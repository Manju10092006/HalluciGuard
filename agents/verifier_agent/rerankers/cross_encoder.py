from __future__ import annotations

import logging
from typing import List

from schemas.models import Passage
from models.model_manager import get_model_manager


class CrossEncoderReranker:
    """Rerank passages with a cross-encoder while preserving fail-soft quality."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large") -> None:
        self.model_name = model_name
        self.model = None
        self._is_available = True
        self._load_attempts = 0
        self.last_diagnostics: dict = {"status": "not_run", "attempted": False,
                                       "inference_executed": False, "degraded": False}

    def diagnostics(self) -> dict:
        return dict(self.last_diagnostics)

    def _load_model(self) -> None:
        if self.model is not None or not self._is_available:
            return
        if self._load_attempts >= 2:
            self._is_available = False
            return

        self._load_attempts += 1
        try:
            self.model = get_model_manager().load_reranker_model(self.model_name)
            self._load_attempts = 0
        except ImportError:
            logging.warning("transformers not installed. CrossEncoderReranker falling back.")
            self._is_available = False
            self.last_diagnostics.update(failure_stage="initialization", error_type="ImportError")
        except Exception as exc:
            logging.warning("Error loading cross-encoder model (%s)", type(exc).__name__)
            self.last_diagnostics.update(failure_stage="initialization", error_type=type(exc).__name__)
            if self._load_attempts >= 2:
                self._is_available = False

    @staticmethod
    def _fallback(passages: List[Passage], k: int) -> List[Passage]:
        """Keep hybrid retrieval scores when cross-encoder inference is unavailable."""
        return [
            p.model_copy(
                update={
                    "relevance_score": max(
                        0.0, min(1.0, float(getattr(p, "relevance_score", 0.0)))
                    )
                }
            )
            for p in passages[:k]
        ]

    def rerank(
        self,
        claim: str,
        passages: List[Passage],
        k: int,
        model_name: str | None = None,
    ) -> List[Passage]:
        self.last_diagnostics = {"status": "not_run", "attempted": False,
                                 "initialization_attempted": False,
                                 "inference_executed": False, "degraded": False,
                                 "failure_stage": None, "error_type": None,
                                 "selected_count": 0}
        if k <= 0 or not passages:
            return []
        self.last_diagnostics["attempted"] = True

        if model_name and model_name != self.model_name:
            self.model_name = model_name
            self.model = None
            self._is_available = True
            self._load_attempts = 0

        self.last_diagnostics["initialization_attempted"] = self.model is None and self._is_available
        self._load_model()

        if not self._is_available or self.model is None:
            logging.warning(
                "Reranking skipped (model unavailable); preserving hybrid retrieval order."
            )
            selected = self._fallback(passages, k)
            self.last_diagnostics.update(status="unavailable", degraded=True,
                                         failure_stage=self.last_diagnostics.get("failure_stage") or "initialization",
                                         selected_count=len(selected))
            return selected

        try:
            pairs = [(claim or "", p.snippet or "") for p in passages]
            raw_scores = self.model.predict(pairs, batch_size=16)
            scores = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)

            if len(scores) != len(passages):
                raise ValueError(
                    f"Reranker returned {len(scores)} scores for {len(passages)} passages"
                )

            scored_passages = list(zip(passages, scores))
            scored_passages.sort(key=lambda item: float(item[1]), reverse=True)

            selected = [
                p.model_copy(update={"relevance_score": float(score)})
                for p, score in scored_passages[:k]
            ]
            self.last_diagnostics.update(status="executed", inference_executed=True,
                                         selected_count=len(selected))
            return selected

        except Exception as exc:
            logging.warning(
                "Reranking failed for %d passages (%s). Preserving hybrid ranking.",
                len(passages),
                type(exc).__name__,
            )
            selected = self._fallback(passages, k)
            self.last_diagnostics.update(status="failed", degraded=True,
                                         failure_stage="inference", error_type=type(exc).__name__,
                                         selected_count=len(selected))
            return selected
