from __future__ import annotations

import logging
import math
import time
from typing import List, Tuple

from schemas.models import Passage
from models.model_manager import get_model_manager


class CrossEncoderReranker:
    """Rerank passages with a cross-encoder while preserving fail-soft quality."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large") -> None:
        self.model_name = model_name
        self.model = None
        self._is_available = True
        self._load_attempts = 0

        # --- §13 execution diagnostics (real-execution-only proof) ---
        # These are set on every rerank()/score_gate_candidates() call so the
        # pipeline and certification mode can tell a REAL BGE inference apart
        # from a fail-soft fallback that merely passed hybrid scores through.
        # A fallback must never be presented as a genuine BGE score.
        self.last_status: str = "not_run"  # executed | degraded | unavailable | not_run
        self.last_inference_executed: bool = False
        self.last_degraded: bool = False
        self.last_device: str = "unknown"
        self.last_latency_ms: int = 0
        self.last_scored_count: int = 0

    def _reset_run_diagnostics(self) -> None:
        self.last_status = "not_run"
        self.last_inference_executed = False
        self.last_degraded = False
        self.last_latency_ms = 0
        self.last_scored_count = 0

    def _detect_device(self) -> str:
        """Best-effort device read from the loaded cross-encoder (never raises)."""
        model = self.model
        if model is None:
            return "unknown"
        for attr in ("_target_device", "device"):
            try:
                dev = getattr(model, attr, None)
                if dev is not None:
                    return str(dev)
            except Exception:
                continue
        inner = getattr(model, "model", None)
        try:
            dev = getattr(inner, "device", None)
            if dev is not None:
                return str(dev)
        except Exception:
            pass
        return "unknown"

    def diagnostics(self) -> dict:
        """Return the last-run execution proof for tracing/certification (§13/§26)."""
        return {
            "component": "bge_reranker",
            "model": self.model_name,
            "loaded": self.model is not None and self._is_available,
            "inference_executed": self.last_inference_executed,
            "degraded": self.last_degraded,
            "status": self.last_status,
            "device": self.last_device,
            "latency_ms": self.last_latency_ms,
            "scored_count": self.last_scored_count,
        }

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
        except Exception as exc:
            logging.warning("Error loading cross-encoder model: %s", exc)
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
        self._reset_run_diagnostics()

        if k <= 0 or not passages:
            return []

        if model_name and model_name != self.model_name:
            self.model_name = model_name
            self.model = None
            self._is_available = True
            self._load_attempts = 0

        self._load_model()

        if not self._is_available:
            logging.warning(
                "Reranking skipped (model unavailable); preserving hybrid retrieval order."
            )
            self.last_status = "unavailable"
            self.last_degraded = True
            self.last_inference_executed = False
            return self._fallback(passages, k)

        try:
            pairs = [(claim or "", p.snippet or "") for p in passages]
            _t0 = time.perf_counter()
            raw_scores = self.model.predict(pairs, batch_size=16)
            self.last_latency_ms = int((time.perf_counter() - _t0) * 1000)
            scores = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)

            if len(scores) != len(passages):
                raise ValueError(
                    f"Reranker returned {len(scores)} scores for {len(passages)} passages"
                )

            scored_passages = list(zip(passages, scores))
            scored_passages.sort(key=lambda item: float(item[1]), reverse=True)

            # Real BGE inference succeeded — record execution proof.
            self.last_status = "executed"
            self.last_inference_executed = True
            self.last_degraded = False
            self.last_device = self._detect_device()
            self.last_scored_count = len(scores)

            return [
                p.model_copy(update={"relevance_score": float(score)})
                for p, score in scored_passages[:k]
            ]

        except Exception as exc:
            logging.warning(
                "Reranking failed for %d passages: %s. Preserving hybrid ranking.",
                len(passages),
                exc,
            )
            # Inference was attempted but failed — this is a DEGRADED result, not
            # a real BGE score. Downstream must not treat the fallback as BGE.
            self.last_status = "degraded"
            self.last_degraded = True
            self.last_inference_executed = False
            return self._fallback(passages, k)

    @staticmethod
    def _normalize_gate_score(raw_score: float) -> float:
        """Map cross-encoder logit to [0, 1] for gate decisions (not calibrated probability)."""
        if raw_score < 0.0:
            raw_score = 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, raw_score))))
        return max(0.0, min(1.0, float(raw_score)))

    def score_gate_candidates(
        self,
        claim: str,
        passages: List[Passage],
        max_candidates: int = 5,
        model_name: str | None = None,
    ) -> Tuple[List[float], List[Passage]]:
        """
        Lightweight bounded BGE scoring for the retrieval quality gate.

        Runs cross-encoder inference on at most ``max_candidates`` usable passages
        to decide whether primary evidence is semantically sufficient before Tavily.
        """
        if not passages or max_candidates <= 0:
            return [], []

        candidates = passages[:max_candidates]
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            self.model = None
            self._is_available = True
            self._load_attempts = 0

        self._load_model()
        if not self._is_available:
            fallback_scores = [
                max(0.0, min(1.0, float(getattr(p, "relevance_score", 0.0) or 0.0)))
                for p in candidates
            ]
            return fallback_scores, candidates

        try:
            pairs = [(claim or "", p.snippet or p.title or "") for p in candidates]
            raw_scores = self.model.predict(pairs, batch_size=min(8, len(pairs)))
            scores_list = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)
            normalized = [self._normalize_gate_score(float(s)) for s in scores_list]
            return normalized, candidates
        except Exception as exc:
            logging.warning("Gate BGE scoring failed: %s", exc)
            fallback_scores = [
                max(0.0, min(1.0, float(getattr(p, "relevance_score", 0.0) or 0.0)))
                for p in candidates
            ]
            return fallback_scores, candidates
