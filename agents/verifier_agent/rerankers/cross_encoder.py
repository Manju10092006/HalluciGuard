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
        self._load_error_type = None

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
        self.last_inference_attempted: bool = False
        self.last_inference_completed: bool = False
        self.last_load_attempted: bool = False
        self.last_error_type: str | None = None
        self.last_failure_stage: str | None = None
        self.last_fallback_reason: str | None = None
        self.last_returned_count: int = 0

    def _reset_run_diagnostics(self) -> None:
        self.last_status = "not_run"
        self.last_inference_executed = False
        self.last_degraded = False
        self.last_latency_ms = 0
        self.last_scored_count = 0
        self.last_device = "unknown"
        self.last_inference_attempted = False
        self.last_inference_completed = False
        self.last_load_attempted = False
        self.last_error_type = None
        self.last_failure_stage = None
        self.last_fallback_reason = None
        self.last_returned_count = 0

    def _record_failure(self, stage: str, error_type: str | None, *, unavailable=False) -> None:
        self.last_status = "unavailable" if unavailable else "degraded"
        self.last_degraded = True
        self.last_inference_executed = False
        self.last_scored_count = 0
        self.last_error_type = error_type
        self.last_failure_stage = stage
        self.last_fallback_reason = f"{stage}:{error_type or 'unavailable'}"

    def _record_success(self, scored_count: int, returned_count: int) -> None:
        self.last_status = "executed"
        self.last_inference_executed = True
        self.last_degraded = False
        self.last_device = self._detect_device()
        self.last_scored_count = scored_count
        self.last_returned_count = returned_count

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
            "available": self.model is not None and self._is_available,
            "load_attempted": self.last_load_attempted,
            "inference_attempted": self.last_inference_attempted,
            "inference_completed": self.last_inference_completed,
            "inference_executed": self.last_inference_executed,
            "degraded": self.last_degraded,
            "status": self.last_status,
            "device": self.last_device,
            "latency_ms": self.last_latency_ms,
            "scored_count": self.last_scored_count,
            "returned_count": self.last_returned_count,
            "failure_stage": self.last_failure_stage,
            "error_type": self.last_error_type,
            "fallback_reason": self.last_fallback_reason,
        }

    def _load_model(self) -> None:
        if self.model is not None or not self._is_available:
            return
        if self._load_attempts >= 2:
            self._is_available = False
            return

        self._load_attempts += 1
        self.last_load_attempted = True
        try:
            self.model = get_model_manager().load_reranker_model(self.model_name)
            self._load_attempts = 0
            self._load_error_type = None
        except ImportError as exc:
            logging.warning("transformers not installed. CrossEncoderReranker falling back.")
            self._is_available = False
            self._load_error_type = type(exc).__name__
        except Exception as exc:
            logging.warning("Error loading cross-encoder model: %s", exc)
            self._load_error_type = type(exc).__name__
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
            self._load_error_type = None

        self._load_model()

        if not self._is_available or self.model is None:
            logging.warning(
                "Reranking skipped (model unavailable); preserving hybrid retrieval order."
            )
            self._record_failure("model_load", self._load_error_type, unavailable=True)
            fallback = self._fallback(passages, k)
            self.last_returned_count = len(fallback)
            return fallback

        stage = "inference"
        _t0 = None
        try:
            pairs = [(claim or "", p.snippet or "") for p in passages]
            _t0 = time.perf_counter()
            self.last_inference_attempted = True
            raw_scores = self.model.predict(pairs, batch_size=16)
            self.last_inference_completed = True
            self.last_latency_ms = int((time.perf_counter() - _t0) * 1000)
            stage = "score_validation"
            scores = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)

            if len(scores) != len(passages):
                raise ValueError(
                    f"Reranker returned {len(scores)} scores for {len(passages)} passages"
                )
            scores = [float(score) for score in scores]
            if not all(math.isfinite(score) for score in scores):
                raise ValueError("Reranker returned non-finite scores")

            scored_passages = list(zip(passages, scores))
            scored_passages.sort(key=lambda item: float(item[1]), reverse=True)

            # Real BGE inference succeeded — record execution proof.
            result = [
                p.model_copy(update={"relevance_score": float(score)})
                for p, score in scored_passages[:k]
            ]
            self._record_success(len(scores), len(result))
            return result

        except Exception as exc:
            logging.warning(
                "Reranking failed for %d passages: %s. Preserving hybrid ranking.",
                len(passages),
                exc,
            )
            # Inference was attempted but failed — this is a DEGRADED result, not
            # a real BGE score. Downstream must not treat the fallback as BGE.
            if _t0 is not None:
                self.last_latency_ms = int((time.perf_counter() - _t0) * 1000)
            self._record_failure(stage, type(exc).__name__)
            fallback = self._fallback(passages, k)
            self.last_returned_count = len(fallback)
            return fallback

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
        self._reset_run_diagnostics()
        if not passages or max_candidates <= 0:
            return [], []

        candidates = passages[:max_candidates]
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            self.model = None
            self._is_available = True
            self._load_attempts = 0
            self._load_error_type = None

        self._load_model()
        if not self._is_available or self.model is None:
            self._record_failure("model_load", self._load_error_type, unavailable=True)
            fallback_scores = [
                max(0.0, min(1.0, float(getattr(p, "relevance_score", 0.0) or 0.0)))
                for p in candidates
            ]
            self.last_returned_count = len(candidates)
            return fallback_scores, candidates

        stage = "inference"
        _t0 = None
        try:
            pairs = [(claim or "", p.snippet or p.title or "") for p in candidates]
            _t0 = time.perf_counter()
            self.last_inference_attempted = True
            raw_scores = self.model.predict(pairs, batch_size=min(8, len(pairs)))
            self.last_inference_completed = True
            self.last_latency_ms = int((time.perf_counter() - _t0) * 1000)
            stage = "score_validation"
            scores_list = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)
            if len(scores_list) != len(candidates):
                raise ValueError(
                    f"Reranker returned {len(scores_list)} scores for {len(candidates)} candidates"
                )
            scores_list = [float(score) for score in scores_list]
            if not all(math.isfinite(score) for score in scores_list):
                raise ValueError("Reranker returned non-finite scores")
            normalized = [self._normalize_gate_score(score) for score in scores_list]
            self._record_success(len(normalized), len(candidates))
            return normalized, candidates
        except Exception as exc:
            logging.warning("Gate BGE scoring failed: %s", exc)
            if _t0 is not None:
                self.last_latency_ms = int((time.perf_counter() - _t0) * 1000)
            self._record_failure(stage, type(exc).__name__)
            fallback_scores = [
                max(0.0, min(1.0, float(getattr(p, "relevance_score", 0.0) or 0.0)))
                for p in candidates
            ]
            self.last_returned_count = len(candidates)
            return fallback_scores, candidates
