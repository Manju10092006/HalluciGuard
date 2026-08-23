from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from schemas.models import EntailmentLabel
from models.model_manager import get_model_manager

logger = logging.getLogger(__name__)


def _canonical_label(
    raw_label: Any, id2label: Optional[Dict[Any, str]] = None
) -> Optional[str]:
    label = str(raw_label or "").strip().lower()
    if "entail" in label or label in {"supports", "support"}:
        return "entailment"
    if "contradict" in label or "refute" in label or label in {"refutes", "refutation"}:
        return "contradiction"
    if "neutral" in label:
        return "neutral"
    if label.startswith("label_"):
        try:
            index = int(label.split("_", 1)[1])
        except (ValueError, IndexError):
            return None
        mapped = id2label.get(index) if id2label else None
        if mapped is None and id2label:
            mapped = id2label.get(str(index))
        if mapped:
            return _canonical_label(mapped, None)
        return {0: "contradiction", 1: "entailment", 2: "neutral"}.get(index)
    return None


def _flatten_predictions(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    if not isinstance(raw, list):
        return []
    if raw and isinstance(raw[0], dict):
        return raw
    flattened: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            flattened.append(item)
        elif isinstance(item, list):
            flattened.extend(x for x in item if isinstance(x, dict))
    return flattened


def _normalize_scores(
    items: List[Dict[str, Any]],
    id2label: Optional[Dict[Any, str]] = None,
) -> Dict[str, float]:
    scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
    for item in items:
        label = _canonical_label(item.get("label"), id2label)
        if label is None:
            continue
        try:
            score = max(0.0, float(item.get("score", 0.0)))
        except (TypeError, ValueError):
            continue
        scores[label] = max(scores[label], score)
    total = sum(scores.values())
    if total > 0:
        scores = {key: value / total for key, value in scores.items()}
    return scores


def _decision(scores: Dict[str, float]) -> Dict[str, Any]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ordered[0]
    second_score = ordered[1][1]
    label = (
        EntailmentLabel.NEUTRAL
        if top_score < 0.45 or (top_score - second_score) < 0.05
        else EntailmentLabel(top_label)
    )
    return {
        "label": label,
        "entailment_score": round(scores["entailment"], 6),
        "contradiction_score": round(scores["contradiction"], 6),
        "neutral_score": round(scores["neutral"], 6),
    }


class NLIEngine:
    """Production-safe NLI wrapper with explicit label mapping and alignment."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base") -> None:
        self.model_name = model_name
        self.pipeline = None
        self._is_available = True

        # --- §15 engine-level execution diagnostics (real-execution-only proof) ---
        # This is the canonical pipeline NLI engine (see nli/__init__.py). It mirrors
        # nli/entailment.py's diagnostics so the pipeline can build a ModelExecutionTrace
        # (§26) and certification can fail-closed on fallback (§28). The decision logic
        # below (_normalize_scores/_decision, batch alignment) is intentionally unchanged.
        self.last_status: str = "not_run"  # executed | degraded | unavailable | not_run
        self.last_inference_executed: bool = False
        self.last_degraded: bool = False
        self.last_device: str = "unknown"
        self.last_latency_ms: int = 0
        self.last_batch_size: int = 0

    def _reset_run_diagnostics(self) -> None:
        self.last_status = "not_run"
        self.last_inference_executed = False
        self.last_degraded = False
        self.last_latency_ms = 0
        self.last_batch_size = 0

    def is_loaded(self) -> bool:
        """True iff the NLI pipeline is loaded and usable."""
        return self.pipeline is not None and self._is_available

    def _detect_device(self) -> str:
        """Best-effort device read from the loaded HF pipeline (never raises)."""
        pipe = self.pipeline
        if pipe is None:
            return "unknown"
        try:
            dev = getattr(pipe, "device", None)
            if dev is not None:
                return str(dev)
        except Exception:
            pass
        model = getattr(pipe, "model", None)
        try:
            dev = getattr(model, "device", None)
            if dev is not None:
                return str(dev)
        except Exception:
            pass
        return "unknown"

    def diagnostics(self) -> dict:
        """Return the last-run execution proof for tracing/certification (§15/§26)."""
        return {
            "component": "deberta_nli",
            "model": self.model_name,
            "loaded": self.is_loaded(),
            "inference_executed": self.last_inference_executed,
            "degraded": self.last_degraded,
            "status": self.last_status,
            "device": self.last_device,
            "latency_ms": self.last_latency_ms,
            "batch_size": self.last_batch_size,
        }

    def _load_model(self) -> None:
        if self.pipeline is not None or not self._is_available:
            return
        try:
            self.pipeline = get_model_manager().load_nli_model(self.model_name)
        except Exception as exc:
            logger.warning(
                "NLI model unavailable (%s); using degraded neutral result", exc
            )
            self._is_available = False

    def _get_id2label(self) -> Optional[Dict[Any, str]]:
        model = getattr(self.pipeline, "model", None)
        config = getattr(model, "config", None)
        return getattr(config, "id2label", None)

    @staticmethod
    def _neutral() -> Dict[str, Any]:
        return {
            "label": EntailmentLabel.NEUTRAL,
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "neutral_score": 1.0,
            "degraded": True,
            "error": "nli_model_unavailable_or_failed",
        }

    def classify(
        self, claim: str, evidence: str, model_name: str | None = None
    ) -> Dict[str, Any]:
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            self.pipeline = None
            self._is_available = True
        self._load_model()
        if not self._is_available or self.pipeline is None:
            return self._neutral()
        try:
            # Premise = evidence, hypothesis = claim.
            raw = self.pipeline({"text": evidence or "", "text_pair": claim or ""})
            scores = _normalize_scores(_flatten_predictions(raw), self._get_id2label())
            return _decision(scores) if sum(scores.values()) > 0 else self._neutral()
        except Exception as exc:
            logger.warning("NLI classification failed: %s", exc)
            return self._neutral()

    def predict(self, claim: str, evidence: str) -> EntailmentLabel:
        return self.classify(claim, evidence)["label"]

    def batch_classify(
        self,
        claim: str,
        evidences: List[str],
        model_name: str | None = None,
    ) -> List[Dict[str, Any]]:
        # §15/§26 execution proof: reset first so an empty call reads "not_run".
        self._reset_run_diagnostics()
        if not evidences:
            return []
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            self.pipeline = None
            self._is_available = True
        self._load_model()
        if not self._is_available or self.pipeline is None:
            self.last_status = "unavailable"
            self.last_degraded = True
            self.last_inference_executed = False
            return [self._neutral() for _ in evidences]
        try:
            batch = [
                {"text": evidence or "", "text_pair": claim or ""}
                for evidence in evidences
            ]
            _t0 = time.perf_counter()
            raw_batch = self.pipeline(batch)
            self.last_latency_ms = int((time.perf_counter() - _t0) * 1000)
            if not isinstance(raw_batch, list) or len(raw_batch) != len(evidences):
                raise ValueError("NLI batch output is not aligned with input batch")
            id2label = self._get_id2label()
            outputs = []
            for raw_item in raw_batch:
                scores = _normalize_scores(_flatten_predictions(raw_item), id2label)
                outputs.append(
                    _decision(scores) if sum(scores.values()) > 0 else self._neutral()
                )
            # Real DeBERTa inference succeeded — record execution proof.
            self.last_status = "executed"
            self.last_inference_executed = True
            self.last_degraded = False
            self.last_device = self._detect_device()
            self.last_batch_size = len(evidences)
            return outputs
        except Exception as exc:
            logger.warning("Batched NLI failed; retrying individually: %s", exc)
            self.last_status = "degraded"
            self.last_degraded = True
            self.last_inference_executed = False
            return [
                self.classify(claim, evidence, model_name=self.model_name)
                for evidence in evidences
            ]
