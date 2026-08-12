from __future__ import annotations

import logging
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
        if not evidences:
            return []
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            self.pipeline = None
            self._is_available = True
        self._load_model()
        if not self._is_available or self.pipeline is None:
            return [self._neutral() for _ in evidences]
        try:
            batch = [
                {"text": evidence or "", "text_pair": claim or ""}
                for evidence in evidences
            ]
            raw_batch = self.pipeline(batch)
            if not isinstance(raw_batch, list) or len(raw_batch) != len(evidences):
                raise ValueError("NLI batch output is not aligned with input batch")
            id2label = self._get_id2label()
            outputs = []
            for raw_item in raw_batch:
                scores = _normalize_scores(_flatten_predictions(raw_item), id2label)
                outputs.append(
                    _decision(scores) if sum(scores.values()) > 0 else self._neutral()
                )
            return outputs
        except Exception as exc:
            logger.warning("Batched NLI failed; retrying individually: %s", exc)
            return [
                self.classify(claim, evidence, model_name=self.model_name)
                for evidence in evidences
            ]
