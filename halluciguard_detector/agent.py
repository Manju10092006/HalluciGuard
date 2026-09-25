"""HalluciGuard orchestration adapter for the reference-grounded detector.

The production graph calls the Detector before n8n evidence retrieval. At that
point this adapter performs claim triage and fails safely toward verification;
it does not invent a truth probability. When evidence is supplied, it executes
the trained model and returns sentence-level grounded predictions.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Iterable

from .detector import Detector
from .text import sentence_spans


class DetectorAgent:
    """Compatibility surface used by HalluciGuard orchestration and services."""

    _detector: Detector | None = None
    _lock = threading.Lock()

    def __init__(self, model_path: str | Path | None = None, device: str | None = None):
        repo_root = Path(__file__).resolve().parent.parent
        configured = os.getenv("HALLUCIGUARD_DETECTOR_MODEL", "").strip()
        selected_path = Path(model_path or configured or repo_root / "artifacts" / "detector-best")
        self.model_path = selected_path if selected_path.is_absolute() else repo_root / selected_path
        self.device = device

    def _get_detector(self) -> Detector:
        if self.__class__._detector is None:
            with self.__class__._lock:
                if self.__class__._detector is None:
                    self.__class__._detector = Detector(self.model_path, self.device)
        return self.__class__._detector

    @staticmethod
    def _evidence_from(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, dict):
            for key in ("evidence", "passages", "documents", "context"):
                if key in value:
                    return DetectorAgent._evidence_from(value[key])
            text = value.get("snippet") or value.get("text") or value.get("content")
            return [str(text)] if text and str(text).strip() else []
        if isinstance(value, Iterable):
            result: list[str] = []
            for item in value:
                result.extend(DetectorAgent._evidence_from(item))
            return result
        return []

    def detect(
        self,
        user_query: str,
        llm_response: str | None = None,
        context: Any = None,
        evidence: Any = None,
        draft_answer: str | None = None,
    ) -> dict[str, Any]:
        answer = str(llm_response if llm_response is not None else draft_answer or "").strip()
        if not answer:
            raise ValueError("llm_response/draft_answer must not be empty")

        evidence_texts = self._evidence_from(evidence)
        if not evidence_texts:
            evidence_texts = self._evidence_from(context)
        if not evidence_texts:
            return self._pre_verification_result(user_query, answer)

        grounded = self._get_detector().detect(
            user_query=user_query,
            draft_answer=answer,
            evidence=evidence_texts,
        )
        claims = []
        for index, sentence in enumerate(grounded.sentences, start=1):
            claims.append(
                {
                    "claim_id": index,
                    "text": sentence.text,
                    "span": [sentence.start, sentence.end],
                    "claim_risk": sentence.hallucination_probability,
                    "label": sentence.label.value,
                    "probabilities": {
                        key.value: value for key, value in sentence.probabilities.items()
                    },
                    "risk_level": sentence.risk.value,
                    "requires_verification": sentence.label.value != "SUPPORTED",
                    "evidence_snippets": sentence.evidence_snippets,
                }
            )
        confidence = max(
            max(sentence.probabilities.values()) for sentence in grounded.sentences
        )
        return {
            "hallucination_probability": grounded.probability,
            "confidence_score": float(confidence),
            "risk_level": grounded.risk.value,
            "next_action": "Verify" if grounded.requires_verification else "Accept",
            "model_source": "halluciguard_detector_ragtruth_deberta",
            "status": "completed",
            "calibrated": True,
            "calibration_applied": True,
            "inference_executed": True,
            "model_loaded": True,
            "model_version": grounded.model_version,
            "calibrator_version": "temperature-scaling-v1",
            "detector_degraded": False,
            "grounded": True,
            "probability_semantics": (
                "max over sentences of calibrated P(CONTRADICTED)+P(NOT_ENOUGH_INFO)"
            ),
            "verification_reason": (
                "one_or_more_claims_not_supported" if grounded.requires_verification else None
            ),
            "claims": claims,
            "warnings": grounded.warnings,
            "diagnostics": {"degraded_reason": None, "evidence_count": len(evidence_texts)},
        }

    def _pre_verification_result(self, user_query: str, answer: str) -> dict[str, Any]:
        spans = sentence_spans(answer)
        claims = [
            {
                "claim_id": index,
                "text": span.text,
                "span": [span.start, span.end],
                "claim_risk": None,
                "label": "UNVERIFIED",
                "probabilities": {},
                "risk_level": "HIGH",
                "requires_verification": True,
                "evidence_snippets": [],
            }
            for index, span in enumerate(spans, start=1)
        ]
        return {
            "hallucination_probability": None,
            "confidence_score": None,
            "risk_level": "HIGH",
            "next_action": "Verify",
            "model_source": "halluciguard_detector_pre_verification_triage",
            "status": "completed",
            "calibrated": False,
            "calibration_applied": False,
            "inference_executed": False,
            "model_loaded": False,
            "model_version": self.model_path.name,
            "calibrator_version": None,
            "detector_degraded": False,
            "grounded": False,
            "probability_semantics": "not_computed_without_evidence",
            "verification_reason": "evidence_required_for_truth_assessment",
            "claims": claims,
            "warnings": [
                "Pre-retrieval triage only; n8n/Verifier must retrieve and evaluate evidence."
            ],
            "diagnostics": {
                "degraded_reason": None,
                "mode": "pre_verification_triage",
                "query_present": bool(user_query.strip()),
                "claim_count": len(claims),
            },
        }
