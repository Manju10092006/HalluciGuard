"""HalluciGuard orchestration adapter for the reference-grounded detector.

The production graph calls the Detector before n8n evidence retrieval. At that
point this adapter performs claim triage and fails safely toward verification;
it does not invent a truth probability. When evidence is supplied, it executes
the trained model and returns sentence-level grounded predictions.
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from .detector import Detector
from .evidence_shapes import normalize_evidence
from .schemas import RiskLevel
from .text import sentence_spans


@lru_cache(maxsize=1)
def _phase1_service():
    from .phase1 import Phase1Service
    return Phase1Service()


class DetectorAgent:
    """Compatibility surface used by HalluciGuard orchestration and services."""

    def __init__(self, model_path: str | Path | None = None, device: str | None = None):
        repo_root = Path(__file__).resolve().parent.parent
        configured = os.getenv("HALLUCIGUARD_DETECTOR_MODEL", "").strip()
        selected_path = Path(model_path or configured or repo_root / "artifacts" / "detector-best")
        self.model_path = selected_path if selected_path.is_absolute() else repo_root / selected_path
        self.device = device
        self._detector: Detector | None = None
        self._lock = threading.Lock()

    def _get_detector(self) -> Detector:
        if self._detector is None:
            with self._lock:
                if self._detector is None:
                    self._detector = Detector(self.model_path, self.device)
        return self._detector

    @staticmethod
    def _evidence_from(value: Any) -> list[str]:
        return normalize_evidence(value).documents

    def detect(
        self,
        user_query: str,
        llm_response: str | None = None,
        context: Any = None,
        evidence: Any = None,
        draft_answer: str | None = None,
        generation_trace: dict[str, Any] | None = None,
        domain: str = "general",
    ) -> dict[str, Any]:
        answer = str(llm_response if llm_response is not None else draft_answer or "").strip()
        if not answer:
            raise ValueError("llm_response/draft_answer must not be empty")

        normalized = normalize_evidence(evidence)
        if not normalized.documents:
            context_normalized = normalize_evidence(context)
            context_normalized.malformed_records += normalized.malformed_records
            context_normalized.truncated_records += normalized.truncated_records
            context_normalized.null_fields += normalized.null_fields
            normalized = context_normalized
        evidence_texts = normalized.documents
        if not evidence_texts:
            result = self._pre_verification_result(user_query, answer)
            result["diagnostics"]["evidence_shape"] = normalized.diagnostics()
            spans = sentence_spans(answer)
            phase1 = _phase1_service().score(
                answer,
                [{"start": span.start, "end": span.end} for span in spans],
                generation_trace,
                domain,
            )
            result["phase1"] = phase1.as_dict()
            if phase1.bypass_eligible:
                result["next_action"] = "Accept"
                result["risk_level"] = "LOW"
                result["verification_reason"] = "phase1_validated_fast_path"
                for claim, risk in zip(result["claims"], phase1.calibrated_risks):
                    claim["phase1_risk"] = risk
                    claim["requires_verification"] = False
                    claim["risk_level"] = "LOW"
            elif phase1.status == "scored":
                for claim, risk in zip(result["claims"], phase1.calibrated_risks):
                    claim["phase1_risk"] = risk
            return result

        grounded = self._get_detector().detect(
            user_query=user_query,
            draft_answer=answer,
            evidence=evidence_texts,
        )
        claims = []
        for index, sentence in enumerate(grounded.sentences, start=1):
            snippet_sources = []
            for snippet in sentence.evidence_snippets:
                matching = {source for document, source in zip(normalized.documents, normalized.document_sources)
                            if snippet in document}
                snippet_sources.append(next(iter(matching)) if len(matching) == 1 and None not in matching else None)
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
                    "requires_verification": sentence.label.value != "SUPPORTED" or sentence.risk == RiskLevel.HIGH,
                    "evidence_snippets": sentence.evidence_snippets,
                    "evidence_source_ids": snippet_sources,
                }
            )
        confidence = max(
            max(sentence.probabilities.values()) for sentence in grounded.sentences
        )
        evidence_shape = normalized.diagnostics()
        return {
            "hallucination_probability": grounded.probability,
            "confidence_score": float(confidence),
            "risk_level": grounded.risk.value,
            "next_action": "Verify" if grounded.requires_verification else "Accept",
            "model_source": "halluciguard_detector_ragtruth_deberta",
            "status": "degraded" if evidence_shape["degraded"] else "completed",
            "calibrated": True,
            "calibration_applied": True,
            "inference_executed": True,
            "model_loaded": True,
            "model_version": grounded.model_version,
            "calibrator_version": "temperature-scaling-v1",
            "detector_degraded": evidence_shape["degraded"],
            "grounded": True,
            "probability_semantics": (
                "max over sentences of calibrated P(CONTRADICTED)+P(NOT_ENOUGH_INFO)"
            ),
            "verification_reason": (
                "one_or_more_claims_not_supported" if grounded.requires_verification else None
            ),
            "claims": claims,
            "warnings": grounded.warnings,
            "diagnostics": {"degraded_reason": "evidence_normalization_incomplete" if evidence_shape["degraded"] else None,
                            "evidence_count": len(evidence_texts), "evidence_shape": evidence_shape,
                            "model_input": getattr(grounded, "input_diagnostics", {})},
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
