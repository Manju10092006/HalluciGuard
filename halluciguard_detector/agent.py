"""HalluciGuard orchestration adapter for the reference-grounded detector.

The production graph calls the Detector before n8n evidence retrieval. At that
point this adapter performs claim triage and fails safely toward verification;
it does not invent a truth probability. When evidence is supplied, it executes
the trained model and returns claim-level grounded predictions.

Result semantics (see ``Detector`` for the authoritative description):
  SUPPORTED      -> the evidence supports the claim.
  CONTRADICTED   -> the evidence conflicts with the claim.
  NOT_ENOUGH_INFO-> the evidence is insufficient; NOT proof of falsehood.
  verification_risk / hallucination_probability -> an operational score for
  downstream triage (P(CONTRADICTED) + P(NOT_ENOUGH_INFO)), not a calibrated
  probability that the claim is objectively false.
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
                    "requires_verification": (
                        getattr(sentence, "label", None).value != "SUPPORTED"
                        if getattr(sentence, "label", None) is not None
                        else True
                    ),
                    "evidence_snippets": getattr(sentence, "evidence_snippets", []),
                    "supported_probability": getattr(sentence, "supported_probability", 0.0),
                    "contradicted_probability": getattr(
                        sentence, "contradicted_probability", 0.0
                    ),
                    "unknown_probability": getattr(sentence, "unknown_probability", 0.0),
                    "non_factual": bool(getattr(sentence, "non_factual", False)),
                    "verification_risk": getattr(
                        sentence,
                        "verification_risk",
                        getattr(sentence, "hallucination_probability", 0.0),
                    ),
                }
            )
        confidence = max(
            max(sentence.probabilities.values(), default=0.0)
            for sentence in grounded.sentences
        )
        verification_risk = float(
            getattr(
                grounded,
                "verification_risk",
                getattr(grounded, "probability", 0.0),
            )
        )
        # Max P(CONTRADICTED) over assessed claims. This is the only field that
        # speaks about falsehood, so it is forwarded separately from
        # verification_risk to keep "refuted" and "still needs checking" apart.
        contradiction_mass = float(
            getattr(
                grounded,
                "contradiction_mass",
                max(
                    (
                        getattr(sentence, "contradicted_probability", 0.0)
                        for sentence in grounded.sentences
                        if not getattr(sentence, "non_factual", False)
                    ),
                    default=0.0,
                ),
            )
        )
        return {
            # Legacy field retained; semantically the operational verification
            # risk (P(CONTRADICTED) + P(NOT_ENOUGH_INFO)), not P(hallucinated).
            "hallucination_probability": grounded.probability,
            "verification_risk": verification_risk,
            "contradiction_mass": contradiction_mass,
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
                "claim_level: SUPPORTED=evidence supports, CONTRADICTED=evidence "
                "conflicts, NOT_ENOUGH_INFO=insufficient evidence (never folded "
                "into contradiction); verification_risk=operational triage score "
                "P(CONTRADICTED)+P(NOT_ENOUGH_INFO), not a probability the claim "
                "is false; contradiction_mass=max P(CONTRADICTED), the only "
                "refutation signal; final decision belongs to the Judge"
            ),
            "verification_reason": (
                "one_or_more_claims_not_supported" if grounded.requires_verification else None
            ),
            "claims": claims,
            "warnings": getattr(grounded, "warnings", []),
            "claim_count": getattr(grounded, "claim_count", len(claims)),
            "supported_count": getattr(grounded, "supported_count", 0),
            "contradicted_count": getattr(grounded, "contradicted_count", 0),
            "unknown_count": getattr(grounded, "unknown_count", 0),
            "non_factual_count": getattr(grounded, "non_factual_count", 0),
            "diagnostics": {
                "degraded_reason": None,
                "evidence_count": len(evidence_texts),
                "claim_count": getattr(grounded, "claim_count", len(claims)),
                "supported_count": getattr(grounded, "supported_count", 0),
                "contradicted_count": getattr(grounded, "contradicted_count", 0),
                "unknown_count": getattr(grounded, "unknown_count", 0),
                "non_factual_count": getattr(grounded, "non_factual_count", 0),
            },
        }

    def _pre_verification_result(self, user_query: str, answer: str) -> dict[str, Any]:
        """Evidence-free triage: split the answer into claims and require verification.

        No evidence means no grounded verification is possible, so this path
        never fabricates NLI probabilities (``hallucination_probability`` and
        ``verification_risk`` are ``None`` and ``inference_executed`` is
        ``False``). Sentence spans are used purely as a triage representation
        of the answer so downstream retrieval knows what to target; they are
        not final claim verdicts. The graph must retrieve evidence and re-run
        the grounded detector before the Judge.
        """
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
            "verification_risk": None,
            "contradiction_mass": None,
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
            "claim_count": len(claims),
            "supported_count": 0,
            "contradicted_count": 0,
            "unknown_count": 0,
            "non_factual_count": 0,
            "diagnostics": {
                "degraded_reason": None,
                "mode": "pre_verification_triage",
                "query_present": bool(user_query.strip()),
                "claim_count": len(claims),
            },
        }
