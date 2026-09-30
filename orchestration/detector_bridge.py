"""Single production integration seam for the HalluciGuard detector.

Before retrieval the detector performs claim triage and routes factual content
to the Verifier without inventing a probability. After retrieval the graph
calls the same bridge with evidence, which loads and executes the trained model.
Any runtime failure fails closed to verification.
"""
from __future__ import annotations

import threading
import math
from typing import Any


_AGENT: Any = None
_LOCK = threading.Lock()


def _get_agent() -> Any:
    global _AGENT
    if _AGENT is None:
        with _LOCK:
            if _AGENT is None:
                from halluciguard_detector import DetectorAgent

                _AGENT = DetectorAgent()
    return _AGENT


def _failclosed(reason: str) -> dict[str, Any]:
    return {
        "hallucination_probability": 0.0,
        "probability_available": False,
        "confidence_score": 0.0,
        "risk_level": "HIGH",
        "next_action": "Verify",
        "model_source": "halluciguard_detector_unavailable",
        "status": "degraded",
        "detector_degraded": True,
        "inference_executed": False,
        "model_loaded": False,
        "model_version": None,
        "calibrator_version": None,
        "calibration_applied": False,
        "grounded": False,
        "probability_semantics": "not_available_detector_failure",
        "verification_reason": "detector_failure",
        "degraded_reason": reason,
        "per_claim_results": [],
    }


def _map_result(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("detector result must be an object")

    def probability(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) and 0.0 <= parsed <= 1.0 else None

    raw_probability = result.get("hallucination_probability")
    valid_probability = probability(raw_probability)
    probability_available = valid_probability is not None
    probability_value = valid_probability if probability_available else 0.0
    raw_confidence = result.get("confidence_score")
    confidence = probability(raw_confidence) or 0.0
    status = str(result.get("status", "completed")).lower()
    degraded = bool(result.get("detector_degraded", status != "completed"))
    next_action = str(result.get("next_action", "Verify"))
    risk_level = str(result.get("risk_level", "HIGH")).upper()
    if risk_level not in {"LOW", "MEDIUM", "HIGH"}:
        risk_level = "HIGH"
        degraded = True
    phase1 = dict(result.get("phase1")) if isinstance(result.get("phase1"), dict) else {}
    risks = phase1.get("calibrated_risks")
    response_risk = probability(phase1.get("response_risk"))
    validated_phase1 = (
        phase1.get("bypass_eligible") is True
        and phase1.get("status") == "scored"
        and phase1.get("mode") == "fast_path"
        and bool(phase1.get("head_id")) and bool(phase1.get("calibration_id"))
        and isinstance(risks, list) and bool(risks)
        and all(probability(item) is not None for item in risks)
        and response_risk is not None
        and abs(response_risk - max(float(item) for item in risks)) < 1e-6
    )
    if not validated_phase1:
        phase1["bypass_eligible"] = False
    grounded_score = bool(result.get("grounded")) and bool(result.get("calibration_applied")) and probability_available
    if status != "completed" or degraded or not (validated_phase1 or grounded_score):
        next_action = "Verify"
        risk_level = "HIGH"

    overall_verify = next_action.lower() == "verify"
    per_claim_results = []
    for index, claim in enumerate(result.get("claims") or [], start=1):
        text = str(claim.get("text") or "").strip()
        if not text:
            continue
        raw_risk = claim.get("claim_risk")
        valid_claim_risk = probability(raw_risk)
        claim_probability = valid_claim_risk if valid_claim_risk is not None else 0.0
        claim_level = str(
            claim.get("risk_level") or ("HIGH" if overall_verify else "LOW")
        ).upper()
        per_claim_results.append(
            {
                "claim_id": f"c{claim.get('claim_id', index)}",
                "text": text,
                "span": claim.get("span"),
                "label": claim.get("label", "UNVERIFIED"),
                "hallucination_probability": claim_probability,
                "probability_available": valid_claim_risk is not None,
                "phase1_risk": claim.get("phase1_risk"),
                "risk_level": claim_level,
                "requires_verification": bool(
                    claim.get("requires_verification", overall_verify)
                    or (valid_claim_risk is None and not validated_phase1)
                ),
            }
        )

    diagnostics = result.get("diagnostics") or {}
    return {
        "hallucination_probability": probability_value,
        "probability_available": probability_available,
        "confidence_score": confidence,
        "risk_level": risk_level,
        "next_action": next_action,
        "model_source": str(result.get("model_source", "halluciguard_detector")),
        "status": status,
        "calibrated": bool(result.get("calibration_applied", False)),
        "calibration_applied": bool(result.get("calibration_applied", False)),
        "inference_executed": bool(result.get("inference_executed", False)),
        "model_loaded": bool(result.get("model_loaded", False)),
        "model_version": result.get("model_version"),
        "calibrator_version": result.get("calibrator_version"),
        "detector_degraded": degraded,
        "grounded": bool(result.get("grounded", False)),
        "probability_semantics": result.get("probability_semantics"),
        "verification_reason": result.get("verification_reason"),
        "degraded_reason": diagnostics.get("degraded_reason") or result.get("degraded_reason"),
        "diagnostics": diagnostics,
        "warnings": result.get("warnings") or [],
        "per_claim_results": per_claim_results,
        "phase1": phase1,
    }


def run_detection(user_query: str, llm_response: str, generation_trace: dict[str, Any] | None = None, domain: str = "general") -> dict[str, Any]:
    """Run evidence-free pre-retrieval triage.

    This intentionally does not load the trained reference-grounded model.
    ``run_grounded_detection`` is the production inference entry point.
    """
    try:
        if generation_trace is None and domain == "general":
            result = _get_agent().detect(user_query, llm_response)
        else:
            result = _get_agent().detect(user_query, llm_response, generation_trace=generation_trace, domain=domain)
        return _map_result(result)
    except Exception as exc:  # fail closed; the graph must still reach Verifier
        return _failclosed(f"detector_failed: {type(exc).__name__}")


def run_grounded_detection(
    user_query: str,
    llm_response: str,
    evidence: list[dict[str, Any]] | list[str],
) -> dict[str, Any]:
    """Execute the trained detector against evidence retrieved by Verifier."""
    try:
        result = _get_agent().detect(
            user_query,
            llm_response,
            evidence=evidence,
        )
        return _map_result(result)
    except Exception as exc:  # fail closed; Judge still receives verifier truth
        return _failclosed(
            f"grounded_detector_failed: {type(exc).__name__}"
        )
