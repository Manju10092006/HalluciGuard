"""Single production integration seam for the HalluciGuard detector.

Before retrieval the detector performs claim triage and routes factual content
to the Verifier without inventing a probability. After retrieval the graph
calls the same bridge with evidence, which loads and executes the trained model.
Any runtime failure fails closed to verification.
"""
from __future__ import annotations

import threading
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
    raw_probability = result.get("hallucination_probability")
    probability_available = raw_probability is not None
    probability = float(raw_probability) if probability_available else 0.0
    raw_confidence = result.get("confidence_score")
    confidence = float(raw_confidence) if raw_confidence is not None else 0.0
    status = str(result.get("status", "completed")).lower()
    degraded = bool(result.get("detector_degraded", status != "completed"))
    next_action = str(result.get("next_action", "Verify"))
    risk_level = str(result.get("risk_level", "HIGH")).upper()
    if status != "completed" or degraded:
        next_action = "Verify"
        risk_level = "HIGH"

    overall_verify = next_action.lower() == "verify"
    per_claim_results = []
    for index, claim in enumerate(result.get("claims") or [], start=1):
        text = str(claim.get("text") or "").strip()
        if not text:
            continue
        raw_risk = claim.get("claim_risk")
        claim_probability = float(raw_risk) if raw_risk is not None else 0.0
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
                "probability_available": raw_risk is not None,
                "risk_level": claim_level,
                "requires_verification": bool(
                    claim.get("requires_verification", overall_verify)
                ),
            }
        )

    diagnostics = result.get("diagnostics") or {}
    return {
        "hallucination_probability": probability,
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
    }


def run_detection(user_query: str, llm_response: str) -> dict[str, Any]:
    """Run evidence-free pre-retrieval triage.

    This intentionally does not load the trained reference-grounded model.
    ``run_grounded_detection`` is the production inference entry point.
    """
    try:
        result = _get_agent().detect(user_query, llm_response)
        return _map_result(result)
    except Exception as exc:  # fail closed; the graph must still reach Verifier
        return _failclosed(f"detector_failed: {type(exc).__name__}: {str(exc)[:160]}")


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
            f"grounded_detector_failed: {type(exc).__name__}: {str(exc)[:160]}"
        )
