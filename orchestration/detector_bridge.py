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
        "hallucination_probability": None,
        "probability_available": False,
        "confidence_score": None,
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
        "claim_count": 0,
        "supported_count": 0,
        "contradicted_count": 0,
        "unknown_count": 0,
        "non_factual_count": 0,
    }


def _map_result(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("invalid_detector_result")
    def valid_score(value: Any) -> float | None:
        if value is None:
            return None
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid_detector_score")
        return float(value)

    raw_probability = result.get("hallucination_probability")
    probability_available = raw_probability is not None
    probability = valid_score(raw_probability)
    raw_confidence = result.get("confidence_score")
    confidence = valid_score(raw_confidence)
    status = str(result.get("status", "unknown")).lower()
    degraded = result.get("detector_degraded", status != "completed") is not False or status != "completed"
    next_action = str(result.get("next_action", "Verify"))
    risk_level = str(result.get("risk_level", "HIGH")).upper()
    if risk_level not in {"LOW", "MEDIUM", "HIGH"} or next_action.lower() not in {"verify", "accept"}:
        raise ValueError("invalid_detector_route")
    grounded = result.get("grounded") is True
    calibrated = result.get("calibration_applied") is True
    executed = result.get("inference_executed") is True
    loaded = result.get("model_loaded") is True
    if degraded or not grounded or not calibrated or not executed or not loaded or probability is None or confidence is None:
        next_action = "Verify"
        risk_level = "HIGH"

    overall_verify = next_action.lower() == "verify"
    per_claim_results = []
    for index, claim in enumerate(result.get("claims") or [], start=1):
        text = str(claim.get("text") or "").strip()
        if not text:
            continue
        raw_risk = claim.get("claim_risk")
        claim_probability = valid_score(raw_risk)
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
                "supported_probability": claim.get("supported_probability", 0.0),
                "contradicted_probability": claim.get("contradicted_probability", 0.0),
                "unknown_probability": claim.get("unknown_probability", 0.0),
                "non_factual": bool(claim.get("non_factual", False)),
                "verification_risk": claim.get(
                    "verification_risk", claim_probability
                ),
            }
        )

    # The refutation signal is derived, never invented. Prefer the canonical
    # answer-level field; otherwise fall back to the claim-level
    # P(CONTRADICTED) values. When the detector produced no probability at all
    # there is no refutation evidence, so it stays None: a 0.0 would falsely
    # assert that nothing was refuted.
    if not probability_available:
        contradiction_mass: float | None = None
    elif result.get("contradiction_mass") is not None:
        contradiction_mass = valid_score(result["contradiction_mass"])
    else:
        contradiction_mass = max(
            (
                valid_score(claim["contradicted_probability"]) or 0.0
                for claim in per_claim_results
                if not claim["non_factual"]
            ),
            default=0.0,
        )

    diagnostics = result.get("diagnostics") or {}
    return {
        # Kept under the historical key for backward compatibility. The value is
        # the detector's operational verification risk (P(CONTRADICTED) +
        # P(NOT_ENOUGH_INFO)), NOT a probability that the answer is false;
        # ``verification_risk`` below is the canonical name.
        "hallucination_probability": probability,
        "phase1": result.get("phase1"),
        "verification_risk": valid_score(result.get("verification_risk", probability)),
        # The only signal about actual refutation, kept separate from the
        # verification risk so downstream cannot mistake "unverified" for
        # "refuted". Verifier and Judge read this alongside the claim counts.
        "contradiction_mass": contradiction_mass,
        "probability_available": probability_available,
        "confidence_score": confidence,
        "risk_level": risk_level,
        "next_action": next_action,
        "model_source": str(result.get("model_source", "halluciguard_detector")),
        "status": status,
        "calibrated": calibrated,
        "calibration_applied": calibrated,
        "inference_executed": executed,
        "model_loaded": loaded,
        "model_version": result.get("model_version"),
        "calibrator_version": result.get("calibrator_version"),
        "detector_degraded": degraded,
        "grounded": grounded,
        "probability_semantics": result.get("probability_semantics"),
        "verification_reason": result.get("verification_reason"),
        "degraded_reason": diagnostics.get("degraded_reason") or result.get("degraded_reason"),
        "diagnostics": diagnostics,
        "warnings": result.get("warnings") or [],
        "per_claim_results": per_claim_results,
        "claim_count": result.get("claim_count", len(per_claim_results)),
        "supported_count": result.get("supported_count", 0),
        "contradicted_count": result.get("contradicted_count", 0),
        "unknown_count": result.get("unknown_count", 0),
        "non_factual_count": result.get("non_factual_count", 0),
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
