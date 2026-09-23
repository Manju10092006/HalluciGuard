"""Detector integration seam for the orchestration graph.

This is the *single* place the graph decides which detector implementation runs.
The **sole production detector is DetV2 Stage-7** (the calibrated DeBERTa encoder,
vendored into this repo under ``detector_v2/`` with its trained weights fetched from
a pinned Hugging Face revision). There is no other detector: the legacy V1 agent has
been removed. If DetV2 fails at runtime the seam **fails closed** (routes to Verify)
rather than fabricating an Accept — it never silently substitutes a different model.

Design goals (all driven by the integration requirements):
  * Preserve the existing detector contract the graph consumes — the canonical six
    fields plus ``per_claim_results`` (V1 shape). DetV2 emits claim detail under a
    different key (``claims``); we translate it here so downstream Verifier/Judge
    behaviour is unchanged.
  * Load the heavy DetV2 encoder **once** (module-level singleton) instead of per
    request — a fresh ``DetectorAgent()`` per call would reload the encoder each time.
  * Fail closed: any error, or a DEGRADED/FAILED DetV2 run, routes to Verify and
    never fabricates an Accept.
  * No CWD-dependent or sibling imports: ``detector_v2`` is the copy vendored inside
    this repo; its encoder weights resolve from the pinned HF model revision (or a
    ``DETECTORV2_ENCODER_DIR`` override) and the calibrator from HalGuard's own
    ``models/stage7_calibration/``.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

# Per-claim risk tiering mirrors the calibrated tier defaults (low=0.30, high=0.50)
# so the claim-level ``risk_level`` the graph reads keeps a consistent meaning.
_LOW_TIER = 0.30
_HIGH_TIER = 0.50

_V2_AGENT: Any = None
_V2_LOCK = threading.Lock()


def _tier(prob: float) -> str:
    if prob >= _HIGH_TIER:
        return "HIGH"
    if prob < _LOW_TIER:
        return "LOW"
    return "MEDIUM"


def _get_v2_agent() -> Any:
    """Build (once) and return the cached DetV2 Stage-7 agent.

    Thread-safe: the graph runs detection in a worker thread, so first-call races
    must not build two encoders. Raises if ``detector_v2`` is not importable or its
    Stage-7 artifacts are present-but-broken (DetV2 itself raises loudly there).
    """
    global _V2_AGENT
    if _V2_AGENT is not None:
        return _V2_AGENT
    with _V2_LOCK:
        if _V2_AGENT is None:
            from detector_v2 import DetectorAgent as _V2DetectorAgent

            # Bare constructor -> DetV2's Stage-7 calibrated default (thresholds and
            # calibration owned entirely by DetV2; we do not touch them).
            _V2_AGENT = _V2DetectorAgent()
    return _V2_AGENT


def _map_v2_claims(ext: Dict[str, Any], overall_verify: bool, overall_prob: float) -> List[Dict[str, Any]]:
    """Translate DetV2's ``claims`` block into the ``per_claim_results`` shape the
    graph's ``_detector_node`` reads (claim_id/text/hallucination_probability/
    risk_level/requires_verification)."""
    results: List[Dict[str, Any]] = []
    for c in ext.get("claims") or []:
        text = str(c.get("text") or "").strip()
        if not text:
            continue
        cr = c.get("claim_risk")
        prob = float(cr) if cr is not None else float(overall_prob)
        level = _tier(prob)
        results.append({
            "claim_id": f"c{c.get('claim_id')}",
            "text": text,
            "hallucination_probability": prob,
            "risk_level": level,
            # A claim needs verification if the response routes to Verify or the
            # claim itself is not clearly low-risk. Fail toward verification.
            "requires_verification": bool(overall_verify or level != "LOW"),
        })
    return results


def _failclosed(model_source: str, reason: str) -> Dict[str, Any]:
    """Detector-contract result that can never be Accepted (fail-closed)."""
    return {
        "hallucination_probability": 0.0,   # honest: no trustworthy number
        "confidence_score": 0.0,
        "risk_level": "HIGH",
        "next_action": "Verify",
        "model_source": model_source,
        "status": "degraded",
        "detector_degraded": True,
        "degraded_reason": reason,
        "per_claim_results": [],
    }


def _run_v2(user_query: str, llm_response: str) -> Dict[str, Any]:
    """Run the DetV2 Stage-7 detector and map its output onto the graph contract."""
    agent = _get_v2_agent()
    ext = agent.detect(user_query, llm_response)  # extended dict (canonical six + claims)

    status = str(ext.get("status", "completed")).lower()
    completed = status == "completed"
    next_action = str(ext.get("next_action", "Verify"))
    risk_level = str(ext.get("risk_level", "HIGH")).upper()
    prob = ext.get("hallucination_probability")
    prob = float(prob) if prob is not None else 0.0
    conf = ext.get("confidence_score")
    conf = float(conf) if conf is not None else 0.0

    # Fail closed on a non-completed DetV2 run: never Accept a degraded result.
    if not completed:
        next_action = "Verify"
        risk_level = "HIGH"

    overall_verify = next_action.lower().endswith("verify")
    per_claim = _map_v2_claims(ext, overall_verify, prob)

    return {
        "hallucination_probability": prob,
        "confidence_score": conf,
        "risk_level": risk_level,
        "next_action": next_action,
        "model_source": str(ext.get("model_source", "detector_v2")),
        "status": status,
        "calibrated": bool(ext.get("calibrated", False)),
        "detector_degraded": not completed,
        "degraded_reason": (ext.get("diagnostics") or {}).get("degraded_reason") if isinstance(ext.get("diagnostics"), dict) else None,
        "per_claim_results": per_claim,
    }


def run_detection(user_query: str, llm_response: str) -> Any:
    """Single detector seam used by the orchestration graph.

    DetV2 is the **only** detector. On any DetV2 runtime failure we fail closed
    (Verify) — we never fabricate an Accept and never substitute a different model.

    Returns a dict (DetV2 output, or the fail-closed contract) normalized by the
    graph's ``_dump`` into the canonical contract.
    """
    try:
        return _run_v2(user_query, llm_response)
    except Exception as v2_exc:  # noqa: BLE001 - fail closed, never crash the graph
        return _failclosed("detector_v2_unavailable", f"v2_failed: {type(v2_exc).__name__}")
