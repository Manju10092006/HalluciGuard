"""Regression tests for the F2 fix: canonical Verifier status must be derived
from the per-stage pipeline status, NOT hardcoded to COMPLETED.

Invariants guarded here:
  * NO_EVIDENCE != TRUE        (a degraded retrieval run is not authoritative)
  * PROVIDER_FAILURE != verdict (a failed stage is not authoritative)

Before the fix, `_build_canonical_verifier_result` hardcoded
`status=ExecutionStatus.COMPLETED`, so an ungrounded Verifier run (retrieval
returned zero passages -> stage "degraded") was presented to the Judge and
Re-Verifier as a fully authoritative, completed investigation.
"""
from __future__ import annotations

from orchestration.graph import _build_canonical_verifier_result
from orchestration.schemas import ExecutionStatus


def _verifier(stage_status: str | None):
    """Minimal raw verifier dict with one verified claim and, optionally, one stage."""
    v: dict = {
        "query_id": "q1",
        "domain": "general",
        "claim_evidence": [
            {
                "claim_id": "c1",
                "claim_text": "Paris is the capital of France.",
                "verdict": "verified",
                "support_score": 0.9,
                "contradiction_score": 0.0,
                "confidence_score": 0.9,
                "evidence": [],
            }
        ],
        "overall_confidence": 0.9,
    }
    if stage_status is not None:
        v["pipeline_stages"] = [{"stage": "retrieval", "status": stage_status}]
    return v


def test_degraded_stage_yields_degraded_status():
    res = _build_canonical_verifier_result(_verifier("degraded"), "q1", "general")
    assert str(res.status).lower().endswith("degraded"), res.status


def test_failed_stage_yields_failed_status():
    res = _build_canonical_verifier_result(_verifier("failed"), "q1", "general")
    assert str(res.status).lower().endswith("failed"), res.status


def test_failed_takes_precedence_over_degraded():
    v = _verifier(None)
    v["pipeline_stages"] = [
        {"stage": "retrieval", "status": "degraded"},
        {"stage": "nli", "status": "failed"},
    ]
    res = _build_canonical_verifier_result(v, "q1", "general")
    assert str(res.status).lower().endswith("failed"), res.status


def test_all_completed_stages_yield_completed():
    v = _verifier(None)
    v["pipeline_stages"] = [{"stage": "retrieval", "status": "completed"}]
    res = _build_canonical_verifier_result(v, "q1", "general")
    assert res.status == ExecutionStatus.COMPLETED or str(res.status).lower().endswith("completed")


def test_missing_pipeline_stages_defaults_completed():
    # Backward compatibility: legacy raw output without pipeline_stages.
    res = _build_canonical_verifier_result(_verifier(None), "q1", "general")
    assert res.status == ExecutionStatus.COMPLETED or str(res.status).lower().endswith("completed")
