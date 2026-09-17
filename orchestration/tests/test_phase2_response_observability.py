"""Phase-2 observability tests: latency breakdown and rich response summary.

Covered defects:
  - #4  the response surfaced to the caller must summarize the multi-stage
        run (verification/judge/reverification) and where latency went
"""
from __future__ import annotations

import pytest

from orchestration.api import VerificationRequest, _execute_verification, _stage_breakdown


def test_stage_breakdown_sums_duplicates_in_trace_order():
    result = {
        "trace": [
            {"node": "detector", "latency_ms": 100},
            {"node": "verifier", "latency_ms": 200},
            {"node": "judge", "latency_ms": 50},
            {"node": "verifier", "latency_ms": 300},
        ]
    }
    breakdown = _stage_breakdown(result)
    by_stage = {entry["stage"]: entry["latency_ms"] for entry in breakdown}

    # "where did the latency go" — duplicate nodes (judge runs twice)
    # are summed into one stage entry.
    assert by_stage["verifier"] == 500
    assert by_stage["detector"] == 100
    assert by_stage["judge"] == 50
    # Order follows first trace appearance, not the dict iteration order.
    assert [entry["stage"] for entry in breakdown] == ["detector", "verifier", "judge"]


async def test_execute_verification_response_surface_summaries(monkeypatch):
    """The caller-facing response must expose the stage summaries plus the
    per-stage latency breakdown, not just a raw verifier blob."""
    from orchestration import api as orchestration_api

    async def fake_run_verification(**kwargs):
        return {
            "execution_id": "ex-99",
            "request_id": "req-99",
            "llm_response": "The draft answer.",
            "draft_response": "The draft answer.",
            "final_response": "The corrected answer.",
            "terminal_status": "verified",
            "answer_status": "confirmed",
            "correction_requested": False,
            "verification_summary": {"overall_status": "verified", "claims_extracted": 2},
            "judge_summary": {"status": "pass", "flagged": 0},
            "reverification_summary": {"passed": True, "remaining_contradictions": 0},
            "trace": [
                {"node": "verifier", "latency_ms": 40},
                {"node": "judge", "latency_ms": 30},
            ],
            "errors": [],
            "active_agents": ["detector", "verifier", "judge"],
            "disabled_agents": [],
        }

    monkeypatch.setattr(orchestration_api, "run_verification", fake_run_verification)

    request = VerificationRequest(
        user_query="Who created Python?",
        llm_response="The draft answer.",
        domain="general",
        request_id="req-99",
        generation_mode="normal",
    )
    resp = await _execute_verification(request)

    assert resp["verification_summary"]["overall_status"] == "verified"
    assert resp["judge_summary"]["status"] == "pass"
    assert resp["reverification_summary"]["passed"] is True
    assert resp["answer_status"] == "confirmed"
    assert resp["terminal_status"] == "verified"

    stages = {entry["stage"]: entry["latency_ms"] for entry in resp["stage_breakdown"]}
    assert stages == {"verifier": 40, "judge": 30}
    assert resp["total_latency_ms"] == 70