"""Regression test for the VERIFY_AGAIN -> human_review dead-terminal-branch bug.

Live repro: a correct answer whose draft decomposed into 1 verified + several
UNVERIFIED (no-evidence / conversational) claims looped VERIFY_AGAIN and was then
forced to human_review — because the graph incremented retry_count and the router
escalated at retry_count >= max_retries BEFORE the Judge's own terminal branch
(relaxed domain -> ACCEPT, strict -> ABSTAIN) could ever run. That branch was
dead code.

The fix makes the Judge node hand evaluate an effective retry_count of
max_retries on the final permitted pass, so the Judge issues its domain-aware
terminal decision instead of a VERIFY_AGAIN the router only converts to
human_review. These tests pin that:
  * early pass (retries remain)  -> VERIFY_AGAIN -> route verifier
  * final pass (retries exhausted, relaxed domain) -> ACCEPT -> route memory
    (NOT human_review)
"""
from __future__ import annotations

import os
import sys

import pytest

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import orchestration.graph as graph


def _verifier_result_one_verified_many_unverified() -> dict:
    """1 strongly-verified claim + 3 no-evidence UNVERIFIED claims (the live shape)."""
    return {
        "query_id": "Q-EIFFEL",
        "domain": "general",
        "overall_confidence": 0.0,  # min() over the unverified claims
        "status": "completed",
        "claim_reports": [
            {
                "claim_id": "c1",
                "claim_text": "The Eiffel Tower is located in Paris, France.",
                "verdict": "verified",
                "support_score": 0.95,
                "contradiction_score": 0.02,
                "confidence_score": 0.95,
                "evidence": [
                    {
                        "evidence_id": "E1",
                        "title": "Eiffel Tower",
                        "source": "wikipedia",
                        "snippet": "The Eiffel Tower is a lattice tower in Paris, France.",
                        "entailment_label": "entailment",
                        "entailment_score": 0.99,
                        "credibility_score": 0.95,
                    }
                ],
            },
            {"claim_id": "c2", "claim_text": "designed by Gustave Eiffel, completed 1889",
             "verdict": "unverified", "support_score": 0.1, "contradiction_score": 0.0,
             "confidence_score": 0.0, "evidence": []},
            {"claim_id": "c3", "claim_text": "Berlin has the Brandenburg Gate",
             "verdict": "unverified", "support_score": 0.1, "contradiction_score": 0.0,
             "confidence_score": 0.0, "evidence": []},
            {"claim_id": "c4", "claim_text": "feel free to ask about other locations",
             "verdict": "unverified", "support_score": 0.0, "contradiction_score": 0.0,
             "confidence_score": 0.0, "evidence": []},
        ],
    }


def _state(retry_count: int) -> dict:
    return {
        "execution_id": "exec-1",
        "request_id": "req-1",
        "user_query": "Where is the Eiffel Tower?",
        "llm_response": "The Eiffel Tower is located in Paris, France. ...",
        "draft_response": "The Eiffel Tower is located in Paris, France. ...",
        "domain": "general",
        "retry_count": retry_count,
        "max_retries": 2,
        "correction_attempt_count": 0,
        "verifier_result": _verifier_result_one_verified_many_unverified(),
        "detector_result": {
            "hallucination_probability": 0.9,
            "confidence_score": 0.9,
            "risk_level": "HIGH",
            "next_action": "Verify",
            "status": "completed",
        },
        "reverification_result": None,
        "inter_agent_bus": [],
        "trace": [],
    }


@pytest.mark.asyncio
async def test_early_pass_still_retries():
    out = await graph._judge_node(_state(retry_count=0))
    assert out["judge_decision"] == "VERIFY_AGAIN"
    assert graph._judge_route({**out, "max_retries": 2}) == "verifier"


@pytest.mark.asyncio
async def test_final_pass_reaches_terminal_accept_not_human_review():
    # retry_count = max_retries -> Judge receives actual retry_count
    # and makes its own terminal decision (ACCEPT on relaxed domain).
    out = await graph._judge_node(_state(retry_count=2))
    assert out["judge_decision"] == "ACCEPT", (
        f"terminal branch not reached; got {out['judge_decision']} ({out['judge'].get('reason')})"
    )
    assert graph._judge_route({**out, "max_retries": 2}) == "memory"
    assert out["terminal_status"] == "accepted"
