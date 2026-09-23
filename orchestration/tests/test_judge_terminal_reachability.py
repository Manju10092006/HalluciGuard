"""Regression test for the VERIFY_AGAIN -> human_review dead-terminal-branch bug.

Live repro: a correct answer whose draft decomposed into 1 verified + several
UNVERIFIED (no-evidence / conversational) claims looped VERIFY_AGAIN and was then
forced to human_review — because the graph incremented retry_count and the router
escalated at retry_count >= max_retries BEFORE the Judge's own terminal branch
could ever run. That branch was dead code.

The fix makes the Judge node hand `evaluate` an effective retry_count of
max_retries on the final permitted pass, so the Judge issues its own terminal
decision instead of a VERIFY_AGAIN the router only converts to human_review.

Under the criticality redesign the Judge classifies each unverified claim as
CORE (leaves a grounding gap the user asked about) or PERIPHERAL (incidental
detail already grounded elsewhere, or unrelated to the query). That makes the
two reachability guarantees below concrete:

  * a GENUINE core gap (the claim that answers the query is ungrounded) still
    retries while budget remains, then reaches a terminal ABSTAIN
    (human_escalation) — never an infinite VERIFY_AGAIN loop; and
  * a SUBSTANTIVELY-GROUNDED answer (core verified, only peripheral unverified
    detail) reaches a terminal ACCEPT and routes to memory — it never falls
    into the dead VERIFY_AGAIN -> human_review branch at all.
"""
from __future__ import annotations

import os
import sys

import pytest

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import orchestration.graph as graph


def _verifier_core_gap() -> dict:
    """The claim that ANSWERS the query is unverified with no verified coverage of
    the query anchors -> a genuine CORE grounding gap."""
    return {
        "query_id": "Q-EIFFEL-COREGAP",
        "domain": "general",
        "overall_confidence": 0.0,
        "status": "completed",
        "claim_reports": [
            {"claim_id": "c1", "claim_text": "The Eiffel Tower is located in Berlin, Germany.",
             "verdict": "unverified", "support_score": 0.1, "contradiction_score": 0.0,
             "confidence_score": 0.0, "evidence": []},
        ],
    }


def _verifier_grounded_with_peripheral() -> dict:
    """1 strongly-verified CORE claim (answers 'where') + peripheral UNVERIFIED
    detail the user did not ask about (designer/date, an unrelated Berlin fact,
    conversational filler)."""
    return {
        "query_id": "Q-EIFFEL",
        "domain": "general",
        "overall_confidence": 0.95,
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


def _state(retry_count: int, verifier_result: dict) -> dict:
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
        "verifier_result": verifier_result,
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
async def test_early_pass_still_retries_on_core_gap():
    # A genuine CORE grounding gap with retries remaining -> VERIFY_AGAIN, route verifier.
    out = await graph._judge_node(_state(retry_count=0, verifier_result=_verifier_core_gap()))
    assert out["judge_decision"] == "VERIFY_AGAIN", (
        f"expected retry on core gap; got {out['judge_decision']} ({out['judge'].get('reason')})"
    )
    assert graph._judge_route({**out, "max_retries": 2}) == "verifier"


@pytest.mark.asyncio
async def test_final_pass_core_gap_reaches_terminal_abstain_not_dead_loop():
    # Retries exhausted on a genuine core gap -> Judge issues its OWN terminal
    # decision (ABSTAIN -> human_escalation), never an infinite VERIFY_AGAIN loop.
    out = await graph._judge_node(_state(retry_count=2, verifier_result=_verifier_core_gap()))
    assert out["judge_decision"] == "ABSTAIN", (
        f"terminal branch not reached; got {out['judge_decision']} ({out['judge'].get('reason')})"
    )
    assert graph._judge_route({**out, "max_retries": 2}) == "human_escalation"


@pytest.mark.asyncio
async def test_grounded_answer_reaches_terminal_accept_not_human_review():
    # Core verified + only peripheral unverified detail: the answer is
    # substantively grounded, so the Judge reaches a terminal ACCEPT and routes
    # to memory -- it never enters the dead VERIFY_AGAIN -> human_review branch,
    # even on the final permitted pass.
    out = await graph._judge_node(
        _state(retry_count=2, verifier_result=_verifier_grounded_with_peripheral())
    )
    assert out["judge_decision"] == "ACCEPT", (
        f"grounded answer not accepted; got {out['judge_decision']} ({out['judge'].get('reason')})"
    )
    assert graph._judge_route({**out, "max_retries": 2}) == "memory"
    assert out["terminal_status"] == "accepted"
