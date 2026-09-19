"""
End-to-End Test Suite for Canonical Judge Integration & Orchestration.
Covers Section 21 Requirements (TEST A through TEST J).
"""

from __future__ import annotations

import os
import sys
import pytest

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from orchestration.graph import build_verification_graph
from orchestration.state import add_trace
from orchestration.schemas import (
    VerifierResult,
    ClaimReport,
    Evidence,
    JudgeResult,
    CorrectionRequest,
    CorrectionResult,
    ReverificationResult,
    VerdictLabel,
    EntailmentLabel,
    ExecutionStatus,
    JudgeDecision,
)


def base_state():
    return {
        "execution_id": "exec-100",
        "request_id": "req-100",
        "user_query": "Tell me about Python.",
        "llm_response": "Python is high-level. Python was created by Steve Jobs. Python supports dynamic typing.",
        "draft_response": "Python is high-level. Python was created by Steve Jobs. Python supports dynamic typing.",
        "domain": "General Knowledge",
        "trace": [],
        "errors": [],
        "inter_agent_bus": [],
        "retry_count": 0,
        "max_retries": 2,
        "correction_attempt_count": 0,
        "active_agents": ["base_llm", "detector", "verifier", "judge", "corrector", "reverifier", "memory"],
        "disabled_agents": [],
    }


def str_val(v):
    return v.value if hasattr(v, "value") else str(v)


# TEST A: Verified claim -> Judge ACCEPT -> Memory
@pytest.mark.asyncio
async def test_e2e_test_a_verified_claim_accepts():
    v_result = VerifierResult(
        query_id="Q-A",
        domain="General Knowledge",
        claim_reports=[
            ClaimReport(
                claim_id="C1",
                claim_text="Guido van Rossum created Python.",
                verdict=VerdictLabel.VERIFIED,
                support_score=0.95,
                contradiction_score=0.05,
                confidence_score=0.95,
                evidence=[Evidence(evidence_id="E1", title="Wiki", source="Wikipedia", snippet="Created by Guido in 1991.", entailment_label=EntailmentLabel.ENTAILMENT)],
            )
        ],
        overall_confidence=0.95,
        status=ExecutionStatus.COMPLETED,
    )

    graph = build_verification_graph(
        node_overrides={
            "detector": lambda s: {"route": "verify", "trace": add_trace(s, "detector", "completed")},
            "verifier": lambda s: {"verifier_result": v_result.model_dump(), "verifier": {"claim_evidence": v_result.model_dump()["claim_reports"]}, "trace": add_trace(s, "verifier", "completed")},
        }
    )
    result = await graph.ainvoke({**base_state(), "llm_response": "Guido van Rossum created Python."})
    nodes = [e["node"] for e in result["trace"]]
    assert "judge" in nodes
    assert "memory" in nodes
    assert result["judge_decision"] == "ACCEPT"


# TEST B & TEST E: Mixed Claims (C1 Verified, C2 Contradicted, C3 Verified) -> Judge CORRECT -> Selective Correction
@pytest.mark.asyncio
async def test_e2e_test_b_and_e_mixed_claims_selective_correction():
    v_result = VerifierResult(
        query_id="Q-BE",
        domain="General Knowledge",
        claim_reports=[
            ClaimReport(
                claim_id="C1",
                claim_text="Python is high-level.",
                verdict=VerdictLabel.VERIFIED,
                support_score=0.95,
                confidence_score=0.95,
                evidence=[Evidence(evidence_id="E1", title="Doc", source="Python.org", snippet="High level", entailment_label=EntailmentLabel.ENTAILMENT)],
            ),
            ClaimReport(
                claim_id="C2",
                claim_text="Python was created by Steve Jobs.",
                verdict=VerdictLabel.CONTRADICTED,
                support_score=0.05,
                contradiction_score=0.95,
                confidence_score=0.95,
                evidence=[Evidence(evidence_id="E2", title="Wiki", source="Wikipedia", snippet="Created by Guido van Rossum", entailment_label=EntailmentLabel.CONTRADICTION)],
            ),
            ClaimReport(
                claim_id="C3",
                claim_text="Python supports dynamic typing.",
                verdict=VerdictLabel.VERIFIED,
                support_score=0.95,
                confidence_score=0.95,
                evidence=[Evidence(evidence_id="E3", title="Doc", source="Python.org", snippet="Dynamic typing", entailment_label=EntailmentLabel.ENTAILMENT)],
            ),
        ],
        overall_confidence=0.95,
        status=ExecutionStatus.COMPLETED,
    )

    graph = build_verification_graph(
        node_overrides={
            "detector": lambda s: {"route": "verify", "trace": add_trace(s, "detector", "completed")},
            "verifier": lambda s: {"verifier_result": v_result.model_dump(), "verifier": {"claim_evidence": v_result.model_dump()["claim_reports"]}, "trace": add_trace(s, "verifier", "completed")},
        }
    )

    result = await graph.ainvoke(base_state())
    nodes = [e["node"] for e in result["trace"]]
    assert "judge" in nodes
    assert "corrector" in nodes
    assert "reverifier" in nodes

    corr_req = result.get("correction_request")
    assert corr_req is not None
    claims_corr = corr_req.get("claims_to_correct", [])
    claims_pres = corr_req.get("claims_to_preserve", [])
    assert len(claims_corr) == 1 and claims_corr[0]["claim_id"] == "C2"
    assert len(claims_pres) == 2 and set(c["claim_id"] for c in claims_pres) == {"C1", "C3"}


# TEST C: Unverified claim -> NOT automatically CORRECT
@pytest.mark.asyncio
async def test_e2e_test_c_unverified_claim_not_correct():
    v_result = VerifierResult(
        query_id="Q-C",
        domain="General Knowledge",
        claim_reports=[
            ClaimReport(
                claim_id="C1",
                claim_text="Unverified statement.",
                verdict=VerdictLabel.UNVERIFIED,
                support_score=0.1,
                contradiction_score=0.0,
                confidence_score=0.4,
                evidence=[],
            )
        ],
        overall_confidence=0.4,
        status=ExecutionStatus.COMPLETED,
    )

    graph = build_verification_graph(
        node_overrides={
            "detector": lambda s: {"route": "verify", "trace": add_trace(s, "detector", "completed")},
            "verifier": lambda s: {"verifier_result": v_result.model_dump(), "verifier": {"claim_evidence": []}, "trace": add_trace(s, "verifier", "completed")},
        }
    )

    result = await graph.ainvoke({**base_state(), "llm_response": "Unverified statement."})
    assert result["judge_decision"] != "CORRECT"


# TEST D: Conflicted claim -> safe handling
@pytest.mark.asyncio
async def test_e2e_test_d_conflicted_claim_safe_handling():
    v_result = VerifierResult(
        query_id="Q-D",
        domain="General Knowledge",
        claim_reports=[
            ClaimReport(
                claim_id="C1",
                claim_text="Conflicted fact.",
                verdict=VerdictLabel.CONFLICTED,
                support_score=0.5,
                contradiction_score=0.5,
                confidence_score=0.5,
                evidence=[],
            )
        ],
        overall_confidence=0.5,
        status=ExecutionStatus.COMPLETED,
    )

    graph = build_verification_graph(
        node_overrides={
            "detector": lambda s: {"route": "verify", "trace": add_trace(s, "detector", "completed")},
            "verifier": lambda s: {"verifier_result": v_result.model_dump(), "verifier": {"claim_evidence": []}, "trace": add_trace(s, "verifier", "completed")},
        }
    )

    result = await graph.ainvoke({**base_state(), "llm_response": "Conflicted fact."})
    assert result["judge_decision"] in ["VERIFY_AGAIN", "ABSTAIN", "REJECT"]


# TEST F: Verifier failure -> ABSTAIN (Never ACCEPT)
@pytest.mark.asyncio
async def test_e2e_test_f_verifier_failure_abstains():
    v_result = VerifierResult(
        query_id="Q-F",
        domain="General Knowledge",
        claim_reports=[],
        evidence=[],
        overall_confidence=0.0,
        status=ExecutionStatus.FAILED,
    )

    graph = build_verification_graph(
        node_overrides={
            "detector": lambda s: {"route": "verify", "trace": add_trace(s, "detector", "completed")},
            "verifier": lambda s: {"verifier_result": v_result.model_dump(), "verifier": {}, "trace": add_trace(s, "verifier", "completed")},
        }
    )

    result = await graph.ainvoke(base_state())
    assert result["judge_decision"] == "ABSTAIN"
    assert "human_escalation" in [e["node"] for e in result["trace"]]


# TEST G: Reverification success -> ACCEPT -> Memory
@pytest.mark.asyncio
async def test_e2e_test_g_reverification_success_accepts():
    v_result = VerifierResult(
        query_id="Q-G",
        domain="General Knowledge",
        claim_reports=[ClaimReport(claim_id="C1", claim_text="Python was created by Guido.", verdict=VerdictLabel.VERIFIED, support_score=0.95, confidence_score=0.95)],
        overall_confidence=0.95,
        status=ExecutionStatus.COMPLETED,
    )

    rev_result = ReverificationResult(passed=True, verifier_result=v_result, remaining_contradictions=0)

    # Invoke judge node directly with reverification_result
    graph = build_verification_graph(
        node_overrides={
            "detector": lambda s: {"route": "verify", "trace": add_trace(s, "detector", "completed")},
            "verifier": lambda s: {"verifier_result": v_result.model_dump(), "reverification_result": rev_result.model_dump(), "trace": add_trace(s, "verifier", "completed")},
        }
    )

    result = await graph.ainvoke({**base_state(), "reverification_result": rev_result.model_dump()})
    assert result["judge_decision"] == "ACCEPT"


# TEST H: Reverification failure -> REJECT when retries exhausted
@pytest.mark.asyncio
async def test_e2e_test_h_reverification_failure_rejects():
    v_result = VerifierResult(
        query_id="Q-H",
        domain="General Knowledge",
        claim_reports=[ClaimReport(claim_id="C1", claim_text="Python is compiled to C++.", verdict=VerdictLabel.CONTRADICTED, contradiction_score=0.95, confidence_score=0.95)],
        overall_confidence=0.2,
        status=ExecutionStatus.COMPLETED,
    )

    rev_result = ReverificationResult(passed=False, verifier_result=v_result, remaining_contradictions=1)

    graph = build_verification_graph(
        node_overrides={
            "detector": lambda s: {"route": "verify", "trace": add_trace(s, "detector", "completed")},
            "verifier": lambda s: {"verifier_result": v_result.model_dump(), "reverification_result": rev_result.model_dump(), "trace": add_trace(s, "verifier", "completed")},
        }
    )

    result = await graph.ainvoke({**base_state(), "reverification_result": rev_result.model_dump()})
    assert result["judge_decision"] == "REJECT"


# TEST I: Missing correction request -> fail closed / human escalation
@pytest.mark.asyncio
async def test_e2e_test_i_missing_correction_request_fails_closed():
    graph = build_verification_graph(
        node_overrides={
            "detector": lambda s: {"route": "verify", "trace": add_trace(s, "detector", "completed")},
            "verifier": lambda s: {"verifier_result": {"query_id": "q", "domain": "general", "claim_reports": []}, "trace": add_trace(s, "verifier", "completed")},
            "judge": lambda s: {"judge_decision": "CORRECT", "correction_request": None, "route": "human_escalation", "trace": add_trace(s, "judge", "completed")},
        }
    )

    result = await graph.ainvoke(base_state())
    nodes = [e["node"] for e in result["trace"]]
    assert "corrector" not in nodes
    assert "human_escalation" in nodes


# TEST J: No infinite loop (bounded retries terminate safely)
@pytest.mark.asyncio
async def test_e2e_test_j_no_infinite_loop_bounded_retries():
    # Force empty claims causing VERIFY_AGAIN
    graph = build_verification_graph(
        node_overrides={
            "detector": lambda s: {"route": "verify", "trace": add_trace(s, "detector", "completed")},
            "verifier": lambda s: {"verifier_result": {"query_id": "q", "domain": "general", "claim_reports": []}, "verifier": {"claim_evidence": []}, "trace": add_trace(s, "verifier", "completed")},
        }
    )

    result = await graph.ainvoke({**base_state(), "retry_count": 0, "max_retries": 2})
    nodes = [e["node"] for e in result["trace"]]
    assert nodes.count("verifier") <= 4
    assert "human_escalation" in nodes or "reject" in nodes
