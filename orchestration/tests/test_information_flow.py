"""Tests for the information-flow correctness fixes (defects 1-5).

Core principle: the Verifier verifies the claims the DRAFT ANSWER makes,
never the user query; the Judge maps decisions to answer-level statuses;
the Corrector runs only when a non-empty CorrectionRequest exists; and the
ReVerifier re-extracts and re-verifies claims from the corrected text.
"""
from __future__ import annotations

import pytest

from orchestration.graph import _reverifier_node, _verifier_node


class FakeSuspiciousClaim:
    def __init__(self, claim_id: str, text: str):
        self.claim_id = claim_id
        self.text = text


class FakeVerifierInputV2:
    def __init__(self, query_id: str, domain: str, suspicious_claims):
        self.query_id = query_id
        self.domain = domain
        self.suspicious_claims = suspicious_claims


class FakePipeline:
    """Captures every VerifierInputV2 it receives and returns verified reports."""

    calls: list = []

    @classmethod
    def reset(cls):
        cls.calls = []

    @classmethod
    async def verify(cls, payload):
        cls.calls.append(payload)
        return {
            "claim_evidence": [
                {
                    "claim_id": c.claim_id,
                    "claim_text": c.text,
                    "verdict": "VERIFIED",
                    "evidence": [
                        {
                            "snippet": "James Gosling created Java at Sun.",
                            "source": "Oracle",
                            "url": "https://example.com/java",
                            "entailment_label": "entailment",
                            "entailment_score": 0.95,
                            "credibility_score": 0.9,
                        }
                    ],
                }
                for c in payload.suspicious_claims
            ]
        }


@pytest.fixture(autouse=True)
def _patch_pipeline(monkeypatch):
    FakePipeline.reset()
    monkeypatch.setattr(
        "orchestration.graph._verifier_imports",
        lambda: (FakePipeline, FakeSuspiciousClaim, FakeVerifierInputV2),
    )


def _base_state(user_query: str, draft: str):
    return {
        "execution_id": "ex-1",
        "request_id": "req-1",
        "user_query": user_query,
        "llm_response": draft,
        "draft_response": draft,
        "domain": "general",
        "trace": [],
        "errors": [],
        "inter_agent_bus": [],
        "active_agents": ["base_llm", "detector", "verifier", "judge", "corrector", "reverifier", "memory"],
        "max_retries": 2,
        "retry_count": 0,
        "correction_attempt_count": 0,
        "reverification_attempt_count": 0,
    }


@pytest.mark.asyncio
async def test_verifier_verifies_draft_answer_claims_not_user_query():
    user_query = "Java was created by Snehith, right?"
    draft = "Java was created by James Gosling in 1991. It is an object-oriented language."

    result = await _verifier_node(_base_state(user_query, draft))

    claims_sent = [c.text for c in FakePipeline.calls[-1].suspicious_claims]
    joined = " ".join(claims_sent).lower()

    assert "snehith" not in joined, "user query must never be verified as the claim"
    assert any("james gosling" in c.lower() for c in claims_sent), "draft claims must be verified"
    assert result["draft_claims"] == claims_sent
    assert result["verification_summary"]["claims_extracted"] == len(claims_sent)
    assert result["verification_summary"]["claims_verified"] >= 1
    assert result["verification_summary"]["overall_status"] in {"verified", "unverified"}


@pytest.mark.asyncio
async def test_verifier_empty_draft_does_not_verify_user_query():
    user_query = "Java was created by Snehith, right?"
    draft = ""

    result = await _verifier_node(_base_state(user_query, draft))

    claims_sent = [c.text for c in FakePipeline.calls[-1].suspicious_claims]
    assert claims_sent == [], "empty draft must produce no claims to verify"


@pytest.mark.asyncio
async def test_reverifier_re_extracts_claims_from_corrected_text():
    state = _base_state(
        user_query="Java was created by Snehith, right?",
        draft="Java was created by Snehith.",
    )
    state["correction_result"] = {
        "corrected_text": "Java was created by James Gosling. It runs on the Java Virtual Machine.",
        "changed_claims": [{"text": "Java was created by Snehith"}],
    }

    result = await _reverifier_node(state)

    claims_sent = [c.text for c in FakePipeline.calls[-1].suspicious_claims]
    joined = " ".join(claims_sent).lower()

    assert "james gosling" in joined, "corrected text claims must be re-verified"
    assert "snehith" not in joined, "stale pre-correction claims must not be reverified"
    assert result["reverification_summary"]["claims_extracted"] == len(claims_sent)
    assert result["reverification_summary"]["passed"] is True
    assert result["reverification_summary"]["remaining_contradictions"] == 0


def test_judge_result_semantics_on_contradicted_draft_claim():
    from orchestration.schemas import (
        ClaimReport,
        ExecutionStatus,
        JudgeDecision,
        VerdictLabel,
        VerifierResult,
    )

    verifier = VerifierResult(
        query_id="q-1",
        domain="general",
        overall_confidence=0.4,
        status=ExecutionStatus.COMPLETED,
        claim_reports=[
            ClaimReport(
                claim_id="dc-1",
                claim_text="Java was created by Snehith.",
                verdict=VerdictLabel.CONTRADICTED,
                contradiction_score=0.9,
            )
        ],
    )

    from agents.judge_agent.judge_agent import JudgeAgent

    result = JudgeAgent().evaluate(
        verifier_result=verifier,
        user_query="Who created Java?",
        original_response="Java was created by Snehith.",
    )

    assert result.decision == JudgeDecision.CORRECT
    assert result.answer_status == "REQUIRES_CORRECTION"
    assert result.correction_requested is True
    assert result.correction_request is not None
    assert result.correction_request.claims_to_correct[0].claim_text == "Java was created by Snehith."


def test_judge_accepted_answer_has_accepted_status():
    from orchestration.schemas import (
        ClaimReport,
        ExecutionStatus,
        JudgeDecision,
        VerdictLabel,
        VerifierResult,
    )

    verifier = VerifierResult(
        query_id="q-1",
        domain="general",
        overall_confidence=0.9,
        status=ExecutionStatus.COMPLETED,
        claim_reports=[
            ClaimReport(
                claim_id="dc-1",
                claim_text="Java was created by James Gosling.",
                verdict=VerdictLabel.VERIFIED,
                support_score=0.9,
            )
        ],
    )

    from agents.judge_agent.judge_agent import JudgeAgent

    result = JudgeAgent().evaluate(
        verifier_result=verifier,
        user_query="Who created Java?",
        original_response="Java was created by James Gosling.",
    )

    assert result.decision == JudgeDecision.ACCEPT
    assert result.answer_status == "ACCEPTED"
    assert result.correction_requested is False
    assert result.correction_request is None


def test_judge_result_backward_compatible_defaults():
    from orchestration.schemas import JudgeResult, JudgeDecision

    legacy = JudgeResult(decision=JudgeDecision.ABSTAIN)
    assert legacy.answer_status == ""
    assert legacy.correction_requested is False