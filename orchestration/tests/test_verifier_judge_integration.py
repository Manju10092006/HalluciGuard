"""
Integration Test Suite for VerifierResult -> JudgeAgent -> JudgeResult Data Flow.
Verifies all Section 11 & 12 contract connection requirements.
"""

from __future__ import annotations

import sys
import os
import pytest

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

judge_dir = os.path.abspath(os.path.join(repo_root, "agents", "judge_agent"))
if judge_dir not in sys.path:
    sys.path.insert(0, judge_dir)

from agents.judge_agent.judge_agent import JudgeAgent
from orchestration.schemas import (
    VerifierResult,
    JudgeResult,
    CorrectionRequest,
    ClaimReport,
    Evidence,
    JudgeDecision,
    SeverityLevel,
    VerdictLabel,
    EntailmentLabel,
    ExecutionStatus,
)


def str_val(v):
    return v.value if hasattr(v, "value") else str(v)


judge = JudgeAgent()


# CASE 1: Single Verified Claim -> Judge ACCEPT
def test_integration_case1_verified_claim_accepts():
    v_result = VerifierResult(
        query_id="Q-V1",
        domain="General Knowledge",
        claim_reports=[
            ClaimReport(
                claim_id="C1",
                claim_text="Paris is the capital of France.",
                verdict=VerdictLabel.VERIFIED,
                support_score=0.98,
                contradiction_score=0.02,
                confidence_score=0.98,
                evidence=[
                    Evidence(
                        evidence_id="E1",
                        title="France Capital",
                        source="Wikipedia",
                        snippet="Paris is the capital and largest city of France.",
                        entailment_label=EntailmentLabel.ENTAILMENT,
                        entailment_score=0.99,
                        credibility_score=0.95,
                    )
                ],
            )
        ],
        evidence=[],
        overall_confidence=0.98,
        status=ExecutionStatus.COMPLETED,
    )

    # Validate VerifierResult schema
    validated_verifier = VerifierResult.model_validate(v_result.model_dump())
    assert validated_verifier.status == ExecutionStatus.COMPLETED

    # Pass directly into Judge
    j_result = judge.evaluate(
        verifier_result=validated_verifier,
        user_query="What is the capital of France?",
        original_response="Paris is the capital of France.",
    )

    # Validate JudgeResult schema
    validated_judge = JudgeResult.model_validate(j_result.model_dump())
    assert str_val(validated_judge.decision) == "ACCEPT"
    assert validated_judge.correction_request is None


# CASE 2: Contradicted Claim -> Judge CORRECT
def test_integration_case2_contradicted_claim_corrects():
    v_result = VerifierResult(
        query_id="Q-V2",
        domain="General Knowledge",
        claim_reports=[
            ClaimReport(
                claim_id="C1",
                claim_text="Python was created by Elon Musk in 1999.",
                verdict=VerdictLabel.CONTRADICTED,
                support_score=0.05,
                contradiction_score=0.95,
                confidence_score=0.95,
                evidence=[
                    Evidence(
                        evidence_id="E1",
                        title="Python History",
                        source="Wikipedia",
                        snippet="Python was created by Guido van Rossum in 1991.",
                        entailment_label=EntailmentLabel.CONTRADICTION,
                        entailment_score=0.98,
                        credibility_score=0.95,
                    )
                ],
            )
        ],
        evidence=[],
        overall_confidence=0.95,
        status=ExecutionStatus.COMPLETED,
    )

    validated_verifier = VerifierResult.model_validate(v_result.model_dump())
    j_result = judge.evaluate(
        verifier_result=validated_verifier,
        user_query="Who created Python?",
        original_response="Python was created by Elon Musk in 1999.",
    )

    validated_judge = JudgeResult.model_validate(j_result.model_dump())
    assert str_val(validated_judge.decision) == "CORRECT"
    assert validated_judge.correction_request is not None
    assert len(validated_judge.correction_request.claims_to_correct) == 1
    assert validated_judge.correction_request.claims_to_correct[0].claim_id == "C1"


# CASE 3: Unverified Claim -> Must NOT automatically become CORRECT
def test_integration_case3_unverified_claim_not_correct():
    v_result = VerifierResult(
        query_id="Q-V3",
        domain="General Knowledge",
        claim_reports=[
            ClaimReport(
                claim_id="C1",
                claim_text="An unverified claim with zero evidence.",
                verdict=VerdictLabel.UNVERIFIED,
                support_score=0.1,
                contradiction_score=0.0,
                confidence_score=0.4,
                evidence=[],
            )
        ],
        evidence=[],
        overall_confidence=0.4,
        status=ExecutionStatus.COMPLETED,
    )

    validated_verifier = VerifierResult.model_validate(v_result.model_dump())
    j_result = judge.evaluate(verifier_result=validated_verifier, retry_count=0)

    validated_judge = JudgeResult.model_validate(j_result.model_dump())
    assert str_val(validated_judge.decision) != "CORRECT"
    assert str_val(validated_judge.decision) in ["VERIFY_AGAIN", "ABSTAIN", "ACCEPT"]


# CASE 4: Multiple Claims (Claim 1 Verified, Claim 2 Contradicted, Claim 3 Verified)
def test_integration_case4_multiple_claims_preserves_verified():
    v_result = VerifierResult(
        query_id="Q-V4",
        domain="General Knowledge",
        claim_reports=[
            ClaimReport(
                claim_id="C1",
                claim_text="Python is high-level.",
                verdict=VerdictLabel.VERIFIED,
                support_score=0.9,
                contradiction_score=0.1,
                confidence_score=0.9,
                evidence=[Evidence(evidence_id="E1", title="Python", source="Docs", snippet="High-level", entailment_label=EntailmentLabel.ENTAILMENT)],
            ),
            ClaimReport(
                claim_id="C2",
                claim_text="Python was created by Steve Jobs.",
                verdict=VerdictLabel.CONTRADICTED,
                support_score=0.1,
                contradiction_score=0.9,
                confidence_score=0.9,
                evidence=[Evidence(evidence_id="E2", title="Python Creator", source="Wikipedia", snippet="Created by Guido van Rossum", entailment_label=EntailmentLabel.CONTRADICTION)],
            ),
            ClaimReport(
                claim_id="C3",
                claim_text="Python supports dynamic typing.",
                verdict=VerdictLabel.VERIFIED,
                support_score=0.9,
                contradiction_score=0.1,
                confidence_score=0.9,
                evidence=[Evidence(evidence_id="E3", title="Python Types", source="Docs", snippet="Dynamic typing", entailment_label=EntailmentLabel.ENTAILMENT)],
            ),
        ],
        evidence=[],
        overall_confidence=0.9,
        status=ExecutionStatus.COMPLETED,
    )

    validated_verifier = VerifierResult.model_validate(v_result.model_dump())
    j_result = judge.evaluate(
        verifier_result=validated_verifier,
        user_query="Tell me about Python.",
        original_response="Python is high-level. Python was created by Steve Jobs. Python supports dynamic typing.",
    )

    validated_judge = JudgeResult.model_validate(j_result.model_dump())
    assert str_val(validated_judge.decision) == "CORRECT"
    cr = validated_judge.correction_request
    assert cr is not None
    assert len(cr.claims_to_correct) == 1 and cr.claims_to_correct[0].claim_id == "C2"
    assert len(cr.claims_to_preserve) == 2 and set(c.claim_id for c in cr.claims_to_preserve) == {"C1", "C3"}


# CASE 5: Verifier Failure Handling
def test_integration_case5_verifier_failure_abstains():
    v_result = VerifierResult(
        query_id="Q-V5",
        domain="General Knowledge",
        claim_reports=[],
        evidence=[],
        overall_confidence=0.0,
        status=ExecutionStatus.FAILED,
    )

    validated_verifier = VerifierResult.model_validate(v_result.model_dump())
    j_result = judge.evaluate(verifier_result=validated_verifier)

    validated_judge = JudgeResult.model_validate(j_result.model_dump())
    assert str_val(validated_judge.decision) == "ABSTAIN"
    assert str_val(validated_judge.status) == "failed"
