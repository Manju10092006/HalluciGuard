"""
Unit tests for canonical supervisor inter-agent contracts.

Covers:
  1. Valid DetectorResult
  2. Invalid probability values (boundary / validation enforcement)
  3. Valid Evidence
  4. Valid ClaimReport
  5. Valid VerifierResult
  6. Valid JudgeResult
  7. Valid CorrectionRequest
  8. Valid CorrectionResult
  9. Valid ReverificationResult
  10. Valid MemoryResult
  11. HalluciGuardState containing all canonical contracts
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
from pydantic import ValidationError

from orchestration.schemas import (
    ClaimReport,
    CorrectionRequest,
    CorrectionResult,
    DetectorResult,
    Evidence,
    ExecutionStatus,
    JudgeDecision,
    JudgeResult,
    MemoryResult,
    MemoryStatus,
    NextAction,
    ReverificationResult,
    RiskLevel,
    SeverityLevel,
    ValidationStatus,
    VerdictLabel,
    VerifierResult,
)
from orchestration.state import HalluciGuardState


# ---------------------------------------------------------------------------
# 1. DetectorResult Tests
# ---------------------------------------------------------------------------

def test_valid_detector_result():
    detector = DetectorResult(
        hallucination_probability=0.15,
        confidence_score=0.85,
        risk_level=RiskLevel.LOW,
        next_action=NextAction.ACCEPT,
        model_source="halueval-distilbert",
    )
    assert detector.hallucination_probability == 0.15
    assert detector.confidence_score == 0.85
    assert detector.risk_level == "LOW"
    assert detector.next_action == "Accept"
    assert detector.status == "completed"
    assert detector.model_source == "halueval-distilbert"

    dumped = detector.model_dump()
    assert dumped["risk_level"] == "LOW"
    assert dumped["next_action"] == "Accept"


# ---------------------------------------------------------------------------
# 2. Invalid Probability & Score Boundaries Tests
# ---------------------------------------------------------------------------

def test_invalid_probability_values():
    # Negative hallucination probability
    with pytest.raises(ValidationError):
        DetectorResult(
            hallucination_probability=-0.01,
            confidence_score=0.85,
            risk_level=RiskLevel.LOW,
            next_action=NextAction.ACCEPT,
        )

    # Hallucination probability > 1.0
    with pytest.raises(ValidationError):
        DetectorResult(
            hallucination_probability=1.05,
            confidence_score=0.85,
            risk_level=RiskLevel.HIGH,
            next_action=NextAction.VERIFY,
        )

    # Negative confidence score
    with pytest.raises(ValidationError):
        DetectorResult(
            hallucination_probability=0.20,
            confidence_score=-0.5,
            risk_level=RiskLevel.LOW,
            next_action=NextAction.ACCEPT,
        )

    # Confidence score > 1.0
    with pytest.raises(ValidationError):
        DetectorResult(
            hallucination_probability=0.20,
            confidence_score=1.5,
            risk_level=RiskLevel.LOW,
            next_action=NextAction.ACCEPT,
        )


# ---------------------------------------------------------------------------
# 3. Evidence Contract Tests
# ---------------------------------------------------------------------------

def test_valid_evidence():
    ev = Evidence(
        evidence_id="ev-101",
        title="Python Overview",
        source="wikipedia",
        url="https://en.wikipedia.org/wiki/Python_(programming_language)",
        snippet="Python was created by Guido van Rossum and first released in 1991.",
        entailment_label="entailment",
        entailment_score=0.98,
        credibility_score=0.95,
    )
    assert ev.evidence_id == "ev-101"
    assert ev.source == "wikipedia"
    assert ev.entailment_label == "entailment"
    assert ev.entailment_score == 0.98
    assert ev.credibility_score == 0.95

    # Test invalid score boundary on Evidence
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="ev-bad",
            source="test",
            snippet="snippet",
            entailment_label="neutral",
            entailment_score=1.2,
        )


# ---------------------------------------------------------------------------
# 4. ClaimReport Contract Tests
# ---------------------------------------------------------------------------

def test_valid_claim_report():
    ev = Evidence(
        evidence_id="ev-1",
        source="wikipedia",
        snippet="Paris is the capital of France.",
        entailment_label="entailment",
        entailment_score=0.99,
        credibility_score=0.95,
    )
    claim = ClaimReport(
        claim_id="c1",
        claim_text="Paris is the capital of France.",
        verdict=VerdictLabel.VERIFIED,
        support_score=0.99,
        contradiction_score=0.01,
        confidence_score=0.95,
        evidence=[ev],
    )
    assert claim.claim_id == "c1"
    assert claim.verdict == "verified"
    assert len(claim.evidence) == 1
    assert claim.evidence[0].evidence_id == "ev-1"


# ---------------------------------------------------------------------------
# 5. VerifierResult Contract Tests
# ---------------------------------------------------------------------------

def test_valid_verifier_result():
    ev = Evidence(
        evidence_id="ev-1",
        source="wikipedia",
        snippet="Paris is the capital of France.",
        entailment_label="entailment",
        entailment_score=0.99,
        credibility_score=0.95,
    )
    claim = ClaimReport(
        claim_id="c1",
        claim_text="Paris is the capital of France.",
        verdict=VerdictLabel.VERIFIED,
        support_score=0.99,
        contradiction_score=0.01,
        confidence_score=0.95,
        evidence=[ev],
    )
    res = VerifierResult(
        query_id="q-1001",
        domain="general",
        claim_reports=[claim],
        evidence=[ev],
        overall_confidence=0.95,
        retrieved_sources_count=3,
        verified_sources_count=1,
        status=ExecutionStatus.COMPLETED,
    )
    assert res.query_id == "q-1001"
    assert res.domain == "general"
    assert len(res.claim_reports) == 1
    assert res.retrieved_sources_count == 3
    assert res.verified_sources_count == 1
    assert res.status == "completed"


# ---------------------------------------------------------------------------
# 6. JudgeResult Contract Tests
# ---------------------------------------------------------------------------

def test_valid_judge_result():
    # Case A: Clean ACCEPT decision without correction payload
    accept_judge = JudgeResult(
        decision=JudgeDecision.ACCEPT,
        severity=SeverityLevel.LOW,
        reason="All claims strongly supported by authoritative evidence.",
        explanation="Calibrated Bayesian confidence 0.96 exceeds acceptance threshold.",
        confidence=0.96,
        correction_request=None,
    )
    assert accept_judge.decision == "ACCEPT"
    assert accept_judge.severity == "LOW"
    assert accept_judge.confidence == 0.96
    assert accept_judge.correction_request is None

    # Case B: CORRECT decision with attached CorrectionRequest
    cr = CorrectionRequest(
        execution_id="exec-42",
        user_query="Who created Python?",
        original_response="Python was created by Elon Musk.",
        claims_to_correct=[
            ClaimReport(
                claim_id="c1",
                claim_text="Python was created by Elon Musk.",
                verdict=VerdictLabel.CONTRADICTED,
                contradiction_score=0.98,
            )
        ],
        claims_to_preserve=[],
        trusted_evidence=[
            Evidence(
                evidence_id="ev-1",
                source="wikipedia",
                snippet="Python was created by Guido van Rossum.",
                entailment_label="contradiction",
                entailment_score=0.98,
                credibility_score=0.90,
            )
        ],
        contradictory_evidence=[],
        correction_instructions="Replace creator claim with Guido van Rossum.",
    )
    correct_judge = JudgeResult(
        decision=JudgeDecision.CORRECT,
        severity=SeverityLevel.HIGH,
        reason="Direct contradiction of core entity creator.",
        explanation="Factually refuted by authoritative ground-truth.",
        confidence=0.92,
        correction_request=cr,
    )
    assert correct_judge.decision == "CORRECT"
    assert correct_judge.correction_request is not None
    assert correct_judge.correction_request.execution_id == "exec-42"


# ---------------------------------------------------------------------------
# 7. CorrectionRequest Contract Tests
# ---------------------------------------------------------------------------

def test_valid_correction_request():
    cr = CorrectionRequest(
        execution_id="exec-123",
        user_query="Tell me about Python and the sky.",
        original_response="The sky is blue. Python was created by Elon Musk.",
        claims_to_correct=[
            ClaimReport(
                claim_id="c2",
                claim_text="Python was created by Elon Musk.",
                verdict=VerdictLabel.CONTRADICTED,
            )
        ],
        claims_to_preserve=[
            ClaimReport(
                claim_id="c1",
                claim_text="The sky is blue.",
                verdict=VerdictLabel.VERIFIED,
            )
        ],
        trusted_evidence=[
            Evidence(
                evidence_id="ev-guido",
                source="python_docs",
                snippet="Guido van Rossum developed Python.",
                entailment_label="contradiction",
            )
        ],
        contradictory_evidence=[],
        correction_instructions="Preserve c1 exactly. Rewrite c2 using ev-guido.",
    )
    assert cr.execution_id == "exec-123"
    assert len(cr.claims_to_correct) == 1
    assert len(cr.claims_to_preserve) == 1
    assert len(cr.trusted_evidence) == 1
    assert "Preserve c1" in cr.correction_instructions


# ---------------------------------------------------------------------------
# 8. CorrectionResult Contract Tests
# ---------------------------------------------------------------------------

def test_valid_correction_result():
    res = CorrectionResult(
        original_text="Python was created by Elon Musk.",
        corrected_text="Python was created by Guido van Rossum.",
        changed_claims=[
            {
                "claim_id": "c2",
                "action": "rewritten_with_evidence",
                "original": "Python was created by Elon Musk.",
                "corrected": "Python was created by Guido van Rossum.",
            }
        ],
        validation_status=ValidationStatus.VALID,
        attempt_count=1,
        status=ExecutionStatus.COMPLETED,
    )
    assert res.original_text == "Python was created by Elon Musk."
    assert res.corrected_text == "Python was created by Guido van Rossum."
    assert res.validation_status == "valid"
    assert res.attempt_count == 1
    assert res.status == "completed"


# ---------------------------------------------------------------------------
# 9. ReverificationResult Contract Tests
# ---------------------------------------------------------------------------

def test_valid_reverification_result():
    ev = Evidence(
        evidence_id="ev-1",
        source="docs",
        snippet="Guido van Rossum created Python.",
        entailment_label="entailment",
        entailment_score=0.99,
        credibility_score=0.95,
    )
    vr = VerifierResult(
        query_id="re-verify-1",
        domain="general",
        claim_reports=[
            ClaimReport(
                claim_id="c1",
                claim_text="Python was created by Guido van Rossum.",
                verdict=VerdictLabel.VERIFIED,
                support_score=0.99,
                evidence=[ev],
            )
        ],
        evidence=[ev],
        overall_confidence=0.98,
        retrieved_sources_count=1,
        verified_sources_count=1,
    )
    reverif = ReverificationResult(
        passed=True,
        verifier_result=vr,
        remaining_contradictions=0,
        status=ExecutionStatus.COMPLETED,
    )
    assert reverif.passed is True
    assert reverif.remaining_contradictions == 0
    assert reverif.verifier_result.overall_confidence == 0.98
    assert reverif.status == "completed"


# ---------------------------------------------------------------------------
# 10. MemoryResult Contract Tests
# ---------------------------------------------------------------------------

def test_valid_memory_result():
    # Successful storage
    mem_success = MemoryResult(
        status=MemoryStatus.STORED,
        stored_count=2,
        fact_ids=["fact-101", "fact-102"],
        reason="Verified facts committed to knowledge graph and vector memory.",
    )
    assert mem_success.status == "stored"
    assert mem_success.stored_count == 2
    assert len(mem_success.fact_ids) == 2

    # Skipped storage
    mem_skipped = MemoryResult(
        status=MemoryStatus.SKIPPED,
        stored_count=0,
        fact_ids=[],
        reason="no_verified_claims_to_persist",
    )
    assert mem_skipped.status == "skipped"
    assert mem_skipped.stored_count == 0


# ---------------------------------------------------------------------------
# 11. HalluciGuardState Containing Canonical Contracts
# ---------------------------------------------------------------------------

def test_halluciguard_state_contains_canonical_contracts():
    detector = DetectorResult(
        hallucination_probability=0.08,
        confidence_score=0.92,
        risk_level=RiskLevel.LOW,
        next_action=NextAction.ACCEPT,
    )
    ev = Evidence(
        evidence_id="ev-1",
        source="wikipedia",
        snippet="Paris is the capital of France.",
        entailment_label="entailment",
    )
    verifier = VerifierResult(
        query_id="req-1",
        claim_reports=[
            ClaimReport(
                claim_id="c1",
                claim_text="Paris is capital of France",
                verdict=VerdictLabel.VERIFIED,
                evidence=[ev],
            )
        ],
        evidence=[ev],
    )
    judge = JudgeResult(
        decision=JudgeDecision.ACCEPT,
        severity=SeverityLevel.LOW,
        confidence=0.95,
    )
    memory = MemoryResult(
        status=MemoryStatus.STORED,
        stored_count=1,
        fact_ids=["f-1"],
    )

    state: HalluciGuardState = {
        "execution_id": "ex-001",
        "request_id": "req-001",
        "user_query": "What is the capital of France?",
        "llm_response": "Paris is the capital of France.",
        "draft_response": "Paris is the capital of France.",
        "final_response": "Paris is the capital of France.",
        "terminal_status": "accepted",
        "domain": "general",
        # Canonical contracts
        "detector_result": detector,
        "verifier_result": verifier,
        "judge_result": judge,
        "memory_result": memory,
        # Legacy backward-compatibility fields
        "detector": detector.model_dump(),
        "verifier": verifier.model_dump(),
        "judge": judge.model_dump(),
        "memory": memory.model_dump(),
        "trace": [],
        "errors": [],
        "inter_agent_bus": [],
    }

    assert state["execution_id"] == "ex-001"
    assert state["detector_result"].risk_level == "LOW"
    assert state["verifier_result"].query_id == "req-1"
    assert state["judge_result"].decision == "ACCEPT"
    assert state["memory_result"].stored_count == 1
    assert state["detector"]["risk_level"] == "LOW"
