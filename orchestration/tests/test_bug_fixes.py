"""Tests for all HalluciGuard bug fixes."""
from __future__ import annotations
import asyncio
import pytest
from pydantic import ValidationError
from orchestration.graph import (
    _aggregate_verification_status, _build_canonical_verifier_result,
    _validate_verdict, _get_verifier_imports,
)
from orchestration.schemas import (
    ClaimReport, ContractViolation, ErrorType, Evidence, ExecutionStatus,
    JudgeDecision, MemoryResult, MemoryStatus, PipelineStatus, VerdictLabel,
    VerificationStatus,
)
from orchestration.state import HalluciGuardState, add_error


class TestNoFabricatedScores:
    def test_evidence_missing_scores_use_zero(self):
        result = _build_canonical_verifier_result(
            verifier={"claim_evidence": [{"claim_id": "c1", "claim_text": "Test.", "verdict": "verified", "evidence": [{"evidence_id": "ev-1", "title": "Test", "source": "test", "snippet": "Test.", "entailment_label": "entailment"}]}]},
            query_id="q-001", domain="general",
        )
        ev = result.claim_reports[0].evidence[0]
        assert ev.entailment_score == 0.0
        assert ev.credibility_score == 0.0

    def test_claim_missing_scores_use_zero(self):
        result = _build_canonical_verifier_result(
            verifier={"claim_evidence": [{"claim_id": "c1", "claim_text": "Test.", "verdict": "verified", "evidence": []}]},
            query_id="q-002", domain="general",
        )
        assert result.claim_reports[0].support_score == 0.0
        assert result.claim_reports[0].confidence_score == 0.0

    def test_missing_overall_confidence_uses_zero(self):
        result = _build_canonical_verifier_result(
            verifier={"claim_evidence": [], },
            query_id="q-003", domain="general",
        )
        assert result.overall_confidence == 0.0


class TestStrictVerdictValidation:
    def test_valid_verdicts_pass(self):
        for v in ["verified", "contradicted", "unverified", "conflicted"]:
            assert _validate_verdict(v) == v
    def test_unknown_verdict_raises_contract_violation(self):
        with pytest.raises(ContractViolation):
            _validate_verdict("verifed")
    def test_build_canonical_result_raises_on_unknown_verdict(self):
        with pytest.raises(ContractViolation):
            _build_canonical_verifier_result(
                verifier={"claim_evidence": [{"verdict": "verifed", "claim_id": "c1", "claim_text": "Test.", "evidence": []}]},
                query_id="q-001", domain="general",
            )


class TestAnswerLevelAggregation:
    def test_all_verified(self):
        claims = [type("C", (), {"verdict": "verified"})(), type("C", (), {"verdict": VerdictLabel.VERIFIED})()]
        assert _aggregate_verification_status(claims) == VerificationStatus.ALL_VERIFIED.value
    def test_any_contradicted(self):
        claims = [type("C", (), {"verdict": "verified"})(), type("C", (), {"verdict": "contradicted"})()]
        assert _aggregate_verification_status(claims) == VerificationStatus.CONTRADICTED.value
    def test_partially_verified(self):
        claims = [type("C", (), {"verdict": "verified"})(), type("C", (), {"verdict": "unverified"})()]
        assert _aggregate_verification_status(claims) == VerificationStatus.PARTIALLY_VERIFIED.value
    def test_empty(self):
        assert _aggregate_verification_status([]) == VerificationStatus.UNVERIFIED.value


class TestMemoryResultCounting:
    def test_memory_result_counts(self):
        mem = MemoryResult(status=MemoryStatus.STORED, stored_count=5, duplicate_count=2, failed_count=1)
        assert mem.stored_count == 5 and mem.duplicate_count == 2 and mem.failed_count == 1
    def test_all_stored(self):
        mem = MemoryResult(status=MemoryStatus.STORED, stored_count=3)
        assert mem.stored_count == 3
    def test_all_failed(self):
        mem = MemoryResult(status=MemoryStatus.FAILED, stored_count=0, failed_count=2)
        assert mem.failed_count == 2


class TestStructuredErrorTaxonomy:
    def test_error_type_enum_values(self):
        assert ErrorType.TIMEOUT.value == "timeout"
        assert ErrorType.NETWORK.value == "network"
        assert ErrorType.CONTRACT.value == "contract"


class TestPipelineStatus:
    def test_pipeline_status_enum_values(self):
        assert PipelineStatus.RUNNING.value == "running"
        assert PipelineStatus.VERIFIED.value == "verified"
        assert PipelineStatus.HUMAN_REVIEW.value == "human_review"


class TestPartialVerificationSemantics:
    def test_verification_status_enum_values(self):
        assert VerificationStatus.ALL_VERIFIED.value == "all_verified"
        assert VerificationStatus.PARTIALLY_VERIFIED.value == "partially_verified"
        assert VerificationStatus.CONFLICTED.value == "conflicted"


class TestJudgeDecision:
    def test_judge_decision_enum_values(self):
        assert JudgeDecision.ACCEPT.value == "ACCEPT"
        assert JudgeDecision.CORRECT.value == "CORRECT"
        assert JudgeDecision.REJECT.value == "REJECT"
        assert JudgeDecision.VERIFY_AGAIN.value == "VERIFY_AGAIN"
        assert JudgeDecision.ABSTAIN.value == "ABSTAIN"


def test_get_verifier_imports_no_syspath_manipulation():
    import sys
    original_path = list(sys.path)
    Pipeline, SuspiciousClaim, VerifierInputV2 = _get_verifier_imports()
    assert sys.path == original_path
    assert Pipeline is not None


def test_state_has_pipeline_status():
    from orchestration.state import HalluciGuardState
    assert "pipeline_status" in HalluciGuardState.__annotations__


def test_state_has_separate_retry_fields():
    from orchestration.state import HalluciGuardState
    annotations = HalluciGuardState.__annotations__
    assert "retry_count" in annotations
    assert "correction_attempt_count" in annotations
    assert "reverification_attempt_count" in annotations
