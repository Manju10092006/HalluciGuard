"""Regression guards for two confirmed Judge arbiter bugs (audited 2026-09-20).

B1 — Dead confidence gate. Rule C accepted whenever every claim was VERIFIED,
     but never checked ``policy.accept_confidence_threshold``. A fully-verified
     answer at low overall_confidence was ACCEPTed verbatim even in a strict
     domain (Healthcare requires 0.88). Fix: Rule C now gates ACCEPT on the
     calibrated confidence, downgrading to VERIFY_AGAIN (budget remaining) or
     ABSTAIN (exhausted) below the threshold.

B4 — Narrow degraded-status guard. Only ``failed`` short-circuited to ABSTAIN;
     ``degraded``/``fallback``/``terminated_unresolved``/``skipped`` fell through
     and a degraded (untrustworthy) verification could be ACCEPTed as fully
     grounded. Fix: all non-authoritative terminal statuses ABSTAIN.

These assert real verdict behavior, deterministically, without the fine-tuned
model — they build canonical VerifierResults directly.
"""

from __future__ import annotations

import pytest

from orchestration.schemas import (
    ClaimReport,
    Evidence,
    EntailmentLabel,
    ExecutionStatus,
    JudgeDecision,
    VerdictLabel,
    VerifierResult,
)
from agents.judge_agent.judge_agent import JudgeAgent


QUERY = "What is the recommended dosage?"
RESPONSE = "The recommended dosage is 500 mg twice daily."


def _supporting_evidence() -> Evidence:
    return Evidence(
        evidence_id="ev-1",
        title="Clinical guideline",
        source="who.int",
        url="https://www.who.int/guideline",
        snippet="The recommended dosage is 500 mg twice daily.",
        entailment_label=EntailmentLabel.ENTAILMENT,
        entailment_score=0.95,
        credibility_score=0.9,
    )


def _all_verified_result(overall_confidence: float, domain: str = "healthcare") -> VerifierResult:
    claim = ClaimReport(
        claim_id="c1",
        claim_text=RESPONSE,
        verdict=VerdictLabel.VERIFIED,
        support_score=0.95,
        contradiction_score=0.02,
        confidence_score=overall_confidence,
        evidence=[_supporting_evidence()],
    )
    return VerifierResult(
        query_id="q1",
        domain=domain,
        claim_reports=[claim],
        evidence=[_supporting_evidence()],
        overall_confidence=overall_confidence,
        status=ExecutionStatus.COMPLETED,
    )


def _evaluate(result: VerifierResult, retry_count: int = 99):
    # retry_count high by default so the "budget exhausted" branch is exercised
    # (isolates the confidence gate from the retry path).
    return JudgeAgent().evaluate(
        verifier_result=result,
        user_query=QUERY,
        original_response=RESPONSE,
        domain=result.domain,
        retry_count=retry_count,
    )


# ---- B1: confidence gate on Rule C -----------------------------------------

def test_high_confidence_verified_is_accepted():
    """Above the domain threshold, all-verified still ACCEPTs (no regression)."""
    res = _evaluate(_all_verified_result(overall_confidence=0.97))
    assert res.decision in (JudgeDecision.ACCEPT, JudgeDecision.ACCEPT.value)


def test_low_confidence_verified_is_not_accepted_in_strict_domain():
    """The exact B1 failure: every claim VERIFIED but overall_confidence 0.30 in
    Healthcare (accept threshold 0.88). Must NOT ACCEPT."""
    res = _evaluate(_all_verified_result(overall_confidence=0.30))
    assert res.decision not in (JudgeDecision.ACCEPT, JudgeDecision.ACCEPT.value), (
        "under-grounded verified claim was ACCEPTed — accept_confidence_threshold "
        "is not being enforced"
    )
    # Budget exhausted -> fail closed to human review.
    assert res.decision in (JudgeDecision.ABSTAIN, JudgeDecision.ABSTAIN.value)


def test_low_confidence_verified_retries_when_budget_remains():
    """Below threshold with retry budget -> VERIFY_AGAIN, not ACCEPT."""
    res = _evaluate(_all_verified_result(overall_confidence=0.30), retry_count=0)
    assert res.decision in (JudgeDecision.VERIFY_AGAIN, JudgeDecision.VERIFY_AGAIN.value)


# ---- B4: non-authoritative status guard ------------------------------------

@pytest.mark.parametrize("bad_status", [
    ExecutionStatus.DEGRADED,
    ExecutionStatus.FALLBACK,
    ExecutionStatus.TERMINATED_UNRESOLVED,
    ExecutionStatus.SKIPPED,
    ExecutionStatus.FAILED,
])
def test_non_authoritative_status_abstains(bad_status):
    """A degraded/fallback/terminated/skipped/failed verification must ABSTAIN,
    never be treated as a clean verification and ACCEPTed."""
    result = _all_verified_result(overall_confidence=0.97)
    result.status = bad_status
    res = _evaluate(result)
    assert res.decision in (JudgeDecision.ABSTAIN, JudgeDecision.ABSTAIN.value), (
        f"status={bad_status} was treated as authoritative and not abstained"
    )
