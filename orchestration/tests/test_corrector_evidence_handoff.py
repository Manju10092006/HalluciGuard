"""Regression guard for the Judge -> Corrector evidence handoff.

Root cause fixed here: for a CONTRADICTED claim the refuting evidence (which
states the *correct* fact) is the grounding the Corrector needs. The Corrector's
binding contract only treats evidence as SUPPORTING when it arrives in
``trusted_evidence``; the Judge used to route it into ``contradictory_evidence``,
so every correction target was skipped as ``no_usable_evidence`` and the
Corrector returned the original response unchanged.

These tests assert, deterministically and without the fine-tuned model:
  1. the Judge emits CORRECT and puts the refuting evidence in trusted_evidence
     (not stranded in contradictory_evidence); and
  2. the resulting CorrectionRequest actually grounds to a correction target
     (``plan.has_targets`` — previously False).
"""

from __future__ import annotations

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

from agents.corrector_agent.corrector.adapter import to_internal_request
from agents.corrector_agent.corrector.config import CorrectorConfig
from agents.corrector_agent.corrector.evidence import ground_request

ORIGINAL = "Java was created by Snehith."
QUERY = "Who created Java?"


def _refuting_evidence() -> Evidence:
    # Refutes the hallucinated claim AND states the correct fact -> this is the
    # grounding a correction must be built on.
    return Evidence(
        evidence_id="ev-1",
        title="Java (programming language)",
        source="wikipedia",
        url="https://en.wikipedia.org/wiki/Java_(programming_language)",
        snippet="Java was created by James Gosling at Sun Microsystems and released in 1995.",
        entailment_label=EntailmentLabel.CONTRADICTION,
        entailment_score=0.95,
        credibility_score=0.9,
    )


def _contradicted_verifier_result() -> VerifierResult:
    claim = ClaimReport(
        claim_id="c1",
        claim_text=ORIGINAL,
        verdict=VerdictLabel.CONTRADICTED,
        support_score=0.05,
        contradiction_score=0.95,
        confidence_score=0.9,
        evidence=[_refuting_evidence()],
    )
    return VerifierResult(
        query_id="q1",
        domain="general",
        claim_reports=[claim],
        evidence=[_refuting_evidence()],
        overall_confidence=0.9,
        status=ExecutionStatus.COMPLETED,
    )


def _judge_correction_request():
    result = JudgeAgent().evaluate(
        verifier_result=_contradicted_verifier_result(),
        user_query=QUERY,
        original_response=ORIGINAL,
        domain="general",
    )
    return result


def test_judge_decides_correct_for_contradicted_claim():
    result = _judge_correction_request()
    assert result.decision in (JudgeDecision.CORRECT, JudgeDecision.CORRECT.value)
    assert result.correction_request is not None


def test_refuting_evidence_reaches_trusted_pool_not_contradictory():
    cr = _judge_correction_request().correction_request
    trusted_ids = {e.evidence_id for e in cr.trusted_evidence}
    contradictory_ids = {e.evidence_id for e in cr.contradictory_evidence}
    # The corrective evidence must be usable grounding (trusted pool)...
    assert "ev-1" in trusted_ids
    # ...and must NOT be stranded only in the contradictory pool, where the
    # corrector would classify it as unusable for grounding.
    assert "ev-1" not in contradictory_ids


def test_correction_request_grounds_to_a_target():
    """The exact failure point of the old behavior: the grounded plan had no
    targets, so the corrector returned the original verbatim. With refuting
    evidence in the trusted pool, the target now grounds."""
    cr = _judge_correction_request().correction_request
    internal = to_internal_request(cr)
    plan = ground_request(internal, CorrectorConfig())
    assert plan.has_targets, (
        "correction target was skipped as no_usable_evidence -> corrector would "
        "return the original unchanged"
    )
    grounded_ids = {
        ev.evidence.evidence_id
        for gt in plan.grounded_targets
        for ev in gt.supporting
    }
    assert "ev-1" in grounded_ids
