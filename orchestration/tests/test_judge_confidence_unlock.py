"""Regression tests reproducing the exact failure the user hit on-screen.

Scenario (the "Eiffel Tower" run):
  * Base LLM produced a correct, well-grounded answer.
  * Verifier returned verdict=VERIFIED but a MODEST evidence confidence (~0.20).
  * Detector (a near-constant ~0.9 classifier) reported hallucination_probability=0.94.
  * OLD build: judge confidence = verifier_conf * (1 - 0.2*det_prob) = ~0.16,
    then a Rule-C accept_confidence_threshold gate (0.75) blocked it ->
    VERIFY_AGAIN -> human_review, even though the answer was correct.

These tests pin the corrected behaviour on current main:
  1. A VERIFIED claim ACCEPTs regardless of a high (untrustworthy) detector prob.
  2. Judge confidence comes SOLELY from the verifier (no detector discount).
"""
from __future__ import annotations

import os
import sys

import pytest

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from agents.judge_agent.judge_agent import JudgeAgent
from orchestration.schemas import (
    VerifierResult,
    ClaimReport,
    Evidence,
    DetectorResult,
    VerdictLabel,
    EntailmentLabel,
    ExecutionStatus,
)


def _str(v):
    return v.value if hasattr(v, "value") else str(v)


def _verified_result(confidence: float) -> VerifierResult:
    return VerifierResult(
        query_id="Q-EIFFEL",
        domain="General Knowledge",
        claim_reports=[
            ClaimReport(
                claim_id="c1",
                claim_text="The Eiffel Tower is located in Paris.",
                verdict=VerdictLabel.VERIFIED,
                support_score=0.95,
                contradiction_score=0.02,
                confidence_score=confidence,
                evidence=[
                    Evidence(
                        evidence_id="E1",
                        title="Eiffel Tower",
                        source="Wikipedia",
                        snippet="Paris hosts the Eiffel Tower, its most recognizable landmark.",
                        entailment_label=EntailmentLabel.ENTAILMENT,
                        entailment_score=0.97,
                        credibility_score=0.95,
                    )
                ],
            )
        ],
        evidence=[],
        overall_confidence=confidence,
        status=ExecutionStatus.COMPLETED,
    )


def _saturated_detector() -> DetectorResult:
    # The broken near-constant classifier: ~0.94 hallucination prob on a true answer.
    return DetectorResult(
        hallucination_probability=0.94,
        confidence_score=0.94,
        risk_level="HIGH",
        next_action="Verify",
        model_source="halueval-detector-final",
        status="completed",
    )


def test_verified_answer_accepts_despite_saturated_detector():
    """The core unlock: a VERIFIED claim must ACCEPT even when the miscalibrated
    detector screams 0.94 hallucination. Detector is triage, not a veto."""
    judge = JudgeAgent()
    result = judge.evaluate(
        verifier_result=_verified_result(confidence=0.90),
        detector_result=_saturated_detector(),
        user_query="Where is the Eiffel Tower?",
        original_response="The Eiffel Tower is located in Paris.",
    )
    assert _str(result.decision) == "ACCEPT", (
        f"expected ACCEPT, got {_str(result.decision)} ({result.reason})"
    )


def test_judge_confidence_is_not_discounted_by_detector():
    """Judge confidence must equal the verifier's confidence, NOT be multiplied
    down by the detector probability (the old (1 - 0.2*det_prob) coupling)."""
    judge = JudgeAgent()
    verifier_conf = 0.80
    result = judge.evaluate(
        verifier_result=_verified_result(confidence=verifier_conf),
        detector_result=_saturated_detector(),
        user_query="Where is the Eiffel Tower?",
        original_response="The Eiffel Tower is located in Paris.",
    )
    # Old coupling would give 0.80 * (1 - 0.2*0.94) = 0.6496.
    # Correct behaviour: confidence tracks the verifier alone.
    assert result.confidence == pytest.approx(verifier_conf, abs=1e-4), (
        f"detector still discounts judge confidence: {result.confidence}"
    )


def test_modest_confidence_verified_answer_is_not_blocked():
    """Even at the modest ~0.20 evidence confidence the user's run produced, a
    VERIFIED verdict must NOT be blocked by a resurrected accept-threshold gate."""
    judge = JudgeAgent()
    result = judge.evaluate(
        verifier_result=_verified_result(confidence=0.20),
        detector_result=_saturated_detector(),
        user_query="Where is the Eiffel Tower?",
        original_response="The Eiffel Tower is located in Paris.",
    )
    assert _str(result.decision) == "ACCEPT", (
        f"a verified answer was blocked at modest confidence: "
        f"{_str(result.decision)} ({result.reason})"
    )
