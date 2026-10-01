"""Regression (audit M5/#11): the HG-012 decision-grade evidence gate is applied
PER CLAIM, not as a whole-response abort. A genuine CONTRADICTED claim (with a
contradiction-labelled snippet) must still route to CORRECT even when an unrelated
VERIFIED claim lacks an entailment-labelled snippet; and a lone weakly-grounded
decisive claim is demoted to unverified rather than aborting the evaluation.
"""
from __future__ import annotations

import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from agents.judge_agent.judge_agent import JudgeAgent, JudgeDecision, DecisionBasis
from orchestration.schemas import (
    VerifierResult,
    ClaimReport,
    VerdictLabel,
    ExecutionStatus,
    Evidence,
)


def _ev(label):
    return Evidence(evidence_id=f"e-{label}", source="fixture", snippet="a non-empty snippet", entailment_label=label)


def _cr(cid, verdict, sup, con, label):
    return ClaimReport(claim_id=cid, claim_text=cid, verdict=verdict, support_score=sup,
                       contradiction_score=con, confidence_score=max(sup, con), evidence=[_ev(label)])


def _evaluate(reports):
    result = VerifierResult(query_id="q", domain="general", claim_reports=reports,
                            evidence=[e for r in reports for e in r.evidence],
                            overall_confidence=0.6, status=ExecutionStatus.COMPLETED)
    return JudgeAgent().evaluate(
        verifier_result=result, detector_result={}, user_query="q",
        original_response="draft", domain="general", reverification_result=None,
        retry_count=0, correction_attempt_count=0,
    )


def test_weak_verified_claim_does_not_suppress_a_contradiction():
    contra = _cr("c1", VerdictLabel.CONTRADICTED, 0.0, 0.9, "contradiction")
    weak_verified = _cr("c2", VerdictLabel.VERIFIED, 0.6, 0.0, "neutral")  # no entailment snippet
    res = _evaluate([contra, weak_verified])
    assert res.decision == JudgeDecision.CORRECT
    assert res.decision_basis != DecisionBasis.INVALID_VERIFIER_INPUT


def test_lone_weakly_grounded_verified_is_demoted_not_gate_aborted():
    weak_verified = _cr("c1", VerdictLabel.VERIFIED, 0.6, 0.0, "neutral")
    res = _evaluate([weak_verified])
    # The whole-response gate-abort (INVALID_VERIFIER_INPUT) must NOT fire.
    assert res.decision_basis != DecisionBasis.INVALID_VERIFIER_INPUT
