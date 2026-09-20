"""Regression tests for JudgeAgent decision-tree defects.

Main defect (confirmed): a substantively-correct answer whose atomic-claim
decomposition yielded one noisy UNVERIFIED sub-claim (alongside genuinely
verified claims, no contradictions) was routed to VERIFY_AGAIN on every pass.
Because decomposition is deterministic the fragment stayed unverified, so the
retry budget was burned on passes with a predetermined outcome, only to ACCEPT
at the very end. Rule C2 now accepts verified-dominant answers immediately in
MODERATE/RELAXED domains, while STRICT/VERY_STRICT keep the conservative path.
"""
from __future__ import annotations

import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from agents.judge_agent.judge_agent import JudgeAgent, _verdict_value
from orchestration.schemas import (
    VerifierResult,
    ClaimReport,
    VerdictLabel,
    ExecutionStatus,
)


def _str(v):
    return v.value if hasattr(v, "value") else str(v)


def _cr(cid, verdict, sup, con, conf):
    return ClaimReport(
        claim_id=cid,
        claim_text=cid,
        verdict=verdict,
        support_score=sup,
        contradiction_score=con,
        confidence_score=conf,
        evidence=[],
    )


def _vr(domain, claims, confidence=0.9):
    return VerifierResult(
        query_id="Q",
        domain=domain,
        claim_reports=claims,
        evidence=[],
        overall_confidence=confidence,
        status=ExecutionStatus.COMPLETED,
    )


def test_verified_dominant_with_minor_unverified_accepts_immediately():
    """The main-defect scenario: 2 verified + 1 noisy unverified, general domain.
    Must ACCEPT on the FIRST pass (retry_count=0), not burn VERIFY_AGAIN retries."""
    vr = _vr(
        "General Knowledge",
        [
            _cr("c1", VerdictLabel.VERIFIED, 0.95, 0.02, 0.95),
            _cr("c2", VerdictLabel.VERIFIED, 0.92, 0.03, 0.92),
            _cr("c3-garbage", VerdictLabel.UNVERIFIED, 0.30, 0.05, 0.30),
        ],
    )
    r = JudgeAgent().evaluate(
        verifier_result=vr, user_query="q", original_response="resp",
        domain="General Knowledge", retry_count=0,
    )
    assert _str(r.decision) == "ACCEPT", f"got {_str(r.decision)} ({r.reason})"
    # confidence still tracks the verifier, not discounted
    assert abs(r.confidence - 0.9) < 1e-4


def test_unverified_majority_is_not_accepted():
    """Guard: if unverified claims outnumber verified, do NOT tolerate — the answer
    is mostly ungrounded, so fall through to VERIFY_AGAIN."""
    vr = _vr(
        "General Knowledge",
        [
            _cr("c1", VerdictLabel.VERIFIED, 0.95, 0.02, 0.95),
            _cr("c2", VerdictLabel.UNVERIFIED, 0.30, 0.05, 0.30),
            _cr("c3", VerdictLabel.UNVERIFIED, 0.25, 0.05, 0.25),
        ],
    )
    r = JudgeAgent().evaluate(
        verifier_result=vr, user_query="q", original_response="resp",
        domain="General Knowledge", retry_count=0,
    )
    assert _str(r.decision) == "VERIFY_AGAIN", f"got {_str(r.decision)} ({r.reason})"


def test_strict_domain_does_not_tolerate_unverified():
    """Healthcare (VERY_STRICT): a minor unverified claim must still NOT be
    silently accepted on the first pass — safety path is preserved."""
    vr = _vr(
        "Healthcare",
        [
            _cr("c1", VerdictLabel.VERIFIED, 0.95, 0.02, 0.95),
            _cr("c2", VerdictLabel.VERIFIED, 0.93, 0.02, 0.93),
            _cr("c3", VerdictLabel.UNVERIFIED, 0.30, 0.05, 0.30),
        ],
    )
    r = JudgeAgent().evaluate(
        verifier_result=vr, user_query="q", original_response="resp",
        domain="Healthcare", retry_count=0,
    )
    assert _str(r.decision) == "VERIFY_AGAIN", f"got {_str(r.decision)} ({r.reason})"


def test_conflicted_claim_is_not_tolerated():
    """A genuine CONFLICTED claim (evidence both ways) must NOT be swept under the
    verified-dominant rule — conflicts are real ambiguity, not decomposition noise."""
    vr = _vr(
        "General Knowledge",
        [
            _cr("c1", VerdictLabel.VERIFIED, 0.95, 0.02, 0.95),
            _cr("c2", VerdictLabel.VERIFIED, 0.90, 0.03, 0.90),
            _cr("c3", VerdictLabel.CONFLICTED, 0.55, 0.55, 0.50),
        ],
    )
    r = JudgeAgent().evaluate(
        verifier_result=vr, user_query="q", original_response="resp",
        domain="General Knowledge", retry_count=0,
    )
    assert _str(r.decision) == "VERIFY_AGAIN", f"got {_str(r.decision)} ({r.reason})"


def test_all_verified_still_accepts():
    """Rule C unchanged: a fully-verified answer accepts."""
    vr = _vr(
        "General Knowledge",
        [
            _cr("c1", VerdictLabel.VERIFIED, 0.95, 0.02, 0.95),
            _cr("c2", VerdictLabel.VERIFIED, 0.93, 0.02, 0.93),
        ],
    )
    r = JudgeAgent().evaluate(
        verifier_result=vr, user_query="q", original_response="resp",
        domain="General Knowledge", retry_count=0,
    )
    assert _str(r.decision) == "ACCEPT"


def test_contradiction_still_corrects():
    """Rule A unchanged for a non-critical domain: a contradiction -> CORRECT."""
    vr = _vr(
        "General Knowledge",
        [
            _cr("c1", VerdictLabel.VERIFIED, 0.9, 0.05, 0.9),
            _cr("c2", VerdictLabel.CONTRADICTED, 0.05, 0.9, 0.9),
        ],
    )
    r = JudgeAgent().evaluate(
        verifier_result=vr, user_query="q", original_response="resp",
        domain="General Knowledge", retry_count=0,
    )
    assert _str(r.decision) == "CORRECT"
    assert r.correction_request is not None


def test_verdict_value_helper_handles_enum_and_string():
    """The robust extractor must map both a raw enum and a plain string to the
    canonical lowercase value (guards the enum-vs-name comparison latent bug)."""
    assert _verdict_value(VerdictLabel.CONTRADICTED) == "contradicted"
    assert _verdict_value("contradicted") == "contradicted"
    assert _verdict_value("VERIFIED") == "verified"
