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
    """Verified CORE claims + one PERIPHERAL unverified detail, general domain.
    The unverified claim is incidental to the query (an aside about a birth year
    when the user asked who founded a company), so it must be tolerated and the
    answer ACCEPTed on the FIRST pass, not thrash through VERIFY_AGAIN.

    Criticality — not a claim count — is what makes this safe: the claims that
    actually answer the query are grounded.
    """
    vr = _vr(
        "General Knowledge",
        [
            _cr("Oracle was founded in 1977", VerdictLabel.VERIFIED, 0.95, 0.02, 0.95),
            _cr("Oracle was founded by Larry Ellison", VerdictLabel.VERIFIED, 0.92, 0.03, 0.92),
            # Peripheral aside: the query never asked about the ocean's depth.
            _cr("The Pacific ocean is very deep", VerdictLabel.UNVERIFIED, 0.30, 0.05, 0.30),
        ],
    )
    r = JudgeAgent().evaluate(
        verifier_result=vr,
        user_query="Who founded Oracle and in what year?",
        original_response="resp",
        domain="General Knowledge", retry_count=0,
    )
    assert _str(r.decision) == "ACCEPT", f"got {_str(r.decision)} ({r.reason})"
    assert r.decision_basis == "PERIPHERAL_UNVERIFIED_TOLERATED_ACCEPT"
    # confidence still tracks the verifier, not discounted
    assert abs(r.confidence - 0.9) < 1e-4


def test_core_unverified_is_not_accepted():
    """The Lamborghini class: the claim that ANSWERS the query is unverified.
    Even though a peripheral verified claim is present, a CORE unverified claim
    must never be swept into an ACCEPT — route to VERIFY_AGAIN (then ABSTAIN)."""
    vr = _vr(
        "General Knowledge",
        [
            # Core to the query but ungrounded -> the whole answer is suspect.
            _cr("Lamborghini was founded in 1948", VerdictLabel.UNVERIFIED, 0.30, 0.05, 0.30),
            # A peripheral verified aside cannot rescue it.
            _cr("Italy is in Europe", VerdictLabel.VERIFIED, 0.95, 0.02, 0.95),
        ],
    )
    r = JudgeAgent().evaluate(
        verifier_result=vr,
        user_query="When was Lamborghini founded?",
        original_response="resp",
        domain="General Knowledge", retry_count=0,
    )
    assert _str(r.decision) != "ACCEPT", f"got {_str(r.decision)} ({r.reason})"
    assert _str(r.decision) == "VERIFY_AGAIN"
    assert r.decision_basis == "CORE_UNVERIFIED_RETRY"


def test_lamborghini_all_unverified_exhausted_abstains_never_accepts():
    """Regression fixture: Lamborghini answer, ALL claims unverified, retries
    exhausted. Must ABSTAIN — never ACCEPT on absence of contradiction."""
    vr = _vr(
        "General Knowledge",
        [
            _cr("Lamborghini was founded in 1948", VerdictLabel.UNVERIFIED, 0.3, 0.05, 0.3),
            _cr("Lamborghini was founded by Ferruccio", VerdictLabel.UNVERIFIED, 0.3, 0.05, 0.3),
        ],
    )
    r = JudgeAgent().evaluate(
        verifier_result=vr,
        user_query="Who founded Lamborghini and when?",
        original_response="resp",
        domain="General Knowledge", retry_count=99,
    )
    assert _str(r.decision) == "ABSTAIN", f"got {_str(r.decision)} ({r.reason})"
    assert r.decision_basis == "CORE_UNVERIFIED_ABSTAIN"


def test_empty_verifier_abstains_never_rejects_on_detector():
    """Zero-claim verifier + high detector prob, retries exhausted. Detector risk
    is a triage prior, not a factual verdict, so the outcome is ABSTAIN — never
    REJECT."""
    vr = _vr("General Knowledge", [])
    r = JudgeAgent().evaluate(
        verifier_result=vr,
        detector_result={"hallucination_probability": 0.95, "confidence_score": 0.9},
        user_query="q", original_response="resp",
        domain="General Knowledge", retry_count=99,
    )
    assert _str(r.decision) == "ABSTAIN", f"got {_str(r.decision)} ({r.reason})"
    assert _str(r.decision) != "REJECT"
    assert r.decision_basis == "NO_EVIDENCE_ABSTAIN"


def test_reverification_failed_with_remaining_contradictions_never_accepts():
    """Vietnam class: reverification passed=False (or remaining>0) must NOT ACCEPT
    even if remaining count is read as 0 in a malformed payload — the gate needs
    BOTH passed AND remaining==0."""
    r = JudgeAgent().evaluate(
        verifier_result=_vr("General Knowledge", []),
        reverification_result={"passed": False, "remaining_contradictions": 0},
        user_query="q", original_response="resp",
        domain="General Knowledge", retry_count=99,
    )
    assert _str(r.decision) != "ACCEPT", f"got {_str(r.decision)} ({r.reason})"
    assert _str(r.decision) == "REJECT"


def test_reverification_passed_accepts():
    """The passing gate: passed AND remaining==0 -> ACCEPT with the stable basis."""
    r = JudgeAgent().evaluate(
        verifier_result=_vr("General Knowledge", []),
        reverification_result={"passed": True, "remaining_contradictions": 0},
        user_query="q", original_response="resp",
        domain="General Knowledge", retry_count=0,
    )
    assert _str(r.decision) == "ACCEPT"
    assert r.decision_basis == "REVERIFICATION_PASSED_ACCEPT"


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


def test_russia_count_tie_does_not_auto_accept():
    """Russia class: verified count == unverified count (a tie). The former
    count-majority rule (verified >= unverified) would ACCEPT on the tie. The
    corrected criticality model must NOT — the unverified claim is core to the
    query, so a real grounding gap remains."""
    vr = _vr(
        "General Knowledge",
        [
            _cr("Russia is the largest country by area", VerdictLabel.VERIFIED, 0.95, 0.02, 0.95),
            _cr("Russia has a population of 900 million", VerdictLabel.UNVERIFIED, 0.30, 0.05, 0.30),
        ],
    )
    r = JudgeAgent().evaluate(
        verifier_result=vr,
        user_query="What is Russia's area and population?",
        original_response="resp",
        domain="General Knowledge", retry_count=0,
    )
    assert _str(r.decision) != "ACCEPT", f"got {_str(r.decision)} ({r.reason})"
    assert _str(r.decision) == "VERIFY_AGAIN"


def test_degraded_verifier_status_never_accepts():
    """F2: a DEGRADED verifier run (retrieval returned zero passages) is NOT
    authoritative, even when the claim verdicts look VERIFIED. The Judge must
    ABSTAIN rather than ACCEPT a verdict resting on an ungrounded run.
    Guards NO_EVIDENCE != TRUE."""
    vr = VerifierResult(
        query_id="Q",
        domain="General Knowledge",
        claim_reports=[
            _cr("Paris is the capital of France", VerdictLabel.VERIFIED, 0.95, 0.02, 0.95),
        ],
        evidence=[],
        overall_confidence=0.9,
        status=ExecutionStatus.DEGRADED,
    )
    r = JudgeAgent().evaluate(
        verifier_result=vr,
        user_query="What is the capital of France?",
        original_response="Paris is the capital of France.",
        domain="General Knowledge", retry_count=0,
    )
    assert _str(r.decision) == "ABSTAIN", f"got {_str(r.decision)} ({r.reason})"


def test_failed_verifier_status_never_accepts():
    """F2 companion: a FAILED verifier run must ABSTAIN (unchanged behavior,
    guarded so the widened status check does not regress the failed path)."""
    vr = VerifierResult(
        query_id="Q",
        domain="General Knowledge",
        claim_reports=[
            _cr("Paris is the capital of France", VerdictLabel.VERIFIED, 0.95, 0.02, 0.95),
        ],
        evidence=[],
        overall_confidence=0.9,
        status=ExecutionStatus.FAILED,
    )
    r = JudgeAgent().evaluate(
        verifier_result=vr,
        user_query="What is the capital of France?",
        original_response="Paris is the capital of France.",
        domain="General Knowledge", retry_count=0,
    )
    assert _str(r.decision) == "ABSTAIN", f"got {_str(r.decision)} ({r.reason})"
