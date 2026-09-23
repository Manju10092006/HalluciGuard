"""
Test fail-closed behavior.
No failure mode must EVER produce LOW_RISK.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from halluciguard_detector.detector import StandaloneDetector
from halluciguard_detector.schemas import RiskLevel, StatusEnum, RoutingDecision, VerificationHint


def test_empty_answer_fails_closed():
    detector = StandaloneDetector()
    resp = detector.detect(user_query="Who created Java?", draft_answer="")
    assert resp.status == StatusEnum.FAILED
    assert "EMPTY_ANSWER" in resp.degraded_reasons
    assert resp.overall.risk_level != RiskLevel.LOW_RISK
    assert resp.routing == RoutingDecision.VERIFY


def test_whitespace_answer_fails_closed():
    detector = StandaloneDetector()
    resp = detector.detect(user_query="Who created Java?", draft_answer="   \n  \t  ")
    assert resp.status == StatusEnum.FAILED
    assert "EMPTY_ANSWER" in resp.degraded_reasons
    assert resp.overall.risk_level != RiskLevel.LOW_RISK


def test_missing_calibration_emits_degraded():
    detector = StandaloneDetector()
    resp = detector.detect(
        user_query="Who created Java?",
        draft_answer="Java was created by James Gosling."
    )
    assert "NO_CALIBRATION" in resp.degraded_reasons
    assert resp.status == StatusEnum.DEGRADED
    assert all(c.calibrated_probability is None for c in resp.claims)
    assert all(c.risk_level != RiskLevel.LOW_RISK for c in resp.claims)


def test_claim_cap_overflow():
    detector = StandaloneDetector()
    supplied_45 = [{"claim_id": f"claim_{i:03d}", "text": f"Claim text number {i}."} for i in range(1, 46)]
    resp = detector.detect(
        user_query="Tell me facts.",
        draft_answer="Many facts...",
        supplied_claims=supplied_45
    )
    assert "CLAIM_CAP" in resp.degraded_reasons
    # Claims past index 40 must have risk_level UNKNOWN and flag CLAIM_CAP
    assert len(resp.claims) == 45
    overflow_claim = resp.claims[41]
    assert overflow_claim.risk_level == RiskLevel.UNKNOWN
    assert "CLAIM_CAP" in overflow_claim.flags
