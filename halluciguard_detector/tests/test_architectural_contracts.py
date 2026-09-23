"""
Tests for architectural contract rules and boundaries.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from halluciguard_detector.detector import StandaloneDetector
from halluciguard_detector.schemas import RiskLevel, RoutingDecision, VerificationHint, StatusEnum


def test_supplied_claim_ids_and_order_preserved():
    detector = StandaloneDetector()
    supplied = [
        {"claim_id": "c_id_100", "text": "Java was created in 1995."},
        {"claim_id": "c_id_101", "text": "Sun Microsystems was the vendor."},
        {"claim_id": "c_id_102", "text": "James Gosling led the team."}
    ]
    resp = detector.detect("Who made Java?", "Java details...", supplied_claims=supplied)
    assert len(resp.claims) == 3
    assert [c.claim_id for c in resp.claims] == ["c_id_100", "c_id_101", "c_id_102"]


def test_one_claim_one_prediction():
    detector = StandaloneDetector()
    supplied = [{"claim_id": "single_01", "text": "Python was created by Guido."}]
    resp = detector.detect("What is Python?", "Python info", supplied_claims=supplied)
    assert len(resp.claims) == 1
    assert resp.claims[0].claim_id == "single_01"


def test_low_risk_never_changes_routing_to_accept():
    """Rule 3: Routing in v1 is ALWAYS VERIFY. LOW_RISK claims never route to ACCEPT."""
    detector = StandaloneDetector()
    resp = detector.detect("What is 2+2?", "2+2 equals 4.")
    assert resp.routing == RoutingDecision.VERIFY


def test_unknown_risk_gets_deep_verification_hint():
    """Rule 4: UNKNOWN risk claims must get DEEP verification hint."""
    detector = StandaloneDetector()
    supplied_45 = [{"claim_id": f"c_{i:03d}", "text": f"Fact {i}."} for i in range(1, 46)]
    resp = detector.detect("Query", "Answer", supplied_claims=supplied_45)
    overflow_claim = resp.claims[41]
    assert overflow_claim.risk_level == RiskLevel.UNKNOWN
    assert overflow_claim.verification_hint == VerificationHint.DEEP


def test_overall_risk_is_max_of_claim_scores():
    detector = StandaloneDetector()
    resp = detector.detect("Who made Java?", "Java was made by Gosling in 1995.")
    max_c_score = max(c.raw_score for c in resp.claims)
    assert resp.overall.risk_score == round(max_c_score, 4)
    assert resp.overall.aggregation == "max"
