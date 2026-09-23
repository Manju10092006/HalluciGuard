"""
Test output contract and schemas.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from halluciguard_detector.detector import StandaloneDetector
from halluciguard_detector.schemas import DetectorResponse, RiskLevel, RoutingDecision, VerificationHint, StatusEnum


def test_detector_contract_output():
    detector = StandaloneDetector()
    resp = detector.detect(
        user_query="Who created Java?",
        draft_answer="Java was created by James Gosling in 1995."
    )
    assert isinstance(resp, DetectorResponse)
    assert resp.request_id == "HG-0001"
    assert resp.routing == RoutingDecision.VERIFY
    assert resp.overall.aggregation == "max"
    assert len(resp.claims) >= 1
    assert all(c.claim_id.startswith("claim_") for c in resp.claims)
    assert all(c.verification_hint in (VerificationHint.STANDARD, VerificationHint.DEEP) for c in resp.claims)


def test_legacy_adapter_never_converts_uncertain_to_no_hallucination():
    detector = StandaloneDetector()
    resp = detector.detect(
        user_query="What is 2+2?",
        draft_answer="2+2 is 4."
    )
    legacy = resp.to_legacy()
    assert "label" in legacy
    assert "probability" in legacy
    assert "risk" in legacy
    # CRITICAL AUDIT RULE: Uncertainty must NEVER be converted to NO_HALLUCINATION
    assert legacy["label"] != "NO_HALLUCINATION"
    assert legacy["label"] in ("HIGH_RISK_REVERIFY", "LOW_RISK_TENTATIVE", "UNKNOWN_UNCERTAIN")


def test_supplied_claims_preserves_ids():
    detector = StandaloneDetector()
    supplied = [
        {"claim_id": "custom_001", "text": "Java was created in 1995."},
        {"claim_id": "custom_002", "text": "Sun Microsystems developed Java."}
    ]
    resp = detector.detect(
        user_query="Tell me about Java.",
        draft_answer="Java was created in 1995.",
        supplied_claims=supplied
    )
    assert len(resp.claims) == 2
    assert resp.claims[0].claim_id == "custom_001"
    assert resp.claims[1].claim_id == "custom_002"
