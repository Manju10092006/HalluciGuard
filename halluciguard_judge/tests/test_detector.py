"""
Tests for JudgeDetector (degraded mode — no trained model needed).
Run: pytest halluciguard_judge/tests/test_detector.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import patch, MagicMock
from halluciguard_judge.detector import JudgeDetector
from halluciguard_judge.models import RiskLevel, RoutingDecision, DetectorOutput


@pytest.fixture(autouse=True)
def reset_shared_state():
    """Reset JudgeDetector shared state before each test."""
    JudgeDetector._shared_classifier = None
    JudgeDetector._shared_extractor = None
    JudgeDetector._classifier_loaded = False
    yield
    JudgeDetector._shared_classifier = None
    JudgeDetector._shared_extractor = None
    JudgeDetector._classifier_loaded = False


def test_empty_query_returns_degraded():
    detector = JudgeDetector()
    result = detector.detect(user_query="", llm_response="Java was created by Gosling.")
    assert result.status == "degraded"
    assert result.routing == RoutingDecision.VERIFY


def test_empty_response_returns_degraded():
    detector = JudgeDetector()
    result = detector.detect(user_query="Who created Java?", llm_response="")
    assert result.status == "degraded"
    assert result.routing == RoutingDecision.VERIFY


def test_degraded_mode_is_fail_closed():
    """Without a trained model, detector MUST route to VERIFY (fail-closed)."""
    detector = JudgeDetector()
    result = detector.detect(
        user_query="Who created Java?",
        llm_response="Java was created by James Gosling at Sun Microsystems in 1995."
    )
    # In degraded mode (no trained checkpoint), routing must be VERIFY
    assert result.routing == RoutingDecision.VERIFY
    assert result.degraded is True


def test_output_is_detectoroutput_type():
    detector = JudgeDetector()
    result = detector.detect(
        user_query="What is Python?",
        llm_response="Python is an interpreted programming language created by Guido van Rossum."
    )
    assert isinstance(result, DetectorOutput)


def test_claims_are_extracted():
    detector = JudgeDetector()
    result = detector.detect(
        user_query="Who created Java?",
        llm_response=(
            "Java was created by Dennis Ritchie in 1972. "
            "It was developed at Sun Microsystems. "
            "It is mainly used for operating systems."
        )
    )
    assert len(result.claims) >= 1
    assert all(hasattr(c, "claim_id") for c in result.claims)
    assert all(hasattr(c, "hallucination_probability") for c in result.claims)


def test_per_claim_probability_in_range():
    detector = JudgeDetector()
    result = detector.detect(
        user_query="Tell me about Java.",
        llm_response="Java was created in 1995. It is object-oriented."
    )
    for claim in result.claims:
        assert 0.0 <= claim.hallucination_probability <= 1.0
        assert 0.0 <= claim.confidence <= 1.0


def test_num_claims_counts_are_consistent():
    detector = JudgeDetector()
    result = detector.detect(
        user_query="Who is Einstein?",
        llm_response="Einstein developed relativity. He won the Nobel Prize in 1921."
    )
    total = result.num_high_risk + result.num_medium_risk + result.num_low_risk
    assert total == result.num_claims
    assert result.num_claims == len(result.claims)


def test_overall_probability_is_max_of_claims():
    detector = JudgeDetector()
    result = detector.detect(
        user_query="Who created Java?",
        llm_response="Java was made by Gosling. Python was made by Guido."
    )
    if result.claims:
        max_prob = max(c.hallucination_probability for c in result.claims)
        assert result.overall_hallucination_probability == max_prob


def test_always_verify_flag():
    from halluciguard_judge.config import JudgeDetectorConfig
    config = JudgeDetectorConfig(always_verify=True)
    detector = JudgeDetector(config=config)
    result = detector.detect(
        user_query="What is 2+2?",
        llm_response="2+2 equals 4."
    )
    # With always_verify=True, must always route to VERIFY
    assert result.routing == RoutingDecision.VERIFY


def test_spec_java_example():
    """
    From the spec:
    LLM response: Java was created by Dennis Ritchie in 1972. 
                  It was developed at Sun Microsystems. 
                  It is mainly used for operating systems.

    Expected claim extraction:
    C1 -> Java was created by Dennis Ritchie.
    C2 -> Java was created in 1972.
    C3 -> Java was developed at Sun Microsystems.
    C4 -> Java is mainly used for operating systems.
    """
    detector = JudgeDetector()
    result = detector.detect(
        user_query="Who created Java?",
        llm_response=(
            "Java was created by Dennis Ritchie in 1972. "
            "It was developed at Sun Microsystems. "
            "It is mainly used for operating systems."
        )
    )
    # Overall output structure is correct
    assert isinstance(result, DetectorOutput)
    assert result.overall_risk in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH)
    assert result.routing in (RoutingDecision.ACCEPT, RoutingDecision.VERIFY)
    assert len(result.claims) >= 1

    # In degraded mode: must route to VERIFY (fail-closed)
    if result.degraded:
        assert result.routing == RoutingDecision.VERIFY
