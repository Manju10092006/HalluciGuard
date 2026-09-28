"""
Tests for ClaimExtractor.
Run: pytest halluciguard_judge/tests/test_claim_extractor.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from halluciguard_judge.claim_extractor import ClaimExtractor
from halluciguard_judge.models import ClaimType


@pytest.fixture
def extractor():
    return ClaimExtractor(max_claims=10, min_claim_length=10)


def test_basic_extraction(extractor):
    response = (
        "Java was created by James Gosling in 1995. "
        "It was originally developed at Sun Microsystems. "
        "Java is designed to be portable across platforms."
    )
    claims = extractor.extract(response, "Who created Java?")
    assert len(claims) >= 1
    assert all(hasattr(c, "claim_id") for c in claims)
    assert all(hasattr(c, "text") for c in claims)
    assert all(hasattr(c, "claim_type") for c in claims)


def test_claim_ids_are_unique(extractor):
    response = "Java was created in 1995. It runs on the JVM. It is object-oriented."
    claims = extractor.extract(response)
    ids = [c.claim_id for c in claims]
    assert len(ids) == len(set(ids)), "Claim IDs must be unique"


def test_numerical_claim_type(extractor):
    response = "The speed of light is 299,792 km/s."
    claims = extractor.extract(response, "What is the speed of light?")
    # Should detect numerical type
    types = [c.claim_type for c in claims]
    assert ClaimType.NUMERICAL in types or ClaimType.FACTUAL in types


def test_empty_response_returns_empty(extractor):
    claims = extractor.extract("", "What is Python?")
    assert claims == []


def test_non_factual_opener_filtered(extractor):
    response = "Sure! Python is a high-level programming language. It was created by Guido van Rossum."
    claims = extractor.extract(response, "What is Python?")
    # "Sure!" line should be filtered
    texts = [c.text.lower() for c in claims]
    assert not any("sure" in t and len(t) < 20 for t in texts)


def test_max_claims_respected():
    extractor = ClaimExtractor(max_claims=3)
    response = " ".join([f"Claim number {i} is a factual statement." for i in range(20)])
    claims = extractor.extract(response)
    assert len(claims) <= 3


def test_min_length_filtered():
    extractor = ClaimExtractor(min_claim_length=20)
    response = "Yes. It is. Python."  # all too short
    claims = extractor.extract(response)
    assert len(claims) == 0


def test_java_example_from_spec(extractor):
    """Reproduce the spec example: Java claims."""
    response = (
        "Java was created by Dennis Ritchie in 1972. "
        "It was developed at Sun Microsystems. "
        "It is mainly used for operating systems."
    )
    claims = extractor.extract(response, "Who created Java?")
    texts = [c.text for c in claims]
    # Should extract at least the first factual claim
    assert any("Dennis Ritchie" in t or "Java" in t for t in texts)
    assert len(claims) >= 2


def test_claim_has_source_span(extractor):
    response = "Albert Einstein developed the theory of relativity in 1905."
    claims = extractor.extract(response)
    for c in claims:
        assert c.source_span is not None
        assert len(c.source_span) > 0
