"""
Test claim extractor functionality.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from halluciguard_detector.claim_extraction.extractor import ClaimExtractor
from halluciguard_detector.claim_extraction.validator import validate_extracted_claim
from halluciguard_detector.schemas import ExtractionMode


def test_sentence_fallback_extraction():
    extractor = ClaimExtractor()
    draft = "Java was created by James Gosling in 1995. It was developed at Sun Microsystems."
    claims, mode, reasons = extractor.extract(
        user_query="Who created Java?",
        draft_answer=draft
    )
    assert len(claims) == 2
    assert claims[0].claim_id == "claim_001"
    assert "James Gosling" in claims[0].text
    assert claims[1].claim_id == "claim_002"
    assert "Sun Microsystems" in claims[1].text


def test_validator_catches_hallucinated_number():
    claim = "Java was created in 1899."
    source = "Java was created in 1995 by James Gosling."
    query = "Who created Java?"
    is_valid, flags = validate_extracted_claim(claim, source, query)
    assert not is_valid
    assert any("1899" in f for f in flags)


def test_validator_passes_valid_claim():
    claim = "Java was created in 1995."
    source = "Java was created in 1995 by James Gosling."
    query = "Who created Java?"
    is_valid, flags = validate_extracted_claim(claim, source, query)
    assert is_valid
    assert len(flags) == 0
