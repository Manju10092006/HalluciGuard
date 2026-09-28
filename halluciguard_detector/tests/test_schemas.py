import pytest
from pydantic import ValidationError

from halluciguard_detector.schemas import DetectRequest


def test_evidence_is_mandatory():
    with pytest.raises(ValidationError):
        DetectRequest(draft_answer="A factual claim.")
