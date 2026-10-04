"""Numeric sentence endings must survive claim decomposition."""
from __future__ import annotations

import pytest

from claims import claim_decomposer


@pytest.fixture
def decomposer(monkeypatch):
    # Exercise the deterministic fallback without loading an external NLP model.
    monkeypatch.setattr(claim_decomposer, "_load_nlp", lambda: None)
    return claim_decomposer.ClaimDecomposer()


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "LinkedIn was founded in 2002 and launched in 2003. "
            "The company is headquartered in California.",
            [
                "LinkedIn was founded in 2002 and launched in 2003",
                "The company is headquartered in California",
            ],
        ),
        (
            "LinkedIn was founded in 2002 and launched in 2003",
            ["LinkedIn was founded in 2002 and launched in 2003"],
        ),
        (
            "LinkedIn was founded in 2002. It was launched in 2003.",
            ["LinkedIn was founded in 2002", "LinkedIn was launched in 2003"],
        ),
        (
            "The release date is May 5, 2003. The next release is in 2004.",
            ["The release date is May 5, 2003", "The next release is in 2004"],
        ),
        (
            "Version is 2. The release is ready.",
            ["Version is 2", "The release is ready"],
        ),
        (
            "The measurement is 2.5 metres. The temperature is 3.75 degrees.",
            ["The measurement is 2.5 metres", "The temperature is 3.75 degrees"],
        ),
        (
            "LinkedIn was founded in 2002; LinkedIn launched in 2003.",
            ["LinkedIn was founded in 2002", "LinkedIn launched in 2003"],
        ),
        (
            "LinkedIn was founded in 2002.\nLinkedIn launched in 2003.",
            ["LinkedIn was founded in 2002", "LinkedIn launched in 2003"],
        ),
    ],
)
def test_decompose_preserves_numeric_values(decomposer, text, expected):
    assert decomposer.decompose(text) == expected


@pytest.mark.parametrize("marker", [".", ")"])
def test_decompose_numbered_list_preserves_years(decomposer, marker):
    text = (
        f"  1{marker} LinkedIn was founded in 2002.\n"
        f"  2{marker} LinkedIn launched in 2003."
    )
    assert decomposer.decompose(text) == [
        "LinkedIn was founded in 2002",
        "LinkedIn launched in 2003",
    ]


def test_coarse_split_recognizes_only_line_start_numbered_markers():
    # Both NLP paths share this segmentation rule. Inline numbers belong to
    # their sentence; explicit numbered markers belong at the start of a line.
    text = " 1. Version is 2. The release is ready.\n\t2. Revenue was 2003."
    segments = [
        segment.strip()
        for segment in claim_decomposer._COARSE_SPLIT_RE.split(text)
        if segment.strip()
    ]
    assert segments == ["Version is 2. The release is ready.", "Revenue was 2003."]
