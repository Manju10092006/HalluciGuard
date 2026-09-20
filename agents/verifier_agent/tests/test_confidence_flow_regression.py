"""Regression tests for the Verifier confidence flow and decomposition quality.

Context: a correct answer ("The Eiffel Tower is located in Paris") was collapsing
to ~0.20 confidence and being sent to human review. Two suspects were the
ClaimDecomposer (reportedly emitting mangled atomic claims such as
"...capital city of Germany to landmarks") and the confidence calibrator.

The claim_merger fix (confidence propagation + dominant-signal blend) is covered
separately in test_merger_confidence_unlock.py. These tests pin the two adjacent
guarantees:

  1. Decomposition of realistic multi-clause / distributive sentences yields
     clean, complete, faithful atomic propositions — never a mangled fragment
     with a duplicated/misattached trailing modifier.
  2. _calibrate_confidence honours a propagated confidence_score (does not
     re-dilute a well-grounded claim) and stays fail-closed on empty evidence.
"""
from __future__ import annotations

import pytest

from claims import ClaimDecomposer
from api.pipeline import VerificationPipeline


# ---------------------------------------------------------------------------
# 1. Decomposition quality — the exact shapes the user reported as garbage.
# ---------------------------------------------------------------------------
def _assert_clean(claims):
    for c in claims:
        norm = c.strip().lower().rstrip(".")
        assert len(norm.split()) >= 3, f"fragment too short: {c!r}"
        assert norm.split()[0] not in {"and", "but", "or", "also", "to"}, (
            f"dangling leading connector: {c!r}"
        )
        # The reported garbage pattern: a distributive split that welds a
        # prefix straight onto a trailing modifier ("...of Germany to landmarks").
        assert " of germany to " not in norm, f"mangled distribution: {c!r}"


def test_berlin_multiclause_decomposition_is_clean():
    """The user's exact failing input must decompose into faithful, complete claims."""
    d = ClaimDecomposer()
    claims = d.decompose(
        "The Eiffel Tower is located in Berlin. Berlin, on the other hand, is "
        "the capital city of Germany, home to many landmarks."
    )
    assert claims, "decomposer must not drop the assertion"
    _assert_clean(claims)
    joined = " ".join(claims).lower()
    assert "eiffel tower is located in berlin" in joined
    assert "capital city of germany" in joined


@pytest.mark.parametrize(
    "sentence,expect_split",
    [
        ("Berlin is the capital city of Germany and home to many landmarks.", True),
        ("The drug cures malaria and dengue in tropical regions.", True),
        ("Water is composed of hydrogen and oxygen atoms.", False),  # compound modifier
        ("Aspirin relieves pain and reduces inflammation in patients.", False),  # VP coord
    ],
)
def test_distributive_split_stays_grammatical(sentence, expect_split):
    d = ClaimDecomposer()
    claims = d.decompose(sentence)
    _assert_clean(claims)
    if expect_split:
        assert len(claims) >= 2, f"expected a distributive split for {sentence!r}"


# ---------------------------------------------------------------------------
# 2. Confidence calibration — propagation and fail-closed edges.
# ---------------------------------------------------------------------------
def test_calibrate_preserves_propagated_confidence():
    """A strong evidence-derived confidence_score must survive calibration."""
    cal = VerificationPipeline._calibrate_confidence
    out = cal({"confidence_score": 0.91, "support_score": 0.95, "contradiction_score": 0.02}, 3, {})
    assert out >= 0.85, f"well-grounded confidence re-diluted to {out}"


def test_calibrate_applies_genuine_conflict_penalty():
    cal = VerificationPipeline._calibrate_confidence
    out = cal({"confidence_score": 0.90}, 2, {"resolution_type": "genuine_conflict"})
    assert out == pytest.approx(0.54, abs=1e-3)  # 0.90 * 0.60


def test_calibrate_failclosed_on_zero_evidence():
    cal = VerificationPipeline._calibrate_confidence
    assert cal({"confidence_score": 0.90}, 0, {}) == 0.0


def test_calibrate_failclosed_on_weak_no_confidence():
    cal = VerificationPipeline._calibrate_confidence
    assert cal({"support_score": 0.10, "contradiction_score": 0.05}, 1, {}) == 0.0
