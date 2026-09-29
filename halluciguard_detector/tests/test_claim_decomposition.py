"""Regression tests for production claim decomposition.

Decomposition is the one part of the pipeline that changes *what* the detector
is asked about before retrieval runs, so a regression here silently changes both
training and evaluation inputs. These pin the current behaviour: multi-sentence
input is split, single-sentence input is not, and non-claimable text is rejected
rather than forced through.
"""
from __future__ import annotations

import pytest

from halluciguard_detector.evidence import decompose_claims, is_checkable

TWO_FACTS = (
    "The company was founded in 2003 by Martin Eberhard. "
    "It serves more than 50 million customers across 50 countries."
)

ONE_FACT = "Java was created by James Gosling at Sun Microsystems and released in 1995."


def test_multi_sentence_input_is_split():
    parts = decompose_claims(TWO_FACTS)
    assert len(parts) == 2
    assert "founded in 2003" in parts[0]
    assert "50 million customers" in parts[1]


def test_each_part_is_individually_checkable():
    # The split is only useful if retrieval and the guard can handle each part.
    for part in decompose_claims(TWO_FACTS) + decompose_claims(ONE_FACT):
        assert is_checkable(part)


def test_a_single_sentence_with_a_coordinator_is_still_split():
    # Not obvious from the name, and easy to regress by accident: decomposition
    # keys on clause structure, not sentence count, so "created by X and
    # released in 1995" becomes two claims even though it is one sentence.
    # This matters for the evidence work, because the claim that retrieval runs
    # against is then a fragment rather than the full input sentence.
    parts = decompose_claims(ONE_FACT)
    assert len(parts) == 2
    assert "James Gosling" in parts[0]
    assert "1995" in parts[1]


def test_a_single_uncoordinated_sentence_is_returned_without_its_full_stop():
    # Decomposition also normalises trailing punctuation away. Harmless for
    # the model, but it is the only normalisation applied to a claim, so it is
    # pinned here: training rows keep the annotated span verbatim while runtime
    # claims are stripped, and that asymmetry belongs in the report rather than
    # being discovered later.
    parts = decompose_claims("Java was created by James Gosling at Sun Microsystems.")
    assert len(parts) == 1
    assert parts[0] == "Java was created by James Gosling at Sun Microsystems"


def test_decomposition_never_returns_an_empty_claim():
    for text in (TWO_FACTS, ONE_FACT, "  ", "First one. Second one."):
        for part in decompose_claims(text):
            assert part.strip()


def test_decomposition_is_deterministic():
    assert decompose_claims(TWO_FACTS) == decompose_claims(TWO_FACTS)


def test_hedged_text_is_not_checkable():
    # Hedged, question-shaped and vacuous text is filtered out rather than
    # scored, so it must not reach the classifier.
    assert not is_checkable("I am not sure whether this is true.")
    assert not is_checkable("What is the capital of France?")


def test_a_factual_sentence_is_checkable():
    assert is_checkable(ONE_FACT)
