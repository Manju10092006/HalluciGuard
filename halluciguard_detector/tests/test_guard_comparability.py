"""Comparability gating for the numeric, year and entity guards.

The bug these pin down: a bare year disagreement was reported as a contradiction
even when the two sentences were about different events. "Tesla was founded in
2003" and "Tesla went public in 2018" are both true statements about unrelated
facts, and manufacturing contradiction mass between them is a false positive in
the exact direction the system is supposed to be conservative in.

A shared relation is therefore required before any numeric or year clash counts.
The gate must stay permissive for genuine same-predicate clashes, including the
awkward case where the shared verb is the first word of the claim and so looks
capitalised like a proper noun.
"""
from __future__ import annotations

import pytest

from halluciguard_detector.text import (
    has_entity_conflict,
    numeric_consistency,
    shared_relation,
)


# ------------------------------------------------- different relation: no flag


@pytest.mark.parametrize(
    "claim, evidence",
    [
        # The originally reported false positive.
        ("Tesla was founded in 2003.", "Tesla went public in 2018."),
        # Same shape, different events.
        ("The treaty was signed in 1618.", "The company was founded in 2019."),
        # Both are correct; only the numbers differ.
        ("Company A was founded in 2003.", "Company B went public in 2019."),
        # Unrelated years entirely.
        ("Java was released in 1995.", "Python was created in 1991."),
    ],
)
def test_unrelated_relations_never_flag_a_year_clash(claim, evidence):
    assert shared_relation(claim, evidence) == set()
    assert numeric_consistency(claim, evidence) == []


# --------------------------------------------------- same relation: must flag


@pytest.mark.parametrize(
    "claim, evidence, fragment",
    [
        (
            "Released in 1995.",
            "The product was released in 1996.",
            "1996",
        ),
        (
            "Java was released in 1995.",
            "Java was released in 1996.",
            "1996",
        ),
        (
            "Company X had 10 million users.",
            "Company X had 5 million users.",
            "5 million",
        ),
    ],
)
def test_same_relation_clashes_are_still_detected(claim, evidence, fragment):
    assert shared_relation(claim, evidence), "same predicate must be comparable"
    conflicts = numeric_consistency(claim, evidence)
    assert conflicts, "a genuine same-relation clash must still be reported"
    assert any(fragment in c for c in conflicts)


def test_sentence_initial_capitalised_predicate_is_not_a_proper_noun():
    """"Released" must be read as a verb, not a name.

    The entity vocabulary is shared across both texts, so a spurious entity hit
    on the claim erases the shared predicate from *both* sides. That is how a
    real 1995-vs-1996 conflict became invisible.
    """
    assert shared_relation("Released in 1995.", "The product was released in 1996.") == {
        "released"
    }


def test_genuine_proper_noun_is_still_excluded():
    """"Company A" / "Company B" stay excluded even though "company" is shared.

    Otherwise every two-sentence comparison that mentions a company would look
    like a shared relation.
    """
    assert "company" not in shared_relation(
        "Company A was founded in 2003.", "Company B went public in 2019."
    )


def test_shared_relation_never_counts_digits_as_relations():
    """Numbers are excluded from the relation vocabulary, not the words.

    "had" and "items" legitimately overlap -- that is the point -- but no digit
    may appear, and the year 5/9 is not a relation.
    """
    shared = shared_relation("It had 5 items.", "It had 9 items.")
    assert shared == {"had", "items"}
    assert not any(w.isdigit() for w in shared)


# ------------------------------------------------------------------ units


def test_mismatched_units_are_not_compared():
    """A percentage is never compared against a raw count."""
    conflicts = numeric_consistency(
        "The conversion rate was 5 percent.", "The conversion rate was 500000."
    )
    assert not any("conflicting percent" in c for c in conflicts)


def test_same_unit_quantity_clash_is_reported():
    conflicts = numeric_consistency(
        "The report counted 12 million records.", "The report counted 9 million records."
    )
    assert any("conflicting million" in c for c in conflicts)


def test_matching_numbers_produce_no_conflict():
    assert numeric_consistency(
        "The report counted 12 million records.", "The report counted 12 million records."
    ) == []


def test_ids_and_dates_are_not_treated_as_quantities():
    """Bare identifiers and ISO dates must not manufacture contradictions."""
    assert (
        numeric_consistency(
            "The record id is 12345.", "The record id is 98765."
        )
        == []
    )


# ------------------------------------------------------------------ entities


def test_entity_guard_flags_a_swapped_fact():
    assert has_entity_conflict(
        "Java was created by Snehith in 1995.",
        "Java was designed by James Gosling at Sun Microsystems in 1995.",
    )


def test_entity_guard_is_conservative_when_facts_agree():
    assert not has_entity_conflict(
        "Java was released in 1995.",
        "Java was designed by James Gosling and released in 1995.",
    )
