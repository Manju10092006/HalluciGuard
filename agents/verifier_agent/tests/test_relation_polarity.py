"""Regression test for polarity-aware relation verification.

The triple extractor is polarity-blind: "X was NOT created by Y" collapses to the
positive triple (X, created_by, Y). Without a polarity gate a TRUE refutation
("Python was not created by Elon Musk") matches truth evidence ("created by Guido
van Rossum"), fires OBJECT_MISMATCH, and manufactures a false ~0.95 contradiction.

The fix stamps sentence-level negation onto extracted triples and, when the claim
is negated, defers to the polarity-aware NLI stage by returning
NO_TRIPLE_EXTRACTED instead of a structural contradiction.
"""
from scorers.relation_verifier import RelationVerifier


def test_negated_claim_defers_instead_of_false_contradiction():
    rv = RelationVerifier()
    # True refutation; evidence states the real creator.
    result = rv.verify_relation(
        claim_text="Python was not created by Elon Musk.",
        evidence_passages=["Python was created by Guido van Rossum in 1991."],
    )
    assert result.status == "NO_TRIPLE_EXTRACTED", (
        f"negated claim must defer to NLI, not manufacture {result.status}"
    )


def test_negation_polarity_is_stamped_on_triples():
    rv = RelationVerifier()
    triples = rv.extract_triples("The Eiffel Tower is not located in Berlin.")
    assert triples, "expected at least one triple"
    assert any(t.negated for t in triples), "negation cue must set Triple.negated"


def test_positive_claim_still_compared_structurally():
    rv = RelationVerifier()
    triples = rv.extract_triples("Paris is the capital of France.")
    assert triples, "expected a triple for a positive assertion"
    assert all(not t.negated for t in triples), "positive claim must not be flagged negated"
