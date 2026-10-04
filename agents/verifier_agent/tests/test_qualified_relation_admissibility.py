"""Offline NLI fixtures test admissibility, not new model inference.

Qualified role extraction supplies evidence scope only. It cannot prove that a
co-founder is a primary/sole founder, resolve "the company" from a query, or use
a mention of an organization in another company's reference title as proof.
"""
from copy import deepcopy

import pytest

from api.pipeline import VerificationPipeline
from schemas.models import Passage
from scorers.relation_verifier import RelationVerifier


CLAIM = "Reid Hoffman is the primary founder of LinkedIn."
REFERENCE = (
    'Retrieved 2023-08-04. Shead, Sam (8 March 2022). '
    '"Reid Hoffman has co-founded his first new company since LinkedIn sale". CNBC'
)


def _passage(snippet, title="Wikipedia: LinkedIn"):
    return Passage(
        title=title, snippet=snippet, source="wikipedia",
        url="https://en.wikipedia.org/wiki/LinkedIn", publication_date="unknown",
        source_id="wiki_linkedin", relevance_score=0.41143,
    )


def _nli(label="contradiction", score=0.993652):
    remainder = (1.0 - score) / 2.0
    result = {"label": label, "entailment_score": remainder,
              "contradiction_score": remainder, "neutral_score": remainder}
    result[f"{label}_score"] = score
    return result


def _select(claim, passage, nli):
    return VerificationPipeline._select_decision_grade_evidence(
        [passage], [nli], claim, RelationVerifier()
    )


@pytest.mark.parametrize("qualifier", ["primary", "main", "principal", "sole", "only"])
def test_role_qualifier_is_preserved_without_structural_proof(qualifier):
    relation = RelationVerifier()
    claim = f"Reid Hoffman is the {qualifier} founder of LinkedIn."
    passage = _passage("Reid Hoffman is a co-founder of LinkedIn.")
    result = relation.verify_relation(claim, [passage])

    assert result.claim_triple.subject == "linkedin"
    assert result.claim_triple.object == "reid hoffman"
    assert qualifier in result.claim_triple.qualifiers
    assert result.status == "NO_TRIPLE_EXTRACTED"
    assert result.combination_rule_applied == ""
    assert result.evidence_triples


def test_cofounder_does_not_force_primary_founder_entailment_floor():
    nli = _nli("neutral", 0.98)
    original = deepcopy(nli)
    selected, scores = _select(CLAIM, _passage("Reid Hoffman is a co-founder of LinkedIn."), nli)

    assert selected == [] and scores == []
    assert nli == original


@pytest.mark.parametrize("label", ["entailment", "contradiction"])
def test_same_org_nli_is_admitted_without_qualified_role_score_floor(label):
    passage = _passage("Reid Hoffman is a co-founder of LinkedIn.")
    nli = _nli(label, 0.62)
    original = deepcopy(nli)
    selected, scores = _select(CLAIM, passage, nli)

    assert selected == [passage]
    assert scores == [original]
    assert nli == original


@pytest.mark.parametrize("label", ["entailment", "contradiction"])
def test_live_mustafa_reference_fixture_is_inadmissible_with_raw_scores_intact(label):
    passage = _passage(REFERENCE, title="Wikipedia: Mustafa Suleyman")
    nli = _nli(label)
    original = deepcopy(nli)
    selected, scores = _select(CLAIM, passage, nli)

    assert selected == [] and scores == []
    assert nli == original


@pytest.mark.parametrize("claim", [CLAIM, "Reid Hoffman is a founder of LinkedIn."])
@pytest.mark.parametrize("label", ["entailment", "contradiction"])
def test_shared_founder_at_other_organization_is_not_same_org_evidence(claim, label):
    passage = _passage("Reid Hoffman is a founder of Inflection AI.", title="Inflection AI")
    nli = _nli(label)
    original = deepcopy(nli)
    selected, scores = _select(claim, passage, nli)

    assert selected == [] and scores == []
    assert nli == original


def test_direct_same_org_negative_quote_remains_admissible():
    passage = _passage('The article states: "Reid Hoffman is not the primary founder of LinkedIn."')
    nli = _nli("contradiction", 0.92)
    original = deepcopy(nli)
    selected, scores = _select(CLAIM, passage, nli)

    assert selected == [passage]
    assert scores == [original]
    assert nli == original


def test_negated_role_claim_retains_evidence_scope_and_defers_to_nli():
    claim = "Reid Hoffman is not the primary founder of LinkedIn."
    passage = _passage("Reid Hoffman is the primary founder of LinkedIn.")
    relation = RelationVerifier().verify_relation(claim, [passage])
    nli = _nli("contradiction", 0.92)
    original = deepcopy(nli)
    selected, scores = _select(claim, passage, nli)

    assert relation.claim_triple.negated
    assert relation.status == "NO_TRIPLE_EXTRACTED"
    assert relation.evidence_triples
    assert selected == [passage]
    assert scores == [original]
    assert nli == original


def test_negated_founder_evidence_cannot_force_positive_structural_match():
    claim = "Reid Hoffman is a founder of LinkedIn."
    passage = _passage("Reid Hoffman is not a founder of LinkedIn.")
    relation = RelationVerifier().verify_relation(claim, [passage])
    nli = _nli("contradiction", 0.92)
    original = deepcopy(nli)
    selected, scores = _select(claim, passage, nli)

    assert relation.status == "NO_TRIPLE_EXTRACTED"
    assert selected == [passage]
    assert scores == [original]


def test_refutation_of_other_company_claim_is_not_admissible():
    passage = _passage(
        "Reid Hoffman is a founder of Inflection AI. The claim about that company is false.",
        title="Wikipedia: Mustafa Suleyman",
    )
    nli = _nli("contradiction")
    original = deepcopy(nli)
    selected, scores = _select(CLAIM, passage, nli)

    assert selected == [] and scores == []
    assert nli == original


@pytest.mark.parametrize("snippet", [
    "LinkedIn lists Reid Hoffman among its founders.",
    "Reid Hoffman helped launch LinkedIn.",
    "LinkedIn was co-founded by Reid Hoffman.",
    "LinkedIn was initially designed and implemented by Reid Hoffman.",
])
def test_relevant_natural_creation_prose_need_not_match_triple_template(snippet):
    passage = _passage(snippet)
    nli = _nli("entailment", 0.62)
    original = deepcopy(nli)
    selected, scores = _select(CLAIM, passage, nli)

    assert selected == [passage]
    assert scores == [original]


@pytest.mark.parametrize("label", ["entailment", "contradiction"])
def test_generic_company_primary_founders_remains_unresolved(label):
    claim = "The company’s primary founders include Reid Hoffman."
    passage = _passage("Reid Hoffman is a founder of LinkedIn.")
    relation_verifier = RelationVerifier()
    relation = relation_verifier.verify_relation(claim, [passage])
    nli = _nli(label)
    original = deepcopy(nli)
    selected, scores = _select(claim, passage, nli)

    assert relation_verifier.has_unresolved_subject(relation.claim_triple)
    assert relation.status == "NO_TRIPLE_EXTRACTED"
    assert selected == [] and scores == []
    assert nli == original


def test_named_possessive_primary_founder_list_keeps_org_and_qualification():
    claim = "LinkedIn’s primary founders include Reid Hoffman."
    passage = _passage("Reid Hoffman is a founder of LinkedIn.")
    relation = RelationVerifier().verify_relation(claim, [passage])
    nli = _nli("entailment", 0.62)
    selected, scores = _select(claim, passage, nli)

    assert relation.claim_triple.subject == "linkedin"
    assert "primary" in relation.claim_triple.qualifiers
    assert relation.status == "NO_TRIPLE_EXTRACTED"
    assert selected == [passage]
    assert scores[0]["entailment_score"] == 0.62


@pytest.mark.parametrize("snippet", [
    "Reid Hoffman helped launch LinkedIn Learning.",
    "Reid Hoffman is a founder of LinkedIn Learning.",
])
def test_natural_scope_does_not_match_different_org_with_shared_name_prefix(snippet):
    nli = _nli("entailment")
    original = deepcopy(nli)
    selected, scores = _select(CLAIM, _passage(snippet), nli)

    assert selected == [] and scores == []
    assert nli == original


@pytest.mark.parametrize("claim,snippet", [
    ("Tim Cook is the CEO of Apple.", "Tim Cook leads Apple."),
    ("Microsoft is headquartered in Redmond.", "Microsoft's headquarters lie in Redmond."),
])
def test_leadership_and_location_scope_allow_clear_natural_prose(claim, snippet):
    passage = _passage(snippet)
    nli = _nli("entailment", 0.62)
    original = deepcopy(nli)
    selected, scores = _select(claim, passage, nli)

    assert selected == [passage]
    assert scores == [original]


def test_existing_unqualified_same_org_structural_match_still_works():
    passage = _passage("Reid Hoffman is a founder of LinkedIn.")
    nli = _nli("neutral", 0.98)
    selected, scores = _select("Reid Hoffman is a founder of LinkedIn.", passage, nli)

    assert selected == [passage]
    assert scores[0]["entailment_score"] >= 0.95
    assert scores[0]["contradiction_score"] == 0.0
