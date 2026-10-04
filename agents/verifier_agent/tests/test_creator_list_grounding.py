"""Exact live LinkedIn excerpts and adversarial, offline relation fixtures.

These tests exercise parsing and evidence admissibility, not neural accuracy.
Names absent from a snippet are never supplied from the query or page title.
"""
from copy import deepcopy

import pytest

from api.pipeline import VerificationPipeline
from schemas.models import Passage
from scorers.relation_verifier import RelationVerifier


FOUNDING = (
    "Section [LinkedIn - Founding from 2002 to 2011]: The company was founded in December 2002 "
    "by Reid Hoffman and the founding team members from PayPal and Socialnet.com "
    "(Allen Blue, Eric Ly, Jean-Luc Vaillant, Lee Hower, Konstantin Guericke, "
    "Stephen Beitzel, David Eves, Ian McNish, Yan Pujante, Chris Saccheri)."
)
ERIC_LY = (
    "2002, Ly co- founded LinkedIn with Reid Hoffman , a Stanford schoolmate, "
    "and several other co-founders, including Jean - Luc Vaillant , Allen Blue , and Konstantin"
)
FULL_CLAIM = (
    "LinkedIn was founded by Reid Hoffman, Allen Blue, Konstantin Guericke, Eric Ly, and Jean‑Luc Vaillant."
)


def passage(snippet, title="LinkedIn"):
    return Passage(title=title, source="wikipedia", publication_date="unknown",
                   snippet=snippet, url="https://en.wikipedia.org/wiki/LinkedIn",
                   source_id="wiki_linkedin", relevance_score=0.75)


def test_live_five_creator_claim_preserves_each_complete_explicit_name():
    triples = RelationVerifier().extract_triples(FULL_CLAIM)
    creators = {triple.object for triple in triples if triple.relation == "created_by"}
    assert creators == {"reid hoffman", "allen blue", "konstantin guericke", "eric ly", "jean-luc vaillant"}
    assert all(triple.subject == "linkedin" for triple in triples)


@pytest.mark.parametrize("rendering", ["Jean-Luc", "Jean‑Luc", "Jean‐Luc", "Jean - Luc"])
def test_typographic_hyphens_preserve_full_creator_name(rendering):
    result = RelationVerifier().verify_relation(
        f"{rendering} Vaillant is a co-founder of LinkedIn.", [passage(FOUNDING)])
    assert result.claim_triple.object == "jean-luc vaillant"
    assert result.status == "MATCH"


def test_complete_live_parenthesized_founding_team_is_bound_to_page_subject():
    relation = RelationVerifier()
    triples = relation._contextual_creation_triples(passage(FOUNDING))
    assert {triple.object for triple in triples} == {
        "reid hoffman", "allen blue", "eric ly", "jean-luc vaillant", "lee hower",
        "konstantin guericke", "stephen beitzel", "david eves", "ian mcnish", "yan pujante", "chris saccheri",
    }
    assert all(triple.subject == "linkedin" for triple in triples)
    assert all("non_exhaustive" in triple.qualifiers for triple in triples)
    assert relation.verify_relation(FULL_CLAIM, [passage(FOUNDING)]).status == "MATCH"


def test_month_year_founding_date_never_becomes_a_location():
    relation = RelationVerifier()
    assert relation._contextual_location_triples(passage(FOUNDING)) == []
    assert not any(triple.relation == "location_of" for triple in relation.extract_triples(
        "LinkedIn was founded in December 2002 by Reid Hoffman."))


@pytest.mark.parametrize("attribution", ["recognized", "recognised", "credited", "acknowledged", "described"])
def test_attributed_role_has_scope_but_is_not_proved_by_bare_creator_membership(attribution):
    claim = f"Jean‑Luc Vaillant is {attribution} as a co‑founder of LinkedIn."
    relation = RelationVerifier()
    result = relation.verify_relation(claim, [passage(FOUNDING)])
    assert result.claim_triple.subject == "linkedin"
    assert result.claim_triple.object == "jean-luc vaillant"
    assert "attributed_role" in result.claim_triple.qualifiers
    assert result.status == "NO_TRIPLE_EXTRACTED"
    assert result.combination_rule_applied == ""
    assert relation.evidence_addresses_relation(result.claim_triple, passage(FOUNDING), result.evidence_triples)


def test_unresolved_attributed_pronoun_remains_unresolved():
    relation = RelationVerifier()
    result = relation.verify_relation(
        "Each of them is recognized as a co‑founder of the company.", [passage(FOUNDING)])
    assert relation.has_unresolved_subject(result.claim_triple)
    assert result.status == "NO_TRIPLE_EXTRACTED"
    assert not relation.evidence_addresses_relation(result.claim_triple, passage(FOUNDING), result.evidence_triples)


def test_irrelevant_paul_allen_death_passage_cannot_contradict_attributed_linkedin_role():
    claim = "Allen Blue is recognized as a co-founder of LinkedIn."
    unrelated = passage("Paul Allen died in 2018. Paul Allen is a co-founder of Microsoft.", title="Paul Allen")
    nli = {"label": "contradiction", "entailment_score": 0.001,
           "contradiction_score": 0.998, "neutral_score": 0.001}
    original = deepcopy(nli)
    selected, scores = VerificationPipeline._select_decision_grade_evidence(
        [unrelated], [nli], claim, RelationVerifier())
    assert selected == [] and scores == []
    assert nli == original


def test_live_truncated_person_page_retains_only_written_creator_names():
    relation = RelationVerifier()
    triples = relation.extract_triples(ERIC_LY)
    creators = {triple.object for triple in triples if triple.relation == "created_by"}
    assert {"ly", "reid hoffman", "jean-luc vaillant", "allen blue", "konstantin"} <= creators
    assert "eric ly" not in creators and "konstantin guericke" not in creators
    assert all("non_exhaustive" in triple.qualifiers for triple in triples if triple.relation == "created_by")
    assert relation.verify_relation(FULL_CLAIM, [passage(ERIC_LY, title="Eric Ly")]).status == "NO_TRIPLE_EXTRACTED"


@pytest.mark.parametrize("person", ["Homer Simpson", "Snehith", "Konstantin Guericke", "Eric Ly"])
def test_nonexhaustive_or_incomplete_creator_evidence_cannot_refute_an_unlisted_person(person):
    result = RelationVerifier().verify_relation(
        f"{person} is a co-founder of LinkedIn.", [passage(ERIC_LY, title="Eric Ly")])
    assert result.status == "NO_TRIPLE_EXTRACTED"
    assert result.combination_rule_applied == ""


def test_truncated_parenthesized_list_does_not_invent_missing_closure_or_names():
    truncated = FOUNDING.split("Eric Ly")[0] + "Eric"
    result = RelationVerifier().verify_relation("Allen Blue is a co-founder of LinkedIn.", [passage(truncated)])
    assert result.status == "NO_TRIPLE_EXTRACTED"
    assert not any(triple.object == "allen blue" for triple in result.evidence_triples)


def test_literal_partial_name_does_not_refute_a_full_name():
    result = RelationVerifier().verify_relation(
        "Konstantin Guericke is a founder of LinkedIn.",
        [passage("LinkedIn was founded by Konstantin.")])
    assert result.status == "NO_TRIPLE_EXTRACTED"
    assert not any(triple.object == "konstantin guericke" for triple in result.evidence_triples)


def test_first_supported_creator_never_proves_an_unsupported_second_creator():
    result = RelationVerifier().verify_relation(
        "LinkedIn was founded by Reid Hoffman and Homer Simpson.", [passage(FOUNDING)])
    assert result.status == "NO_TRIPLE_EXTRACTED"
    assert result.combination_rule_applied == ""


def test_two_correct_complete_creators_can_both_be_grounded():
    result = RelationVerifier().verify_relation(
        "Bill Gates and Paul Allen founded Microsoft.",
        [passage("Founded in 1975 by Bill Gates and Paul Allen.", title="Microsoft")])
    assert result.status == "MATCH"


def test_mixed_compound_claim_does_not_match_only_its_creator_component():
    result = RelationVerifier().verify_relation(
        "LinkedIn was founded by Reid Hoffman. LinkedIn is headquartered in Boston.", [passage(FOUNDING)])
    assert result.status == "NO_TRIPLE_EXTRACTED"


@pytest.mark.parametrize("extra_role", ["the primary founder", "the sole founder", "recognized as a co-founder"])
def test_same_person_bare_role_does_not_hide_a_remaining_qualified_role(extra_role):
    result = RelationVerifier().verify_relation(
        "Reid Hoffman is a founder of LinkedIn. "
        f"Reid Hoffman is {extra_role} of LinkedIn.", [passage(FOUNDING)])
    assert result.status == "NO_TRIPLE_EXTRACTED"
    assert result.combination_rule_applied == ""


def test_negated_founding_team_evidence_cannot_prove_positive_membership():
    result = RelationVerifier().verify_relation(
        "Allen Blue is a co-founder of LinkedIn.", [passage(FOUNDING.replace("was founded", "was not founded"))])
    assert result.status == "NO_TRIPLE_EXTRACTED"
    assert all(triple.negated for triple in result.evidence_triples if triple.relation == "created_by")


@pytest.mark.parametrize("snippet,title", [
    ("", "Reid Hoffman is a founder of LinkedIn"),
    ("Microsoft was founded by Bill Gates.", "LinkedIn"),
    ("According to LinkedIn, Microsoft was founded by Bill Gates.", "LinkedIn"),
    ("LinkedIn is a partner of Microsoft, founded in 1975 by Bill Gates and Paul Allen.", "LinkedIn"),
    ("LinkedIn is a subsidiary of Microsoft founded in 1975 by Bill Gates and Paul Allen.", "LinkedIn"),
    ("Reid Hoffman has co-founded his first new company since LinkedIn sale.", "LinkedIn"),
    ("He founded a company.", "LinkedIn"),
])
def test_titles_and_ambiguous_mentions_do_not_supply_missing_creation_relationship(snippet, title):
    result = RelationVerifier().verify_relation("Reid Hoffman is a founder of LinkedIn.", [passage(snippet, title)])
    assert result.status == "NO_TRIPLE_EXTRACTED"


def test_unresolved_creator_pronoun_is_not_expanded_from_page_title():
    result = RelationVerifier().verify_relation(
        "Bill Gates is a founder of Microsoft.", [passage("He founded Microsoft.", title="Bill Gates")])
    assert result.status == "NO_TRIPLE_EXTRACTED"
    assert result.evidence_triples == []
