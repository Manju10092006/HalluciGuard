"""Regression coverage for title-anchored passive creation evidence."""

from schemas.models import Passage
from scorers.evidence_scorer import EvidenceScorer
from scorers.relation_verifier import RelationVerifier


def _microsoft_passage() -> Passage:
    return Passage(
        title="Wikipedia: Microsoft (Overview)",
        source="wikipedia",
        url="https://en.wikipedia.org/wiki/Microsoft#Overview",
        publication_date="unknown",
        snippet=(
            "Section [Microsoft - Overview]: A Big Tech company. "
            "Founded in 1975 by Bill Gates and Paul Allen to market BASIC "
            "interpreters for the Altair 8800."
        ),
        source_id="wiki_microsoft_overview",
        source_confidence_hint=0.8,
    )


def test_title_anchored_passive_creation_detects_wrong_founder() -> None:
    result = RelationVerifier().verify_relation(
        "Snehith founded Microsoft.",
        [_microsoft_passage()],
    )

    assert result.status == "OBJECT_MISMATCH"
    assert result.claim_triple is not None
    assert result.claim_triple.subject == "microsoft"
    assert any(t.subject == "microsoft" for t in result.evidence_triples)
    assert any("bill gates" in t.object for t in result.evidence_triples)


def test_title_anchored_passive_creation_supports_correct_founders() -> None:
    result = RelationVerifier().verify_relation(
        "Bill Gates and Paul Allen founded Microsoft.",
        [_microsoft_passage()],
    )

    assert result.status == "MATCH"


def test_unanchored_passive_text_does_not_invent_a_subject() -> None:
    passage = _microsoft_passage().model_copy(
        update={"title": "", "snippet": "Founded in 1975 by Bill Gates."}
    )

    assert RelationVerifier()._contextual_creation_triples(passage) == []


def test_structured_founder_mismatch_survives_low_reranker_score() -> None:
    passage = _microsoft_passage().model_copy(update={"relevance_score": 0.01})
    scores = EvidenceScorer().score_evidence(
        claim="Snehith founded Microsoft.",
        passages=[passage],
        nli_results=[
            {
                "label": "contradiction",
                "entailment_score": 0.001,
                "contradiction_score": 0.999,
                "neutral_score": 0.0,
            }
        ],
        domain="general",
    )

    assert scores["verdict"].value == "contradicted"
    assert scores["contradiction_score"] >= 0.50
    assert scores["confidence_score"] >= 0.40
