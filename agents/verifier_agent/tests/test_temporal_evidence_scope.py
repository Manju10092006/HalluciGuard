"""Historical context must not masquerade as a current factual refutation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.pipeline import VerificationPipeline
from schemas.models import Passage, VerdictLabel
from scorers.evidence_scorer import EvidenceScorer
from scorers.relation_verifier import RelationVerifier


def passage(snippet, title, suffix):
    return Passage(title=title, snippet=snippet, source="wikipedia",
                   url=f"https://example.org/{suffix}", publication_date="",
                   relevance_score=0.99)


def nli(*, support=0.0, contradiction=0.0):
    return {"entailment_score": support, "contradiction_score": contradiction,
            "neutral_score": max(0.0, 1.0 - support - contradiction),
            "label": "contradiction" if contradiction > support else "entailment",
            "degraded": False}


CURRENT = "Paris is the capital of France."
HISTORICAL = passage(
    "Section [Capital city - Capitals that are not the seat of government]: "
    "Kingdom of France: The traditional capital was Paris, though from 1682 to 1789 "
    "the seat of government was at the Palace of Versailles.",
    "Wikipedia: Capital city (Capitals that are not the seat of government)", "history",
)
CURRENT_SUPPORT = passage("Paris is the capital and largest city of France.", "Paris", "current")
CURRENT_CONTRADICTION = passage("Lyon is the capital of France, not Paris.", "France", "contradiction")


def test_historical_only_passage_is_not_current_contradiction():
    relation = RelationVerifier()
    assert relation.is_temporally_inapplicable(CURRENT, HISTORICAL)
    assert relation.verify_relation(CURRENT, [HISTORICAL]).status != "OBJECT_MISMATCH"
    selected, results = VerificationPipeline._select_decision_grade_evidence(
        [HISTORICAL], [nli(contradiction=0.995)], CURRENT, relation
    )
    assert selected == results == []
    assert EvidenceScorer().score_evidence(CURRENT, [HISTORICAL], [nli(contradiction=0.995)], "general")["verdict"] == VerdictLabel.UNVERIFIED


def test_present_support_survives_historical_distractor():
    selected, results = VerificationPipeline._select_decision_grade_evidence(
        [HISTORICAL, CURRENT_SUPPORT],
        [nli(contradiction=0.995), nli(support=0.99)], CURRENT, RelationVerifier()
    )
    assert selected == [CURRENT_SUPPORT]
    assert EvidenceScorer().score_evidence(CURRENT, selected, results, "general")["verdict"] == VerdictLabel.VERIFIED


def test_genuine_current_contradiction_and_past_claim_remain_checkable():
    relation = RelationVerifier()
    assert not relation.is_temporally_inapplicable(CURRENT, CURRENT_CONTRADICTION)
    assert not relation.is_temporally_inapplicable("Paris was the traditional capital of the Kingdom of France.", HISTORICAL)
    selected, results = VerificationPipeline._select_decision_grade_evidence(
        [CURRENT_CONTRADICTION], [nli(contradiction=0.995)], CURRENT, relation
    )
    assert selected
    assert EvidenceScorer().score_evidence(CURRENT, selected, results, "general")["verdict"] == VerdictLabel.CONTRADICTED


def test_mixed_present_and_historical_statements_are_not_blindly_discarded():
    mixed = passage(
        "The traditional capital was Lyon in 1700. Today, Paris is the capital of France.",
        "Capitals", "mixed",
    )
    assert not RelationVerifier().is_temporally_inapplicable(CURRENT, mixed)
