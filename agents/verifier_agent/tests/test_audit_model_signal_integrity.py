"""Rules must not synthesize neural probabilities. Fixtures are diagnostic only."""
import pytest
from schemas.models import Passage
from api.pipeline import VerificationPipeline
from scorers.relation_verifier import RelationVerifier
from scorers.evidence_scorer import EvidenceScorer


def passage(text, title="Test"):
    return Passage(title=title, snippet=text, source="wiki", source_id="s",
                   url="https://example.com/test", publication_date="", relevance_score=.9)


def neutral():
    return {"label": "neutral", "entailment_score": .02, "contradiction_score": .03,
            "neutral_score": .95, "validity_factor": 1.0}


@pytest.mark.parametrize("claim,text,expected_class", [
    # Confident relation OBJECT_MISMATCH -> CONTRADICTING even when the (mocked)
    # NLI is silent. The deterministic relation check IS corroborating evidence;
    # letting it vote restores the wrong-creator/wrong-location contradictions the
    # pure-NLI path misses (audit H1/H7-H9). NLI probabilities are NOT mutated.
    ("Java was created by Snehith.", "Java was created by James Gosling.", "CONTRADICTING"),
    # Confident relation MATCH on verbatim evidence -> SUPPORTING.
    ("Java was created by James Gosling.", "Java was created by James Gosling.", "SUPPORTING"),
    # No extractable relation triple -> defer entirely to NLI (neutral -> NEUTRAL).
    ("Nottingham is in England.", "An unrelated claim about medicine is false.", "NEUTRAL"),
    ("Rust was created by Graydon Hoare.", "Rust is memory safe. A different rumor was debunked.", "NEUTRAL"),
])
def test_relation_votes_as_corroboration_without_mutating_nli(claim, text, expected_class):
    p, nli = passage(text), neutral()
    original = dict(nli)
    # HG-007 guarantee preserved: the scorer never rewrites model probabilities.
    assert EvidenceScorer().classify_evidence(claim, p, nli) == expected_class
    assert nli == original
    # Decision-grade SELECTION still requires a real NLI signal, so a neutral-NLI
    # passage is not promoted into the selected set by the rule alone (relation-
    # grounded selection without any NLI signal is tracked separately as H8).
    selected, predictions = VerificationPipeline._select_decision_grade_evidence(
        [p], [nli], claim, RelationVerifier())
    assert nli == original
    assert selected == predictions == []


def test_selected_model_output_is_not_mutated_by_relation():
    p = passage("Java was created by James Gosling.")
    nli = {"entailment_score": .8, "contradiction_score": .1, "neutral_score": .1, "label": "entailment"}
    original = dict(nli)
    _, selected = VerificationPipeline._select_decision_grade_evidence(
        [p], [nli], "Java was created by James Gosling.", RelationVerifier())
    assert nli == original
    assert selected[0]["entailment_score"] == .8
    assert selected[0]["score_provenance"] == "nli"


def test_title_is_not_prefix_of_factual_subject():
    result = RelationVerifier().verify_relation(
        "Paris is the capital of France.",
        [passage("Paris is the capital of France.", title="France")])
    assert result.status == "MATCH"


def test_explicit_plural_creators_are_both_preserved():
    result = RelationVerifier().verify_relation(
        "Microsoft was created by Paul Allen.",
        [passage("Microsoft was created by Bill Gates and Paul Allen.")])
    assert result.status == "MATCH"


@pytest.mark.parametrize("claim,text", [
    ("Java was created by Snehith.", "It was alleged that Java was created by Snehith."),
    ("Java was created by Snehith.", "Java was not created by Snehith."),
    ("Java was created by James Gosling in 2005.", "Java was created by James Gosling in 1995."),
    ("Java was created by James Gosling and Python was created by Snehith.",
     "Java was created by James Gosling."),
])
def test_qualified_negated_or_compound_relation_cannot_prove_match(claim, text):
    result = RelationVerifier().verify_relation(claim, [passage(text)])
    assert result.status != "MATCH"


def test_duplicate_templates_are_not_compound_relations():
    result = RelationVerifier().verify_relation(
        "The Eiffel Tower is located in Paris.", [passage("The Eiffel Tower is located in Paris.")])
    assert result.status == "MATCH"


@pytest.mark.parametrize("items", [
    [{"label": "entailment", "score": 1.0}],
    [{"label": "not_entailment", "score": 1.0}],
    [{"label": "entailment", "score": float("nan")},
     {"label": "contradiction", "score": 0.0}, {"label": "neutral", "score": 0.0}],
    [{"label": "entailment", "score": .7},
     {"label": "contradiction", "score": .7}, {"label": "neutral", "score": .7}],
])
def test_invalid_nli_scores_are_rejected(items):
    from nli.robust_entailment import _normalize_scores
    with pytest.raises(ValueError):
        _normalize_scores(items)


@pytest.mark.parametrize("claim,text,expected", [
    ("Paris is the capital of France.",
     "Section [Capital city - Capitals that are not the seat of government]: "
     "Kingdom of France: The traditional capital was Paris, though from 1682 to 1789 "
     "the seat of government was at the Palace of Versailles.", "NEUTRAL"),
    ("Paris is the capital of France.",
     "Between 1682 and 1789 the seat of government was Versailles.", "NEUTRAL"),
    ("Paris is the capital of France.",
     "From 1682 to 1789 the seat was Versailles. Paris is not the capital of France.", "CONTRADICTING"),
    ("Paris is the capital of France in 1700.",
     "From 1682 to 1789 the seat was Versailles, not Paris, France.", "CONTRADICTING"),
    ("Paris is the capital of France.", "Paris is not the capital of France.", "CONTRADICTING"),
    ("Paris was the capital of France.",
     "From 1682 to 1789 the capital of France was Versailles, not Paris.", "CONTRADICTING"),
])
def test_historical_context_abstains_without_changing_model_scores(claim, text, expected):
    nli = {"label": "contradiction", "entailment_score": .002,
           "contradiction_score": .995, "neutral_score": .003}
    before = dict(nli)
    assert EvidenceScorer().classify_evidence(claim, passage(text), nli) == expected
    assert nli == before


def test_conflict_weights_use_the_probability_of_the_reported_class():
    from scorers.conflict_resolver import ConflictResolver
    from formatters.citation_formatter import CitationFormatter
    scorer = EvidenceScorer()
    formatter = CitationFormatter(evidence_scorer=scorer)
    nli = {"label": "contradiction", "entailment_score": .001,
           "contradiction_score": .998, "neutral_score": .001}
    contradicted = formatter.format_evidence(
        passage("Paris is not the capital of France."), nli, .8,
        claim="Paris is the capital of France.")
    support = {"entailment_label": "entailment", "entailment_score": .6, "credibility_score": .8}
    # Both evidence directions are substantial; .001 entailment on the
    # contradiction must not make it disappear as majority support.
    assert ConflictResolver().resolve([support, contradicted])["resolution_type"] == "genuine_conflict"
    assert ConflictResolver().resolve([support, contradicted.model_dump()])["resolution_type"] == "genuine_conflict"


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), -1.0, 1.5, "private-error"])
def test_conflict_resolver_ignores_invalid_probabilities(score):
    from scorers.conflict_resolver import ConflictResolver
    evidence = [
        {"entailment_label": "entailment", "entailment_score": .9, "credibility_score": .8},
        {"entailment_label": "contradiction", "nli_contradiction": score, "credibility_score": .8},
    ]
    assert ConflictResolver().resolve(evidence)["resolution_type"] == "unanimous_support"
