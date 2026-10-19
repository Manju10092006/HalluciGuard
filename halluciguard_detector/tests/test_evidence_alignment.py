"""Regression tests for the evidence-shape arms and the error categoriser.

The three arms exist to hold everything constant except one variable, so the
tests are mostly about that variable actually changing: joined evidence must be
longer than top-1 evidence, and the top-1 arms must select the *same* snippet
given the same pool.
"""
from __future__ import annotations

import pytest

from halluciguard_detector.error_analysis import categorise
from halluciguard_detector.evidence_shapes import (
    PRODUCTION_SHAPE,
    SHAPES,
    EvidenceRecord,
    build_pool,
    lexical_joined,
    lexical_top1,
    stratified_subsample,
)
from halluciguard_detector.hard_negatives import HARD_CASES, to_rows
from halluciguard_detector.nli_input import enforce_single_snippet
from halluciguard_detector.text import lexical_evidence

CLAIM = "The company was founded in 2003 by Martin Eberhard and Marc Tarpenning."
SENTENCES = [
    "Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning in San Francisco.",
    "The company serves more than 50 million customers across 50 countries.",
    "Revenue reached 5 million dollars in 2010 and has grown every year since.",
    "The Chief Executive Officer is a woman who joined in 2018.",
]


def test_the_three_arms_are_registered():
    assert set(SHAPES) == {"lexical_joined", "lexical_top1", PRODUCTION_SHAPE}


def test_build_pool_puts_the_source_first():
    pool = build_pool("primary sentence", ["distractor one", "distractor two"])
    assert pool[0] == "primary sentence"
    assert len(pool) == 3


def test_build_pool_deduplicates():
    pool = build_pool("same", ["same", "same"])
    assert pool == ["same"]


def test_lexical_joined_records_that_it_joined_several_snippets():
    joined = lexical_joined(CLAIM, build_pool(SENTENCES[0], SENTENCES[1:]))
    top1 = lexical_top1(CLAIM, build_pool(SENTENCES[0], SENTENCES[1:]))
    # snippet_count is the structural marker the contract check keys on: 1 is
    # production, more than 1 is the old join.
    assert top1.snippet_count == 1
    assert joined.snippet_count > 1


def test_top1_evidence_is_a_verbatim_member_of_the_pool():
    top1 = lexical_top1(CLAIM, build_pool(SENTENCES[0], SENTENCES[1:]))
    assert top1.evidence in SENTENCES


def test_lexical_joined_is_longer_than_top1():
    joined = lexical_joined(CLAIM, build_pool(SENTENCES[0], SENTENCES[1:]))
    top1 = lexical_top1(CLAIM, build_pool(SENTENCES[0], SENTENCES[1:]))
    # The whole point of the experiment: the old arm concatenated, the
    # production arm does not.
    assert len(joined.evidence) > len(top1.evidence)
    assert enforce_single_snippet(top1.evidence) == top1.evidence


def test_lexical_arms_agree_on_the_selected_sentence():
    # Same selector, different join: isolating the join is the variable.
    pool = build_pool(SENTENCES[0], SENTENCES[1:])
    joined = lexical_joined(CLAIM, pool)
    top1 = lexical_top1(CLAIM, pool)
    assert top1.evidence in joined.evidence


def test_record_reports_metadata_without_inventing_retrieval_scores():
    record = lexical_top1(CLAIM, build_pool(SENTENCES[0], SENTENCES[1:]))
    assert isinstance(record, EvidenceRecord)
    assert record.route == "lexical_top1"
    # The lexical arms do not run the retriever, so they must not claim a score
    # from one. A number that was never computed is worse than an absent one.
    assert "similarity" not in record.extra
    assert "rerank_score" not in record.extra
    assert record.degraded is False


def test_subsample_is_deterministic_and_nests():
    rows = [
        {"id": index, "label": label}
        for index, label in enumerate(["SUPPORTED"] * 40 + ["CONTRADICTED"] * 8 + ["NOT_ENOUGH_INFO"] * 12)
    ]
    first = stratified_subsample(rows, 20, seed=42)
    second = stratified_subsample(rows, 20, seed=42)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 20
    # A smaller draw from the same seed must come from the same ordering, so
    # the cheap run is a subset of the expensive one.
    small = stratified_subsample(rows, 10, seed=42)
    assert {row["id"] for row in small} <= {row["id"] for row in first}


def test_subsample_keeps_all_three_labels_present():
    rows = [{"id": i, "label": label} for i, label in enumerate(
        ["SUPPORTED"] * 30 + ["CONTRADICTED"] * 6 + ["NOT_ENOUGH_INFO"] * 6
    )]
    drawn = stratified_subsample(rows, 15, seed=7)
    assert {row["label"] for row in drawn} == {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"}


def test_lexical_evidence_respects_a_limit_of_one():
    selected = lexical_evidence(CLAIM, build_pool(SENTENCES[0], SENTENCES[1:]), limit=1)
    assert len(selected) == 1


# --- error categoriser ------------------------------------------------------
# Each case is a fabricated CONTRADICTED false negative, so the expected label
# is the category the rule set is supposed to reach. The rules are heuristic and
# these tests pin their current behaviour, not a claim that the labels are
# correct ground truth.


@pytest.mark.parametrize(
    "claim,evidence,expected",
    [
        (
            "Tesla was founded in 1995.",
            "Tesla was founded in 2003 by Martin Eberhard in San Francisco.",
            "DATE",
        ),
        (
            "Revenue reached 5 million dollars.",
            "Revenue reached 8 million dollars last year.",
            "NUMBER",
        ),
        (
            "Java was created by Snehith.",
            "Java was created by James Gosling at Sun Microsystems.",
            "ENTITY",
        ),
        (
            "Safari runs on Windows.",
            "Safari is a browser developed exclusively for macOS and iOS, not Windows.",
            "NEGATION",
        ),
        (
            "Tesla was founded in 2003 and acquired by Google in 2018.",
            "Tesla was founded in 2003 by Martin Eberhard.",
            "CLAIM_DECOMPOSITION",
        ),
        (
            "Company X has 5 million users.",
            "Bananas are a tropical fruit grown in many countries.",
            "RETRIEVAL",
        ),        (
            "Apple acquired Company A in 2014.",
            "Apple has partnered with a wide range of companies.",
            "RELATION",
        ),
        (
            "The treaty was signed in 1618 and ended a war.",
            "The treaty was signed in 1618 in a European city.",
            "CLAIM_DECOMPOSITION",
        ),
    ],
)
def test_categoriser_reaches_the_expected_bucket(claim, evidence, expected):
    assert categorise(claim, evidence)["category"] == expected


def test_empty_evidence_is_attributed_to_retrieval():
    assert categorise("Something happened in 2003.", "")["category"] == "RETRIEVAL"


def test_a_structured_record_is_not_blamed_on_retrieval():
    # ~12% of RAGTruth test rows come from JSON sources that cannot be split
    # into sentences. Labelling those "retrieval" hides a dataset-preparation
    # defect behind a plausible-sounding explanation.
    verdict = categorise(
        "free Wi-Fi",
        '{"address": "4421 Hollister Ave", "attributes": {"Ambience": {"casual": true}}}',
    )
    assert verdict["category"] == "UNSEGMENTED_SOURCE"


def test_a_short_claim_covered_by_the_evidence_is_not_a_retrieval_miss():
    # One-word claims used to be mislabelled purely for being short.
    verdict = categorise(
        "Somali-Canadian",
        "The sisters were born in the Somali capital but fled to Dadaab after the 1991 war.",
    )
    assert verdict["category"] != "RETRIEVAL"


def test_production_arm_reports_the_snippet_the_model_actually_sees(monkeypatch):
    # ``select_evidence`` returns a ranked list; ``Detector.detect`` scores only
    # the first element. A real run recorded snippet_count=3 on every row as a
    # result of counting the ranked list, which made in-contract production
    # evidence look like a multi-snippet regression.
    from halluciguard_detector import evidence_shapes

    monkeypatch.setattr(
        evidence_shapes,
        "production_top1",
        evidence_shapes.production_top1,
    )
    fake_module = type(
        "_Fake",
        (),
        {
            "DEFAULT_POOL_K": 20,
            "select_evidence": staticmethod(
                lambda claim, candidates, **kwargs: ["first", "second", "third"]
            ),
        },
    )
    monkeypatch.setitem(
        __import__("sys").modules, "halluciguard_detector.evidence", fake_module
    )
    record = evidence_shapes.production_top1("a claim", ["first", "second"])
    assert record.evidence == "first"
    assert record.snippet_count == 1
    assert record.extra["ranked_snippets"] == 3
    assert record.extra["unused_ranked"] == ["second", "third"]


def test_categoriser_never_returns_an_unknown_bucket():
    verdict = categorise(
        "The weather in Berlin was unusually warm last spring.",
        "Berlin reported record rainfall during the same period.",
    )
    assert verdict["category"] in {
        "UNSEGMENTED_SOURCE", "RETRIEVAL", "CLAIM_DECOMPOSITION", "DATE",
        "NUMBER", "ENTITY", "NEGATION", "RELATION", "TEMPORAL", "MULTI_HOP",
        "NLI_MODEL", "OTHER",
    }
    assert verdict["reason"]


# --- hard negatives --------------------------------------------------------


def test_hard_cases_render_through_the_production_contract():
    rows = to_rows(HARD_CASES)
    assert len(rows) == len(HARD_CASES)
    for row in rows:
        # Single snippet on the evidence side, exactly like production.
        enforce_single_snippet(row["evidence"])
        assert row["label"] in {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"}
        assert row["label_id"] in (0, 1, 2)


def test_hard_set_is_not_all_contradictions():
    # A probe set of only contradictions would reward a model for saying
    # CONTRADICTED to everything, so "not enough info" cases are deliberate.
    labels = {case.expected for case in HARD_CASES}
    assert labels == {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"}


def test_hard_cases_cover_the_documented_difficulty_types():
    # Asserted on the structured case ids, not the prose: the notes are
    # documentation and re-wording one should not fail a behavioural test.
    ids = " ".join(case.case_id for case in HARD_CASES)
    for topic in ("year", "number", "entity", "relation", "negation", "temporal", "multi_hop"):
        assert topic in ids, topic
