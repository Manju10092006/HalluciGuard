"""The pre-model relevance gate must not discard distinct factual excerpts."""
import pytest

from services.evidence_ranker import _dedup, rank_evidence


def passage(text, url="https://example.test/company#History", source_id="history"):
    return {"title": "Company (History)", "source": "fixture", "source_id": source_id,
            "url": url, "snippet": text}


@pytest.mark.parametrize("first,second", [
    ("The company was founded in 2002.", "The company was founded in 2003."),
    ("The medicine is approved for adults.", "The medicine is not approved for adults."),
    ("Alice and Bob founded the company.", "Alice and Carol founded the company."),
    ("The laptop costs 1000 dollars.", "The laptop costs 1500 dollars."),
    ("The company was founded by Alice.", "The company appointed Bob as CEO in 2024."),
])
def test_same_source_distinct_facts_survive_relevance_dedup(first, second):
    values = [passage(first), passage(second)]
    kept, dropped = _dedup(values)
    assert kept == values and dropped == []
    ranked = rank_evidence(values, claim_text="Who founded the company?")
    assert len(ranked) == 2
    assert all(not item.discarded for item in ranked)


def test_identical_same_source_excerpt_is_observable_duplicate():
    values = [passage("Company was founded by Alice."),
              passage("Company  was\n founded by Alice.")]
    kept, dropped = _dedup(values)
    assert kept == values[:1]
    assert dropped == [(values[1], "duplicate_source_excerpt")]


def test_identical_independent_sources_keep_both_provenances():
    values = [passage("Company was founded by Alice."),
              passage("Company was founded by Alice.", "https://independent.test/company")]
    assert _dedup(values) == (values, [])


def test_urlless_ids_disambiguate_sources_without_erasing_distinct_excerpts():
    values = [passage("Company was founded by Alice.", "", "one"),
              passage("Company was founded by Alice.", "", "two"),
              passage("Company launched a product in 2003.", "", "one")]
    assert _dedup(values) == (values, [])


def test_unknown_sources_and_empty_excerpts_are_not_silently_deduplicated():
    values = [passage("A claim.", "", ""), passage("A claim.", "", ""),
              passage(""), passage(" ")]
    assert _dedup(values) == (values, [])
