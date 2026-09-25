"""Offline tests for the deterministic multi-signal evidence ranker.

Centres on the Snehith/Google regression: an unrelated same-name page must be
discarded while genuine counter-evidence about Google's real founders is kept.
"""

from __future__ import annotations

from services.evidence_ranker import (
    DISCARD,
    STRONG,
    USABLE,
    WEAK,
    rank_and_select,
    rank_evidence,
    score_passage,
    select_evidence,
    source_family,
)


def _p(title, snippet, url, hint=0.7, date=""):
    return {
        "title": title,
        "snippet": snippet,
        "url": url,
        "source_confidence_hint": hint,
        "publication_date": date,
    }


CLAIM = "Snehith is the founder of Google."
QUERIES = ["who founded Google", "Google founder"]

IRRELEVANT_SAMENAME = _p(
    "Snehith (actor)",
    "Snehith is an Indian film actor known for his roles in Telugu cinema.",
    "https://en.wikipedia.org/wiki/Snehith_(actor)",
)
COUNTER_EVIDENCE = _p(
    "Google - Founders",
    "Google was founded in 1998 by Larry Page and Sergey Brin while they were "
    "PhD students at Stanford University.",
    "https://en.wikipedia.org/wiki/Google",
)


def test_article_section_is_not_deduplicated_against_article_lead():
    """Deep sections carry relation evidence absent from generic lead text."""
    claim = "Rust was created by Graydon Hoare while working at Mozilla."
    lead = _p(
        "Rust (programming language)",
        "Rust is a general-purpose programming language emphasizing safety and performance.",
        "https://en.wikipedia.org/wiki/Rust_(programming_language)",
        hint=0.85,
    )
    early_years = _p(
        "Rust (programming language) — Early years",
        "Graydon Hoare began developing Rust at Mozilla Research in 2006.",
        "https://en.wikipedia.org/wiki/Rust_(programming_language)#2006-2009_Early_years",
        hint=0.85,
    )
    ranked = rank_evidence(
        [lead, early_years], claim_text=claim, search_queries=["Rust Graydon Hoare Mozilla"]
    )
    assert len(ranked) == 2
    section = next(item for item in ranked if "Early years" in str(item.passage["title"]))
    assert section.discard_reason not in {"duplicate_canonical_url", "duplicate_url"}
    assert section.score > next(item.score for item in ranked if item is not section)


def test_counter_evidence_outranks_same_name_distractor():
    ranked = rank_evidence(
        [IRRELEVANT_SAMENAME, COUNTER_EVIDENCE], claim_text=CLAIM, search_queries=QUERIES
    )
    top = ranked[0]
    assert "Google" in str(top.passage["title"]) or "google" in str(top.passage["url"])
    # The founders page must score strictly higher than the actor distractor.
    scores = {r.passage["title"]: r.score for r in ranked}
    assert scores["Google - Founders"] > scores["Snehith (actor)"]


def test_same_name_distractor_is_not_decision_grade():
    ranked = rank_evidence([IRRELEVANT_SAMENAME], claim_text=CLAIM, search_queries=QUERIES)
    selected = select_evidence(ranked)
    # The actor page shares only the subject token; it must not be selected as
    # decision-grade counter/support evidence.
    assert all("actor" not in str(s.passage["title"]).lower() for s in selected)


def test_high_authority_relevant_source_is_strong_or_usable():
    ranked = rank_evidence([COUNTER_EVIDENCE], claim_text=CLAIM, search_queries=QUERIES)
    assert ranked[0].tier in (STRONG, USABLE)


def test_source_family_diversity_caps_wikipedia():
    passages = [
        _p(f"Google fact {i}", "Google was founded in 1998 by Larry Page and Sergey Brin.",
           f"https://en.wikipedia.org/wiki/Google_{i}")
        for i in range(5)
    ]
    selected, _ = rank_and_select(passages, claim_text=CLAIM, search_queries=QUERIES)
    families = [s.family for s in selected]
    # Five wikipedia.org hits collapse to one family; the cap must bound them.
    assert families.count("wikipedia.org") <= 3


def test_source_family_grouping():
    assert source_family(_p("t", "s", "https://en.wikipedia.org/wiki/X")) == "wikipedia.org"
    assert source_family(_p("t", "s", "https://pubmed.ncbi.nlm.nih.gov/123")) == "nih.gov"


def test_discard_reason_recorded_for_audit():
    ranked = rank_evidence([IRRELEVANT_SAMENAME], claim_text=CLAIM, search_queries=QUERIES)
    select_evidence(ranked)
    obs = [r.observability() for r in ranked]
    assert all("discard_reason" in o and "score" in o and "tier" in o for o in obs)


def test_duplicate_urls_are_deduped():
    dup = _p("Google", "Google was founded by Larry Page and Sergey Brin.",
             "https://en.wikipedia.org/wiki/Google")
    ranked = rank_evidence([dup, dict(dup)], claim_text=CLAIM, search_queries=QUERIES)
    reasons = [r.discard_reason for r in ranked if r.discarded]
    assert any("duplicate" in r for r in reasons)


def test_verified_claim_finds_strong_support():
    claim = "Google was founded in 1998."
    ranked = rank_evidence([COUNTER_EVIDENCE], claim_text=claim, search_queries=["Google founded 1998"])
    assert ranked[0].tier in (STRONG, USABLE)
    assert not ranked[0].discarded


def test_completely_irrelevant_passage_discarded():
    junk = _p("Cooking pasta", "Boil water and add salt before the pasta.",
              "https://example.com/pasta")
    ranked = rank_evidence([junk], claim_text=CLAIM, search_queries=QUERIES)
    assert ranked[0].tier in (WEAK, DISCARD)
