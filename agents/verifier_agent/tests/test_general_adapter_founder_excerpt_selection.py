"""Offline Wikipedia selection regressions using full HTTP response fixtures."""

from types import SimpleNamespace

import pytest

from adapters.general import GeneralAdapter
from services.evidence_ranker import rank_and_select


TITLE = "Aurora Circuit"
DESCRIPTION = (
    "Aurora Circuit sells computer equipment and provides subscription services "
    "to business customers through a network of regional offices."
)
LONG_LEAD = " ".join([
    DESCRIPTION,
    "Its annual reports describe product sales, customer contracts, regional "
    "operations, research spending, and the management of its current businesses.",
    "The company operates distribution centers and supplies components to "
    "customers in many markets through partnerships with local service providers.",
    "Corporate publications include detailed descriptions of its current "
    "services, offices, logistics, and the products available to its customers.",
])


class WikipediaHTTP:
    def __init__(self, summary, extract="", section_failure=False):
        self.summary = summary
        self.extract = extract
        self.section_failure = section_failure
        self.calls = []

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("/search/page"):
            data = {"pages": [{"title": TITLE, "excerpt": DESCRIPTION}]}
        elif "/page/summary/" in url:
            data = {"extract": self.summary}
        elif kwargs.get("params", {}).get("prop") == "extracts":
            if self.section_failure:
                raise RuntimeError("fixture section failure")
            data = {"query": {"pages": {"42": {"extract": self.extract}}}}
        else:
            raise AssertionError(f"Unexpected Wikipedia request: {url} {kwargs}")
        return SimpleNamespace(json=lambda: data)

    @property
    def deep_calls(self):
        return [call for call in self.calls if call[1].get("params", {}).get("prop") == "extracts"]


def use_http(monkeypatch, summary=LONG_LEAD, extract="", section_failure=False):
    client = WikipediaHTTP(summary, extract, section_failure)
    monkeypatch.setattr("adapters.general.get_client", lambda: client)
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("predicate", ["founder", "founders", "founded", "founding", "co-founder"])
async def test_founding_history_chunk_precedes_acquired_company_founders(monkeypatch, predicate):
    history = (
        "In 1982, the company was founded by Ada North and Ben Vale, initially "
        "as a small engineering partnership rather than its present corporation."
    )
    modern = "\n".join([
        "== Recent acquisitions ==",
        "Aurora Circuit's founder information appeared in an acquisition "
        "announcement when Aurora Circuit acquired Lumen, which was founded "
        "by Cara Stone to sell industrial components to local customers.",
        "== Executive announcements ==",
        "Aurora Circuit's founder information appeared in a recent profile "
        "of Aurora Circuit's executive, a founder of a separate consultancy "
        "that worked with regional clients on procurement programs.",
        "== Partnerships ==",
        "Aurora Circuit's founder information appeared in a partner profile "
        "when Aurora Circuit contracted with a founder of another business "
        "to provide regional technical services and maintain equipment.",
        "== History ==",
        history,
    ])
    client = use_http(monkeypatch, extract=modern)

    passages = await GeneralAdapter().search(f"{TITLE} {predicate}")

    assert client.deep_calls
    assert passages[1].title == f"Wikipedia: {TITLE} (History)"
    assert passages[1].snippet == f"Section [{TITLE} - History]: {history}"
    assert passages[1].url == "https://en.wikipedia.org/wiki/Aurora_Circuit#History"
    assert passages[1].relevance_score == 0.0
    assert passages[1].source_confidence_hint == 0.80


@pytest.mark.asyncio
async def test_founding_lead_sentence_after_character_400_is_visible(monkeypatch):
    relation = f"{TITLE} was founded by Ada North and Ben Vale in 1982."
    summary = f"{LONG_LEAD} {relation}"
    assert "founded" not in summary[:400]
    client = use_http(monkeypatch, summary=summary)

    passages = await GeneralAdapter().search(f"{TITLE} founded")

    assert len(passages) == 1
    excerpt = passages[0].snippet.removeprefix(f"Article [{TITLE}]: ")
    assert relation in excerpt
    assert excerpt in summary
    assert len(excerpt) <= 400
    assert client.deep_calls == []
    assert passages[0].source_id == "wiki_aurora_circuit"
    assert passages[0].relevance_score == 0.0
    assert passages[0].source_confidence_hint == 0.85


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [f"{TITLE} products", f"{TITLE} foundation", f"{TITLE} found"])
async def test_ordinary_queries_keep_existing_prefix_and_request_behavior(monkeypatch, query):
    summary = f"{LONG_LEAD} Its foundation and found documents describe products."
    client = use_http(monkeypatch, summary=summary)

    passages = await GeneralAdapter().search(query)

    assert len(passages) == 1
    assert passages[0].snippet == f"Article [{TITLE}]: {summary[:400]}"
    assert len(client.calls) == 2
    assert client.deep_calls == []


@pytest.mark.asyncio
async def test_no_founding_relation_preserves_deterministic_legacy_fallback(monkeypatch):
    modern = f"{TITLE} operates {TITLE} regional service offices for commercial customers."
    history = "The historical archives record early product catalogs and regional office addresses."
    extract = f"== Offices ==\n{modern}\n== History ==\n{history}"
    client = use_http(monkeypatch, extract=extract)
    adapter = GeneralAdapter()

    first = await adapter.search(f"{TITLE} founder")
    second = await adapter.search(f"{TITLE} founder")

    assert [p.model_dump() for p in first] == [p.model_dump() for p in second]
    assert first[0].snippet == f"Article [{TITLE}]: {LONG_LEAD[:400]}"
    assert first[1].snippet == f"Section [{TITLE} - Offices]: {modern}"
    assert first[2].snippet == f"Section [{TITLE} - History]: {history}"
    assert len(client.deep_calls) == 2
    assert all("founded" not in p.snippet.lower() for p in first)


@pytest.mark.asyncio
@pytest.mark.parametrize("relation", [
    f"{TITLE} was not founded by Elin West; the archived attribution names Ada North and Ben Vale.",
    f"{TITLE} was initially founded as a partnership before it was incorporated as a company.",
    f"Ada North and Ben Vale were legally recognized as co-founders of {TITLE} following a settlement.",
])
async def test_selected_lead_preserves_negation_and_relation_qualifiers(monkeypatch, relation):
    qualifier = "The attribution remains disputed in later accounts."
    summary = f"{LONG_LEAD} {relation} {qualifier}"
    use_http(monkeypatch, summary=summary)

    passages = await GeneralAdapter().search(f"{TITLE} founder")

    excerpt = passages[0].snippet.removeprefix(f"Article [{TITLE}]: ")
    assert relation in excerpt
    assert qualifier in excerpt
    assert excerpt in summary
    assert len(excerpt) <= 400


@pytest.mark.asyncio
async def test_distinct_history_chunks_share_source_and_survive_ranking_caller(monkeypatch):
    first_chunk = (
        f"{TITLE} was founded by Ada North in 1982, according to early company "
        "records describing the original partnership, its engineering workshop, "
        "regional customers, initial product designs, and the archived "
        "registration documents that accompanied its first commercial contracts."
    )
    second_chunk = (
        f"{TITLE} was also co-founded by Ben Vale in 1982, according to the "
        "same company records describing his role in the original partnership, "
        "the early engineering projects, customer contracts, and the "
        "registration documents preserved with the company's historical archives."
    )
    assert len(first_chunk) < 400 and len(second_chunk) < 400
    assert len(first_chunk) + len(second_chunk) > 400
    use_http(monkeypatch, extract=f"== History ==\n{first_chunk} {second_chunk}")

    passages = await GeneralAdapter().search(f"{TITLE} founder")
    history = [p for p in passages if p.title.endswith("(History)")]
    selected, ranked = rank_and_select(
        history, claim_text=f"{TITLE} founder", search_queries=[f"{TITLE} founded by"], top_k=2
    )

    assert len(history) == 2
    assert history[0].url == history[1].url
    assert history[0].source_id == history[1].source_id
    assert {p.snippet for p in history} == {
        f"Section [{TITLE} - History]: {first_chunk}",
        f"Section [{TITLE} - History]: {second_chunk}",
    }
    assert {item.passage.snippet for item in selected} == {p.snippet for p in history}
    assert all(item.discard_reason != "duplicate_source_excerpt" for item in ranked)


@pytest.mark.asyncio
async def test_section_failure_returns_only_the_retrieved_lead(monkeypatch):
    client = use_http(monkeypatch, section_failure=True)

    passages = await GeneralAdapter().search(f"{TITLE} founder")

    assert len(client.deep_calls) == 1
    assert len(passages) == 1
    assert passages[0].snippet == f"Article [{TITLE}]: {LONG_LEAD[:400]}"
