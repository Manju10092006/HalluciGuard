"""V1.1 Quality Gate Tests — 11 deterministic mock tests for WebEnhancedAdapter."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from schemas.models import Passage
from adapters.web_enhanced import WebEnhancedAdapter


def _make_passage(
    title="Test Article",
    source="wikipedia",
    url="https://en.wikipedia.org/wiki/Test",
    snippet="This is a valid test passage with enough content for verification purposes.",
    relevance_score=0.7,
    source_id="wiki_test",
):
    return Passage(
        title=title,
        source=source,
        url=url,
        publication_date="2024-01-01",
        snippet=snippet,
        source_id=source_id,
        relevance_score=relevance_score,
    )


@pytest.fixture
def mock_primary():
    adapter = MagicMock()
    adapter.search = AsyncMock(return_value=[])
    adapter.name = "mock_primary"
    return adapter


@pytest.fixture
def enhanced(mock_primary, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test_key_123")
    return WebEnhancedAdapter(mock_primary, tavily_api_key="test_key_123")


# Test 1: Primary returns 5 highly relevant passages -> Tavily NOT called
@pytest.mark.asyncio
async def test_primary_sufficient_skips_tavily(enhanced, mock_primary):
    mock_primary.search.return_value = [
        _make_passage(url=f"https://example.com/page{i}", relevance_score=0.6)
        for i in range(5)
    ]

    with patch("adapters.web_retriever.TavilyWebRetriever") as MockRetriever:
        mock_tavily = MockRetriever.return_value
        mock_tavily.search = AsyncMock()
        enhanced._web_retriever = mock_tavily

        passages = await enhanced.search("test query", k=5, retrieval_mode="hybrid")

    trace = enhanced.last_retrieval_trace
    assert trace is not None
    assert trace.primary.sufficient is True
    assert trace.tavily.called is False
    assert trace.primary.relevant_count >= 1


# Test 2: Primary returns 5 irrelevant passages -> Tavily called
@pytest.mark.asyncio
async def test_primary_irrelevant_triggers_tavily(enhanced, mock_primary):
    mock_primary.search.return_value = [
        _make_passage(url=f"https://example.com/page{i}", relevance_score=0.05)
        for i in range(5)
    ]

    mock_tavily = MagicMock()
    mock_tavily.search = AsyncMock(return_value=[_make_passage(url="https://tavily.com/result1")])
    mock_tavily.credibility_of = MagicMock(return_value=0.7)
    enhanced._web_retriever = mock_tavily

    passages = await enhanced.search("test query", k=5, retrieval_mode="hybrid")

    trace = enhanced.last_retrieval_trace
    assert trace is not None
    assert trace.primary.sufficient is False
    assert trace.tavily.called is True
    assert mock_tavily.search.called


# Test 3: Primary returns empty list -> Tavily called
@pytest.mark.asyncio
async def test_primary_empty_triggers_tavily(enhanced, mock_primary):
    mock_primary.search.return_value = []

    mock_tavily = MagicMock()
    mock_tavily.search = AsyncMock(return_value=[_make_passage(url="https://tavily.com/r1")])
    mock_tavily.credibility_of = MagicMock(return_value=0.7)
    enhanced._web_retriever = mock_tavily

    passages = await enhanced.search("test query", k=5, retrieval_mode="hybrid")

    trace = enhanced.last_retrieval_trace
    assert trace is not None
    assert "EMPTY" in trace.primary.reason
    assert trace.tavily.called is True


# Test 4: Primary raises exception -> Tavily called
@pytest.mark.asyncio
async def test_primary_exception_triggers_tavily(enhanced, mock_primary):
    mock_primary.search.side_effect = Exception("Connection timeout")

    mock_tavily = MagicMock()
    mock_tavily.search = AsyncMock(return_value=[_make_passage(url="https://tavily.com/r1")])
    mock_tavily.credibility_of = MagicMock(return_value=0.7)
    enhanced._web_retriever = mock_tavily

    passages = await enhanced.search("test query", k=5, retrieval_mode="hybrid")

    trace = enhanced.last_retrieval_trace
    assert trace is not None
    assert trace.primary.reason == "PRIMARY_ERROR"
    assert trace.tavily.called is True


# Test 5: One strong passage -> sufficient (1 >= min_relevant=1, 0.8 >= min_top=0.3)
@pytest.mark.asyncio
async def test_one_strong_passage_sufficient(enhanced, mock_primary):
    mock_primary.search.return_value = [
        _make_passage(url="https://example.com/strong", relevance_score=0.8)
    ]

    mock_tavily = MagicMock()
    mock_tavily.search = AsyncMock()
    enhanced._web_retriever = mock_tavily

    passages = await enhanced.search("test query", k=5, retrieval_mode="hybrid")

    trace = enhanced.last_retrieval_trace
    assert trace is not None
    assert trace.primary.sufficient is True
    assert trace.tavily.called is False


# Test 6: Same URL in primary and Tavily -> deduplicated
@pytest.mark.asyncio
async def test_url_dedup_on_merge(enhanced, mock_primary):
    shared_url = "https://example.com/shared-page"
    mock_primary.search.return_value = [
        _make_passage(url=shared_url, relevance_score=0.1)
    ]

    mock_tavily = MagicMock()
    mock_tavily.search = AsyncMock(return_value=[
        _make_passage(url=shared_url, source="tavily_web")
    ])
    mock_tavily.credibility_of = MagicMock(return_value=0.7)
    enhanced._web_retriever = mock_tavily

    passages = await enhanced.search("test query", k=5, retrieval_mode="hybrid")

    urls = [p.url for p in passages]
    assert urls.count(shared_url) == 1


# Test 7: Tavily returns 3 passages, 2 have empty snippets -> usable_count == 1
@pytest.mark.asyncio
async def test_tavily_partial_failure(enhanced, mock_primary):
    mock_primary.search.return_value = []

    mock_tavily = MagicMock()
    mock_tavily.search = AsyncMock(return_value=[
        _make_passage(url="https://t1.com", snippet="This is a good passage with enough text for NLI."),
        _make_passage(url="https://t2.com", snippet=""),
        _make_passage(url="https://t3.com", snippet="   "),
    ])
    mock_tavily.credibility_of = MagicMock(return_value=0.7)
    enhanced._web_retriever = mock_tavily

    passages = await enhanced.search("test query", k=5, retrieval_mode="hybrid")

    trace = enhanced.last_retrieval_trace
    assert trace is not None
    assert trace.tavily.usable_count == 1


# Test 8: --force-tavily (tavily_only) -> primary NOT called
@pytest.mark.asyncio
async def test_force_tavily_mode(enhanced, mock_primary):
    mock_tavily = MagicMock()
    mock_tavily.search = AsyncMock(return_value=[_make_passage(url="https://t1.com")])
    mock_tavily.credibility_of = MagicMock(return_value=0.7)
    enhanced._web_retriever = mock_tavily

    passages = await enhanced.search("test query", k=5, retrieval_mode="tavily_only")

    assert not mock_primary.search.called
    trace = enhanced.last_retrieval_trace
    assert trace is not None
    assert trace.primary.called is False
    assert trace.tavily.called is True


# Test 9: primary_only -> Tavily NOT called even if primary has 0 results
@pytest.mark.asyncio
async def test_primary_only_mode(enhanced, mock_primary):
    mock_primary.search.return_value = []

    mock_tavily = MagicMock()
    mock_tavily.search = AsyncMock()
    enhanced._web_retriever = mock_tavily

    passages = await enhanced.search("test query", k=5, retrieval_mode="primary_only")

    assert not mock_tavily.search.called
    trace = enhanced.last_retrieval_trace
    assert trace is not None
    assert trace.tavily.called is False
    assert trace.primary.called is True


# Test 10: tavily_only explicit -> primary skipped, tavily called
@pytest.mark.asyncio
async def test_tavily_only_mode(enhanced, mock_primary):
    mock_tavily = MagicMock()
    mock_tavily.search = AsyncMock(return_value=[_make_passage(url="https://t1.com")])
    mock_tavily.credibility_of = MagicMock(return_value=0.7)
    enhanced._web_retriever = mock_tavily

    passages = await enhanced.search("test query", k=5, retrieval_mode="tavily_only")

    assert not mock_primary.search.called
    trace = enhanced.last_retrieval_trace
    assert trace.primary.called is False
    assert trace.tavily.called is True


# Test 11: hybrid mode, primary returns 0 -> primary first, then tavily fallback
@pytest.mark.asyncio
async def test_hybrid_primary_then_fallback(enhanced, mock_primary):
    mock_primary.search.return_value = []

    mock_tavily = MagicMock()
    mock_tavily.search = AsyncMock(return_value=[_make_passage(url="https://t1.com")])
    mock_tavily.credibility_of = MagicMock(return_value=0.7)
    enhanced._web_retriever = mock_tavily

    passages = await enhanced.search("test query", k=5, retrieval_mode="hybrid")

    assert mock_primary.search.called
    assert mock_tavily.search.called
    trace = enhanced.last_retrieval_trace
    assert trace.primary.called is True
    assert trace.tavily.called is True
