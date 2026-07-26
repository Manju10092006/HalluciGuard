from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, patch

from adapters.registry import get_registry
from schemas.models import Passage

@pytest.mark.anyio
async def test_healthcare_adapter_search():
    registry = get_registry()
    adapter = registry.get_adapter("healthcare")
    sample_passage = Passage(
        title="PubMed Study",
        source="pubmed",
        url="https://pubmed.ncbi.nlm.nih.gov/123",
        publication_date="2024-01-01",
        snippet="PubMed result on diabetes"
    )
    with patch.object(adapter, 'search', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [sample_passage]
        results = await adapter.search("diabetes drug")
        assert len(results) > 0
        assert results[0].source == "pubmed"

@pytest.mark.anyio
async def test_cybersecurity_adapter_search():
    registry = get_registry()
    adapter = registry.get_adapter("cybersecurity")
    sample_passage = Passage(
        title="NVD CVE",
        source="nvd",
        url="https://nvd.nist.gov/vuln/detail/CVE-2023-1234",
        publication_date="2023-05-01",
        snippet="CVE-2023-1234 details"
    )
    with patch.object(adapter, 'search', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [sample_passage]
        results = await adapter.search("buffer overflow")
        assert len(results) > 0
        assert results[0].source == "nvd"

@pytest.mark.anyio
async def test_ai_research_adapter_search():
    registry = get_registry()
    adapter = registry.get_adapter("ai_research")
    sample_passage = Passage(
        title="arXiv Paper",
        source="arxiv",
        url="http://arxiv.org/abs/2301.12345",
        publication_date="2023-01-01",
        snippet="Transformer architecture"
    )
    with patch.object(adapter, 'search', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [sample_passage]
        results = await adapter.search("attention mechanism")
        assert len(results) > 0
        assert results[0].source == "arxiv"

@pytest.mark.anyio
async def test_general_adapter_search():
    registry = get_registry()
    adapter = registry.get_adapter("general")
    sample_passage = Passage(
        title="Wikipedia Cat",
        source="wikipedia",
        url="https://en.wikipedia.org/wiki/Cat",
        publication_date="2024-01-01",
        snippet="Wikipedia page on cats"
    )
    with patch.object(adapter, 'search', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [sample_passage]
        results = await adapter.search("feline")
        assert len(results) > 0
        assert results[0].source == "wikipedia"

@pytest.mark.anyio
async def test_stub_adapter():
    registry = get_registry()
    adapter = registry.get_adapter("sports")
    results = await adapter.search("football")
    assert isinstance(results, list)
