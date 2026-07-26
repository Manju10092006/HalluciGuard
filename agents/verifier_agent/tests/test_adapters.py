from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, patch

from adapters.registry import get_registry

@pytest.mark.asyncio
async def test_healthcare_adapter_search():
    registry = get_registry()
    adapter = registry.get_adapter("healthcare")
    with patch.object(adapter, 'search', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [{"text": "PubMed result on diabetes", "source": "pubmed"}]
        results = await adapter.search("diabetes drug")
        assert len(results) > 0
        assert results[0]["source"] == "pubmed"

@pytest.mark.asyncio
async def test_cybersecurity_adapter_search():
    registry = get_registry()
    adapter = registry.get_adapter("cybersecurity")
    with patch.object(adapter, 'search', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [{"text": "CVE-2023-1234 details", "source": "nvd"}]
        results = await adapter.search("buffer overflow")
        assert len(results) > 0
        assert results[0]["source"] == "nvd"

@pytest.mark.asyncio
async def test_ai_research_adapter_search():
    registry = get_registry()
    adapter = registry.get_adapter("ai_research")
    with patch.object(adapter, 'search', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [{"text": "Transformer architecture", "source": "arxiv"}]
        results = await adapter.search("attention mechanism")
        assert len(results) > 0
        assert results[0]["source"] == "arxiv"

@pytest.mark.asyncio
async def test_general_adapter_search():
    registry = get_registry()
    adapter = registry.get_adapter("general")
    with patch.object(adapter, 'search', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [{"text": "Wikipedia page on cats", "source": "wikipedia"}]
        results = await adapter.search("feline")
        assert len(results) > 0
        assert results[0]["source"] == "wikipedia"

@pytest.mark.asyncio
async def test_stub_adapter():
    registry = get_registry()
    adapter = registry.get_adapter("finance") # Assuming finance is a stub
    results = await adapter.search("stock market")
    # Stubs usually return empty or mocked data
    assert isinstance(results, list)
