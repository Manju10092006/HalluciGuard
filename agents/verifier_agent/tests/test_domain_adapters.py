import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from schemas.models import Passage
from adapters.cybersecurity import CybersecurityAdapter
from adapters.general import GeneralAdapter
from adapters.finance import FinanceAdapter
from adapters.ai_research import AiResearchAdapter
from adapters.legal_general import LegalGeneralAdapter


# ── Cybersecurity Adapter Tests ──────────────────────────────

def test_cybersecurity_credibility_mapping():
    adapter = CybersecurityAdapter()
    assert adapter.credibility_of("nvd_cve_2021_44228") == 0.99
    assert adapter.credibility_of("cisa_cve_2021_44228") == 0.98
    assert adapter.credibility_of("circl_cve_2021_44228") == 0.97
    assert adapter.credibility_of("mitre_t1059") == 0.97
    assert adapter.credibility_of("github_advisory_ghsa_123") == 0.96


@pytest.mark.asyncio
async def test_cybersecurity_cve_direct_lookup():
    adapter = CybersecurityAdapter()
    mock_client = MagicMock()

    # Mock NVD direct response
    mock_nvd_res = MagicMock()
    mock_nvd_res.json.return_value = {
        "vulnerabilities": [
            {
                "cve": {
                    "id": "CVE-2021-44228",
                    "descriptions": [{"value": "Apache Log4j2 JNDI remote code execution vulnerability (Log4Shell)."}],
                    "published": "2021-12-10",
                }
            }
        ]
    }
    mock_client.get = AsyncMock(return_value=mock_nvd_res)

    passages = await adapter._search_cve_direct(mock_client, "CVE-2021-44228")
    assert len(passages) == 1
    p = passages[0]
    assert p.source == "nvd"
    assert "CVE-2021-44228" in p.title
    assert "Log4Shell" in p.snippet
    assert p.relevance_score >= 0.80


@pytest.mark.asyncio
async def test_cybersecurity_circl_fallback():
    adapter = CybersecurityAdapter()
    mock_client = MagicMock()

    # NVD fails, CIRCL succeeds
    mock_circl_res = MagicMock()
    mock_circl_res.json.return_value = {
        "id": "CVE-2021-44228",
        "summary": "Log4j JNDI flaw allows remote command execution.",
        "Published": "2021-12-10",
    }
    mock_client.get = AsyncMock(side_effect=[Exception("NVD Timeout"), mock_circl_res])

    passages = await adapter._search_cve_direct(mock_client, "CVE-2021-44228")
    assert len(passages) == 1
    p = passages[0]
    assert p.source == "circl"
    assert "Log4j JNDI flaw" in p.snippet


@pytest.mark.asyncio
async def test_cybersecurity_source_mode():
    adapter = CybersecurityAdapter()
    adapter._search_nvd = AsyncMock(return_value=[Passage(title="NVD", source="nvd", url="u1", publication_date="", snippet="s")])
    adapter._search_cisa = AsyncMock(return_value=[Passage(title="CISA", source="cisa", url="u2", publication_date="", snippet="s")])

    # Test CISA source mode
    res = await adapter.search("CVE-2021-44228", k=5, source_mode="cybersecurity-cisa")
    assert adapter._search_cisa.called
    assert not adapter._search_nvd.called
    assert len(res) == 1


# ── General (Wikipedia) Adapter Tests ────────────────────────

@pytest.mark.asyncio
async def test_general_adapter_page_summary():
    adapter = GeneralAdapter()
    mock_client = MagicMock()

    mock_search_res = MagicMock()
    mock_search_res.json.return_value = {
        "pages": [
            {"title": "Flat Earth", "excerpt": "<span>archaic concept</span>"}
        ]
    }

    mock_summary_res = MagicMock()
    mock_summary_res.json.return_value = {
        "extract": "Flat Earth is an archaic and scientifically disproven conception of Earth's shape as a plane or disk."
    }

    mock_client.get = AsyncMock(side_effect=[mock_search_res, mock_summary_res])

    with patch("adapters.general.get_client", return_value=mock_client):
        passages = await adapter.search("The Earth is flat", k=5)
        assert len(passages) == 1
        p = passages[0]
        assert p.source == "wikipedia"
        assert "scientifically disproven" in p.snippet
        assert p.source_confidence_hint >= 0.80


def test_general_adapter_noise_filtering():
    assert GeneralAdapter._is_noisy_title("Tanith Lee bibliography", "The Earth is flat") is True
    assert GeneralAdapter._is_noisy_title("List of songs by Queen", "Queen is a rock band") is True
    assert GeneralAdapter._is_noisy_title("Python (programming language)", "Python was created by Guido") is False


# ── Finance Adapter Tests ────────────────────────────────────

@pytest.mark.asyncio
async def test_finance_source_mode():
    adapter = FinanceAdapter()
    adapter._search_sec = AsyncMock(return_value=[Passage(title="SEC", source="sec_edgar", url="u1", publication_date="", snippet="s")])
    adapter._search_worldbank = AsyncMock(return_value=[Passage(title="WB", source="worldbank", url="u2", publication_date="", snippet="s")])

    res = await adapter.search("Apple revenue", k=5, source_mode="finance-sec")
    assert adapter._search_sec.called
    assert not adapter._search_worldbank.called
    assert len(res) == 1


# ── AI Research Adapter Tests ────────────────────────────────

def test_arxiv_query_sanitization():
    adapter = AiResearchAdapter()
    clean = adapter._sanitize_arxiv_query("Attention Is All You Need (2017) [Vaswani et al.]")
    assert "(" not in clean
    assert ")" not in clean
    assert "[" not in clean
    assert "Attention" in clean


@pytest.mark.asyncio
async def test_ai_research_source_mode():
    adapter = AiResearchAdapter()
    adapter._search_arxiv = AsyncMock(return_value=[Passage(title="arXiv", source="arxiv", url="u1", publication_date="", snippet="s")])
    adapter._search_semanticscholar = AsyncMock(return_value=[Passage(title="S2", source="semantic_scholar", url="u2", publication_date="", snippet="s")])

    res = await adapter.search("transformer models", k=5, source_mode="ai-arxiv")
    assert adapter._search_arxiv.called
    assert not adapter._search_semanticscholar.called
    assert len(res) == 1


# ── Legal Adapter Tests ──────────────────────────────────────

@pytest.mark.asyncio
async def test_legal_source_mode():
    adapter = LegalGeneralAdapter()
    adapter._search_courtlistener = AsyncMock(return_value=[Passage(title="Court", source="courtlistener", url="u1", publication_date="", snippet="s")])
    adapter._search_wikipedia = AsyncMock(return_value=[Passage(title="Wiki", source="wikipedia", url="u2", publication_date="", snippet="s")])

    res = await adapter.search("Miranda v. Arizona", k=5, source_mode="legal-courtlistener")
    assert adapter._search_courtlistener.called
    assert not adapter._search_wikipedia.called
    assert len(res) == 1
