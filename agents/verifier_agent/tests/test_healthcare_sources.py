import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from schemas.models import Passage
from adapters.healthcare import HealthcareAdapter


def test_credibility_mapping():
    adapter = HealthcareAdapter()
    assert adapter.credibility_of("who_mdg_001") == 0.99
    assert adapter.credibility_of("openfda_aspirin") == 0.98
    assert adapter.credibility_of("pubmed_12345") == 0.97
    assert adapter.credibility_of("pmid_12345") == 0.97
    assert adapter.credibility_of("pmc_67890") == 0.96
    assert adapter.credibility_of("clinicaltrials_NCT123") == 0.95


def test_query_strategy_routing():
    adapter = HealthcareAdapter()
    
    # Drug claim
    sources_drug = adapter._determine_query_strategy("Aspirin tablet cures headache", drug_name="aspirin")
    assert "openfda" in sources_drug
    assert "pubmed" in sources_drug
    
    # Clinical trial claim
    sources_trial = adapter._determine_query_strategy("Phase 3 clinical trial of mRNA vaccine in patients")
    assert "clinicaltrials" in sources_trial
    assert "pubmed" in sources_trial
    
    # Public health claim
    sources_who = adapter._determine_query_strategy("WHO reported cholera outbreak mortality rate")
    assert "who" in sources_who
    assert "pubmed" in sources_who


@pytest.mark.asyncio
async def test_source_mode_override():
    adapter = HealthcareAdapter()
    
    with patch("adapters.healthcare.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        adapter._search_pubmed = AsyncMock(return_value=[Passage(title="T1", source="pubmed", url="http://p1", publication_date="", snippet="s")])
        adapter._search_openfda = AsyncMock(return_value=[Passage(title="T2", source="openfda", url="http://p2", publication_date="", snippet="s")])
        adapter._search_who = AsyncMock(return_value=[Passage(title="T3", source="who", url="http://p3", publication_date="", snippet="s")])
        
        # Test pubmed only mode
        res1 = await adapter.search("aspirin", k=5, source_mode="healthcare-pubmed")
        assert adapter._search_pubmed.called
        assert not adapter._search_openfda.called
        assert not adapter._search_who.called
        assert len(res1) == 1
        
        # Reset mocks
        adapter._search_pubmed.reset_mock()
        adapter._search_openfda.reset_mock()
        
        # Test fda only mode
        res2 = await adapter.search("aspirin", k=5, source_mode="healthcare-fda")
        assert adapter._search_openfda.called
        assert not adapter._search_pubmed.called
        assert len(res2) == 1


@pytest.mark.asyncio
async def test_pubmed_parsing():
    adapter = HealthcareAdapter()
    mock_client = MagicMock()
    
    # Mock esearch response
    mock_search_res = MagicMock()
    mock_search_res.json.return_value = {"esearchresult": {"idlist": ["12345"]}}
    
    # Mock efetch response with XML
    mock_fetch_res = MagicMock()
    mock_fetch_res.text = """
    <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345</PMID>
                <Article>
                    <Abstract>
                        <AbstractText>Aspirin relieves mild to moderate pain and fever.</AbstractText>
                    </Abstract>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    </PubmedArticleSet>
    """
    
    # Mock esummary response
    mock_summary_res = MagicMock()
    mock_summary_res.json.return_value = {
        "result": {
            "12345": {
                "title": "Aspirin in clinical therapy",
                "pubdate": "2023 Jan",
            }
        }
    }
    
    mock_client.get = AsyncMock(side_effect=[mock_search_res, mock_fetch_res, mock_summary_res])
    
    passages = await adapter._search_pubmed(mock_client, "aspirin", 5)
    assert len(passages) == 1
    p = passages[0]
    assert p.source == "pubmed"
    assert p.url == "https://pubmed.ncbi.nlm.nih.gov/12345/"
    assert "Aspirin relieves mild to moderate pain" in p.snippet
    assert p.source_id == "pubmed_12345"


@pytest.mark.asyncio
async def test_pubmed_central_parsing():
    adapter = HealthcareAdapter()
    mock_client = MagicMock()
    
    mock_search_res = MagicMock()
    mock_search_res.json.return_value = {"esearchresult": {"idlist": ["99999"]}}
    
    mock_summary_res = MagicMock()
    mock_summary_res.json.return_value = {
        "result": {
            "99999": {
                "title": "Full text study of drug",
                "pubdate": "2022",
            }
        }
    }
    
    mock_fetch_res = MagicMock()
    mock_fetch_res.text = """
    <pmc-articleset>
        <article>
            <front>
                <article-meta>
                    <article-id pub-id-type="pmc">99999</article-id>
                    <abstract><p>Clinical efficacy was strongly supported in randomized trial.</p></abstract>
                </article-meta>
            </front>
        </article>
    </pmc-articleset>
    """
    
    mock_client.get = AsyncMock(side_effect=[mock_search_res, mock_summary_res, mock_fetch_res])
    
    passages = await adapter._search_pubmed_central(mock_client, "drug study", 5)
    assert len(passages) == 1
    p = passages[0]
    assert p.source == "pubmed_central"
    assert p.url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC99999/"
    assert "Clinical efficacy was strongly supported" in p.snippet
    assert p.source_id == "pmc_99999"


@pytest.mark.asyncio
async def test_openfda_multiword_query():
    adapter = HealthcareAdapter()
    mock_client = MagicMock()
    
    mock_fda_res = MagicMock()
    mock_fda_res.json.return_value = {
        "results": [
            {
                "openfda": {
                    "brand_name": ["Ferrous Sulfate"],
                    "generic_name": ["Iron Supplement"]
                },
                "indications_and_usage": [
                    "For treatment of iron deficiency anemia."
                ],
                "purpose": ["Mineral Supplement"]
            }
        ]
    }
    mock_client.get = AsyncMock(return_value=mock_fda_res)
    
    passages = await adapter._search_openfda(mock_client, "ferrous sulfate", "ferrous sulfate iron tablet", 5)
    assert len(passages) == 1
    p = passages[0]
    assert p.source == "openfda"
    assert "Ferrous Sulfate" in p.title
    assert "iron deficiency anemia" in p.snippet


@pytest.mark.asyncio
async def test_who_indicator_parsing():
    adapter = HealthcareAdapter()
    mock_client = MagicMock()
    
    mock_who_res = MagicMock()
    mock_who_res.json.return_value = {
        "value": [
            {
                "IndicatorName": "Maternal mortality ratio (per 100 000 live births)",
                "IndicatorCode": "MDG_0000000026"
            }
        ]
    }
    mock_client.get = AsyncMock(return_value=mock_who_res)
    
    passages = await adapter._search_who(mock_client, "maternal mortality ratio", 5)
    assert len(passages) == 1
    p = passages[0]
    assert p.source == "who"
    assert "who_mdg_0000000026" in p.source_id
    assert "Maternal mortality ratio" in p.title


@pytest.mark.asyncio
async def test_clinicaltrials_parsing():
    adapter = HealthcareAdapter()
    mock_client = MagicMock()
    
    mock_ct_res = MagicMock()
    mock_ct_res.json.return_value = {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT01234567",
                        "briefTitle": "Phase 3 Study of Drug X in Hypertension"
                    },
                    "statusModule": {
                        "overallStatus": "COMPLETED"
                    },
                    "descriptionModule": {
                        "briefSummary": "This trial investigates drug X versus placebo."
                    }
                }
            }
        ]
    }
    mock_client.get = AsyncMock(return_value=mock_ct_res)
    
    passages = await adapter._search_clinicaltrials(mock_client, "drug x hypertension", 5)
    assert len(passages) == 1
    p = passages[0]
    assert p.source == "clinicaltrials"
    assert p.url == "https://clinicaltrials.gov/study/NCT01234567"
    assert "COMPLETED" in p.snippet


def test_source_diversity_selection():
    p_pub1 = Passage(title="P1", source="pubmed", url="u1", publication_date="", snippet="s", relevance_score=0.9)
    p_pub2 = Passage(title="P2", source="pubmed", url="u2", publication_date="", snippet="s", relevance_score=0.8)
    p_fda = Passage(title="F1", source="openfda", url="u3", publication_date="", snippet="s", relevance_score=0.85)
    p_who = Passage(title="W1", source="who", url="u4", publication_date="", snippet="s", relevance_score=0.75)
    
    passages = [p_pub1, p_pub2, p_fda, p_who]
    selected = HealthcareAdapter._select_source_diverse_passages(passages, k=3)
    assert len(selected) == 3
    sources = {p.source for p in selected}
    assert sources == {"pubmed", "openfda", "who"}
