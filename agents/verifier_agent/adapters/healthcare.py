from __future__ import annotations

import asyncio
import logging
from typing import List

import httpx

from schemas.models import Passage, AdapterMetadata

logger = logging.getLogger(__name__)

class HealthcareAdapter:
    def __init__(self) -> None:
        self.name = "healthcare"

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self.name,
            version="1.0.0",
            supported_domains=["pubmed", "openfda", "clinicaltrials"],
            supports_live_search=True,
            cacheable=True,
            priority=10,
            max_results=50,
            is_stub=False
        )

    def credibility_of(self, source_id: str) -> float:
        return 0.95

    async def search(self, query: str, k: int = 5) -> List[Passage]:
        passages: List[Passage] = []
        try:
            async with httpx.AsyncClient() as client:
                results = await asyncio.gather(
                    self._search_pubmed(client, query, k),
                    self._search_openfda(client, query, k),
                    self._search_clinicaltrials(client, query, k),
                    return_exceptions=True
                )
                
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Healthcare source search error: {result}")
                    elif isinstance(result, list):
                        passages.extend(result)
        except Exception as e:
            logger.error(f"Failed healthcare search: {e}")
            
        return sorted(passages, key=lambda x: x.relevance_score, reverse=True)[:k] if passages else passages

    async def _search_pubmed(self, client: httpx.AsyncClient, query: str, k: int) -> List[Passage]:
        try:
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmode=json&retmax={k}"
            search_res = await client.get(search_url, timeout=10.0)
            search_res.raise_for_status()
            
            ids = search_res.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []
                
            ids_str = ",".join(ids)
            summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={ids_str}&retmode=json"
            summary_res = await client.get(summary_url, timeout=10.0)
            summary_res.raise_for_status()
            
            result = summary_res.json().get("result", {})
            passages = []
            for pm_id in ids:
                if pm_id in result:
                    item = result[pm_id]
                    title = item.get("title", f"PubMed Article {pm_id}")
                    pub_date = item.get("pubdate", "2024")
                    passages.append(Passage(
                        title=title,
                        source="pubmed",
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pm_id}/",
                        publication_date=pub_date,
                        snippet=f"{title}. Published in PubMed ({pub_date}). ID: {pm_id}.",
                        source_id=f"pubmed_{pm_id}",
                        relevance_score=0.9
                    ))
            return passages
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []

    async def _search_openfda(self, client: httpx.AsyncClient, query: str, k: int) -> List[Passage]:
        try:
            url = f"https://api.fda.gov/drug/label.json?search={query}&limit={k}"
            res = await client.get(url, timeout=10.0)
            res.raise_for_status()
            
            data = res.json().get("results", [])
            passages = []
            for item in data:
                brand = item.get("openfda", {}).get("brand_name", ["Unknown Brand"])[0]
                desc_list = item.get("description", []) or item.get("indications_and_usage", [])
                desc = desc_list[0] if desc_list else "FDA approved drug label details."
                if desc:
                    passages.append(Passage(
                        title=f"FDA Label: {brand}",
                        source="openfda",
                        url="https://api.fda.gov/drug/label.json",
                        publication_date="2024",
                        snippet=f"Drug Label [{brand}]: {desc[:300]}",
                        source_id="openfda",
                        relevance_score=0.85
                    ))
            return passages
        except Exception as e:
            logger.error(f"OpenFDA search error: {e}")
            return []

    async def _search_clinicaltrials(self, client: httpx.AsyncClient, query: str, k: int) -> List[Passage]:
        try:
            url = f"https://clinicaltrials.gov/api/v2/studies?query.term={query}&pageSize={k}&format=json"
            res = await client.get(url, timeout=10.0)
            res.raise_for_status()
            
            data = res.json().get("studies", [])
            passages = []
            for study in data:
                protocol = study.get("protocolSection", {})
                nctId = protocol.get("identificationModule", {}).get("nctId", "")
                title = protocol.get("identificationModule", {}).get("briefTitle", "Clinical Study")
                status = protocol.get("statusModule", {}).get("overallStatus", "Unknown")
                summary = protocol.get("descriptionModule", {}).get("briefSummary", "")
                
                passages.append(Passage(
                    title=f"Clinical Trial [{nctId}]: {title}",
                    source="clinicaltrials",
                    url=f"https://clinicaltrials.gov/study/{nctId}",
                    publication_date="2024",
                    snippet=f"Study Status [{status}]: {summary[:300]}",
                    source_id=f"clinicaltrials_{nctId}",
                    relevance_score=0.88
                ))
            return passages
        except Exception as e:
            logger.error(f"ClinicalTrials search error: {e}")
            return []
