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
                    title = item.get("title", "")
                    passages.append(Passage(
                        text=title,
                        source_id=f"pubmed_{pm_id}",
                        source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pm_id}/",
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
                brand = item.get("openfda", {}).get("brand_name", [""])[0]
                desc = item.get("description", [""])[0] or item.get("indications_and_usage", [""])[0]
                if desc:
                    passages.append(Passage(
                        text=f"{brand}: {desc}",
                        source_id="openfda",
                        source_url="https://api.fda.gov/drug/label.json",
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
                title = protocol.get("identificationModule", {}).get("briefTitle", "")
                status = protocol.get("statusModule", {}).get("overallStatus", "")
                summary = protocol.get("descriptionModule", {}).get("briefSummary", "")
                
                if summary:
                    passages.append(Passage(
                        text=f"Title: {title}\nStatus: {status}\nSummary: {summary}",
                        source_id=f"clinicaltrials_{nctId}",
                        source_url=f"https://clinicaltrials.gov/study/{nctId}",
                        relevance_score=0.88
                    ))
            return passages
        except Exception as e:
            logger.error(f"ClinicalTrials search error: {e}")
            return []

    def credibility_of(self, source_id: str) -> float:
        if source_id.startswith("pubmed"): return 0.95
        if source_id.startswith("openfda"): return 0.98
        if source_id.startswith("clinicaltrials"): return 0.92
        return 0.8
