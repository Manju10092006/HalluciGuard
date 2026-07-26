from __future__ import annotations

import asyncio
import logging
from typing import List
from bs4 import BeautifulSoup

import httpx

from schemas.models import Passage, AdapterMetadata

logger = logging.getLogger(__name__)

class AiResearchAdapter:
    def __init__(self) -> None:
        self.name = "ai_research"

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self.name,
            version="1.0.0",
            supported_domains=["arxiv", "semanticscholar", "crossref"],
            supports_live_search=True,
            cacheable=True,
            priority=10,
            max_results=50,
            is_stub=False
        )

    def credibility_of(self, source_id: str) -> float:
        if source_id.startswith("arxiv"): return 0.90
        if source_id.startswith("s2"): return 0.95
        if source_id.startswith("crossref"): return 0.92
        return 0.85

    async def search(self, query: str, k: int = 5) -> List[Passage]:
        passages: List[Passage] = []
        try:
            async with httpx.AsyncClient() as client:
                results = await asyncio.gather(
                    self._search_arxiv(client, query, k),
                    self._search_semanticscholar(client, query, k),
                    self._search_crossref(client, query, k),
                    return_exceptions=True
                )
                
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"AI research source search error: {result}")
                    elif isinstance(result, list):
                        passages.extend(result)
        except Exception as e:
            logger.error(f"Failed ai_research search: {e}")
            
        return sorted(passages, key=lambda x: x.relevance_score, reverse=True)[:k] if passages else passages

    async def _search_arxiv(self, client: httpx.AsyncClient, query: str, k: int) -> List[Passage]:
        try:
            url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={k}"
            res = await client.get(url, timeout=10.0)
            res.raise_for_status()
            
            soup = BeautifulSoup(res.text, "xml")
            entries = soup.find_all("entry")
            
            passages = []
            for entry in entries:
                title = entry.title.text.strip() if entry.title else "arXiv Paper"
                summary = entry.summary.text.strip() if entry.summary else ""
                url_id = entry.id.text.strip() if entry.id else ""
                published = entry.published.text.strip() if entry.published else "2024"
                
                passages.append(Passage(
                    title=f"arXiv Paper: {title}",
                    source="arxiv",
                    url=url_id,
                    publication_date=published[:10],
                    snippet=f"Paper Title [{title}]: {summary[:300]}",
                    source_id=f"arxiv_{url_id.split('/')[-1]}",
                    relevance_score=0.9
                ))
            return passages
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return []

    async def _search_semanticscholar(self, client: httpx.AsyncClient, query: str, k: int) -> List[Passage]:
        try:
            url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={k}&fields=title,abstract,url,year,externalIds"
            res = await client.get(url, timeout=10.0)
            res.raise_for_status()
            
            data = res.json().get("data", [])
            passages = []
            for item in data:
                title = item.get("title", "Paper")
                abstract = item.get("abstract", "")
                url_val = item.get("url", "https://semanticscholar.org")
                year = item.get("year", "2024")
                
                passages.append(Passage(
                    title=f"Semantic Scholar: {title}",
                    source="semantic_scholar",
                    url=url_val or "https://semanticscholar.org",
                    publication_date=str(year),
                    snippet=f"Abstract [{title} ({year})]: {abstract[:300]}",
                    source_id=f"s2_{item.get('paperId', 'paper')}",
                    relevance_score=0.92
                ))
            return passages
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            return []

    async def _search_crossref(self, client: httpx.AsyncClient, query: str, k: int) -> List[Passage]:
        try:
            url = f"https://api.crossref.org/works?query={query}&rows={k}&select=title,abstract,URL,published-print,container-title"
            headers = {"User-Agent": "HalluciGuard/2.0 (mailto:compliance@halluciguard.ai)"}
            res = await client.get(url, headers=headers, timeout=10.0)
            res.raise_for_status()
            
            items = res.json().get("message", {}).get("items", [])
            passages = []
            for item in items:
                title = item.get("title", ["Crossref Work"])[0]
                url_val = item.get("URL", "https://crossref.org")
                abstract = item.get("abstract", "")
                
                passages.append(Passage(
                    title=f"Crossref: {title}",
                    source="crossref",
                    url=url_val,
                    publication_date="2024",
                    snippet=f"Publication [{title}]: {abstract[:300]}",
                    source_id=f"crossref_{url_val.split('/')[-1]}",
                    relevance_score=0.88
                ))
            return passages
        except Exception as e:
            logger.error(f"Crossref search error: {e}")
            return []
