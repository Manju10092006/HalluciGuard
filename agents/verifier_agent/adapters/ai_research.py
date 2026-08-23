from __future__ import annotations

import logging
from typing import List
from bs4 import BeautifulSoup

from config.settings import get_settings
from schemas.models import Passage, AdapterMetadata
from utils.async_executor import gather_results
from utils.http_client import ResilientHttpClient, get_client

logger = logging.getLogger(__name__)

class AiResearchAdapter:
    def __init__(self) -> None:
        self.name = "ai_research"
        self.semantic_scholar_key = get_settings().semantic_scholar_key

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
        if source_id.startswith("arxiv"): return 0.95
        if source_id.startswith("s2") or source_id.startswith("semantic"): return 0.96
        if source_id.startswith("crossref"): return 0.94
        return 0.90

    async def search(
        self,
        query: str,
        k: int = 5,
        source_mode: Optional[str] = None,
        **kwargs,
    ) -> List[Passage]:
        passages: List[Passage] = []
        try:
            client = get_client()

            if source_mode:
                mode_clean = source_mode.lower().strip()
                if mode_clean in ("ai-arxiv", "arxiv"):
                    return await self._search_arxiv(client, query, k)
                if mode_clean in ("ai-semanticscholar", "semanticscholar", "s2"):
                    return await self._search_semanticscholar(client, query, k)
                if mode_clean in ("ai-crossref", "crossref"):
                    return await self._search_crossref(client, query, k)

            results = await gather_results([
                self._search_arxiv(client, query, k),
                self._search_semanticscholar(client, query, k),
                self._search_crossref(client, query, k),
            ])
                
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"AI research source search error: {result}")
                elif isinstance(result, list):
                    passages.extend(result)
        except Exception as e:
            logger.error(f"Failed ai_research search: {e}")

        # Deduplicate passages
        seen = set()
        deduped = []
        for p in passages:
            key = p.url or p.source_id
            if key not in seen:
                seen.add(key)
                deduped.append(p)
            
        return sorted(deduped, key=lambda x: x.relevance_score, reverse=True)[:k] if deduped else deduped

    def _sanitize_arxiv_query(self, query: str) -> str:
        import re
        clean = re.sub(r"[^\w\s\-]", " ", query)
        words = [w for w in clean.split() if len(w) > 2]
        return " ".join(words[:6]) if words else query[:50]

    async def _search_arxiv(self, client: ResilientHttpClient, query: str, k: int) -> List[Passage]:
        try:
            clean_q = self._sanitize_arxiv_query(query)
            res = await client.get(
                "https://export.arxiv.org/api/query",
                adapter_name=self.name,
                params={"search_query": f"all:{clean_q}", "start": 0, "max_results": k},
            )
            
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
                    relevance_score=0.0,
                    source_confidence_hint=0.80,
                ))
            return passages
        except Exception as e:
            logger.warning(f"arXiv search search failed. URL: https://export.arxiv.org/api/query. Error: {e}")
            return []

    async def _search_semanticscholar(self, client: ResilientHttpClient, query: str, k: int) -> List[Passage]:
        try:
            headers = {"x-api-key": self.semantic_scholar_key} if self.semantic_scholar_key else None
            res = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                adapter_name=self.name,
                params={"query": query, "limit": k, "fields": "title,abstract,url,year,externalIds"},
                headers=headers,
            )
            
            data = res.json().get("data", [])
            passages = []
            for item in data:
                title = item.get("title", "Paper")
                abstract = item.get("abstract", "")
                url_val = item.get("url", "https://semanticscholar.org")
                year = item.get("year", "unknown")
                
                passages.append(Passage(
                    title=f"Semantic Scholar: {title}",
                    source="semantic_scholar",
                    url=url_val or "https://semanticscholar.org",
                    publication_date=str(year),
                    snippet=f"Abstract [{title} ({year})]: {abstract[:300]}",
                    source_id=f"s2_{item.get('paperId', 'paper')}",
                    relevance_score=0.0,
                    source_confidence_hint=0.85,
                ))
            return passages
        except Exception as e:
            logger.warning(f"Semantic Scholar search search failed. URL: https://api.semanticscholar.org/graph/v1/paper/search. Error: {e}")
            return []

    async def _search_crossref(self, client: ResilientHttpClient, query: str, k: int) -> List[Passage]:
        try:
            headers = {"User-Agent": "HalluciGuard/2.0 (mailto:compliance@halluciguard.ai)"}
            res = await client.get(
                "https://api.crossref.org/works",
                adapter_name=self.name,
                headers=headers,
                params={"query": query, "rows": k, "select": "title,abstract,URL,published-print,container-title,published-online"},
            )
            
            items = res.json().get("message", {}).get("items", [])
            passages = []
            for item in items:
                title = item.get("title", ["Crossref Work"])[0]
                url_val = item.get("URL", "https://crossref.org")
                abstract = item.get("abstract", "")
                published = item.get("published-print") or item.get("published-online") or {}
                date_parts = published.get("date-parts", [["2024"]])
                publication_date = "-".join(str(part) for part in date_parts[0]) if date_parts and date_parts[0] else "2024"
                
                passages.append(Passage(
                    title=f"Crossref: {title}",
                    source="crossref",
                    url=url_val,
                    publication_date=publication_date,
                    snippet=f"Publication [{title}]: {abstract[:300]}",
                    source_id=f"crossref_{url_val.split('/')[-1]}",
                    relevance_score=0.0,
                    source_confidence_hint=0.80,
                ))
            return passages
        except Exception as e:
            logger.warning(f"Crossref search search failed. URL: https://api.crossref.org/works. Error: {e}")
            return []
