from __future__ import annotations

import logging
from typing import List
from bs4 import BeautifulSoup

from config.settings import get_settings
from schemas.models import Passage, AdapterMetadata
from utils.async_executor import gather_results
from utils.http_client import ResilientHttpClient, get_client

logger = logging.getLogger(__name__)

class LegalGeneralAdapter:
    def __init__(self) -> None:
        self.name = "legal_general"
        self.courtlistener_key = get_settings().courtlistener_key

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self.name,
            version="1.0.0",
            supported_domains=["courtlistener", "wikipedia"],
            supports_live_search=True,
            cacheable=True,
            priority=10,
            max_results=20,
            is_stub=False
        )

    def credibility_of(self, source_id: str) -> float:
        if source_id.startswith("courtlistener"): return 0.95
        if source_id.startswith("wiki"): return 0.80
        return 0.75

    async def search(
        self,
        query: str,
        k: int = 5,
        source_mode: Optional[str] = None,
    ) -> List[Passage]:
        passages: List[Passage] = []
        try:
            client = get_client()

            if source_mode:
                mode_clean = source_mode.lower().strip()
                if mode_clean in ("legal-courtlistener", "courtlistener"):
                    return await self._search_courtlistener(client, query, k)
                if mode_clean in ("legal-wikipedia", "wikipedia"):
                    return await self._search_wikipedia(client, query, k)

            results = await gather_results([
                self._search_courtlistener(client, query, k),
                self._search_wikipedia(client, query, k),
            ])
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Legal source search error: {result}")
                elif isinstance(result, list):
                    passages.extend(result)
                
        except Exception as e:
            logger.error(f"Failed legal general search: {e}")

        # Deduplicate passages
        seen = set()
        deduped = []
        for p in passages:
            key = p.url or p.source_id
            if key not in seen:
                seen.add(key)
                deduped.append(p)
            
        return sorted(deduped, key=lambda x: x.relevance_score, reverse=True)[:k] if deduped else deduped

    async def _search_courtlistener(self, client: ResilientHttpClient, query: str, k: int) -> List[Passage]:
        try:
            headers = {"Authorization": f"Token {self.courtlistener_key}"} if self.courtlistener_key else None
            res = await client.get(
                "https://www.courtlistener.com/api/rest/v4/search/",
                adapter_name=self.name,
                params={"q": query, "type": "o", "order_by": "score desc"},
                headers=headers,
            )
            results = res.json().get("results", [])
            passages = []
            for item in results[:k]:
                case_name = item.get("caseName") or item.get("case_name") or "CourtListener Opinion"
                absolute_url = item.get("absolute_url") or item.get("cluster", "")
                url = f"https://www.courtlistener.com{absolute_url}" if str(absolute_url).startswith("/") else absolute_url
                snippet = item.get("snippet") or item.get("plain_text") or item.get("suitNature") or ""
                snippet = BeautifulSoup(str(snippet), "html.parser").get_text()
                date_filed = item.get("dateFiled") or item.get("date_filed") or "2024"
                passages.append(Passage(
                    title=f"CourtListener: {case_name}",
                    source="courtlistener",
                    url=url or "https://www.courtlistener.com/",
                    publication_date=str(date_filed)[:10],
                    snippet=f"Legal opinion [{case_name}]: {snippet[:300]}",
                    source_id=f"courtlistener_{item.get('cluster_id', item.get('id', 'opinion'))}",
                    relevance_score=0.5
                ))
            return passages
        except Exception as e:
            logger.warning(f"CourtListener search search failed. URL: https://www.courtlistener.com/api/rest/v4/search/. Error: {e}")
            return []

    async def _search_wikipedia(self, client: ResilientHttpClient, query: str, k: int) -> List[Passage]:
        try:
            search_query = f"{query} law legal act"
            res = await client.get(
                "https://en.wikipedia.org/w/rest.php/v1/search/page",
                adapter_name=self.name,
                params={"q": search_query, "limit": k},
                headers={"User-Agent": "HalluciGuard/2.0 (compliance@halluciguard.ai)"},
            )
            
            search_results = res.json().get("pages", [])
            passages = []
            for item in search_results:
                title = item.get("title", "Legal Article")
                snippet = BeautifulSoup(item.get("excerpt", ""), "html.parser").get_text()
                title_url = item.get("key", title.replace(" ", "_"))
                
                passages.append(Passage(
                    title=f"Wikipedia Legal: {title}",
                    source="wikipedia",
                    url=f"https://en.wikipedia.org/wiki/{title_url}",
                    publication_date="unknown",
                    snippet=f"Legal Reference [{title}]: {snippet[:300]}",
                    source_id=f"wiki_{title_url}",
                    relevance_score=0.5
                ))
            return passages
        except Exception as e:
            logger.warning(f"Wikipedia legal search search failed. URL: https://en.wikipedia.org/w/rest.php/v1/search/page. Error: {e}")
            return []
