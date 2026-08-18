from __future__ import annotations

import logging
import urllib.parse
from typing import List
from bs4 import BeautifulSoup

from schemas.models import Passage, AdapterMetadata
from utils.http_client import get_client

logger = logging.getLogger(__name__)

class GeneralAdapter:
    def __init__(self) -> None:
        self.name = "general"

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self.name,
            version="1.0.0",
            supported_domains=["wikipedia"],
            supports_live_search=True,
            cacheable=True,
            priority=5,
            max_results=50,
            is_stub=False
        )

    def credibility_of(self, source_id: str) -> float:
        return 0.80

    def _clean_query(self, query: str) -> str:
        import re
        clean = re.sub(r'[*#_`]', '', query or '')
        clean = re.sub(r'[\r\n\t]+', ' ', clean).strip()
        words = clean.split()
        if len(words) > 25:
            clean = " ".join(words[:25])
        return clean[:200]

    async def search(self, query: str, k: int = 5) -> List[Passage]:
        passages: List[Passage] = []
        cleaned_q = self._clean_query(query)
        if not cleaned_q:
            return passages

        client = get_client()
        headers = {"User-Agent": "HalluciGuard/2.0 (compliance@halluciguard.ai)"}

        # Primary: the documented Wikipedia REST search endpoint.
        try:
            res = await client.get(
                "https://en.wikipedia.org/w/rest.php/v1/search/page",
                adapter_name=self.name,
                params={"q": cleaned_q, "limit": k},
                headers=headers,
            )
            items = res.json().get("pages", [])
            for item in items:
                title = item.get("title", "Wikipedia Article")
                snippet_html = item.get("excerpt", "") or item.get("description", "")
                snippet = BeautifulSoup(snippet_html, "html.parser").get_text().strip()
                title_url = urllib.parse.quote(title.replace(' ', '_'), safe='')

                passages.append(Passage(
                    title=f"Wikipedia: {title}",
                    source="wikipedia",
                    url=f"https://en.wikipedia.org/wiki/{title_url}",
                    publication_date="unknown",
                    snippet=f"Article [{title}]: {snippet[:300]}",
                    source_id=f"wiki_{title_url}",
                    relevance_score=0.7
                ))
            if passages:
                return passages
        except Exception as e:
            logger.warning(f"Wikipedia Action API search failed for query '{cleaned_q}': {e}. Trying REST API fallback.")

        # Fallback: MediaWiki Action API when the REST endpoint is unavailable.
        try:
            res = await client.get(
                "https://en.wikipedia.org/w/api.php",
                adapter_name=self.name,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": cleaned_q,
                    "format": "json",
                    "srlimit": k,
                },
                headers=headers,
            )
            search_results = res.json().get("query", {}).get("search", [])
            for item in search_results:
                title = item.get("title", "Wikipedia Article")
                snippet_html = item.get("snippet", "")
                snippet = BeautifulSoup(snippet_html, "html.parser").get_text().strip()
                title_url = urllib.parse.quote(title.replace(' ', '_'), safe='')
                
                passages.append(Passage(
                    title=f"Wikipedia: {title}",
                    source="wikipedia",
                    url=f"https://en.wikipedia.org/wiki/{title_url}",
                    publication_date="unknown",
                    snippet=f"Article [{title}]: {snippet[:300]}",
                    source_id=f"wiki_{title_url}",
                    relevance_score=0.5
                ))
        except Exception as e:
            logger.error(f"Failed general search fallback: {e}")
            
        return passages
