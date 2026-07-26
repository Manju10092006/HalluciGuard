from __future__ import annotations

import logging
from typing import List
from bs4 import BeautifulSoup

import httpx

from schemas.models import Passage, AdapterMetadata

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

    async def search(self, query: str, k: int = 5) -> List[Passage]:
        passages: List[Passage] = []
        try:
            async with httpx.AsyncClient() as client:
                url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json&srlimit={k}"
                res = await client.get(url, timeout=10.0)
                res.raise_for_status()
                
                search_results = res.json().get("query", {}).get("search", [])
                for item in search_results:
                    title = item.get("title", "Wikipedia Article")
                    snippet_html = item.get("snippet", "")
                    snippet = BeautifulSoup(snippet_html, "html.parser").get_text()
                    title_url = title.replace(" ", "_")
                    
                    passages.append(Passage(
                        title=f"Wikipedia: {title}",
                        source="wikipedia",
                        url=f"https://en.wikipedia.org/wiki/{title_url}",
                        publication_date="2024",
                        snippet=f"Article [{title}]: {snippet[:300]}",
                        source_id=f"wiki_{title_url}",
                        relevance_score=0.8
                    ))
        except Exception as e:
            logger.error(f"Failed general search: {e}")
            
        return passages
