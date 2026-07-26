from __future__ import annotations

import logging
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

    async def search(self, query: str, k: int = 5) -> List[Passage]:
        passages: List[Passage] = []
        try:
            client = get_client()
            res = await client.get(
                "https://en.wikipedia.org/w/rest.php/v1/search/page",
                adapter_name=self.name,
                params={"q": query, "limit": k},
                headers={"User-Agent": "HalluciGuard/2.0 (compliance@halluciguard.ai)"},
            )
            
            search_results = res.json().get("pages", [])
            for item in search_results:
                title = item.get("title", "Wikipedia Article")
                snippet_html = item.get("excerpt", "") or item.get("description", "")
                snippet = BeautifulSoup(snippet_html, "html.parser").get_text()
                title_url = item.get("key", title.replace(" ", "_"))
                
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
