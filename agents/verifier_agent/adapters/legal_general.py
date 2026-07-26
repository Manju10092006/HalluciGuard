from __future__ import annotations

import asyncio
import logging
from typing import List

import httpx

from schemas.models import Passage, AdapterMetadata

logger = logging.getLogger(__name__)

class LegalGeneralAdapter:
    def __init__(self) -> None:
        self.name = "legal_general"
        self._curated_acts = {
            "ipc": "Indian Penal Code, major sections cover criminal offenses.",
            "crpc": "Code of Criminal Procedure, sets out the procedure for the administration of criminal law in India.",
            "constitution": "Constitution of India, the supreme law of India.",
            "it act 2000": "Information Technology Act, 2000 deals with cybercrime and electronic commerce.",
            "bns 2023": "Bharatiya Nyaya Sanhita, 2023 replaces the IPC."
        }

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self.name,
            version="1.0.0",
            supported_domains=["wikipedia", "curated_acts"],
            supports_live_search=True,
            cacheable=True,
            priority=10,
            max_results=20,
            is_stub=False
        )

    async def search(self, query: str, k: int = 5) -> List[Passage]:
        passages: List[Passage] = []
        try:
            # Check curated acts
            query_lower = query.lower()
            for act_key, act_desc in self._curated_acts.items():
                if act_key in query_lower:
                    passages.append(Passage(
                        text=act_desc,
                        source_id=f"curated_{act_key.replace(' ', '_')}",
                        source_url="",
                        relevance_score=0.9
                    ))

            async with httpx.AsyncClient() as client:
                wiki_results = await self._search_wikipedia(client, query, k)
                passages.extend(wiki_results)
                
        except Exception as e:
            logger.error(f"Failed legal general search: {e}")
            
        return sorted(passages, key=lambda x: x.relevance_score, reverse=True)[:k] if passages else passages

    async def _search_wikipedia(self, client: httpx.AsyncClient, query: str, k: int) -> List[Passage]:
        try:
            search_query = f"{query} law legal act"
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={search_query}&format=json&srlimit={k}"
            res = await client.get(url, timeout=10.0)
            res.raise_for_status()
            
            search_results = res.json().get("query", {}).get("search", [])
            passages = []
            for item in search_results:
                title = item.get("title", "")
                snippet = item.get("snippet", "").replace('<span class="searchmatch">', '').replace('</span>', '')
                title_url = title.replace(" ", "_")
                
                passages.append(Passage(
                    text=f"{title}: {snippet}",
                    source_id=f"wiki_{title_url}",
                    source_url=f"https://en.wikipedia.org/wiki/{title_url}",
                    relevance_score=0.8
                ))
            return passages
        except Exception as e:
            logger.error(f"Wikipedia legal search error: {e}")
            return []

    def credibility_of(self, source_id: str) -> float:
        # Honest about limited coverage
        return 0.75
