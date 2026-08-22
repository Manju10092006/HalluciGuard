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

    async def _fetch_page_summary(self, client: object, title: str) -> str:
        """Fetch the clean lead extract from Wikipedia REST summary API."""
        try:
            title_url = urllib.parse.quote(title.replace(' ', '_'), safe='')
            res = await client.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{title_url}",
                adapter_name=self.name,
                headers={"User-Agent": "HalluciGuard/2.0 (compliance@halluciguard.ai)"},
            )
            data = res.json()
            extract = data.get("extract", "")
            if extract and len(extract.strip()) > 30:
                return extract.strip()
        except Exception:
            pass
        return ""

    @staticmethod
    def _is_noisy_title(title: str, query: str, description: str = "") -> bool:
        """Filter out bibliographies/discographies/albums/songs/entertainment titles unless requested in the claim."""
        t_lower = title.lower()
        q_lower = query.lower()
        d_lower = (description or "").lower()

        entertainment_tags = (
            "bibliography", "discography", "filmography", "list of songs", "list of",
            "(album)", "(song)", "(film)", "(soundtrack)", "(tv series)", "(single)",
            "(band)", "(video game)", "(novel)", "(short story)", "(play)"
        )
        for noise in entertainment_tags:
            if noise in t_lower and noise not in q_lower:
                return True

        # Check description tag from Wikipedia search API
        if d_lower:
            for desc_noise in ("studio album", "song by", "film directed by", "television series", "single by", "band"):
                if desc_noise in d_lower and desc_noise not in q_lower:
                    return True

        return False

    async def search(self, query: str, k: int = 5) -> List[Passage]:
        passages: List[Passage] = []
        cleaned_q = self._clean_query(query)
        if not cleaned_q:
            return passages

        client = get_client()
        headers = {"User-Agent": "HalluciGuard/2.0 (compliance@halluciguard.ai)"}

        candidate_pages = []

        # Primary: Wikipedia REST search endpoint
        try:
            res = await client.get(
                "https://en.wikipedia.org/w/rest.php/v1/search/page",
                adapter_name=self.name,
                params={"q": cleaned_q, "limit": max(k, 5)},
                headers=headers,
            )
            candidate_pages = res.json().get("pages", [])
        except Exception as e:
            logger.warning(f"Wikipedia REST search failed for '{cleaned_q}': {e}. Trying Action API.")

        # Fallback: MediaWiki Action API
        if not candidate_pages:
            try:
                res = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    adapter_name=self.name,
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": cleaned_q,
                        "format": "json",
                        "srlimit": max(k, 5),
                    },
                    headers=headers,
                )
                candidate_pages = res.json().get("query", {}).get("search", [])
            except Exception as e:
                logger.error(f"Failed general search fallback: {e}")
                return []

        # Process candidates and fetch lead extracts
        for item in candidate_pages:
            title = item.get("title", "Wikipedia Article")
            description = item.get("description", "")
            if self._is_noisy_title(title, query, description):
                continue

            snippet_html = item.get("excerpt", "") or description or item.get("snippet", "")
            fallback_snippet = BeautifulSoup(snippet_html, "html.parser").get_text().strip()
            
            # Fetch authoritative lead paragraph
            summary_extract = await self._fetch_page_summary(client, title)
            final_snippet = summary_extract or fallback_snippet
            if not final_snippet or len(final_snippet.strip()) < 20:
                continue

            title_url = urllib.parse.quote(title.replace(' ', '_'), safe='')
            passages.append(Passage(
                title=f"Wikipedia: {title}",
                source="wikipedia",
                url=f"https://en.wikipedia.org/wiki/{title_url}",
                publication_date="unknown",
                snippet=f"Article [{title}]: {final_snippet[:400]}",
                source_id=f"wiki_{title_url.lower()}",
                relevance_score=0.85 if summary_extract else 0.70,
            ))
            if len(passages) >= k:
                break

        return passages
