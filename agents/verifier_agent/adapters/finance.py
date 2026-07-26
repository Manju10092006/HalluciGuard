from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

import httpx

from schemas.models import Passage, AdapterMetadata

logger = logging.getLogger(__name__)

class FinanceAdapter:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.name = "finance"
        self.api_key = api_key

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self.name,
            version="1.0.0",
            supported_domains=["sec", "worldbank", "alphavantage"],
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
                tasks = [
                    self._search_sec(client, query, k),
                    self._search_worldbank(client, query, k)
                ]
                if self.api_key:
                    tasks.append(self._search_alphavantage(client, query, k))
                    
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Finance source search error: {result}")
                    elif isinstance(result, list):
                        passages.extend(result)
        except Exception as e:
            logger.error(f"Failed finance search: {e}")
            
        return sorted(passages, key=lambda x: x.relevance_score, reverse=True)[:k] if passages else passages

    async def _search_sec(self, client: httpx.AsyncClient, query: str, k: int) -> List[Passage]:
        try:
            url = f"https://efts.sec.gov/LATEST/search-index?q={query}&dateRange=custom&startdt=2020-01-01&forms=10-K,10-Q,8-K&hits.hits.total=value"
            headers = {"User-Agent": "HalluciGuard/2.0 (compliance@halluciguard.ai)"}
            res = await client.get(url, headers=headers, timeout=10.0)
            res.raise_for_status()
            
            hits = res.json().get("hits", {}).get("hits", [])
            passages = []
            for item in hits[:k]:
                source = item.get("_source", {})
                date = source.get("filing_date", "")
                entity = source.get("display_names", [""])[0] if source.get("display_names") else ""
                desc = source.get("file_description", "")
                
                passages.append(Passage(
                    text=f"SEC Filing by {entity} on {date}: {desc}",
                    source_id=f"sec_{entity}",
                    source_url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={entity}",
                    relevance_score=0.9
                ))
            return passages
        except Exception as e:
            logger.error(f"SEC search error: {e}")
            return []

    async def _search_worldbank(self, client: httpx.AsyncClient, query: str, k: int) -> List[Passage]:
        try:
            url = f"https://api.worldbank.org/v2/country/all?format=json&per_page=50"
            res = await client.get(url, timeout=10.0)
            res.raise_for_status()
            
            data = res.json()
            passages = []
            if len(data) > 1:
                items = data[1]
                query_lower = query.lower()
                for item in items:
                    name = item.get("name", "")
                    if query_lower in name.lower() or query_lower in item.get("id", "").lower():
                        passages.append(Passage(
                            text=f"World Bank Data for {name} (Capital: {item.get('capitalCity', '')}, Region: {item.get('region', {}).get('value', '')})",
                            source_id=f"wb_{item.get('id')}",
                            source_url=f"https://data.worldbank.org/country/{item.get('id')}",
                            relevance_score=0.85
                        ))
                        if len(passages) >= k:
                            break
            return passages
        except Exception as e:
            logger.error(f"World Bank search error: {e}")
            return []

    async def _search_alphavantage(self, client: httpx.AsyncClient, query: str, k: int) -> List[Passage]:
        # Implementation would extract ticker from query and search. Placeholder for logic.
        try:
            # Assuming 'query' is a ticker symbol for simplicity in this example
            ticker = query.split()[0].upper()
            url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={self.api_key}"
            res = await client.get(url, timeout=10.0)
            res.raise_for_status()
            
            data = res.json()
            if "Symbol" in data:
                return [Passage(
                    text=f"Alpha Vantage Overview for {data['Symbol']}: {data.get('Description', '')[:200]}...",
                    source_id=f"alpha_{data['Symbol']}",
                    source_url=f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={data['Symbol']}",
                    relevance_score=0.88
                )]
            return []
        except Exception as e:
            logger.error(f"Alpha Vantage search error: {e}")
            return []

    def credibility_of(self, source_id: str) -> float:
        if source_id.startswith("sec"): return 0.98
        if source_id.startswith("wb"): return 0.95
        if source_id.startswith("alpha"): return 0.9
        return 0.8
