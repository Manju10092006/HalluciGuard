from __future__ import annotations

import logging
import re
import urllib.parse
from typing import List, Optional

from config.settings import get_settings
from schemas.models import Passage, AdapterMetadata
from utils.async_executor import gather_results
from utils.http_client import ResilientHttpClient, get_client
from claims.entity_resolver import EntityResolver

logger = logging.getLogger(__name__)


class FinanceAdapter:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.name = "finance"
        settings = get_settings()
        self.api_key = api_key or settings.alpha_vantage_key
        self.sec_user_agent = settings.sec_user_agent
        self.entity_resolver = EntityResolver()

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
            is_stub=False,
        )

    def credibility_of(self, source_id: str) -> float:
        if source_id.startswith("sec"):
            return 0.98
        if source_id.startswith("wb"):
            return 0.95
        if source_id.startswith("alpha"):
            return 0.90
        return 0.85

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
                if mode_clean in ("finance-sec", "sec"):
                    return await self._search_sec(client, query, k)
                if mode_clean in ("finance-worldbank", "worldbank", "wb"):
                    return await self._search_worldbank(client, query, k)
                if mode_clean in ("finance-alpha", "alphavantage"):
                    return await self._search_alphavantage(client, query, k)

            # Resolve financial entities (Company name, Ticker, CIK)
            resolution = self.entity_resolver.resolve(query, "finance")
            sec_query = resolution.identifiers.get(
                "company_name", resolution.canonical_query or query
            )
            ticker = resolution.identifiers.get("ticker")

            tasks = [
                self._search_sec(client, sec_query, k),
                self._search_worldbank(client, resolution.canonical_query or query, k),
            ]
            if self.api_key:
                tasks.append(self._search_alphavantage(client, ticker or query, k))

            results = await gather_results(tasks)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Finance source search error: {result}")
                elif isinstance(result, list):
                    passages.extend(result)
        except Exception as e:
            logger.error(f"Failed finance search: {e}")

        # Deduplicate passages by URL
        seen = set()
        deduped = []
        for p in passages:
            key = p.url or p.source_id
            if key not in seen:
                seen.add(key)
                deduped.append(p)

        return (
            sorted(deduped, key=lambda x: x.relevance_score, reverse=True)[:k]
            if deduped
            else deduped
        )

    async def _search_sec(
        self, client: ResilientHttpClient, query: str, k: int
    ) -> List[Passage]:
        try:
            headers = {
                "User-Agent": self.sec_user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
            # Clean up query term for EFTS search
            search_q = query.strip()
            if len(search_q.split()) > 5:
                search_q = " ".join(search_q.split()[:3])

            res = await client.get(
                "https://efts.sec.gov/LATEST/search-index",
                adapter_name=self.name,
                headers=headers,
                params={
                    "q": search_q,
                    "dateRange": "custom",
                    "startdt": "2020-01-01",
                    "forms": "10-K,10-Q,8-K",
                    "hits.hits.total": "value",
                },
            )

            hits = res.json().get("hits", {}).get("hits", [])
            passages = []
            for item in hits[:k]:
                source = item.get("_source", {})
                date = source.get("filing_date", "unknown")
                entity = (
                    source.get("display_names", ["EDGAR Entity"])[0]
                    if source.get("display_names")
                    else "EDGAR Entity"
                )
                form = source.get("form", "Filing")
                desc = source.get("file_description", f"SEC {form} Filing Document.")

                passages.append(
                    Passage(
                        title=f"SEC Filing ({form}): {entity}",
                        source="sec_edgar",
                        url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={urllib.parse.quote_plus(entity)}",
                        publication_date=str(date)[:10],
                        snippet=f"SEC Form {form} by {entity} ({date}): {desc[:350]}",
                        source_id=f"sec_{entity}",
                        relevance_score=0.0,
                        source_confidence_hint=0.80,
                    )
                )
            return passages
        except Exception as e:
            logger.warning(f"SEC search search failed. URL: https://efts.sec.gov/LATEST/search-index. Error: {e}")
            return []

    async def _search_worldbank(
        self, client: ResilientHttpClient, query: str, k: int
    ) -> List[Passage]:
        try:
            res = await client.get(
                "https://api.worldbank.org/v2/indicator",
                adapter_name=self.name,
                params={"format": "json", "per_page": 100, "source": 2},
            )

            data = res.json()
            passages = []
            if len(data) > 1:
                items = data[1]
                query_words = [w.lower() for w in query.split() if len(w) > 3]
                for item in items:
                    name = str(item.get("name", "")).lower()
                    source_note = str(item.get("sourceNote", "")).lower()
                    ind_id = str(item.get("id", "")).lower()

                    if any(
                        qw in name or qw in ind_id or qw in source_note
                        for qw in query_words
                    ):
                        passages.append(
                            Passage(
                                title=f"World Bank Indicator: {item.get('name')}",
                                source="world_bank",
                                url=f"https://data.worldbank.org/indicator/{item.get('id')}",
                                publication_date="unknown",
                                snippet=f"World Bank indicator {item.get('id')} [{item.get('name')}]: {source_note[:350]}",
                                source_id=f"wb_{item.get('id')}",
                                relevance_score=0.0,
                                source_confidence_hint=0.75,
                            )
                        )
                        if len(passages) >= k:
                            break
            return passages
        except Exception as e:
            logger.warning(f"World Bank search search failed. URL: https://api.worldbank.org/v2/indicator. Error: {e}")
            return []

    async def _search_alphavantage(
        self, client: ResilientHttpClient, query: str, k: int
    ) -> List[Passage]:
        try:
            match = re.search(r'\b[A-Z]{1,5}\b', query)
            if match:
                ticker = match.group(0)
            else:
                capitalized = [w for w in query.split() if w.istitle()]
                ticker = capitalized[0].upper() if capitalized else query.split()[0].upper()
                
            res = await client.get(
                "https://www.alphavantage.co/query",
                adapter_name=self.name,
                params={
                    "function": "OVERVIEW",
                    "symbol": ticker,
                    "apikey": self.api_key,
                },
            )

            data = res.json()
            if "Symbol" in data:
                return [
                    Passage(
                        title=f"Alpha Vantage: {data['Symbol']} Overview",
                        source="alpha_vantage",
                        url=f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={data['Symbol']}",
                        publication_date="unknown",
                        snippet=f"Overview for {data.get('Name', data['Symbol'])} ({data['Symbol']}): {data.get('Description', '')[:350]}",
                        source_id=f"alpha_{data['Symbol']}",
                        relevance_score=0.0,
                        source_confidence_hint=0.85,
                    )
                ]
            return []
        except Exception as e:
            logger.warning(f"Alpha Vantage search search failed. URL: https://www.alphavantage.co/query. Error: {e}")
            return []
