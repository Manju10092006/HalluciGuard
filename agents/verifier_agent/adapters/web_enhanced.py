"""
HalluciGuard Verifier Agent — Web-Enhanced Domain Adapter.

Wraps any existing domain adapter with Tavily web retrieval as a fallback.
When the primary adapter returns insufficient evidence, Tavily supplements.

Integration into existing pipeline:
  - Does NOT replace existing adapters
  - Does NOT change DomainAdapter Protocol
  - Registered alongside existing adapters in AdapterRegistry
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from schemas.models import Passage, AdapterMetadata

logger = logging.getLogger(__name__)

# Minimum passages from primary adapter before triggering web fallback
MIN_PRIMARY_EVIDENCE = 2


class WebEnhancedAdapter:
    """
    Adapter wrapper that adds Tavily web retrieval as fallback.

    Usage:
        primary = GeneralAdapter()
        enhanced = WebEnhancedAdapter(primary)
        passages = await enhanced.search("query")
        # If primary returns < MIN_PRIMARY_EVIDENCE passages,
        # Tavily supplements with web evidence.
    """

    def __init__(
        self,
        primary_adapter: object,
        min_primary: int = MIN_PRIMARY_EVIDENCE,
        tavily_api_key: Optional[str] = None,
    ) -> None:
        self._primary = primary_adapter
        self._min_primary = min_primary
        self._tavily_key = tavily_api_key if tavily_api_key is not None else os.environ.get("TAVILY_API_KEY", "")
        self._web_retriever = None  # Lazy

        # Mirror the primary adapter's name so registry routing still works
        self.name = getattr(primary_adapter, "name", "general")
        self.sources_attempted: List[str] = []
        self.sources_succeeded: List[str] = []
        self.sources_failed: List[str] = []

    def _get_web_retriever(self):
        key = self._tavily_key or os.environ.get("TAVILY_API_KEY", "")
        if self._web_retriever is None or not getattr(self._web_retriever, "_api_key", None):
            from .web_retriever import TavilyWebRetriever
            self._web_retriever = TavilyWebRetriever(api_key=key)
        return self._web_retriever

    @property
    def metadata(self) -> AdapterMetadata:
        primary_meta = getattr(self._primary, "metadata", None)
        if primary_meta:
            return AdapterMetadata(
                name=self.name,
                version=primary_meta.version,
                supported_domains=primary_meta.supported_domains + ["tavily_web"],
                supports_live_search=True,
                cacheable=primary_meta.cacheable,
                priority=primary_meta.priority,
                max_results=primary_meta.max_results,
                is_stub=False,
            )
        return AdapterMetadata(
            name=self.name,
            version="1.0.0",
            supported_domains=["tavily_web"],
            supports_live_search=True,
            cacheable=True,
            priority=5,
            max_results=10,
            is_stub=False,
        )

    def credibility_of(self, source_id: str) -> float:
        """Delegate to primary adapter for known sources, web retriever for web sources."""
        if source_id.startswith("tavily_"):
            retriever = self._get_web_retriever()
            return retriever.credibility_of(source_id)
        if hasattr(self._primary, "credibility_of"):
            return self._primary.credibility_of(source_id)
        return 0.70

    async def search(self, query: str, k: int = 5) -> List[Passage]:
        """
        Search primary adapter first; if insufficient results, supplement with Tavily.
        """
        self.sources_attempted = []
        self.sources_succeeded = []
        self.sources_failed = []

        # ── Primary adapter search ─────────────────────────────────
        primary_passages: List[Passage] = []
        primary_name = getattr(self._primary, "name", "primary")
        self.sources_attempted.append(primary_name)

        try:
            search_fn = getattr(self._primary, "search")
            primary_passages = await search_fn(query, k)
            if primary_passages:
                self.sources_succeeded.append(primary_name)
                logger.info(
                    "Primary adapter '%s' returned %d passages",
                    primary_name,
                    len(primary_passages),
                )
            else:
                logger.info("Primary adapter '%s' returned 0 passages", primary_name)
        except Exception as e:
            logger.error("Primary adapter '%s' failed: %s", primary_name, e)
            self.sources_failed.append(primary_name)

        # Carry forward any source tracking from primary adapter
        if hasattr(self._primary, "sources_attempted"):
            self.sources_attempted = list(getattr(self._primary, "sources_attempted", []))
        if hasattr(self._primary, "sources_succeeded"):
            self.sources_succeeded = list(getattr(self._primary, "sources_succeeded", []))
        if hasattr(self._primary, "sources_failed"):
            self.sources_failed = list(getattr(self._primary, "sources_failed", []))

        # ── Web fallback if insufficient ───────────────────────────
        if len(primary_passages) >= self._min_primary:
            return primary_passages

        active_key = self._tavily_key or os.environ.get("TAVILY_API_KEY", "").strip()
        if not active_key:
            logger.info("Tavily API key not configured; skipping web fallback")
            return primary_passages

        self.sources_attempted.append("tavily_web")
        web_needed = max(1, k - len(primary_passages))

        try:
            retriever = self._get_web_retriever()
            web_passages = await retriever.search(query, k=web_needed)
            if web_passages:
                self.sources_succeeded.append("tavily_web")
                logger.info(
                    "Tavily web fallback returned %d passages",
                    len(web_passages),
                )
            else:
                logger.info("Tavily web fallback returned 0 passages")
        except Exception as e:
            logger.error("Tavily web fallback failed: %s", e)
            self.sources_failed.append("tavily_web")
            web_passages = []

        # ── Merge: primary first, web supplement ───────────────────
        # Deduplicate by normalized URL
        seen_urls = set()
        merged: List[Passage] = []
        for p in primary_passages + web_passages:
            from .web_retriever import _normalize_url
            norm = _normalize_url(p.url)
            if norm not in seen_urls:
                seen_urls.add(norm)
                merged.append(p)

        return merged[:k]
