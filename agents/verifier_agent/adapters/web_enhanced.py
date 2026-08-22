"""
HalluciGuard Verifier Agent — Web-Enhanced Domain Adapter.

Wraps any existing domain adapter with quality-based Tavily web retrieval fallback.
Instead of counting raw passages, assesses primary evidence quality using BGE relevance
scores before deciding whether to invoke Tavily.

Retrieval modes:
  - hybrid (default): Primary first, Tavily fallback if quality gate fails
  - primary_only: Only primary adapter, no Tavily
  - tavily_only: Skip primary, go straight to Tavily (diagnostic mode)

Returns a RetrievalTrace alongside passages for full observability.
"""
from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Tuple

from schemas.models import Passage, AdapterMetadata
from schemas.retrieval_trace import (
    RetrievalTrace,
    PrimaryRetrievalTrace,
    TavilyRetrievalTrace,
    MergedTrace,
    FinalTrace,
)

logger = logging.getLogger(__name__)


class WebEnhancedAdapter:
    """
    Adapter wrapper that adds quality-based Tavily web retrieval fallback.

    Quality Gate Logic:
      1. Run primary adapter
      2. Assess passage quality using BGE relevance scores
      3. If quality is sufficient → skip Tavily
      4. If quality is insufficient → call Tavily → merge → deduplicate

    The quality gate considers:
      - Number of usable passages (non-empty title/snippet/URL)
      - Number of relevant passages (BGE score >= relevance_threshold)
      - Top relevance score
      - Source diversity
    """

    def __init__(
        self,
        primary_adapter: object,
        tavily_api_key: Optional[str] = None,
        min_primary: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._primary = primary_adapter
        self._tavily_key = tavily_api_key if tavily_api_key is not None else os.environ.get("TAVILY_API_KEY", "")
        self._web_retriever = None  # Lazy

        # Mirror the primary adapter's name so registry routing still works
        self.name = getattr(primary_adapter, "name", "general")
        self.sources_attempted: List[str] = []
        self.sources_succeeded: List[str] = []
        self.sources_failed: List[str] = []
        self.last_retrieval_trace: Optional[RetrievalTrace] = None

    def _get_settings(self):
        """Lazy-load settings to avoid circular imports."""
        from config.settings import get_settings
        return get_settings()

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

    @staticmethod
    def _is_usable_passage(p: Passage) -> bool:
        """Check if a passage has meaningful content for evidence assessment."""
        has_snippet = bool(p.snippet and len(p.snippet.strip()) > 20)
        has_url = bool(p.url and p.url.startswith("http"))
        has_title = bool(p.title and len(p.title.strip()) > 2)
        return has_snippet and (has_url or has_title)

    def _assess_primary_quality(
        self,
        passages: List[Passage],
        claim: str,
    ) -> PrimaryRetrievalTrace:
        """
        Assess whether primary passages provide sufficient decision-quality evidence.

        Uses existing BGE relevance scores already assigned by adapters/retrievers.
        Does NOT run a second BGE inference — reuses the relevance_score field.

        Quality Gate Criteria (configurable via settings):
          - At least min_relevant_passages with relevance >= relevance_threshold
          - Top relevance score >= min_top_relevance
        """
        settings = self._get_settings()

        usable = [p for p in passages if self._is_usable_passage(p)]
        usable_count = len(usable)

        if usable_count == 0:
            return PrimaryRetrievalTrace(
                called=True,
                result_count=len(passages),
                usable_count=0,
                relevant_count=0,
                top_relevance=0.0,
                source_diversity=0,
                sufficient=False,
                reason="PRIMARY_EMPTY",
            )

        import re
        stopwords = {
            "is", "the", "a", "an", "in", "of", "to", "and", "or", "for", "with",
            "that", "this", "from", "was", "were", "been", "have", "has", "had",
            "associated", "related", "called", "known", "named", "most", "dangerous"
        }
        claim_terms = [
            w.lower() for w in re.findall(r"[a-zA-Z0-9]+", claim)
            if len(w) > 3 and w.lower() not in stopwords and not (w.isdigit() and len(w) == 4)
        ]

        # Compute honest gate-time relevance from lexical overlap + confidence hint, or explicit score
        gate_relevance_scores = []
        for p in usable:
            if float(getattr(p, "relevance_score", 0.0) or 0.0) > 0.0:
                gate_relevance_scores.append(float(p.relevance_score))
            else:
                p_text = f"{p.title} {p.snippet}".lower()
                overlap = sum(1 for t in claim_terms if t in p_text)
                overlap_ratio = overlap / len(claim_terms) if claim_terms else 1.0
                hint = float(getattr(p, "source_confidence_hint", 0.0) or 0.70)
                score = round(overlap_ratio * 0.7 + hint * 0.3, 4)
                gate_relevance_scores.append(score)

        top_relevance = max(gate_relevance_scores) if gate_relevance_scores else 0.0
        relevant_count = sum(
            1 for score in gate_relevance_scores
            if score >= settings.relevance_threshold
        )

        # Source diversity: count unique source identifiers
        unique_sources = set()
        for p in usable:
            source_key = p.source or p.source_id or "unknown"
            unique_sources.add(source_key)
        source_diversity = len(unique_sources)

        # Check claim term coverage in primary passages to detect alias/colloquial gaps
        if claim_terms:
            combined_corpus = " ".join((p.title + " " + p.snippet).lower() for p in usable)
            covered_terms = sum(1 for term in claim_terms if term in combined_corpus)
            coverage_ratio = covered_terms / len(claim_terms)
        else:
            coverage_ratio = 1.0
            covered_terms = 0

        if len(claim_terms) <= 2:
            sufficient_coverage = (covered_terms >= 1)
        elif claim_terms:
            sufficient_coverage = (coverage_ratio >= 0.50)
        else:
            sufficient_coverage = True

        # Quality Gate Decision: must have relevant passages, top relevance, and sufficient term coverage
        sufficient = (
            relevant_count >= settings.min_relevant_passages
            and top_relevance >= settings.min_top_relevance
            and sufficient_coverage
        )

        if sufficient:
            reason = "PRIMARY_EVIDENCE_SUFFICIENT"
        elif not sufficient_coverage:
            reason = f"PRIMARY_EVIDENCE_INSUFFICIENT_TERM_COVERAGE ({covered_terms}/{len(claim_terms)} terms covered)"
        elif relevant_count == 0:
            reason = "PRIMARY_EVIDENCE_INSUFFICIENT_RELEVANCE"
        elif top_relevance < settings.min_top_relevance:
            reason = f"PRIMARY_TOP_RELEVANCE_TOO_LOW ({top_relevance:.3f} < {settings.min_top_relevance})"
        else:
            reason = f"PRIMARY_EVIDENCE_INSUFFICIENT ({relevant_count} relevant < {settings.min_relevant_passages} required)"

        return PrimaryRetrievalTrace(
            called=True,
            result_count=len(passages),
            usable_count=usable_count,
            relevant_count=relevant_count,
            top_relevance=round(top_relevance, 4),
            source_diversity=source_diversity,
            sufficient=sufficient,
            reason=reason,
        )

    async def search(
        self,
        query: str,
        k: int = 5,
        retrieval_mode: str = "hybrid",
        source_mode: Optional[str] = None,
    ) -> List[Passage]:
        """
        Search with quality-based Tavily fallback.

        Args:
            query: The claim/search query.
            k: Maximum passages to return.
            retrieval_mode: "hybrid" (default), "primary_only", "tavily_only"
            source_mode: Optional domain-specific source override (e.g. "healthcare-pubmed")

        Returns:
            List of Passage objects. Also sets self.last_retrieval_trace.
        """
        self.sources_attempted = []
        self.sources_succeeded = []
        self.sources_failed = []

        trace = RetrievalTrace(
            requested_domain=self.name,
            primary_adapter=getattr(self._primary, "name", "unknown"),
            retrieval_mode=retrieval_mode,
        )

        # ── MODE: tavily_only (diagnostic / --force-tavily) ────────
        if retrieval_mode == "tavily_only":
            trace.primary = PrimaryRetrievalTrace(
                called=False, reason="SKIPPED_TAVILY_ONLY_MODE"
            )
            passages = await self._run_tavily(
                query, k, trace, reason="FORCE_TAVILY"
            )
            self.last_retrieval_trace = trace
            return passages

        # ── Primary adapter search ─────────────────────────────────
        primary_passages: List[Passage] = []
        primary_name = getattr(self._primary, "name", "primary")
        self.sources_attempted.append(primary_name)
        primary_start = time.time()
        primary_error = None

        try:
            search_fn = getattr(self._primary, "search")
            if source_mode:
                try:
                    primary_passages = await search_fn(query, k, source_mode=source_mode)
                except TypeError:
                    primary_passages = await search_fn(query, k)
            else:
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
            primary_error = str(e)[:200]

        primary_latency = int((time.time() - primary_start) * 1000)

        # Carry forward source tracking from primary adapter
        if hasattr(self._primary, "sources_attempted"):
            self.sources_attempted = list(getattr(self._primary, "sources_attempted", []))
        if hasattr(self._primary, "sources_succeeded"):
            self.sources_succeeded = list(getattr(self._primary, "sources_succeeded", []))
        if hasattr(self._primary, "sources_failed"):
            self.sources_failed = list(getattr(self._primary, "sources_failed", []))

        # ── Assess primary quality ─────────────────────────────────
        if primary_error:
            trace.primary = PrimaryRetrievalTrace(
                called=True,
                result_count=0,
                sufficient=False,
                reason="PRIMARY_ERROR",
                latency_ms=primary_latency,
                error=primary_error,
            )
        else:
            trace.primary = self._assess_primary_quality(primary_passages, query)
            trace.primary.latency_ms = primary_latency

        top_hint = max([float(getattr(p, "source_confidence_hint", 0.0) or 0.70) for p in primary_passages], default=0.0)
        from schemas.retrieval_trace import GateRelevanceAuditTrace
        trace.gate_relevance_audit = GateRelevanceAuditTrace(
            source_confidence_hint=round(top_hint, 4),
            gate_time_relevance_signal=trace.primary.top_relevance,
            final_bge_relevance_score=0.0,
            signals_agree=True,
        )

        # ── MODE: primary_only ─────────────────────────────────────
        if retrieval_mode == "primary_only":
            trace.tavily = TavilyRetrievalTrace(
                called=False, reason="SKIPPED_PRIMARY_ONLY_MODE"
            )
            trace.merged = MergedTrace(
                candidate_count=len(primary_passages),
                deduplicated_count=len(primary_passages),
            )
            trace.final = FinalTrace(reranked_count=len(primary_passages))
            self.last_retrieval_trace = trace
            return primary_passages[:k]

        # ── Quality gate decision ──────────────────────────────────
        if trace.primary.sufficient:
            trace.tavily = TavilyRetrievalTrace(
                called=False, reason="PRIMARY_EVIDENCE_SUFFICIENT"
            )
            trace.merged = MergedTrace(
                candidate_count=len(primary_passages),
                deduplicated_count=len(primary_passages),
            )
            trace.final = FinalTrace(reranked_count=len(primary_passages))
            self.last_retrieval_trace = trace
            return primary_passages[:k]

        # ── Tavily fallback ────────────────────────────────────────
        web_passages = await self._run_tavily(
            query, k, trace, reason=trace.primary.reason
        )

        # ── Merge + deduplicate ────────────────────────────────────
        from .web_retriever import _normalize_url
        seen_urls = set()
        merged: List[Passage] = []

        for p in primary_passages + web_passages:
            norm = _normalize_url(p.url) if p.url else p.title
            if norm not in seen_urls:
                seen_urls.add(norm)
                merged.append(p)

        trace.merged = MergedTrace(
            candidate_count=len(primary_passages) + len(web_passages),
            deduplicated_count=len(merged),
        )
        trace.final = FinalTrace(reranked_count=len(merged))

        self.last_retrieval_trace = trace
        return merged[:k]

    async def _run_tavily(
        self,
        query: str,
        k: int,
        trace: RetrievalTrace,
        reason: str,
    ) -> List[Passage]:
        """Execute Tavily search with trace recording."""
        active_key = self._tavily_key if self._tavily_key is not None else os.environ.get("TAVILY_API_KEY", "").strip()
        if not active_key:
            logger.info("Tavily API key not configured; skipping web fallback")
            trace.tavily = TavilyRetrievalTrace(
                called=False,
                reason="TAVILY_API_KEY_NOT_CONFIGURED",
            )
            return []

        self.sources_attempted.append("tavily_web")
        tavily_start = time.time()

        try:
            retriever = self._get_web_retriever()
            include_domains = None
            if self.name == "healthcare":
                include_domains = [
                    "who.int", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
                    "fda.gov", "cdc.gov", "nih.gov", "mayoclinic.org", "webmd.com",
                ]

            web_passages = await retriever.search(query, k=k, include_domains=include_domains)
            tavily_latency = int((time.time() - tavily_start) * 1000)

            usable = [p for p in web_passages if self._is_usable_passage(p)]

            if web_passages:
                self.sources_succeeded.append("tavily_web")
                logger.info(
                    "Tavily web fallback returned %d passages (%d usable)",
                    len(web_passages),
                    len(usable),
                )

            trace.tavily = TavilyRetrievalTrace(
                called=True,
                reason=reason,
                query=query[:200],
                result_count=len(web_passages),
                extracted_count=len(web_passages),
                usable_count=len(usable),
                latency_ms=tavily_latency,
                domains_requested=include_domains or [],
            )
            return web_passages

        except Exception as e:
            tavily_latency = int((time.time() - tavily_start) * 1000)
            logger.error("Tavily web fallback failed: %s", e)
            self.sources_failed.append("tavily_web")
            trace.tavily = TavilyRetrievalTrace(
                called=True,
                reason=reason,
                query=query[:200],
                latency_ms=tavily_latency,
                error=str(e)[:200],
            )
            return []
