"""
HalluciGuard Verifier Agent — Tavily Web Evidence Retriever.

Provides real web evidence retrieval via Tavily Search + Extract.
This is a SUPPLEMENTARY retriever, not a replacement for domain adapters.

Integration pattern:
  Domain Adapter (Wikipedia/PubMed/arXiv) → primary evidence
  TavilyWebRetriever → fallback when primary returns insufficient evidence

Never uses Tavily's generated answer as evidence — only real page content.
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urldefrag, urlparse, urlencode, parse_qs, urlunparse

from schemas.models import Passage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trusted domain credibility scores
# ---------------------------------------------------------------------------
TRUSTED_DOMAINS: Dict[str, float] = {
    # Government / official
    "who.int": 0.98,
    "cdc.gov": 0.98,
    "nih.gov": 0.97,
    "ncbi.nlm.nih.gov": 0.97,
    "fda.gov": 0.97,
    "nvd.nist.gov": 1.0,
    "nist.gov": 0.98,
    "sec.gov": 0.97,
    "congress.gov": 0.97,
    "europa.eu": 0.96,
    "gov.uk": 0.96,
    "whitehouse.gov": 0.95,
    # Academic / research
    "arxiv.org": 0.95,
    "scholar.google.com": 0.90,
    "semanticscholar.org": 0.90,
    "pubmed.ncbi.nlm.nih.gov": 0.97,
    "nature.com": 0.95,
    "science.org": 0.95,
    "ieee.org": 0.94,
    "acm.org": 0.94,
    "jstor.org": 0.93,
    "springer.com": 0.93,
    "wiley.com": 0.92,
    # Encyclopedic
    "en.wikipedia.org": 0.82,
    "britannica.com": 0.90,
    # News (major)
    "reuters.com": 0.88,
    "apnews.com": 0.88,
    "bbc.com": 0.87,
    "bbc.co.uk": 0.87,
    "nytimes.com": 0.86,
    "theguardian.com": 0.85,
    "washingtonpost.com": 0.85,
    # Tech / standards
    "w3.org": 0.95,
    "ietf.org": 0.95,
    "python.org": 0.92,
    "docs.python.org": 0.92,
    "developer.mozilla.org": 0.92,
    "github.com": 0.80,
    "stackoverflow.com": 0.75,
}

# Tracking parameters to strip during URL normalization
_TRACKING_PARAMS = frozenset([
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "ref", "source", "spm",
])


def _normalize_url(url: str) -> str:
    """Normalize a URL for deduplication: remove fragments, tracking params, trailing slashes."""
    if not url or not url.strip():
        return ""
    url, _frag = urldefrag(url)
    parsed = urlparse(url)
    # Remove tracking query params
    qs = parse_qs(parsed.query, keep_blank_values=False)
    cleaned_qs = {k: v for k, v in qs.items() if k.lower() not in _TRACKING_PARAMS}
    clean_query = urlencode(cleaned_qs, doseq=True) if cleaned_qs else ""
    # Normalize path (strip trailing slash, lowercase scheme+host)
    path = parsed.path.rstrip("/") or "/"
    normalized = urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        path,
        parsed.params,
        clean_query,
        "",  # no fragment
    ))
    return normalized


def _domain_credibility(url: str) -> float:
    """Look up the credibility score for a URL's domain."""
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        # Try exact match first
        if hostname in TRUSTED_DOMAINS:
            return TRUSTED_DOMAINS[hostname]
        # Try parent domain (e.g. "news.bbc.co.uk" -> "bbc.co.uk")
        parts = hostname.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[i:])
            if parent in TRUSTED_DOMAINS:
                return TRUSTED_DOMAINS[parent]
    except Exception:
        pass
    return 0.60  # Unknown domain baseline


def _clean_markdown_text(text: str) -> str:
    """Clean markdown artifacts, images, links, navigation junk from extracted web text."""
    if not text:
        return ""
    # Remove markdown images: ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Convert markdown links [text](url) -> text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove table markers and horizontal rules
    text = re.sub(r'\|[-:\s|]+\|', ' ', text)
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'[-*_]{3,}', ' ', text)
    # Remove table of contents / contents blocks
    text = re.sub(r'##+\s*(Contents|Table of contents)\b.*?(?=\n\n|\n[A-Z]|\Z)', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove jump to content, search wikipedia, and navigation lines
    lines = []
    for line in text.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue
        if re.match(r'^(jump to|search|sign up|log in|navigation|table of contents|menu|skip to\b|contents\b|\* \d)', line_clean, re.IGNORECASE):
            continue
        lines.append(line_clean)
    cleaned = " ".join(lines)
    # Collapse whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _extract_best_passage(full_content: str, search_snippet: str, query: str, max_chars: int = 500) -> str:
    """
    Select the most informative passage from extracted page content or search snippet.
    """
    cleaned_full = _clean_markdown_text(full_content)
    cleaned_snippet = _clean_markdown_text(search_snippet)

    stopwords = {"is", "the", "a", "an", "in", "of", "to", "and", "or", "for", "on", "at", "by", "with", "that", "this", "was", "are", "it", "as", "be"}
    query_terms = set(re.findall(r'\w+', query.lower())) - stopwords

    # Split cleaned full text into sentence groupings / paragraphs
    sentences = re.split(r'(?<=[.!?])\s+', cleaned_full)
    paragraphs = []
    curr = []
    curr_len = 0
    for s in sentences:
        s_clean = s.strip()
        if not s_clean or len(s_clean) < 15 or s_clean.startswith(('+', '*', '#', '|')):
            continue
        curr.append(s_clean)
        curr_len += len(s_clean) + 1
        if curr_len >= 200:
            paragraphs.append(" ".join(curr))
            curr = []
            curr_len = 0
    if curr:
        paragraphs.append(" ".join(curr))

    best_para = ""
    best_score = -1
    for p in paragraphs[:25]:
        if p.startswith(('#', '*', '+', '|')) or len(p) < 40:
            continue
        p_lower = p.lower()
        match_count = sum(1 for term in query_terms if term in p_lower)
        if match_count > best_score:
            best_score = match_count
            best_para = p

    # If search_snippet is informative and has high term match, prefer search_snippet
    if cleaned_snippet and len(cleaned_snippet) >= 30:
        snippet_score = sum(1 for term in query_terms if term in cleaned_snippet.lower())
        if snippet_score >= best_score:
            return cleaned_snippet[:max_chars].strip()

    if best_para:
        return best_para[:max_chars].strip()

    return (cleaned_snippet or cleaned_full)[:max_chars].strip()


class TavilyWebRetriever:
    """
    Web evidence retriever using Tavily Search + Extract APIs.

    Usage:
        retriever = TavilyWebRetriever()
        passages = await retriever.search("Paris is the capital of France", k=5)
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("TAVILY_API_KEY", "")
        self._client = None  # Lazy-init

    def _get_client(self):
        """Lazy-initialize Tavily client."""
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "TAVILY_API_KEY not set. Add it to .env or pass api_key to TavilyWebRetriever."
                )
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self._api_key)
            except ImportError:
                raise RuntimeError(
                    "tavily-python not installed. Run: pip install tavily-python"
                )
        return self._client

    async def search(
        self,
        query: str,
        k: int = 5,
        search_depth: str = "advanced",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Passage]:
        """
        Search the web via Tavily and return normalized Passage objects.

        Steps:
        1. Tavily Search → candidate URLs + snippets
        2. Tavily Extract → full page content for top URLs
        3. Normalize into Passage objects with real provenance
        """
        if not query or not query.strip():
            return []

        client = self._get_client()
        passages: List[Passage] = []
        seen_urls: Set[str] = set()
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # ── Step 1: Tavily Search ──────────────────────────────────
        search_results = []
        try:
            logger.info("Tavily search: query=%r, depth=%s, k=%d", query, search_depth, k)
            response = client.search(
                query=query,
                search_depth=search_depth,
                max_results=min(k * 2, 10),  # Over-fetch for dedup headroom
                include_answer=False,  # NEVER use generated answers as evidence
                include_domains=include_domains or [],
                exclude_domains=exclude_domains or [],
            )
            search_results = response.get("results", [])
            logger.info("Tavily search returned %d results", len(search_results))
        except Exception as e:
            logger.error("Tavily search failed: %s", e)
            return []

        if not search_results:
            return []

        # ── Step 2: Collect URLs for extraction ─────────────────────
        candidate_urls: List[str] = []
        url_metadata: Dict[str, Dict[str, Any]] = {}  # normalized_url -> search metadata

        for result in search_results:
            raw_url = result.get("url", "")
            if not raw_url:
                continue
            norm_url = _normalize_url(raw_url)
            if norm_url in seen_urls:
                continue
            seen_urls.add(norm_url)
            candidate_urls.append(raw_url)
            url_metadata[norm_url] = {
                "title": result.get("title", ""),
                "search_snippet": result.get("content", ""),
                "score": result.get("score", 0.0),
                "raw_url": raw_url,
            }

        # ── Step 3: Tavily Extract (page content) ──────────────────
        extracted_content: Dict[str, str] = {}
        extract_urls = candidate_urls[:min(k + 2, 7)]  # Limit extraction calls

        if extract_urls:
            try:
                logger.info("Tavily extract: %d URLs", len(extract_urls))
                extract_response = client.extract(urls=extract_urls)
                for item in extract_response.get("results", []):
                    url = item.get("url", "")
                    content = item.get("raw_content", "") or item.get("text", "")
                    if url and content:
                        norm = _normalize_url(url)
                        extracted_content[norm] = content
                logger.info("Tavily extract returned content for %d URLs", len(extracted_content))
            except Exception as e:
                logger.warning("Tavily extract failed (using search snippets): %s", e)

        # ── Step 4: Build Passage objects ──────────────────────────
        for norm_url, meta in url_metadata.items():
            raw_url = meta["raw_url"]
            title = meta["title"] or "Web Source"
            search_snippet = meta["search_snippet"]
            full_content = extracted_content.get(norm_url, "")

            snippet = _extract_best_passage(
                full_content=full_content,
                search_snippet=search_snippet,
                query=query,
                max_chars=600,
            )
            if not snippet or len(snippet) < 5:
                continue

            # Determine source identifier from domain
            try:
                domain = urlparse(raw_url).hostname or "web"
            except Exception:
                domain = "web"

            credibility = _domain_credibility(raw_url)

            passage = Passage(
                title=f"Web: {title[:100]}",
                source=f"tavily:{domain}",
                url=raw_url,
                publication_date=timestamp,
                snippet=snippet,
                source_id=f"tavily_{_normalize_url(raw_url)[:80]}",
                relevance_score=min(float(meta.get("score", 0.5)), 1.0),
            )
            passages.append(passage)

            if len(passages) >= k:
                break

        logger.info(
            "TavilyWebRetriever produced %d passages for query=%r",
            len(passages),
            query[:60],
        )
        return passages

    def credibility_of(self, url_or_source_id: str) -> float:
        """Return credibility for a URL or source_id."""
        return _domain_credibility(url_or_source_id)

    @property
    def is_available(self) -> bool:
        """Check if Tavily API key is configured."""
        return bool(self._api_key)
