from __future__ import annotations
from typing import List

from schemas.models import Passage

class DuplicateRemover:
    """Removes duplicate or highly overlapping passages."""

    def __init__(self, overlap_threshold: float = 0.85) -> None:
        self.overlap_threshold = overlap_threshold

    def remove_duplicates(self, passages: List[Passage]) -> List[Passage]:
        """
        Deduplicate passages based on URLs, content overlap, and title/source.
        
        Args:
            passages: The list of passages to deduplicate.
            
        Returns:
            Deduplicated list of passages.
        """
        unique_passages: List[Passage] = []
        seen_urls = set()
        seen_title_sources = set()
        
        # Sort by relevance score to keep the best ones first
        sorted_passages = sorted(
            passages, 
            key=lambda p: p.relevance_score if p.relevance_score is not None else 0.0, 
            reverse=True
        )
        
        from adapters.web_retriever import _normalize_url

        for p in sorted_passages:
            # Strategy 1: Canonical URL match
            norm_url = _normalize_url(p.url) if p.url else None
            if norm_url:
                if norm_url in seen_urls:
                    continue
                
            # Strategy 2: Same title + same source (for URL-less passages)
            source_val = p.source
            title_source_key = None
            if not norm_url and p.title and source_val:
                title_source_key = f"{p.title.lower()}::{source_val.lower()}"
                if title_source_key in seen_title_sources:
                    continue
                
            # Strategy 3: Token overlap >85% between snippets
            is_overlap = False
            for up in unique_passages:
                overlap = self._compute_token_overlap(p.snippet, up.snippet)
                if overlap > self.overlap_threshold:
                    is_overlap = True
                    break
                    
            if not is_overlap:
                unique_passages.append(p)
                if norm_url:
                    seen_urls.add(norm_url)
                elif p.url:
                    seen_urls.add(p.url)
                if title_source_key:
                    seen_title_sources.add(title_source_key)
                
        return unique_passages

    def _compute_token_overlap(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity on word tokens."""
        import re
        set1 = set(re.findall(r"\b\w+\b", text1.lower())) if text1 else set()
        set2 = set(re.findall(r"\b\w+\b", text2.lower())) if text2 else set()
        if not set1 or not set2:
            return 0.0
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)
