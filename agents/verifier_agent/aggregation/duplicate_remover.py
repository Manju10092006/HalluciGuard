from __future__ import annotations
from typing import List

from schemas.models import Passage

class DuplicateRemover:
    """Remove repeated excerpts without discarding distinct factual evidence."""

    def __init__(self, overlap_threshold: float = 0.85) -> None:
        self.overlap_threshold = overlap_threshold

    def remove_duplicates(self, passages: List[Passage]) -> List[Passage]:
        """
        Deduplicate the same normalized excerpt from the same document.

        A shared document URL is not proof of duplicate content: sections of an
        article can establish different facts. Likewise, high token overlap can
        hide a changed date, quantity, negation, or word order. Preserve these
        passages and let the downstream ranker and verifier evaluate them.

        Only whitespace is normalized in content; punctuation, case, values,
        and word order remain significant. Independent sources remain separate
        even when they repeat identical text, preserving their provenance.
        
        Args:
            passages: The list of passages to deduplicate.
            
        Returns:
            Deduplicated list of passages.
        """
        unique_passages: List[Passage] = []
        seen_excerpts = set()
        
        # Sort by relevance score to keep the best ones first
        sorted_passages = sorted(
            passages, 
            key=lambda p: p.relevance_score if p.relevance_score is not None else 0.0, 
            reverse=True
        )
        
        from adapters.web_retriever import _normalize_url

        for p in sorted_passages:
            content = " ".join((p.snippet or "").split())
            norm_url = _normalize_url(p.url) if p.url else ""
            if norm_url:
                document_key = ("url", norm_url)
            elif p.source_id or (p.title and p.source):
                document_key = ("record", p.source, p.source_id, p.title)
            else:
                # No reliable document identity or substantive content means
                # there is no basis for declaring another passage a duplicate.
                unique_passages.append(p)
                continue

            if not content:
                unique_passages.append(p)
                continue
            excerpt_key = (document_key, content)
            if excerpt_key in seen_excerpts:
                continue
            seen_excerpts.add(excerpt_key)
            unique_passages.append(p)

        return unique_passages

    def _compute_token_overlap(self, text1: str, text2: str) -> float:
        """Compute diagnostic overlap, not a safe factual-duplicate decision."""
        import re
        set1 = set(re.findall(r"\b\w+\b", text1.lower())) if text1 else set()
        set2 = set(re.findall(r"\b\w+\b", text2.lower())) if text2 else set()
        if not set1 or not set2:
            return 0.0
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)
