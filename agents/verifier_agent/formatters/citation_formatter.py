from __future__ import annotations
from typing import List, Dict, Any
from schemas.models import Passage, EvidenceItem

class CitationFormatter:
    """Formats evidence passages into EvidenceItem schemas."""

    def format_evidence(self, passage: Passage, nli_result: Dict[str, Any], credibility: float) -> EvidenceItem:
        """
        Combine passage metadata with NLI scores and credibility into an EvidenceItem.
        """
        # Truncate snippet to 500 chars max
        snippet = passage.snippet
        if len(snippet) > 500:
            snippet = snippet[:497] + "..."
            
        url = passage.url if passage.url and passage.url.startswith("http") else ""
        
        # Format dates consistently
        pub_date = passage.publication_date or ""
        if len(pub_date) > 10:
            pub_date = pub_date[:10]  # rough YYYY-MM-DD truncation
            
        return EvidenceItem(
            source_name=passage.source_name,
            snippet=snippet,
            url=url,
            publication_date=pub_date,
            entailment_label=nli_result['label'],
            entailment_score=nli_result['entailment_score'],
            source_credibility=credibility
        )

    def format_all(self, passages: List[Passage], nli_results: List[Dict[str, Any]], domain: str, reliability_manager: Any) -> List[EvidenceItem]:
        """
        Batch format all passages.
        """
        formatted_items = []
        for passage, nli in zip(passages, nli_results):
            source_id = passage.source_name or "unknown"
            credibility = reliability_manager.get_credibility(domain, source_id)
            formatted_items.append(self.format_evidence(passage, nli, credibility))
            
        return formatted_items
