from __future__ import annotations
from typing import List, Dict, Any
from schemas.models import Passage, EvidenceItem, EntailmentLabel

class CitationFormatter:
    """Formats evidence passages into EvidenceItem schemas."""

    def format_evidence(self, passage: Passage, nli_result: Dict[str, Any], credibility: float) -> EvidenceItem:
        """
        Combine passage metadata with NLI scores and credibility into an EvidenceItem.
        """
        snippet = passage.snippet
        if len(snippet) > 500:
            snippet = snippet[:497] + "..."

        url = passage.url if passage.url and passage.url.startswith("http") else ""

        pub_date = passage.publication_date or ""
        if len(pub_date) > 10:
            pub_date = pub_date[:10]

        label = nli_result.get('label', EntailmentLabel.NEUTRAL)
        if not isinstance(label, EntailmentLabel):
            if label in ('entailment', 'supports'):
                label = EntailmentLabel.ENTAILMENT
            elif label in ('contradiction', 'contradicts'):
                label = EntailmentLabel.CONTRADICTION
            else:
                label = EntailmentLabel.NEUTRAL

        return EvidenceItem(
            title=getattr(passage, 'title', '') or "Reference Passage",
            source=getattr(passage, 'source', '') or getattr(passage, 'source_id', '') or "unknown",
            snippet=snippet,
            url=url,
            publication_date=pub_date,
            entailment_label=label,
            entailment_score=float(nli_result.get('entailment_score', 0.5)),
            credibility_score=float(credibility)
        )

    def format_all(self, passages: List[Passage], nli_results: List[Dict[str, Any]], domain: str, reliability_manager: Any) -> List[EvidenceItem]:
        """
        Batch format all passages.
        """
        formatted_items = []
        for passage, nli in zip(passages, nli_results):
            source_id = getattr(passage, 'source_id', '') or passage.source or "unknown"
            credibility = reliability_manager.get_credibility(domain, source_id)
            formatted_items.append(self.format_evidence(passage, nli, credibility))

        return formatted_items
