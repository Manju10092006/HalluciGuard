from __future__ import annotations
from typing import List, Dict, Any
from schemas.models import Passage, EvidenceItem, EntailmentLabel

class CitationFormatter:
    """Formats evidence passages into EvidenceItem schemas."""

    def __init__(self, evidence_scorer: Any = None) -> None:
        self.evidence_scorer = evidence_scorer

    def format_evidence(
        self,
        passage: Passage,
        nli_result: Dict[str, Any],
        credibility: float,
        claim: str = "",
    ) -> EvidenceItem:
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

        if self.evidence_scorer and claim:
            ev_class = self.evidence_scorer.classify_evidence(claim, passage, nli_result)
            if ev_class == "SUPPORTING":
                label = EntailmentLabel.ENTAILMENT
            elif ev_class == "CONTRADICTING":
                label = EntailmentLabel.CONTRADICTION
            else:
                label = EntailmentLabel.NEUTRAL
        else:
            label_raw = nli_result.get('label', EntailmentLabel.NEUTRAL)
            if not isinstance(label_raw, EntailmentLabel):
                if label_raw in ('entailment', 'supports'):
                    label = EntailmentLabel.ENTAILMENT
                    ev_class = "SUPPORTING"
                elif label_raw in ('contradiction', 'contradicts'):
                    label = EntailmentLabel.CONTRADICTION
                    ev_class = "CONTRADICTING"
                else:
                    label = EntailmentLabel.NEUTRAL
                    ev_class = "NEUTRAL"
            else:
                label = label_raw
                ev_class = label_raw.value.upper()

        source_name = getattr(passage, 'source', '') or getattr(passage, 'source_id', '') or "unknown"
        if nli_result.get('degraded', False):
            source_name += " [estimated]"

        entail_score = float(nli_result.get('entailment_score', 0.0))
        contra_score = float(nli_result.get('contradiction_score', 0.0))
        neutral_score = float(nli_result.get('neutral_score', 0.0))
        adapter_score = float(getattr(passage, 'source_confidence_hint', 0.0))
        bge_score = float(getattr(passage, 'relevance_score', 0.0))

        return EvidenceItem(
            title=getattr(passage, 'title', '') or "Reference Passage",
            source=source_name,
            snippet=snippet,
            url=url,
            publication_date=pub_date,
            entailment_label=label,
            entailment_score=entail_score,
            credibility_score=float(credibility),
            source_confidence_hint=adapter_score,
            adapter_score=adapter_score,
            bge_score=bge_score,
            nli_entailment=entail_score,
            nli_contradiction=contra_score,
            nli_neutral=neutral_score,
            classification=ev_class,
        )

    def format_all(
        self,
        passages: List[Passage],
        nli_results: List[Dict[str, Any]],
        domain: str,
        reliability_manager: Any,
        claim: str = "",
    ) -> List[EvidenceItem]:
        """
        Batch format all passages.
        """
        formatted_items = []
        for passage, nli in zip(passages, nli_results):
            source_id = getattr(passage, 'source_id', '') or passage.source or "unknown"
            credibility = reliability_manager.get_credibility(domain, source_id)
            formatted_items.append(self.format_evidence(passage, nli, credibility, claim=claim))

        return formatted_items
