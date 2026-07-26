from __future__ import annotations
from typing import Dict, Any, List
from schemas.models import EntailmentLabel, VerdictLabel, EvidenceItem

class ExplanationGenerator:
    """Generates human-readable explanations for the verification results."""

    def generate(self, claim_text: str, evidence_items: List[Any], verdict: str | VerdictLabel, scores: Dict[str, float], conflict_resolution: Dict[str, Any]) -> str:
        """
        Generate a human-readable explanation paragraph.

        Args:
            claim_text: The original claim text.
            evidence_items: List of evidence items (dict or EvidenceItem objects).
            verdict: The final verdict string/enum.
            scores: Dictionary with support_score, contradiction_score, trust_score.
            conflict_resolution: Dictionary with conflict resolution details.

        Returns:
            Formatted explanation string.
        """
        if not evidence_items:
            return "No supporting or contradicting evidence was found from any authoritative source."

        total_evidence = len(evidence_items)

        supports = []
        contradicts = []

        for e in evidence_items:
            label = getattr(e, 'entailment_label', None) or (e.get('entailment_label') if isinstance(e, dict) else None)
            if label in (EntailmentLabel.ENTAILMENT, 'entailment', 'supports'):
                supports.append(e)
            elif label in (EntailmentLabel.CONTRADICTION, 'contradiction', 'contradicts'):
                contradicts.append(e)

        # Helper to extract credibility
        def get_cred(item: Any) -> float:
            if isinstance(item, dict):
                return float(item.get('credibility_score', item.get('source_credibility', 0.5)))
            return float(getattr(item, 'credibility_score', 0.5))

        most_credible = max(evidence_items, key=get_cred)

        if isinstance(most_credible, dict):
            source_name = most_credible.get('source', most_credible.get('source_name', 'Unknown'))
            credibility = get_cred(most_credible)
            snippet = most_credible.get('snippet', '')
            pub_date = most_credible.get('publication_date', 'Unknown date')
        else:
            source_name = getattr(most_credible, 'source', 'Unknown')
            credibility = get_cred(most_credible)
            snippet = getattr(most_credible, 'snippet', '')
            pub_date = getattr(most_credible, 'publication_date', 'Unknown date')

        if len(snippet) > 150:
            snippet = snippet[:147] + "..."

        verdict_str = str(verdict.value if isinstance(verdict, VerdictLabel) else verdict)

        if verdict_str == VerdictLabel.LIKELY_HALLUCINATED.value:
            explanation = f"{len(contradicts)} out of {total_evidence} trusted sources contradict this claim. "
            explanation += f"The most credible source ({source_name}, credibility: {credibility:.2f}) states: \"{snippet}\" Published {pub_date}. "
            if not supports:
                explanation += "No supporting evidence was found from any authoritative database. "
        elif verdict_str == VerdictLabel.VERIFIED.value:
            explanation = f"{len(supports)} out of {total_evidence} trusted sources support this claim. "
            explanation += f"The most credible source ({source_name}, credibility: {credibility:.2f}) states: \"{snippet}\" Published {pub_date}. "
        else:
            explanation = f"There is mixed evidence regarding this claim. {len(supports)} sources support it while {len(contradicts)} contradict it. "
            explanation += f"A highly credible source ({source_name}, credibility: {credibility:.2f}) states: \"{snippet}\" "

        if conflict_resolution and conflict_resolution.get('resolution_type') == 'genuine_conflict':
            explanation += conflict_resolution.get('explanation', '')

        return explanation.strip()
