from __future__ import annotations
from typing import Dict, Any, List
from schemas.models import EntailmentLabel, VerdictLabel

class ExplanationGenerator:
    """Generates human-readable explanations for the verification results."""

    def generate(self, claim_text: str, evidence_items: List[Dict[str, Any]], verdict: str, scores: Dict[str, float], conflict_resolution: Dict[str, Any]) -> str:
        """
        Generate a human-readable explanation paragraph.
        
        Args:
            claim_text: The original claim text.
            evidence_items: List of evidence items (dictionaries).
            verdict: The final verdict string.
            scores: Dictionary with support_score, contradiction_score, trust_score.
            conflict_resolution: Dictionary with conflict resolution details.
            
        Returns:
            Formatted explanation string.
        """
        if not evidence_items:
            return "No supporting or contradicting evidence was found from any authoritative source."

        total_evidence = len(evidence_items)
        supports = [e for e in evidence_items if e.get('entailment_label') == EntailmentLabel.SUPPORTS]
        contradicts = [e for e in evidence_items if e.get('entailment_label') == EntailmentLabel.CONTRADICTS]
        
        # Find the most credible source among all evidence
        most_credible = max(evidence_items, key=lambda e: e.get('source_credibility', 0.0))
        
        source_name = most_credible.get('source_name', 'Unknown')
        credibility = most_credible.get('source_credibility', 0.0)
        snippet = most_credible.get('snippet', '')
        pub_date = most_credible.get('publication_date', 'Unknown date')

        if verdict == VerdictLabel.LIKELY_HALLUCINATED:
            explanation = f"{len(contradicts)} out of {total_evidence} trusted sources contradict this claim. "
            explanation += f"The most credible source ({source_name}, credibility: {credibility:.2f}) states: \"{snippet}\" Published {pub_date}. "
            if not supports:
                explanation += "No supporting evidence was found from any authoritative database. "
        elif verdict == VerdictLabel.VERIFIED:
            explanation = f"{len(supports)} out of {total_evidence} trusted sources support this claim. "
            explanation += f"The most credible source ({source_name}, credibility: {credibility:.2f}) states: \"{snippet}\" Published {pub_date}. "
        else:
            explanation = f"There is mixed evidence regarding this claim. {len(supports)} sources support it while {len(contradicts)} contradict it. "
            explanation += f"A highly credible source ({source_name}, credibility: {credibility:.2f}) states: \"{snippet}\" "
            
        if conflict_resolution and conflict_resolution.get('resolution_type') == 'genuine_conflict':
            explanation += conflict_resolution.get('explanation', '')

        return explanation.strip()
