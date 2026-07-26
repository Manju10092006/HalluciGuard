from __future__ import annotations
from typing import List, Dict, Any
from datetime import datetime

from schemas.models import Passage, VerdictLabel, EntailmentLabel
from .source_reliability import SourceReliabilityManager

class EvidenceScorer:
    """Scores evidence based on entailment, credibility, and agreement."""

    def __init__(self, credibility_config: Dict[str, Any]) -> None:
        self.reliability_manager = SourceReliabilityManager()
        # Mocking config load for brevity, typically would pass it to reliability_manager

    def score_evidence(self, claim: str, passages: List[Passage], nli_results: List[Dict[str, Any]], domain: str) -> Dict[str, Any]:
        """
        Score a set of evidence passages for a claim.
        
        Returns:
            Dict containing support_score, contradiction_score, trust_score, verdict.
        """
        if not passages or not nli_results:
            return {
                'support_score': 0.0,
                'contradiction_score': 0.0,
                'trust_score': 0.0,
                'verdict': VerdictLabel.MIXED_EVIDENCE
            }

        # Calculate agreement bonus
        supports_count = sum(1 for res in nli_results if res['label'] == EntailmentLabel.SUPPORTS)
        contradicts_count = sum(1 for res in nli_results if res['label'] == EntailmentLabel.CONTRADICTS)
        
        weighted_scores = []
        
        for passage, nli in zip(passages, nli_results):
            entailment_margin = nli['entailment_score'] - nli['contradiction_score']
            
            source_id = passage.source_name or "unknown"
            credibility_weight = self.reliability_manager.get_credibility(domain, source_id)
            
            pub_date_str = passage.publication_date or datetime.now().isoformat()
            recency_factor = self.reliability_manager.compute_recency_factor(pub_date_str)
            
            # Agreement bonus
            if nli['label'] == EntailmentLabel.SUPPORTS:
                agreement_bonus = 1.0 + (0.1 * supports_count)
            elif nli['label'] == EntailmentLabel.CONTRADICTS:
                agreement_bonus = 1.0 + (0.1 * contradicts_count)
            else:
                agreement_bonus = 1.0
                
            weighted_score = entailment_margin * credibility_weight * recency_factor * agreement_bonus
            weighted_scores.append(weighted_score)
            
        positive_scores = [s for s in weighted_scores if s > 0]
        negative_scores = [s for s in weighted_scores if s < 0]
        
        support_score = sum(positive_scores) / len(positive_scores) if positive_scores else 0.0
        # Average of negative scores (absolute value for logic)
        contradiction_score = abs(sum(negative_scores) / len(negative_scores)) if negative_scores else 0.0
        
        # Clamp scores between 0 and 1
        support_score = max(0.0, min(1.0, support_score))
        contradiction_score = max(0.0, min(1.0, contradiction_score))
        
        trust_score = support_score * (1.0 - contradiction_score)
        
        if trust_score > 0.6:
            verdict = VerdictLabel.VERIFIED
        elif trust_score < 0.3:
            verdict = VerdictLabel.LIKELY_HALLUCINATED
        else:
            verdict = VerdictLabel.MIXED_EVIDENCE
            
        return {
            'support_score': support_score,
            'contradiction_score': contradiction_score,
            'trust_score': trust_score,
            'verdict': verdict
        }
