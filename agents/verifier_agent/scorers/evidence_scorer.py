from __future__ import annotations
from typing import List, Dict, Any
from datetime import datetime

from schemas.models import Passage, VerdictLabel, EntailmentLabel
from .source_reliability import SourceReliabilityManager

class EvidenceScorer:
    """Scores evidence based on entailment, credibility, and agreement."""

    def __init__(self, credibility_config: Dict[str, Any] = None) -> None:
        self.reliability_manager = SourceReliabilityManager()

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
                'verdict': VerdictLabel.INSUFFICIENT_EVIDENCE
            }

        # Calculate agreement bonus
        supports_count = sum(1 for res in nli_results if res.get('label') == EntailmentLabel.ENTAILMENT)
        contradicts_count = sum(1 for res in nli_results if res.get('label') == EntailmentLabel.CONTRADICTION)
        
        weighted_scores = []
        
        for passage, nli in zip(passages, nli_results):
            entailment_score = float(nli.get('entailment_score', 0.5))
            contradiction_score_val = float(nli.get('contradiction_score', 0.0))
            entailment_margin = entailment_score - contradiction_score_val
            
            source_id = getattr(passage, 'source_id', '') or passage.source or "unknown"
            credibility_weight = self.reliability_manager.get_credibility(domain, source_id)
            
            pub_date_str = passage.publication_date or datetime.now().isoformat()
            recency_factor = self.reliability_manager.compute_recency_factor(pub_date_str)
            
            # Agreement bonus
            if nli.get('label') == EntailmentLabel.ENTAILMENT:
                agreement_bonus = 1.0 + (0.1 * supports_count)
            elif nli.get('label') == EntailmentLabel.CONTRADICTION:
                agreement_bonus = 1.0 + (0.1 * contradicts_count)
            else:
                agreement_bonus = 1.0

            weighted_score = entailment_margin * credibility_weight * recency_factor * agreement_bonus
            weighted_scores.append(weighted_score)
            
        positive_scores = [s for s in weighted_scores if s > 0]
        negative_scores = [s for s in weighted_scores if s < 0]
        
        support_score = sum(positive_scores) / len(positive_scores) if positive_scores else 0.0
        contradiction_score = abs(sum(negative_scores) / len(negative_scores)) if negative_scores else 0.0
        
        # Clamp scores between 0 and 1
        support_score = max(0.0, min(1.0, support_score))
        contradiction_score = max(0.0, min(1.0, contradiction_score))
        
        trust_score = support_score * (1.0 - contradiction_score)
        
        if support_score == 0.0 and contradiction_score == 0.0:
            verdict = VerdictLabel.INSUFFICIENT_EVIDENCE
        elif trust_score > 0.5 or support_score > 0.6:
            verdict = VerdictLabel.VERIFIED
        elif contradiction_score > 0.5:
            verdict = VerdictLabel.LIKELY_HALLUCINATED
        else:
            verdict = VerdictLabel.MIXED_EVIDENCE
            
        return {
            'support_score': support_score,
            'contradiction_score': contradiction_score,
            'trust_score': trust_score,
            'verdict': verdict
        }
