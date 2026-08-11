from __future__ import annotations

import logging

class ClaimMerger:
    """Merges evidence and verdicts from sub-claims."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def merge_results(self, sub_claim_reports: list[dict]) -> dict:
        """
        Combine evidence and verdicts from multiple sub-claim verifications.
        
        Args:
            sub_claim_reports: List of dictionaries containing sub-claim reports.
            
        Returns:
            Merged report dictionary.
        """
        if not sub_claim_reports:
            return {}

        merged_evidence = []
        seen_evidence = set()
        
        total_support = 0.0
        total_contradict = 0.0
        total_trust = 0.0
        total_weight = 0.0
        
        for report in sub_claim_reports:
            scores = report.get('scores', {})
            support = scores.get('support_score', 0.0)
            contradict = scores.get('contradiction_score', 0.0)
            trust = scores.get('trust_score', 0.0)
            
            weight = trust + 0.1  # small base weight
            total_weight += weight
            
            total_support += support * weight
            total_contradict += contradict * weight
            total_trust += trust
            
            for evidence in report.get('evidence_items', []):
                if isinstance(evidence, dict):
                    snippet = evidence.get('snippet', '')
                else:
                    snippet = getattr(evidence, 'snippet', '')

                if snippet and snippet not in seen_evidence:
                    seen_evidence.add(snippet)
                    merged_evidence.append(evidence)

        count = len(sub_claim_reports)
        avg_support = total_support / total_weight if total_weight > 0 else 0.0
        avg_contradict = total_contradict / total_weight if total_weight > 0 else 0.0
        avg_trust = total_trust / count if count > 0 else 0.0
        
        if avg_trust > 0.5 or avg_support > 0.6:
            overall_verdict = 'verified'
        elif avg_contradict > 0.5:
            overall_verdict = 'likely_hallucinated'
        elif avg_support == 0.0 and avg_contradict == 0.0:
            overall_verdict = 'insufficient_evidence'
        else:
            overall_verdict = 'mixed_evidence'

        self.logger.debug("Merge decision: %s (weighted_support=%.2f, weighted_contradict=%.2f, avg_trust=%.2f)", overall_verdict, avg_support, avg_contradict, avg_trust)

        return {
            'verdict': overall_verdict,
            'scores': {
                'support_score': avg_support,
                'contradiction_score': avg_contradict,
                'trust_score': avg_trust
            },
            'evidence_items': merged_evidence
        }

    def merge(self, sub_claim_reports: list[dict]) -> dict:
        """Alias for merge_results."""
        return self.merge_results(sub_claim_reports)
