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

        # A single suspicious claim is decomposed into atomic sub-claims. When one
        # sub-claim is strongly grounded (e.g. 0.97 NLI entailment) it must NOT be
        # washed out by weaker siblings. We therefore aggregate the dominant signal
        # with a max-biased blend rather than a pure weighted mean, and we PROPAGATE
        # the evidence-derived confidence_score (previously dropped, which forced the
        # downstream calibrator into a low fallback and collapsed a well-grounded
        # answer to ~0.20 confidence).
        support_scores: list[float] = []
        contradict_scores: list[float] = []
        confidence_scores: list[float] = []
        total_trust = 0.0

        for report in sub_claim_reports:
            scores = report.get('scores', {})
            support = float(scores.get('support_score', 0.0) or 0.0)
            contradict = float(scores.get('contradiction_score', 0.0) or 0.0)
            trust = float(scores.get('trust_score', 0.0) or 0.0)
            conf = float(scores.get('confidence_score', 0.0) or 0.0)

            support_scores.append(support)
            contradict_scores.append(contradict)
            confidence_scores.append(conf)
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

        def _dominant(values: list[float]) -> float:
            # 70% strongest sub-claim + 30% mean: the best-grounded sub-claim leads,
            # but broad weak agreement still contributes. Clamped to [0, 1].
            if not values:
                return 0.0
            peak = max(values)
            mean = sum(values) / len(values)
            return max(0.0, min(1.0, 0.70 * peak + 0.30 * mean))

        avg_support = _dominant(support_scores)
        avg_contradict = _dominant(contradict_scores)
        avg_trust = total_trust / count if count > 0 else 0.0
        merged_confidence = _dominant(confidence_scores)

        if avg_support >= 0.30 and avg_contradict >= 0.30:
            overall_verdict = 'conflicted'
        elif avg_support >= 0.30 and avg_support > avg_contradict + 0.10:
            overall_verdict = 'verified'
        elif avg_contradict >= 0.30 and avg_contradict > avg_support + 0.10:
            overall_verdict = 'contradicted'
        elif avg_support > avg_contradict and avg_support >= 0.20:
            overall_verdict = 'verified'
        elif avg_contradict > avg_support and avg_contradict >= 0.20:
            overall_verdict = 'contradicted'
        else:
            overall_verdict = 'unverified'

        self.logger.debug("Merge decision: %s (weighted_support=%.2f, weighted_contradict=%.2f, avg_trust=%.2f)", overall_verdict, avg_support, avg_contradict, avg_trust)

        return {
            'verdict': overall_verdict,
            'scores': {
                'support_score': avg_support,
                'contradiction_score': avg_contradict,
                'trust_score': avg_trust,
                'confidence_score': merged_confidence,
            },
            'evidence_items': merged_evidence
        }

    def merge(self, sub_claim_reports: list[dict]) -> dict:
        """Alias for merge_results."""
        return self.merge_results(sub_claim_reports)
