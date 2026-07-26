from __future__ import annotations
from typing import Dict, Any, List
from schemas.models import EntailmentLabel

class ConflictResolver:
    """Resolves conflicts between contradictory evidence."""

    def resolve(self, evidence_items: List[Any]) -> Dict[str, Any]:
        """
        Analyze conflicting evidence and provide a resolution.

        Args:
            evidence_items: List of formatted evidence items (dictionaries or EvidenceItem objects).

        Returns:
            Dict containing resolution_type, explanation, confidence_adjustment.
        """
        if not evidence_items:
            return {
                'resolution_type': 'no_evidence',
                'explanation': 'No evidence available to resolve.',
                'confidence_adjustment': 0.0
            }

        support_weight = 0.0
        contradict_weight = 0.0

        for item in evidence_items:
            if isinstance(item, dict):
                label = item.get('entailment_label')
                credibility = item.get('credibility_score', item.get('source_credibility', 0.5))
            else:
                label = getattr(item, 'entailment_label', None)
                credibility = getattr(item, 'credibility_score', 0.5)

            if label in (EntailmentLabel.ENTAILMENT, 'entailment', 'supports'):
                support_weight += credibility
            elif label in (EntailmentLabel.CONTRADICTION, 'contradiction', 'contradicts'):
                contradict_weight += credibility

        if contradict_weight == 0 and support_weight > 0:
            return {
                'resolution_type': 'unanimous_support',
                'explanation': 'All evidence uniformly supports the claim.',
                'confidence_adjustment': 0.0
            }

        if support_weight == 0 and contradict_weight > 0:
            return {
                'resolution_type': 'unanimous_contradiction',
                'explanation': 'All evidence uniformly contradicts the claim.',
                'confidence_adjustment': 0.0
            }

        if support_weight > (2 * contradict_weight):
            return {
                'resolution_type': 'majority_support',
                'explanation': 'Supporting evidence significantly outweighs contradicting evidence.',
                'confidence_adjustment': 0.0
            }

        if contradict_weight > (2 * support_weight):
            return {
                'resolution_type': 'majority_contradiction',
                'explanation': 'Contradicting evidence significantly outweighs supporting evidence.',
                'confidence_adjustment': 0.0
            }

        return {
            'resolution_type': 'genuine_conflict',
            'explanation': 'There is a genuine conflict between credible sources.',
            'confidence_adjustment': -0.2
        }
