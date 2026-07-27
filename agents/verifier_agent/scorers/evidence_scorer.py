from __future__ import annotations
from typing import List, Dict, Any
from datetime import datetime

from schemas.models import Passage, VerdictLabel, EntailmentLabel
from .source_reliability import SourceReliabilityManager


class EvidenceScorer:
    """Scores evidence based on entailment, credibility, recency, and agreement."""

    def __init__(self, credibility_config: Dict[str, Any] = None) -> None:
        self.reliability_manager = SourceReliabilityManager()

    def score_evidence(
        self,
        claim: str,
        passages: List[Passage],
        nli_results: List[Dict[str, Any]],
        domain: str,
    ) -> Dict[str, Any]:
        """
        Score a set of evidence passages for a claim.

        Returns:
            Dict containing support_score, contradiction_score, trust_score, verdict.
        """
        if not passages or not nli_results:
            return {
                "support_score": 0.0,
                "contradiction_score": 0.0,
                "trust_score": 0.0,
                "verdict": VerdictLabel.INSUFFICIENT_EVIDENCE,
            }

        # Calculate agreement counts
        supports_count = sum(
            1 for res in nli_results if res.get("label") == EntailmentLabel.ENTAILMENT
        )
        contradicts_count = sum(
            1 for res in nli_results if res.get("label") == EntailmentLabel.CONTRADICTION
        )

        support_weights = []
        contradiction_weights = []

        for passage, nli in zip(passages, nli_results):
            entailment_score = float(nli.get("entailment_score", 0.0))
            contradiction_score_val = float(nli.get("contradiction_score", 0.0))
            label = nli.get("label")

            source_id = (
                getattr(passage, "source_id", "")
                or getattr(passage, "source", "")
                or "unknown"
            )
            credibility_weight = self.reliability_manager.get_credibility(
                domain, source_id
            )

            pub_date_str = passage.publication_date or datetime.now().isoformat()
            recency_factor = self.reliability_manager.compute_recency_factor(
                pub_date_str
            )

            # Passage relevance score contribution
            relevance = (
                getattr(passage, "relevance_score", 0.9)
                if getattr(passage, "relevance_score", 0.0) > 0
                else 0.9
            )

            if label == EntailmentLabel.ENTAILMENT or entailment_score > contradiction_score_val:
                agreement_bonus = 1.0 + (0.05 * supports_count)
                weight = (
                    entailment_score
                    * credibility_weight
                    * recency_factor
                    * relevance
                    * agreement_bonus
                )
                support_weights.append(weight)

            elif label == EntailmentLabel.CONTRADICTION or contradiction_score_val > entailment_score:
                agreement_bonus = 1.0 + (0.05 * contradicts_count)
                weight = (
                    contradiction_score_val
                    * credibility_weight
                    * recency_factor
                    * relevance
                    * agreement_bonus
                )
                contradiction_weights.append(weight)

        support_score = (
            sum(support_weights) / len(support_weights) if support_weights else 0.0
        )
        contradiction_score = (
            sum(contradiction_weights) / len(contradiction_weights)
            if contradiction_weights
            else 0.0
        )

        # Clamp between 0.0 and 1.0
        support_score = round(max(0.0, min(1.0, support_score)), 4)
        contradiction_score = round(max(0.0, min(1.0, contradiction_score)), 4)

        # Calculate trust score
        trust_score = round(support_score * (1.0 - (0.8 * contradiction_score)), 4)

        # Determine Verdict
        if support_score == 0.0 and contradiction_score == 0.0:
            verdict = VerdictLabel.INSUFFICIENT_EVIDENCE
        elif support_score > 0.4 and contradiction_score < 0.25:
            verdict = VerdictLabel.VERIFIED
        elif contradiction_score > 0.4 and support_score < 0.25:
            verdict = VerdictLabel.LIKELY_HALLUCINATED
        elif support_score > 0.2 and contradiction_score > 0.2:
            verdict = VerdictLabel.MIXED_EVIDENCE
        elif support_score > 0.25:
            verdict = VerdictLabel.VERIFIED
        elif contradiction_score > 0.25:
            verdict = VerdictLabel.LIKELY_HALLUCINATED
        else:
            verdict = VerdictLabel.INSUFFICIENT_EVIDENCE

        return {
            "support_score": support_score,
            "contradiction_score": contradiction_score,
            "trust_score": trust_score,
            "verdict": verdict,
        }
