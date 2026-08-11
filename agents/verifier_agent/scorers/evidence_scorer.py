from __future__ import annotations

import math
from typing import Any, Dict, List

from schemas.models import Passage, VerdictLabel, EntailmentLabel
from .source_reliability import SourceReliabilityManager


class EvidenceScorer:
    """Combines NLI strength, retrieval relevance, source trust, and recency."""

    def __init__(self, credibility_config: Dict[str, Any] | None = None) -> None:
        self.reliability_manager = SourceReliabilityManager()

    @staticmethod
    def _bounded(value: Any, default: float = 0.0) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _relevance_weight(raw_score: Any) -> float:
        """Map arbitrary cross-encoder scores into a stable 0..1 weight."""
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            return 0.5

        # Cross-encoder scores are logits, not probabilities. Sigmoid keeps
        # negative/positive model outputs meaningful without hard thresholds.
        if score < 0.0 or score > 1.0:
            score = 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, score))))
        return max(0.05, min(1.0, score))

    def score_evidence(
        self,
        claim: str,
        passages: List[Passage],
        nli_results: List[Dict[str, Any]],
        domain: str,
    ) -> Dict[str, Any]:
        if not passages or not nli_results:
            return {
                "support_score": 0.0,
                "contradiction_score": 0.0,
                "trust_score": 0.0,
                "verdict": VerdictLabel.INSUFFICIENT_EVIDENCE,
            }

        support_weights: List[float] = []
        contradiction_weights: List[float] = []
        supporting_sources: set[str] = set()
        contradicting_sources: set[str] = set()

        for passage, nli in zip(passages, nli_results):
            entailment = self._bounded(nli.get("entailment_score"))
            contradiction = self._bounded(nli.get("contradiction_score"))
            label = nli.get("label")

            source_id = (
                getattr(passage, "source_id", "")
                or getattr(passage, "source", "")
                or "unknown"
            )
            source_id = str(source_id)

            credibility = self._bounded(
                self.reliability_manager.get_credibility(domain, source_id),
                0.8,
            )
            recency = self._bounded(
                self.reliability_manager.compute_recency_factor(
                    getattr(passage, "publication_date", "")
                ),
                1.0,
            )
            relevance = self._relevance_weight(
                getattr(passage, "relevance_score", 0.5)
            )

            base_weight = credibility * recency * relevance

            if label == EntailmentLabel.ENTAILMENT or entailment > contradiction:
                support_weights.append(entailment * base_weight)
                supporting_sources.add(source_id)
            elif label == EntailmentLabel.CONTRADICTION or contradiction > entailment:
                contradiction_weights.append(contradiction * base_weight)
                contradicting_sources.add(source_id)

        # Use the strongest evidence as the main signal, then add a modest
        # independent-source bonus. This avoids averaging away a strong source.
        support_max = max(support_weights, default=0.0)
        contradiction_max = max(contradiction_weights, default=0.0)
        support_avg = sum(support_weights) / len(support_weights) if support_weights else 0.0
        contradiction_avg = (
            sum(contradiction_weights) / len(contradiction_weights)
            if contradiction_weights else 0.0
        )

        support_bonus = min(0.15, 0.05 * max(0, len(supporting_sources) - 1))
        contradiction_bonus = min(0.15, 0.05 * max(0, len(contradicting_sources) - 1))

        support_score = min(
            1.0,
            0.70 * support_max + 0.30 * support_avg + support_bonus,
        )
        contradiction_score = min(
            1.0,
            0.70 * contradiction_max
            + 0.30 * contradiction_avg
            + contradiction_bonus,
        )

        # Contradiction should reduce trust strongly, while independent support
        # raises trust without allowing raw evidence count to dominate.
        trust_score = max(
            0.0,
            min(
                1.0,
                support_score
                * (1.0 - 0.90 * contradiction_score)
                + 0.05 * min(1.0, len(supporting_sources) / 2.0),
            ),
        )

        if support_score < 0.20 and contradiction_score < 0.20:
            verdict = VerdictLabel.INSUFFICIENT_EVIDENCE
        elif contradiction_score >= 0.55 and contradiction_score > support_score + 0.10:
            verdict = VerdictLabel.LIKELY_HALLUCINATED
        elif support_score >= 0.55 and support_score > contradiction_score + 0.10:
            verdict = VerdictLabel.VERIFIED
        elif support_score >= 0.30 and contradiction_score >= 0.30:
            verdict = VerdictLabel.MIXED_EVIDENCE
        elif support_score > contradiction_score and support_score >= 0.30:
            verdict = VerdictLabel.VERIFIED
        elif contradiction_score > support_score and contradiction_score >= 0.30:
            verdict = VerdictLabel.LIKELY_HALLUCINATED
        else:
            # Evidence scores are too low to make a confident decision
            verdict = VerdictLabel.INSUFFICIENT_EVIDENCE

        return {
            "support_score": round(support_score, 4),
            "contradiction_score": round(contradiction_score, 4),
            "trust_score": round(trust_score, 4),
            "verdict": verdict,
        }
