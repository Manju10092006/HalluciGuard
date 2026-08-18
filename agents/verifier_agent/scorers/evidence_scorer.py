from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from schemas.models import Passage, VerdictLabel, EntailmentLabel
from .source_reliability import SourceReliabilityManager


class EvidenceScorer:
    """
    Combine NLI strength, retrieval quality, source trust, and recency into
    mathematically grounded evidence scores, classifications, and calibrated confidence.
    """

    MIN_NLI_SIGNAL = 0.40

    def __init__(self, credibility_config: Dict[str, Any] | None = None) -> None:
        self.reliability_manager = SourceReliabilityManager()

    @staticmethod
    def _bounded(value: Any, default: float = 0.0) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def relevance_weight(raw_score: Any) -> float:
        """Map cross-encoder scores into a calibrated evidence weight in [0.20, 1.0]."""
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            return 0.50

        if score < 0.0:
            score = 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, score))))

        score = max(0.0, min(1.0, score))
        # Cross-encoder scores for contradiction premises (which discuss the subject but omit the false entity)
        # range around 0.02 - 0.20. Calibrate with power transform so candidate passages provide meaningful weight.
        calibrated = math.pow(score, 0.25) if score > 0.0 else 0.0
        return max(0.20, min(1.0, calibrated))

    @staticmethod
    def _is_non_assertive_claim_context(claim: str, passage_text: str) -> bool:
        """Reject passages that discuss a claim as a belief, fiction, proverb, myth, etymology, or phrase."""
        import re

        stopwords = {
            "is", "the", "a", "an", "in", "of", "to", "and", "or", "for", "on",
            "at", "by", "with", "that", "this", "was", "are", "it", "as", "be",
        }
        claim_terms = set(re.findall(r"[a-z0-9]+", claim.lower())) - stopwords
        text_lower = passage_text.lower()
        snippet_terms = set(re.findall(r"[a-z0-9]+", text_lower))

        overlap = claim_terms.intersection(snippet_terms)
        if len(overlap) < max(1, min(2, len(claim_terms))):
            return False

        context_terms = {
            "allegation", "alleged", "belief", "fiction", "fictional",
            "folklore", "hoax", "legend", "myth", "mythical", "purported",
            "fable", "proverb", "idiom", "saying", "trope", "metaphor",
            "fanciful", "absurdity", "nursery", "rhyme", "untrue", "false",
            "disproven", "debunked", "misconception", "expression", "phrase",
            "idiomatic", "children", "folktale", "origin", "origins",
            "etymology", "meaning", "riddle", "joke", "metaphorical",
        }
        if bool(context_terms.intersection(snippet_terms)):
            return True

        phrase_patterns = [
            "why do we say", "where did the phrase", "meaning of", "story behind",
            "did people really believe", "origin of the phrase",
        ]
        if any(pat in text_lower for pat in phrase_patterns):
            return True

        return False

    def classify_passage(
        self,
        claim: str,
        passage: Passage,
        nli: Dict[str, Any],
    ) -> str:
        """
        Classify a single passage into exactly ONE mutually exclusive category:
        'SUPPORTING', 'CONTRADICTING', 'NEUTRAL', 'IRRELEVANT'.
        """
        rel_score = float(getattr(passage, "relevance_score", 0.5) or 0.0)
        if rel_score < 0.001:
            return "IRRELEVANT"

        # Guard against myths / proverbs / common misconceptions matching
        # assertively if the passage qualifies them as untrue, figurative, or an idiom.
        full_text = f"{getattr(passage, 'title', '')} {passage.snippet}"
        if self._is_non_assertive_claim_context(claim, full_text):
            return "NEUTRAL"

        entailment = self._bounded(nli.get("entailment_score"))
        contradiction = self._bounded(nli.get("contradiction_score"))
        neutral = self._bounded(nli.get("neutral_score"))
        label = str(nli.get("label", "")).lower()

        if ("entailment" in label or entailment >= self.MIN_NLI_SIGNAL) and entailment > contradiction:
            if entailment >= self.MIN_NLI_SIGNAL:
                return "SUPPORTING"
            return "NEUTRAL"
        elif ("contradiction" in label or contradiction >= self.MIN_NLI_SIGNAL) and contradiction > entailment:
            if contradiction >= self.MIN_NLI_SIGNAL:
                return "CONTRADICTING"
            return "NEUTRAL"
        else:
            return "NEUTRAL"

    def score_evidence(
        self,
        claim: str,
        passages: List[Passage],
        nli_results: List[Dict[str, Any]],
        domain: str,
    ) -> Dict[str, Any]:
        """
        Score and aggregate evidence from passages and NLI classifications.
        Enforces invariant: supporting_count + contradicting_count + neutral_count + irrelevant_count == total_passages.
        """
        if not passages or not nli_results:
            return {
                "support_score": 0.0,
                "contradiction_score": 0.0,
                "trust_score": 0.0,
                "confidence_score": 0.0,
                "verdict": VerdictLabel.UNVERIFIED,
                "evidence_classification_counts": {
                    "supporting": 0,
                    "contradicting": 0,
                    "neutral": 0,
                    "irrelevant": 0,
                    "total": 0,
                },
            }

        support_weights: List[float] = []
        contradiction_weights: List[float] = []
        supporting_sources: set[str] = set()
        contradicting_sources: set[str] = set()

        classification_counts = {
            "supporting": 0,
            "contradicting": 0,
            "neutral": 0,
            "irrelevant": 0,
            "total": len(passages),
        }

        for passage, nli in zip(passages, nli_results):
            evidence_class = self.classify_passage(claim, passage, nli)
            classification_counts[evidence_class.lower()] += 1

            if evidence_class == "IRRELEVANT" or evidence_class == "NEUTRAL":
                continue

            entailment = self._bounded(nli.get("entailment_score"))
            contradiction = self._bounded(nli.get("contradiction_score"))
            neutral = self._bounded(nli.get("neutral_score"))

            source_id = str(
                getattr(passage, "source_id", "")
                or getattr(passage, "source", "")
                or "unknown"
            )

            credibility = self._bounded(
                self.reliability_manager.get_credibility(domain, source_id),
                0.80,
            )
            recency = self._bounded(
                self.reliability_manager.compute_recency_factor(
                    getattr(passage, "publication_date", "")
                ),
                1.0,
            )
            relevance = self.relevance_weight(
                getattr(passage, "relevance_score", 0.5)
            )

            base_weight = credibility * recency * relevance

            if evidence_class == "SUPPORTING":
                support_signal = entailment * (1.0 - 0.35 * neutral)
                effective_support = support_signal * base_weight
                support_weights.append(effective_support)
                supporting_sources.add(source_id)
            elif evidence_class == "CONTRADICTING":
                contradiction_signal = contradiction * (1.0 - 0.35 * neutral)
                effective_contradiction = contradiction_signal * base_weight
                contradiction_weights.append(effective_contradiction)
                contradicting_sources.add(source_id)

        # Invariant assertion check
        total_classified = (
            classification_counts["supporting"]
            + classification_counts["contradicting"]
            + classification_counts["neutral"]
            + classification_counts["irrelevant"]
        )
        assert total_classified == len(passages), f"Classification invariant violated: {total_classified} != {len(passages)}"

        support_max = max(support_weights, default=0.0)
        contradiction_max = max(contradiction_weights, default=0.0)
        support_avg = sum(support_weights) / len(support_weights) if support_weights else 0.0
        contradiction_avg = (
            sum(contradiction_weights) / len(contradiction_weights)
            if contradiction_weights
            else 0.0
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

        # Trust Score calculation
        if support_score > contradiction_score and support_score >= 0.25:
            trust_score = max(
                0.0,
                min(
                    1.0,
                    support_score * (1.0 - 0.90 * contradiction_score)
                    + 0.05 * min(1.0, len(supporting_sources) / 2.0),
                ),
            )
        else:
            trust_score = 0.0

        # Verdict Policy
        if support_score >= 0.30 and contradiction_score >= 0.30 and abs(support_score - contradiction_score) < 0.15:
            verdict = VerdictLabel.CONFLICTED
        elif support_score >= 0.30 and support_score > contradiction_score:
            verdict = VerdictLabel.VERIFIED
        elif contradiction_score >= 0.25 and contradiction_score > support_score:
            verdict = VerdictLabel.CONTRADICTED
        else:
            verdict = VerdictLabel.UNVERIFIED

        # Calibrated Confidence calculation (independent of trust_score, reflecting evidence certainty & consensus)
        primary_evidence_strength = max(support_score, contradiction_score)
        if primary_evidence_strength >= 0.25:
            consensus_factor = max(0.10, 1.0 - min(support_score, contradiction_score))
            verified_count = len(support_weights) + len(contradiction_weights)
            count_factor = 0.75 + 0.25 * min(1.0, verified_count / 3.0)
            confidence_score = round(
                max(0.0, min(1.0, primary_evidence_strength * consensus_factor * count_factor)),
                4,
            )
        else:
            confidence_score = 0.0

        return {
            "verdict": verdict,
            "support_score": round(support_score, 4),
            "contradiction_score": round(contradiction_score, 4),
            "trust_score": round(trust_score, 4),
            "confidence_score": confidence_score,
            "evidence_classification_counts": classification_counts,
        }

