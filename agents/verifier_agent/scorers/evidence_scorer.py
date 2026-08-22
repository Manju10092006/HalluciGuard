from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

from schemas.models import Passage, VerdictLabel, EntailmentLabel
from .source_reliability import SourceReliabilityManager
from .relation_verifier import RelationVerifier


class EvidenceScorer:
    """
    Combine NLI strength, retrieval quality, source trust, and recency into
    mathematically grounded evidence scores, classifications, and calibrated confidence.

    V1.2 Evidence Semantics:
      - Strict Relevance Gate: passages with relevance below threshold (default 0.20) are IRRELEVANT
      - 4 Mutually Exclusive Classes: SUPPORTING, CONTRADICTING, NEUTRAL, IRRELEVANT
      - Invariant: sum(classification_counts) == total_passages
      - Effective Weight: relevance_weight * nli_strength * credibility * recency * validity_factor
      - URL-level deduplication: multiple passages from same canonical URL only contribute once
      - Calibrated Confidence: independent of Trust, non-zero for both VERIFIED and CONTRADICTED
    """

    MIN_NLI_SIGNAL = 0.35  # Aligned with decision-grade evidence selection threshold

    def __init__(self, credibility_config: Dict[str, Any] | None = None) -> None:
        self.reliability_manager = SourceReliabilityManager()
        self.relation_verifier = RelationVerifier()

    def _get_relevance_gate_threshold(self) -> float:
        """Get the relevance gate threshold from settings, fallback to 0.20."""
        try:
            from config.settings import get_settings
            return float(get_settings().evidence_relevance_gate)
        except Exception:
            return 0.20

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
        # Cross-encoder scores for contradiction premises range around 0.02 - 0.20.
        # Calibrate with power transform so candidate passages provide meaningful weight.
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
            "fanciful", "absurdity", "nursery", "rhyme", "expression", "phrase",
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

    def classify_evidence(
        self,
        claim: str,
        passage: Passage,
        nli: Dict[str, Any],
    ) -> str:
        """
        Classify a single passage into exactly ONE mutually exclusive category:
        'SUPPORTING', 'CONTRADICTING', 'NEUTRAL', 'IRRELEVANT'.

        Decision Flow:
          1. Relevance Gate: If relevance_score < evidence_relevance_gate (default 0.20) -> 'IRRELEVANT'
          2. Validity Check: If NLI inference was degraded or marked invalid -> 'NEUTRAL'
          3. Relation Check Override: If object/relation mismatch -> 'CONTRADICTING' (bypasses word-coverage suppression)
          4. Non-assertive Context: If passage discusses myth/idiom/misconception -> 'NEUTRAL'
          5. NLI Signal:
             - Entailment >= MIN_NLI_SIGNAL and Entailment > Contradiction -> 'SUPPORTING'
             - Contradiction >= MIN_NLI_SIGNAL and Contradiction > Entailment -> 'CONTRADICTING'
             - Otherwise -> 'NEUTRAL'
        """
        gate_threshold = self._get_relevance_gate_threshold()
        rel_score = float(getattr(passage, "relevance_score", 0.5) or 0.0)
        hint_score = float(getattr(passage, "source_confidence_hint", 0.0) or 0.0)
        if max(rel_score, hint_score) < gate_threshold and rel_score < gate_threshold:
            return "IRRELEVANT"

        # Check if NLI inference was degraded or invalid
        if nli.get("nli_degraded", False) or nli.get("validity_factor", 1.0) == 0.0:
            return "NEUTRAL"

        # Structured Relation Verification Check
        rel_result = self.relation_verifier.verify_relation(claim, [passage])
        if rel_result.status in ("OBJECT_MISMATCH", "RELATION_MISMATCH"):
            # Direct contradiction at relation level overrides false entailment or neutral
            # Bypasses the word-coverage suppression check completely
            return "CONTRADICTING"
        elif rel_result.status == "MATCH":
            return "SUPPORTING"

        # Guard against myths / proverbs / common misconceptions matching
        # assertively if the passage qualifies them as untrue, figurative, or an idiom.
        full_text = f"{getattr(passage, 'title', '')} {passage.snippet}"
        snippet_lower = full_text.lower()
        refutation_phrases = (
            "disproven", "debunked", "misconception", "hoax", "untrue",
            "falsely", "refuted", "no evidence", "scientifically disproven",
            "incorrectly claimed", "not true", "not associated", "is false",
        )
        has_explicit_refutation = any(rf in snippet_lower for rf in refutation_phrases)
        if has_explicit_refutation and not any(neg in claim.lower() for neg in ("not", "never", "disproven", "false", "myth")):
            return "CONTRADICTING"

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
                # If passage lacks key predicate terms and has no explicit refutation phrases, it is neutral context
                claim_words = [
                    w.lower() for w in re.findall(r"[a-zA-Z0-9]+", claim)
                    if len(w) > 3 and w.lower() not in {
                        "is", "the", "a", "an", "in", "of", "to", "and", "or", "for", "with",
                        "that", "this", "from", "was", "were", "been", "have", "has", "had",
                        "associated", "related", "called", "known", "named"
                    } and not (w.isdigit() and len(w) == 4)
                ]
                if len(claim_words) >= 2:
                    snippet_lower = full_text.lower()
                    covered = sum(1 for w in claim_words if w in snippet_lower)
                    refutation_phrases = (
                        "disproven", "debunked", "misconception", "hoax", "untrue",
                        "falsely", "refuted", "no evidence", "myth", "incorrectly claimed",
                        "not true", "not associated", "not related", "is false",
                    )
                    has_explicit_refutation = any(
                        rf in snippet_lower for rf in refutation_phrases
                    )
                    if covered == 0 and not has_explicit_refutation:
                        return "NEUTRAL"

                return "CONTRADICTING"
            return "NEUTRAL"
        else:
            return "NEUTRAL"

    # Backwards-compatibility alias
    classify_passage = classify_evidence

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
        Deduplicates evidence from identical canonical URLs to prevent double-counting.
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

        classification_counts = {
            "supporting": 0,
            "contradicting": 0,
            "neutral": 0,
            "irrelevant": 0,
            "total": len(passages),
        }

        # Track per-URL max weight to prevent multi-chunk duplicate inflation
        url_support_weights: Dict[str, float] = {}
        url_contradiction_weights: Dict[str, float] = {}
        supporting_sources: set[str] = set()
        contradicting_sources: set[str] = set()

        for passage, nli in zip(passages, nli_results):
            rel_result = self.relation_verifier.verify_relation(claim, [passage])
            evidence_class = self.classify_evidence(claim, passage, nli)
            classification_counts[evidence_class.lower()] += 1

            if evidence_class in ("IRRELEVANT", "NEUTRAL"):
                continue

            entailment = self._bounded(nli.get("entailment_score"))
            contradiction = self._bounded(nli.get("contradiction_score"))
            neutral = self._bounded(nli.get("neutral_score"))
            validity_factor = 0.0 if nli.get("nli_degraded", False) else float(nli.get("validity_factor", 1.0))

            # Apply relation verification signal calibration
            if rel_result.status in ("OBJECT_MISMATCH", "RELATION_MISMATCH"):
                contradiction = max(contradiction, 0.95)
                entailment = 0.0
                neutral = 0.05
            elif rel_result.status == "MATCH":
                entailment = max(entailment, 0.95)
                contradiction = 0.0
                neutral = 0.05

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

            # Canonical URL / doc key for deduplication
            url_key = (getattr(passage, "url", "") or source_id or getattr(passage, "title", "")).strip().lower()

            base_weight = credibility * recency * relevance * validity_factor

            if evidence_class == "SUPPORTING":
                support_signal = entailment * (1.0 - 0.35 * neutral)
                effective_support = support_signal * base_weight
                # Keep strongest passage per canonical URL
                if url_key not in url_support_weights or effective_support > url_support_weights[url_key]:
                    url_support_weights[url_key] = effective_support
                supporting_sources.add(source_id)
            elif evidence_class == "CONTRADICTING":
                contradiction_signal = contradiction * (1.0 - 0.35 * neutral)
                effective_contradiction = contradiction_signal * base_weight
                # Keep strongest passage per canonical URL
                if url_key not in url_contradiction_weights or effective_contradiction > url_contradiction_weights[url_key]:
                    url_contradiction_weights[url_key] = effective_contradiction
                contradicting_sources.add(source_id)

        # Invariant assertion check
        total_classified = (
            classification_counts["supporting"]
            + classification_counts["contradicting"]
            + classification_counts["neutral"]
            + classification_counts["irrelevant"]
        )
        assert total_classified == len(passages), f"Classification invariant violated: {total_classified} != {len(passages)}"

        support_weights = list(url_support_weights.values())
        contradiction_weights = list(url_contradiction_weights.values())

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

        # Trust Score calculation (reflects supporting source reliability)
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

