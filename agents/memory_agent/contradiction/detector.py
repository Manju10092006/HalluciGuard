from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "had", "her", "was", "one", "our", "out", "has", "his", "how",
    "its", "may", "new", "now", "old", "see", "way", "who", "did",
    "get", "let", "say", "she", "too", "use", "that", "with", "have",
    "this", "will", "your", "from", "they", "been", "said", "each",
    "make", "like", "than", "them", "then", "what", "when", "were",
    "does", "over", "into", "more",
}

_NEGATION_RE = re.compile(
    r"\bnot\b|\bnever\b|\bno\b|\bnone\b|\bno longer\b|\bwithout\b|"
    r"doesn'?t\b|does not\b|do not\b|don'?t\b|didn'?t\b|did not\b|didnt\b|"
    r"isn'?t\b|is not\b|ain'?t\b|aren'?t\b|are not\b|"
    r"wasn'?t\b|was not\b|weren'?t\b|were not\b|"
    r"can'?t\b|cannot\b|can not\b|cant\b|won'?t\b|will not\b|"
    r"shouldn'?t\b|wouldn'?t\b"
)

_TEMPORAL_VERB_RE = re.compile(
    r"\b(founded|created|invented|discovered|established|launched|"
    r"originated|began|started|released|occurred|introduced)\b",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(r"\b\d{1,4}(?:[.,]\d+)?\b")


class ContradictionDetector:
    """Stage-2 confirmation that a candidate is an actual logical contradiction.

    Vector similarity only surfaces *candidate* pairs. This stage decides
    whether two similar claims truly contradict each other using:

    1. NLI (premise/hypothesis classification) when ``use_nli`` is enabled, or
    2. an explicit, deterministic structured comparison otherwise:
       - negation asymmetry on otherwise-overlapping content, and
       - conflicting temporal values (e.g. "founded in 1998" vs "founded in 1999").

    If neither confirms, the pair is NOT a contradiction. Similarity alone is
    never treated as proof.
    """

    def __init__(
        self,
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        use_nli: bool = False,
        nli_threshold: float = 0.5,
        overlap_threshold: float = 0.6,
    ):
        self._nli_model = nli_model
        self._use_nli = use_nli
        self._nli_threshold = nli_threshold
        self._overlap_threshold = overlap_threshold
        self._nli_pipeline = None
        self._nli_attempted = False
        self.nli_available = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def confirm(self, claim_a: str, claim_b: str) -> tuple[bool, Optional[str]]:
        """Return (is_contradiction, method) for two claims.

        method is 'nli' or 'structured' when confirmed and None otherwise.
        """
        if self._use_nli:
            nli_confirmed = self._confirm_nli(claim_a, claim_b)
            if nli_confirmed is not None:
                return nli_confirmed
            # NLI unavailable/failed -> fall through to structured comparison
        return self._confirm_structured(claim_a, claim_b)

    # ------------------------------------------------------------------
    # NLI path (preferred when enabled and available)
    # ------------------------------------------------------------------

    def _confirm_nli(self, claim_a: str, claim_b: str) -> Optional[tuple[bool, str]]:
        self._ensure_nli()
        if not self.nli_available or self._nli_pipeline is None:
            return None

        # Directional checks: hypothesis B given premise A, and A given B.
        for premise, hypothesis in ((claim_a, claim_b), (claim_b, claim_a)):
            try:
                result = self._nli_pipeline({"text": premise, "text_pair": hypothesis})
                if result and isinstance(result[0], list):
                    result = result[0]
                label = str(result[0].get("label", "")).lower()
                score = float(result[0].get("score", 0.0))
                if "contradict" in label and score >= self._nli_threshold:
                    return True, "nli"
            except Exception as e:
                logger.warning("NLI pair check failed, falling back: %s", e)
                return None
        return False, "nli"

    def _ensure_nli(self) -> None:
        if self._nli_attempted:
            return
        self._nli_attempted = True
        try:
            from transformers import pipeline

            self._nli_pipeline = pipeline(
                "text-classification", model=self._nli_model
            )
            self.nli_available = True
            logger.info("NLI contradiction model loaded: %s", self._nli_model)
        except Exception as e:
            self.nli_available = False
            self._nli_pipeline = None
            logger.warning(
                "NLI contradiction model %s unavailable (%s); "
                "falling back to structured comparison.",
                self._nli_model, e,
            )

    # ------------------------------------------------------------------
    # Structured path (deterministic, no model download)
    # ------------------------------------------------------------------

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z]{3,}", text.lower())) - _STOPWORDS

    def _negated_predicates(self, claim: str) -> set[str]:
        """Content word after a negation marker (stem-prefix), if any.

        E.g. "does not cure cancer" -> {"cure"}. A negation only counts as a
        contradiction when the negated predicate is actually asserted by the
        other claim.
        """
        predicates: set[str] = set()
        for m in _NEGATION_RE.finditer(claim):
            tail = re.findall(r"[a-z]{3,}", claim[m.end():].lower())
            for word in tail:
                if word not in _STOPWORDS:
                    predicates.add(word[:4])
                    break
        return predicates

    def _confirm_structured(self, claim_a: str, claim_b: str) -> tuple[bool, Optional[str]]:
        toks_a = self._content_tokens(claim_a)
        toks_b = self._content_tokens(claim_b)
        if not toks_a or not toks_b:
            return False, None

        min_size = min(len(toks_a), len(toks_b))
        overlap = len(toks_a & toks_b) / max(1, min_size) if min_size else 0.0

        neg_a = bool(_NEGATION_RE.search(claim_a))
        neg_b = bool(_NEGATION_RE.search(claim_b))

        # Same subject matter, exactly one side negates, AND the negated
        # predicate is asserted by the other side (e.g. "cures" vs "does not cure").
        if overlap >= self._overlap_threshold and neg_a != neg_b:
            a_text = " ".join(toks_a)
            b_text = " ".join(toks_b)
            negated_in_a = self._negated_predicates(claim_a)
            negated_in_b = self._negated_predicates(claim_b)
            if any(p in a_text for p in negated_in_b) or any(
                p in b_text for p in negated_in_a
            ):
                return True, "structured"
            return False, None

        # Same temporal verb with conflicting values, e.g. founded 1998 vs 1999.
        if (
            _TEMPORAL_VERB_RE.search(claim_a)
            and _TEMPORAL_VERB_RE.search(claim_b)
            and overlap >= self._overlap_threshold
        ):
            nums_a = set(_NUMBER_RE.findall(claim_a))
            nums_b = set(_NUMBER_RE.findall(claim_b))
            shared = nums_a & nums_b
            if nums_a and nums_b and shared != nums_a and nums_b - nums_a:
                return True, "structured"

        return False, None