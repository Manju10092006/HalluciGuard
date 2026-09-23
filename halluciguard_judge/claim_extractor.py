"""
halluciguard_judge / claim_extractor.py
────────────────────────────────────────
Atomic claim extraction from LLM responses.

PIPELINE POSITION: FIRST — runs BEFORE the classifier.
    LLM Response -> ClaimExtractor -> [Claim, ...] -> Classifier

Architecture:
    1. Sentence-split the response (nltk punkt tokenizer)
    2. Filter trivial / non-factual sentences
    3. (Optional) Sub-sentence splitting for compound claims
    4. Classify each claim by type (FACTUAL / NUMERICAL / ENTITY / CAUSAL)
    5. Return ordered list of Claim objects with source_span

Inspired by:
    - RefChecker (amazon-science): claim-level triple extraction idea
    - HalluciGuard spec: atomic factual units feed Verifier one-per-claim

NO external API calls in this module. CPU-only, fast (<50ms typical).
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import List, Optional

from .models import Claim, ClaimType

logger = logging.getLogger(__name__)

# ── Patterns ──────────────────────────────────────────────────────────────────

# Sentences that are almost certainly non-factual and should be dropped
_NON_FACTUAL_PATTERNS = [
    r"^(sure|of course|certainly|here('s| is)|let me|i('ll| will)|thank|please|happy to)",
    r"^\s*(yes|no|ok|okay)\s*[.,!]?\s*$",
    r"^(in (summary|conclusion|short)|to summarize|overall)",
    r"^(note|disclaimer|important|warning)\s*:",
]
_NON_FACTUAL_RE = re.compile(
    "|".join(_NON_FACTUAL_PATTERNS), re.IGNORECASE
)

# Signals that a sentence likely contains a verifiable factual claim
_FACTUAL_SIGNALS = [
    r"\b(is|are|was|were|has|have|had|will|would|can|created|invented|founded|built|developed|published|released|discovered|born|died)\b",
    r"\b(in \d{4}|\d{4}–\d{4}|\d+\s*(years?|percent|%|km|miles?|kg|lbs?))\b",
    r"\b(the|a|an)\s+\w+\s+(is|are|was|were)\b",
]
_FACTUAL_RE = re.compile("|".join(_FACTUAL_SIGNALS), re.IGNORECASE)

# Numerical / date patterns
_NUMERICAL_RE = re.compile(
    r"\b\d{4}\b|\b\d+[\.,]\d+\b|\b\d+\s*(percent|%|million|billion|thousand|km|kg|lbs?|years?)\b",
    re.IGNORECASE
)

# Entity patterns (proper nouns — simple heuristic)
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

# Compound clause splitters
_COMPOUND_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+|(?<=,)\s+(?:and|but|while|whereas|however|although|though)\s+",
    re.IGNORECASE
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences. Uses nltk if available, else regex."""
    try:
        import nltk
        try:
            tokenizer = nltk.data.load("tokenizers/punkt_tab/english.pickle")
        except Exception:
            try:
                nltk.download("punkt_tab", quiet=True)
                tokenizer = nltk.data.load("tokenizers/punkt_tab/english.pickle")
            except Exception:
                try:
                    nltk.download("punkt", quiet=True)
                    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
                except Exception:
                    tokenizer = None
        if tokenizer:
            return tokenizer.tokenize(text.strip())
    except ImportError:
        pass

    # Fallback: simple regex sentence split
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _classify_claim_type(text: str) -> ClaimType:
    """Heuristically classify the claim type."""
    if _NUMERICAL_RE.search(text):
        return ClaimType.NUMERICAL
    if _ENTITY_RE.search(text):
        return ClaimType.ENTITY
    causal_words = r"\b(because|caused|led to|resulted in|due to|as a result|therefore)\b"
    if re.search(causal_words, text, re.IGNORECASE):
        return ClaimType.CAUSAL
    if _FACTUAL_RE.search(text):
        return ClaimType.FACTUAL
    return ClaimType.GENERAL


def _estimate_importance(text: str, claim_type: ClaimType) -> str:
    """Estimate claim importance: HIGH | MEDIUM | LOW."""
    if claim_type in (ClaimType.NUMERICAL, ClaimType.ENTITY):
        return "HIGH"
    if claim_type == ClaimType.FACTUAL:
        return "HIGH" if len(text.split()) > 6 else "MEDIUM"
    if claim_type == ClaimType.CAUSAL:
        return "MEDIUM"
    return "LOW"


# ──────────────────────────────────────────────────────────────────────────────
# ClaimExtractor
# ──────────────────────────────────────────────────────────────────────────────

class ClaimExtractor:
    """Extracts atomic factual claims from an LLM response.

    Usage
    -----
        extractor = ClaimExtractor(max_claims=10, min_claim_length=10)
        claims = extractor.extract(llm_response, user_query)
        # -> List[Claim]

    Design principles:
    - No API calls. CPU-only. Fast.
    - Returns Claim objects (not raw strings) so downstream agents
      have structured metadata.
    - Respects max_claims cap to prevent cost runaway.
    """

    def __init__(
        self,
        max_claims: int = 15,
        min_claim_length: int = 10,
        split_compound: bool = True,
    ) -> None:
        self.max_claims = max_claims
        self.min_claim_length = min_claim_length
        self.split_compound = split_compound

    def extract(
        self,
        llm_response: str,
        user_query: Optional[str] = None,
    ) -> List[Claim]:
        """Extract atomic factual claims from an LLM response.

        Args:
            llm_response: The draft answer from the Base LLM.
            user_query:   Optional — used for importance scoring context.

        Returns:
            Ordered list of Claim objects.
        """
        if not llm_response or not llm_response.strip():
            logger.warning("[ClaimExtractor] Empty llm_response received.")
            return []

        sentences = _split_sentences(llm_response)
        candidates: List[str] = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            # Filter non-factual openers
            if _NON_FACTUAL_RE.match(sent):
                continue
            # Must be long enough
            if len(sent) < self.min_claim_length:
                continue

            # Optionally split compound sentences into sub-claims
            if self.split_compound:
                sub = _COMPOUND_SPLIT_RE.split(sent)
                sub = [s.strip() for s in sub if len(s.strip()) >= self.min_claim_length]
                candidates.extend(sub if sub else [sent])
            else:
                candidates.append(sent)

        # Deduplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for c in candidates:
            norm = c.lower().strip()
            if norm not in seen:
                seen.add(norm)
                unique.append(c)

        # Cap claims
        unique = unique[: self.max_claims]

        claims: List[Claim] = []
        for idx, text in enumerate(unique, start=1):
            ctype = _classify_claim_type(text)
            importance = _estimate_importance(text, ctype)
            claims.append(
                Claim(
                    claim_id=f"C{idx:03d}",
                    text=text,
                    claim_type=ctype,
                    source_span=text,
                    importance=importance,
                )
            )

        logger.info(
            "[ClaimExtractor] Extracted %d claims from response (%d chars).",
            len(claims),
            len(llm_response),
        )
        return claims


__all__ = ["ClaimExtractor"]
