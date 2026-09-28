"""Lightweight, deterministic claim typing for the Detector Agent.

Claim typing has exactly ONE job: decide what kind of factual evidence a claim
needs so the Detector can (a) avoid flagging opinions as hallucinations and
(b) refuse to call a specific factual claim SUPPORTED from generic related
evidence. It never decides truth by itself.

Typing is deliberately heuristic and LLM-independent so it works on normal
developer hardware without model downloads. When in doubt a claim is typed
FACTUAL (the safe default for an evidence-verification gate).
"""

from __future__ import annotations

import re
from enum import Enum

from typing import List


class ClaimType(str, Enum):
    """Coarse, deterministic factual-claim type used by the Detector."""

    FACTUAL = "FACTUAL"
    NUMERICAL = "NUMERICAL"
    TEMPORAL = "TEMPORAL"
    ENTITY = "ENTITY"
    RELATIONAL = "RELATIONAL"
    COMPARATIVE = "COMPARATIVE"
    OPINION = "OPINION"


# Opinion / non-factual surface cues. Conservative on purpose: a claim should
# only be treated as non-factual when it clearly asserts preference or belief.
_OPINION_RE = re.compile(
    r"\b(?:"
    r"in my opinion|i think|i believe|i feel|i prefer|as far as i'm concerned|"
    r"in my view|my favorite|personally"
    r")\b",
    re.IGNORECASE,
)

# Superlative / value-judgment adjectives — "the best X", "the worst Y".
_SUPERLATIVE_RE = re.compile(
    r"\b(?:"
    r"best|worst|greatest|favorite|most (?:useful|powerful|popular|impressive|"
    r"beautiful|important|effective|reliable|beautiful|amazing)|"
    r"beautiful|amazing|terrible|awful|brilliant|wonderful"
    r")\b",
    re.IGNORECASE,
)

# Comparative constructions ("more than", "higher than", "bigger", "-er forms").
_COMPARATIVE_RE = re.compile(
    r"\b(?:"
    r"more than|less than|higher than|lower than|bigger than|smaller than|"
    r"faster than|slower than|better than|worse than|larger than|"
    r"over\s+\d|under\s+\d|above\s+\d|below\s+\d|"
    r"compared to|compared with|exceeds|beat|surpassed"
    r")\b|"
    r"\b\w{2,}er\s+than\b",
    re.IGNORECASE,
)

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_MONTH_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?%")
_CURRENCY_RE = re.compile(r"[£€$]\s?\d")
_NUMBER_RE = re.compile(r"\b\d+(?:[,.]\d+)*\b")
_MAGNITUDE_RE = re.compile(
    r"\b(?:thousand|million|billion|trillion)\b", re.IGNORECASE
)
_UNIT_RE = re.compile(
    r"\b(?:km|kg|kmh|mph|gb|tb|hz|ghz|runs|points|goals|years|meters|metres|"
    r"miles|dollars|usd|euros|pounds|percent|%)\b",
    re.IGNORECASE,
)

# Relational predicates typical of factual relation claims.
_RELATIONAL_RE = re.compile(
    r"\b(?:acquired|purchased|bought|merged|acqui-hired|founded|established|"
    r"created|headquartered|located|owns|subsidiary|acquired by|purchased by|"
    r"founded by|led by|succeeded by|wrote|authored|designed|developed)\b[\w'’'-]*",
    re.IGNORECASE,
)


def _has_opinion_cues(claim: str) -> bool:
    if _OPINION_RE.search(claim):
        return True
    if _SUPERLATIVE_RE.search(claim):
        # "the best X" is only an opinion when it is a value judgment, not a
        # factual description ("the best-selling album" is factual). We are
        # intentionally conservative: bare superlatives count as opinion.
        return True
    return False


def _has_temporal_anchor(claim: str) -> bool:
    if _YEAR_RE.search(claim):
        return True
    if _MONTH_RE.search(claim):
        return True
    return bool(re.search(r"\b(?:in|by|during|since|until|from)\s+(?:the\s+)?\d{4}\b", claim, re.IGNORECASE))


def _has_numeric_anchor(claim: str) -> bool:
    return bool(
        _PERCENT_RE.search(claim)
        or _CURRENCY_RE.search(claim)
        or _NUMBER_RE.search(claim)
        or _MAGNITUDE_RE.search(claim)
    )


def _has_comparative_anchor(claim: str) -> bool:
    return bool(_COMPARATIVE_RE.search(claim))


def classify_claim_type(claim: str) -> ClaimType:
    """Classify a claim into a coarse factual type.

    Priority order: OPINION > TEMPORAL > NUMERICAL > COMPARATIVE > RELATIONAL
    > ENTITY > FACTUAL. NUMERICAL/TEMPORAL/COMPARATIVE are "specific-anchor"
    types: the Detector must not call them SUPPORTED without evidence that
    addresses the anchor.
    """
    text = (claim or "").strip()
    if not text:
        return ClaimType.FACTUAL

    if _has_opinion_cues(text):
        return ClaimType.OPINION
    if _has_temporal_anchor(text):
        return ClaimType.TEMPORAL
    if _has_numeric_anchor(text):
        return ClaimType.NUMERICAL
    if _has_comparative_anchor(text):
        return ClaimType.COMPARATIVE
    if _RELATIONAL_RE.search(text):
        return ClaimType.RELATIONAL
    # A capitalized / multi-token entity anywhere suggests an entity-focused claim.
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text):
        return ClaimType.ENTITY
    return ClaimType.FACTUAL


def has_specific_anchor(claim: str) -> bool:
    """True when the claim carries a number/date/percent/quantity/comparison.

    Such claims are NOT supported by merely-related evidence. Evidence must
    explicitly address the anchor (e.g. contain the same or a conflicting
    number/date).
    """
    return (
        _has_numeric_anchor(claim)
        or _has_temporal_anchor(claim)
        or _has_comparative_anchor(claim)
    )


def needs_specific_evidence(claim: str) -> bool:
    """Alias for ``has_specific_anchor`` used by the evidence-sufficiency layer."""
    return has_specific_anchor(claim)


def extract_anchors(claim: str) -> List[str]:
    """Extract the specific numeric/temporal anchors named in ``claim``.

    Returns normalized anchor tokens (e.g. "2016", "10 million", "100 runs").
    Used for debugging and for the evidence-sufficiency guard.
    """
    text = (claim or "").strip()
    anchors: List[str] = []

    years = _YEAR_RE.findall(text)
    anchors.extend(years)

    percents = _PERCENT_RE.findall(text)
    anchors.extend(percents)

    currencies = _CURRENCY_RE.findall(text)
    anchors.extend(currencies)

    # number + magnitude (e.g. "10 million").
    for match in re.finditer(r"\b(\d+(?:[,.]\d+)*)\s*(thousand|million|billion|trillion)\b", text, re.IGNORECASE):
        anchors.append(f"{match.group(1)} {match.group(2).lower()}")

    # number + unit (e.g. "100 runs", "300 km").
    for match in re.finditer(r"\b(\d+(?:[,.]\d+)*)\s*(runs|points|goals|km|kg|mph|years|meters|metres|miles|dollars)\b", text, re.IGNORECASE):
        anchors.append(f"{match.group(1)} {match.group(2).lower()}")

    return anchors


__all__ = [
    "ClaimType",
    "classify_claim_type",
    "has_specific_anchor",
    "needs_specific_evidence",
    "extract_anchors",
]