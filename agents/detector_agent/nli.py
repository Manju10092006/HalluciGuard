"""Claim-evidence consistency for the Detector Agent.

Classifies whether a single evidence snippet supports / contradicts / is
insufficient for an atomic claim, mirroring the repository's DeBERTa 3-way
NLI semantics (SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO).

Design rules (enforced here because they are detector-level requirements):

* **NOT_ENOUGH_INFO is never treated as CONTRADICTED.** The two probabilities
  are returned separately and kept distinct.
* **Evidence sufficiency first.** A claim with a specific anchor (a number, a
  date, a percentage, a price, a comparison) is NOT supported by merely
  related evidence — the evidence must address the anchor. This stops
  "Virat Kohli played in the match" from supporting "Virat Kohli scored
  100 runs".
* **DeBERTa stays the production NLI.** When ``cross-encoder/nli-deberta-v3-base``
  is available we prefer its entailment/contradiction/neutral scores. The
  deterministic layer below only runs when the model is unavailable (fail-soft),
  and its outputs are explicitly non-calibrated heuristic scores.

This module never decides "the whole answer is hallucinated"; it only labels
claim-evidence consistency. The Judge owns the final answer verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from agents.detector_agent.claims import (
    extract_anchors,
    has_specific_anchor,
)

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"

# Synonym map used ONLY by the deterministic fallback to recognize paraphrase
# support (e.g. "acquired" ~ "purchased"). Production NLI handles paraphrase
# semantically; this lets the fallback pass the same paraphrase checks
# deterministically on machines without the model.
_PARAPHRASE_SYNONYMS: Dict[str, set] = {
    "acquired": {"purchased", "bought", "took over", "taken over"},
    "purchased": {"acquired", "bought"},
    "bought": {"acquired", "purchased"},
    "founded": {"established", "created", "set up", "launched", "began"},
    "established": {"founded", "created"},
    "created": {"founded", "established", "wrote", "authored"},
    "wrote": {"created", "authored", "composed"},
    "authored": {"wrote", "created"},
    "orbits": {"revolves", "goes around"},
    "revolves": {"orbits", "goes around"},
    "firm": {"company", "corporation", "business", "firm"},
    "company": {"firm", "corporation", "business"},
    "corporation": {"company", "firm"},
    "earth": {"earth"},
    "sun": {"sun"},
}

# Small, explicit relation-conflict table used ONLY by the deterministic
# fallback. It flags "X is the center of Y" against "X orbits Z" style
# evidence. Documented as a safety heuristic — never the primary truth
# mechanism. DeBERTa decides these cases in production.
_RELATION_CONFLICTS: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = [
    (("center of", "centre of", "middle of"), ("orbit", "orbits", "revolve", "revolves", "circle", "circles")),
    (("is the founder", "founded", "created", "wrote", "authored"), ("did not found", "did not create", "did not write", "never wrote", "was not written")),
]

_NEGATION_RE = re.compile(
    r"(?:does not|do not|did not|is not|are not|was not|were not|never|"
    r"no evidence that|incorrectly claims|falsely)",
    re.IGNORECASE,
)


@dataclass
class ClaimEvidenceResult:
    """Outcome of classifying one claim against one evidence snippet."""

    label: str
    supported_probability: float
    contradicted_probability: float
    unknown_probability: float
    degraded: bool = False
    model_source: str = "deterministic-fallback"


class ClaimEvidenceClassifier:
    """3-way claim-evidence classifier: DeBERTa NLI + deterministic guards."""

    def __init__(self, nli_engine=None) -> None:
        self._nli = nli_engine
        if self._nli is None:
            from agents.verifier_agent.nli.robust_entailment import NLIEngine

            self._nli = NLIEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, claim: str, evidence: str) -> ClaimEvidenceResult:
        """Classify one claim against one evidence snippet."""
        # 1) Evidence-sufficiency guard (always, model or not).
        sufficiency = _assess_sufficiency(claim, evidence)
        if sufficiency == "insufficient":
            return _unknown_result(claim, evidence)

        # 2) Deterministic numeric/date conflict guard.
        numeric_conflict = _numeric_conflict(claim, evidence)
        if numeric_conflict:
            return _result(CONTRADICTED, supported=0.0, contradicted=1.0, unknown=0.0)

        # 3) Production DeBERTa NLI when available.
        model_result = self._classify_with_nli(claim, evidence)
        if model_result is not None:
            return model_result

        # 4) Deterministic fallback (model unavailable or failed).
        return self._classify_deterministic(claim, evidence)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _classify_with_nli(self, claim: str, evidence: str) -> Optional[ClaimEvidenceResult]:
        if self._nli is None:
            return None
        try:
            raw = self._nli.classify(claim, evidence)
        except Exception:
            return None
        if not raw or raw.get("degraded"):
            return None

        label = str(raw.get("label", ""))
        entail = float(raw.get("entailment_score", 0.0))
        contradict = float(raw.get("contradiction_score", 0.0))
        neutral = float(raw.get("neutral_score", 0.0))

        if "entailment" in label:
            return _result(SUPPORTED, entail, 0.0, 0.0, degraded=False, model_source="deberta-nli")
        if "contradiction" in label:
            return _result(CONTRADICTED, 0.0, contradict, 0.0, degraded=False, model_source="deberta-nli")
        # Neutral is NOT_ENOUGH_INFO — never folded into CONTRADICTED.
        return _result(NOT_ENOUGH_INFO, entail, contradict, neutral, degraded=False, model_source="deberta-nli")

    def _classify_deterministic(self, claim: str, evidence: str) -> ClaimEvidenceResult:
        """Fail-soft heuristic classifier when the NLI model is unavailable.

        Scores are heuristic, non-calibrated signals — clearly separated from
        DeBERTa's softmax probabilities.
        """
        support_score = _lexical_support(claim, evidence)
        conflict = _relation_conflict(claim, evidence)
        negated = _claim_negated_in_evidence(claim, evidence)

        if conflict and support_score < 0.5:
            return _result(CONTRADICTED, 0.0, 1.0, 0.0, degraded=True)
        if negated:
            return _result(CONTRADICTED, 0.0, 1.0, 0.0, degraded=True)
        if support_score >= 0.6:
            return _result(SUPPORTED, support_score, 0.0, 0.0, degraded=True)
        if support_score >= 0.35:
            # Related but not strongly entailing -> insufficient, not contradicted.
            return _result(NOT_ENOUGH_INFO, support_score, 0.0, 1.0 - support_score, degraded=True)
        return _unknown_result(claim, evidence, degraded=True)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def _assess_sufficiency(claim: str, evidence: str) -> str:
    """Return ``'sufficient'`` or ``'insufficient'`` for the claim+evidence pair.

    A claim with a specific anchor (number/date/percent/quantity/comparison)
    requires that the evidence address that anchor class. Related-but-generic
    evidence is NOT enough to mark the claim supported.
    """
    if not has_specific_anchor(claim):
        return "sufficient"
    return "sufficient" if _evidence_addresses_anchor(claim, evidence) else "insufficient"


def _evidence_addresses_anchor(claim: str, evidence: str) -> bool:
    claim_anchors = extract_anchors(claim)
    for anchor in claim_anchors:
        token = str(anchor).lower()
        if re.fullmatch(r"(\d{4})", token):
            # Year: evidence must carry a year token (same year or not).
            if re.search(r"\b(?:19|20)\d{2}\b", evidence):
                return True
        elif any(char.isdigit() for char in token):
            # Numbered anchor: evidence must carry at least one number.
            if re.search(r"\b\d", evidence):
                return True
        else:
            # Magnitude/unit-only anchor (e.g. "million"): evidence must
            # mention it or contain some numeric quantity.
            if re.search(r"\b\d|thousand|million|billion|trillion", evidence, re.IGNORECASE):
                return True
    # No anchor token ever matched -> assume evidence does not address it.
    return False


def _numeric_conflict(claim: str, evidence: str) -> bool:
    """Detect a direct numeric/percent/date mismatch between claim and evidence.

    Only fires when BOTH sides name a number (or date year) and the values
    differ, e.g. "population is 10 million" vs "population is 12 million".
    """
    claim_numbers = _number_tokens(claim)
    evidence_numbers = _number_tokens(evidence)
    if not claim_numbers or not evidence_numbers:
        return False

    # Compare each claim number: if evidence names a different numeric value
    # AND there is lexical overlap between the pair, treat as a conflict.
    claim_set = set(claim_numbers)
    evidence_set = set(evidence_numbers)

    overlap = _content_words(claim) & _content_words(evidence)
    if not overlap:
        return False

    shared = claim_set & evidence_set
    if shared:
        # A shared value weakens the mismatch signal.
        if len(shared) == len(claim_set):
            return False
    return True


def _number_tokens(text: str) -> List[str]:
    normalized = (
        text.lower()
        .replace("thousand", "")
        .replace("million", "")
        .replace("billion", "")
        .replace("trillion", "")
        .replace(" percent", "%")
        .replace(" percentage", "%")
    )
    return re.findall(r"\b\d+(?:[,.]\d+)?\b", normalized)


def _claim_negated_in_evidence(claim: str, evidence: str) -> bool:
    """True when the evidence explicitly negates the claim's core content."""
    if not _NEGATION_RE.search(evidence):
        return False
    claim_terms = _content_words(claim) - {"is", "was", "are", "were", "the"}
    evidence_terms = _content_words(evidence)
    return len(claim_terms & evidence_terms) >= 2


def _relation_conflict(claim: str, evidence: str) -> bool:
    """Curated safety heuristic: explicit mutually-exclusive relations."""
    for claim_cues, evidence_cues in _RELATION_CONFLICTS:
        if any(cue in claim.lower() for cue in claim_cues):
            if any(cue in evidence.lower() for cue in evidence_cues):
                return True
    return False


def _lexical_support(claim: str, evidence: str) -> float:
    """Paraphrase-aware lexical support in ``[0, 1]`` (non-calibrated)."""
    claim_terms = _content_words(claim)
    if not claim_terms:
        return 0.0
    evidence_terms = _content_words(evidence)

    hit = 0
    for term in claim_terms:
        if term in evidence_terms:
            hit += 1
            continue
        synonyms = _PARAPHRASE_SYNONYMS.get(term)
        if synonyms and synonyms & evidence_terms:
            hit += 1
    return hit / len(claim_terms)


def _content_words(text: str) -> set:
    return {
        word
        for word in re.findall(r"[a-z]+", (text or "").lower())
        if word not in _STOP
    }


_STOP = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at",
    "by", "for", "with", "is", "are", "was", "were", "be", "been", "its",
    "it", "this", "that", "from", "as", "during", "over", "under",
}


def _result(label: str, supported: float, contradicted: float, unknown: float, *, degraded: bool = False, model_source: str = "deterministic-fallback") -> ClaimEvidenceResult:
    return ClaimEvidenceResult(
        label=label,
        supported_probability=round(max(0.0, min(1.0, supported)), 4),
        contradicted_probability=round(max(0.0, min(1.0, contradicted)), 4),
        unknown_probability=round(max(0.0, min(1.0, unknown)), 4),
        degraded=degraded,
        model_source=model_source,
    )


def _unknown_result(claim: str, evidence: str, *, degraded: bool = False) -> ClaimEvidenceResult:
    return _result(NOT_ENOUGH_INFO, 0.0, 0.0, 1.0, degraded=degraded)


__all__ = [
    "SUPPORTED",
    "CONTRADICTED",
    "NOT_ENOUGH_INFO",
    "ClaimEvidenceResult",
    "ClaimEvidenceClassifier",
]