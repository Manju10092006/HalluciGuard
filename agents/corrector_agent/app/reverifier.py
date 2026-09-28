"""ReVerifier — independent post-correction verification.

WHY THIS EXISTS
---------------
The Corrector repairs an answer, but a repair can itself introduce a NEW
hallucination: turning "Java was created by Snehith." into "Java was created by
James Gosling in 1991." fixes the creator yet fabricates a year (the evidence
says 1995). The pre-correction Judge (``app/judge.py``) cannot catch this — it
only checks whether the *originally flagged* wording was removed and whether the
evidence text was echoed, so a brand-new wrong fact is invisible to it.

WHAT THIS MODULE DOES (the contract)
------------------------------------
It RE-EXTRACTS claims from the CORRECTED answer (never assuming the Corrector
touched only the one claim we expected), verifies each re-extracted claim against
the same retrieved evidence, and returns a verdict per claim:

* ``SUPPORTED``     — grounded in supporting evidence.
* ``CONTRADICTED``  — restates contradiction evidence, or asserts a salient value
                      (number/date) that conflicts with supporting evidence.
* ``UNKNOWN``       — insufficient evidence. NOT treated as false.

Output contract:
    success -> {"status": "VERIFIED",
                "claims": [{claim_id, text, verdict, confidence}, ...],
                "overall_confidence": <float>}
    failure -> {"status": "FAILED",
                "failed_claims": [{text, verdict}, ...]}

FAILED means "route back to correction / Judge", never "emit the answer".

BOUNDARIES
----------
This is a SEPARATE verification event from the pre-correction Judge. It consumes
the pre-correction inputs read-only (query, corrected answer, original claims,
correction info, retrieved evidence, Judge decision) and MUST NOT mutate them:
the original verifier verdicts (``payload.claims``) survive untouched so both
events are kept. Pure and deterministic — no network, no model, no I/O. An
optional claim extractor / evidence scorer can be injected for testing, but the
defaults are self-contained so the pipeline never depends on a live API key.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence

from pydantic import BaseModel, Field

from app.models import (
    CorrectionPlan,
    JudgeVerificationPayload,
    JudgeVerificationResult,
)

# ---------------------------------------------------------------------------
# Verdict vocabulary (distinct from the pre-correction ClaimStatus enum)
# ---------------------------------------------------------------------------
SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
UNKNOWN = "UNKNOWN"

# Overlap (Jaccard on content tokens) at/above which a re-extracted claim is
# considered "about" an evidence passage. Kept conservative: too low and every
# sentence matches every passage; too high and paraphrase escapes detection.
_RELEVANCE_THRESHOLD = 0.18
# Strong overlap with a *contradiction* passage is itself a contradiction signal.
_CONTRADICTION_OVERLAP = 0.35


class ReverifiedClaim(BaseModel):
    """One claim re-extracted from the CORRECTED answer and independently judged."""

    claim_id: str
    text: str
    verdict: str  # SUPPORTED | CONTRADICTED | UNKNOWN
    confidence: float = 0.0
    rationale: str = ""


class ReverificationResult(BaseModel):
    """The ReVerifier's verdict on the corrected answer as a whole.

    Deliberately a NEW structure: it never overwrites the pre-correction
    ``JudgeVerificationResult`` or ``payload.claims``, so both verification
    events remain inspectable.
    """

    status: str  # VERIFIED | FAILED
    claims: List[ReverifiedClaim] = Field(default_factory=list)
    overall_confidence: float = 0.0

    @property
    def is_verified(self) -> bool:
        return self.status == "VERIFIED"

    @property
    def failed_claims(self) -> List[ReverifiedClaim]:
        return [c for c in self.claims if c.verdict == CONTRADICTED]

    def to_contract(self) -> dict:
        """Render the exact success/failure output contract shape."""
        if self.status == "VERIFIED":
            return {
                "status": "VERIFIED",
                "claims": [
                    {
                        "claim_id": c.claim_id,
                        "text": c.text,
                        "verdict": c.verdict,
                        "confidence": round(c.confidence, 4),
                    }
                    for c in self.claims
                ],
                "overall_confidence": round(self.overall_confidence, 4),
            }
        return {
            "status": "FAILED",
            "failed_claims": [
                {"text": c.text, "verdict": c.verdict} for c in self.failed_claims
            ],
        }


# ---------------------------------------------------------------------------
# Pure text primitives (no dependencies) — the re-extraction + salient values
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")
# A proper-noun-ish entity: a capitalised word not at the very start is a decent
# deterministic signal; used only for relevance, never as the sole verdict.
_ENTITY = re.compile(r"\b[A-Z][a-zA-Z]+\b")

_STOPWORDS = frozenset(
    {
        "the", "and", "was", "were", "are", "for", "with", "that", "this",
        "from", "has", "have", "had", "into", "than", "then", "been", "being",
        "which", "what", "who", "whom", "whose", "when", "where", "does", "did",
        "will", "would", "could", "should", "can", "may", "might", "there",
        "their", "they", "them", "his", "her", "its", "our", "your",
    }
)


def _norm_number(tok: str) -> str:
    """Normalise a numeric token so '1,991' and '1991' compare equal."""
    return tok.replace(",", "").rstrip(".")


def _content_tokens(text: str) -> set:
    """Lowercased content words (len>3, non-stopword) for overlap scoring."""
    return {
        w.lower()
        for w in _WORD.findall(text)
        if len(w) > 3 and w.lower() not in _STOPWORDS
    }


def _numbers(text: str) -> set:
    return {_norm_number(m) for m in _NUMBER.findall(text)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def default_claim_extractor(corrected_answer: str) -> List[str]:
    """Re-extract candidate claims from the CORRECTED answer.

    Deterministic sentence segmentation. This is intentionally independent of the
    Corrector's plan: the whole point is to inspect what the corrected text
    *actually* asserts, including any sentence the Corrector added or altered
    beyond the one claim we asked it to fix.
    """
    text = (corrected_answer or "").strip()
    if not text:
        return []
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text)]
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# The verifier
# ---------------------------------------------------------------------------

class _Passage:
    """A pre-tokenised evidence passage."""

    __slots__ = ("text", "tokens", "numbers")

    def __init__(self, text: str):
        self.text = text or ""
        self.tokens = _content_tokens(self.text)
        self.numbers = _numbers(self.text)


def _classify_claim(
    claim_text: str,
    supporting: Sequence[_Passage],
    contradictory: Sequence[_Passage],
) -> ReverifiedClaim:
    """Judge ONE re-extracted claim against evidence.

    Order of decision:
      1. Strong overlap with a contradiction passage -> CONTRADICTED.
      2. Overlaps a supporting passage but asserts a salient value (number/date)
         that conflicts with that passage's value -> CONTRADICTED (this is the
         secondary-hallucination catch, e.g. a wrong year).
      3. Overlaps a supporting passage with no value conflict -> SUPPORTED.
      4. No meaningful overlap with any evidence -> UNKNOWN (insufficient
         evidence; never treated as false).
    """
    claim_tokens = _content_tokens(claim_text)
    claim_numbers = _numbers(claim_text)

    # Best supporting / contradicting passage by content overlap.
    best_sup = max(
        supporting, key=lambda p: _jaccard(claim_tokens, p.tokens), default=None
    )
    best_con = max(
        contradictory, key=lambda p: _jaccard(claim_tokens, p.tokens), default=None
    )
    sup_overlap = _jaccard(claim_tokens, best_sup.tokens) if best_sup else 0.0
    con_overlap = _jaccard(claim_tokens, best_con.tokens) if best_con else 0.0

    # 1. Explicit contradiction evidence for what this claim restates.
    if best_con is not None and con_overlap >= _CONTRADICTION_OVERLAP:
        return ReverifiedClaim(
            claim_id="",
            text=claim_text,
            verdict=CONTRADICTED,
            confidence=round(min(0.99, 0.5 + con_overlap), 4),
            rationale="claim restates content covered by contradiction evidence",
        )

    # 2 & 3. Relevant to supporting evidence -> check for a conflicting value.
    if best_sup is not None and sup_overlap >= _RELEVANCE_THRESHOLD:
        # A salient value the claim asserts that the matched supporting passage
        # does NOT contain, while the passage DOES assert a value of that kind,
        # is a newly-introduced wrong fact (the Java-year case).
        conflicting = claim_numbers - best_sup.numbers
        if conflicting and best_sup.numbers:
            return ReverifiedClaim(
                claim_id="",
                text=claim_text,
                verdict=CONTRADICTED,
                confidence=round(min(0.99, 0.5 + sup_overlap), 4),
                rationale=(
                    "asserts value(s) "
                    + ", ".join(sorted(conflicting))
                    + " absent from supporting evidence, which asserts "
                    + ", ".join(sorted(best_sup.numbers))
                ),
            )
        return ReverifiedClaim(
            claim_id="",
            text=claim_text,
            verdict=SUPPORTED,
            confidence=round(min(0.99, sup_overlap + 0.15), 4),
            rationale="grounded in supporting evidence",
        )

    # 4. Nothing to check it against.
    return ReverifiedClaim(
        claim_id="",
        text=claim_text,
        verdict=UNKNOWN,
        confidence=0.0,
        rationale="insufficient evidence to support or contradict",
    )


class ReVerifier:
    """Independent post-correction verifier (the ReVerifier stage).

    Inject ``claim_extractor`` to substitute a smarter re-extractor (e.g. an LLM
    claim splitter); the default is deterministic and offline so tests and the
    pipeline never require a live API key or network.
    """

    def __init__(
        self,
        claim_extractor: Optional[Callable[[str], List[str]]] = None,
    ):
        self._extract = claim_extractor or default_claim_extractor

    def reverify(
        self,
        payload: JudgeVerificationPayload,
        plan: CorrectionPlan,
        corrected_answer: str,
        judge_result: Optional[JudgeVerificationResult] = None,
    ) -> ReverificationResult:
        """Re-extract claims from ``corrected_answer`` and verify each.

        Read-only over every input. ``plan`` and ``judge_result`` are part of the
        contract's input set (they explain what the Corrector believed it did),
        but the verdict is derived from the corrected TEXT and the evidence, not
        from the Corrector's own claims about its output.
        """
        supporting = [_Passage(ev.passageText) for ev in payload.supportingEvidence]
        contradictory = [
            _Passage(ev.passageText) for ev in payload.contradictionEvidence
        ]

        extracted = self._extract(corrected_answer)
        claims: List[ReverifiedClaim] = []
        for i, sentence in enumerate(extracted):
            verdict = _classify_claim(sentence, supporting, contradictory)
            verdict.claim_id = f"rc{i + 1}"
            claims.append(verdict)

        # A single CONTRADICTED claim fails the whole answer — a correction that
        # introduced a new false fact must not be emitted. UNKNOWN never fails:
        # insufficient evidence is not falsity.
        has_contradiction = any(c.verdict == CONTRADICTED for c in claims)
        status = "FAILED" if has_contradiction else "VERIFIED"

        overall = (
            sum(c.confidence for c in claims) / len(claims) if claims else 0.0
        )

        return ReverificationResult(
            status=status,
            claims=claims,
            overall_confidence=round(overall, 4),
        )
