"""Authorization + claim targeting for the Corrector Agent (Phase 2, Step 2).

AUTHORITY MODEL — the Corrector is NOT the Judge:
    * ``claims_to_correct`` is the ONLY correction authorization.
    * ``claims_to_preserve`` is LOCKED content and can never become a target.
    * No target is ever invented. A claim that cannot be located unambiguously
      inside the original response is SKIPPED, never guessed at.

MATCHING LADDER (strictest first; the first level that yields a UNIQUE match wins):
    1. EXACT         — claim text equals span text verbatim
    2. NORMALIZED    — equal after unicode/quote fold, casefold, whitespace collapse
    3. CONTAINMENT   — normalized claim occurs inside exactly one span
    4. TOKEN_OVERLAP — last resort, allowed ONLY when the best candidate clears
                       ``token_overlap_threshold`` AND beats the runner-up by
                       ``token_overlap_margin`` AND the claim has at least
                       ``min_claim_tokens_for_overlap`` tokens

FAIL-SAFE RULES:
    * ambiguous at any level (two or more equally good candidates) -> skip
    * not found                                                    -> skip
    * span locked by a preserved claim                             -> skip
    * claim_id present in both authorization lists                 -> skip
    * span integrity violation                                     -> skip everything

There is deliberately NO first-sentence fallback and no "closest guess". This
module performs NO evidence binding, NO model generation, NO validation, and NO
reconstruction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from .config import CorrectorConfig
from .contracts import (
    CorrectionPlan,
    CorrectionTarget,
    InternalClaim,
    InternalCorrectionRequest,
    MatchStrategy,
    SentenceSpan,
    SkippedClaim,
    SkipReason,
)
from .sentence_utils import (
    extract_sentence_spans,
    find_span_integrity_error,
    normalize_whitespace_case,
    token_coverage,
    tokenize,
)

__all__ = ["ClaimMatch", "match_claim_to_span", "build_correction_plan"]


class ClaimMatch:
    """Result of locating one claim: either a unique span, or a skip reason."""

    __slots__ = ("span", "strategy", "score", "skip_reason", "detail")

    def __init__(
        self,
        span: Optional[SentenceSpan] = None,
        strategy: Optional[MatchStrategy] = None,
        score: float = 0.0,
        skip_reason: Optional[SkipReason] = None,
        detail: str = "",
    ) -> None:
        self.span = span
        self.strategy = strategy
        self.score = score
        self.skip_reason = skip_reason
        self.detail = detail

    @property
    def matched(self) -> bool:
        """Return True if a sentence span was matched for this claim."""
        return self.span is not None


def _unique(candidates: Sequence[SentenceSpan]) -> Tuple[Optional[SentenceSpan], bool]:
    """Return (span, ambiguous). A single candidate is unique; 2+ is ambiguous."""
    if not candidates:
        return None, False
    if len(candidates) == 1:
        return candidates[0], False
    return None, True


def _ambiguous(level: str, candidates: Sequence[SentenceSpan]) -> ClaimMatch:
    ids = ", ".join(c.sentence_id for c in candidates)
    return ClaimMatch(
        skip_reason=SkipReason.AMBIGUOUS_MATCH,
        detail=f"{level} matched {len(candidates)} spans ({ids}); refusing to guess",
    )


def match_claim_to_span(
    claim_text: str,
    spans: Sequence[SentenceSpan],
    config: Optional[CorrectorConfig] = None,
) -> ClaimMatch:
    """Locate ``claim_text`` in ``spans`` using the strict ladder, or fail safe."""
    cfg = config or CorrectorConfig()

    if not claim_text or not claim_text.strip():
        return ClaimMatch(
            skip_reason=SkipReason.EMPTY_CLAIM_TEXT, detail="claim text is empty"
        )
    if not spans:
        return ClaimMatch(
            skip_reason=SkipReason.NO_SENTENCE_SPANS,
            detail="original response produced no sentence spans",
        )

    # --- 1. EXACT -----------------------------------------------------------
    exact = [s for s in spans if s.text == claim_text]
    span, ambiguous = _unique(exact)
    if span is not None:
        return ClaimMatch(span, MatchStrategy.EXACT, 1.0)
    if ambiguous:
        return _ambiguous("exact", exact)

    # --- 2. NORMALIZED ------------------------------------------------------
    norm_claim = normalize_whitespace_case(claim_text)
    if not norm_claim:
        return ClaimMatch(
            skip_reason=SkipReason.EMPTY_CLAIM_TEXT,
            detail="claim text normalizes to empty",
        )

    norm_spans = [(s, normalize_whitespace_case(s.text)) for s in spans]
    normalized = [s for s, ns in norm_spans if ns == norm_claim]
    span, ambiguous = _unique(normalized)
    if span is not None:
        return ClaimMatch(span, MatchStrategy.NORMALIZED, 1.0)
    if ambiguous:
        return _ambiguous("normalized", normalized)

    # --- 3. CONTAINMENT (claim inside span only, never the reverse) ---------
    # The reverse direction (span inside claim) is intentionally NOT accepted:
    # it would let a long claim capture an unrelated short sentence.
    contained = [s for s, ns in norm_spans if norm_claim in ns]
    span, ambiguous = _unique(contained)
    if span is not None:
        return ClaimMatch(span, MatchStrategy.CONTAINMENT, 1.0)
    if ambiguous:
        return _ambiguous("containment", contained)

    # --- 4. TOKEN_OVERLAP (guarded last resort) ----------------------------
    claim_tokens = set(tokenize(claim_text))
    if len(claim_tokens) < cfg.min_claim_tokens_for_overlap:
        return ClaimMatch(
            skip_reason=SkipReason.CLAIM_NOT_FOUND,
            detail=(
                f"claim has {len(claim_tokens)} distinct token(s); minimum "
                f"{cfg.min_claim_tokens_for_overlap} required for overlap matching"
            ),
        )

    scored: List[Tuple[float, SentenceSpan]] = []
    for s in spans:
        score = token_coverage(claim_tokens, set(tokenize(s.text)))
        if score > 0.0:
            scored.append((score, s))

    if not scored:
        return ClaimMatch(
            skip_reason=SkipReason.CLAIM_NOT_FOUND,
            detail="no sentence shares any token with the claim",
        )

    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best_span = scored[0]

    if best_score < cfg.token_overlap_threshold:
        return ClaimMatch(
            skip_reason=SkipReason.CLAIM_NOT_FOUND,
            detail=(
                f"best token coverage {best_score:.2f} is below threshold "
                f"{cfg.token_overlap_threshold:.2f}"
            ),
        )

    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if best_score - runner_up < cfg.token_overlap_margin:
        return ClaimMatch(
            skip_reason=SkipReason.AMBIGUOUS_MATCH,
            detail=(
                f"token coverage {best_score:.2f} vs runner-up {runner_up:.2f} "
                f"(margin {cfg.token_overlap_margin:.2f} not met); refusing to guess"
            ),
        )

    return ClaimMatch(best_span, MatchStrategy.TOKEN_OVERLAP, best_score)


def _locate_preserved(
    claims: Sequence[InternalClaim],
    spans: Sequence[SentenceSpan],
    cfg: CorrectorConfig,
) -> Tuple[Set[str], List[SkippedClaim]]:
    """Resolve preserved claims to the sentence_ids that must stay locked.

    A preserved claim that cannot be located is REPORTED, not ignored: later
    steps must be able to see that a lock could not be positioned.
    """
    locked: Set[str] = set()
    unlocated: List[SkippedClaim] = []
    for claim in claims:
        match = match_claim_to_span(claim.claim_text, spans, cfg)
        if match.matched and match.span is not None:
            locked.add(match.span.sentence_id)
        else:
            reason = match.skip_reason or SkipReason.CLAIM_NOT_FOUND
            unlocated.append(
                SkippedClaim(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    reason=reason.value,
                    detail=match.detail,
                )
            )
    return locked, unlocated


def build_correction_plan(
    request: InternalCorrectionRequest,
    config: Optional[CorrectorConfig] = None,
) -> CorrectionPlan:
    """Produce the authorized, located, verified set of correction targets.

    Ordering matters: preserved claims are located FIRST so their spans are
    already locked before any authorized claim is considered.
    """
    cfg = config or CorrectorConfig()
    original = request.original_response
    spans = extract_sentence_spans(original)

    # Requirement 13 — verify byte accuracy before trusting any offset.
    integrity_error = find_span_integrity_error(original, spans)
    if integrity_error is not None:
        return CorrectionPlan(
            spans=spans,
            targets=[],
            locked_sentence_ids=[],
            skipped_claims=[
                SkippedClaim(
                    claim_id=c.claim_id,
                    claim_text=c.claim_text,
                    reason=SkipReason.SPAN_INTEGRITY_FAILED.value,
                    detail=integrity_error,
                )
                for c in request.claims_to_correct
            ],
            integrity_verified=False,
        )

    locked_ids, unlocated_preserved = _locate_preserved(
        request.claims_to_preserve, spans, cfg
    )
    preserved_claim_ids = {c.claim_id for c in request.claims_to_preserve}

    targets_by_sentence: Dict[str, CorrectionTarget] = {}
    ordered_sentence_ids: List[str] = []
    skipped: List[SkippedClaim] = []

    for claim in request.claims_to_correct:
        # A claim appearing in BOTH lists is a contradictory authorization.
        # Preservation wins; the Corrector never resolves Judge conflicts.
        if claim.claim_id in preserved_claim_ids:
            skipped.append(
                SkippedClaim(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    reason=SkipReason.CLAIM_ALSO_IN_PRESERVE_LIST.value,
                    detail="claim_id present in claims_to_preserve; preservation wins",
                )
            )
            continue

        match = match_claim_to_span(claim.claim_text, spans, cfg)
        if not match.matched or match.span is None:
            reason = match.skip_reason or SkipReason.CLAIM_NOT_FOUND
            skipped.append(
                SkippedClaim(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    reason=reason.value,
                    detail=match.detail,
                )
            )
            continue

        span = match.span
        if span.sentence_id in locked_ids:
            skipped.append(
                SkippedClaim(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    reason=SkipReason.SPAN_LOCKED_BY_PRESERVED_CLAIM.value,
                    detail=f"{span.sentence_id} is locked by a preserved claim",
                )
            )
            continue

        existing = targets_by_sentence.get(span.sentence_id)
        if existing is None:
            targets_by_sentence[span.sentence_id] = CorrectionTarget(
                sentence_id=span.sentence_id,
                claim_ids=[claim.claim_id],
                claim_texts=[claim.claim_text],
                original_text=span.text,
                start_offset=span.start_offset,
                end_offset=span.end_offset,
                match_strategy=(
                    match.strategy.value if match.strategy else MatchStrategy.EXACT.value
                ),
                match_score=match.score,
            )
            ordered_sentence_ids.append(span.sentence_id)
        else:
            # Several authorized claims on one sentence: merge, keeping the
            # weakest strategy/score so downstream sees the true confidence.
            existing.claim_ids.append(claim.claim_id)
            existing.claim_texts.append(claim.claim_text)
            if match.score < existing.match_score:
                existing.match_score = match.score
                existing.match_strategy = (
                    match.strategy.value if match.strategy else existing.match_strategy
                )

    targets = [targets_by_sentence[sid] for sid in ordered_sentence_ids]

    # Post-conditions, enforced with real checks (NOT `assert`, which `python -O`
    # strips): every emitted target must be byte-accurate and must not be locked.
    # A violation here is a logic error, so the whole plan fails safe to zero
    # targets rather than handing a bad offset to the reconstruction step.
    for target in targets:
        if original[target.start_offset:target.end_offset] != target.original_text:
            return CorrectionPlan(
                spans=spans,
                targets=[],
                locked_sentence_ids=sorted(locked_ids),
                skipped_claims=skipped
                + [
                    SkippedClaim(
                        claim_id=cid,
                        claim_text="",
                        reason=SkipReason.SPAN_INTEGRITY_FAILED.value,
                        detail=(
                            f"{target.sentence_id}: target offsets do not reproduce "
                            f"original_text; discarding all targets"
                        ),
                    )
                    for cid in target.claim_ids
                ],
                unlocated_preserved_claims=unlocated_preserved,
                integrity_verified=False,
            )
        if target.sentence_id in locked_ids:  # pragma: no cover - unreachable guard
            return CorrectionPlan(
                spans=spans,
                targets=[],
                locked_sentence_ids=sorted(locked_ids),
                skipped_claims=skipped
                + [
                    SkippedClaim(
                        claim_id=cid,
                        claim_text="",
                        reason=SkipReason.SPAN_LOCKED_BY_PRESERVED_CLAIM.value,
                        detail=f"{target.sentence_id} locked; discarding all targets",
                    )
                    for cid in target.claim_ids
                ],
                unlocated_preserved_claims=unlocated_preserved,
                integrity_verified=False,
            )

    return CorrectionPlan(
        spans=spans,
        targets=targets,
        locked_sentence_ids=sorted(locked_ids),
        skipped_claims=skipped,
        unlocated_preserved_claims=unlocated_preserved,
        integrity_verified=True,
    )
