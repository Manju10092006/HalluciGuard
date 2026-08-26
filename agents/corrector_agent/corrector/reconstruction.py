"""Step 6 — deterministic, fail-closed response reconstruction.

This is the ONLY component that turns accepted, validated corrections back into a
whole response. It never regenerates the response and never repairs a mismatch:
it splices accepted sentence text into the ORIGINAL response at byte offsets and
proves — before returning anything — that every byte outside the accepted spans
is identical to the original.

THE ONLY LEGITIMATE PATH INTO A CORRECTED RESPONSE
--------------------------------------------------
    model output -> CorrectionCandidate -> Step 5 validation
                 -> accepted TargetResolution -> reconstruction -> CorrectionResult

No text reaches the caller unless it came from an ACCEPTED
:class:`TargetResolution`. Reconstruction consumes:

* ``original_response`` — the verbatim response the Corrector was given;
* ``plan.spans`` / ``plan.grounded_targets`` — byte-accurate sentence offsets
  from Step 2/3 (``original_response[start:end] == span.text``, ordered,
  non-overlapping, inter-sentence whitespace excluded — see ``sentence_utils``);
* ``outcome.accepted`` — Step 5's accepted resolutions, matched to offsets by
  ``sentence_id``.

FAIL-CLOSED CONTRACT
--------------------
If ANY integrity check fails — an accepted target with no span, disagreeing
offset sources, a slice that is not byte-identical to the recorded original, an
overlap, an empty accepted text, or a control splice that does not reproduce the
original — this returns a FAILURE SENTINEL (``Reconstruction.ok is False``). It
NEVER returns a partially reconstructed response. The caller (Step 6 wiring) maps
that sentinel to ``build_reconstruction_fallback_result`` and preserves the
original response verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .contracts import (
    CorrectionTarget,
    GroundedPlan,
    SentenceSpan,
    TargetResolution,
    ValidationOutcome,
)
from .sentence_utils import find_span_integrity_error

__all__ = ["Reconstruction", "reconstruct"]


@dataclass(frozen=True)
class Reconstruction:
    """Result of a reconstruction attempt.

    ``ok`` is True only when every accepted correction was spliced AND the bytes
    outside the accepted spans were proven identical to the original. When ``ok``
    is False, ``text`` is empty and ``failure_reason`` names the first violated
    invariant — the caller must fall back to the original response, never to
    ``text``.
    """

    ok: bool
    text: str = ""
    changed_sentence_ids: Tuple[str, ...] = ()
    failure_reason: str = ""


def _fail(reason: str) -> Reconstruction:
    return Reconstruction(ok=False, failure_reason=reason)


def reconstruct(
    original_response: str,
    plan: GroundedPlan,
    outcome: ValidationOutcome,
) -> Reconstruction:
    """Splice accepted corrections into ``original_response`` by byte offset.

    Returns a successful :class:`Reconstruction` only when the splice is provably
    faithful: exactly the accepted sentence spans change and every other byte is
    preserved. Any inconsistency yields a failure sentinel (``ok=False``) so the
    caller preserves the original. This function performs NO repair and NO
    regeneration.
    """
    accepted = list(outcome.accepted)
    if not accepted:
        # correct() only calls reconstruct when there is at least one accepted
        # resolution; being asked to "reconstruct" nothing is a caller-logic
        # inconsistency, so fail closed rather than silently return the original
        # dressed as a correction.
        return _fail("no_accepted_resolutions")

    # The entire offset scheme is only trustworthy if the plan's spans still
    # satisfy the byte-accuracy invariant against THIS response. Verify first.
    if not plan.spans:
        return _fail("plan_has_no_spans")
    span_error = find_span_integrity_error(original_response, list(plan.spans))
    if span_error is not None:
        return _fail(f"span_integrity_failed: {span_error}")

    spans_by_id: Dict[str, SentenceSpan] = {}
    for span in plan.spans:
        # Duplicate ids would make an accepted match ambiguous; integrity check
        # above already rejects adjacent duplicates, this catches non-adjacent.
        if span.sentence_id in spans_by_id:
            return _fail(f"duplicate_span_id: {span.sentence_id}")
        spans_by_id[span.sentence_id] = span

    targets_by_id: Dict[str, CorrectionTarget] = {
        gt.target.sentence_id: gt.target for gt in plan.grounded_targets
    }
    locked = set(plan.locked_sentence_ids)

    edits: List[Tuple[int, int, str, str]] = []  # (start, end, new_text, sentence_id)
    for resolution in accepted:
        sid = resolution.sentence_id

        # An accepted target that is not a real, authorized, grounded target — or
        # that collides with a preserved (locked) sentence — is an authorization
        # breach. Refuse; never rewrite a sentence the plan did not authorize.
        target = targets_by_id.get(sid)
        if target is None:
            return _fail(f"accepted_target_not_grounded: {sid}")
        if sid in locked:
            return _fail(f"accepted_target_is_locked: {sid}")

        span = spans_by_id.get(sid)
        if span is None:
            return _fail(f"accepted_target_has_no_span: {sid}")

        # The two independent offset sources (Step 2 spans and the Step 3 target)
        # must agree exactly, or the offsets cannot be trusted.
        if (span.start_offset, span.end_offset) != (
            target.start_offset,
            target.end_offset,
        ):
            return _fail(f"offset_sources_disagree: {sid}")

        start, end = span.start_offset, span.end_offset
        if not (0 <= start < end <= len(original_response)):
            return _fail(f"offsets_out_of_bounds: {sid}")

        slice_text = original_response[start:end]
        # Redundant with the span-integrity check, but reconstruction re-proves the
        # bytes it is about to replace against BOTH recorded originals.
        if slice_text != span.text:
            return _fail(f"slice_mismatch_span: {sid}")
        if slice_text != target.original_text:
            return _fail(f"slice_mismatch_target: {sid}")
        if slice_text != resolution.original_text:
            return _fail(f"slice_mismatch_resolution: {sid}")

        # Step 5 guarantees accepted_text is non-empty for an accepted decision;
        # verify rather than trust, so an empty splice can never occur.
        new_text = resolution.accepted_text
        if not new_text:
            return _fail(f"empty_accepted_text: {sid}")

        edits.append((start, end, new_text, sid))

    # Order by start offset and prove the accepted spans are strictly
    # non-overlapping. Sentence spans are non-overlapping by construction; a
    # violation here means the inputs are inconsistent, so fail closed.
    edits.sort(key=lambda e: e[0])
    for prev, curr in zip(edits, edits[1:]):
        if curr[0] < prev[1]:
            return _fail(f"overlapping_accepted_spans: {prev[3]} / {curr[3]}")

    # Build the corrected text and, in lock-step, a CONTROL splice that reinserts
    # each span's ORIGINAL text. If the control reproduces the response
    # byte-for-byte, the offset partition is proven faithful: every byte outside
    # the accepted spans is preserved, so substituting accepted_text changes ONLY
    # those spans. This is the core, non-circular fail-closed proof.
    corrected_parts: List[str] = []
    control_parts: List[str] = []
    cursor = 0
    for start, end, new_text, _sid in edits:
        gap = original_response[cursor:start]
        corrected_parts.append(gap)
        control_parts.append(gap)
        corrected_parts.append(new_text)
        control_parts.append(original_response[start:end])
        cursor = end
    tail = original_response[cursor:]
    corrected_parts.append(tail)
    control_parts.append(tail)

    if "".join(control_parts) != original_response:
        return _fail("control_splice_does_not_reproduce_original")

    return Reconstruction(
        ok=True,
        text="".join(corrected_parts),
        changed_sentence_ids=tuple(sid for _s, _e, _t, sid in edits),
    )
