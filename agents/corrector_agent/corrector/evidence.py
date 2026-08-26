"""Evidence binding for the Corrector Agent (Phase 2, Step 3).

BINDING AUTHORITY — where evidence for a target may come from:
    The canonical ``Evidence`` model carries NO claim linkage field (no
    ``claim_id``). The ONLY structural claim -> evidence association in
    ``orchestration/schemas.py`` is ``ClaimReport.evidence``. The top-level
    ``CorrectionRequest.trusted_evidence`` / ``contradictory_evidence`` pools are
    therefore *unattributed*: they say WHAT the Judge trusts, never FOR WHICH
    CLAIM.

    Consequently:
        * a target is bound ONLY to the evidence carried by its OWN authorized
          claims (``claim.evidence``);
        * the pools are used ONLY to classify evidence the claim already owns
          (supporting vs contradictory) — they NEVER supply evidence;
        * there is deliberately NO broad fallback to the full
          ``trusted_evidence`` list for a target that has none of its own.

    The "no broad fallback" rule is not just a convention here: it is enforced
    by an explicit post-condition (``_find_binding_violation``) that fails the
    whole plan safe if any bound evidence_id is not owned by the target's own
    claims.

CLASSIFICATION LADDER (first signal that applies wins):
    1. in trusted_evidence only        -> supporting     (trusted_pool)
    2. in contradictory_evidence only  -> contradictory  (contradictory_pool)
    3. in BOTH pools                   -> contradictory  (pool_conflict)
    4. in neither pool                 -> by entailment_label (entailment_label)
                                          entailment    -> supporting
                                          contradiction -> contradictory
                                          neutral/other -> neutral

    Pool membership outranks the NLI label on purpose: the pools are the
    Judge's explicit authority signal, while ``entailment_label`` describes the
    evidence's relation to the *hallucinated* claim. Evidence that refutes a
    hallucination is exactly what the Judge places in ``trusted_evidence``.

FAIL-SAFE RULES:
    * a target with no supporting evidence is SKIPPED with
      ``SkipReason.NO_USABLE_EVIDENCE`` — never grounded on borrowed or invented
      evidence (Step 3 requirement 7);
    * duplicate evidence_ids collapse to the first occurrence;
    * an evidence_id that resolves to conflicting roles within one target is
      demoted to contradictory (grounding authority is never granted under
      ambiguity);
    * evidence with a blank evidence_id is not bound (its provenance cannot be
      preserved) and is recorded in ``provenance_warnings``;
    * an unverified ``CorrectionPlan`` is never grounded.

EVIDENCE CONTENT IS DATA, NEVER INSTRUCTIONS:
    Snippets are copied verbatim — no rewriting, no truncation, no merging, no
    normalization. Text that reads like an injected instruction is FLAGGED
    (``BoundEvidence.suspected_instruction_text``) so the Step 4 prompt layer
    can fence it; it is never edited here.

This module performs NO model generation, NO validation/retry, and NO
reconstruction.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .config import CorrectorConfig
from .contracts import (
    BoundEvidence,
    CorrectionPlan,
    CorrectionTarget,
    EvidenceClassifier,
    EvidenceRole,
    GroundedPlan,
    GroundedTarget,
    InternalClaim,
    InternalCorrectionRequest,
    InternalEvidence,
    SkippedClaim,
    SkipReason,
)
from .targeting import build_correction_plan

__all__ = [
    "classify_evidence",
    "looks_like_instruction",
    "bind_evidence",
    "ground_request",
]

# Canonical EntailmentLabel values, plus tolerated synonyms from upstream
# producers. Anything unrecognized falls through to NEUTRAL (fail safe: an
# unknown label never grants grounding authority).
_SUPPORT_LABELS = frozenset({"entailment", "entails", "entailed", "support", "supports", "supported"})
_CONTRADICT_LABELS = frozenset({"contradiction", "contradicts", "contradicted", "refutes", "refuted"})

# Heuristic detector for prompt-injection-shaped evidence text. Detection only:
# the matched text is never altered, only flagged.
_INSTRUCTION_PATTERNS = re.compile(
    r"(?:"
    r"ignore\s+(?:all\s+|any\s+|the\s+)*(?:previous|prior|above|preceding|earlier)\s+"
    r"(?:instruction|prompt|direction|rule|message|context)"
    r"|disregard\s+(?:all\s+|any\s+|the\s+)*(?:previous|prior|above|preceding|earlier)\b"
    r"|forget\s+(?:everything|all\s+previous|what\s+you)\b"
    r"|(?:^|\n)\s*(?:system|assistant|user)\s*:"
    r"|<\s*/?\s*(?:system|instruction|prompt)s?\b"
    r"|\[\s*(?:system|instructions?|prompt)\s*\]"
    r"|new\s+instructions?\s*:"
    r"|you\s+(?:must|should|shall|are\s+to)\s+(?:now\s+)?"
    r"(?:instead\b|ignore\b|output\b|respond\b|reply\b|say\b|write\b)"
    r"|override\s+(?:the\s+)?(?:previous|prior|system)\b"
    r"|do\s+not\s+correct\s+(?:this|the|any)\b"
    r"|(?:reveal|print|repeat)\s+(?:your|the)\s+(?:system\s+)?prompt"
    r")",
    re.IGNORECASE,
)


def looks_like_instruction(text: str) -> bool:
    """True when ``text`` reads like an injected instruction rather than data.

    Advisory only. Nothing in this module changes evidence content based on the
    result; the flag exists so the prompt layer can quarantine the snippet.
    """
    if not text:
        return False
    return _INSTRUCTION_PATTERNS.search(text) is not None


def classify_evidence(
    evidence: InternalEvidence,
    trusted_ids: Set[str],
    contradictory_ids: Set[str],
) -> Tuple[EvidenceRole, EvidenceClassifier]:
    """Decide the role of ONE evidence item the claim already owns.

    The pools are consulted for MEMBERSHIP only. This function can never add an
    evidence item — it only labels one that was passed in.
    """
    eid = evidence.evidence_id
    in_trusted = eid in trusted_ids
    in_contradictory = eid in contradictory_ids

    if in_trusted and in_contradictory:
        # The Judge trusted and refuted the same passage. Never treat a
        # conflicted passage as authoritative grounding.
        return EvidenceRole.CONTRADICTORY, EvidenceClassifier.POOL_CONFLICT
    if in_trusted:
        return EvidenceRole.SUPPORTING, EvidenceClassifier.TRUSTED_POOL
    if in_contradictory:
        return EvidenceRole.CONTRADICTORY, EvidenceClassifier.CONTRADICTORY_POOL

    label = (evidence.entailment_label or "").strip().casefold()
    if label in _SUPPORT_LABELS:
        return EvidenceRole.SUPPORTING, EvidenceClassifier.ENTAILMENT_LABEL
    if label in _CONTRADICT_LABELS:
        return EvidenceRole.CONTRADICTORY, EvidenceClassifier.ENTAILMENT_LABEL
    return EvidenceRole.NEUTRAL, EvidenceClassifier.ENTAILMENT_LABEL


def _claims_by_id(claims: Sequence[InternalClaim]) -> Dict[str, List[InternalClaim]]:
    """Index authorized claims by id.

    A list per id (not a single claim) so that a duplicated claim_id carrying
    different evidence never loses evidence to a dict overwrite.
    """
    index: Dict[str, List[InternalClaim]] = {}
    for claim in claims:
        index.setdefault(claim.claim_id, []).append(claim)
    return index


def _unique_claim_ids(claim_ids: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for cid in claim_ids:
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered


def _owned_evidence_ids(
    target: CorrectionTarget, index: Dict[str, List[InternalClaim]]
) -> Set[str]:
    """Every evidence_id reachable from the target's OWN authorized claims."""
    owned: Set[str] = set()
    for cid in _unique_claim_ids(target.claim_ids):
        for claim in index.get(cid, ()):
            for ev in claim.evidence:
                if ev.evidence_id:
                    owned.add(ev.evidence_id)
    return owned


def _bind_target(
    target: CorrectionTarget,
    index: Dict[str, List[InternalClaim]],
    trusted_ids: Set[str],
    contradictory_ids: Set[str],
    warnings: List[str],
) -> GroundedTarget:
    """Bind ONE target to the evidence of its own claims. Adds nothing else."""
    ordered_ids: List[str] = []
    items: Dict[str, InternalEvidence] = {}
    roles: Dict[str, Tuple[EvidenceRole, EvidenceClassifier]] = {}

    for cid in _unique_claim_ids(target.claim_ids):
        claims = index.get(cid)
        if not claims:
            warnings.append(
                f"{target.sentence_id}/{cid}: authorized claim not found in "
                f"claims_to_correct; no evidence bound for it"
            )
            continue

        for claim in claims:
            for ev in claim.evidence:
                eid = ev.evidence_id
                if not eid or not eid.strip():
                    warnings.append(
                        f"{target.sentence_id}/{cid}: evidence with blank "
                        f"evidence_id dropped (provenance cannot be preserved)"
                    )
                    continue

                role, classifier = classify_evidence(ev, trusted_ids, contradictory_ids)

                if eid not in items:
                    # Verbatim reference — content is never edited or merged.
                    items[eid] = ev
                    roles[eid] = (role, classifier)
                    ordered_ids.append(eid)
                    continue

                previous_role = roles[eid][0]
                if previous_role != role:
                    # Same passage, incompatible roles across this target's
                    # claims. Demote: never grant grounding authority under
                    # ambiguity.
                    roles[eid] = (
                        EvidenceRole.CONTRADICTORY,
                        EvidenceClassifier.ROLE_CONFLICT,
                    )
                    warnings.append(
                        f"{target.sentence_id}: evidence '{eid}' resolved to both "
                        f"'{previous_role.value}' and '{role.value}'; demoted to "
                        f"'{EvidenceRole.CONTRADICTORY.value}'"
                    )

    supporting: List[BoundEvidence] = []
    contradictory: List[BoundEvidence] = []
    neutral: List[BoundEvidence] = []

    for eid in ordered_ids:
        evidence = items[eid]
        role, classifier = roles[eid]
        bound = BoundEvidence(
            evidence=evidence,
            role=role.value,
            classified_by=classifier.value,
            suspected_instruction_text=looks_like_instruction(evidence.snippet),
        )
        if role is EvidenceRole.SUPPORTING:
            supporting.append(bound)
        elif role is EvidenceRole.CONTRADICTORY:
            contradictory.append(bound)
        else:
            neutral.append(bound)

    return GroundedTarget(
        target=target,
        supporting=supporting,
        contradictory=contradictory,
        neutral=neutral,
    )


def _no_evidence_skips(
    target: CorrectionTarget, grounded: GroundedTarget
) -> List[SkippedClaim]:
    """One skip record per authorized claim on an ungroundable target."""
    detail = (
        f"{target.sentence_id}: no supporting evidence "
        f"(contradictory={len(grounded.contradictory)}, "
        f"neutral={len(grounded.neutral)}); refusing to borrow or invent evidence"
    )
    texts = list(target.claim_texts)
    skips: List[SkippedClaim] = []
    for i, cid in enumerate(target.claim_ids):
        skips.append(
            SkippedClaim(
                claim_id=cid,
                claim_text=texts[i] if i < len(texts) else "",
                reason=SkipReason.NO_USABLE_EVIDENCE.value,
                detail=detail,
            )
        )
    return skips


def _find_binding_violation(
    original: str,
    grounded_targets: Sequence[GroundedTarget],
    index: Dict[str, List[InternalClaim]],
) -> Optional[str]:
    """Post-condition check. Returns a reason string on violation, else ``None``.

    This is the machine-checked proof of the two Step 3 invariants:
        1. no bound evidence originates outside the target's own claims
           (i.e. no broad ``trusted_evidence`` fallback ever happened);
        2. the target offsets still reproduce the original text byte for byte.

    Implemented as a real check, not ``assert`` — ``python -O`` strips asserts
    and would erase a safety-critical guarantee.
    """
    for grounded in grounded_targets:
        target = grounded.target
        if original[target.start_offset:target.end_offset] != target.original_text:
            return (
                f"{target.sentence_id}: target offsets do not reproduce "
                f"original_text"
            )
        owned = _owned_evidence_ids(target, index)
        for eid in grounded.all_evidence_ids:
            if eid not in owned:
                return (
                    f"{target.sentence_id}: bound evidence '{eid}' is not owned by "
                    f"the target's own claims (broad-evidence fallback detected)"
                )
        seen: Set[str] = set()
        for eid in grounded.all_evidence_ids:
            if eid in seen:
                return f"{target.sentence_id}: evidence '{eid}' bound more than once"
            seen.add(eid)
    return None


def _failed_plan(
    plan: CorrectionPlan,
    extra_skips: Sequence[SkippedClaim] = (),
    warnings: Sequence[str] = (),
) -> GroundedPlan:
    """Fail-safe result: zero grounded targets, nothing lost from the record."""
    return GroundedPlan(
        spans=plan.spans,
        grounded_targets=[],
        locked_sentence_ids=plan.locked_sentence_ids,
        skipped_claims=list(plan.skipped_claims) + list(extra_skips),
        unlocated_preserved_claims=plan.unlocated_preserved_claims,
        provenance_warnings=list(warnings),
        integrity_verified=False,
    )


def bind_evidence(
    request: InternalCorrectionRequest,
    plan: CorrectionPlan,
) -> GroundedPlan:
    """Bind every authorized target to ONLY its own claims' evidence.

    Targets that end up without usable supporting evidence are moved into
    ``skipped_claims`` with ``no_usable_evidence`` so a later step can map them
    to the appropriate canonical status. They are never grounded on unrelated
    evidence and never dropped silently.
    """
    if not plan.integrity_verified:
        # Step 2 already failed safe; never ground an unverified plan.
        return _failed_plan(
            plan,
            warnings=["correction plan failed integrity verification; nothing bound"],
        )

    index = _claims_by_id(request.claims_to_correct)
    trusted_ids = {e.evidence_id for e in request.trusted_evidence if e.evidence_id}
    contradictory_ids = {
        e.evidence_id for e in request.contradictory_evidence if e.evidence_id
    }

    warnings: List[str] = []
    grounded_targets: List[GroundedTarget] = []
    skipped: List[SkippedClaim] = list(plan.skipped_claims)

    for target in plan.targets:
        grounded = _bind_target(target, index, trusted_ids, contradictory_ids, warnings)
        if grounded.has_usable_support:
            grounded_targets.append(grounded)
        else:
            skipped.extend(_no_evidence_skips(target, grounded))

    violation = _find_binding_violation(
        request.original_response, grounded_targets, index
    )
    if violation is not None:
        # A logic error here would mean handing unowned evidence or a bad offset
        # to generation. Discard every target rather than proceed.
        collapsed = [
            SkippedClaim(
                claim_id=cid,
                claim_text="",
                reason=SkipReason.SPAN_INTEGRITY_FAILED.value,
                detail=f"{violation}; discarding all grounded targets",
            )
            for g in grounded_targets
            for cid in g.target.claim_ids
        ]
        return _failed_plan(plan, extra_skips=collapsed, warnings=warnings + [violation])

    return GroundedPlan(
        spans=plan.spans,
        grounded_targets=grounded_targets,
        locked_sentence_ids=plan.locked_sentence_ids,
        skipped_claims=skipped,
        unlocated_preserved_claims=plan.unlocated_preserved_claims,
        provenance_warnings=warnings,
        integrity_verified=True,
    )


def ground_request(
    request: InternalCorrectionRequest,
    config: Optional[CorrectorConfig] = None,
) -> GroundedPlan:
    """Convenience: Step 2 targeting followed by Step 3 evidence binding."""
    plan = build_correction_plan(request, config)
    return bind_evidence(request, plan)
