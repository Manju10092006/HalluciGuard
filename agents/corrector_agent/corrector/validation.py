"""Step 5 — the validation pipeline: order, fail-closed wrapping, entailment seam.

WHAT THIS MODULE OWNS
---------------------
:mod:`validators` holds ten independent gates that deliberately know nothing
about each other. This module owns everything they leave undecided:

* **ORDER** — cheapest first, and nothing semantic before authorization is
  settled (see :data:`GATE_ORDER`).
* **SHORT-CIRCUITING** — gates 1–3 stop the pipeline, so a candidate aimed at an
  unauthorized sentence is never measured against any evidence.
* **FAIL-CLOSED WRAPPING** — every gate call is wrapped, so "a gate never raises"
  is *enforced* here rather than merely intended there. Any unexpected exception
  becomes ``VALIDATION_ERROR`` and rejects the candidate. The wrapper is not
  redundant with the gates' totality; it is what makes totality non-negotiable.
* **CONVERSION** — pure :class:`validators.Finding` values become pydantic
  ``ValidationFailure`` models. Keeping that conversion here is what lets the
  safety-critical gate layer stay importable with nothing installed.
* **THE ENTAILMENT SEAM** — an optional injected scorer, additive only.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not generate, retry, splice, reverify, or repair. It reaches a verdict on
ONE candidate for ONE target and stops. Bounded retry is :mod:`retry`; splicing
the accepted sentence back into the response is Step 6.

ABSTENTION (approved policy)
----------------------------
A legitimate abstention ("Current evidence is insufficient to support this
claim.") is recognised as valid safe behaviour: it bypasses minimal-edit
rejection and is not treated as unsupported hallucinated content. It is NOT a
factual correction, so ``passed`` stays False and the target routes to
``TargetDecision.UNRESOLVED``. No new supervisor-level status is introduced —
``UNRESOLVED`` already exists in the internal decision enum and maps onto the
existing canonical builders in :mod:`adapter` at Step 6.

ACCEPTANCE
----------
Conjunctive, never a majority vote or a score. ``passed`` requires an empty
failure list AND every recorded per-dimension signal to be satisfied
independently. A gate that returned no finding but left its dimension
unsatisfied still blocks acceptance — the two conditions are checked separately
on purpose, so that a future gate whose signal and findings disagree fails
closed rather than open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from .config import CorrectorConfig
from .contracts import (
    CandidateValidation,
    CorrectionCandidate,
    GroundedPlan,
    GroundedTarget,
    PromptBundle,
    ValidationFailure,
)
from .failure_codes import ValidationFailureCode as Code
from .validators import (
    EvidenceView,
    Finding,
    alignment_score,
    check_authorization,
    check_contradiction,
    check_dates,
    check_entities,
    check_evidence_grounding,
    check_minimal_edit,
    check_numbers,
    check_original_preservation,
    check_structure,
    check_target_identity,
    check_unsupported_addition,
    detect_abstention,
    edit_similarity,
)

__all__ = [
    "GATE_ORDER",
    "SHORT_CIRCUIT_GATES",
    "EntailmentJudgement",
    "EntailmentScorer",
    "NullEntailmentScorer",
    "ValidationContext",
    "build_evidence_pairs",
    "build_context",
    "build_contexts",
    "validate_candidate",
]


# The pipeline's order of operations, named so tests can assert it has not
# silently changed. Order is a safety property here, not a style choice.
GATE_ORDER: Tuple[str, ...] = (
    "structure",             # 1  shape only
    "authorization",         # 2  may this sentence be touched at all
    "target_identity",       # 3  is this the sentence we asked about
    "original_preservation",  # 4  no other span, no whole-response regeneration
    "numbers",               # 5  value-level
    "dates",                 # 6  component-normalized
    "entities",              # 7  proper-noun set diff
    "unsupported_addition",  # 8  candidate \ original must be grounded
    "evidence_grounding",    # 9  is the sentence built from the evidence
    "contradiction",         # 10 echo ladder + retained unsupported values
    "minimal_edit",          # 11 repair vs rewrite (soft, retryable)
    "entailment",            # 12 optional, injected, additive only
)

# A candidate that fails any of these is never measured against evidence: it is
# malformed, unauthorized, or about a different sentence entirely.
SHORT_CIRCUIT_GATES: Tuple[str, ...] = (
    "structure",
    "authorization",
    "target_identity",
)


# ---------------------------------------------------------------------------
# The entailment seam
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntailmentJudgement:
    """An injected scorer's verdict on "does the evidence entail this sentence?".

    The scorer owns its own calibration. The Corrector deliberately does not hold
    an entailment threshold: thresholds belong with the model that produced the
    score, and inventing one here would be a second, unvalidated calibration.
    """

    entailed: bool
    score: float = 0.0
    label: str = ""
    detail: str = ""


class EntailmentScorer(Protocol):
    """Optional NLI seam. Injected at the call site, never imported.

    The Corrector must not import ``agents.verifier_agent.*``: that package uses
    verifier-rooted absolute imports which only resolve under the root
    ``pytest.ini`` pythonpath, and it pulls ``torch`` at import time. Injection
    keeps the dependency out of the Corrector's import graph entirely.

    Returning ``None`` means UNAVAILABLE — explicitly not "entailed".
    """

    def score(self, premise: str, hypothesis: str) -> Optional[EntailmentJudgement]:
        """Score entailment between premise and hypothesis, returning None if unavailable."""
        ...


class NullEntailmentScorer:
    """The default scorer: always unavailable.

    ``None`` is returned rather than an "entailed" verdict so that an absent ML
    stack can never be mistaken for a passing entailment check. Gates 1–11 alone
    catch the recorded echo failure, so the safety guarantee never depends on
    this seam being filled.
    """

    def score(self, premise: str, hypothesis: str) -> Optional[EntailmentJudgement]:
        """Always return None to indicate entailment scoring is unavailable."""
        return None


# ---------------------------------------------------------------------------
# Per-target context
# ---------------------------------------------------------------------------

def build_evidence_pairs(
    target: GroundedTarget,
) -> Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...]]:
    """Split a grounded target into ``(supporting, contradictory)`` id/text pairs.

    GROUNDING RULE (inherited from Steps 3 and 4): only SUPPORTING evidence
    grounds a correction. Contradictory evidence is a contradiction signal and
    neutral evidence is neither — merging them into one pool is the specific bug
    that lets a verbatim echo score as "grounded", and it is not repeated here.

    Items with a blank snippet are dropped: they cannot ground anything, and
    keeping their id would overstate the grounding pool.
    """
    supporting = tuple(
        (str(bound.evidence_id), bound.evidence.snippet)
        for bound in target.supporting
        if (bound.evidence.snippet or "").strip()
    )
    contradictory = tuple(
        (str(bound.evidence_id), bound.evidence.snippet)
        for bound in target.contradictory
        if (bound.evidence.snippet or "").strip()
    )
    return supporting, contradictory


@dataclass(frozen=True)
class ValidationContext:
    """Everything one target's candidates are validated against.

    Built once per target and reused across retry attempts, so every attempt is
    judged against byte-identical inputs. Immutable by construction: a gate or a
    scorer cannot widen the grounding pool or the authorization set.
    """

    sentence_id: str
    original_text: str = ""
    claim_ids: Tuple[str, ...] = ()
    claim_texts: Tuple[str, ...] = ()
    query_text: str = ""
    authorized_sentence_ids: Tuple[str, ...] = ()
    authorized_claim_ids: Tuple[str, ...] = ()
    locked_sentence_ids: Tuple[str, ...] = ()
    other_span_texts: Tuple[str, ...] = ()
    supporting_pairs: Tuple[Tuple[str, str], ...] = ()
    contradictory_pairs: Tuple[Tuple[str, str], ...] = ()
    rendered_evidence_ids: Tuple[str, ...] = ()

    def evidence_view(
        self, evidence_ids: Optional[Sequence[str]] = None
    ) -> EvidenceView:
        """The grounding pool, optionally narrowed to the ids actually rendered.

        The Corrector must not validate against evidence the model never saw:
        that would credit the model for a fact it had no access to. When
        ``evidence_ids`` is empty or None the full supporting pool is used;
        otherwise the pool is intersected with it.

        ``evidence_ids`` is safe to trust because Step 4's ``output_parser``
        populates ``CorrectionCandidate.evidence_ids_used`` from
        ``PromptBundle.evidence_ids`` — it is the render manifest, not model
        output, so a model cannot use it to widen its own grounding pool.
        """
        wanted = {str(eid) for eid in (evidence_ids or ())}
        if wanted:
            supporting = tuple(p for p in self.supporting_pairs if p[0] in wanted)
        else:
            supporting = self.supporting_pairs
        return EvidenceView.from_texts(
            supporting=supporting, contradictory=self.contradictory_pairs
        )


def build_context(
    *,
    target: GroundedTarget,
    plan: Optional[GroundedPlan] = None,
    bundle: Optional[PromptBundle] = None,
    query_text: str = "",
) -> ValidationContext:
    """Assemble a target's :class:`ValidationContext` from Step 2–4 outputs.

    ``authorized_sentence_ids`` comes from the PLAN, not from the target: gate 2
    must be able to tell "this id belongs to another target" apart from "this id
    exists nowhere", and only the plan knows the full authorized set. With no
    plan supplied the authorized set is this target alone, which is the strict
    direction (any other id fails authorization).
    """
    supporting, contradictory = build_evidence_pairs(target)

    if plan is not None:
        authorized_sentence_ids = tuple(
            str(grounded.sentence_id) for grounded in plan.grounded_targets
        )
        locked = tuple(str(sid) for sid in plan.locked_sentence_ids)
        other_spans = tuple(
            span.text
            for span in plan.spans
            if str(span.sentence_id) != str(target.sentence_id)
        )
    else:
        authorized_sentence_ids = (str(target.sentence_id),)
        locked = ()
        other_spans = ()

    rendered = tuple(str(eid) for eid in (bundle.evidence_ids if bundle else ()))

    return ValidationContext(
        sentence_id=str(target.sentence_id),
        original_text=target.target.original_text,
        claim_ids=tuple(str(cid) for cid in target.target.claim_ids),
        claim_texts=tuple(target.target.claim_texts),
        query_text=query_text,
        authorized_sentence_ids=authorized_sentence_ids,
        authorized_claim_ids=tuple(str(cid) for cid in target.target.claim_ids),
        locked_sentence_ids=locked,
        other_span_texts=other_spans,
        supporting_pairs=supporting,
        contradictory_pairs=contradictory,
        rendered_evidence_ids=rendered,
    )


def build_contexts(
    *,
    plan: GroundedPlan,
    bundles: Sequence[PromptBundle] = (),
    query_text: str = "",
) -> Dict[str, ValidationContext]:
    """One context per grounded target, keyed by ``sentence_id``."""
    by_sentence = {
        str(bundle.sentence_id): bundle for bundle in bundles if bundle.build_ok
    }
    return {
        str(target.sentence_id): build_context(
            target=target,
            plan=plan,
            bundle=by_sentence.get(str(target.sentence_id)),
            query_text=query_text,
        )
        for target in plan.grounded_targets
    }


# ---------------------------------------------------------------------------
# Fail-closed gate execution
# ---------------------------------------------------------------------------

@dataclass
class _GateRecord:
    """What one gate did: whether it ran, and what it found."""

    ran: bool = False
    findings: Tuple[Finding, ...] = ()

    @property
    def clean(self) -> bool:
        """True only when the gate RAN and found nothing.

        A gate that never ran is not evidence of safety, so ``clean`` is False
        for a skipped gate. Every dimension therefore defaults to unsatisfied.
        """
        return self.ran and not self.findings


def _validation_error(gate: str, exc: BaseException) -> Finding:
    """Convert an unexpected exception into a terminal, explainable failure.

    The exception type and message are recorded; no traceback and no candidate
    text are copied into the detail, so a crash can never become a channel for
    model-authored text.
    """
    return Finding(
        Code.VALIDATION_ERROR.value,
        f"validation gate {gate!r} raised and the candidate was rejected",
        {"gate": gate, "error_type": type(exc).__name__, "error": str(exc)[:200]},
        retryable=False,
    )


def _run(record: Dict[str, _GateRecord], gate: str, fn, **kwargs) -> _GateRecord:
    """Run one gate, guaranteeing it neither raises nor fails open."""
    try:
        findings = tuple(fn(**kwargs) or ())
    except Exception as exc:  # noqa: BLE001 — deliberate: fail closed on anything
        findings = (_validation_error(gate, exc),)
    entry = _GateRecord(ran=True, findings=findings)
    record[gate] = entry
    return entry


def _measure(fn, **kwargs) -> float:
    """Compute a diagnostic number without ever letting it break validation."""
    try:
        return float(fn(**kwargs))
    except Exception:  # noqa: BLE001 — a diagnostic must never reject a candidate
        return 0.0


def _as_float(value) -> float:
    """Coerce an injected scorer's number for the record, defaulting to 0.0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_failure(finding: Finding) -> ValidationFailure:
    """Pure :class:`Finding` -> pydantic ``ValidationFailure``."""
    return ValidationFailure(
        code=finding.code,
        reason=finding.reason,
        detail=dict(finding.detail or {}),
        retryable=bool(finding.retryable),
    )


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

def validate_candidate(
    *,
    candidate: CorrectionCandidate,
    context: ValidationContext,
    config: Optional[CorrectorConfig] = None,
    scorer: Optional[EntailmentScorer] = None,
) -> CandidateValidation:
    """Validate ONE candidate against ONE authorized, grounded target.

    Never raises: every gate is wrapped, every diagnostic is wrapped, and any
    unexpected fault becomes a terminal ``VALIDATION_ERROR`` that rejects the
    candidate. The returned verdict is the sole basis for the retry decision in
    :mod:`retry`; this function makes no generation call and has no side effect.
    """
    cfg = config or CorrectorConfig()
    records: Dict[str, _GateRecord] = {gate: _GateRecord() for gate in GATE_ORDER}
    evidence = context.evidence_view(candidate.evidence_ids_used)
    candidate_text = candidate.corrected_text
    original_text = context.original_text

    # --- Gate 1: structure (short-circuits) --------------------------------
    if not _run(
        records,
        "structure",
        check_structure,
        sentence_id=candidate.sentence_id,
        candidate_text=candidate_text,
        is_proposed=candidate.is_proposed,
        status=candidate.status,
        enforce_single_sentence=cfg.enforce_single_sentence,
    ).clean:
        return _assemble(candidate, context, cfg, records, evidence, None)

    # --- Gate 2: authorization (short-circuits) ---------------------------
    if not _run(
        records,
        "authorization",
        check_authorization,
        sentence_id=candidate.sentence_id,
        claim_ids=candidate.claim_ids,
        authorized_sentence_ids=context.authorized_sentence_ids,
        authorized_claim_ids=context.authorized_claim_ids,
        locked_sentence_ids=context.locked_sentence_ids,
    ).clean:
        return _assemble(candidate, context, cfg, records, evidence, None)

    # --- Gate 3: target identity (short-circuits) -------------------------
    identity = _run(
        records,
        "target_identity",
        check_target_identity,
        candidate_sentence_id=candidate.sentence_id,
        expected_sentence_id=context.sentence_id,
    )
    # Pipeline invariant, checked here rather than in a gate because it compares a
    # candidate FIELD against the context rather than judging the text: the
    # original the model was shown must be the original we are correcting. A
    # mismatch means the candidate belongs to a different plan or a stale span,
    # which no retry can repair.
    if identity.clean and candidate.original_text and original_text:
        if candidate.original_text != original_text:
            records["target_identity"] = _GateRecord(
                ran=True,
                findings=(
                    Finding(
                        Code.TARGET_MISMATCH.value,
                        "candidate carries a different original sentence than the target",
                        {
                            "expected_length": len(original_text),
                            "received_length": len(candidate.original_text),
                        },
                        retryable=False,
                    ),
                ),
            )
    if not records["target_identity"].clean:
        return _assemble(candidate, context, cfg, records, evidence, None)

    # --- Abstention classification ----------------------------------------
    # Deliberately AFTER gates 1-3: a malformed, unauthorized, or misdirected
    # candidate earns no policy exemption merely by containing the words
    # "evidence is insufficient".
    try:
        verdict = detect_abstention(candidate_text=candidate_text, evidence=evidence)
        is_abstention = bool(verdict.is_abstention)
    except Exception as exc:  # noqa: BLE001 — unclassifiable means "not exempt"
        records["structure"] = _GateRecord(
            ran=True,
            findings=records["structure"].findings
            + (_validation_error("detect_abstention", exc),),
        )
        return _assemble(candidate, context, cfg, records, evidence, None)

    # --- Gates 4-7: always run, abstention or not -------------------------
    # An abstention asserts no value, so these are naturally clean for a genuine
    # one; a "smuggled" abstention that does assert a value has already been
    # denied legitimacy by detect_abstention, so it gets no exemption here.
    _run(
        records,
        "original_preservation",
        check_original_preservation,
        candidate_text=candidate_text,
        other_span_texts=context.other_span_texts,
    )
    _run(
        records,
        "numbers",
        check_numbers,
        candidate_text=candidate_text,
        original_text=original_text,
        evidence=evidence,
    )
    _run(
        records,
        "dates",
        check_dates,
        candidate_text=candidate_text,
        original_text=original_text,
        evidence=evidence,
    )
    _run(
        records,
        "entities",
        check_entities,
        candidate_text=candidate_text,
        original_text=original_text,
        evidence=evidence,
    )

    # --- Gates 8-11: the policy-aware semantic gates ----------------------
    # All four run to completion even after a failure: retry needs the complete
    # failure set to build one prompt that addresses every problem, rather than
    # discovering them one attempt at a time.
    _run(
        records,
        "unsupported_addition",
        check_unsupported_addition,
        candidate_text=candidate_text,
        original_text=original_text,
        evidence=evidence,
        query_text=context.query_text,
        is_abstention=is_abstention,
    )
    _run(
        records,
        "evidence_grounding",
        check_evidence_grounding,
        candidate_text=candidate_text,
        evidence=evidence,
        alignment_threshold=cfg.evidence_alignment_threshold,
        is_abstention=is_abstention,
    )
    _run(
        records,
        "contradiction",
        check_contradiction,
        candidate_text=candidate_text,
        original_text=original_text,
        evidence=evidence,
        claim_texts=context.claim_texts,
        is_abstention=is_abstention,
    )
    _run(
        records,
        "minimal_edit",
        check_minimal_edit,
        candidate_text=candidate_text,
        original_text=original_text,
        min_similarity=cfg.minimal_edit_min_similarity,
        max_expansion=cfg.max_expansion_ratio,
        is_abstention=is_abstention,
    )

    # --- Gate 12: optional entailment, additive only ----------------------
    judgement = _entailment_stage(
        records=records,
        cfg=cfg,
        scorer=scorer,
        evidence=evidence,
        candidate_text=candidate_text,
        is_abstention=is_abstention,
    )

    return _assemble(
        candidate, context, cfg, records, evidence, judgement, is_abstention=is_abstention
    )


def _entailment_stage(
    *,
    records: Dict[str, _GateRecord],
    cfg: CorrectorConfig,
    scorer: Optional[EntailmentScorer],
    evidence: EvidenceView,
    candidate_text: str,
    is_abstention: bool,
) -> Optional[EntailmentJudgement]:
    """Run the injected scorer, if there is one and if it can change anything.

    ADDITIVE ONLY: this stage may add a failure and can never remove one, so an
    injected scorer cannot rescue a candidate that failed a deterministic hard
    gate. It is therefore skipped when gates 1–11 already failed — its verdict
    could not alter the outcome, and calling a model to confirm a decision
    already made only spends time.

    FAIL CLOSED: with ``require_entailment_check=True`` and no usable judgement,
    nothing can be accepted in this configuration, so the failure is recorded as
    terminal rather than retryable — retrying cannot install an NLI model.
    """
    prior_failures = any(records[gate].findings for gate in GATE_ORDER)
    judgement: Optional[EntailmentJudgement] = None
    findings: List[Finding] = []

    if scorer is not None and not prior_failures and not is_abstention:
        try:
            returned = scorer.score(evidence.supporting_text, candidate_text)
        except Exception as exc:  # noqa: BLE001 — an injected scorer is untrusted
            findings.append(_validation_error("entailment", exc))
            returned = None

        if returned is not None:
            entailed = getattr(returned, "entailed", None)
            if isinstance(entailed, bool):
                judgement = returned
                if not entailed:
                    findings.append(
                        Finding(
                            Code.EVIDENCE_UNSUPPORTED.value,
                            "supporting evidence does not entail the corrected sentence",
                            {
                                "label": str(getattr(returned, "label", "")),
                                "score": _as_float(getattr(returned, "score", 0.0)),
                            },
                            retryable=bool(evidence.has_supporting),
                        )
                    )
            else:
                # An injected object that does not answer the question asked is a
                # fault, not a pass.
                findings.append(
                    Finding(
                        Code.VALIDATION_ERROR.value,
                        "injected entailment scorer returned an unusable judgement",
                        {"returned_type": type(returned).__name__},
                        retryable=False,
                    )
                )

    if cfg.require_entailment_check and judgement is None and not is_abstention:
        findings.append(
            Finding(
                Code.VALIDATION_ERROR.value,
                "entailment verification is required but no scorer was available",
                {"scorer": type(scorer).__name__ if scorer else "none"},
                retryable=False,
            )
        )

    records["entailment"] = _GateRecord(ran=scorer is not None, findings=tuple(findings))
    return judgement


def _assemble(
    candidate: CorrectionCandidate,
    context: ValidationContext,
    cfg: CorrectorConfig,
    records: Mapping[str, _GateRecord],
    evidence: EvidenceView,
    judgement: Optional[EntailmentJudgement],
    *,
    is_abstention: bool = False,
) -> CandidateValidation:
    """Turn gate records into the verdict, conjunctively and conservatively."""
    findings: List[Finding] = []
    for gate in GATE_ORDER:
        findings.extend(records[gate].findings)

    structural_valid = records["structure"].clean
    target_authorized = (
        records["authorization"].clean and records["target_identity"].clean
    )
    # An abstention is not scored against the evidence (gate 9 exempts it), so it
    # is recorded as NOT grounded rather than trivially grounded. Recording a
    # dimension as satisfied because it was skipped is how a fail-open bug starts.
    evidence_grounded = records["evidence_grounding"].clean and not is_abstention
    contradiction_detected = bool(records["contradiction"].findings)
    unsupported_addition_detected = bool(records["unsupported_addition"].findings)

    passed = bool(
        not findings
        and not is_abstention
        and structural_valid
        and target_authorized
        and records["original_preservation"].clean
        and records["numbers"].clean
        and records["dates"].clean
        and records["entities"].clean
        and not unsupported_addition_detected
        and evidence_grounded
        and not contradiction_detected
        and records["minimal_edit"].clean
        and not records["entailment"].findings
    )

    # INVARIANT: a rejection must always be explainable. A candidate that is not
    # accepted, is not an abstention, and carries no failure would be an opaque
    # refusal, so the fault is recorded explicitly instead of being hidden.
    if not passed and not findings and not is_abstention:
        findings.append(
            Finding(
                Code.VALIDATION_ERROR.value,
                "candidate was not accepted but no gate recorded a failure",
                {"gates_run": [g for g in GATE_ORDER if records[g].ran]},
                retryable=False,
            )
        )

    return CandidateValidation(
        sentence_id=candidate.sentence_id or context.sentence_id,
        passed=passed,
        failures=[_as_failure(f) for f in findings],
        structural_valid=structural_valid,
        target_authorized=target_authorized,
        evidence_grounded=evidence_grounded,
        contradiction_detected=contradiction_detected,
        unsupported_addition_detected=unsupported_addition_detected,
        entity_consistent=records["entities"].clean,
        numbers_consistent=records["numbers"].clean,
        dates_consistent=records["dates"].clean,
        is_abstention=is_abstention,
        evidence_alignment_score=_measure(
            alignment_score, candidate_text=candidate.corrected_text, evidence=evidence
        ),
        minimal_edit_similarity=_measure(
            edit_similarity,
            candidate_text=candidate.corrected_text,
            original_text=context.original_text,
        ),
        entailment_available=judgement is not None,
        evidence_ids_validated=list(evidence.supporting_ids),
    )
