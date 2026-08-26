"""Step 5 — the validation gates: ten independent, pure, total predicates.

DESIGN CONTRACT
---------------
Every gate in this module:

* is a free function that takes explicit inputs and returns a tuple of
  :class:`Finding` — it never mutates its arguments and holds no state;
* is INDEPENDENT: no gate calls another, so each is unit-testable in isolation
  and the pipeline in :mod:`validation` owns the ordering;
* is TOTAL: it does not raise on any input, including empty and adversarial text
  (:mod:`validation` still wraps every call, because "does not raise" must be
  enforced, not merely intended);
* DECIDES but never REPAIRS: a gate reports why a candidate is unacceptable and
  never returns a rewritten sentence. The Corrector must not author corrections.

Acceptance is CONJUNCTIVE: a candidate is accepted only when every hard gate
returns no finding. There is no scoring, no majority vote, and no gate whose pass
can offset another's failure.

PURITY
------
stdlib + :mod:`sentence_utils` + :mod:`text_features` + :mod:`failure_codes` only.
No pydantic, no torch, no transformers, no filesystem, no clock, no randomness.
:mod:`validation` converts :class:`Finding` into the pydantic
``ValidationFailure`` model; keeping that conversion out of here is what lets the
safety-critical gate logic be imported and exercised with nothing installed.

WHAT CATCHES THE RECORDED FAILURE
---------------------------------
``training/data/contract_v4/test_benchmark_report.json`` sample 1 is the model
echoing the target sentence back verbatim, plus a trailing period. It passes every
structural check and every ADDITIVE check — the benchmark recorded
``unsupported_numbers: []`` and ``unsupported_entities: []`` precisely because
nothing was added. :func:`check_contradiction` is what catches it:

* rung 3 fires on the recorded output: candidate and original are equal after
  punctuation folding, so the "add a period" difference is not a correction;
* rung 4 fires on every non-verbatim evasion measured against the same evidence —
  a trailing clause, ``forty-eight`` for ``48``, or ``48 h`` for ``48 hours`` all
  retain the disputed value ``48`` / ``48 hour`` while the supporting evidence
  contains no such value.

MEASURED, NOT ASSUMED: :func:`check_evidence_grounding` scores this echo at
exactly 0.5000 (3 of its 6 content tokens appear in the supporting evidence),
which PASSES the default 0.5 threshold. Gate 9 therefore does NOT catch the
recorded failure and is not credited with doing so; the correct rewrite scores
1.0000 by the same measure. Gate 9 earns its place on other failures, and gate 10
is the one guarantee for this class.

Redundancy is deliberate: no single heuristic is the only thing standing between
a hallucination and the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Tuple

from .failure_codes import ValidationFailureCode as Code
from .sentence_utils import (
    normalize_for_matching,
    normalize_whitespace_case,
    token_coverage,
)
from .text_features import (
    ABSTENTION_VOCABULARY,
    abstention_phrase,
    content_tokens,
    control_characters,
    extract_dates,
    extract_entities,
    extract_numbers,
    extract_quantities,
    leaked_scaffolding,
    length_ratio,
    looks_multi_sentence,
    similarity_ratio,
)

__all__ = [
    "Finding",
    "EvidenceView",
    "AbstentionVerdict",
    "MAX_CANDIDATE_CHARS",
    "ABSTENTION_MAX_CHARS",
    "MIN_SPAN_ECHO_CHARS",
    "alignment_score",
    "edit_similarity",
    "detect_abstention",
    "check_structure",
    "check_authorization",
    "check_target_identity",
    "check_original_preservation",
    "check_numbers",
    "check_dates",
    "check_entities",
    "check_unsupported_addition",
    "check_evidence_grounding",
    "check_contradiction",
    "check_minimal_edit",
]

# A single corrected sentence has a bounded plausible length. `max_new_tokens` is
# 256 (~1000 chars); anything past this is not a sentence.
MAX_CANDIDATE_CHARS = 2000

# An abstention is a short, fixed safety statement. A long passage containing an
# abstention phrase somewhere inside it is not an abstention.
ABSTENTION_MAX_CHARS = 240

# Ignore very short "other span" strings when looking for whole-response
# regeneration: a 5-character span appears inside normal prose by chance.
MIN_SPAN_ECHO_CHARS = 24

_SALIENT_KEYS = ("numbers", "dates", "quantities", "entities")


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    """One concrete reason a candidate is unacceptable.

    ``detail`` carries the offending values so no rejection is ever opaque —
    ``{"retained_unsupported": ["48", "48 hour"]}`` explains itself.
    ``retryable`` is a per-instance judgement: the same code can be retryable in
    one context and terminal in another (``EVIDENCE_UNSUPPORTED`` is retryable
    when supporting evidence exists and terminal when none does).
    """

    code: str
    reason: str
    detail: Dict[str, Any] = field(default_factory=dict)
    retryable: bool = False


def _salient(text: str) -> Dict[str, FrozenSet[str]]:
    """All checkable values in ``text``, as immutable sets."""
    return {
        "numbers": frozenset(extract_numbers(text)),
        "dates": frozenset(extract_dates(text)),
        "quantities": frozenset(
            f"{value} {unit}" for value, unit in extract_quantities(text)
        ),
        "entities": frozenset(extract_entities(text)),
    }


def _union(parts: Iterable[Mapping[str, FrozenSet[str]]]) -> Dict[str, FrozenSet[str]]:
    merged = {key: frozenset() for key in _SALIENT_KEYS}
    for part in parts:
        for key in _SALIENT_KEYS:
            merged[key] = merged[key] | part.get(key, frozenset())
    return merged


@dataclass(frozen=True)
class EvidenceView:
    """The grounding pool for one target, precomputed once.

    GROUNDING RULE: only SUPPORTING evidence grounds a correction, and only the
    supporting evidence whose ids were actually rendered into the prompt. The
    Corrector must not validate against evidence the model never saw — that would
    credit the model for a fact it had no access to.

    Contradictory and neutral evidence are carried for signal and diagnostics
    only. They never ground anything.
    """

    supporting_ids: Tuple[str, ...] = ()
    contradictory_ids: Tuple[str, ...] = ()
    supporting_text: str = ""
    contradictory_text: str = ""
    supporting_tokens: FrozenSet[str] = frozenset()
    supporting_salient: Mapping[str, FrozenSet[str]] = field(
        default_factory=lambda: {key: frozenset() for key in _SALIENT_KEYS}
    )

    @classmethod
    def from_texts(
        cls,
        *,
        supporting: Sequence[Tuple[str, str]] = (),
        contradictory: Sequence[Tuple[str, str]] = (),
    ) -> "EvidenceView":
        """Build from ``(evidence_id, text)`` pairs. Order-independent."""
        supporting_text = "\n".join(text for _, text in supporting)
        contradictory_text = "\n".join(text for _, text in contradictory)
        return cls(
            supporting_ids=tuple(str(eid) for eid, _ in supporting),
            contradictory_ids=tuple(str(eid) for eid, _ in contradictory),
            supporting_text=supporting_text,
            contradictory_text=contradictory_text,
            supporting_tokens=frozenset(content_tokens(supporting_text)),
            supporting_salient=_salient(supporting_text),
        )

    @property
    def has_supporting(self) -> bool:
        return bool(self.supporting_ids) and bool(self.supporting_text.strip())


@dataclass(frozen=True)
class AbstentionVerdict:
    """Whether a candidate is a LEGITIMATE safe abstention.

    POLICY: a legitimate abstention is recognised as valid safe behaviour, is
    exempt from minimal-edit rejection, and is not treated as unsupported
    hallucinated content — but it is NOT a factual correction, so the target
    routes to UNRESOLVED rather than ACCEPTED. Shape alone is not enough: a
    sentence that says "evidence is insufficient" and then asserts a value is a
    smuggled claim, not an abstention.
    """

    is_abstention: bool = False
    phrase: Optional[str] = None
    rejected_because: str = ""


# ---------------------------------------------------------------------------
# Shared measures
# ---------------------------------------------------------------------------
#
# These are the two continuous numbers the pipeline RECORDS on every verdict, and
# they are exposed here so that the number recorded is by construction the number
# a gate DECIDED on. One formula, one source of truth: if a threshold is ever
# retuned, the diagnostic and the decision cannot drift apart.
#
# They are measures, not gates: they return a float and never a verdict.

def alignment_score(*, candidate_text: str, evidence: EvidenceView) -> float:
    """Fraction of the candidate's content tokens found in the supporting evidence.

    Gate 9's measure. Returns 0.0 when the candidate carries no content token,
    which is the conservative direction (nothing measurable is not "grounded").
    """
    candidate_content = content_tokens(candidate_text)
    if not candidate_content:
        return 0.0
    return token_coverage(set(candidate_content), set(evidence.supporting_tokens))


def edit_similarity(*, candidate_text: str, original_text: str) -> float:
    """Similarity of the candidate to the ORIGINAL sentence. Gate 11's measure.

    Returns 0.0 when either side is blank — gate 11 declines to judge minimality
    in that case, and 0.0 records "not measured" without implying a verdict.
    """
    body = (candidate_text or "").strip()
    original = (original_text or "").strip()
    if not body or not original:
        return 0.0
    return similarity_ratio(body, original)


def detect_abstention(
    *,
    candidate_text: str,
    evidence: EvidenceView,
) -> AbstentionVerdict:
    """Classify a candidate as a legitimate abstention, or explain why not."""
    phrase = abstention_phrase(candidate_text)
    if phrase is None:
        return AbstentionVerdict(False, None, "no abstention phrase")

    body = (candidate_text or "").strip()
    if len(body) > ABSTENTION_MAX_CHARS:
        return AbstentionVerdict(
            False, phrase, f"too long to be an abstention ({len(body)} chars)"
        )
    if looks_multi_sentence(body):
        return AbstentionVerdict(False, phrase, "more than one sentence")

    # An abstention asserts nothing specific. Any salient value it does carry
    # must already be grounded in the supporting evidence.
    candidate_salient = _salient(body)
    smuggled = {
        key: sorted(candidate_salient[key] - evidence.supporting_salient.get(key, frozenset()))
        for key in _SALIENT_KEYS
    }
    if any(smuggled.values()):
        offending = {key: values for key, values in smuggled.items() if values}
        return AbstentionVerdict(
            False, phrase, f"asserts ungrounded values {offending}"
        )

    return AbstentionVerdict(True, phrase, "")


# ---------------------------------------------------------------------------
# Gate 1 — structural
# ---------------------------------------------------------------------------

def check_structure(
    *,
    sentence_id: str,
    candidate_text: str,
    is_proposed: bool,
    status: str = "",
    enforce_single_sentence: bool = True,
) -> Tuple[Finding, ...]:
    """Shape of the candidate, independent of meaning.

    Parsing already happened in Step 4's ``output_parser``; this gate does NOT
    re-parse and does NOT repair. It re-checks the invariants that matter for
    safety and adds the ones parsing could not know about: leaked prompt
    scaffolding, control characters, sentence count and length bounds.
    """
    findings: list = []

    if not is_proposed:
        return (
            Finding(
                Code.MODEL_FAILURE.value,
                "candidate was not proposed by the model",
                {"status": status},
                retryable=True,
            ),
        )

    body = candidate_text or ""
    if not body.strip():
        findings.append(
            Finding(
                Code.STRUCTURAL_FAILURE.value,
                "corrected sentence is empty",
                {},
                retryable=True,
            )
        )
        # Nothing further is meaningful on empty text.
        return tuple(findings)

    if not (sentence_id or "").strip():
        findings.append(
            Finding(
                Code.STRUCTURAL_FAILURE.value,
                "candidate carries no sentence id",
                {},
                retryable=False,
            )
        )

    if len(body) > MAX_CANDIDATE_CHARS:
        findings.append(
            Finding(
                Code.STRUCTURAL_FAILURE.value,
                "corrected sentence exceeds the plausible length of one sentence",
                {"length": len(body), "limit": MAX_CANDIDATE_CHARS},
                retryable=True,
            )
        )

    leaked = leaked_scaffolding(body)
    if leaked:
        findings.append(
            Finding(
                Code.STRUCTURAL_FAILURE.value,
                "corrected sentence contains prompt scaffolding",
                {"markers": leaked},
                retryable=True,
            )
        )

    controls = control_characters(body)
    if controls:
        findings.append(
            Finding(
                Code.STRUCTURAL_FAILURE.value,
                "corrected sentence contains control characters",
                {"characters": controls},
                retryable=True,
            )
        )

    if enforce_single_sentence and looks_multi_sentence(body):
        findings.append(
            Finding(
                Code.STRUCTURAL_FAILURE.value,
                "corrected sentence contains more than one sentence",
                {},
                retryable=True,
            )
        )

    return tuple(findings)


# ---------------------------------------------------------------------------
# Gate 2 — authorization
# ---------------------------------------------------------------------------

def check_authorization(
    *,
    sentence_id: str,
    claim_ids: Sequence[str] = (),
    authorized_sentence_ids: Sequence[str] = (),
    authorized_claim_ids: Sequence[str] = (),
    locked_sentence_ids: Sequence[str] = (),
) -> Tuple[Finding, ...]:
    """The candidate may only touch what the Judge authorised.

    This is the gate that stops a prompt-injected or confused model from
    rewriting a sentence nobody asked it to touch. Authorization failures are
    TERMINAL: no rephrased prompt makes an unauthorized edit acceptable, and
    retrying would only invite the model to try again on forbidden ground.
    """
    findings: list = []
    target = (sentence_id or "").strip()
    authorized = [str(sid) for sid in authorized_sentence_ids]
    locked = {str(sid) for sid in locked_sentence_ids}

    if target not in authorized:
        return (
            Finding(
                Code.AUTHORIZATION_FAILURE.value,
                "candidate targets a sentence that was not authorized for correction",
                {"sentence_id": target, "authorized": authorized},
                retryable=False,
            ),
        )

    if authorized.count(target) != 1:
        findings.append(
            Finding(
                Code.AUTHORIZATION_FAILURE.value,
                "sentence id is not unique among authorized targets",
                {"sentence_id": target, "occurrences": authorized.count(target)},
                retryable=False,
            )
        )

    if target in locked:
        findings.append(
            Finding(
                Code.AUTHORIZATION_FAILURE.value,
                "candidate targets a locked sentence",
                {"sentence_id": target},
                retryable=False,
            )
        )

    permitted = {str(cid) for cid in authorized_claim_ids}
    offered = {str(cid) for cid in claim_ids}
    unauthorized = sorted(offered - permitted)
    if unauthorized:
        findings.append(
            Finding(
                Code.AUTHORIZATION_FAILURE.value,
                "candidate cites claims not authorized for this target",
                {"claim_ids": unauthorized, "authorized": sorted(permitted)},
                retryable=False,
            )
        )

    return tuple(findings)


# ---------------------------------------------------------------------------
# Gate 3 — target identity
# ---------------------------------------------------------------------------

def check_target_identity(
    *,
    candidate_sentence_id: str,
    expected_sentence_id: str,
) -> Tuple[Finding, ...]:
    """EXACT string equality of the sentence id.

    No normalization, no casefolding, no prefix or numeric comparison. Sentence
    ids are ``S1``, ``S2``, ... ``S10``, so any prefix-based or startswith-based
    comparison would let ``S1`` satisfy a request for ``S10``. Terminal: a
    candidate for the wrong sentence cannot be repaired into one for the right
    sentence.
    """
    if candidate_sentence_id == expected_sentence_id:
        return ()
    return (
        Finding(
            Code.TARGET_MISMATCH.value,
            "candidate sentence id does not match the requested target",
            {"expected": expected_sentence_id, "received": candidate_sentence_id},
            retryable=False,
        ),
    )


# ---------------------------------------------------------------------------
# Gate 4 — original-content preservation
# ---------------------------------------------------------------------------

def check_original_preservation(
    *,
    candidate_text: str,
    other_span_texts: Sequence[str] = (),
) -> Tuple[Finding, ...]:
    """The candidate must replace ONE sentence, not regenerate the response.

    A model that returns several of the response's other sentences has ignored
    the single-target mandate. Splicing that back in (Step 6) would duplicate or
    destroy untouched content, so it is rejected here.
    """
    body = normalize_for_matching(candidate_text or "")
    if not body:
        return ()

    echoed = []
    for span in other_span_texts:
        normalized = normalize_for_matching(span or "")
        if len(normalized) < MIN_SPAN_ECHO_CHARS:
            continue
        if normalized in body:
            echoed.append(span)

    if not echoed:
        return ()

    return (
        Finding(
            Code.STRUCTURAL_FAILURE.value,
            "corrected sentence reproduces other sentences of the response",
            {"echoed_spans": echoed[:5], "echoed_count": len(echoed)},
            retryable=True,
        ),
    )


# ---------------------------------------------------------------------------
# Gates 5-7 — added specific values must be grounded
# ---------------------------------------------------------------------------

def _added_ungrounded(
    *,
    candidate_text: str,
    original_text: str,
    evidence: EvidenceView,
    key: str,
) -> Tuple[str, ...]:
    """Values the candidate introduces that neither the original nor evidence has.

    Removing a value is not a failure — deleting a falsehood is a correction.
    Only ADDITIONS carry a grounding burden here; retained values are gate 10's
    concern.
    """
    candidate_values = _salient(candidate_text)[key]
    original_values = _salient(original_text)[key]
    grounded = evidence.supporting_salient.get(key, frozenset())
    return tuple(sorted((candidate_values - original_values) - grounded))


def check_numbers(
    *,
    candidate_text: str,
    original_text: str,
    evidence: EvidenceView,
) -> Tuple[Finding, ...]:
    """Numbers introduced by the correction must appear in supporting evidence.

    Comparison is VALUE-level and word-boundary anchored, so ``48`` is not
    considered grounded by an unrelated ``1948`` elsewhere in the evidence, and a
    spelled-out ``forty-eight`` is the same value as ``48``.
    """
    ungrounded = _added_ungrounded(
        candidate_text=candidate_text,
        original_text=original_text,
        evidence=evidence,
        key="numbers",
    )
    quantities = _added_ungrounded(
        candidate_text=candidate_text,
        original_text=original_text,
        evidence=evidence,
        key="quantities",
    )
    if not ungrounded and not quantities:
        return ()
    return (
        Finding(
            Code.NUMBER_MISMATCH.value,
            "correction introduces numbers not present in the supporting evidence",
            {"numbers": list(ungrounded), "quantities": list(quantities)},
            retryable=True,
        ),
    )


def check_dates(
    *,
    candidate_text: str,
    original_text: str,
    evidence: EvidenceView,
) -> Tuple[Finding, ...]:
    """Dates introduced by the correction must appear in supporting evidence.

    Dates are component-normalized, so ``2024-05-03``, ``May 3, 2024`` and
    ``03/05/2024`` are one value and a format change is not mistaken for a new
    fact.
    """
    ungrounded = _added_ungrounded(
        candidate_text=candidate_text,
        original_text=original_text,
        evidence=evidence,
        key="dates",
    )
    if not ungrounded:
        return ()
    return (
        Finding(
            Code.DATE_MISMATCH.value,
            "correction introduces dates not present in the supporting evidence",
            {"dates": list(ungrounded)},
            retryable=True,
        ),
    )


def check_entities(
    *,
    candidate_text: str,
    original_text: str,
    evidence: EvidenceView,
) -> Tuple[Finding, ...]:
    """Named entities introduced by the correction must be grounded.

    An entity counts as grounded when the evidence names it, or when every token
    of it appears in the evidence — so ``Rossum`` is grounded by
    ``Guido van Rossum``. Extraction is regex-based (no spaCy, no new
    dependency); residual imprecision produces extra rejections rather than extra
    acceptances.
    """
    candidate_entities = _salient(candidate_text)["entities"]
    original_entities = _salient(original_text)["entities"]
    evidence_entities = evidence.supporting_salient.get("entities", frozenset())

    ungrounded = []
    for entity in sorted(candidate_entities - original_entities):
        if entity in evidence_entities:
            continue
        if any(entity in known or known in entity for known in evidence_entities):
            continue
        pieces = {piece for piece in entity.split() if len(piece) > 1}
        if pieces and pieces <= evidence.supporting_tokens:
            continue
        ungrounded.append(entity)

    if not ungrounded:
        return ()
    return (
        Finding(
            Code.ENTITY_MISMATCH.value,
            "correction introduces entities not present in the supporting evidence",
            {"entities": ungrounded},
            retryable=True,
        ),
    )


# ---------------------------------------------------------------------------
# Gate 8 — unsupported addition (content level)
# ---------------------------------------------------------------------------

def check_unsupported_addition(
    *,
    candidate_text: str,
    original_text: str,
    evidence: EvidenceView,
    query_text: str = "",
    is_abstention: bool = False,
) -> Tuple[Finding, ...]:
    """Content the correction adds must come from authorized material.

    Permitted sources for NEW wording are the supporting evidence, the original
    sentence, and the user's query. Anything else is the model supplying content
    from its parameters — which is exactly the hallucination this agent exists to
    prevent.

    ABSTENTION EXEMPTION: a legitimate abstention necessarily adds its own safety
    vocabulary ("evidence", "insufficient", "support", "claim"). That vocabulary
    is not hallucinated content, so it is permitted here. The abstention still
    does not become an accepted correction — routing is the pipeline's decision,
    not this gate's.
    """
    added = content_tokens(candidate_text) - content_tokens(original_text)
    if not added:
        return ()

    permitted = set(evidence.supporting_tokens)
    permitted |= content_tokens(original_text)
    if query_text:
        permitted |= content_tokens(query_text)
    if is_abstention:
        permitted |= set(ABSTENTION_VOCABULARY)

    ungrounded = sorted(added - permitted)
    if not ungrounded:
        return ()
    return (
        Finding(
            Code.UNSUPPORTED_ADDITION.value,
            "correction adds content that no authorized source supports",
            {"tokens": ungrounded},
            retryable=True,
        ),
    )


# ---------------------------------------------------------------------------
# Gate 9 — evidence grounding / alignment
# ---------------------------------------------------------------------------

def check_evidence_grounding(
    *,
    candidate_text: str,
    evidence: EvidenceView,
    alignment_threshold: float = 0.5,
    is_abstention: bool = False,
) -> Tuple[Finding, ...]:
    """How much of the candidate actually comes from the supporting evidence.

    Measured as the fraction of the candidate's content tokens found in the
    supporting evidence. This is a GLOBAL counterpart to gate 8's per-token check:
    gate 8 asks "is each new token authorized?", this asks "is the sentence as a
    whole built from the evidence?".

    CALIBRATION NOTE (measured, not assumed): on the recorded echo this scores
    exactly 0.5000, which PASSES the default 0.5 threshold, while the correct
    rewrite scores 1.0000. This gate therefore does not catch that echo —
    :func:`check_contradiction` does — and the default threshold is untuned for
    this metric. Alignment failure is retryable rather than terminal for that
    reason.

    With NO supporting evidence there is nothing to ground against, and the
    failure is TERMINAL — retrying cannot conjure evidence, and pressing the
    model for another attempt with no evidence is an invitation to fabricate.
    """
    if is_abstention:
        # An abstention deliberately draws its wording from safety vocabulary
        # rather than from the evidence. Scoring it here would punish correct
        # behaviour; it is routed as UNRESOLVED by the pipeline regardless.
        return ()

    if not evidence.has_supporting:
        return (
            Finding(
                Code.EVIDENCE_UNSUPPORTED.value,
                "no supporting evidence was available to ground a correction",
                {
                    "supporting_ids": list(evidence.supporting_ids),
                    "contradictory_ids": list(evidence.contradictory_ids),
                },
                retryable=False,
            ),
        )

    candidate_content = content_tokens(candidate_text)
    if not candidate_content:
        return (
            Finding(
                Code.EVIDENCE_UNSUPPORTED.value,
                "corrected sentence carries no content to ground",
                {},
                retryable=True,
            ),
        )

    score = alignment_score(candidate_text=candidate_text, evidence=evidence)
    if score >= alignment_threshold:
        return ()
    return (
        Finding(
            Code.EVIDENCE_UNSUPPORTED.value,
            "correction is not sufficiently grounded in the supporting evidence",
            {
                "alignment": round(score, 4),
                "threshold": alignment_threshold,
                "ungrounded_tokens": sorted(
                    set(candidate_content) - set(evidence.supporting_tokens)
                )[:20],
            },
            retryable=True,
        ),
    )


# ---------------------------------------------------------------------------
# Gate 10 — contradiction / echo ladder
# ---------------------------------------------------------------------------

def check_contradiction(
    *,
    candidate_text: str,
    original_text: str,
    evidence: EvidenceView,
    claim_texts: Sequence[str] = (),
    is_abstention: bool = False,
) -> Tuple[Finding, ...]:
    """Did the model actually correct anything, or restate the falsehood?

    THE GATE THAT CATCHES THE RECORDED FAILURE. The Judge already decided this
    sentence needs correcting, so returning it unchanged is not a conservative
    edit — it is a failure to correct, dressed as compliance.

    A four-rung ladder, cheapest first:

    1. exact string equality;
    2. equality after whitespace and case folding;
    3. equality after punctuation folding (defeats "add a period" evasion);
    4. retained salient values that the supporting evidence does not support
       (defeats rewording that keeps the falsehood).

    Rung 4 is SCOPED to values the disputed claims actually mention, when claim
    text is available. Without that scoping, correcting "the Eiffel Tower in
    Berlin is 300m tall" to "...in Paris..." would be rejected for retaining
    ``300`` — a number nobody disputed. With no claim text the scope falls back
    to all retained values, which fails safe.
    """
    if is_abstention:
        # An abstention is not an echo: it asserts nothing and retains nothing.
        # POLICY: it must not be treated as unsupported hallucinated content.
        return ()

    candidate = candidate_text or ""
    original = original_text or ""

    if candidate == original:
        return (
            Finding(
                Code.EVIDENCE_CONTRADICTION.value,
                "corrected sentence is identical to the sentence that needed correcting",
                {"rung": "exact"},
                retryable=True,
            ),
        )

    if normalize_whitespace_case(candidate) == normalize_whitespace_case(original):
        return (
            Finding(
                Code.EVIDENCE_CONTRADICTION.value,
                "corrected sentence differs from the original only in whitespace or case",
                {"rung": "whitespace_case"},
                retryable=True,
            ),
        )

    if normalize_for_matching(candidate) == normalize_for_matching(original):
        return (
            Finding(
                Code.EVIDENCE_CONTRADICTION.value,
                "corrected sentence differs from the original only in punctuation",
                {"rung": "punctuation"},
                retryable=True,
            ),
        )

    candidate_salient = _salient(candidate)
    original_salient = _salient(original)
    disputed = _union([_salient(text) for text in claim_texts]) if claim_texts else None

    retained_unsupported: Dict[str, list] = {}
    for key in _SALIENT_KEYS:
        retained = candidate_salient[key] & original_salient[key]
        retained = retained - evidence.supporting_salient.get(key, frozenset())
        if disputed is not None:
            retained = retained & disputed[key]
        if retained:
            retained_unsupported[key] = sorted(retained)

    if not retained_unsupported:
        return ()

    return (
        Finding(
            Code.EVIDENCE_CONTRADICTION.value,
            "corrected sentence still asserts disputed values the evidence does not support",
            {
                "rung": "retained_values",
                "retained_unsupported": retained_unsupported,
                "scoped_to_claims": disputed is not None,
                "contradictory_evidence_present": bool(evidence.contradictory_ids),
            },
            retryable=True,
        ),
    )


# ---------------------------------------------------------------------------
# Gate 11 — minimal edit (soft)
# ---------------------------------------------------------------------------

def check_minimal_edit(
    *,
    candidate_text: str,
    original_text: str,
    min_similarity: float = 0.25,
    max_expansion: float = 2.5,
    is_abstention: bool = False,
) -> Tuple[Finding, ...]:
    """Is this a repair, or a rewrite?

    A correction should change what is wrong and leave the rest alone. A
    candidate that shares almost nothing with the original has replaced the
    sentence rather than corrected it, which loses content the Judge never
    disputed.

    SOFT gate: it blocks acceptance and is retryable, but it is a quality signal
    rather than a safety one, so it never contributes to a terminal rejection on
    its own.

    ABSTENTION EXEMPTION (policy): an abstention is by nature nothing like the
    original sentence. Rejecting it for non-minimality would penalise the exact
    behaviour the model was trained to produce when evidence is inadequate.

    The measure here is similarity to the ORIGINAL, which is computable at
    runtime. It is deliberately NOT ``benchmark_engine``'s ``minimal_edit_score``,
    which compares against the GOLD sentence and requires a label production
    never has.
    """
    if is_abstention:
        return ()

    body = (candidate_text or "").strip()
    original = (original_text or "").strip()
    if not body or not original:
        return ()

    findings: list = []
    similarity = edit_similarity(candidate_text=body, original_text=original)
    expansion = length_ratio(body, original)

    if similarity < min_similarity:
        findings.append(
            Finding(
                Code.NON_MINIMAL_EDIT.value,
                "correction rewrites the sentence instead of repairing it",
                {"similarity": round(similarity, 4), "minimum": min_similarity},
                retryable=True,
            )
        )

    if expansion > max_expansion:
        findings.append(
            Finding(
                Code.NON_MINIMAL_EDIT.value,
                "correction is substantially longer than the sentence it replaces",
                {"length_ratio": round(expansion, 4), "maximum": max_expansion},
                retryable=True,
            )
        )

    return tuple(findings)
