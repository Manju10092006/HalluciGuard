"""Step 4 — strict structured-output parsing for the Corrector Agent.

REJECT, NEVER REPAIR (requirement 13)
-------------------------------------
The correction model must return exactly one JSON object::

    {"sentence_id": "<authorized id>", "corrected_sentence": "<rewritten sentence>"}

Anything else is rejected with an explicit reason. This module deliberately does
NOT:

* scan prose for the first ``{...}`` it can find (the reference implementation's
  greedy ``(\\{[\\s\\S]*\\})`` scan lets a model prepend arbitrary commentary — or
  a second object — and still "succeed");
* coerce non-string values, rename near-miss keys, or fill in missing fields;
* silently ignore unexpected keys (we cannot claim to have understood output we
  only partly parsed);
* rewrite, strip or reflow ``corrected_sentence``. It is carried VERBATIM.
  Whitespace normalisation is a reconstruction concern (Step 6), not something
  to apply silently here;
* carry text forward when the model answered for the WRONG sentence. The
  reference returned the unauthorized text in ``corrected_text`` alongside an
  INVALID status; that text is dropped here so it cannot leak into a splice.

The single structural allowance is unwrapping one complete markdown code fence
that encloses the entire output — a lossless unwrap of a container, not a repair
of content.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .contracts import (
    CandidateStatus,
    CorrectionCandidate,
    PromptBundle,
    RejectionReason,
)

__all__ = [
    "REQUIRED_KEYS",
    "extract_json_object",
    "parse_candidate",
    "rejected_candidate",
]

REQUIRED_KEYS = ("sentence_id", "corrected_sentence")

# Matches ONLY when a code fence encloses the entire output.
_WHOLE_FENCE = re.compile(r"\A```[A-Za-z0-9_+-]*[ \t]*\r?\n?(.*?)\r?\n?```\Z", re.DOTALL)

_DETAIL_LIMIT = 240


def _clip(text: str) -> str:
    if len(text) <= _DETAIL_LIMIT:
        return text
    return text[:_DETAIL_LIMIT] + "…[clipped]"


def extract_json_object(
    raw: Any,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """Parse ``raw`` into a JSON object, or explain why it is not one.

    Returns ``(obj, reason, detail)``; ``obj`` is None whenever ``reason`` is set.
    """
    if raw is None:
        return None, RejectionReason.EMPTY_OUTPUT.value, "model returned None"
    if not isinstance(raw, str):
        return (
            None,
            RejectionReason.NOT_JSON.value,
            f"model output is {type(raw).__name__}, not text",
        )

    text = raw.strip()
    if not text:
        return None, RejectionReason.EMPTY_OUTPUT.value, "model returned empty text"

    fenced = _WHOLE_FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
        if not text:
            return (
                None,
                RejectionReason.EMPTY_OUTPUT.value,
                "model returned an empty code fence",
            )

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return (
            None,
            RejectionReason.NOT_JSON.value,
            f"output is not valid JSON: {_clip(str(exc))}",
        )

    if not isinstance(parsed, dict):
        return (
            None,
            RejectionReason.NOT_JSON_OBJECT.value,
            f"output parsed as {type(parsed).__name__}, expected a JSON object",
        )
    return parsed, "", ""


def rejected_candidate(
    bundle: PromptBundle,
    reason: RejectionReason,
    detail: str,
    raw: str = "",
) -> CorrectionCandidate:
    """Build a rejected candidate that carries NO corrected text."""
    return CorrectionCandidate(
        sentence_id=bundle.sentence_id,
        claim_ids=list(bundle.claim_ids),
        original_text=bundle.original_sentence,
        corrected_text="",
        status=CandidateStatus.REJECTED.value,
        rejection_reason=reason.value,
        rejection_detail=detail,
        evidence_ids_used=list(bundle.evidence_ids),
        raw_output=raw,
    )


def parse_candidate(raw: Any, bundle: PromptBundle) -> CorrectionCandidate:
    """Parse one model output against the prompt that produced it.

    ``status == proposed`` means only "structurally valid JSON for the authorized
    target". Evidence alignment, unsupported-content checks and retry are Step 5.
    """
    raw_text = raw if isinstance(raw, str) else ""

    parsed, reason, detail = extract_json_object(raw)
    if parsed is None:
        return rejected_candidate(
            bundle, RejectionReason(reason), detail, raw_text
        )

    keys = set(parsed.keys())
    missing = [key for key in REQUIRED_KEYS if key not in keys]
    if missing:
        return rejected_candidate(
            bundle,
            RejectionReason.MISSING_FIELD,
            f"missing required field(s): {', '.join(missing)}",
            raw_text,
        )

    unexpected = sorted(keys - set(REQUIRED_KEYS))
    if unexpected:
        return rejected_candidate(
            bundle,
            RejectionReason.UNEXPECTED_FIELDS,
            "output contains unrecognized field(s), refusing to partially "
            f"interpret it: {', '.join(unexpected)}",
            raw_text,
        )

    wrong_types: List[str] = [
        key for key in REQUIRED_KEYS if not isinstance(parsed[key], str)
    ]
    if wrong_types:
        types = ", ".join(f"{k}={type(parsed[k]).__name__}" for k in wrong_types)
        return rejected_candidate(
            bundle,
            RejectionReason.WRONG_TYPE,
            f"expected string field(s), got: {types}",
            raw_text,
        )

    returned_id = parsed["sentence_id"]
    corrected = parsed["corrected_sentence"]

    if not corrected.strip():
        return rejected_candidate(
            bundle,
            RejectionReason.EMPTY_CORRECTED_SENTENCE,
            "corrected_sentence is empty or whitespace only",
            raw_text,
        )

    # Authorization check: the model may only answer for the target it was
    # given. The unauthorized text is intentionally NOT carried forward.
    if returned_id != bundle.sentence_id:
        return rejected_candidate(
            bundle,
            RejectionReason.SENTENCE_ID_MISMATCH,
            "model answered for sentence_id "
            f"{returned_id!r}, authorized target is {bundle.sentence_id!r}; "
            "discarding the unauthorized text",
            raw_text,
        )

    return CorrectionCandidate(
        sentence_id=bundle.sentence_id,
        claim_ids=list(bundle.claim_ids),
        original_text=bundle.original_sentence,
        corrected_text=corrected,  # VERBATIM — no strip, no repair
        status=CandidateStatus.PROPOSED.value,
        rejection_reason="",
        rejection_detail="",
        evidence_ids_used=list(bundle.evidence_ids),
        raw_output=raw_text,
    )
