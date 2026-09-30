"""The single canonical NLI input contract for the grounded Detector.

The Detector is a pairwise classifier. Whatever it reads at runtime it must
also have read during training, and until now the pair was built in two places
-- the training collator and the runtime classifier -- that happened to agree
but were free to drift. This module owns the contract so there is exactly one
definition to change, and every path (dataset prep, training, dev, test, and
production inference) goes through it.

The contract is not a design choice; it is a *description* of what the
production pipeline already sends to DeBERTa, measured rather than invented.
Traced from ``detector.Detector.detect``:

* **One claim, one evidence snippet.** ``detect`` calls
  ``evidence.select_evidence`` and classifies ``snippets[0]`` only. The rest of
  the ranked list is kept on the response for inspection and is never shown to
  the model. The evidence side is therefore a single string, never a
  concatenation of several retrieved snippets.
* **Order.** ``evidence`` is the first sequence, ``claim`` the second, matching
  the released ``microsoft/deberta-v3-xsmall`` sentence-pair convention.
  Reversing them is a real failure mode -- DeBERTa would score the inverted
  relation -- so the swap lives in exactly one place.
* **No markers.** The production pair carries no ``"Claim:"`` / ``"Evidence:"``
  prefix and no separator string. Adding them would move the model off the
  distribution it was trained and shipped on, so they are explicitly ``None``
  here rather than merely absent.
* **Truncation.** ``longest_first``, so the longer of the two is trimmed. There
  is no character-level pre-truncation in production: the tokenizer's
  ``max_length`` (256, from the checkpoint's ``calibration.json``) is the only
  length limit that exists.
* **Length.** ``max_length`` is supplied by the caller so the shipped
  calibration value stays the single source of truth.

``MAX_EVIDENCE_SNIPPETS`` and :func:`enforce_single_snippet` exist because the
historical failure mode was structural rather than textual: dataset prep joined
up to six retrieved sentences into one string, so the *evidence side* was long
and got truncated away from the back. That silently taught the model to expect
a paragraph and served it a sentence. :func:`contract_violations` measures how
often a dataset still does that, so the regression is visible instead of
implied.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "MAX_EVIDENCE_SNIPPETS",
    "TRUNCATION_STRATEGY",
    "CLAIM_PREFIX",
    "EVIDENCE_PREFIX",
    "prepare_nli_input",
    "encode_nli_pair",
    "enforce_single_snippet",
    "contract_spec",
    "contract_violations",
    "TOKENIZER_KWARGS",
]

#: Production classifies ``snippets[0]`` and nothing else, so the evidence side
#: of an NLI pair is one selected snippet. Enforced rather than documented.
MAX_EVIDENCE_SNIPPETS = 1

#: Applied by the tokenizer on every path. ``longest_first`` trims the longer
#: side, which is why a long training evidence blob loses its tail.
TRUNCATION_STRATEGY = "longest_first"

#: The production pair is bare text. These are ``None`` on purpose: they are
#: pinned here so a future edit has to make the distribution change explicitly
#: rather than slipping a marker into one call site.
CLAIM_PREFIX: str | None = None
EVIDENCE_PREFIX: str | None = None

#: Encoding options shared by training, evaluation and runtime. ``padding`` is
#: resolved by the tokenizer from the longest member of the batch, so a short
#: single-snippet batch stays cheap while a long joined-evidence batch is
#: still padded consistently.
TOKENIZER_KWARGS: dict[str, Any] = {
    "padding": True,
    "truncation": TRUNCATION_STRATEGY,
}


def prepare_nli_input(claim: str, evidence: str) -> tuple[str, str]:
    """Return the ordered ``(evidence, claim)`` pair used for every NLI call.

    The claim/evidence argument order matches how callers think about the
    example, while the returned order matches what the tokenizer consumes.
    Reversing them is a real failure mode -- DeBERTa would score the inverted
    relation -- so the swap lives in exactly one place.

    Neither string is rewritten. Whitespace, casing, and truncation-affecting
    characters are left untouched so training and runtime stay byte-identical
    for identical inputs. Callers holding several retrieved snippets must call
    :func:`enforce_single_snippet` first and pass the one string it returns.
    """
    evidence = enforce_single_snippet(evidence)
    return (evidence, claim)


def enforce_single_snippet(evidence: Any) -> str:
    """Return exactly one evidence string, refusing to silently concatenate.

    A ``str`` passes through unchanged. A single-element sequence is unwrapped.
    Anything else raises: joining several retrieved snippets into one string is
    the exact train/runtime mismatch this module exists to prevent, and a
    silent ``" ".join`` here is how it was reintroduced once already.
    """
    if isinstance(evidence, str):
        return evidence
    if isinstance(evidence, Sequence):
        items = [item for item in evidence if isinstance(item, str) and item.strip()]
        if len(items) == 1:
            return items[0]
        raise ValueError(
            f"NLI input takes {MAX_EVIDENCE_SNIPPETS} evidence snippet, got "
            f"{len(items)}. Select one snippet (e.g. select_evidence(...)[0]) "
            f"instead of passing a list; concatenating snippets changes the "
            f"training distribution and the length the model sees."
        )
    raise TypeError(f"evidence must be a str, got {type(evidence).__name__}")


def encode_nli_pair(
    tokenizer: Any,
    claim: str,
    evidence: str,
    *,
    max_length: int,
) -> Any:
    """Tokenize one claim against one evidence string, per the contract.

    This is the only place a Detector input becomes tensors. Training, dev,
    test and runtime all end up here, which is what makes "the model is scored
    on the distribution it was trained on" a structural property rather than a
    convention.
    """
    evidence_seq, claim_seq = prepare_nli_input(claim, evidence)
    return tokenizer(
        [evidence_seq],
        [claim_seq],
        return_tensors="pt",
        max_length=max_length,
        **TOKENIZER_KWARGS,
    )


def contract_spec(max_length: int) -> dict[str, Any]:
    """Machine-readable description of the contract, for reports and docs."""
    return {
        "pair_order": ["evidence", "claim"],
        "evidence_snippets": MAX_EVIDENCE_SNIPPETS,
        "claim_prefix": CLAIM_PREFIX,
        "evidence_prefix": EVIDENCE_PREFIX,
        "separator": None,
        "truncation": TRUNCATION_STRATEGY,
        "max_sequence_length": int(max_length),
        "max_claim_chars": None,
        "max_evidence_chars": None,
        "char_level_pre_truncation": False,
        "note": (
            "No character-level cap exists. The only length limit is the "
            "tokenizer's max_length, applied with truncation="
            f"{TRUNCATION_STRATEGY!r}, which trims the longer side. A dataset "
            "that joins several snippets onto the evidence side is therefore "
            "out of contract, and the tail of that evidence is what gets cut."
        ),
    }


def contract_violations(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_length: int,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Measure how far a dataset sits from the contract.

    Reported, never raised: the point of the measurement is to make the
    mismatch *visible* on a dataset that already exists, not to refuse to
    evaluate it. ``truncated`` needs a tokenizer; without one it is reported as
    ``None`` rather than guessed, because "how much of this evidence is cut" is
    exactly the kind of number that must not be invented.
    """
    rows = list(rows)
    total = len(rows)
    if not total:
        return {"rows": 0}

    evidence_lengths = [len(str(row.get("evidence", ""))) for row in rows]
    claim_lengths = [len(str(row.get("claim", ""))) for row in rows]
    empty_evidence = sum(1 for length in evidence_lengths if length == 0)

    truncated: int | None = None
    evidence_tokens: list[int] | None = None
    if tokenizer is not None:
        evidence_tokens = [
            len(ids) for ids in tokenizer([str(row.get("evidence", "")) for row in rows])["input_ids"]
        ]
        claim_tokens = [
            len(ids) for ids in tokenizer([str(row.get("claim", "")) for row in rows])["input_ids"]
        ]
        # +3 for [CLS] / [SEP] / [SEP], which the tokenizer adds per sequence.
        truncated = sum(1 for a, b in zip(evidence_tokens, claim_tokens) if a + b + 3 > max_length)

    return {
        "rows": total,
        "empty_evidence_rows": empty_evidence,
        "evidence_chars": {
            "mean": sum(evidence_lengths) / total,
            "max": max(evidence_lengths),
        },
        "claim_chars": {
            "mean": sum(claim_lengths) / total,
            "max": max(claim_lengths),
        },
        "evidence_tokens": {
            "mean": (sum(evidence_tokens) / total) if evidence_tokens else None,
            "max": max(evidence_tokens) if evidence_tokens else None,
        },
        "truncated_rows": truncated,
        "truncated_fraction": (truncated / total) if truncated is not None else None,
    }
