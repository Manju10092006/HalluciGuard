"""Regression tests for the canonical NLI input contract.

These lock the *measured* production shape rather than an intended one. If a
future change swaps the pair order, adds a marker, or starts joining snippets
onto the evidence side, the change has to happen here, visibly, instead of
drifting in one call site.
"""
from __future__ import annotations

import pytest

from halluciguard_detector.nli_input import (
    CLAIM_PREFIX,
    EVIDENCE_PREFIX,
    MAX_EVIDENCE_SNIPPETS,
    TRUNCATION_STRATEGY,
    contract_spec,
    contract_violations,
    encode_nli_pair,
    enforce_single_snippet,
    prepare_nli_input,
)

CLAIM = "Java was created by James Gosling."
EVIDENCE = "Java was created by James Gosling at Sun Microsystems and released in 1995."


def test_production_classifies_a_single_snippet():
    # ``Detector.detect`` scores ``snippets[0]`` only, so one snippet is the
    # contract, not a simplification of it.
    assert MAX_EVIDENCE_SNIPPETS == 1


def test_pair_order_is_evidence_then_claim():
    assert prepare_nli_input(CLAIM, EVIDENCE) == (EVIDENCE, CLAIM)


def test_claim_and_evidence_are_not_rewritten():
    # Byte-identical pass-through: any normalisation applied here would
    # desynchronise training from runtime for identical inputs.
    messy = "  Java   was\ncreated by James Gosling.  "
    evidence, claim = prepare_nli_input(messy, "  Java\twas created  ")
    assert claim == messy
    assert evidence == "  Java\twas created  "


def test_no_markers_are_injected():
    spec = contract_spec(256)
    assert CLAIM_PREFIX is None
    assert EVIDENCE_PREFIX is None
    assert spec["separator"] is None
    evidence, claim = prepare_nli_input(CLAIM, EVIDENCE)
    assert "Claim:" not in claim and "Evidence:" not in evidence


def test_longest_first_truncation_is_pinned():
    assert TRUNCATION_STRATEGY == "longest_first"
    assert contract_spec(256)["truncation"] == "longest_first"


def test_char_level_pre_truncation_is_honestly_absent():
    # Documented as None instead of a number that is not enforced anywhere.
    spec = contract_spec(256)
    assert spec["max_evidence_chars"] is None
    assert spec["max_claim_chars"] is None
    assert spec["char_level_pre_truncation"] is False


def test_single_snippet_sequence_is_unwrapped():
    assert enforce_single_snippet([EVIDENCE]) == EVIDENCE


def test_joining_snippets_is_refused():
    # The historical bug was ``" ".join(snippets)``. A silent join here is how
    # it came back once already, so a multi-snippet list must raise.
    with pytest.raises(ValueError) as excinfo:
        enforce_single_snippet([EVIDENCE, "A second sentence."])
    assert "1 evidence snippet" in str(excinfo.value)


def test_prepare_refuses_a_snippet_list():
    with pytest.raises(ValueError):
        prepare_nli_input(CLAIM, [EVIDENCE, "A second sentence."])


def test_non_string_evidence_is_rejected():
    with pytest.raises(TypeError):
        enforce_single_snippet(123)


class _StubTokenizer:
    """Minimal tokenizer that records how the pair was handed to it."""

    def __init__(self):
        self.calls = []

    def __call__(self, first, second=None, **kwargs):
        self.calls.append((first, second, kwargs))
        return {"input_ids": [[1, 2, 3]]}


def test_encode_nli_pair_uses_the_contract_and_passes_max_length():
    tokenizer = _StubTokenizer()
    encode_nli_pair(tokenizer, CLAIM, EVIDENCE, max_length=256)
    first, second, kwargs = tokenizer.calls[0]
    assert first == [EVIDENCE]
    assert second == [CLAIM]
    assert kwargs["max_length"] == 256
    assert kwargs["truncation"] == "longest_first"
    assert kwargs["padding"] is True


def test_contract_violations_measures_rather_than_raises():
    rows = [
        {"claim": CLAIM, "evidence": EVIDENCE},
        {"claim": "", "evidence": ""},
    ]
    stats = contract_violations(rows, max_length=256)
    # A dataset that already exists must still be measurable, even when it is
    # out of contract -- otherwise the measurement is only available for data
    # that does not need it.
    assert stats["rows"] == 2
    assert stats["empty_evidence_rows"] == 1


def test_truncation_count_is_none_without_a_tokenizer():
    # "How much evidence is cut" must not be guessed.
    stats = contract_violations([{"claim": CLAIM, "evidence": EVIDENCE}], max_length=256)
    assert stats["truncated_rows"] is None
    assert stats["truncated_fraction"] is None


def test_truncation_count_uses_the_tokenizer_when_available():
    class _Counting:
        """Token count follows the text, so evidence and claims differ."""

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[0] * len(text) for text in texts]}

    rows = [
        {"claim": "short", "evidence": "x" * 300},
        {"claim": "y" * 10, "evidence": "z"},
    ]
    stats = contract_violations(rows, max_length=256, tokenizer=_Counting())
    # Row 0: 300 + 5 + 3 > 256 -> truncated. Row 1: 1 + 10 + 3 = 14 -> fits.
    assert stats["truncated_rows"] == 1
    assert stats["truncated_fraction"] == 0.5
    assert stats["evidence_tokens"]["mean"] == 150.5
