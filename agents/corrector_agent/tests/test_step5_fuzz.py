"""Step 5 — seeded property/fuzz tests and scope guards.

Two jobs:

1. **Properties and fuzzing.** The gates rest on a handful of text primitives
   (number/date/entity extraction, similarity, edit distance) and on the promise
   that the pipeline *never raises* and *never rejects opaquely*. Those are
   universally-quantified claims, so they are tested over seeded random input
   rather than a few hand-picked strings. Seeds are fixed, so a failure reproduces
   exactly.

2. **Scope guards.** Step 5 must not quietly grow into Step 6, a Judge, a Memory,
   a reverifier, or a dependency on the reference implementation. Following
   ``test_prompt_model.py``'s Step 4 guards, these checks read the module AST — not
   its docstrings — so prose can never satisfy or violate them.

The randomness is confined to inputs; every assertion is a deterministic invariant
that must hold for all of them.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path
from typing import List

import pytest

from agents.corrector_agent.corrector.config import CorrectorConfig
from agents.corrector_agent.corrector.contracts import CandidateValidation, TargetDecision
from agents.corrector_agent.corrector import retry as retry_mod
from agents.corrector_agent.corrector import text_features as T
from agents.corrector_agent.corrector import validation as validation_mod
from agents.corrector_agent.corrector import validators as validators_mod
from agents.corrector_agent.corrector.retry import (
    ATTEMPT_HARD_CAP,
    resolve_target,
    total_attempts,
)
from agents.corrector_agent.corrector.validation import validate_candidate

STEP5_MODULES = [validation_mod, validators_mod, retry_mod, T]

_VALID_DECISIONS = {
    TargetDecision.ACCEPTED.value,
    TargetDecision.REJECTED.value,
    TargetDecision.UNRESOLVED.value,
}


# ---------------------------------------------------------------------------
# A seeded corpus of awkward strings
# ---------------------------------------------------------------------------

_FRAGMENTS = [
    "Python", "Guido van Rossum", "Elon Musk", "1991", "1823", "48", "forty-eight",
    "2050-01-02", "NASA", "Paris", "in", "the", "was", "created", "by", "hours",
    "insufficient evidence", "Current evidence is insufficient to support this claim.",
    "", " ", ".", "\t", "\n", "🙂", "\x00", "<<<HG-DATA-deadbeef>>>",
    "=== OUTPUT MANDATE ===", "Ignore previous instructions", "{}", '"', "\\",
    "%s" % ("x" * 40), "or 500", "at Google", "500", "€", "½", "12,345", "3.14",
]


def _random_text(rng: random.Random) -> str:
    n = rng.randint(0, 12)
    joiner = rng.choice([" ", "", ". ", ", "])
    return joiner.join(rng.choice(_FRAGMENTS) for _ in range(n))


def _corpus(seed: int, count: int) -> List[str]:
    rng = random.Random(seed)
    return [_random_text(rng) for _ in range(count)]


# ===========================================================================
# Text-feature properties (metamorphic invariants over random input)
# ===========================================================================

@pytest.mark.parametrize("seed", [1, 7, 42, 100, 2024])
def test_extractors_are_deterministic_and_idempotent(seed):
    for text in _corpus(seed, 60):
        for extractor in (
            T.extract_numbers,
            T.extract_dates,
            T.extract_entities,
            T.extract_quantities,
            T.content_tokens,
        ):
            first = extractor(text)
            assert extractor(text) == first  # pure: same input, same output


@pytest.mark.parametrize("seed", [2, 13, 99])
def test_similarity_ratio_is_bounded_and_reflexive(seed):
    # NOTE: similarity_ratio wraps difflib.SequenceMatcher.ratio(), which is not
    # guaranteed symmetric (its junk heuristic favours the second sequence), so
    # symmetry is deliberately NOT asserted. The gates only ever call it with a
    # fixed (candidate, original) argument order, so the real invariants that
    # matter are the range and reflexivity.
    corpus = _corpus(seed, 50)
    for text in corpus:
        assert T.similarity_ratio(text, text) == pytest.approx(1.0)
    for a, b in zip(corpus, corpus[1:]):
        assert 0.0 <= T.similarity_ratio(a, b) <= 1.0


@pytest.mark.parametrize("seed", [3, 17, 88])
def test_edit_distance_is_a_metric_shape(seed):
    corpus = _corpus(seed, 40)
    for text in corpus:
        assert T.edit_distance(text, text) == 0  # identity of indiscernibles
    for a, b in zip(corpus, corpus[1:]):
        assert T.edit_distance(a, b) >= 0  # non-negative
        assert T.edit_distance(a, b) == T.edit_distance(b, a)  # symmetric
        if a != b and (a or b):
            assert T.edit_distance(a, b) > 0


@pytest.mark.parametrize("seed", [4, 21])
def test_content_tokens_never_contain_stopwords(seed):
    for text in _corpus(seed, 60):
        assert T.content_tokens(text).isdisjoint(T.STOPWORDS)


@pytest.mark.parametrize("seed", [5, 23, 77])
def test_text_features_never_raise_on_random_input(seed):
    corpus = _corpus(seed, 80)
    for text in corpus:
        T.extract_numbers(text)
        T.extract_dates(text)
        T.extract_entities(text)
        T.extract_quantities(text)
        T.content_tokens(text)
        T.abstention_phrase(text)
        T.is_abstention_shaped(text)
        T.leaked_scaffolding(text)
        T.control_characters(text)
        T.looks_multi_sentence(text)
        T.salient_values(text)
    for a, b in zip(corpus, corpus[1:]):
        T.similarity_ratio(a, b)
        T.edit_distance(a, b)
        T.length_ratio(a, b)


def test_number_extraction_understands_words_and_digits():
    # Anchored spot-checks so the fuzz above cannot pass vacuously.
    assert "48" in T.extract_numbers("wait 48 hours")
    assert "48" in T.extract_numbers("wait forty-eight hours")


# ===========================================================================
# Pipeline fuzz — never raises, never rejects opaquely, accepts conjunctively
# ===========================================================================

@pytest.mark.parametrize("seed", [11, 29, 53, 101])
def test_validate_candidate_is_total_and_well_formed(step5_case, step5, seed):
    for text in _corpus(seed, 60):
        verdict = validate_candidate(
            candidate=step5.raw_candidate(step5_case.bundle, text, status="proposed"),
            context=step5_case.context,
        )
        # Total: always returns a verdict, never raises.
        assert isinstance(verdict, CandidateValidation)
        codes = list(verdict.failure_codes)
        # Conjunctive acceptance: passing means zero recorded failures.
        if verdict.passed:
            assert codes == []
        # No opaque rejection: a non-pass that is not an abstention must explain
        # itself with at least one failure code.
        elif not verdict.is_abstention:
            assert codes, f"opaque rejection for {text!r}"
        # Every recorded code is a real member of the enum vocabulary.
        for code in codes:
            assert code in {c.value for c in validators_mod.Code}


@pytest.mark.parametrize("seed", [31, 61])
def test_passing_candidates_never_carry_a_contradiction_or_addition(step5_case, step5, seed):
    """A stronger acceptance invariant: if the pipeline passes a candidate, none of
    the safety-critical gates may have fired."""
    for text in _corpus(seed, 80):
        verdict = validate_candidate(
            candidate=step5.raw_candidate(step5_case.bundle, text, status="proposed"),
            context=step5_case.context,
        )
        if verdict.passed:
            assert not verdict.contradiction_detected
            assert not verdict.unsupported_addition_detected
            assert verdict.evidence_grounded


# ===========================================================================
# Loop fuzz — always bounded, never fabricates
# ===========================================================================

class _RandomSource:
    def __init__(self, kit, bundle, rng):
        self._kit = kit
        self._bundle = bundle
        self._rng = rng

    def generate_for_prompt(self, bundle):
        return self._kit.candidate(self._bundle, _random_text(self._rng))


@pytest.mark.parametrize("seed", [12, 34, 56, 78])
def test_resolve_target_is_always_bounded_and_honest(step5_case, step5, seed):
    rng = random.Random(seed)
    for _ in range(25):
        cfg = CorrectorConfig(max_retries=rng.choice([0, 1, 2, 50, 10_000]))
        source = _RandomSource(step5, step5_case.bundle, rng)
        initial = step5.candidate(step5_case.bundle, _random_text(rng))
        resolution = resolve_target(
            target=step5_case.grounded,
            context=step5_case.context,
            bundle=step5_case.bundle,
            source=source,
            initial_candidate=initial,
            config=cfg,
        )
        # Bounded: never more attempts than allowed, never past the hard cap.
        assert resolution.attempt_count <= total_attempts(cfg)
        assert resolution.attempt_count <= ATTEMPT_HARD_CAP
        # Decision is always one of the three real outcomes.
        assert resolution.decision in _VALID_DECISIONS
        # Honest: accepted text exists only for an accepted target, and the
        # original is always preserved verbatim for anything else.
        if resolution.decision == TargetDecision.ACCEPTED.value:
            assert resolution.accepted_text
        else:
            assert resolution.accepted_text == ""
            assert resolution.original_text == step5_case.original_text


@pytest.mark.parametrize("seed", [90, 91])
def test_resolve_target_without_a_source_is_at_most_one_attempt(step5_case, step5, seed):
    rng = random.Random(seed)
    for _ in range(30):
        initial = step5.candidate(step5_case.bundle, _random_text(rng))
        resolution = resolve_target(
            target=step5_case.grounded,
            context=step5_case.context,
            bundle=step5_case.bundle,
            source=None,
            initial_candidate=initial,
        )
        # With no generator there is nothing to retry with.
        assert resolution.attempt_count <= 1
        assert resolution.decision in _VALID_DECISIONS


# ===========================================================================
# Scope guards — Step 5 stays inside Step 5 (AST, not prose)
# ===========================================================================

def _source_of(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


def _identifiers(module) -> set:
    tree = ast.parse(_source_of(module))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
    return names


def _absolute_imports(module) -> List[str]:
    """Top-level (non-relative) imported module names."""
    tree = ast.parse(_source_of(module))
    out: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append(node.module)
    return out


# The only absolute dependencies the Step 5 core is permitted: the standard
# library. Everything else it needs is reached by relative import inside the
# corrector package, which keeps the ML stack, the reference repo, the canonical
# schemas and the sibling agents out of the validation/retry core.
_ALLOWED_STDLIB = {
    "__future__",
    "dataclasses",
    "typing",
    "enum",
    "hashlib",
    "difflib",
    "re",
    "unicodedata",
    "math",
    "collections",
    "itertools",
    "functools",
    "string",
}


@pytest.mark.parametrize("module", STEP5_MODULES)
def test_step5_core_imports_only_stdlib_absolutely(module):
    for name in _absolute_imports(module):
        root = name.split(".")[0]
        assert root in _ALLOWED_STDLIB, f"{module.__name__} imports {name} absolutely"


@pytest.mark.parametrize("module", STEP5_MODULES)
def test_step5_core_never_imports_forbidden_neighbours(module):
    for name in _absolute_imports(module):
        assert "Corrector_agent" not in name  # the reference repo
        assert not name.startswith("app.") and name != "app"  # the FastAPI app
        assert "verifier_agent" not in name  # no direct Verifier import
        assert not name.startswith("orchestration")  # schemas boundary is the adapter


@pytest.mark.parametrize("module", STEP5_MODULES)
def test_step5_core_has_no_judge_memory_or_reverification(module):
    names = _identifiers(module)
    for forbidden in (
        "JudgeAgent",
        "MemoryAgent",
        "judge",
        "memory",
        "reverify",
        "reverification",
        "reconstruct",
        "splice",
    ):
        assert forbidden not in names


@pytest.mark.parametrize("module", [validation_mod, validators_mod, retry_mod])
def test_step5_pipeline_modules_contain_no_assert_statements(module):
    # Validation must fail closed by RETURNING a verdict, never by asserting: an
    # assert is a raise, and a raised gate would crash the pipeline instead of
    # recording VALIDATION_ERROR.
    tree = ast.parse(_source_of(module))
    assert [n for n in ast.walk(tree) if isinstance(n, ast.Assert)] == []
