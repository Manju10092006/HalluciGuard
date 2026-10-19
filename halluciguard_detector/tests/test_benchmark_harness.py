"""Benchmark invariants and evidence-strategy selection.

These test the *harness*, not the model: no checkpoint is loaded, so they run
fast and they pin the semantic contracts that must hold no matter how good or
bad the checkpoint is. Model quality is reported by ``benchmark.py`` as a
measurement, never asserted here.
"""
from __future__ import annotations

import pytest

from halluciguard_detector.benchmark import (
    FALLBACK_ROUTES,
    BenchmarkCase,
    _check_invariant,
    _label_text,
)
from halluciguard_detector.evidence_alignment import PROBES, STRATEGIES, _select
from halluciguard_detector.schemas import ClaimLabel


class _Result:
    """Minimal stand-in for ``DetectResponse``."""

    def __init__(self, label="NO_HALLUCINATION", warnings=None, contradiction_mass=0.0):
        self.label = label
        self.warnings = warnings or []
        self.contradiction_mass = contradiction_mass
        self.verification_risk = 0.0


def _claim(label, *, non_factual=False, route="hybrid", degraded=False, **extra):
    payload = {
        "label": label,
        "non_factual": non_factual,
        "verification_risk": 0.5,
        "contradicted_probability": 0.2,
        "unknown_probability": 0.3,
        "evidence_route": route,
        "evidence_degraded": degraded,
    }
    payload.update(extra)
    return payload


def _case(invariant):
    return BenchmarkCase(
        case_id="c",
        category="cat",
        claim="a claim",
        evidence=("some evidence",),
        invariant=invariant,
    )


# ------------------------------------------------------------------ invariants


def test_unknown_must_never_be_reported_as_refutation():
    case = _case("unknown_is_not_refutation")
    ok, _ = _check_invariant(case, _Result(), [_claim("NOT_ENOUGH_INFO")])
    assert ok
    # P(CONTRADICTED) > 0 is honest uncertainty and must not fail the invariant.
    ok, _ = _check_invariant(case, _Result(contradiction_mass=0.234), [_claim("NOT_ENOUGH_INFO")])
    assert ok
    # Being *decided* as refuted is the real failure.
    ok, _ = _check_invariant(case, _Result(label="HALLUCINATION"), [_claim("CONTRADICTED")])
    assert not ok


def test_opinion_is_never_refuted():
    case = _case("opinion_never_refuted")
    # A purely non-factual claim is excluded from checking entirely.
    ok, detail = _check_invariant(case, _Result(), [_claim("NOT_ENOUGH_INFO", non_factual=True)])
    assert ok and "excluded" in detail
    # The real failure: a factual claim alongside the opinion gets refuted.
    ok, _ = _check_invariant(
        case,
        _Result(label="HALLUCINATION"),
        [_claim("SUPPORTED", non_factual=True), _claim("CONTRADICTED")],
    )
    assert not ok


def test_compound_claims_must_each_be_scored_independently():
    case = _case("compound_claims_evaluated_independently")
    # Two claims, both SUPPORTED: still independently evaluated, and the model
    # is allowed to agree with itself. Requiring differing labels would be
    # testing model quality, not semantics.
    ok, _ = _check_invariant(
        case, _Result(), [_claim("SUPPORTED"), _claim("SUPPORTED")]
    )
    assert ok
    # A single claim means decomposition failed.
    ok, _ = _check_invariant(case, _Result(), [_claim("SUPPORTED")])
    assert not ok


def test_guards_must_notice_a_clash_without_being_able_to_manufacture_one():
    numeric = _case("same_relation_numeric_clash_signals")
    entity = _case("same_relation_entity_swap_signals")
    ok, _ = _check_invariant(numeric, _Result(warnings=["Number/date mismatch flagged between claim and evidence: 1996"]), [])
    assert ok
    ok, _ = _check_invariant(entity, _Result(warnings=["Named-entity conflict on a shared relation raised contradiction confidence for one claim."]), [])
    assert ok
    # Silence means the guard did not fire.
    ok, _ = _check_invariant(numeric, _Result(warnings=[]), [])
    assert not ok


def test_different_relation_must_be_reported_as_not_a_contradiction():
    case = _case("different_relation_is_not_contradiction")
    ok, _ = _check_invariant(
        case, _Result(warnings=["Named-entity mismatch noted; relation differs, so it is not treated as an automatic contradiction."]), []
    )
    assert ok
    ok, _ = _check_invariant(case, _Result(warnings=[]), [])
    assert not ok


def test_degraded_route_must_be_visible_and_flagged():
    case = _case("degraded_route_is_visible")
    ok, _ = _check_invariant(case, _Result(), [_claim("SUPPORTED", route="lexical", degraded=True)])
    assert ok
    # A fallback route that is not flagged degraded is the bug this checks.
    ok, _ = _check_invariant(case, _Result(), [_claim("SUPPORTED", route="lexical", degraded=False)])
    assert not ok
    # A missing route is never acceptable.
    ok, _ = _check_invariant(case, _Result(), [_claim("SUPPORTED", route=None)])
    assert not ok
    # A clean route legitimately reports degraded=False.
    ok, _ = _check_invariant(case, _Result(), [_claim("SUPPORTED", route="hybrid", degraded=False)])
    assert ok


def test_fallback_routes_are_the_documented_ones():
    assert "lexical" in FALLBACK_ROUTES
    assert "hybrid" not in FALLBACK_ROUTES, "production route is not a fallback"


def test_unknown_invariant_is_reported_not_silently_passed():
    case = _case("something_new")
    ok, detail = _check_invariant(case, _Result(), [])
    assert not ok
    assert "unhandled invariant" in detail


def test_label_text_unwraps_enums():
    assert _label_text(ClaimLabel.CONTRADICTED) == "CONTRADICTED"
    assert _label_text("SUPPORTED") == "SUPPORTED"
    assert _label_text(None) is None


# ------------------------------------------------------- evidence strategies


def test_training_strategy_returns_one_joined_evidence_string():
    """Training pairs the classifier with concatenated snippets, not one."""
    joined = _select("training_lexical_joined", "Java was created by James Gosling in 1995.", ("Java was created by James Gosling at Sun Microsystems in 1995.", "Python was created by Guido van Rossum."))
    assert len(joined) == 1
    assert "Java was created by James Gosling" in joined[0]


def test_single_snippet_strategies_return_at_most_one_snippet():
    corpus = ("Java was created by James Gosling in 1995.", "Python was created by Guido van Rossum.")
    for strategy in ("lexical_top1", "hybrid_top1", "hybrid_rerank"):
        snippets = _select(strategy, "Java was created by James Gosling in 1995.", corpus)
        assert len(snippets) <= 1, f"{strategy} must hand the classifier one snippet"


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        _select("nope", "claim", ("doc",))


def test_probes_declare_expected_labels_from_the_real_vocabulary():
    assert PROBES, "the alignment experiment must not be empty"
    for probe in PROBES:
        if probe.expected_label is not None:
            assert probe.expected_label in {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"}
    assert "training_lexical_joined" in STRATEGIES
    assert "hybrid_rerank" in STRATEGIES
