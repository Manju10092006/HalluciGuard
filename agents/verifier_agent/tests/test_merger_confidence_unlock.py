"""Regression tests for the claim-merger confidence unlock.

Root cause proven earlier: when a single suspicious claim is decomposed into
several atomic sub-claims, ``ClaimMerger.merge_results`` used to (a) DROP the
evidence-derived ``confidence_score`` entirely and (b) weighted-average the
support signal with tiny ``trust + 0.1`` weights. A strongly-grounded sub-claim
(0.97 NLI entailment) was therefore washed out by weaker siblings, collapsing a
well-grounded answer to ~0.20 confidence and forcing the Judge into
VERIFY_AGAIN -> human_review even for correct answers.

These tests pin the fixed behaviour:
  1. confidence_score is propagated through the merge (never silently dropped).
  2. a strong sub-claim is NOT diluted below a weak sibling (max-biased blend).
"""
from __future__ import annotations

from claims import ClaimMerger


def _report(support: float, contradict: float, trust: float, confidence: float):
    return {
        "scores": {
            "support_score": support,
            "contradiction_score": contradict,
            "trust_score": trust,
            "confidence_score": confidence,
        },
        "evidence_items": [],
    }


def test_confidence_score_is_propagated_not_dropped():
    merger = ClaimMerger()
    merged = merger.merge_results([_report(0.9, 0.05, 0.85, 0.91)])
    scores = merged["scores"]
    assert "confidence_score" in scores, "merger must propagate confidence_score"
    # single strong sub-claim: confidence must stay high, not collapse to ~0.2
    assert scores["confidence_score"] >= 0.85


def test_strong_subclaim_not_diluted_by_weak_siblings():
    """A 0.97-grounded sub-claim mixed with weak-but-present siblings must keep
    the aggregate confidence high (dominant-signal blend), not average down."""
    merger = ClaimMerger()
    reports = [
        _report(0.95, 0.02, 0.9, 0.95),   # strongly grounded
        _report(0.30, 0.05, 0.2, 0.30),   # weak sibling
        _report(0.25, 0.05, 0.15, 0.25),  # weak sibling
    ]
    merged = merger.merge_results(reports)
    conf = merged["scores"]["confidence_score"]
    sup = merged["scores"]["support_score"]
    # Old pure-mean would give ~0.50; dominant blend keeps the strong claim leading.
    assert conf >= 0.70, f"strong sub-claim diluted to {conf}"
    assert sup >= 0.70, f"support diluted to {sup}"
    # verdict must still be verified (support dominates, no real contradiction)
    assert merged["verdict"] == "verified"


def test_all_weak_stays_low():
    """Sanity: genuinely weak evidence must NOT be inflated by the blend."""
    merger = ClaimMerger()
    reports = [_report(0.10, 0.05, 0.05, 0.10), _report(0.12, 0.04, 0.05, 0.12)]
    merged = merger.merge_results(reports)
    assert merged["scores"]["confidence_score"] < 0.30
    assert merged["verdict"] == "unverified"


def test_empty_reports_safe():
    merger = ClaimMerger()
    assert merger.merge_results([]) == {}
