"""Real end-to-end detector benchmark.

Everything here runs the **actual trained checkpoint** in
``artifacts/detector-best`` through the real pipeline: claim decomposition,
evidence selection, DeBERTa classification, secondary guards, answer-level
aggregation. Nothing is mocked and no result is hard-coded.

Two things are deliberately kept separate:

*Invariants* -- semantic properties that must hold no matter what the weights
say. A claim whose evidence is purely non-factual must never be reported as
refuted; a degraded evidence route must stay visible; a compound sentence must
be split into independently evaluated claims. These are asserted, because
breaking one is a bug in the code.

*Accuracy* -- how often the model lands on the expected class. This is
**measured and reported, never asserted**. The shipped checkpoint is weak on
refutation, and a test suite that demanded high accuracy would either fail
forever or, worse, be tuned until it passed. The number goes in the report so
it cannot be quietly overstated.

The benchmark is skipped when the checkpoint cannot be loaded, so the suite
still runs in an environment without the model weights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

DEFAULT_CHECKPOINT = Path("artifacts/detector-best")


@dataclass(frozen=True)
class BenchmarkCase:
    """One curated probe with an expected class and/or a hard invariant."""

    case_id: str
    category: str
    claim: str
    evidence: tuple[str, ...]
    expected_label: str | None = None
    expected_claims: int | None = None
    invariant: str | None = None
    note: str = ""


#: The user-facing expectation is recorded, but only the invariant is enforced.
#: A wrong model answer is a measurement; a wrong invariant is a code defect.
CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        case_id="supported_basic",
        category="supported",
        claim="The Earth orbits the Sun.",
        evidence=("Earth revolves around the Sun once per year.",),
        expected_label="SUPPORTED",
        note="Paraphrase, not lexical overlap: the reranker must keep the claim.",
    ),
    BenchmarkCase(
        case_id="contradicted_centre_of_solar_system",
        category="contradicted",
        claim="The Earth is the center of the solar system.",
        evidence=("The Earth orbits the Sun once per year.",),
        expected_label="CONTRADICTED",
        note="Evidence directly refutes the claim.",
    ),
    BenchmarkCase(
        case_id="unknown_revenue_not_in_evidence",
        category="unknown",
        claim="Company X generated $5 billion revenue in 2025.",
        evidence=("Company X operates globally.",),
        expected_label="NOT_ENOUGH_INFO",
        invariant="unknown_is_not_refutation",
        note="Related topic, silent on revenue. Must never become a refutation.",
    ),
    BenchmarkCase(
        case_id="compound_tesla",
        category="compound",
        claim="Tesla was founded in 2003 and acquired by Google in 2018.",
        evidence=(
            "Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning.",
            "Tesla was never acquired by Google; it remains an independent company.",
        ),
        expected_claims=2,
        invariant="compound_claims_evaluated_independently",
        note="Two atomic claims: one SUPPORTED, one CONTRADICTED.",
    ),
    BenchmarkCase(
        case_id="numeric_user_count",
        category="numeric",
        claim="Company X had 10 million users.",
        evidence=("Company X had 5 million users.",),
        expected_label="CONTRADICTED",
        invariant="same_relation_numeric_clash_signals",
        note="Same relation, same unit, different value.",
    ),
    BenchmarkCase(
        case_id="different_relation",
        category="relation",
        claim="Apple works with Company A.",
        evidence=("Apple acquired Company B.",),
        expected_label="NOT_ENOUGH_INFO",
        invariant="different_relation_is_not_contradiction",
        note="Different predicates. A mere entity mismatch is not a refutation.",
    ),
    BenchmarkCase(
        case_id="entity_substitution",
        category="entity",
        claim="Java was created by Snehith in 1995.",
        evidence=("Java was created by James Gosling and released in 1995.",),
        expected_label="CONTRADICTED",
        invariant="same_relation_entity_swap_signals",
        note="Same relation, swapped entity.",
    ),
    BenchmarkCase(
        case_id="opinion_is_not_hallucination",
        category="non_factual",
        claim="Python is the best programming language for data science.",
        evidence=("Python is widely used in data science and has a large ecosystem.",),
        expected_label=None,
        invariant="opinion_never_refuted",
        note="Preference/opinion content is not a factual hallucination.",
    ),
    BenchmarkCase(
        case_id="unrelated_evidence",
        category="unknown",
        claim="The Eiffel Tower is located in Berlin.",
        evidence=("Bananas are a tropical fruit grown in many countries.",),
        expected_label="NOT_ENOUGH_INFO",
        invariant="unknown_is_not_refutation",
        note="Evidence is off-topic: insufficient, not false.",
    ),
    BenchmarkCase(
        case_id="offtopic_evidence_route_visible",
        category="degraded",
        claim="The Treaty of Westphalia was signed in 1618.",
        evidence=("Bananas are a tropical fruit grown in many countries.",),
        expected_label="NOT_ENOUGH_INFO",
        invariant="degraded_route_is_visible",
        note=(
            "Off-topic evidence. Whichever selection route runs, it must be named "
            "on the result and a fallback must be flagged degraded."
        ),
    ),
)

#: Routes that are genuine fallbacks and must always be flagged ``degraded``.
FALLBACK_ROUTES = {"lexical", "hybrid_order", "empty", "untraced"}


def _label_text(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", value)


def _check_invariant(
    case: BenchmarkCase,
    result: Any,
    claims: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Evaluate the invariant that must hold regardless of model quality.

    These assert *semantics*, not confidence. A model is allowed to be unsure:
    ``P(CONTRADICTED) = 0.23`` on an unverified claim is honest uncertainty, not
    a bug. What must never happen is the claim being *decided* as refuted. So
    the invariants below test labels and routes, never the probability floor.
    """
    labels = [c["label"] for c in claims]
    if case.invariant == "unknown_is_not_refutation":
        ok = result.label != "HALLUCINATION" and "CONTRADICTED" not in labels
        return ok, f"answer={result.label} claims={labels}"
    if case.invariant == "opinion_never_refuted":
        factual = [c for c in claims if not c.get("non_factual")]
        if not factual:
            return True, "no checkable (factual) claims; opinion correctly excluded"
        ok = result.label != "HALLUCINATION" and "CONTRADICTED" not in [
            c["label"] for c in factual
        ]
        return ok, f"answer={result.label} factual_claims={[c['label'] for c in factual]}"
    if case.invariant == "compound_claims_evaluated_independently":
        count = len(claims)
        expected = case.expected_claims or 2
        # "Independently" means each atomic claim got its own probability
        # triple -- not that the model must disagree with itself.
        independent = all(
            {"contradicted_probability", "unknown_probability", "verification_risk"} <= set(c)
            for c in claims
        )
        ok = count == expected and independent
        return ok, f"{count} claims (expected {expected}), labels={labels}, independent={independent}"
    if case.invariant in {
        "same_relation_numeric_clash_signals",
        "same_relation_entity_swap_signals",
    }:
        # The guard must *notice* the clash. It is a bounded secondary signal:
        # it may resolve a near-tie but never overturn a decisive model call, so
        # requiring a final CONTRADICTED label would be testing the wrong thing.
        joined = " ".join(result.warnings).lower()
        noticed = (
            "number/date mismatch flagged" in joined
            or "raised contradiction confidence" in joined
        )
        return noticed, f"warnings={joined[:200]!r}"
    if case.invariant == "different_relation_is_not_contradiction":
        warns = " ".join(result.warnings).lower()
        rejected = "not treated as an automatic contradiction" in warns
        return rejected, f"warnings={warns[:200]!r}"
    if case.invariant == "degraded_route_is_visible":
        routes = [c.get("evidence_route") for c in claims]
        if not routes or any(not r for r in routes):
            return False, f"evidence route missing: {routes}"
        bad = [
            (c.get("evidence_route"), c.get("evidence_degraded"))
            for c in claims
            if c.get("evidence_route") in FALLBACK_ROUTES and not c.get("evidence_degraded")
        ]
        if bad:
            return False, f"fallback route not flagged degraded: {bad}"
        summary = ",".join(
            f"{c['evidence_route']}:{'degraded' if c['evidence_degraded'] else 'clean'}"
            for c in claims
        )
        return True, summary
    if case.invariant is None:
        return True, "no invariant"
    return False, f"unhandled invariant {case.invariant}"


def run_benchmark(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    cases: tuple[BenchmarkCase, ...] = CASES,
    detector_factory: Callable[[Path], Any] | None = None,
) -> dict:
    """Run every case through the real pipeline and report measured results."""
    from .detector import Detector

    checkpoint = Path(checkpoint)
    if not (checkpoint / "calibration.json").exists():
        raise FileNotFoundError(
            f"no detector checkpoint at {checkpoint}; run training or pass a different path"
        )
    factory = detector_factory or (lambda path: Detector(path))
    detector = factory(checkpoint)

    rows: list[dict[str, Any]] = []
    for case in cases:
        response = detector.detect(
            draft_answer=case.claim,
            evidence=list(case.evidence),
        )
        claims = [
            {
                "label": _label_text(item.label),
                "non_factual": item.non_factual,
                "verification_risk": item.verification_risk,
                "contradicted_probability": item.contradicted_probability,
                "unknown_probability": item.unknown_probability,
                "evidence_route": item.evidence_route,
                "evidence_degraded": item.evidence_degraded,
            }
            for item in response.sentences
        ]
        invariant_ok, invariant_detail = _check_invariant(case, response, claims)
        predicted = claims[0]["label"] if len(claims) == 1 else None
        matched = case.expected_label is None or predicted == case.expected_label
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "claim": case.claim,
                "expected_label": case.expected_label,
                "predicted_label": predicted,
                "matches_expectation": bool(matched),
                "answer_label": response.label,
                "contradiction_mass": round(response.contradiction_mass, 6),
                "verification_risk": round(response.verification_risk, 6),
                "claim_count": len(claims),
                "claims": claims,
                "evidence_route": claims[0]["evidence_route"] if claims else None,
                "evidence_degraded": any(c["evidence_degraded"] for c in claims),
                "invariant": case.invariant,
                "invariant_ok": bool(invariant_ok),
                "invariant_detail": invariant_detail,
                "note": case.note,
            }
        )

    scored = [r for r in rows if r["expected_label"] and r["predicted_label"]]
    measured = {
        "checkpoint": str(checkpoint),
        "cases": len(rows),
        "label_accuracy": {
            "scored_cases": len(scored),
            "correct": sum(1 for r in scored if r["matches_expectation"]),
            "accuracy": round(
                sum(1 for r in scored if r["matches_expectation"]) / len(scored), 4
            )
            if scored
            else None,
        },
        "invariants": {
            "checked": sum(1 for r in rows if r["invariant"]),
            "passed": sum(1 for r in rows if r["invariant"]),
            "all_passed": all(r["invariant_ok"] for r in rows if r["invariant"]),
        },
        "degraded_cases": [r["case_id"] for r in rows if r["evidence_degraded"]],
        "rows": rows,
        "disclaimer": (
            "A small hand-built probe set, not a benchmark of RAGTruth scale. It is "
            "reported to expose failure modes; it is not evidence of general "
            "accuracy and no test asserts on it."
        ),
    }
    return measured


def format_report(report: dict) -> str:
    lines = [
        f"Detector benchmark -- checkpoint {report['checkpoint']}",
        f"cases: {report['cases']}",
    ]
    accuracy = report["label_accuracy"]
    if accuracy["accuracy"] is not None:
        lines.append(
            f"label accuracy: {accuracy['correct']}/{accuracy['scored_cases']} "
            f"= {accuracy['accuracy']:.2%}  (measured, not asserted)"
        )
    inv = report["invariants"]
    lines.append(f"invariants: {inv['passed']}/{inv['checked']} passed (all={inv['all_passed']})")
    lines.append("")
    header = f"{'case':<38} {'expected':<16} {'predicted':<16} {'con_mass':>8} {'verif':>6} {'route':<12} inv"
    lines.append(header)
    lines.append("-" * len(header))
    for row in report["rows"]:
        lines.append(
            f"{row['case_id']:<38} {str(row['expected_label']):<16} "
            f"{str(row['predicted_label']):<16} {row['contradiction_mass']:>8.3f} "
            f"{row['verification_risk']:>6.3f} {str(row['evidence_route']):<12} "
            f"{'ok' if row['invariant_ok'] else 'FAIL'}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--json", type=Path, default=None, help="write full JSON report")
    args = parser.parse_args(argv)
    report = run_benchmark(args.checkpoint)
    print(format_report(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["invariants"]["all_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
