"""A small hard-negative probe set for the contradiction axis.

These are **evaluation cases, not production rules**. Nothing here is consulted
by ``Detector.detect``; no case changes a label or a probability at runtime.
They exist to answer "what kind of contradiction does the model actually miss?"
with concrete examples instead of a recall number.

The taxonomy follows the ways a claim can be refuted by an evidence sentence:

``DIRECT``
    Same predicate, same slot, different value. The easy case.
``DIFFERENT_YEAR`` / ``DIFFERENT_NUMBER``
    A quantity or a date is replaced. The slots match, the value does not.
``DIFFERENT_ENTITY``
    A named entity is swapped while the relation is identical.
``DIFFERENT_RELATION``
    A different predicate on the same entities ("partnered" vs "acquired").
    This is **not** a contradiction: the evidence does not refute the claim,
    it simply does not speak to it. Included precisely so a model that
    over-fires on relation mismatch is caught.
``NEGATION``
    The evidence asserts the negative of the claim.
``TEMPORAL``
    The fact is true but at a different time, so the claim is unstated for the
    period it asserts.
``MULTI_HOP``
    Refuting the claim needs two evidence sentences, not one.

The expected label is the *human* judgement of whether the evidence refutes the
claim, which is deliberately not always ``CONTRADICTED``. A probe that only
contained contradictions would reward a model for saying CONTRADICTED to
everything.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


@dataclass(frozen=True)
class HardCase:
    """One claim, the single production-shaped evidence snippet, and truth."""

    case_id: str
    claim: str
    evidence: str
    expected: str
    #: Retrieval pool the evidence would be selected from, so the case can be
    #: driven through the *real* production selector rather than a stub.
    distractors: Sequence[str] = ()
    note: str = ""


_EARTH = (
    "The Earth revolves around the Sun once per year, completing one orbit in about 365.25 days.",
    "The Sun sits at the centre of the solar system and the Earth orbits it.",
)
_TESLA = (
    "Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning in San Francisco.",
    "Tesla remains an independent American company and has never been acquired by another firm.",
)
_JAVA = (
    "Java was created by James Gosling at Sun Microsystems and released in 1995.",
    "Java is a high-level programming language designed to have few implementation dependencies.",
)
_APPLE = (
    "Apple Inc. is an American multinational technology company headquartered in Cupertino.",
    "Apple has partnered with a wide range of companies across its supply chain.",
)
_SAFARI = (
    "Safari is a web browser developed by Apple and first released in 2003.",
    "Safari is available on macOS, iOS and iPadOS.",
)
_TREATY = (
    "The Treaty of Westphalia was signed in 1618, ending the Thirty Years' War in Europe.",
    "The Peace of Westphalia reshaped the political map of Europe for centuries.",
)
_POPULATION = (
    "Company X serves more than 50 million customers across 50 countries.",
    "Company X expanded its service footprint to 50 countries.",
)

HARD_CASES: tuple[HardCase, ...] = (
    # ---- direct contradiction: the case that must work -------------------
    HardCase(
        "direct_year",
        "Tesla was founded in 1995.",
        _TESLA[0],
        CONTRADICTED,
        note="Same predicate, same slot, different year.",
    ),
    HardCase(
        "direct_number",
        "Company X serves 5 million customers.",
        "Company X serves more than 50 million customers across 50 countries.",
        CONTRADICTED,
        note="Same unit, different magnitude.",
    ),
    HardCase(
        "direct_entity",
        "Java was created by Snehith Babu.",
        _JAVA[0],
        CONTRADICTED,
        note="Same relation, different person.",
    ),
    # ---- supported -------------------------------------------------------
    HardCase(
        "supported_exact",
        "Java was created by James Gosling at Sun Microsystems in 1995.",
        _JAVA[0],
        SUPPORTED,
    ),
    HardCase(
        "supported_paraphrase",
        "Safari was first released in 2003.",
        _SAFARI[0],
        SUPPORTED,
    ),
    # ---- different year: NOT the same fact, so not a contradiction -------
    HardCase(
        "different_year_event",
        "Tesla was founded in 2003.",
        "Tesla remained an independent American company throughout the 2010s.",
        NOT_ENOUGH_INFO,
        note=(
            "The evidence mentions Tesla and a year, but about a different "
            "event. Treating this as a year conflict manufactures contradiction "
            "from an unrelated fact."
        ),
    ),
    # ---- different relation: evidence does not speak to the claim -------
    HardCase(
        "different_relation",
        "Apple acquired Company A in 2014.",
        "Apple has partnered with a wide range of companies across its supply chain.",
        NOT_ENOUGH_INFO,
        note=(
            "Partnered is not acquired. This is the case a naive entity-swap "
            "rule gets wrong."
        ),
    ),
    HardCase(
        "different_relation_same_entities",
        "Tesla was acquired by Google in 2018.",
        _TESLA[1],
        CONTRADICTED,
        note="The evidence explicitly denies the acquisition.",
    ),
    # ---- negation --------------------------------------------------------
    HardCase(
        "negation",
        "Safari runs on Windows.",
        "Safari is a web browser developed exclusively for macOS and iOS.",
        CONTRADICTED,
    ),
    # ---- temporal --------------------------------------------------------
    HardCase(
        "temporal_stale",
        "Tesla's headquarters is still in San Francisco as of 2026.",
        "Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning in San Francisco.",
        NOT_ENOUGH_INFO,
        note="True of the past, not stated for the asserted period.",
    ),
    # ---- multi-hop -------------------------------------------------------
    HardCase(
        "multi_hop",
        "The browser that Apple first released in 2003 is developed by Apple.",
        _SAFARI[1],
        NOT_ENOUGH_INFO,
        note=(
            "Supportable only by joining two evidence sentences; with a single "
            "snippet it is genuinely under-determined."
        ),
    ),
    # ---- off-topic retrieval --------------------------------------------
    HardCase(
        "retrieval_miss",
        "The Treaty of Westphalia was signed in 1618.",
        _TREATY[1],
        NOT_ENOUGH_INFO,
        note="Plausible but wrong snippet: same topic, no supporting date.",
    ),
    HardCase(
        "retrieval_miss_supported",
        "The Treaty of Westphalia was signed in 1618.",
        _TREATY[0],
        SUPPORTED,
    ),
)


def case_to_row(case: HardCase, *, select: bool = False) -> Dict[str, Any]:
    """Render a case as a dataset row using the canonical contract fields.

    By default the case's own evidence snippet is used verbatim, so the probe
    measures the *classifier*. With ``select=True`` the snippet is instead
    chosen by the real production selector from the case pool, which measures
    the whole claim+retrieval path and can turn an expected label into a
    retrieval failure -- reported as such rather than silently reclassified.
    """
    from .evidence_shapes import build_pool
    from .nli_input import MAX_EVIDENCE_SNIPPETS, enforce_single_snippet
    from .text import lexical_evidence
    from .training import ID_TO_LABEL

    if select:
        from .evidence import select_evidence

        snippets = select_evidence(case.claim, build_pool(case.evidence, case.distractors))
        evidence = enforce_single_snippet(snippets[0] if snippets else "")
    else:
        selected = lexical_evidence(
            case.claim,
            build_pool(case.evidence, case.distractors),
            limit=MAX_EVIDENCE_SNIPPETS,
        )
        evidence = enforce_single_snippet(selected[0] if selected else "")
    return {
        "id": f"hard:{case.case_id}",
        "claim": case.claim,
        "evidence": evidence,
        "label": case.expected,
        "label_id": {name: index for index, name in ID_TO_LABEL.items()}[case.expected],
        "granularity": "hard_negative",
        "note": case.note,
    }


def to_rows(
    cases: Sequence[HardCase] = HARD_CASES, *, select: bool = False
) -> List[Dict[str, Any]]:
    return [case_to_row(case, select=select) for case in cases]


def evaluate_cases(
    model_dir: Path,
    cases: Sequence[HardCase] = HARD_CASES,
    *,
    max_length: int = 256,
    batch_size: int = 16,
    select: bool = False,
) -> Dict[str, Any]:
    """Run the probe set through a real checkpoint and report every case.

    All cases are reported, including the ones the model gets right. A probe
    set that prints only failures invites cherry-picking, and this one is small
    enough (13 cases) to read in full.
    """
    import torch

    from .calibration import class_probabilities
    from .experiments import score_rows
    from .training import ID_TO_LABEL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = to_rows(cases, select=select)
    scored = score_rows(
        Path(model_dir), rows, device=device, max_length=max_length, batch_size=batch_size
    )
    from .calibration import load_calibration

    calibration = load_calibration(Path(model_dir) / "calibration.json")
    probabilities = class_probabilities(
        scored["logits"], float(calibration.get("temperature", 1.0))
    )
    predicted = probabilities.argmax(axis=1)

    results: List[Dict[str, Any]] = []
    correct = 0
    contradicted_total = 0
    contradicted_recalled = 0
    for case, row, prob, pred in zip(cases, rows, probabilities, predicted):
        name = ID_TO_LABEL[int(pred)]
        hit = name == case.expected
        correct += int(hit)
        if case.expected == CONTRADICTED:
            contradicted_total += 1
            contradicted_recalled += int(name == CONTRADICTED)
        results.append(
            {
                "case_id": case.case_id,
                "claim": case.claim,
                "evidence": row["evidence"],
                "expected": case.expected,
                "predicted": name,
                "correct": hit,
                "p_supported": round(float(prob[0]), 6),
                "p_contradicted": round(float(prob[1]), 6),
                "p_unknown": round(float(prob[2]), 6),
                "note": case.note,
            }
        )
    return {
        "checkpoint": str(model_dir),
        "cases": len(cases),
        "accuracy": round(correct / max(1, len(cases)), 4),
        "expected_contradicted": contradicted_total,
        "contradicted_detected": contradicted_recalled,
        "contradicted_detection_rate": (
            round(contradicted_recalled / contradicted_total, 4) if contradicted_total else None
        ),
        "note": (
            "A tiny, hand-written probe set. It is a diagnostic for failure "
            "modes, not a benchmark: 13 cases cannot support a performance "
            "claim, and the accuracy here is not comparable to the RAGTruth "
            "metrics."
        ),
        "results": results,
    }


def main(argv: List[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--select", action="store_true", help="run the production selector")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("reports/detector_v2_hard_negatives.json"))
    parser.add_argument("--rows-out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = evaluate_cases(
        args.checkpoint, max_length=args.max_length, select=args.select
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.rows_out:
        with args.rows_out.open("w", encoding="utf-8") as handle:
            for row in to_rows(select=args.select):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for entry in report["results"]:
        flag = "ok " if entry["correct"] else "MISS"
        print(
            f"{flag} {entry['case_id']:<32} expected={entry['expected']:<18} "
            f"predicted={entry['predicted']:<18} p(contra)={entry['p_contradicted']:.3f}"
        )
    print(
        f"\naccuracy {report['accuracy']} | contradicted detection "
        f"{report['contradicted_detected']}/{report['expected_contradicted']}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
