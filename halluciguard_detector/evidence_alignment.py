"""Train/runtime evidence-alignment experiment (P0).

There is a concrete distribution mismatch between how training pairs are built
and how the runtime pairs are formed.

**Training** (``data.py``) joins up to six lexical snippets into one string::

    evidence = " ".join(lexical_evidence(claim, [source_info]))

so the classifier reads a concatenated, topically-loose paragraph.

**Runtime** (``detector.py``) classifies ``(best, claim)`` where ``best`` is
``snippets[0]`` -- a *single* reranked snippet.

Same model, same claim, different evidence shape. This module measures what
that actually costs, using the real checkpoint, by running one fixed claim set
through four evidence strategies:

``training_lexical_joined``
    exactly what ``data.py`` writes to disk.
``lexical_top1``
    lexical retrieval, but only the best snippet.
``hybrid_top1``
    shared BM25 + dense fusion, no cross-encoder rerank.
``hybrid_rerank``
    the production path.

The output is a measured comparison: how often each strategy disagrees with
production, and by how much the triage score moves. No strategy is assumed to
be best -- the numbers decide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import halluciguard_detector.evidence as evidence_module
from .detector import LABELS
from .evidence import DEFAULT_POOL_K, DEFAULT_RERANK_TOP
from .text import lexical_evidence

STRATEGIES = (
    "training_lexical_joined",
    "lexical_top1",
    "hybrid_top1",
    "hybrid_rerank",
)

PRODUCTION_STRATEGY = "hybrid_rerank"

#: A small fixed pool so retrieval has real candidates to rank. Distractors are
#: deliberately topically close, because the whole question is whether a
#: weaker selector picks the wrong one.
CORPUS: tuple[str, ...] = (
    "The Earth revolves around the Sun once per year, completing one orbit in about 365.25 days.",
    "The Earth is not the centre of the solar system; the Sun sits at its centre.",
    "Bananas are a tropical fruit grown in many countries across Asia, Africa and Latin America.",
    "Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning in San Francisco.",
    "Tesla remains an independent American company and has never been acquired by Google.",
    "Apple Inc. is an American multinational technology company headquartered in Cupertino.",
    "Apple has partnered with a wide range of companies across its supply chain.",
    "Company X operates globally and serves customers in more than fifty countries.",
    "Java is a high-level programming language designed to have as few implementation dependencies as possible.",
    "Java was created by James Gosling at Sun Microsystems and released in 1995.",
    "Python is widely used in data science and has a large ecosystem of libraries.",
    "The Treaty of Westphalia was signed in 1618, ending the Thirty Years' War in Europe.",
)


@dataclass(frozen=True)
class Probe:
    """A claim to route through every evidence strategy."""

    probe_id: str
    claim: str
    expected_label: str | None = None
    note: str = ""


PROBES: tuple[Probe, ...] = (
    Probe("earth_orbits", "The Earth orbits the Sun.", "SUPPORTED"),
    Probe("earth_centre", "The Earth is the center of the solar system.", "CONTRADICTED"),
    Probe("java_creator", "Java was created by James Gosling in 1995.", "SUPPORTED"),
    Probe("java_creator_wrong", "Java was created by Snehith in 1995.", "CONTRADICTED"),
    Probe("tesla_founding", "Tesla was founded in 2003.", "SUPPORTED"),
    Probe("tesla_acquired", "Tesla was acquired by Google in 2018.", "CONTRADICTED"),
    Probe("apple_partner", "Apple works with Company A.", "NOT_ENOUGH_INFO"),
    Probe("treaty_year", "The Treaty of Westphalia was signed in 1618.", "SUPPORTED"),
    Probe("revenue_absent", "Company X generated $5 billion revenue in 2025.", "NOT_ENOUGH_INFO"),
)


def _select(strategy: str, claim: str, corpus: tuple[str, ...]) -> list[str]:
    """Return the evidence snippet list a given strategy would hand the model."""
    if strategy == "training_lexical_joined":
        # What data.py stores: up to six lexical snippets, concatenated by the
        # caller. Returned here as a single string so the classifier sees the
        # same thing it saw during training.
        joined = " ".join(lexical_evidence(claim, list(corpus)))
        return [joined] if joined else []
    if strategy == "lexical_top1":
        return lexical_evidence(claim, list(corpus), limit=1)
    if strategy == "hybrid_top1":
        # Read the singletons off the module, not by name: they are created
        # lazily by _ensure_verifier() and a `from ... import` at import time
        # would freeze them at None.
        if not evidence_module._ensure_verifier():
            return lexical_evidence(claim, list(corpus), limit=1)
        retriever = evidence_module._hybrid_retriever
        if retriever is None:
            return lexical_evidence(claim, list(corpus), limit=1)
        passages = [evidence_module._to_passage(t) for t in corpus]
        merged = retriever.retrieve(claim, passages, k=DEFAULT_POOL_K)
        return [p.snippet for p in merged[:1] if p.snippet]
    if strategy == "hybrid_rerank":
        if not evidence_module._ensure_verifier():
            return lexical_evidence(claim, list(corpus), limit=1)[:1]
        retriever = evidence_module._hybrid_retriever
        reranker = evidence_module._reranker
        if retriever is None or reranker is None:
            return lexical_evidence(claim, list(corpus), limit=DEFAULT_RERANK_TOP)[:1]
        passages = [evidence_module._to_passage(t) for t in corpus]
        merged = retriever.retrieve(claim, passages, k=DEFAULT_POOL_K)
        ranked = reranker.rerank(claim, merged[:DEFAULT_POOL_K], k=DEFAULT_RERANK_TOP)
        return [p.snippet for p in ranked if p.snippet][:1]
    raise ValueError(f"unknown strategy: {strategy}")


def run_evidence_alignment(
    checkpoint: Path = Path("artifacts/detector-best"),
    probes: tuple[Probe, ...] = PROBES,
    corpus: tuple[str, ...] = CORPUS,
) -> dict:
    """Run every probe through every strategy with the real checkpoint."""
    import torch

    from .calibration import apply_temperature
    from .detector import Detector

    checkpoint = Path(checkpoint)
    if not (checkpoint / "calibration.json").exists():
        raise FileNotFoundError(f"no detector checkpoint at {checkpoint}")
    detector = Detector(checkpoint)
    shared_ready = bool(
        evidence_module._ensure_verifier()
        and evidence_module._hybrid_retriever is not None
    )

    rows: list[dict[str, Any]] = []
    for probe in probes:
        row: dict[str, Any] = {
            "probe_id": probe.probe_id,
            "claim": probe.claim,
            "expected_label": probe.expected_label,
            "strategies": {},
        }
        for strategy in STRATEGIES:
            snippets = _select(strategy, probe.claim, corpus)
            if not snippets:
                row["strategies"][strategy] = {
                    "evidence": "",
                    "label": None,
                    "verification_risk": None,
                    "contradiction_probability": None,
                }
                continue
            encoded = detector.tokenizer(
                [snippets[0]],
                [probe.claim],
                padding=True,
                truncation="longest_first",
                max_length=detector.max_length,
                return_tensors="pt",
            ).to(detector.device)
            with torch.inference_mode():
                logits = detector.model(**encoded).logits
            probabilities = apply_temperature(logits, detector.temperature)[0].cpu().numpy()
            predicted = LABELS[int(probabilities.argmax())]
            row["strategies"][strategy] = {
                "evidence": snippets[0],
                "evidence_chars": len(snippets[0]),
                "label": getattr(predicted, "value", predicted),
                "verification_risk": round(float(probabilities[1] + probabilities[2]), 6),
                "contradiction_probability": round(float(probabilities[1]), 6),
                "unknown_probability": round(float(probabilities[2]), 6),
            }
        rows.append(row)

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "shared_retrieval_available": shared_ready,
        "probes": len(probes),
        "corpus_documents": len(corpus),
        "strategies": {},
        "rows": rows,
    }
    for strategy in STRATEGIES:
        agreements = []
        deltas = []
        correct = 0
        scored = 0
        for row in rows:
            entry = row["strategies"][strategy]
            production = row["strategies"][PRODUCTION_STRATEGY]
            if entry["label"] is None or production["label"] is None:
                continue
            agreements.append(entry["label"] == production["label"])
            deltas.append(
                abs(
                    (entry["verification_risk"] or 0.0)
                    - (production["verification_risk"] or 0.0)
                )
            )
            if row["expected_label"] and entry["label"]:
                scored += 1
                correct += int(entry["label"] == row["expected_label"])
        summary["strategies"][strategy] = {
            "label_agreement_with_production": round(sum(agreements) / len(agreements), 4)
            if agreements
            else None,
            "mean_abs_verification_risk_delta": round(sum(deltas) / len(deltas), 6)
            if deltas
            else None,
            "max_abs_verification_risk_delta": round(max(deltas), 6) if deltas else None,
            "label_accuracy_on_probe_set": round(correct / scored, 4) if scored else None,
            "scored_probes": scored,
        }

    train_vs_runtime = summary["strategies"]["training_lexical_joined"]
    summary["finding"] = (
        "Training pairs the classifier with up to six concatenated lexical snippets; "
        "runtime pairs it with one reranked snippet. "
        f"Label agreement between the two is {train_vs_runtime['label_agreement_with_production']} "
        f"with a mean verification-risk shift of "
        f"{train_vs_runtime['mean_abs_verification_risk_delta']}. "
        "That gap is the measured cost of the mismatch, on this probe set."
    )
    return summary


def format_alignment_report(summary: dict) -> str:
    lines = [
        f"Evidence alignment -- checkpoint {summary['checkpoint']}",
        f"probes: {summary['probes']}  corpus: {summary['corpus_documents']}  "
        f"shared retrieval available: {summary['shared_retrieval_available']}",
        "",
        f"{'strategy':<26} {'agree(prod)':>12} {'mean|dR|':>10} {'max|dR|':>9} {'probe acc':>11}",
        "-" * 72,
    ]
    for strategy, stats in summary["strategies"].items():
        lines.append(
            f"{strategy:<26} {str(stats['label_agreement_with_production']):>12} "
            f"{str(stats['mean_abs_verification_risk_delta']):>10} "
            f"{str(stats['max_abs_verification_risk_delta']):>9} "
            f"{str(stats['label_accuracy_on_probe_set']):>11}"
        )
    lines.append("")
    lines.append(summary["finding"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/detector-best"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = run_evidence_alignment(args.checkpoint)
    print(format_alignment_report(summary))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
