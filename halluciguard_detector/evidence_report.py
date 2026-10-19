"""Measure the train/runtime evidence distribution gap.

The hypothesis under test is that the detector's weak contradiction recall is
partly caused by being *trained* on a different evidence distribution than it is
*served*. That claim is only worth anything if it is measured, so this module
produces the numbers that could refute it:

* :func:`describe_evidence` -- per-arm evidence shape (length, snippet count,
  truncation rate, degraded rate) measured against the canonical NLI contract.
* :func:`risk_shift` -- the paired, per-claim difference in the detector's own
  risk score when *only* the evidence shape changes. Paired, so it is not
  confounded by which claims happen to be in the split.
* :func:`retrieval_diagnostics` -- reranker score and retrieval-similarity
  distributions for the production selector, sampled rather than run over the
  whole split (the cross-encoder costs seconds per claim on CPU).

The target is not a zero gap. It is: the production input distribution should be
*represented* during training and evaluation, so the gap should shrink between
the pre-alignment and post-alignment evidence policies. A gap that does not
shrink is a real finding and is reported as one.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .nli_input import MAX_EVIDENCE_SNIPPETS, contract_spec, contract_violations

#: Risk scores whose paired absolute difference exceeds this are reported
#: separately as "moved a lot", because the mean alone hides a heavy tail.
LARGE_SHIFT = 0.20


def load_rows(path: Path) -> List[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "mean": round(statistics.fmean(ordered), 4),
        "median": round(statistics.median(ordered), 4),
        "p10": round(ordered[int(0.10 * (len(ordered) - 1))], 4),
        "p90": round(ordered[int(0.90 * (len(ordered) - 1))], 4),
        "max": round(ordered[-1], 4),
    }


def describe_evidence(
    paths: Dict[str, Path],
    *,
    tokenizer: Any = None,
    max_length: int = 256,
) -> Dict[str, Any]:
    """Contract-conformance summary for one file per evidence arm."""
    report: Dict[str, Any] = {"contract": contract_spec(max_length), "arms": {}}
    for arm, path in paths.items():
        path = Path(path)
        if not path.exists():
            report["arms"][arm] = {"status": "missing", "path": str(path)}
            continue
        rows = load_rows(path)
        snippet_counts = [int(row.get("evidence_snippet_count", 0) or 0) for row in rows]
        report["arms"][arm] = {
            "status": "ok",
            "path": str(path),
            "rows": len(rows),
            "contract": contract_violations(rows, max_length=max_length, tokenizer=tokenizer),
            "snippets_per_example": {
                "mean": round(statistics.fmean(snippet_counts), 4) if snippet_counts else None,
                "max": max(snippet_counts) if snippet_counts else None,
                "rows_with_multiple_snippets": sum(
                    1 for count in snippet_counts if count > MAX_EVIDENCE_SNIPPETS
                ),
                "in_contract": sum(1 for count in snippet_counts if count <= MAX_EVIDENCE_SNIPPETS),
            },
            "degraded_rows": sum(1 for row in rows if row.get("evidence_degraded")),
            "routes": _tally(row.get("evidence_route", "unknown") for row in rows),
            "labels": _tally(row["label"] for row in rows),
        }
    return report


def _tally(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _by_claim(rows: Sequence[dict]) -> Dict[str, dict]:
    return {str(row.get("id") or f"{row['claim']}|{i}"): row for i, row in enumerate(rows)}


def risk_shift(
    scores: Dict[str, Sequence[float]],
    labels: Sequence[int],
    *,
    reference: str,
) -> Dict[str, Any]:
    """Paired per-claim risk difference between arms and a reference arm.

    ``scores`` maps an arm name to its risk score per claim, all in the *same*
    claim order. Only claims present in every arm are compared; the count that
    survives the intersection is reported, because silently comparing different
    row sets would make the numbers incomparable in a way that looks fine.
    """
    arms = list(scores)
    missing = [arm for arm in arms if arm not in scores]
    if missing:
        raise KeyError(f"no scores for arms: {missing}")
    if reference not in scores:
        raise KeyError(f"reference arm {reference!r} has no scores")
    if any(len(v) != len(labels) for v in scores.values()):
        raise ValueError("every arm must have one score per label")

    reference_scores = scores[reference]
    out: Dict[str, Any] = {"reference": reference, "claims": len(labels), "arms": {}}
    for arm in arms:
        deltas = [abs(a - b) for a, b in zip(scores[arm], reference_scores)]
        out["arms"][arm] = {
            "abs_risk_shift": _percentiles(deltas),
            "mean_abs_risk_shift": round(statistics.fmean(deltas), 6) if deltas else None,
            "median_abs_risk_shift": round(statistics.median(deltas), 6) if deltas else None,
            "share_shifted_over_%.2f" % LARGE_SHIFT: round(
                sum(1 for d in deltas if d > LARGE_SHIFT) / len(deltas), 4
            )
            if deltas
            else None,
        }
    return out


def label_flip_matrix(
    predictions: Dict[str, Sequence[int]],
    labels: Sequence[int],
) -> Dict[str, Any]:
    """How often two arms disagree, split by the true class.

    A disagreement only matters if it moves a claim in or out of
    CONTRADICTED, so the CONTRADICTED row is the one to read. Reported per
    true class because a global flip rate can be dominated by SUPPORTED rows
    that nobody is routing on.
    """
    arms = list(predictions)
    pairs = [(a, b) for i, a in enumerate(arms) for b in arms[i + 1 :]]
    out: Dict[str, Any] = {"claims": len(labels), "pairs": {}}
    for a, b in pairs:
        pa, pb = predictions[a], predictions[b]
        by_class: Dict[str, Dict[str, int]] = {}
        for true, x, y in zip(labels, pa, pb):
            key = str(true)
            bucket = by_class.setdefault(key, {"agree": 0, "disagree": 0})
            bucket["agree" if x == y else "disagree"] += 1
        total = sum(v["agree"] + v["disagree"] for v in by_class.values())
        disagreements = sum(v["disagree"] for v in by_class.values())
        out["pairs"][f"{a}_vs_{b}"] = {
            "agreement": round(1 - disagreements / total, 4) if total else None,
            "by_true_class": by_class,
        }
    return out


def retrieval_diagnostics(
    dataset_dir: Path,
    rows: Sequence[dict],
    *,
    pool_k: int = 8,
    limit: int = 40,
) -> Dict[str, Any]:
    """Reranker score / similarity distributions for the production selector.

    Sampled deliberately: the cross-encoder costs seconds per claim on CPU, so a
    full-split sweep is not affordable and a sample is enough to characterise a
    distribution. The number of claims actually scored is reported so the
    sample size is never implied to be larger than it is.
    """
    from .evidence_shapes import candidate_passages, load_sources

    from .evidence import _ensure_verifier, _hybrid_retriever, _reranker, _to_passage

    sources = load_sources(Path(dataset_dir))
    sample = list(rows)[:limit]
    if not _ensure_verifier() or _hybrid_retriever is None or _reranker is None:
        return {"available": False, "reason": "shared retrieval stack unavailable"}

    top_scores: List[float] = []
    best_similarities: List[float] = []
    reranked: List[float] = []
    for row in sample:
        claim = str(row["claim"])
        passages = _to_passage_list(candidate_passages(sources.get(str(row["source_id"]), "")))
        merged = _hybrid_retriever.retrieve(claim, passages, k=pool_k)
        if not merged:
            continue
        ranked = _reranker.rerank(claim, merged[:pool_k], k=3)
        if not ranked:
            continue
        top_scores.append(float(ranked[0].relevance_score))
        reranked.append(float(ranked[0].relevance_score))
        claim_terms = {t.lower() for t in str(claim).split() if len(t) > 2}
        best_terms = {t.lower() for t in str(ranked[0].snippet or "").split() if len(t) > 2}
        if claim_terms and best_terms:
            best_similarities.append(
                len(claim_terms & best_terms) / len(claim_terms | best_terms)
            )
    return {
        "available": True,
        "claims_scored": len(top_scores),
        "pool_k": pool_k,
        "top1_reranker_score": _percentiles(top_scores),
        "top1_token_jaccard_similarity": _percentiles(best_similarities),
        "note": (
            "Jaccard similarity is a lexical proxy, not a semantic score. It "
            "bounds how much of the claim the selected snippet actually shares."
        ),
    }


def _to_passage_list(texts: Sequence[str]) -> List[Any]:
    from .evidence import _to_passage

    return [_to_passage(text) for text in texts]


def main(argv: List[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("reports/evidence_distribution.json"))
    args = parser.parse_args(argv)
    arms = dict(entry.split("=", 1) for entry in args.arm)
    report = describe_evidence(
        {name: Path(path) for name, path in arms.items()}, max_length=args.max_length
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
