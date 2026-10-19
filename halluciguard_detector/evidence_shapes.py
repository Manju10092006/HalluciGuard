"""Evidence-shape arms used to isolate the train/runtime mismatch.

The Detector shows DeBERTa exactly one evidence string per claim. What that
string *is* used to be decided in two places that disagreed:

* ``data.iter_ragtruth_examples`` built training evidence by taking the top six
  lexical sentences for the claim and **joining** them into one string.
* ``detector.Detector.detect`` asks ``evidence.select_evidence`` for ranked
  snippets, keeps the full list on the response for display, and classifies on
  ``snippets[0]`` only.

Measured on the released RAGTruth split, the joined form averages 277 DeBERTa
tokens against an 18-token claim, so **41% of training examples overflow the
256-token budget** and have the tail of their evidence cut away by
``longest_first``. Production feeds one short reranked sentence that essentially
never truncates. The model is trained mostly on truncated multi-snippet blobs
and served on single snippets.

This module builds the arms from one shared candidate pool so that *only the
selection policy differs*:

``lexical_joined``
    Top six lexical sentences, space-joined. The released baseline, kept
    reproducible so Experiment A is the same thing that was shipped.
``lexical_top1``
    Top one lexical sentence. Same selector, joined string removed -- this
    isolates "how much of the mismatch is the join" from "how much is the
    selector".
``production_top1``
    Top one sentence from the real production path: the same
    ``select_evidence`` call ``detect`` makes, at the same pool size and rerank
    depth.

Candidate sentences are produced identically for every arm, so sentence
segmentation is held constant and cannot confound the result.

Scope, stated plainly: the candidate pool is the sentence set of the claim's
*own* RAGTruth source document. Real production retrieves over a multi-document
corpus, so the pool here is easier than production's. That makes this a
controlled test of evidence *shape*, not a simulation of retrieval difficulty.
:func:`build_pool` can add foreign-document distractors when retrieval
difficulty is the thing under test.

Every arm returns an :class:`EvidenceRecord`, not a bare string, so a report can
state how many snippets went in, how long the result was, which selection route
ran, and whether it was degraded -- instead of assuming all three are equal.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from .nli_input import MAX_EVIDENCE_SNIPPETS
from .text import lexical_evidence, sentence_spans

#: How many lexical sentences the released baseline joined per claim. Changing
#: this changes the baseline, so it is a named constant rather than a literal.
LEXICAL_JOIN_LIMIT = 6

#: Foreign-document sentences mixed into the pool by :func:`build_pool` when a
#: harder retrieval problem is wanted. Zero by default: the arms must differ
#: only in selection policy.
DISTRACTOR_COUNT = 0


@dataclass
class EvidenceRecord:
    """One arm's evidence for one claim, plus what it cost to produce."""

    evidence: str
    #: How many source snippets were combined into ``evidence``. 1 is the
    #: production contract; >1 is the joined-string baseline.
    snippet_count: int = MAX_EVIDENCE_SNIPPETS
    #: Which selection route produced it (see ``evidence.select_evidence``).
    route: str = "unknown"
    #: True when the shared retrieval stack was unavailable or failed closed.
    degraded: bool = False
    reason: str = ""
    #: Length of the candidate pool the selector was allowed to choose from.
    pool_size: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def candidate_passages(source_info: str) -> List[str]:
    """Return the sentence-level candidate pool for one source document.

    Production hands ``select_evidence`` a list of already-segmented passages
    and never segments internally, so the pool granularity is the caller's
    choice. Building it here from sentence spans mirrors a retrieval layer that
    returns passages, and keeps all arms on identical candidates.
    """
    return [span.text for span in sentence_spans(source_info) if span.text.strip()]


def build_pool(source_info: str, distractors: Sequence[str] = ()) -> List[str]:
    """Candidate pool for one claim: own-document sentences plus distractors.

    The own-document sentences come first so that, with no distractors, every
    arm sees byte-identical candidates. Distractors model the harder production
    case where the claim's source competes with the rest of the corpus.
    """
    pool = candidate_passages(source_info)
    for text in distractors:
        if text.strip() and text not in pool:
            pool.append(text)
    return pool


def lexical_joined(claim: str, candidates: Sequence[str]) -> EvidenceRecord:
    """Baseline shape: top-N lexical sentences joined into one string."""
    selected = lexical_evidence(claim, list(candidates), limit=LEXICAL_JOIN_LIMIT)
    return EvidenceRecord(
        evidence=" ".join(selected),
        snippet_count=len(selected),
        route="lexical_joined",
        degraded=False,
        pool_size=len(candidates),
    )


def lexical_top1(claim: str, candidates: Sequence[str]) -> EvidenceRecord:
    """Single best lexical sentence, no reranking."""
    selected = lexical_evidence(claim, list(candidates), limit=MAX_EVIDENCE_SNIPPETS)
    return EvidenceRecord(
        evidence=selected[0] if selected else "",
        snippet_count=len(selected),
        route="lexical_top1",
        degraded=False,
        pool_size=len(candidates),
    )


def production_top1(
    claim: str,
    candidates: Sequence[str],
    *,
    pool_k: int | None = None,
    dense_model: str | None = None,
) -> EvidenceRecord:
    """Single best snippet from the real production selection path.

    Calls ``select_evidence`` with the detector's own pool width, so this is
    the same code path ``detect`` runs, not a re-implementation of it. When the
    shared retrieval stack is unavailable the call fails soft to lexical
    selection; that is recorded as ``degraded`` rather than being presented as a
    production-shaped pick, because a degraded pick measured as if it were
    production evidence would overstate the alignment.

    Note that ``select_evidence`` returns a *ranked list* while ``detect``
    scores only its first element, so this arm reports ``snippet_count=1`` and
    keeps the rest under ``extra["unused_ranked"]``. Counting the ranked list
    would have made every production row look like a multi-snippet regression.
    """
    # Imported lazily: the production path pulls in the shared Verifier stack
    # (cross-encoder, and an embedding model when enabled), which must never be
    # loaded as a side effect of importing this module.
    from .evidence import DEFAULT_POOL_K, select_evidence

    trace: Dict[str, Any] = {}
    snippets = select_evidence(
        claim,
        list(candidates),
        pool_k=pool_k if pool_k is not None else DEFAULT_POOL_K,
        dense_model=dense_model,
        trace=trace,
    )
    usable = [s for s in snippets if s and s.strip()]
    # ``snippet_count`` is the contract marker: 1 means the evidence side is a
    # single snippet, which is what the detector actually sends. It must count
    # what reached the model, not what the selector ranked. Recording
    # ``len(usable)`` here reported 3 on every row of a real run, because
    # ``select_evidence`` returns the top-3 reranked list while
    # ``Detector.detect`` classifies ``snippets[0]`` -- so every production row
    # looked out of contract when it was in contract.
    extra: Dict[str, Any] = {"ranked_snippets": len(usable)}
    if len(usable) > 1:
        # Kept for inspection only; never concatenated onto the evidence side.
        extra["unused_ranked"] = usable[1:]
    return EvidenceRecord(
        evidence=usable[0] if usable else "",
        snippet_count=1 if usable else 0,
        route=str(trace.get("route", "unknown")),
        degraded=bool(trace.get("degraded")),
        reason=str(trace.get("reason", "")),
        pool_size=len(candidates),
        extra=extra,
    )


#: Arm name -> selector. The arms are compared on two axes that are genuinely
#: conflated in the released baseline: how many snippets are joined, and which
#: selector chose them.
SHAPES: Dict[str, Callable[[str, Sequence[str]], EvidenceRecord]] = {
    "lexical_joined": lexical_joined,
    "lexical_top1": lexical_top1,
    "production_top1": production_top1,
}

#: The arm the detector is actually served.
PRODUCTION_SHAPE = "production_top1"


def shape_evidence(claim: str, source_info: str, shape: str) -> str:
    """Return just the evidence string for an arm (convenience wrapper)."""
    return shape_record(claim, source_info, shape).evidence


def shape_record(
    claim: str,
    source_info: str,
    shape: str,
    *,
    distractors: Sequence[str] = (),
    pool_k: int | None = None,
) -> EvidenceRecord:
    """Build one arm's :class:`EvidenceRecord` for a claim."""
    if shape not in SHAPES:
        raise ValueError(f"unknown evidence shape {shape!r}; expected one of {sorted(SHAPES)}")
    if shape == PRODUCTION_SHAPE and pool_k is not None:
        return production_top1(claim, build_pool(source_info, distractors), pool_k=pool_k)
    return SHAPES[shape](claim, build_pool(source_info, distractors))


def load_sources(dataset_dir: Path) -> Dict[str, str]:
    """Map ``source_id`` to its ``source_info`` text.

    Most RAGTruth sources carry a plain string, but a large majority (2,022 of
    2,965) are structured records (name/address/hours) for the Data2txt and QA
    tasks. ``iter_ragtruth_examples`` serialises those with sorted keys; the
    same coercion is applied here so all arms are built from byte-identical
    source text. Reusing the baseline's own coercion keeps the arms comparable
    instead of introducing a second, subtly different serialisation.
    """
    sources: Dict[str, str] = {}
    with (dataset_dir / "source_info.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source_info = row.get("source_info")
            if not isinstance(source_info, str):
                source_info = json.dumps(source_info, ensure_ascii=False, sort_keys=True)
            sources[str(row["source_id"])] = source_info
    return sources


def load_records(path: Path) -> List[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def stratified_subsample(
    records: Sequence[dict],
    size: int,
    seed: int,
) -> List[dict]:
    """Take a class-proportional, deterministic subsample.

    The per-class ratio of the input is preserved, so a reduced run keeps the
    natural evaluation distribution instead of silently reweighting the
    minority classes -- which would move contradiction precision and recall
    for reasons that have nothing to do with the evidence pipeline. Selection
    is seeded and therefore reproducible, and the relative order within a class
    is the input order, so the choice is stable across reruns.
    """
    records = list(records)
    if size <= 0 or size >= len(records):
        return records
    rng = random.Random(seed)
    buckets: Dict[str, List[dict]] = {}
    for row in records:
        buckets.setdefault(row["label"], []).append(row)
    total = len(records)
    taken: List[dict] = []
    for label in sorted(buckets):
        pool = list(buckets[label])
        rng.shuffle(pool)
        share = max(1, round(size * len(pool) / total))
        taken.extend(pool[:share])
    rng.shuffle(taken)
    return taken


def class_capped_subsample(
    records: Sequence[dict],
    per_class: int,
    seed: int,
) -> List[dict]:
    """Take up to ``per_class`` examples of each class, deterministically.

    The natural test split is 91% SUPPORTED, so a size-limited stratified draw
    still yields only a few dozen CONTRADICTED rows -- far too few for a
    contradiction F1 to mean anything. Capping *per class* rather than per
    dataset fixes that while keeping every arm on the identical row set.

    This is a **diagnostic** split, not a natural-distribution metric: the
    class mix is deliberately reweighted, so absolute precision/recall here do
    not transfer. It is valid for comparing arms, because all arms see the same
    rows in the same proportion, and the shipped baseline's natural-
    distribution numbers are reported separately.
    """
    records = list(records)
    rng = random.Random(seed)
    buckets: Dict[str, List[dict]] = {}
    for row in records:
        buckets.setdefault(row["label"], []).append(row)
    taken: List[dict] = []
    for label in sorted(buckets):
        pool = list(buckets[label])
        rng.shuffle(pool)
        taken.extend(pool[:per_class])
    rng.shuffle(taken)
    return taken


def _subsample(
    records: Sequence[dict],
    size: int,
    seed: int,
    per_class: int | None,
) -> List[dict]:
    return (
        class_capped_subsample(records, per_class, seed)
        if per_class
        else stratified_subsample(records, size, seed)
    )


def build_split(
    data_dir: Path,
    out_dir: Path,
    dataset_dir: Path,
    shape: str,
    seed: int,
    sizes: Dict[str, int],
    *,
    pool_k: int | None = None,
    per_class: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    """Write one arm's splits with the requested evidence shape.

    ``sizes`` maps a split name to a subsample size (``0``/absent keeps the
    full split). ``per_class`` optionally caps a split *per label* instead --
    used for the ``metric`` split, where the natural class mix would leave too
    few CONTRADICTED rows to measure.
    """
    sources = load_sources(dataset_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class = per_class or {}
    summary: Dict[str, Any] = {"shape": shape, "seed": seed, "splits": {}}
    for name, size in sizes.items():
        # The diagnostic split is a re-balanced view of the *test* rows, not a
        # new partition: same claims, different class mix. Reading it from
        # test.jsonl guarantees that, instead of relying on a second file that
        # could drift out of sync.
        source_name = "test" if name == "metric" else name
        records = load_records(Path(data_dir) / f"{source_name}.jsonl")
        subset = _subsample(records, size, seed, per_class.get(name))
        empty = 0
        degraded = 0
        routes: Dict[str, int] = {}
        lengths: List[int] = []
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in subset:
                source_info = sources.get(str(row["source_id"]), "")
                record = shape_record(row["claim"], source_info, shape, pool_k=pool_k)
                if not record.evidence:
                    empty += 1
                if record.degraded:
                    degraded += 1
                routes[record.route] = routes.get(record.route, 0) + 1
                lengths.append(len(record.evidence))
                row = dict(row)
                row["evidence"] = record.evidence
                row["evidence_shape"] = shape
                row["evidence_route"] = record.route
                row["evidence_degraded"] = record.degraded
                row["evidence_snippet_count"] = record.snippet_count
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts: Dict[str, int] = {}
        for row in subset:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
        summary["splits"][name] = {
            "total": len(subset),
            "labels": counts,
            "empty_evidence": empty,
            "degraded_evidence": degraded,
            "routes": dict(sorted(routes.items())),
            "mean_evidence_chars": round(sum(lengths) / max(1, len(lengths)), 2),
            "per_class_cap": per_class.get(name),
        }
    (out_dir / "shape_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("halluciguard_detector/data"))
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("halluciguard_detector/third_party/RAGTruth/dataset"),
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--shapes", nargs="+", default=sorted(SHAPES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-k", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=0, help="0 = keep the full split")
    parser.add_argument("--dev-size", type=int, default=0)
    parser.add_argument("--test-size", type=int, default=0)
    parser.add_argument(
        "--metric-per-class",
        type=int,
        default=0,
        help="per-class cap for the diagnostic 'metric' split (0 = skip)",
    )
    args = parser.parse_args(argv)

    sizes: Dict[str, int] = {
        "train": args.train_size,
        "dev": args.dev_size,
        "test": args.test_size,
    }
    per_class: Dict[str, int] = {}
    if args.metric_per_class:
        sizes["metric"] = 0
        per_class["metric"] = args.metric_per_class
    for shape in args.shapes:
        summary = build_split(
            args.data_dir,
            args.out_root / shape,
            args.dataset_dir,
            shape,
            args.seed,
            sizes,
            pool_k=args.pool_k,
            per_class=per_class,
        )
        print(shape, json.dumps(summary["splits"]), flush=True)
    return 0

if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
