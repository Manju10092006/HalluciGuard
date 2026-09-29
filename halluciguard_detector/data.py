"""RAGTruth -> three-class claim examples.

The human annotation is **span level**, but a naive conversion collapses every
sentence that touches an annotated span into that span's class. For

    "Java was created by James Gosling in 1995 and is widely used."

a single annotated span does not make the whole sentence false, and training on
that sentence teaches the model exactly the conflation this detector is meant
to avoid. Three things are done here instead:

1. **Coverage is measured, not assumed.** A sentence is labelled from its
   annotations only when the annotated problem region actually covers it.
2. **Mixed sentences are split, not guessed.** When a sentence contains both
   sufficient and problem spans, each maximal annotated region is emitted as
   its own example using that region's own human label. No label is invented.
3. **Unrecognised label vocabulary is reported.** An unknown ``label_type`` is
   treated conservatively as insufficient evidence (matching the previous
   behaviour) *and* counted, so a dataset revision with new spellings shows up
   in the stats instead of silently skewing the label mix.

Known limitation: RAGTruth annotates spans, not the atomic claims the runtime
decomposer produces, and this conversion cannot recover atomisation. Span-level
examples are therefore a closer match than sentence-level ones but still not
identical to runtime claims; the split counts in ``stats.json`` quantify how
much of the data is sentence-level versus span-level.

**Evidence shape** is the other half of the train/runtime contract. The claim
labels above are unchanged by it, but the *evidence* a label is attached to is
not: the detector is served one reranked snippet and used to be trained on up
to six lexical snippets joined into one string. The shape is therefore an
explicit parameter (``evidence_shape``, defaulting to the production arm) so
the released ``lexical_joined`` baseline stays reproducible for comparison
instead of being quietly overwritten. Every emitted row records which route
selected its evidence and whether that selection was degraded, so a training
set built on a fail-soft fallback can never be mistaken for one built on real
production selection.
"""

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .evidence_shapes import PRODUCTION_SHAPE, shape_record
from .text import sentence_spans


LABEL_TO_ID = {"SUPPORTED": 0, "CONTRADICTED": 1, "NOT_ENOUGH_INFO": 2}

#: ``label_type`` -> markers. Matching is *substring*-based, deliberately.
#:
#: The shipped RAGTruth vocabulary is decorated, not bare: the four real
#: ``label_type`` values are ``"Evident Conflict"``, ``"Subtle Conflict"``,
#: ``"Evident Baseless Info"`` and ``"Subtle Baseless Info"``. An exact-match
#: table silently fails on all four and dumps every example into the unknown
#: fallback, which would flip the whole training set. Marker sets keep the
#: prefixes free while staying explicit and reviewable.
LABEL_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "CONTRADICTED": ("conflict", "contradict"),
    "NON_FACTUAL": ("non_factual", "nonfactual", "opinion"),
    "SUPPORTED": ("evidence_sufficient", "non_hallucinated", "supported"),
    "NOT_ENOUGH_INFO": (
        "baseless",
        "hallucinat",
        "evidence_insufficient",
        "insufficient_evidence",
    ),
}

#: (marker, concept) pairs, longest marker first. Longest-match-wins keeps the
#: result independent of dict ordering and resolves the one genuine overlap in
#: the vocabulary: ``"non_hallucinated"`` contains ``"hallucinat"``, so the
#: longer SUPPORTED marker has to win or every non-hallucinated span would be
#: labelled NOT_ENOUGH_INFO.
_MARKER_PAIRS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        ((marker, concept) for concept, markers in LABEL_TYPE_ALIASES.items() for marker in markers),
        key=lambda pair: (-len(pair[0]), pair[0]),
    )
)

#: A sentence is only labelled from its problem annotations when they cover at
#: least this fraction of it. Below that the sentence is mixed and gets split
#: into per-span examples instead of being forced into one class.
FULL_COVERAGE = 0.8


def _overlaps(start: int, end: int, annotation: dict) -> bool:
    return start < int(annotation["end"]) and end > int(annotation["start"])


def _overlap_length(start: int, end: int, annotation: dict) -> int:
    return max(0, min(end, int(annotation["end"])) - max(start, int(annotation["start"])))


def _normalise_label_type(raw: str) -> str:
    """Lowercase and fold separators so ``Non Factual`` == ``non_factual``.

    The dataset is not consistent about underscores versus spaces, so matching
    on a normalised form is what keeps ``"Non Factual"`` from falling into the
    unknown bucket.
    """
    return re.sub(r"[\s\-]+", "_", str(raw).strip().lower())


def _match_concept(raw: str) -> str | None:
    """Longest-marker substring match of a raw ``label_type`` onto a concept."""
    value = _normalise_label_type(raw)
    if not value:
        return None
    for marker, concept in _MARKER_PAIRS:
        if marker in value:
            return concept
    return None


def annotation_concept(annotation: dict) -> str:
    """Map one RAGTruth annotation onto a concept name.

    Unknown vocabulary falls back to ``NOT_ENOUGH_INFO``: the conservative
    reading, and the one this converter used before the alias table existed.
    Callers should surface the unknown value via :func:`unknown_label_types`
    rather than trust the fallback.
    """
    return _match_concept(str(annotation.get("label_type", ""))) or "NOT_ENOUGH_INFO"


def unknown_label_types(annotations: Iterable[dict]) -> set[str]:
    return {
        _normalise_label_type(a.get("label_type", ""))
        for a in annotations
        if _match_concept(str(a.get("label_type", ""))) is None
    }


def _relevant(annotations: list[dict], start: int, end: int) -> list[dict]:
    return [
        a
        for a in annotations
        if _overlaps(start, end, a) and not a.get("implicit_true")
    ]


def _coverage(start: int, end: int, annotations: list[dict], concepts: set[str]) -> float:
    """Fraction of ``[start, end)`` covered by annotations of the given concepts.

    Measured as a *union* of intervals. Summing per-annotation overlap lengths
    double-counts nested or duplicated annotations, which could push coverage to
    1.0 for a sentence that is only half annotated and get it force-labelled as
    if a human had marked the whole thing.
    """
    span = max(1, end - start)
    intervals: list[tuple[int, int]] = []
    for annotation in annotations:
        if annotation_concept(annotation) not in concepts:
            continue
        lo = max(start, int(annotation["start"]))
        hi = min(end, int(annotation["end"]))
        if hi > lo:
            intervals.append((lo, hi))
    if not intervals:
        return 0.0
    intervals.sort()
    covered = 0
    current_lo, current_hi = intervals[0]
    for lo, hi in intervals[1:]:
        if lo > current_hi:
            covered += current_hi - current_lo
            current_lo, current_hi = lo, hi
        else:
            current_hi = max(current_hi, hi)
    covered += current_hi - current_lo
    return min(1.0, covered / span)


def _non_factual_only(start: int, end: int, annotations: list[dict]) -> bool:
    """True when the sentence is entirely opinion/non-factual content.

    Such sentences are dropped rather than labelled: an opinion is not evidence
    of falsehood, and forcing it into a factual class would inject exactly the
    noise this converter is meant to remove.
    """
    relevant = _relevant(annotations, start, end)
    if not relevant:
        return False
    return _coverage(start, end, relevant, {"NON_FACTUAL"}) >= FULL_COVERAGE


def _uniform_sentence_label(start: int, end: int, annotations: list[dict]) -> str | None:
    """Label a sentence only when its annotations agree on one class.

    Returns ``None`` when the sentence is mixed, so the caller can split it
    rather than guess. ``NON_FACTUAL`` content is ignored entirely: an opinion
    span is not evidence of falsehood.
    """
    relevant = _relevant(annotations, start, end)
    if not relevant:
        return "SUPPORTED"
    for concept in ("CONTRADICTED", "NOT_ENOUGH_INFO", "SUPPORTED"):
        if _coverage(start, end, relevant, {concept}) >= FULL_COVERAGE:
            return concept
    return None


_REGION_EDGE = ".,;:!?()[]\"'"


def _snap_region(text: str, lo: int, hi: int) -> tuple[int, int]:
    """Grow a region to whitespace/punctuation boundaries.

    Annotation offsets follow the dataset's own tokenisation, which does not
    always land on a word edge. Emitting ``"ames Gosling in"`` as a training
    claim teaches the model mangled text, so each region is grown outwards to
    the nearest natural boundary instead.
    """
    lo = max(0, min(lo, len(text)))
    hi = max(0, min(hi, len(text)))
    while lo > 0 and not text[lo - 1].isspace() and text[lo - 1] not in _REGION_EDGE:
        lo -= 1
    while hi < len(text) and not text[hi].isspace() and text[hi] not in _REGION_EDGE:
        hi += 1
    while lo < hi and text[lo].isspace():
        lo += 1
    while hi > lo and text[hi - 1].isspace():
        hi -= 1
    return lo, hi


def _regions(start: int, end: int, annotations: list[dict], text: str) -> list[tuple[str, str]]:
    """Split a mixed sentence into ``(region_text, label)`` pairs.

    Only the annotated regions are emitted, and each carries the human label of
    the annotations covering it. Regions dominated by non-factual content are
    skipped rather than labelled.
    """
    relevant = sorted(
        (a for a in _relevant(annotations, start, end)),
        key=lambda a: int(a["start"]),
    )
    if not relevant:
        return []
    merged: list[list] = []
    for annotation in relevant:
        span = (int(annotation["start"]), int(annotation["end"]), annotation_concept(annotation))
        if merged and span[0] <= merged[-1][1]:
            previous = merged[-1]
            previous[1] = max(previous[1], span[1])
            # A region that mixes problem and sufficient text is not cleanly
            # classifiable; keep the stronger, non-SUPPORTED reading.
            if previous[2] != span[2] and "SUPPORTED" in (previous[2], span[2]):
                previous[2] = "NOT_ENOUGH_INFO" if "CONTRADICTED" not in (previous[2], span[2]) else "CONTRADICTED"
        else:
            merged.append(list(span))
    out: list[tuple[str, str]] = []
    for region_start, region_end, concept in merged:
        if concept == "NON_FACTUAL":
            continue
        lo, hi = _snap_region(text, max(region_start, start), min(region_end, end))
        if hi <= lo:
            continue
        region_text = text[lo:hi].strip()
        if region_text:
            out.append((region_text, concept))
    return out



def _load_sources(dataset_dir: Path) -> dict[str, dict[str, str]]:
    """Load ``source_id -> {"source_info", "task_type"}`` once, coercing text.

    ``source_info`` is a plain string for prose sources and a JSON object for
    the Data2txt/QA records. The object form is serialised with sorted keys,
    exactly as the original per-row conversion did, so the bytes fed to the
    evidence selector are unchanged.
    """
    sources: dict[str, dict[str, str]] = {}
    with (dataset_dir / "source_info.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source_info = row.get("source_info")
            if not isinstance(source_info, str):
                source_info = json.dumps(source_info, ensure_ascii=False, sort_keys=True)
            sources[str(row["source_id"])] = {
                "source_info": source_info,
                "task_type": str(row.get("task_type") or ""),
            }
    return sources


def _evidence_for(
    claim: str,
    sources: dict[str, dict[str, str]],
    source_id: str,
    evidence_shape: str,
    pool_k: int | None,
    stats: Counter | None,
) -> Any:
    """Select this arm's evidence for a claim and tally what it cost.

    The ``source_id -> source`` map is threaded through instead of being
    reloaded per row: ``prepare_ragtruth`` iterates ~125k examples, and the
    production arm calls into hybrid retrieval per claim, so re-reading a
    15 MB JSONL for every sentence would dominate preparation time.
    """
    source = sources.get(str(source_id)) or {}
    record = shape_record(
        claim,
        source.get("source_info", ""),
        evidence_shape,
        pool_k=pool_k,
    )
    if stats is not None:
        stats[f"evidence_route:{record.route}"] += 1
        if record.degraded:
            stats["evidence_degraded_examples"] += 1
        if not record.evidence:
            stats["evidence_empty_examples"] += 1
    return record


def iter_ragtruth_examples(
    dataset_dir: Path,
    split: str,
    stats: Counter | None = None,
    *,
    evidence_shape: str = PRODUCTION_SHAPE,
    sources: dict[str, dict[str, str]] | None = None,
    pool_k: int | None = None,
) -> Iterable[dict]:
    """Yield claim examples for one split.

    A uniformly annotated sentence yields one sentence-level example. A mixed
    sentence yields one example per annotated region, which is the closest
    claim-level labelling the span annotations support. ``stats`` (if given)
    accumulates the counters reported in ``stats.json``.

    ``evidence_shape`` selects which evidence policy builds the evidence string;
    it does not touch label derivation, which stays exactly as PR #53 defined
    it. Pass ``sources`` to reuse an already-loaded source map across splits.
    """
    sources = _load_sources(dataset_dir) if sources is None else sources
    with (dataset_dir / "response.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != split or row["quality"] != "good":
                continue
            task_type = (sources.get(str(row["source_id"])) or {}).get("task_type", "")
            annotations = row.get("labels") or []
            for vocabulary in unknown_label_types(annotations):
                if stats is not None:
                    stats[f"unknown_label_type:{vocabulary or '<empty>'}"] += 1
            for span in sentence_spans(row["response"]):
                record = _evidence_for(
                    span.text, sources, row["source_id"], evidence_shape, pool_k, stats
                )
                label = _uniform_sentence_label(span.start, span.end, annotations)
                if label is not None:
                    if stats is not None:
                        stats["sentence_level_examples"] += 1
                    yield {
                        "id": f"{row['id']}:{span.start}-{span.end}",
                        "source_id": str(row["source_id"]),
                        "task_type": task_type,
                        "evidence": record.evidence,
                        "evidence_shape": evidence_shape,
                        "evidence_route": record.route,
                        "evidence_degraded": record.degraded,
                        "evidence_snippet_count": record.snippet_count,
                        "claim": span.text,
                        "label": label,
                        "label_id": LABEL_TO_ID[label],
                        "granularity": "sentence",
                    }
                    continue
                # Mixed: split into the annotated regions instead of guessing.
                if _non_factual_only(span.start, span.end, annotations):
                    if stats is not None:
                        stats["non_factual_sentences_dropped"] += 1
                    continue
                regions = _regions(span.start, span.end, annotations, row["response"])
                if stats is not None:
                    stats["mixed_sentences_split"] += 1
                if not regions:
                    if stats is not None:
                        stats["mixed_sentences_dropped"] += 1
                    continue
                for index, (region_text, region_label) in enumerate(regions):
                    if stats is not None:
                        stats["span_level_examples"] += 1
                    region = _evidence_for(
                        region_text, sources, row["source_id"], evidence_shape, pool_k, stats
                    )
                    yield {
                        "id": f"{row['id']}:{span.start}-{span.end}#{index}",
                        "source_id": str(row["source_id"]),
                        "task_type": task_type,
                        "evidence": region.evidence,
                        "evidence_shape": evidence_shape,
                        "evidence_route": region.route,
                        "evidence_degraded": region.degraded,
                        "evidence_snippet_count": region.snippet_count,
                        "claim": region_text,
                        "label": region_label,
                        "label_id": LABEL_TO_ID[region_label],
                        "granularity": "span",
                    }




def prepare_ragtruth(
    dataset_dir: Path,
    output_dir: Path,
    seed: int = 42,
    dev_fraction: float = 0.1,
    max_supported_ratio: float = 2.0,
    *,
    evidence_shape: str = PRODUCTION_SHAPE,
    pool_k: int | None = None,
) -> dict:
    """Create leakage-safe splits grouped by source document.

    ``evidence_shape`` defaults to the production arm so the generated
    training pairs match what the detector is served. ``"lexical_joined"``
    reproduces the pre-alignment baseline byte-for-byte, which is what makes
    the Experiment A/B comparison a controlled one rather than a comparison of
    two codebases that drifted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    counters: Counter = Counter()
    sources = _load_sources(dataset_dir)
    train_all = list(
        iter_ragtruth_examples(
            dataset_dir, "train", counters,
            evidence_shape=evidence_shape, sources=sources, pool_k=pool_k,
        )
    )
    test = list(
        iter_ragtruth_examples(
            dataset_dir, "test", counters,
            evidence_shape=evidence_shape, sources=sources, pool_k=pool_k,
        )
    )
    source_ids = sorted({x["source_id"] for x in train_all})
    rng = random.Random(seed)
    rng.shuffle(source_ids)
    dev_ids = set(source_ids[: max(1, round(len(source_ids) * dev_fraction))])
    raw_splits = {
        "train": [x for x in train_all if x["source_id"] not in dev_ids],
        "dev": [x for x in train_all if x["source_id"] in dev_ids],
        "test": test,
    }
    # Downsample only the training majority class. Evaluation remains natural-distribution.
    train = raw_splits["train"]
    positives = [x for x in train if x["label"] != "SUPPORTED"]
    supported = [x for x in train if x["label"] == "SUPPORTED"]
    rng.shuffle(supported)
    supported = supported[: int(max_supported_ratio * max(1, len(positives)))]
    raw_splits["train"] = positives + supported
    rng.shuffle(raw_splits["train"])
    stats = {}
    for name, records in raw_splits.items():
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats[name] = {
            "total": len(records),
            "labels": dict(Counter(x["label"] for x in records)),
            "granularity": dict(Counter(x.get("granularity", "sentence") for x in records)),
            "evidence_route": dict(
                Counter(x.get("evidence_route", "unknown") for x in records)
            ),
            "degraded_evidence": sum(1 for x in records if x.get("evidence_degraded")),
            "empty_evidence": sum(1 for x in records if not x.get("evidence")),
        }
    stats["evidence_shape"] = evidence_shape
    stats["label_noise_controls"] = {
        "full_coverage_threshold": FULL_COVERAGE,
        "known_label_markers": sorted(marker for marker, _ in _MARKER_PAIRS),
        "counters": dict(sorted(counters.items())),
        "note": (
            "Sentence-level examples are emitted only when annotated problem spans "
            "cover the whole sentence. Mixed sentences are split into per-span "
            "examples using the human label of each annotated region. "
            "'unknown_label_type:*' counters mean the dataset used a label_type this "
            "converter does not recognise; those were treated as NOT_ENOUGH_INFO."
        ),
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
