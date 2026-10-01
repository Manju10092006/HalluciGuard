"""Ingest genuinely labeled, same-generation traces for the Phase 1 head."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .phase1 import FEATURE_NAMES, FEATURE_SCHEMA, LABELS, GenerationTrace, claim_features
from .text import sentence_spans


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Require a human label for claims in the original generation event."""
    response = record.get("response")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("response must be nonempty text")
    if record.get("label_source") != "human":
        raise ValueError("label_source must be 'human'; inferred labels are not training truth")
    if not all(isinstance(record.get(k), str) and record[k].strip() for k in ("event_id", "group_id", "dataset_id")):
        raise ValueError("event_id, group_id and dataset_id are required")
    trace = GenerationTrace.model_validate(record.get("trace"))
    if not trace.complete or trace.text_sha256 != hashlib.sha256(response.encode()).hexdigest():
        raise ValueError("trace must match this exact generated response")
    claims = record.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("nonempty claims list required")
    label_unit = record.get("label_unit", "atomic_claim")
    if label_unit not in ("sentence", "atomic_claim"):
        raise ValueError("label_unit must be sentence or atomic_claim")
    if label_unit == "sentence":
        expected_spans = [(span.start, span.end) for span in sentence_spans(response)]
        actual_spans = [(int(claim["start"]), int(claim["end"])) for claim in claims]
        if actual_spans != expected_spans:
            raise ValueError("sentence labels must cover every sentence exactly")
    out = []
    for claim in claims:
        label = claim.get("label")
        if label not in LABELS:
            raise ValueError(f"unknown claim label: {label!r}")
        start, end = int(claim["start"]), int(claim["end"])
        if not 0 <= start < end <= len(response):
            raise ValueError("claim offset outside response")
        if claim.get("text") != response[start:end]:
            raise ValueError("claim text and response offsets disagree")
        out.append({
            "event_id": record["event_id"], "group_id": record["group_id"],
            "dataset_id": record["dataset_id"], "claim_id": str(claim.get("claim_id") or len(out) + 1),
            "label": label, "label_id": LABELS[label],
            "features": claim_features(trace, response, start, end),
            "model_id": trace.model_id, "model_revision": trace.model_revision,
            "tokenizer_id": trace.tokenizer_id, "tokenizer_revision": trace.tokenizer_revision,
            "feature_schema": FEATURE_SCHEMA,
            "label_unit": label_unit,
        })
    return out


def annotation_template(input_jsonl: Path, output_jsonl: Path) -> dict[str, Any]:
    """Prepare unlabelled annotation tasks from real, exact generation events.

    Sentence spans are suggestions only: annotators must split compound claims,
    remove non-factual text, and supply independently checked labels.
    """
    if output_jsonl.exists():
        raise FileExistsError("annotation output already exists")
    if not input_jsonl.is_file():
        raise FileNotFoundError(f"generation events missing: {input_jsonl}")
    if input_jsonl.resolve() == output_jsonl.resolve():
        raise ValueError("annotation output must not overwrite generation events")
    tasks: list[dict[str, Any]] = []
    seen_events: set[str] = set()
    for line_no, line in enumerate(input_jsonl.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        response = record.get("response")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"line {line_no}: missing generated response")
        if not all(isinstance(record.get(k), str) and record[k].strip() for k in ("event_id", "group_id", "dataset_id")):
            raise ValueError(f"line {line_no}: event_id, group_id and dataset_id required")
        if record["event_id"] in seen_events:
            raise ValueError(f"line {line_no}: duplicate event_id")
        trace = GenerationTrace.model_validate(record.get("trace"))
        if not trace.complete or trace.text_sha256 != hashlib.sha256(response.encode()).hexdigest():
            raise ValueError(f"line {line_no}: trace does not match response")
        spans = sentence_spans(response)
        tasks.append({
            "event_id": record["event_id"], "group_id": record["group_id"],
            "dataset_id": record["dataset_id"], "response": response,
            "query": record.get("query", ""),
            "trace": trace.model_dump(), "label_source": "UNLABELED",
            "label_unit": "sentence",
            "claims": [{"claim_id": f"c{index}", "text": span.text,
                        "start": span.start, "end": span.end, "label": None}
                       for index, span in enumerate(spans, start=1)],
        })
        seen_events.add(record["event_id"])
    if not tasks:
        raise ValueError("no generation events found")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks), encoding="utf-8")
    return {"events": len(tasks), "status": "unlabelled_requires_human_review", "output": str(output_jsonl)}


def prepare(input_jsonl: Path, output_dir: Path, seed: int = 42) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("prepared output directory is not empty")
    if not input_jsonl.is_file():
        raise FileNotFoundError(f"labeled generation file missing: {input_jsonl}")
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_events: set[str] = set()
    seen_generations: set[str] = set()
    compatibility: tuple[str, str, str, str] | None = None
    annotation_unit: str | None = None
    with input_jsonl.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
                event_id = record.get("event_id")
                if event_id in seen_events:
                    raise ValueError("duplicate event_id")
                prepared = validate_record(record)
                generation_key = _hash_json([record["trace"]["text_sha256"], record["response"]])
                if generation_key in seen_generations:
                    raise ValueError("duplicate generated response across events")
                key = tuple(prepared[0][name] for name in ("model_id", "model_revision", "tokenizer_id", "tokenizer_revision"))
                if compatibility is not None and key != compatibility:
                    raise ValueError("mixed generator/checkpoint versions")
                compatibility = key
                if annotation_unit is not None and prepared[0]["label_unit"] != annotation_unit:
                    raise ValueError("mixed label units in dataset")
                annotation_unit = prepared[0]["label_unit"]
                seen_events.add(event_id)
                seen_generations.add(generation_key)
                rows.extend(prepared)
            except Exception as exc:
                rejected.append({"line": line_no, "reason": type(exc).__name__})
    if not rows:
        raise ValueError(f"no valid labeled events; rejected={len(rejected)}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 8:
        raise ValueError("at least eight independent groups are required for four leakage-safe splits")
    n = len(keys)
    train_end = min(n - 3, max(1, round(0.60 * n)))
    validation_end = min(n - 2, train_end + max(1, round(0.15 * n)))
    calibration_end = min(n - 1, validation_end + max(1, round(0.15 * n)))
    assignments = {group: ("train" if index < train_end else
                           "validation" if index < validation_end else
                           "calibration" if index < calibration_end else "test")
                   for index, group in enumerate(keys)}
    splits = {name: [] for name in ("train", "validation", "calibration", "test")}
    for group, group_rows in groups.items():
        splits[assignments[group]].extend(group_rows)
    if any(not values for values in splits.values()):
        raise ValueError("all four splits must contain labeled claims")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as target:
            for value in values:
                target.write(json.dumps(value, sort_keys=True) + "\n")
    manifest = {
        "schema": "hg-phase1-dataset-v1", "source": str(input_jsonl),
        "source_sha256": hashlib.sha256(input_jsonl.read_bytes()).hexdigest(),
        "seed": seed, "feature_schema": FEATURE_SCHEMA, "feature_names": list(FEATURE_NAMES),
        "label_unit": annotation_unit,
        "generator": dict(zip(("model_id", "model_revision", "tokenizer_id", "tokenizer_revision"), compatibility)),
        "total_events": len(seen_events), "total_claims": len(rows), "rejected_count": len(rejected),
        "rejected": rejected,
        "splits": {name: {"claims": len(values), "groups": len({v["group_id"] for v in values}),
                          "labels": dict(Counter(v["label"] for v in values))} for name, values in splits.items()},
        "group_assignment_sha256": _hash_json(assignments),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
