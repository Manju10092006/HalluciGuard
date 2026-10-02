"""Reproducible RAGTruth fine-tuning pilot; never promotes production weights.

Uses genuine prepared rows without label changes. Training is class-balanced by
seeded subsampling; validation/calibration preserve their sampled prevalence and
are source-separated. Test rows are unchanged and never used for selection.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from halluciguard_detector.training import train, evaluate, load_rows


def prepare(source: Path, output: Path, per_class: int, limit: int, seed: int) -> dict:
    if output.exists():
        raise FileExistsError("pilot output already exists")
    if per_class < 1 or limit < 3:
        raise ValueError("pilot sample limits must be positive")
    rows = {name: load_rows(source / f"{name}.jsonl") for name in ("train", "dev", "test")}
    groups = {name: {str(r["source_id"]) for r in value} for name, value in rows.items()}
    if any(groups[a] & groups[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("source data groups overlap")
    rng = random.Random(seed)
    train_rows = []
    for label in (0, 1, 2):
        candidates = [r for r in rows["train"] if r["label_id"] == label]
        if len(candidates) < per_class:
            raise ValueError("insufficient genuine training examples")
        train_rows.extend(rng.sample(candidates, per_class))
    rng.shuffle(train_rows)
    dev_groups = sorted(groups["dev"])
    rng.shuffle(dev_groups)
    if len(dev_groups) < 2:
        raise ValueError("independent calibration requires at least two dev source groups")
    validation_groups = set(dev_groups[:len(dev_groups) // 2])
    validation = [r for r in rows["dev"] if str(r["source_id"]) in validation_groups]
    calibration = [r for r in rows["dev"] if str(r["source_id"]) not in validation_groups]
    for sample in (validation, calibration):
        rng.shuffle(sample)
        del sample[limit:]
        if {r["label_id"] for r in sample} != {0, 1, 2}:
            raise ValueError("pilot validation/calibration lacks a class")
    output.mkdir(parents=True)
    data = output / "data"
    data.mkdir()
    final = {"train": train_rows, "dev": validation, "calibration": calibration, "test": rows["test"]}
    summary = {"seed": seed, "source_directory": str(source), "splits": {}}
    for name, sample in final.items():
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample)
        (data / f"{name}.jsonl").write_text(payload, encoding="utf-8")
        summary["splits"][name] = {
            "count": len(sample), "labels": dict(Counter(r["label"] for r in sample)),
            "groups": len({str(r["source_id"]) for r in sample}),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }
    (output / "preparation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="microsoft/deberta-v3-xsmall")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--per-class", type=int, default=2000)
    parser.add_argument("--validation-limit", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    start = time.perf_counter()
    print("PILOT_PREPARATION", json.dumps(prepare(
        args.data, args.output, args.per_class, args.validation_limit, args.seed)), flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trained = train(args.output / "data", args.output / "checkpoint",
        base_model=args.base_model, epochs=args.epochs, batch_size=8,
        seed=args.seed, calibration_data=args.output / "data/calibration.jsonl",
        selection_metric="macro_f1", local_files_only=True, gradient_checkpointing=True)
    print("PILOT_TRAINED", json.dumps(trained), flush=True)
    pilot = evaluate(args.output / "data", args.output / "checkpoint")
    # Read-only original evaluation: never overwrite baseline metrics/predictions.
    baseline = evaluate(args.output / "data", args.baseline, save=False)
    comparison = {"training_seconds_and_evaluation": time.perf_counter() - start,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "initialization": args.base_model, "pilot": pilot, "baseline": baseline,
        "production_promoted": False,
        "limitations": ["class-balanced subsample, not full-corpus training",
            "prepared span labels are not guaranteed atomic claims",
            "legacy joined evidence is not production top-one evidence",
            "no low-risk bypass release evaluation"]}
    (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print("PILOT_COMPARISON", json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main()
