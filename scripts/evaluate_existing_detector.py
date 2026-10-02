"""Read-only checkpoint evaluation on genuine, source-separated prepared data.
Writes a new report directory, never checkpoint metrics or calibration files.
"""
import argparse
import hashlib
import json
from pathlib import Path
from halluciguard_detector.training import evaluate, load_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("evaluation report directory already exists")
    splits = {s: load_rows(args.data / f"{s}.jsonl") for s in ("train", "dev", "test")}
    groups = {s: {str(r["source_id"]) for r in rows} for s, rows in splits.items()}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if groups[a] & groups[b]:
            raise ValueError("source group overlap")
    config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    if {str(k): v for k, v in config["id2label"].items()} != {
        "0": "SUPPORTED", "1": "CONTRADICTED", "2": "NOT_ENOUGH_INFO"
    }:
        raise ValueError("checkpoint label mapping differs from prepared data")
    calibration = json.loads((args.checkpoint / "calibration.json").read_text(encoding="utf-8"))
    hashes = {str(f.relative_to(args.checkpoint)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in args.checkpoint.iterdir() if f.is_file()}
    metrics = evaluate(args.data, args.checkpoint, batch_size=args.batch_size,
        max_length=int(calibration["max_length"]), save=False)
    if any(hashlib.sha256((args.checkpoint / f).read_bytes()).hexdigest() != h for f, h in hashes.items()):
        raise RuntimeError("checkpoint artifact changed during evaluation")
    report = {"checkpoint": str(args.checkpoint.resolve()), "data": str(args.data.resolve()),
        "checkpoint_hashes": hashes, "calibration": calibration,
        "split_counts": {s: len(r) for s, r in splits.items()}, "source_group_disjoint": True,
        "test_sha256": hashlib.sha256((args.data / "test.jsonl").read_bytes()).hexdigest(),
        "metrics": metrics, "interpretation": "existing checkpoint measurement; no training or improvement claim"}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"split_counts": report["split_counts"], "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
