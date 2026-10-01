"""Reproducible Phase 1 data, training, and held-out evaluation commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="collect genuine unlabelled local generation traces")
    collect.add_argument("--input", type=Path, required=True, help="JSONL with prompt_id and query")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--model-id", required=True)
    collect.add_argument("--model-revision", required=True)
    collect.add_argument("--tokenizer-id", required=True)
    collect.add_argument("--tokenizer-revision", required=True)
    collect.add_argument("--dataset-id", required=True)
    collect.add_argument("--max-new-tokens", type=int, default=128)
    collect.add_argument("--seed", type=int, default=42)
    annotate = sub.add_parser("annotation-template", help="make unlabelled tasks from exact generation events")
    annotate.add_argument("--input", type=Path, required=True)
    annotate.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare", help="validate and split genuine labeled generation traces")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    preflight = sub.add_parser("preflight", help="check prepared data and training prerequisites")
    preflight.add_argument("--data-dir", type=Path, required=True)
    preflight.add_argument("--output-dir", type=Path, required=True)
    train = sub.add_parser("train", help="fit head, calibrate, select validation routing policy")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-false-accept", type=float, default=0.01)
    train.add_argument("--min-bypass", type=int, default=100)
    evaluate = sub.add_parser("evaluate", help="one-time held-out test evaluation")
    evaluate.add_argument("--data-dir", type=Path, required=True)
    evaluate.add_argument("--head-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        from .phase1_collect import collect as run
        result = run(args.input, args.output, model_id=args.model_id,
                     model_revision=args.model_revision, tokenizer_id=args.tokenizer_id,
                     tokenizer_revision=args.tokenizer_revision, dataset_id=args.dataset_id,
                     max_new_tokens=args.max_new_tokens, seed=args.seed)
    elif args.command == "annotation-template":
        from .phase1_data import annotation_template as run
        result = run(args.input, args.output)
    elif args.command == "prepare":
        from .phase1_data import prepare as run
        result = run(args.input, args.output_dir, args.seed)
    elif args.command == "preflight":
        from .phase1_train import preflight as run
        result = run(args.data_dir, args.output_dir)
    elif args.command == "train":
        from .phase1_train import train as run
        result = run(args.data_dir, args.output_dir, epochs=args.epochs,
                     learning_rate=args.learning_rate, seed=args.seed,
                     max_false_accept=args.max_false_accept, min_bypass=args.min_bypass)
    else:
        from .phase1_train import evaluate as run
        result = run(args.data_dir, args.head_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
