"""Paired genuine held-out inference. No downloads, training or artifact overwrite."""
import argparse
from pathlib import Path
from halluciguard_detector.paired_evaluation import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--raw-responses", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--exclude-cross-split-duplicates", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    run(args.data, args.checkpoint, args.output, args.batch_size, args.raw_responses,
        args.exclude_cross_split_duplicates)


if __name__ == "__main__":
    main()
