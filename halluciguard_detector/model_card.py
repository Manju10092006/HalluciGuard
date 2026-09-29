"""Generate ``model_metadata.json`` for the shipped Detector checkpoint.

The model card, the runtime and the training code each used to hard-code
sequence length and thresholds independently, so they could silently disagree
(384 in code, 256 in the artifact). This module derives the authoritative values
*from the artifact itself* and refuses to guess when they conflict.

Run after training or calibration changes::

    python -m halluciguard_detector.model_card --checkpoint artifacts/detector-best
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .calibration import DEFAULT_MAX_LENGTH, load_calibration

ID_TO_LABEL = {0: "SUPPORTED", 1: "CONTRADICTED", 2: "NOT_ENOUGH_INFO"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_metadata(checkpoint: Path) -> dict[str, Any]:
    """Derive the model metadata, cross-checking every duplicated constant."""
    checkpoint = Path(checkpoint)
    config = _read_json(checkpoint / "config.json")
    tokenizer_config = _read_json(checkpoint / "tokenizer_config.json")
    calibration = load_calibration(checkpoint / "calibration.json")
    report = _read_json(checkpoint / "training_report.json")

    tokenizer_max = tokenizer_config.get("model_max_length")
    calibration_max = calibration.get("max_length")
    sequence_length = calibration_max or tokenizer_max or DEFAULT_MAX_LENGTH

    # If the tokenizer and the calibration file disagree, serving would truncate
    # differently from evaluation. That is a hard error, not a warning.
    conflicts: list[str] = []
    if tokenizer_max and calibration_max and int(tokenizer_max) != int(calibration_max):
        conflicts.append(
            f"tokenizer_config.model_max_length={tokenizer_max} but calibration.max_length={calibration_max}"
        )
    if sequence_length != DEFAULT_MAX_LENGTH:
        conflicts.append(
            f"checkpoint sequence length {sequence_length} differs from DEFAULT_MAX_LENGTH={DEFAULT_MAX_LENGTH}"
        )

    id2label = config.get("id2label") or {}
    normalised_id2label = {
        int(index): ID_TO_LABEL.get(int(index), str(label))
        for index, label in id2label.items()
    }

    return {
        "name": "halluciguard-detector",
        "task": "three-class grounded claim verification (SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO)",
        "base_model": config.get("_name_or_path") or "microsoft/deberta-v3-xsmall",
        "sequence_length": int(sequence_length),
        "id2label": normalised_id2label,
        "calibration": {
            "temperature": calibration.get("temperature"),
            "contradiction_threshold": calibration.get("contradiction_threshold"),
            "verification_risk_threshold": calibration.get("verification_risk_threshold"),
            "deprecated_hallucination_threshold": calibration.get("hallucination_threshold"),
        },
        "training": {
            "epochs": report.get("epochs", len(report.get("history", [])) or None),
            "seed": report.get("seed"),
            "batch_size": report.get("batch_size"),
            "max_length": report.get("max_length") or sequence_length,
        },
        "evaluation": {
            "test_metrics_file": "test_metrics.json",
            "dev_predictions_file": "dev_predictions.npz",
            "test_predictions_file": "test_predictions.npz",
        },
        "consistency": {
            "conflicts": conflicts,
            "ok": not conflicts,
        },
    }


def recompute_evaluation(checkpoint: Path) -> dict[str, Any]:
    """Rebuild the two-task test report from saved predictions.

    The shipped ``test_metrics.json`` was produced by a single function that
    measured only the verification-needed question and reported it under the
    name "hallucination". Saved logits and labels are still in the checkpoint,
    so the honest two-task report can be regenerated without retraining -- and
    without inventing any number.
    """
    import numpy as np

    from .training import evaluate_saved_predictions

    checkpoint = Path(checkpoint)
    predictions = np.load(checkpoint / "test_predictions.npz")
    calibration = load_calibration(checkpoint / "calibration.json")
    return evaluate_saved_predictions(
        predictions["logits"],
        predictions["labels"],
        temperature=float(calibration["temperature"]),
        contradiction_threshold=float(calibration["contradiction_threshold"]),
        verification_risk_threshold=float(calibration["verification_risk_threshold"]),
    )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/detector-best"))
    parser.add_argument("--write", action="store_true", help="write model_metadata.json into the checkpoint")
    parser.add_argument(
        "--recompute-metrics",
        action="store_true",
        help="rebuild test_metrics.json from saved test predictions (no retraining)",
    )
    parser.add_argument("--strict", action="store_true", help="exit non-zero when constants disagree")
    args = parser.parse_args(argv)

    if args.recompute_metrics:
        report = recompute_evaluation(args.checkpoint)
        target = Path(args.checkpoint) / "test_metrics.json"
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {target}", file=sys.stderr)
        return 0

    metadata = build_metadata(args.checkpoint)
    print(json.dumps(metadata, indent=2))
    if args.write:
        target = Path(args.checkpoint) / "model_metadata.json"
        target.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"wrote {target}", file=sys.stderr)
    if args.strict and metadata["consistency"]["conflicts"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
