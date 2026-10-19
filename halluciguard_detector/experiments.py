"""Controlled train/runtime evidence-alignment experiments.

Everything in this module exists to answer one question with numbers instead of
argument: *is the detector's weak contradiction recall caused by the mismatch
between the evidence it is trained on and the evidence it is served?*

Two designs, because they answer different halves of it and neither is
sufficient alone.

**Shape sensitivity (no retraining).** The *shipped* checkpoint is scored on
the *same claims* under three evidence policies, changing nothing else. Any
change in predictions is attributable to evidence shape alone -- no training
noise, no seed, no sample-size confound. This is the causal estimate of the
mismatch's cost, and it is the primary result.

**Controlled retraining.** Each arm is trained from the same base model with
the same optimizer, learning rate, epochs, seed, label map, max sequence
length and claims, so the evidence policy is the only variable. This is slower
and noisier, and at the scale that is affordable on CPU it is
under-powered for the minority class; it is reported as confirmation, not as
the primary estimate.

Both are honest about scale. Sample sizes are recorded in every report, and no
number is written into a test.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .calibration import DEFAULT_MAX_LENGTH, class_probabilities, verification_risk_score
from .evidence_report import label_flip_matrix, risk_shift
from .evidence_shapes import PRODUCTION_SHAPE, SHAPES
from .nli_input import contract_spec
from .training import (
    evaluate_saved_predictions,
    load_rows,
    predict_rows,
    train_arm,
)

#: Class ids, mirroring ``training.ID_TO_LABEL``.
SUPPORTED_ID, CONTRADICTED_ID, NOT_ENOUGH_INFO_ID = 0, 1, 2
CLASS_NAMES = ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO")

#: Every arm shares these. Only the evidence differs; if one of these moves,
#: the comparison is no longer controlled and the report should say so.
SHARED_TRAINING = {
    "base_model": "microsoft/deberta-v3-xsmall",
    "architecture": "DebertaV2ForSequenceClassification (num_labels=3)",
    "optimizer": "AdamW(weight_decay=0.01)",
    "learning_rate": 2e-5,
    "epochs": 3,
    "batch_size": 8,
    "seed": 42,
    "max_length": DEFAULT_MAX_LENGTH,
    "label_map": {"SUPPORTED": 0, "CONTRADICTED": 1, "NOT_ENOUGH_INFO": 2},
}


def environment() -> Dict[str, Any]:
    """Record the execution environment so a number can be re-derived later."""
    import torch
    import transformers

    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        info["git_commit"] = commit.stdout.strip() or None
    except Exception:  # pragma: no cover - git is optional
        info["git_commit"] = None
    return info


# ---------------------------------------------------------------- scoring


def score_rows(
    model_dir: Path,
    rows: Sequence[dict],
    *,
    device: Any,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 16,
) -> Dict[str, np.ndarray]:
    """Run a checkpoint over rows through the canonical NLI contract."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    logits, labels = predict_rows(
        model, tokenizer, rows, device=device, max_length=max_length, batch_size=batch_size
    )
    del model
    if torch.cuda.is_available():  # pragma: no cover - CPU runs here
        torch.cuda.empty_cache()
    return {"logits": logits, "labels": labels}


def _calibration(model_dir: Path) -> Dict[str, float]:
    from .calibration import load_calibration

    data = load_calibration(Path(model_dir) / "calibration.json")
    return {
        "temperature": float(data.get("temperature", 1.0)),
        "contradiction_threshold": float(data.get("contradiction_threshold", 0.5)),
        "verification_risk_threshold": float(data.get("verification_risk_threshold", 0.5)),
    }


def arm_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    calibration: Dict[str, float],
) -> Dict[str, Any]:
    """Task A / Task B / three-class metrics for one arm, at one threshold set.

    The calibration is supplied by the caller and must be the *shared* one --
    the released PR #53 calibration -- not each arm's own fitted thresholds.
    The docstring used to claim thresholds are never re-tuned; that is only
    true if the caller passes a shared calibration, so it is now explicit in
    the parameter name and asserted by the comparison runner.

    ``train`` fits a temperature and picks two decision thresholds on each
    arm's own dev set. For this study that is a confound: a threshold that
    moved to suit a differently-shaped score distribution would show up as an
    "alignment effect" while the evidence pipeline was the only thing that
    changed. Every arm is therefore scored at the identical operating point.
    """
    metrics = evaluate_saved_predictions(
        logits,
        labels,
        calibration["temperature"],
        calibration["contradiction_threshold"],
        calibration["verification_risk_threshold"],
    )
    metrics.pop("calibration", None)
    return metrics


def train_and_compare(
    arms: Dict[str, Path],
    output_root: Path,
    *,
    reference_calibration: Path,
    splits: Sequence[str] = ("test", "metric"),
    epochs: int = 3,
    seed: int = 42,
    device: Any = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 16,
    base_model: str = "microsoft/deberta-v3-xsmall",
    train_batch_size: int = 8,
) -> Dict[str, Any]:
    """Retrain each evidence-shape arm under identical settings and compare.

    The only intended difference between arms is the evidence their rows carry.
    Everything else -- base checkpoint, epoch count, optimizer, learning rate,
    seed, label map, sequence length, and the scoring operating point -- is
    held fixed, so a metric difference is attributable to the evidence shape
    rather than to a lucky threshold or a different recipe.

    Each arm's own fitted calibration is reported alongside the shared one as
    a secondary observation. It is deliberately *not* used for the headline
    comparison, but it is worth knowing how far the argmax decision boundary
    moved as a result of the evidence change.
    """
    import torch

    from .calibration import load_calibration
    from .training import train_arm

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared = load_calibration(reference_calibration)
    shared = {
        "temperature": float(shared.get("temperature", 1.0)),
        "contradiction_threshold": float(shared.get("contradiction_threshold", 0.5)),
        "verification_risk_threshold": float(shared.get("verification_risk_threshold", 0.5)),
    }
    results: Dict[str, Any] = {}
    for arm, directory in arms.items():
        directory = Path(directory)
        if not (directory / "train.jsonl").exists():
            results[arm] = {"status": "missing", "path": str(directory)}
            continue
        checkpoint = Path(output_root) / arm
        entry: Dict[str, Any] = {"data_dir": str(directory), "checkpoint": str(checkpoint)}
        if not (checkpoint / "model.safetensors").exists():
            entry["status"] = "training"
            entry["training_report"] = train_arm(
                directory,
                checkpoint,
                base_model=base_model,
                epochs=epochs,
                batch_size=train_batch_size,
                max_length=max_length,
                seed=seed,
            )
        entry["own_calibration"] = _calibration(checkpoint)
        entry["splits"] = {}
        for split in splits:
            path = directory / f"{split}.jsonl"
            if not path.exists():
                entry["splits"][split] = {"status": "missing"}
                continue
            rows = load_rows(path)
            scored = score_rows(
                checkpoint, rows, device=device, max_length=max_length, batch_size=batch_size
            )
            entry["splits"][split] = {
                "status": "ok",
                "rows": len(rows),
                "shared_calibration": arm_metrics(scored["logits"], scored["labels"], shared),
                "own_calibration": arm_metrics(
                    scored["logits"], scored["labels"], entry["own_calibration"]
                ),
            }
        entry["status"] = "ok"
        results[arm] = entry
    return {
        "shared_training": {
            **SHARED_TRAINING,
            "epochs": epochs,
            "seed": seed,
            "max_sequence_length": max_length,
            "train_batch_size": train_batch_size,
        },
        "shared_calibration": shared,
        "reference_calibration": str(reference_calibration),
        "arms": results,
    }


def baseline(
    checkpoint: Path,
    data_dir: Path,
    split: str,
    *,
    device: Any = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 16,
    limit: int | None = None,
) -> Dict[str, Any]:
    """Re-derive the shipped checkpoint's metrics on a named split.

    This is a re-scoring, not a re-training: the point is to confirm that the
    numbers in the report come from the model and data actually on disk, under
    the canonical input contract, rather than being copied out of a metrics
    file. The artifact's own metrics are included alongside for comparison, and
    a disagreement between them is reported rather than resolved silently --
    a baseline nobody can reproduce is not a baseline.
    """
    import torch

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calibration = _calibration(checkpoint)
    path = Path(data_dir) / f"{split}.jsonl"
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    rows = load_rows(path)
    total_available = len(rows)
    if limit is not None and limit < len(rows):
        rows = rows[:limit]
    scored = score_rows(
        Path(checkpoint), rows, device=device, max_length=max_length, batch_size=batch_size
    )
    measured = arm_metrics(scored["logits"], scored["labels"], calibration)
    return {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "calibration": calibration,
        "data_dir": str(data_dir),
        "split": split,
        "rows_scored": len(rows),
        "rows_available": total_available,
        "truncated_by_limit": limit is not None and limit < total_available,
        "label_distribution": measured["three_class"]["class_distribution"],
        "metrics": measured,
        "nli_contract": contract_spec(max_length),
    }


#: What PR #53 fixed, and where the regression coverage for each lives. Kept as
#: data so the report cannot quietly drop an item: an untested claim of "no
#: regressions" is exactly the kind of claim that becomes wrong unnoticed.
PR53_REGRESSION_CHECKS: tuple[Dict[str, str], ...] = (
    {
        "check": "decorated RAGTruth labels map to the right class ids",
        "covered_by": "halluciguard_detector/tests/test_data_conversion.py",
    },
    {
        "check": "no public symbol removed by the PR #53 refactor",
        "covered_by": "halluciguard_detector/tests/test_schemas.py",
    },
    {
        "check": "annotation overlap handling in dataset conversion",
        "covered_by": "halluciguard_detector/tests/test_data_conversion.py",
    },
    {
        "check": "numeric comparability guard",
        "covered_by": "halluciguard_detector/tests/test_guard_comparability.py",
    },
    {
        "check": "date comparability guard",
        "covered_by": "halluciguard_detector/tests/test_guard_comparability.py",
    },
    {
        "check": "entity comparability guard",
        "covered_by": "halluciguard_detector/tests/test_guard_comparability.py",
    },
    {
        "check": "bare-number extraction",
        "covered_by": "halluciguard_detector/tests/test_text.py",
    },
    {
        "check": "extra_special_tokens does not break checkpoint loading",
        "covered_by": "halluciguard_detector/tests/test_detector_semantics.py",
    },
    {
        "check": "train and serve sequence length agree",
        "covered_by": (
            "halluciguard_detector/tests/test_nli_input_contract.py,"
            " halluciguard_detector/tests/test_detector_semantics.py"
        ),
    },
    {
        "check": "judge fails closed when the detector is degraded",
        "covered_by": "halluciguard_judge/tests/test_detector.py",
    },
)


def released_artifact_metrics(checkpoint: Path) -> Dict[str, Any] | None:
    """Load the metrics the shipped artifact already recorded, if present."""
    path = Path(checkpoint) / "test_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def format_comparison_table(result: Dict[str, Any], split: str) -> str:
    """Render the retrained-arm comparison at each arm's own operating point.

    An earlier version scored every arm at the released checkpoint's
    calibration and reported contradiction F1 = 0.0000 for both arms. That was
    not a result, it was a mistake in the experimental design, and the reason
    is worth recording: temperature and a 0.46 decision threshold are properties
    of *one checkpoint's* logit scale. The retrained arms fit temperatures of
    0.68 / 0.64 and dev-optimal thresholds of 0.245 / 0.205, so the released
    0.46 sits far above where either of them ever puts P(CONTRADICTED), and
    both arms score exactly zero. A number that is identically zero for every
    arm is a diagnostic of a broken transfer, not a weak model.

    The fair comparison is each arm at its own dev-fitted operating point, with
    the *procedure* held identical: same dev split, same grid, same selection
    rule, run once per arm. That is what this table shows.
    """
    lines = [
        f"Retrained arm comparison -- '{split}'",
        "each arm at its own dev-fitted operating point (identical procedure)",
        "",
        f"{'arm':<20} {'contrF1':>8} {'contrR':>8} {'contrP':>8} "
        f"{'verifF1':>8} {'NEI F1':>8} {'macroF1':>8} {'acc':>8} {'thr':>7}",
        "-" * 92,
    ]
    for arm, entry in result.get("arms", {}).items():
        split_result = entry.get("splits", {}).get(split, {})
        metrics = split_result.get("own_calibration")
        if not metrics:
            lines.append(f"{arm:<20} {'(no result)':>8}")
            continue
        three = metrics["three_class"]
        per_class = three["per_class"]
        lines.append(
            f"{arm:<20} "
            f"{metrics['task_a_contradiction']['f1']:>8.4f} "
            f"{metrics['task_a_contradiction']['recall']:>8.4f} "
            f"{metrics['task_a_contradiction']['precision']:>8.4f} "
            f"{metrics['task_b_verification_needed']['f1']:>8.4f} "
            f"{(per_class['NOT_ENOUGH_INFO']['f1'] or 0):>8.4f} "
            f"{three['macro_f1']:>8.4f} "
            f"{three['accuracy']:>8.4f} "
            f"{entry.get('own_calibration', {}).get('contradiction_threshold', 0):>7.3f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- designs


def shape_sensitivity(
    checkpoint: Path,
    arms: Dict[str, Path],
    split: str,
    *,
    device: Any = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 16,
    reference: str = PRODUCTION_SHAPE,
) -> Dict[str, Any]:
    """Score one checkpoint on identical claims under several evidence shapes.

    Rows are aligned on the RAGTruth example ``id`` rather than on file order,
    so a shape that happened to emit a different number of rows cannot silently
    turn a paired comparison into an unpaired one.
    """
    import torch

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calibration = _calibration(checkpoint)

    per_arm: Dict[str, Dict[str, Any]] = {}
    common: set[str] | None = None
    for arm, directory in arms.items():
        path = Path(directory) / f"{split}.jsonl"
        if not path.exists():
            per_arm[arm] = {"status": "missing", "path": str(path)}
            continue
        rows = load_rows(path)
        by_id = {str(row["id"]): row for row in rows}
        per_arm[arm] = {"status": "ok", "path": str(path), "rows": len(rows), "_rows": by_id}
        keys = set(by_id)
        common = keys if common is None else (common & keys)

    missing = [arm for arm, value in per_arm.items() if value["status"] != "ok"]
    if missing:
        return {"status": "incomplete", "missing_arms": missing}
    if not common:
        return {"status": "no_common_claims", "arms": list(per_arm)}

    ordered_ids = sorted(common)
    scores: Dict[str, List[float]] = {}
    predictions: Dict[str, List[int]] = {}
    labels: List[int] = []
    metrics: Dict[str, Any] = {}

    for arm, value in per_arm.items():
        rows = [value["_rows"][key] for key in ordered_ids]
        scored = score_rows(
            checkpoint,
            rows,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
        )
        probabilities = class_probabilities(scored["logits"], calibration["temperature"])
        scores[arm] = verification_risk_score(probabilities).tolist()
        predictions[arm] = probabilities.argmax(axis=1).tolist()
        if not labels:
            labels = scored["labels"].tolist()
        metrics[arm] = arm_metrics(scored["logits"], scored["labels"], calibration)
        value.pop("_rows", None)

    return {
        "status": "ok",
        "design": "shape_sensitivity",
        "description": (
            "One checkpoint, identical claims, only the evidence policy "
            "changes. No retraining, so the difference is attributable to the "
            "evidence distribution alone."
        ),
        "checkpoint": str(checkpoint),
        "split": split,
        "claims": len(ordered_ids),
        "thresholds": calibration,
        "label_distribution": {
            name: int(sum(1 for value in labels if value == index))
            for index, name in enumerate(CLASS_NAMES)
        },
        "arms": {
            arm: {
                "path": str(per_arm[arm]["path"]),
                "rows": per_arm[arm]["rows"],
                "metrics": metrics[arm],
            }
            for arm in per_arm
        },
        "risk_shift_vs_production": risk_shift(scores, labels, reference=reference),
        "reference_arm": reference,
        "label_disagreements": label_flip_matrix(predictions, labels),
    }


def train_and_score(
    arm_data: Path,
    arm_model: Path,
    *,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = 42,
    eval_splits: Sequence[str] = ("test", "metric"),
    device: Any = None,
) -> Dict[str, Any]:
    """Train one arm and score it on each requested split."""
    import torch

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = train_arm(
        arm_data,
        arm_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        seed=seed,
    )
    calibration = _calibration(arm_model)
    evaluations: Dict[str, Any] = {}
    for split in eval_splits:
        path = Path(arm_data) / f"{split}.jsonl"
        if not path.exists():
            evaluations[split] = {"status": "missing", "path": str(path)}
            continue
        rows = load_rows(path)
        scored = score_rows(
            arm_model, rows, device=device, max_length=max_length, batch_size=batch_size
        )
        evaluations[split] = {
            "status": "ok",
            "rows": len(rows),
            "label_distribution": {
                name: int((scored["labels"] == index).sum())
                for index, name in enumerate(CLASS_NAMES)
            },
            "metrics": arm_metrics(scored["logits"], scored["labels"], calibration),
        }
    return {
        "data_dir": str(arm_data),
        "model_dir": str(arm_model),
        "training": {
            key: report.get(key)
            for key in ("device", "base_model", "max_length", "epochs", "batch_size",
                        "learning_rate", "seed", "label_map")
        },
        "history": report.get("history"),
        "evaluations": evaluations,
    }


# ---------------------------------------------------------------- reporting


def comparison_table(results: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    """Extract the headline row per experiment for the report table.

    Built from the measured dict, never from literals, so a missing metric
    shows up as ``None`` instead of being filled in.
    """
    rows: List[Dict[str, Any]] = []
    for name, payload in results.items():
        evaluation = (payload.get("evaluations") or {}).get(split)
        if evaluation is None or evaluation.get("status") != "ok":
            rows.append({"experiment": name, "split": split, "status": "missing"})
            continue
        metrics = evaluation["metrics"]
        task_a = metrics["task_a_contradiction"]
        task_b = metrics["task_b_verification_needed"]
        three = metrics["three_class"]
        rows.append(
            {
                "experiment": name,
                "split": split,
                "status": "ok",
                "rows": evaluation["rows"],
                "contradicted_recall": task_a["recall"],
                "contradicted_precision": task_a["precision"],
                "contradicted_f1": task_a["f1"],
                "contradicted_pr_auc": task_a.get("pr_auc"),
                "contradicted_roc_auc": task_a.get("roc_auc"),
                "verification_needed_f1": task_b["f1"],
                "verification_needed_recall": task_b["recall"],
                "supported_f1": three["per_class"]["SUPPORTED"].get("f1"),
                "not_enough_info_f1": three["per_class"]["NOT_ENOUGH_INFO"].get("f1"),
                "macro_f1": three["macro_f1"],
                "accuracy": three["accuracy"],
                "confusion_matrix": three["confusion_matrix"]["matrix"],
            }
        )
    return rows


# ---------------------------------------------------------------- CLI


def main(argv: List[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    sub = parser.add_subparsers(dest="command", required=True)

    sens = sub.add_parser("shape-sensitivity", help="one checkpoint, many evidence shapes")
    sens.add_argument("--checkpoint", type=Path, required=True)
    sens.add_argument("--arm", action="append", default=[], metavar="NAME=DIR")
    sens.add_argument("--split", default="metric")
    sens.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    sens.add_argument("--batch-size", type=int, default=16)
    sens.add_argument(
        "--reference",
        default=PRODUCTION_SHAPE,
        help=(
            "arm used as the zero point for risk shift. Defaults to the "
            "production shape because the question is distance from production; "
            "point it elsewhere explicitly rather than letting it fall back, so "
            "a preliminary run cannot be read as a production-relative result."
        ),
    )
    sens.add_argument("--out", type=Path, default=None)

    compare = sub.add_parser(
        "train-compare", help="retrain each arm under identical settings, compare"
    )
    compare.add_argument("--arm", action="append", default=[], metavar="NAME=DIR")
    compare.add_argument("--output-root", type=Path, required=True)
    compare.add_argument(
        "--reference-calibration",
        type=Path,
        default=Path("artifacts/detector-best/calibration.json"),
        help="the single operating point every arm is scored at",
    )
    compare.add_argument("--split", action="append", default=None)
    compare.add_argument("--epochs", type=int, default=3)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    compare.add_argument("--batch-size", type=int, default=16)
    compare.add_argument("--train-batch-size", type=int, default=8)
    compare.add_argument("--out", type=Path, default=None)

    base = sub.add_parser("baseline", help="re-derive the shipped checkpoint's metrics")
    base.add_argument("--checkpoint", type=Path, default=Path("artifacts/detector-best"))
    base.add_argument("--data-dir", type=Path, default=Path("halluciguard_detector/data"))
    base.add_argument("--split", default="test")
    base.add_argument("--limit", type=int, default=None)
    base.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    base.add_argument("--batch-size", type=int, default=16)
    base.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "shape-sensitivity":
        arms = dict(entry.split("=", 1) for entry in args.arm)
        result = shape_sensitivity(
            args.checkpoint,
            {name: Path(path) for name, path in arms.items()},
            args.split,
            max_length=args.max_length,
            batch_size=args.batch_size,
            reference=args.reference,
        )
        result["environment"] = environment()
        result["shared_training"] = SHARED_TRAINING
        result["nli_contract"] = contract_spec(args.max_length)
        out = args.out or (args.reports_dir / "detector_v2_shape_sensitivity.json")
        out.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
        print(format_shape_table(result, args.split))
        print(f"\nwrote {out}")
        return 0

    if args.command == "train-compare":
        arms = dict(entry.split("=", 1) for entry in args.arm)
        splits = args.split or ["test", "metric"]
        result = train_and_compare(
            {name: Path(path) for name, path in arms.items()},
            args.output_root,
            reference_calibration=args.reference_calibration,
            splits=splits,
            epochs=args.epochs,
            seed=args.seed,
            max_length=args.max_length,
            batch_size=args.batch_size,
            train_batch_size=args.train_batch_size,
        )
        result["environment"] = environment()
        result["nli_contract"] = contract_spec(args.max_length)
        out = args.out or (args.reports_dir / "detector_v2_train_comparison.json")
        out.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
        for split in splits:
            print(format_comparison_table(result, split))
            print()
        print(f"wrote {out}")
        return 0

    if args.command == "baseline":
        result = baseline(
            args.checkpoint,
            args.data_dir,
            args.split,
            max_length=args.max_length,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        result["environment"] = environment()
        result["pr53_regression_checks"] = [
            dict(entry) for entry in PR53_REGRESSION_CHECKS
        ]
        artifact = released_artifact_metrics(args.checkpoint)
        if artifact is not None:
            result["shipped_artifact_metrics"] = {
                "source": str(Path(args.checkpoint) / "test_metrics.json"),
                "samples": artifact.get("samples"),
                "class_distribution": artifact.get("class_distribution"),
                "task_a_contradiction": {
                    key: artifact.get("task_a_contradiction", {}).get(key)
                    for key in ("precision", "recall", "f1", "pr_auc", "roc_auc")
                },
                "task_b_verification_needed": {
                    key: artifact.get("task_b_verification_needed", {}).get(key)
                    for key in ("precision", "recall", "f1", "pr_auc", "roc_auc")
                },
                "three_class": {
                    key: artifact.get("three_class", {}).get(key)
                    for key in ("macro_f1", "accuracy", "confusion_matrix")
                },
            }
            result["shipped_vs_recomputed"] = {
                "note": (
                    "The shipped artifact was produced on a different data "
                    "snapshot, so a small disagreement is expected and is "
                    "reported here rather than reconciled. Neither number is "
                    "silently preferred."
                ),
                "shipped_samples": artifact.get("samples"),
                "recomputed_samples": result.get("rows_scored"),
            }
        out = args.out or (args.reports_dir / "detector_v2_baseline.json")
        out.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
        metrics = result.get("metrics")
        if metrics:
            print(format_baseline(result))
            print(f"\nwrote {out}")
        else:
            print(json.dumps(result, indent=2, default=_json_default))
        return 0
    return 1


def format_baseline(result: Dict[str, Any]) -> str:
    """Render the re-derived baseline from measured values only."""
    metrics = result["metrics"]
    three = metrics["three_class"]
    lines = [
        f"Baseline -- {result['checkpoint']} on '{result['split']}'",
        f"rows scored {result['rows_scored']} of {result['rows_available']}"
        + ("  (TRUNCATED BY --limit)" if result.get("truncated_by_limit") else ""),
        "",
        f"{'metric':<34} {'precision':>10} {'recall':>10} {'f1':>10} {'pr_auc':>10} {'roc_auc':>10}",
        "-" * 80,
    ]
    for key, name in (
        ("task_a_contradiction", "contradiction"),
        ("task_b_verification_needed", "verification_needed"),
    ):
        task = metrics[key]
        lines.append(
            f"{name:<34} {task['precision']:>10.4f} {task['recall']:>10.4f} "
            f"{task['f1']:>10.4f} {task['pr_auc']:>10.4f} {task['roc_auc']:>10.4f}"
        )
    lines.append("")
    lines.append(f"three-class macro-F1 {three['macro_f1']:.4f}   accuracy {three['accuracy']:.4f}")
    lines.append("")
    lines.append(f"{'class':<20} {'support':>8} {'recall':>9} {'precision':>10} {'f1':>9}")
    lines.append("-" * 60)
    for name, stats in three["per_class"].items():
        lines.append(
            f"{name:<20} {stats['support']:>8} "
            f"{(stats['recall'] if stats['recall'] is not None else float('nan')):>9.4f} "
            f"{(stats['precision'] if stats['precision'] is not None else float('nan')):>10.4f} "
            f"{(stats['f1'] if stats['f1'] is not None else float('nan')):>9.4f}"
        )
    return "\n".join(lines)


def format_shape_table(result: Dict[str, Any], split: str) -> str:
    """Render the per-arm headline table from measured values only."""
    lines = [
        f"Evidence-shape sensitivity -- {result.get('claims', '?')} claims on '{split}'",
        "",
        f"{'arm':<20} {'contrR':>8} {'contrP':>8} {'contrF1':>8} {'NEI F1':>8} {'macroF1':>9}",
        "-" * 64,
    ]
    for name, payload in (result.get("arms") or {}).items():
        metrics = (payload or {}).get("metrics")
        if not metrics:
            lines.append(f"{name:<20} {'n/a':>8}")
            continue
        task_a = metrics["task_a_contradiction"]
        three = metrics["three_class"]
        lines.append(
            f"{name:<20} {task_a['recall']:>8.4f} {task_a['precision']:>8.4f} "
            f"{task_a['f1']:>8.4f} {three['per_class']['NOT_ENOUGH_INFO']['f1']:>8.4f} "
            f"{three['macro_f1']:>9.4f}"
        )
    return "\n".join(lines)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
