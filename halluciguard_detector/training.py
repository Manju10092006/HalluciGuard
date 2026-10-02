import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .calibration import (
    DEFAULT_MAX_LENGTH,
    DEPRECATED_HALLUCINATION_THRESHOLD,
    class_probabilities,
    contradiction_score,
    fit_temperature,
    load_calibration,
    save_calibration,
    verification_risk_score,
)
from .nli_input import TOKENIZER_KWARGS, prepare_nli_input


ID_TO_LABEL = {0: "SUPPORTED", 1: "CONTRADICTED", 2: "NOT_ENOUGH_INFO"}
SUPPORTED_ID, CONTRADICTED_ID, NOT_ENOUGH_INFO_ID = 0, 1, 2


class JsonlDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(line) for line in path.open(encoding="utf-8")]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _collator(tokenizer, max_length: int):
    def collate(rows):
        # One shared NLI contract for training, dev and test. The pair order and
        # the truncation policy must not drift from the runtime path, otherwise
        # the model is scored on a distribution it never saw.
        pairs = [prepare_nli_input(x["claim"], x["evidence"]) for x in rows]
        batch = tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            max_length=max_length,
            return_tensors="pt",
            **TOKENIZER_KWARGS,
        )
        batch["labels"] = torch.tensor([x["label_id"] for x in rows], dtype=torch.long)
        return batch

    return collate


def predict_rows(
    model,
    tokenizer,
    rows,
    *,
    device,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Score already-selected rows through the canonical NLI contract.

    Every arm in the evidence-alignment study goes through this function, so a
    difference between two arms can only come from the rows themselves and not
    from a difference in how they were encoded. ``rows`` are dicts with
    ``claim`` / ``evidence`` / ``label_id``; the order of the returned arrays
    matches the input order, which is what makes a paired per-claim
    comparison between arms valid.
    """
    loader = DataLoader(
        _ListDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collator(tokenizer, max_length),
    )
    return _evaluate(model, loader, device)


class _ListDataset(Dataset):
    """Adapter so in-memory rows can reuse the single shared collator."""

    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def load_rows(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits.cpu()
            all_logits.append(logits)
            all_labels.append(labels)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def _expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            continue
        total += float(mask.mean()) * abs(
            float(confidence[mask].mean()) - float(correct[mask].mean())
        )
    return total


def _reliability_bins(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[dict] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        out.append(
            {
                "lower": round(float(low), 4),
                "upper": round(float(high), 4),
                "count": int(mask.sum()),
                "mean_confidence": round(float(confidence[mask].mean()), 6) if mask.any() else None,
                "empirical_rate": round(float(correct[mask].mean()), 6) if mask.any() else None,
            }
        )
    return out


def binary_task_metrics(
    score: np.ndarray,
    truth: np.ndarray,
    threshold: float = 0.5,
    task: str = "",
) -> dict:
    """Metrics for one binary view of the three-class model.

    The caller supplies the score and the ground truth, which is the whole
    point: contradiction and verification-needed are different questions over
    the same distribution and must be measured separately.
    """
    score = np.asarray(score, dtype=np.float64)
    truth = np.asarray(truth).astype(int)
    pred = (score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    positives = int(truth.sum())
    negatives = int(len(truth) - positives)
    out = {
        "task": task,
        "score": "P(CONTRADICTED)" if task == "contradiction" else "P(CONTRADICTED)+P(NOT_ENOUGH_INFO)",
        "positive_class": "CONTRADICTED" if task == "contradiction" else "CONTRADICTED+NOT_ENOUGH_INFO",
        "support": {"positives": positives, "negatives": negatives, "total": int(len(truth))},
        "positive_rate": float(positives / max(1, len(truth))),
        "accuracy": float((truth == pred).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "false_negative_rate": float(fn / max(1, fn + tp)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": float(threshold),
        "brier": float(np.mean((score - truth) ** 2)),
        "ece": _expected_calibration_error(score, truth.astype(np.float64)),
        "reliability": _reliability_bins(score, truth.astype(np.float64)),
    }
    # Ranking metrics are undefined when one class is absent from the split.
    if 0 < positives < len(truth):
        out["roc_auc"] = float(roc_auc_score(truth, score))
        out["pr_auc"] = float(average_precision_score(truth, score))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
        out["note"] = "ranking metrics undefined: only one class present"
    return out


def contradiction_metrics(
    logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0, threshold: float = 0.5
) -> dict:
    """Task A -- does the evidence REFUTE the claim?

    Positive: CONTRADICTED. Negative: SUPPORTED + NOT_ENOUGH_INFO.
    """
    probabilities = class_probabilities(logits, temperature)
    return binary_task_metrics(
        contradiction_score(probabilities),
        (np.asarray(labels) == CONTRADICTED_ID).astype(int),
        threshold,
        task="contradiction",
    )


def verification_needed_metrics(
    logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0, threshold: float = 0.5
) -> dict:
    """Task B -- does the claim still need checking?

    Positive: CONTRADICTED + NOT_ENOUGH_INFO. Negative: SUPPORTED.
    """
    probabilities = class_probabilities(logits, temperature)
    return verification_risk_metrics_from_probabilities(
        probabilities, labels, threshold
    )


def verification_risk_metrics_from_probabilities(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> dict:
    return binary_task_metrics(
        verification_risk_score(probabilities),
        (np.asarray(labels) != SUPPORTED_ID).astype(int),
        threshold,
        task="verification_needed",
    )


def three_class_metrics(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> dict:
    """Argmax view of the native three-way decision, plus per-class accuracy.

    Accuracy here is not comparable to the binary tasks: it is dominated by the
    SUPPORTED majority, so it is reported only alongside per-class recall.
    """
    probabilities = class_probabilities(logits, temperature)
    labels = np.asarray(labels).astype(int)
    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1, 2])
    # Per-class P/R/F1 are reported alongside macro-F1 because macro-F1 hides
    # which class is carrying the average. Contradiction F1 in particular is the
    # axis this detector is weakest on, and it is invisible inside the mean.
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=[0, 1, 2], average=None, zero_division=0
    )
    per_class = {}
    for index, name in ((0, "SUPPORTED"), (1, "CONTRADICTED"), (2, "NOT_ENOUGH_INFO")):
        support = int((labels == index).sum())
        predicted_count = int((predicted == index).sum())
        hits = int(((labels == index) & (predicted == index)).sum())
        per_class[name] = {
            "support": support,
            "predicted": predicted_count,
            "correct": hits,
            # Recall: of the true class instances, how many were recovered.
            "recall": round(hits / support, 6) if support else None,
            # Precision: of the predicted instances, how many were right.
            "precision": round(hits / predicted_count, 6) if predicted_count else None,
            "f1": round(float(f1[index]), 6) if support or predicted_count else None,
        }
    return {
        "accuracy": float((labels == predicted).mean()),
        "macro_f1": float(
            precision_recall_fscore_support(
                labels, predicted, labels=[0, 1, 2], average="macro", zero_division=0
            )[2]
        ),
        "confusion_matrix": {
            "labels": ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"],
            "rows_are_true": True,
            "matrix": matrix.tolist(),
        },
        "per_class": per_class,
        "class_distribution": {
            name: per_class[name]["support"] for name in ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO")
        },
        "majority_class_share": round(
            float(max((labels == index).mean() for index in (0, 1, 2))), 6
        ),
    }


def calibration_report(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> dict:
    """Multiclass calibration of all three probabilities.

    Brier score and ECE are reported per class as well as overall, because a
    good average can hide a class whose confidence is meaningless. A softmax
    output should still be read as a *model score* unless these numbers
    support a stronger claim on the measured distribution.
    """
    probabilities = class_probabilities(logits, temperature)
    labels = np.asarray(labels).astype(int)
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(labels)), labels] = 1.0
    per_class = {}
    for index, name in ((0, "SUPPORTED"), (1, "CONTRADICTED"), (2, "NOT_ENOUGH_INFO")):
        confidence = probabilities[:, index]
        truth = one_hot[:, index]
        per_class[name] = {
            "brier": round(float(np.mean((confidence - truth) ** 2)), 6),
            "ece": round(_expected_calibration_error(confidence, truth), 6),
            "mean_confidence": round(float(confidence.mean()), 6),
            "empirical_frequency": round(float(truth.mean()), 6),
        }
    return {
        "temperature": float(temperature),
        "multiclass_brier": round(float(np.mean((probabilities - one_hot) ** 2)), 6),
        "multiclass_ece": round(
            float(
                np.mean(
                    [
                        _expected_calibration_error(probabilities[:, i], one_hot[:, i])
                        for i in range(3)
                    ]
                )
            ),
            6,
        ),
        "per_class": per_class,
        "interpretation": (
            "These metrics describe the supplied evaluation labels. Temperature-fitting "
            "provenance must be established separately; metrics do not transfer to other "
            "domains without re-measurement."
        ),
    }


def best_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    task: str,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """F1-optimal threshold for one task, selected on the given split."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)
    probabilities = class_probabilities(logits, temperature)
    if task == "contradiction":
        score = contradiction_score(probabilities)
        truth = (np.asarray(labels) == CONTRADICTED_ID).astype(int)
    elif task == "verification_needed":
        score = verification_risk_score(probabilities)
        truth = (np.asarray(labels) != SUPPORTED_ID).astype(int)
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown task: {task}")
    best_f1, best_threshold_value = -1.0, 0.5
    for candidate in grid:
        metrics = binary_task_metrics(score, truth, float(candidate), task=task)
        if metrics["f1"] > best_f1:
            best_f1, best_threshold_value = metrics["f1"], float(candidate)
    return best_threshold_value, best_f1


def evaluate_saved_predictions(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    contradiction_threshold: float,
    verification_risk_threshold: float,
) -> dict:
    """Assemble the full honest report for one split."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(logits) or not len(labels):
        raise ValueError("evaluation labels must be nonempty and aligned with logits")
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("evaluation labels must use the exact three-class mapping")
    return {
        "samples": int(len(labels)),
        "class_distribution": {
            name: int((np.asarray(labels) == index).sum())
            for index, name in ((0, "SUPPORTED"), (1, "CONTRADICTED"), (2, "NOT_ENOUGH_INFO"))
        },
        "three_class": three_class_metrics(logits, labels, temperature),
        "task_a_contradiction": contradiction_metrics(
            logits, labels, temperature, contradiction_threshold
        ),
        "task_b_verification_needed": verification_needed_metrics(
            logits, labels, temperature, verification_risk_threshold
        ),
        "calibration": calibration_report(logits, labels, temperature),
    }


#: Deprecated. Kept so existing notebooks/callers keep working, but the name
#: encoded the exact confusion this module removes: it always measured
#: verification-needed, never contradiction. Use the two explicit functions.
def binary_metrics(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0, threshold: float = 0.5):
    return verification_needed_metrics(logits, labels, temperature, threshold)



def train(
    data_dir: Path,
    output_dir: Path,
    base_model: str = "microsoft/deberta-v3-xsmall",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = 42,
    calibration_data: Path | None = None,
    selection_metric: str = "verification_f1",
    local_files_only: bool = False,
    gradient_checkpointing: bool = False,
) -> dict:
    if selection_metric not in {"verification_f1", "macro_f1"}:
        raise ValueError("unsupported checkpoint selection metric")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"model output already exists: {output_dir}")
    recovery_dir = output_dir.parent / f"{output_dir.name}-last"
    if recovery_dir.exists() and any(recovery_dir.iterdir()):
        raise FileExistsError(f"recovery output already exists: {recovery_dir}")
    train_rows = load_rows(data_dir / "train.jsonl")
    dev_rows = load_rows(data_dir / "dev.jsonl")
    if not train_rows or not dev_rows:
        raise ValueError("train and dev splits must be nonempty")
    for name, rows in (("train", train_rows), ("dev", dev_rows)):
        for row in rows:
            if row.get("label_id") not in ID_TO_LABEL or row.get("label") != ID_TO_LABEL[row["label_id"]]:
                raise ValueError(f"invalid {name} label")
    train_groups = {str(row["source_id"]) for row in train_rows if row.get("source_id") is not None}
    dev_groups = {str(row["source_id"]) for row in dev_rows if row.get("source_id") is not None}
    if train_groups & dev_groups:
        raise ValueError("train/dev source groups overlap")
    calibration_rows = load_rows(calibration_data) if calibration_data is not None else dev_rows
    if not calibration_rows:
        raise ValueError("calibration split must be nonempty")
    for row in calibration_rows:
        if row.get("label_id") not in ID_TO_LABEL or row.get("label") != ID_TO_LABEL[row["label_id"]]:
            raise ValueError("invalid calibration label")
    if calibration_data is not None:
        if any(not str(row.get("source_id") or "").strip() for row in train_rows + dev_rows + calibration_rows):
            raise ValueError("independent calibration requires source group identifiers")
        calibration_groups = {str(row["source_id"]) for row in calibration_rows}
        if calibration_groups & (train_groups | dev_groups):
            raise ValueError("calibration source groups overlap training or validation")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id={v: k for k, v in ID_TO_LABEL.items()},
        local_files_only=local_files_only,
    ).float().to(device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    collate = _collator(tokenizer, max_length)
    train_loader = DataLoader(JsonlDataset(data_dir / "train.jsonl"), batch_size=batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(JsonlDataset(data_dir / "dev.jsonl"), batch_size=batch_size, shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.06), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1 = -1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
            if step % 100 == 0 or step == len(train_loader):
                print(
                    f"epoch={epoch + 1}/{epochs} step={step}/{len(train_loader)} "
                    f"loss={np.mean(losses[-100:]):.4f}",
                    flush=True,
                )
        # Persist completed work before evaluation; a late resource error must not
        # discard a full epoch. This is overwritten only by another completed epoch.
        recovery_dir = output_dir.parent / f"{output_dir.name}-last"
        model.save_pretrained(recovery_dir, safe_serialization=True)
        tokenizer.save_pretrained(recovery_dir)
        torch.cuda.empty_cache()
        logits, labels = _evaluate(model, dev_loader, device)
        # Model selection uses verification-needed F1: that is the operational
        # triage question the runtime actually routes on. Contradiction F1 is
        # still reported every epoch so the trade-off stays visible instead of
        # being optimised away silently.
        verification_metrics = verification_needed_metrics(logits, labels)
        contradiction_view = contradiction_metrics(logits, labels)
        metrics = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "task_b_verification_needed_f1": verification_metrics["f1"],
            "task_a_contradiction_f1": contradiction_view["f1"],
            "macro_f1": three_class_metrics(logits, labels)["macro_f1"],
        }
        history.append(metrics)
        selection_score = metrics["macro_f1"] if selection_metric == "macro_f1" else verification_metrics["f1"]
        if selection_score > best_f1:
            best_f1 = selection_score
            model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)
            np.savez_compressed(output_dir / "dev_predictions.npz", logits=logits, labels=labels)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    saved = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    calibration_loader = DataLoader(
        _ListDataset(calibration_rows), batch_size=batch_size, shuffle=False, collate_fn=collate,
    )
    logits, labels = _evaluate(saved, calibration_loader, device)
    temperature = fit_temperature(logits, labels)
    # One threshold per question, both selected on dev only.
    contradiction_threshold, contradiction_f1 = best_threshold(
        logits, labels, temperature, "contradiction"
    )
    verification_risk_threshold, verification_f1 = best_threshold(
        logits, labels, temperature, "verification_needed"
    )
    save_calibration(
        output_dir / "calibration.json",
        temperature,
        contradiction_threshold,
        verification_risk_threshold,
        max_length,
    )
    report = {
        "device": str(device),
        "base_model": base_model,
        "max_length": int(max_length),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "label_map": ID_TO_LABEL,
        "selection_metric": selection_metric,
        "calibration_split": "independent" if calibration_data is not None else "dev",
        "calibration_sample_count": len(calibration_rows),
        "gradient_checkpointing": gradient_checkpointing,
        "temperature": temperature,
        "thresholds": {
            "contradiction": {
                "value": contradiction_threshold,
                "selected_on": "calibration" if calibration_data is not None else "dev",
                "dev_f1": contradiction_f1,
                "optimises": "Task A: P(CONTRADICTED) vs CONTRADICTED",
            },
            "verification_risk": {
                "value": verification_risk_threshold,
                "selected_on": "calibration" if calibration_data is not None else "dev",
                "dev_f1": verification_f1,
                "optimises": "Task B: P(CONTRADICTED)+P(NOT_ENOUGH_INFO) vs non-SUPPORTED",
            },
            DEPRECATED_HALLUCINATION_THRESHOLD: {
                "value": verification_risk_threshold,
                "status": "deprecated mirror of verification_risk_threshold",
            },
        },
        "calibration_evaluation" if calibration_data is not None else "dev_evaluation": evaluate_saved_predictions(
            logits,
            labels,
            temperature,
            contradiction_threshold,
            verification_risk_threshold,
        ),
        "history": history,
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate(
    data_dir: Path,
    model_dir: Path,
    batch_size: int = 16,
    max_length: int = DEFAULT_MAX_LENGTH,
    split: str = "test",
    save: bool = True,
) -> dict:
    """Score a checkpoint on one split of a prepared data directory.

    ``split`` selects the JSONL file. The evidence-alignment study adds a
    class-capped ``metric`` split to the same directories, and both go through
    the identical code path -- a different split must not mean different
    encoding.
    """
    model_dir = Path(model_dir)
    calibration = load_calibration(model_dir / "calibration.json")
    saved_length = int(calibration.get("max_length", DEFAULT_MAX_LENGTH))
    if max_length != saved_length:
        raise ValueError(f"evaluation max_length {max_length} differs from checkpoint {saved_length}")
    if save and any((model_dir / f"{split}_{suffix}").exists() for suffix in ("metrics.json", "predictions.npz")):
        raise FileExistsError(f"evaluation artifacts already exist for split {split}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    rows = load_rows(Path(data_dir) / f"{split}.jsonl")
    logits, labels = predict_rows(
        model, tokenizer, rows, device=device, max_length=max_length, batch_size=batch_size
    )
    temperature = float(calibration.get("temperature", 1.0))
    contradiction_threshold = float(calibration.get("contradiction_threshold", 0.5))
    verification_risk_threshold = float(
        calibration.get("verification_risk_threshold", calibration.get(DEPRECATED_HALLUCINATION_THRESHOLD, 0.5))
    )
    metrics = evaluate_saved_predictions(
        logits,
        labels,
        temperature,
        contradiction_threshold,
        verification_risk_threshold,
    )
    metrics["split"] = split
    metrics["thresholds"] = {
        "contradiction": contradiction_threshold,
        "verification_risk": verification_risk_threshold,
    }
    if save:
        (Path(model_dir) / f"{split}_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        np.savez_compressed(Path(model_dir) / f"{split}_predictions.npz", logits=logits, labels=labels)
    return metrics


def train_arm(
    data_dir: Path,
    output_dir: Path,
    base_model: str = "microsoft/deberta-v3-xsmall",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = 42,
) -> dict:
    """Train one evidence-shape arm.

    A thin alias for :func:`train` that exists so the experiment runner can
    state, in one place, that every arm shares the architecture, optimizer,
    learning rate, epoch count, seed, label map and max sequence length. Only
    the ``data_dir`` -- i.e. the evidence the rows carry -- is allowed to vary.
    """
    return train(
        data_dir,
        output_dir,
        base_model=base_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        seed=seed,
    )

