import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .calibration import fit_temperature, save_calibration


ID_TO_LABEL = {0: "SUPPORTED", 1: "CONTRADICTED", 2: "NOT_ENOUGH_INFO"}


class JsonlDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = []
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{path.name} line {number}: empty JSONL row")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name} line {number}: malformed JSON") from exc
                if not isinstance(row, dict) or not isinstance(row.get("claim"), str) or not row["claim"].strip() or not isinstance(row.get("evidence"), str) or not row["evidence"].strip() or type(row.get("label_id")) is not int or row["label_id"] not in ID_TO_LABEL:
                    raise ValueError(f"{path.name} line {number}: malformed training example")
                if "label" in row and row["label"] != ID_TO_LABEL[row["label_id"]]:
                    raise ValueError(f"{path.name} line {number}: label and label_id disagree")
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _collator(tokenizer, max_length: int):
    def collate(rows):
        batch = tokenizer(
            [x["evidence"] for x in rows],
            [x["claim"] for x in rows],
            padding=True,
            truncation="longest_first",
            max_length=max_length,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor([x["label_id"] for x in rows], dtype=torch.long)
        return batch

    return collate


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


def binary_metrics(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0, threshold: float = 0.5):
    if logits.ndim != 2 or logits.shape[1] != 3 or labels.ndim != 1 or len(logits) != len(labels) or not len(labels):
        raise ValueError("expected nonempty [N,3] logits and [N] labels")
    if logits.dtype.kind not in "fi" or labels.dtype.kind not in "iu":
        raise ValueError("logits must be numeric and labels must be integer class IDs")
    if not np.isfinite(logits).all() or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("nonfinite logits or unsupported labels")
    if not np.isfinite(temperature) or temperature <= 0 or not 0 <= threshold <= 1:
        raise ValueError("invalid calibration temperature or threshold")
    probabilities = torch.softmax(torch.tensor(logits) / temperature, dim=-1).numpy()
    hallu_probability = 1.0 - probabilities[:, 0]
    truth = (labels != 0).astype(int)
    pred = (hallu_probability >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(truth, pred, average="binary", zero_division=0)
    accuracy = float((truth == pred).mean())
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (hallu_probability >= low) & (hallu_probability <= high if high == 1 else hallu_probability < high)
        if mask.any():
            ece += mask.mean() * abs(hallu_probability[mask].mean() - truth[mask].mean())
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(truth, hallu_probability)) if len(set(truth)) == 2 else None,
        "pr_auc": float(average_precision_score(truth, hallu_probability)) if len(set(truth)) == 2 else None,
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "false_negative_rate": float(fn / max(1, fn + tp)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "ece": float(ece),
        "brier": float(np.mean((hallu_probability - truth) ** 2)),
        "class_counts": {ID_TO_LABEL[i]: int((labels == i).sum()) for i in ID_TO_LABEL},
        "threshold": threshold,
    }


def train(
    data_dir: Path,
    output_dir: Path,
    base_model: str = "microsoft/deberta-v3-xsmall",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 384,
    seed: int = 42,
) -> dict:
    if epochs < 1 or batch_size < 1 or max_length < 1:
        raise ValueError("epochs, batch size, and maximum length must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("detector output directory is not empty; preserve existing artifacts")
    recovery_dir = output_dir.parent / f"{output_dir.name}-last"
    if recovery_dir.exists() and any(recovery_dir.iterdir()):
        raise FileExistsError("detector recovery directory is not empty; preserve existing artifacts")
    train_data = JsonlDataset(data_dir / "train.jsonl")
    dev_data = JsonlDataset(data_dir / "dev.jsonl")
    if not train_data or not dev_data:
        raise ValueError("training requires nonempty train and dev datasets")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id={v: k for k, v in ID_TO_LABEL.items()},
    ).float().to(device)
    collate = _collator(tokenizer, max_length)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, collate_fn=collate)
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
        model.save_pretrained(recovery_dir, safe_serialization=True)
        tokenizer.save_pretrained(recovery_dir)
        torch.cuda.empty_cache()
        logits, labels = _evaluate(model, dev_loader, device)
        metrics = binary_metrics(logits, labels)
        metrics.update({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})
        history.append(metrics)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)
            np.savez_compressed(output_dir / "dev_predictions.npz", logits=logits, labels=labels)
    saved = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    logits, labels = _evaluate(saved, dev_loader, device)
    temperature = fit_temperature(logits, labels)
    thresholds = np.linspace(0.25, 0.8, 112)
    scored = [(binary_metrics(logits, labels, temperature, float(t))["f1"], float(t)) for t in thresholds]
    threshold = max(scored)[1]
    save_calibration(output_dir / "calibration.json", temperature, threshold, max_length)
    report = {"device": str(device), "base_model": base_model, "temperature": temperature, "best_threshold": threshold, "history": history}
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate(data_dir: Path, model_dir: Path, batch_size: int = 16, max_length: int | None = None) -> dict:
    if (model_dir / "test_metrics.json").exists() or (model_dir / "test_predictions.npz").exists():
        raise FileExistsError("test evaluation artifacts already exist; preserve measured results")
    calibration = json.loads((model_dir / "calibration.json").read_text(encoding="utf-8"))
    effective_length = int(max_length if max_length is not None else calibration["max_length"])
    if batch_size < 1 or effective_length < 1:
        raise ValueError("batch size and maximum length must be positive")
    if effective_length != int(calibration["max_length"]):
        raise ValueError("evaluation maximum length must match saved calibration")
    test_data = JsonlDataset(data_dir / "test.jsonl")
    if not test_data:
        raise ValueError("evaluation requires a nonempty test dataset")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=_collator(tokenizer, effective_length))
    logits, labels = _evaluate(model, loader, device)
    metrics = binary_metrics(logits, labels, calibration["temperature"], calibration["hallucination_threshold"])
    metrics["samples"] = int(len(labels))
    (model_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(model_dir / "test_predictions.npz", logits=logits, labels=labels)
    return metrics
