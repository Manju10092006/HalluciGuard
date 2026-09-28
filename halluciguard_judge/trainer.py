"""
halluciguard_judge / trainer.py
────────────────────────────────
Fine-tuning script for DeBERTa-v3-base hallucination classifier.

Training data sources (as recommended in the spec):
    1. HaluBench  — partially synthetic, multi-domain QA
    2. RAGTruth   — human-labelled, naturally generated LLM responses (~18k)
    3. HaluEval   — existing project dataset (QA, dialogue, summarisation)
    4. Custom HalluciGuard data  — production-style (added incrementally)

Input format (per example):
    text:  "Query: {user_query}\nClaim: {claim_text}"
    label: 0 = NOT_HALLUCINATION, 1 = HALLUCINATION

Training strategy:
    - DeBERTa-v3-base (microsoft/deberta-v3-base)
    - Binary cross-entropy loss
    - Class-weighted loss to handle imbalance
    - Early stopping on validation F1
    - Checkpoint saved at best validation F1
    - Temperature calibration on held-out calibration set

Usage:
    python -m halluciguard_judge.trainer \
        --data_dir halluciguard_judge/datasets/training \
        --output_dir halluciguard_judge/checkpoints/deberta-v3-hallucination \
        --epochs 5 \
        --batch_size 16 \
        --max_length 512
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

LABEL_MAP = {0: "NOT_HALLUCINATION", 1: "HALLUCINATION"}
BASE_MODEL = "microsoft/deberta-v3-base"


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class HallucinationDataset(Dataset):
    """PyTorch Dataset for hallucination binary classification.

    Each sample is a dict:
        {
            "query": "Who created Java?",
            "claim": "Java was created by Dennis Ritchie.",
            "label": 1   # 1 = hallucination
        }

    Or a flat format with "text" and "label" (pre-formatted).
    """

    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Support both formats
        if "text" in sample:
            text = sample["text"]
        else:
            query = sample.get("query", sample.get("user_query", ""))
            claim = sample.get("claim", sample.get("claim_text", sample.get("response", "")))
            text = f"Query: {query}\nClaim: {claim}"

        label = int(sample["label"])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros(1)).squeeze(0)
            if "token_type_ids" in encoding else torch.zeros(self.max_length, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Data Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_halueval_samples(filepath: str) -> List[Dict]:
    """Load HaluEval-format samples (QA task).

    Expected format (from HaluEval GitHub):
        [{"question": "...", "answer": "...", "hallucination": "yes|no"}, ...]
    """
    samples = []
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            label = 1 if str(item.get("hallucination", "no")).lower() == "yes" else 0
            samples.append({
                "query": item.get("question", item.get("user_query", "")),
                "claim": item.get("answer", item.get("response", "")),
                "label": label,
                "source": "halueval",
            })
    except Exception as e:
        logger.error("[Trainer] Failed to load HaluEval: %s", e)
    logger.info("[Trainer] Loaded %d samples from HaluEval: %s", len(samples), filepath)
    return samples


def load_ragtruth_samples(filepath: str) -> List[Dict]:
    """Load RAGTruth-format samples.

    Expected format (from ParticleMedia/RAGTruth):
        [{"query": "...", "response": "...", "labels": [{"label": "...", "start": ..., "end": ...}]}]

    A response is hallucinated if it has any non-CORRECT label.
    """
    samples = []
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            response = item.get("response", item.get("answer", ""))
            query = item.get("query", item.get("question", ""))
            labels = item.get("labels", [])
            # If any label is not "correct" or "None" -> hallucination
            if labels:
                label = 1 if any(
                    str(l.get("label", "correct")).lower() not in ("correct", "none", "")
                    for l in labels
                ) else 0
            else:
                label = int(item.get("label", item.get("hallucination", 0)))
            samples.append({
                "query": query,
                "claim": response,
                "label": label,
                "source": "ragtruth",
            })
    except Exception as e:
        logger.error("[Trainer] Failed to load RAGTruth: %s", e)
    logger.info("[Trainer] Loaded %d samples from RAGTruth: %s", len(samples), filepath)
    return samples


def load_custom_samples(filepath: str) -> List[Dict]:
    """Load custom HalluciGuard production-style samples.

    Format: [{"query": "...", "claim": "...", "label": 0|1, "source": "..."}]
    """
    samples = []
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            samples.append({
                "query": item.get("query", item.get("user_query", "")),
                "claim": item.get("claim", item.get("response", "")),
                "label": int(item.get("label", 0)),
                "source": item.get("source", "custom"),
            })
    except Exception as e:
        logger.error("[Trainer] Failed to load custom data: %s", e)
    logger.info("[Trainer] Loaded %d custom samples from %s", len(samples), filepath)
    return samples


def compute_class_weights(samples: List[Dict]) -> Tuple[float, float]:
    """Compute class weights to handle label imbalance."""
    total = len(samples)
    n_pos = sum(1 for s in samples if s["label"] == 1)
    n_neg = total - n_pos
    if n_pos == 0 or n_neg == 0:
        return 1.0, 1.0
    # Inverse frequency weighting
    w_neg = total / (2 * n_neg)
    w_pos = total / (2 * n_pos)
    return w_neg, w_pos


# ──────────────────────────────────────────────────────────────────────────────
# Calibration (Temperature Scaling)
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_temperature(
    model,
    tokenizer,
    val_samples: List[Dict],
    max_length: int = 512,
    device: str = "cpu",
) -> float:
    """Find optimal temperature for probability calibration.

    Minimises Expected Calibration Error (ECE) on the validation set.
    Returns the optimal temperature (applied as logits / temperature).
    """
    try:
        import numpy as np
        from scipy.optimize import minimize_scalar
    except ImportError:
        logger.warning("[Trainer] scipy not available — skipping calibration, using T=1.0.")
        return 1.0

    val_dataset = HallucinationDataset(val_samples, tokenizer, max_length)
    from torch.utils.data import DataLoader
    loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    all_logits = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    all_logits_np = np.concatenate(all_logits, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)

    def ece_with_temperature(T: float) -> float:
        scaled = all_logits_np / T
        exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
        pos_probs = probs[:, 1]

        # ECE: 15 bins
        n_bins = 15
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (pos_probs >= bin_edges[i]) & (pos_probs < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            acc = all_labels_np[mask].mean()
            conf = pos_probs[mask].mean()
            ece += mask.sum() * abs(acc - conf)
        return float(ece / len(all_labels_np))

    result = minimize_scalar(ece_with_temperature, bounds=(0.1, 5.0), method="bounded")
    optimal_T = float(result.x)
    logger.info("[Trainer] Optimal calibration temperature: %.4f (ECE=%.4f)", optimal_T, result.fun)
    return optimal_T


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ──────────────────────────────────────────────────────────────────────────────

def train(
    data_dir: str,
    output_dir: str,
    base_model: str = BASE_MODEL,
    epochs: int = 5,
    batch_size: int = 16,
    max_length: int = 512,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    patience: int = 3,
    seed: int = 42,
) -> None:
    """Full training pipeline for the DeBERTa hallucination classifier.

    Steps:
        1. Load training data from all sources
        2. Compute class weights (handle imbalance)
        3. Fine-tune DeBERTa-v3-base
        4. Early stopping on validation F1
        5. Temperature calibration on cal set
        6. Save checkpoint + config

    Args:
        data_dir:       Directory containing training JSON files.
        output_dir:     Where to save the fine-tuned checkpoint.
        base_model:     HuggingFace model ID to fine-tune from.
        epochs:         Max training epochs.
        batch_size:     Per-device training batch size.
        max_length:     Max tokenisation length.
        learning_rate:  AdamW learning rate.
        warmup_ratio:   Fraction of steps for LR warmup.
        weight_decay:   L2 regularisation.
        patience:       Early stopping patience (epochs without F1 improvement).
        seed:           Random seed.
    """
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            EarlyStoppingCallback,
        )
        import numpy as np
        from sklearn.metrics import f1_score, precision_score, recall_score
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        logger.error("[Trainer] Missing dependency: %s", e)
        raise

    torch.manual_seed(seed)
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Load all data sources ─────────────────────────────────────────────────
    all_samples: List[Dict] = []

    for fname in data_path.glob("halueval*.json"):
        all_samples.extend(load_halueval_samples(str(fname)))
    for fname in data_path.glob("ragtruth*.json"):
        all_samples.extend(load_ragtruth_samples(str(fname)))
    for fname in data_path.glob("custom*.json"):
        all_samples.extend(load_custom_samples(str(fname)))
    # Generic format fallback
    for fname in data_path.glob("train*.json"):
        all_samples.extend(load_custom_samples(str(fname)))

    if not all_samples:
        raise ValueError(
            f"No training data found in {data_dir}. "
            "Expected files matching halueval*.json, ragtruth*.json, or custom*.json"
        )

    logger.info("[Trainer] Total samples loaded: %d", len(all_samples))
    pos = sum(1 for s in all_samples if s["label"] == 1)
    neg = len(all_samples) - pos
    logger.info("[Trainer] Label distribution: HAL=%d NOT_HAL=%d (%.1f%%)", pos, neg, 100*pos/len(all_samples))

    # ── Split: train / val / cal ──────────────────────────────────────────────
    train_val, cal_samples = train_test_split(all_samples, test_size=0.1, random_state=seed, stratify=[s["label"] for s in all_samples])
    train_samples, val_samples = train_test_split(train_val, test_size=0.111, random_state=seed, stratify=[s["label"] for s in train_val])
    # -> roughly 80/10/10

    logger.info("[Trainer] Split: train=%d val=%d cal=%d", len(train_samples), len(val_samples), len(cal_samples))

    # ── Load tokenizer + model ────────────────────────────────────────────────
    logger.info("[Trainer] Loading base model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)

    # ── Class weights ─────────────────────────────────────────────────────────
    w_neg, w_pos = compute_class_weights(train_samples)
    class_weights = torch.tensor([w_neg, w_pos])
    logger.info("[Trainer] Class weights: NOT_HAL=%.3f HAL=%.3f", w_neg, w_pos)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = HallucinationDataset(train_samples, tokenizer, max_length)
    val_dataset   = HallucinationDataset(val_samples,   tokenizer, max_length)

    # ── Metrics ───────────────────────────────────────────────────────────────
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        f1  = f1_score(labels, preds, zero_division=0)
        prec = precision_score(labels, preds, zero_division=0)
        rec  = recall_score(labels, preds, zero_division=0)
        acc  = (preds == labels).mean()
        return {"f1": f1, "precision": prec, "recall": rec, "accuracy": acc}

    # ── Training arguments ────────────────────────────────────────────────────
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    training_args = TrainingArguments(
        output_dir=str(out_path / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        seed=seed,
        dataloader_num_workers=0,  # Windows compatibility
        report_to="none",
        fp16=(device_str == "cuda"),
    )

    # Custom trainer with class-weighted loss
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weights = class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("[Trainer] Starting training...")
    trainer.train()

    # ── Calibration ───────────────────────────────────────────────────────────
    logger.info("[Trainer] Running temperature calibration on cal set...")
    optimal_temp = calibrate_temperature(
        model=model,
        tokenizer=tokenizer,
        val_samples=cal_samples,
        max_length=max_length,
        device=device_str,
    )

    # ── Save final model + calibration config ─────────────────────────────────
    final_dir = out_path
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    calib_config = {
        "temperature": optimal_temp,
        "base_model": base_model,
        "max_length": max_length,
        "label_map": LABEL_MAP,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
    }
    with open(final_dir / "calibration_config.json", "w") as f:
        json.dump(calib_config, f, indent=2)

    logger.info("[Trainer] Saved fine-tuned model to: %s", final_dir)
    logger.info("[Trainer] Temperature: %.4f", optimal_temp)
    logger.info("[Trainer] Training complete.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train DeBERTa-v3 hallucination classifier")
    parser.add_argument("--data_dir",   required=True, help="Directory with training JSON files")
    parser.add_argument("--output_dir", required=True, help="Where to save the fine-tuned model")
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--max_length", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--patience",   type=int,   default=3)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
    )
