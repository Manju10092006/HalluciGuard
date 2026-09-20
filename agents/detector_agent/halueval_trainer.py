"""
HaluEval Fine-Tuning Trainer — v2 (root-cause fix build).

Key changes:
  - Default output dir: artifacts/halueval-detector-final/
  - Class-weighted loss via custom Trainer subclass.
  - Verbose GPU/VRAM info before training.
  - general_upsample_factor passed through.
  - NEVER overwrites artifacts/halueval-detector/ (legacy).

Usage:
    # Smoke test
    python -m agents.detector_agent.halueval_trainer --max-rows 500

    # Full training
    python -m agents.detector_agent.halueval_trainer --epochs 3 --batch-size 16

Environment variables (optional):
    HALUEVAL_BASE_MODEL       - HF model name (default: distilbert-base-uncased)
    HALUEVAL_OUTPUT_DIR       - Output dir (default: artifacts/halueval-detector-final)
    HALUEVAL_MAX_LENGTH       - Max token length (default: 384)
    HALUEVAL_EPOCHS           - Epochs (default: 3)
    HALUEVAL_LR               - LR (default: 2e-5)
    HALUEVAL_TRAIN_BATCH_SIZE - Train batch (default: 16)
    HALUEVAL_EVAL_BATCH_SIZE  - Eval batch (default: 32)
    HALUEVAL_SEED             - Seed (default: 42)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import DatasetDict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.detector_agent.halueval_dataset import HaluEvalConfig, load_halueval


# ============================================================
# WEIGHTED TRAINER — handles class imbalance in general config
# ============================================================

class WeightedTrainer(Trainer):
    """Trainer subclass that applies class-frequency inverse weights to the loss."""

    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ============================================================
# HELPERS
# ============================================================

def get_config_from_env() -> dict:
    return {
        "base_model":       os.environ.get("HALUEVAL_BASE_MODEL", "distilbert-base-uncased"),
        "output_dir":       os.environ.get("HALUEVAL_OUTPUT_DIR", "artifacts/halueval-detector-final"),
        "max_length":       int(os.environ.get("HALUEVAL_MAX_LENGTH", "384")),
        "epochs":           int(os.environ.get("HALUEVAL_EPOCHS", "3")),
        "lr":               float(os.environ.get("HALUEVAL_LR", "2e-5")),
        "train_batch_size": int(os.environ.get("HALUEVAL_TRAIN_BATCH_SIZE", "16")),
        "eval_batch_size":  int(os.environ.get("HALUEVAL_EVAL_BATCH_SIZE", "32")),
        "seed":             int(os.environ.get("HALUEVAL_SEED", "42")),
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall":    recall_score(labels, predictions, zero_division=0),
        "f1":        f1_score(labels, predictions, zero_division=0),
    }


def print_gpu_info():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024 ** 3
        print(f"  GPU Name:    {props.name}")
        print(f"  CUDA:        {torch.version.cuda}")
        print(f"  VRAM:        {vram_gb:.1f} GB")
        print(f"  Compute Cap: {props.major}.{props.minor}")
    else:
        print("  GPU: NONE (CPU mode)")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on HaluEval (v2)")
    parser.add_argument("--max-rows",   type=int,   default=None, help="Max rows/config for smoke test")
    parser.add_argument("--base-model", type=str,   default=None)
    parser.add_argument("--output-dir", type=str,   default=None)
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--general-upsample", type=int, default=5,
                        help="Upsample factor for general config (default 5)")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable class-weighted loss")
    args = parser.parse_args()

    cfg = get_config_from_env()
    if args.base_model:  cfg["base_model"] = args.base_model
    if args.output_dir:  cfg["output_dir"] = args.output_dir
    if args.epochs:      cfg["epochs"]     = args.epochs
    if args.lr:          cfg["lr"]         = args.lr
    if args.batch_size:  cfg["train_batch_size"] = args.batch_size

    # SAFETY: refuse to overwrite the legacy model directory
    output_dir = os.path.abspath(cfg["output_dir"])
    legacy_dir = os.path.abspath("artifacts/halueval-detector")
    if output_dir == legacy_dir:
        print("ERROR: Output directory matches legacy model path!")
        print("       Use --output-dir artifacts/halueval-detector-final")
        sys.exit(1)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = torch.cuda.is_available()

    print("=" * 70)
    print("HALLUCIGUARD — HALUEVAL DETECTOR TRAINING v2 (root-cause fix)")
    print("=" * 70)
    print_gpu_info()
    print(f"  FP16:           {use_fp16}")
    print(f"  Base Model:     {cfg['base_model']}")
    print(f"  Output Dir:     {output_dir}")
    print(f"  Max Length:     {cfg['max_length']}")
    print(f"  Epochs:         {cfg['epochs']}")
    print(f"  Learning Rate:  {cfg['lr']}")
    print(f"  Train Batch:    {cfg['train_batch_size']}")
    print(f"  Eval Batch:     {cfg['eval_batch_size']}")
    print(f"  Seed:           {cfg['seed']}")
    print(f"  Max Rows/Config:{args.max_rows or 'ALL'}")
    print(f"  General Upsample: {args.general_upsample}x")
    print(f"  Class Weights:  {not args.no_class_weights}")
    print("=" * 70)

    # ---------------------------------------------------------------- Step 1
    print("\n[Step 1] Loading HaluEval dataset (fixed v2)...")
    halueval_cfg = HaluEvalConfig(
        max_rows_per_config=args.max_rows,
        seed=cfg["seed"],
        general_upsample_factor=args.general_upsample,
    )
    splits = load_halueval(halueval_cfg)

    # ---------------------------------------------------------------- Step 2
    print(f"\n[Step 2] Loading tokenizer: {cfg['base_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg["max_length"],
            padding=False,
        )

    print("[Step 2] Tokenizing...")
    tokenized = splits.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    print(f"[Step 2] Done. Train={len(tokenized['train'])} Val={len(tokenized['validation'])} Test={len(tokenized['test'])}")

    # ---------------------------------------------------------------- Step 3
    print(f"\n[Step 3] Loading model: {cfg['base_model']}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_model"],
        num_labels=2,
        id2label={0: "NO_HALLUCINATION", 1: "HALLUCINATION"},
        label2id={"NO_HALLUCINATION": 0, "HALLUCINATION": 1},
    )
    print(f"[Step 3] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compute class weights from training set
    # Use list comprehension — works regardless of underlying list vs tensor type
    train_labels_list = tokenized["train"]["label"]
    n0 = sum(1 for l in train_labels_list if int(l) == 0)
    n1 = sum(1 for l in train_labels_list if int(l) == 1)
    n_total = n0 + n1
    if not args.no_class_weights and n0 > 0 and n1 > 0:
        w0 = n_total / (2.0 * n0)
        w1 = n_total / (2.0 * n1)
        class_weights = torch.tensor([w0, w1], dtype=torch.float)
        print(f"[Step 3] Class weights: w0={w0:.4f}  w1={w1:.4f}  (n0={n0}, n1={n1})")
    else:
        class_weights = None
        print(f"[Step 3] No class weights applied. (n0={n0}, n1={n1})")

    # ---------------------------------------------------------------- Step 4
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        num_train_epochs=cfg["epochs"],
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=cfg["seed"],
        logging_steps=100,
        report_to="none",
        fp16=use_fp16,
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print(f"\n[Step 4] Training ({cfg['epochs']} epochs) on {device}...")
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time
    print(f"[Step 4] Done in {elapsed:.1f}s. Loss: {train_result.training_loss:.4f}")

    # ---------------------------------------------------------------- Step 5
    print("\n[Step 5] Evaluating on test set...")
    test_results = trainer.evaluate(tokenized["test"])
    print("[Step 5] Test Results:")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ---------------------------------------------------------------- Step 6
    print("\n[Step 6] Classification report...")
    predictions = trainer.predict(tokenized["test"])
    preds  = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    print(classification_report(labels, preds,
                                target_names=["NO_HALLUCINATION", "HALLUCINATION"],
                                digits=4))
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    # ---------------------------------------------------------------- Step 7
    print(f"\n[Step 7] Saving to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model":          cfg["base_model"],
        "dataset":             "pminervini/HaluEval",
        "version":             "v2-root-cause-fix",
        "dataset_changes":     [
            "Removed ungrounded QA/dialogue duplicates",
            f"General config upsampled {args.general_upsample}x",
            "Sample-level (source_id) group splitting",
            "Canonical format_detector_input() formatter",
            "Class-weighted loss",
        ],
        "max_rows_per_config": args.max_rows,
        "train_examples":      len(tokenized["train"]),
        "val_examples":        len(tokenized["validation"]),
        "test_examples":       len(tokenized["test"]),
        "class_n0_train":      n0,
        "class_n1_train":      n1,
        "class_w0":            float(class_weights[0]) if class_weights is not None else 1.0,
        "class_w1":            float(class_weights[1]) if class_weights is not None else 1.0,
        "epochs":              cfg["epochs"],
        "learning_rate":       cfg["lr"],
        "max_length":          cfg["max_length"],
        "seed":                cfg["seed"],
        "device":              device,
        "training_loss":       float(train_result.training_loss),
        "training_time_seconds": round(elapsed, 1),
        "test_accuracy":       float(test_results.get("eval_accuracy", 0)),
        "test_precision":      float(test_results.get("eval_precision", 0)),
        "test_recall":         float(test_results.get("eval_recall", 0)),
        "test_f1":             float(test_results.get("eval_f1", 0)),
        "confusion_matrix":    cm.tolist(),
    }
    with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("[Step 7] Metadata saved.")

    # ---------------------------------------------------------------- Step 8
    print("\n[Step 8] Verifying model reload from disk...")
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reload_tok = AutoTokenizer.from_pretrained(output_dir)
    reload_mod = AutoModelForSequenceClassification.from_pretrained(output_dir)
    reload_mod.eval()

    # Verify label mapping
    assert reload_mod.config.id2label[0] == "NO_HALLUCINATION"
    assert reload_mod.config.id2label[1] == "HALLUCINATION"
    print(f"  Labels: {reload_mod.config.id2label}")

    from agents.detector_agent.halueval_dataset import format_detector_input
    test_text = format_detector_input("What is the capital of France?",
                                      "The capital of France is Paris.")
    inputs = reload_tok(test_text, return_tensors="pt", truncation=True, max_length=384)
    with torch.no_grad():
        probs = torch.softmax(reload_mod(**inputs).logits, dim=-1)
    print(f"  Paris test: P(no_halluc)={probs[0][0]:.4f}  P(halluc)={probs[0][1]:.4f}")
    print(f"  Loaded from: {os.path.abspath(output_dir)}")
    print("[Step 8] RELOAD PASSED")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model:      {os.path.abspath(output_dir)}")
    print(f"  Train rows: {len(tokenized['train'])}")
    print(f"  Accuracy:   {test_results.get('eval_accuracy', 0):.4f}")
    print(f"  F1:         {test_results.get('eval_f1', 0):.4f}")
    print(f"  Time:       {elapsed:.1f}s")
    print("=" * 70)

    return metadata


if __name__ == "__main__":
    main()
