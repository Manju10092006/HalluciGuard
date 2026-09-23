"""
scripts / train_deberta.py
──────────────────────────
Fine-tune DeBERTa-v3-base as the M2 claim-risk classifier (v0.1 diagnostic).

Input format MUST match inference (halluciguard_detector/models/deberta.py):
    text_a = claim
    text_b = "[QUESTION] {query} [ANSWER] {answer}"
    truncation = only_second, max_length = 384
    label 1 = high verification risk, label 0 = low risk
    risk probability = softmax(logits)[:, 1]

Sized for a 6GB GPU: batch 8 + grad-accum 4 (effective 32), fp16, max_len 320.

Usage:
    python halluciguard_detector/scripts/train_deberta.py --epochs 2
Outputs a checkpoint dir the detector/live-runner can point DETECTOR_CHECKPOINT at.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import roc_auc_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

BASE = "microsoft/deberta-v3-base"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
DEFAULT_OUT = Path(__file__).parent.parent / "data" / "checkpoints" / "deberta-halueval-v0.1"


def _fmt_context(query: str, answer: str) -> str:
    return f"[QUESTION] {query} [ANSWER] {answer}"


def load_split(name: str):
    with open(DATA_DIR / f"HALUEVAL_{name}.json", encoding="utf-8") as f:
        rows = json.load(f)
    return rows


def make_dataset(rows, tokenizer, max_length: int):
    claims = [r["claim"] for r in rows]
    contexts = [_fmt_context(r["query"], r["answer"]) for r in rows]
    labels = [int(r["label"]) for r in rows]
    enc = tokenizer(
        claims, contexts,
        truncation="only_second", max_length=max_length, padding=False,
    )
    enc["labels"] = labels
    return Dataset.from_dict(enc)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)
    out = {"f1": f1_score(labels, preds, zero_division=0)}
    if len(np.unique(labels)) > 1:
        out["auroc"] = roc_auc_score(labels, probs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3.0)
    # DeBERTa-v3 needs a low LR; higher LRs diverge to NaN with disentangled attn.
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=320)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    # DeBERTa-v3 must train in fp32:
    #  - fp16 -> "Attempting to unscale FP16 gradients"
    #  - bf16 -> forward pass numerically corrupts, model stays at random (AUROC~0.5)
    # fp32 is stable given a low LR + Adam eps 1e-6 + grad clipping (set below).
    on_cuda = torch.cuda.is_available()
    use_bf16 = False
    use_fp16 = False
    print(f"[train] device={'cuda' if on_cuda else 'cpu'} precision=fp32 (DeBERTa-v3 requirement)")

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2)

    train_rows, dev_rows = load_split("train"), load_split("dev")
    print(f"[train] train={len(train_rows)} dev={len(dev_rows)}")
    train_ds = make_dataset(train_rows, tokenizer, args.max_length)
    dev_ds = make_dataset(dev_rows, tokenizer, args.max_length)

    targs = TrainingArguments(
        output_dir=str(Path(args.out) / "_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.10,
        weight_decay=0.01,
        adam_epsilon=1e-6,      # 1e-8 default destabilizes DeBERTa-v3 -> NaN
        max_grad_norm=1.0,      # clip; forward can spike early
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"[train] final dev metrics: {metrics}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    with open(out / "dev_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"[train] saved checkpoint -> {out}")
    print(f"[train] point the detector at it:  set DETECTOR_CHECKPOINT={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
