"""
scripts / train_detector.py
───────────────────────────
Download REAL hallucination / fact-verification data from the internet and
fine-tune the M2 DeBERTa-v3-base claim-risk detector.

Data sources (pulled live from the HuggingFace Hub)
    HaluEval QA (pminervini/HaluEval, config "qa")
        question + right_answer      -> label 0 (SUPPORTED, low risk)
        question + hallucinated_answer -> label 1 (CONTRADICTED, high risk)
        Naturally balanced pairs; context = the question.
    FEVER gold evidence (copenlu/fever_gold_evidence)
        claim + SUPPORTS  -> label 0
        claim + REFUTES   -> label 1
        NOT ENOUGH INFO   -> dropped (not a binary risk signal)
        Real human-annotated factuality; context = the evidence.

Why a MIX (spec §9, and this project's memory):
    Training on HaluEval alone made the detector shortcut-learn "what a HaluEval
    hallucination looks like" and emit near-constant scores on real claims.
    Mixing a second, differently-constructed source (FEVER) breaks that single
    -distribution shortcut. We also report per-source eval so shortcutting is
    visible rather than hidden behind one aggregate number.

Model input format (must match models/deberta.py exactly — no train/serve skew)
    text_a = claim
    text_b = "[QUESTION] {context} [ANSWER] {answer_or_claim}"
    truncation = only_second, max_length = 384

Training notes (this project's memory: DeBERTa-v3 fp32-only gotchas)
    - fp16 -> gradient/overflow errors; bf16 -> stuck/broken runs. Use fp32.
    - lower LR (default 1e-5), load_best_model_at_end on eval AUROC-ish metric.

Usage
    python halluciguard_detector/scripts/train_detector.py \
        --max-per-source 4000 --epochs 3
    (add --smoke to do a tiny fast run that proves the pipeline end-to-end)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_detector")

CKPT_DIR = _ROOT / "data" / "checkpoints"
DATA_DIR = _ROOT / "data" / "processed"
MANIFEST_DIR = _ROOT / "data" / "manifests"
BASE_MODEL = "microsoft/deberta-v3-base"
MAX_LEN = 384


# ─────────────────────────────────────────────────────────────────────────────
# 1. Download + normalize real data to (claim, context, label, source) rows
# ─────────────────────────────────────────────────────────────────────────────
def _ctx(context: str, answer: str) -> str:
    """Build text_b in the SAME format the served detector uses."""
    return f"[QUESTION] {context} [ANSWER] {answer}"


def load_halueval_qa(max_rows: int) -> List[dict]:
    from datasets import load_dataset
    logger.info("Downloading HaluEval QA ...")
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    rows: List[dict] = []
    n = min(len(ds), max_rows) if max_rows else len(ds)
    for i in range(n):
        ex = ds[i]
        q = (ex.get("question") or "").strip()
        right = (ex.get("right_answer") or "").strip()
        wrong = (ex.get("hallucinated_answer") or "").strip()
        if q and right:
            rows.append({"claim": right, "context": _ctx(q, right), "label": 0, "source": "halueval_qa"})
        if q and wrong:
            rows.append({"claim": wrong, "context": _ctx(q, wrong), "label": 1, "source": "halueval_qa"})
    logger.info("  HaluEval QA rows: %d", len(rows))
    return rows


def load_fever(max_rows: int) -> List[dict]:
    from datasets import load_dataset
    logger.info("Downloading FEVER gold evidence ...")
    ds = load_dataset("copenlu/fever_gold_evidence", split="train")
    label_map = {"SUPPORTS": 0, "REFUTES": 1}  # drop NOT ENOUGH INFO
    rows: List[dict] = []
    count = 0
    for ex in ds:
        lbl = ex.get("label")
        if lbl not in label_map:
            continue
        claim = (ex.get("claim") or "").strip()
        ev = ex.get("evidence")
        if isinstance(ev, list):
            ev = " ".join(str(e) for e in ev)
        ev = (ev or "").strip()
        if not claim:
            continue
        rows.append({"claim": claim, "context": _ctx(ev[:1000], claim),
                     "label": label_map[lbl], "source": "fever"})
        count += 1
        if max_rows and count >= max_rows:
            break
    logger.info("  FEVER rows: %d", len(rows))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 2. Split (stratified by source+label, leakage-safe by row)
# ─────────────────────────────────────────────────────────────────────────────
def split_rows(rows: List[dict], seed: int, ratios=(0.8, 0.1, 0.1)):
    rng = random.Random(seed)
    buckets: Dict[Tuple[str, int], List[dict]] = {}
    for r in rows:
        buckets.setdefault((r["source"], r["label"]), []).append(r)
    train, val, test = [], [], []
    for _, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_tr, n_va = int(ratios[0] * n), int(ratios[1] * n)
        train += items[:n_tr]
        val += items[n_tr:n_tr + n_va]
        test += items[n_tr + n_va:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 3. Train
# ─────────────────────────────────────────────────────────────────────────────
def _metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    preds = (probs >= 0.5).astype(int)
    out = {"f1": float(f1_score(labels, preds, zero_division=0)),
           "precision": float(precision_score(labels, preds, zero_division=0)),
           "recall": float(recall_score(labels, preds, zero_division=0))}
    try:
        out["auroc"] = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan")
    except Exception:
        out["auroc"] = float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train M2 DeBERTa detector on real internet data")
    ap.add_argument("--max-per-source", type=int, default=4000, help="cap raw examples per source")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-5)          # low LR (fp32 stability)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(CKPT_DIR / "deberta-mixed-v1"))
    ap.add_argument("--sources", default="halueval,fever", help="comma list: halueval,fever")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to prove the pipeline")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    if args.smoke:
        args.max_per_source = 60
        args.epochs = 1.0

    # ---- 1. data
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    rows: List[dict] = []
    if "halueval" in sources:
        rows += load_halueval_qa(args.max_per_source)
    if "fever" in sources:
        rows += load_fever(args.max_per_source)
    if not rows:
        ap.error("no data loaded")

    train, val, test = split_rows(rows, args.seed)
    logger.info("Split: train=%d val=%d test=%d", len(train), len(val), len(test))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("REAL_train", train), ("REAL_val", val), ("REAL_test", test)]:
        json.dump(part, open(DATA_DIR / f"{name}.json", "w", encoding="utf-8"), ensure_ascii=False)

    # ---- 2. tokenize
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)
    from datasets import Dataset

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    def to_hf(part):
        return Dataset.from_dict({
            "claim": [r["claim"] for r in part],
            "context": [r["context"] for r in part],
            "label": [r["label"] for r in part],
            "source": [r["source"] for r in part],
        })

    def tokenize(batch):
        return tok(batch["claim"], batch["context"], truncation="only_second",
                   max_length=MAX_LEN, padding="max_length")

    ds_train = to_hf(train).map(tokenize, batched=True)
    ds_val = to_hf(val).map(tokenize, batched=True)
    ds_test = to_hf(test).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        m = _metrics(np.array(labels), probs)
        return m

    targs = TrainingArguments(
        output_dir=str(Path(args.out) / "_hf"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auroc",
        greater_is_better=True,
        fp16=False, bf16=False,          # fp32 only (project memory: fp16/bf16 broke)
        # CRITICAL: DeBERTa-v3 in fp32 diverges to NaN on the FIRST optimizer
        # step with the default Adam epsilon (1e-8) — the denominator is too
        # small and the update explodes. eps=1e-6 keeps it finite; grad
        # clipping is a secondary safety belt. (Diagnosed empirically:
        # step0 loss=0.75 healthy, step1 -> NaN with eps=1e-8; stable with 1e-6.)
        adam_epsilon=1e-6,
        max_grad_norm=1.0,
        logging_steps=25,
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds_train,
                      eval_dataset=ds_val, compute_metrics=compute_metrics)

    logger.info("Training (fp32, lr=%.1e, epochs=%.1f) ...", args.lr, args.epochs)
    trainer.train()

    # ---- 3. save served checkpoint
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    logger.info("Saved checkpoint -> %s", out)

    # ---- 4. evaluate: overall + PER SOURCE (exposes shortcut learning)
    def eval_split(ds, tag) -> dict:
        pred = trainer.predict(ds)
        probs = torch.softmax(torch.tensor(pred.predictions), dim=-1)[:, 1].numpy()
        labels = np.array(ds["label"])
        srcs = np.array(ds["source"])
        report = {"overall": _metrics(labels, probs)}
        for s in sorted(set(srcs.tolist())):
            mask = srcs == s
            if mask.sum() > 0:
                report[s] = _metrics(labels[mask], probs[mask])
        logger.info("[%s] %s", tag, json.dumps(report, indent=2))
        return report

    results = {
        "base_model": BASE_MODEL,
        "sources": sources,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "hyperparams": {"lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
                        "precision": "fp32", "max_len": MAX_LEN, "seed": args.seed},
        "val": eval_split(ds_val, "VAL"),
        "test": eval_split(ds_test, "TEST"),
        "checkpoint": str(out),
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(MANIFEST_DIR / "train_results.json", "w", encoding="utf-8"), indent=2)

    logger.info("=" * 70)
    logger.info("  TRAINING COMPLETE -> %s", out)
    logger.info("  TEST overall AUROC=%.4f F1=%.4f",
                results["test"]["overall"]["auroc"], results["test"]["overall"]["f1"])
    for s in sources:
        if s == "halueval" and "halueval_qa" in results["test"]:
            r = results["test"]["halueval_qa"]
        elif s in results["test"]:
            r = results["test"][s]
        else:
            continue
        logger.info("  TEST[%s] AUROC=%.4f F1=%.4f", s, r["auroc"], r["f1"])
    logger.info("  full report -> %s", MANIFEST_DIR / "train_results.json")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
