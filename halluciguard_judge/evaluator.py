"""
halluciguard_judge / evaluator.py
──────────────────────────────────
Benchmark evaluator: compare new JudgeDetector vs existing DistilBERT detector.

Metrics computed:
    - Accuracy, Precision, Recall, F1
    - False Positive Rate (FPR), False Negative Rate (FNR)
    - ECE (Expected Calibration Error)
    - Average latency per sample
    - Comparison table: new vs old

Usage:
    python -m halluciguard_judge.evaluator \
        --dataset halluciguard_judge/datasets/sample_halueval.json \
        --dataset_type halueval \
        --output results.json

Success criterion (from spec):
    Can this new standalone detector correctly distinguish true vs hallucinated
    responses on a production-style test set WITHOUT behaving like the current
    ~0.9 constant classifier?

    Target: F1 >= 0.78 on HaluEval test set.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _load_dataset(filepath: str, dataset_type: str) -> List[Dict]:
    """Load evaluation dataset in standard format.

    Returns list of: {"query": str, "claim": str, "label": int}
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    if dataset_type == "halueval":
        for item in data:
            label = 1 if str(item.get("hallucination", "no")).lower() == "yes" else 0
            samples.append({
                "query": item.get("question", item.get("user_query", "")),
                "claim": item.get("answer", item.get("response", "")),
                "label": label,
            })
    elif dataset_type == "ragtruth":
        for item in data:
            labels = item.get("labels", [])
            label = 1 if any(
                str(l.get("label", "correct")).lower() not in ("correct", "none", "")
                for l in labels
            ) else int(item.get("label", 0))
            samples.append({
                "query": item.get("query", ""),
                "claim": item.get("response", ""),
                "label": label,
            })
    else:  # custom / generic
        for item in data:
            samples.append({
                "query": item.get("query", item.get("user_query", "")),
                "claim": item.get("claim", item.get("response", "")),
                "label": int(item.get("label", 0)),
            })

    logger.info("[Evaluator] Loaded %d samples from %s (%s)", len(samples), filepath, dataset_type)
    return samples


def _compute_metrics(y_true: List[int], y_pred: List[int], probs: List[float]) -> Dict:
    """Compute classification and calibration metrics."""
    import numpy as np

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    probs_np  = np.array(probs)

    tp = int(((y_pred_np == 1) & (y_true_np == 1)).sum())
    tn = int(((y_pred_np == 0) & (y_true_np == 0)).sum())
    fp = int(((y_pred_np == 1) & (y_true_np == 0)).sum())
    fn = int(((y_pred_np == 0) & (y_true_np == 1)).sum())

    total = len(y_true)
    accuracy  = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # ECE (15 bins)
    n_bins = 15
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs_np >= bin_edges[i]) & (probs_np < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        acc_bin  = y_true_np[mask].mean()
        conf_bin = probs_np[mask].mean()
        ece += mask.sum() * abs(acc_bin - conf_bin)
    ece /= total

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "fpr":       round(fpr,       4),
        "fnr":       round(fnr,       4),
        "ece":       round(ece,       4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total,
    }


def evaluate_new_detector(
    samples: List[Dict],
    threshold: float = 0.5,
) -> Tuple[Dict, float]:
    """Evaluate the new JudgeDetector.

    Returns: (metrics_dict, avg_latency_seconds)
    """
    from halluciguard_judge.detector import JudgeDetector

    detector = JudgeDetector()

    y_true, y_pred, probs = [], [], []
    latencies = []

    for sample in samples:
        t0 = time.time()
        result = detector.detect(
            user_query=sample["query"],
            llm_response=sample["claim"],
        )
        latency = time.time() - t0
        latencies.append(latency)

        prob = result.overall_hallucination_probability
        pred = 1 if prob >= threshold else 0

        y_true.append(sample["label"])
        y_pred.append(pred)
        probs.append(prob)

    metrics = _compute_metrics(y_true, y_pred, probs)
    metrics["avg_latency_s"] = round(sum(latencies) / len(latencies), 3)
    metrics["p50_latency_s"] = round(sorted(latencies)[len(latencies) // 2], 3)
    return metrics, metrics["avg_latency_s"]


def evaluate_old_detector(
    samples: List[Dict],
    threshold: float = 0.5,
) -> Optional[Tuple[Dict, float]]:
    """Evaluate the existing DistilBERT detector for comparison.

    Returns None if old detector is unavailable.
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from agents.detector_agent.detector import DetectorAgent

        old_detector = DetectorAgent()

        y_true, y_pred, probs = [], [], []
        latencies = []

        for sample in samples:
            t0 = time.time()
            result = old_detector.detect(
                user_query=sample["query"],
                llm_response=sample["claim"],
            )
            latency = time.time() - t0
            latencies.append(latency)

            prob = result.hallucination_probability
            pred = 1 if prob >= threshold else 0

            y_true.append(sample["label"])
            y_pred.append(pred)
            probs.append(prob)

        metrics = _compute_metrics(y_true, y_pred, probs)
        metrics["avg_latency_s"] = round(sum(latencies) / len(latencies), 3)
        return metrics, metrics["avg_latency_s"]

    except Exception as e:
        logger.warning("[Evaluator] Could not evaluate old detector: %s", e)
        return None


def print_comparison_table(new_metrics: Dict, old_metrics: Optional[Dict]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("  HALLUCINATION DETECTOR BENCHMARK COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<20} {'NEW (DeBERTa)':<20} {'OLD (DistilBERT)':<20}")
    print("-" * 70)

    old = old_metrics or {}
    for key in ["accuracy", "precision", "recall", "f1", "fpr", "fnr", "ece", "avg_latency_s"]:
        new_val = f"{new_metrics.get(key, 'N/A'):.4f}" if isinstance(new_metrics.get(key), float) else str(new_metrics.get(key, "N/A"))
        old_val = f"{old.get(key, 'N/A'):.4f}" if isinstance(old.get(key), float) else str(old.get(key, "N/A"))
        print(f"  {key:<18} {new_val:<20} {old_val:<20}")

    print("=" * 70)

    # Check success criterion
    f1 = new_metrics.get("f1", 0.0)
    if f1 >= 0.78:
        print(f"  ✅ SUCCESS: F1={f1:.4f} >= 0.78 target")
    else:
        print(f"  ❌ NOT YET: F1={f1:.4f} < 0.78 target — keep training")
    print("=" * 70 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate hallucination detectors")
    parser.add_argument("--dataset",      required=True, help="Path to evaluation JSON")
    parser.add_argument("--dataset_type", default="halueval",
                        choices=["halueval", "ragtruth", "custom"])
    parser.add_argument("--output",       default="eval_results.json")
    parser.add_argument("--threshold",    type=float, default=0.5)
    parser.add_argument("--compare_old",  action="store_true",
                        help="Also evaluate old detector for comparison")
    args = parser.parse_args()

    samples = _load_dataset(args.dataset, args.dataset_type)

    print(f"\n[Evaluator] Evaluating NEW detector on {len(samples)} samples...")
    new_metrics, _ = evaluate_new_detector(samples, args.threshold)

    old_metrics = None
    if args.compare_old:
        print("[Evaluator] Evaluating OLD detector for comparison...")
        result = evaluate_old_detector(samples, args.threshold)
        if result:
            old_metrics, _ = result

    print_comparison_table(new_metrics, old_metrics)

    output = {
        "new_detector": new_metrics,
        "old_detector": old_metrics,
        "dataset": args.dataset,
        "n_samples": len(samples),
        "threshold": args.threshold,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[Evaluator] Results saved to {args.output}")
