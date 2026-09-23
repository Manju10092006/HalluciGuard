"""
scripts / fit_calibration.py
────────────────────────────
Fit calibration + select routing thresholds on DEV data only, then report
(but do not tune on) the frozen test split.

Pipeline (handoff spec §20-23):
    raw scores + labels (DEV)
        -> select_best_calibrator  (lowest dev ECE)
        -> calibrate DEV probs
        -> select_thresholds       (LOW miss <= 10%, HIGH precision >= 60%)
        -> freeze
        -> apply ONCE to TEST, report ECE / band stats

IMPORTANT: this script needs REAL raw model scores aligned to labels. It reads
them from a JSON file of records: [{"raw_score": float, "label": 0|1}, ...].
It does NOT fabricate scores from labels. If you don't yet have real DeBERTa
scores dumped per claim, generate them first with the model — do not synthesise.

Usage:
    python scripts/fit_calibration.py --dev <dev_scores.json> --test <test_scores.json>
"""

from __future__ import annotations
import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from halluciguard_detector.calibration import (  # noqa: E402
    select_best_calibrator,
    expected_calibration_error,
)
from halluciguard_detector.thresholds import select_thresholds  # noqa: E402


def _load(path: Path):
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    raw = [float(r["raw_score"]) for r in records]
    labels = [int(r["label"]) for r in records]
    return raw, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True, help="JSON of [{raw_score,label}] on DEV")
    ap.add_argument("--test", help="JSON of [{raw_score,label}] on frozen TEST")
    ap.add_argument("--out", default="config/calibration_fitted.json")
    ap.add_argument("--low-miss", type=float, default=0.10)
    ap.add_argument("--low-cov-min", type=float, default=0.20)
    ap.add_argument("--high-prec", type=float, default=0.60)
    args = ap.parse_args()

    dev_raw, dev_labels = _load(Path(args.dev))
    print(f"[fit] DEV: {len(dev_raw)} claims, positives={sum(dev_labels)}")

    # 1. Calibration selected on DEV only.
    calibrator, cal_report = select_best_calibrator(dev_raw, dev_labels, dataset_name="DEV")
    print(f"[fit] calibration ECE by method: {cal_report}")

    dev_probs = calibrator.calibrate(dev_raw)

    # 2. Thresholds selected on DEV calibrated probs only.
    sel = select_thresholds(
        dev_probs,
        dev_labels,
        low_miss_target=args.low_miss,
        low_coverage_min=args.low_cov_min,
        high_precision_target=args.high_prec,
    )
    print(f"[fit] thresholds (DEV): {sel.to_dict()}")

    fitted = {
        "calibration": calibrator.describe(),
        "calibration_dev_report": cal_report,
        "thresholds": sel.to_dict(),
    }

    # 3. Report on frozen TEST — evaluate once, never tune.
    if args.test:
        test_raw, test_labels = _load(Path(args.test))
        test_probs = calibrator.calibrate(test_raw)
        test_ece = expected_calibration_error(test_probs, test_labels)
        import numpy as np

        tp = np.asarray(test_probs)
        ty = np.asarray(test_labels)
        low_mask = tp <= sel.t_low
        high_mask = tp >= sel.t_high
        test_report = {
            "n_claims": len(test_raw),
            "ece": round(test_ece, 4),
            "low_coverage": round(float(low_mask.mean()), 4),
            "low_miss_rate": round(float(ty[low_mask].mean()) if low_mask.any() else 0.0, 4),
            "high_coverage": round(float(high_mask.mean()), 4),
            "high_precision": round(float(ty[high_mask].mean()) if high_mask.any() else 0.0, 4),
        }
        fitted["frozen_test_report"] = test_report
        print(f"[fit] FROZEN TEST report: {test_report}")
        print(f"[fit] G3 (ECE<=0.10): {'PASS' if test_ece <= 0.10 else 'FAIL'} (ECE={test_ece:.4f})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fitted, f, indent=2)
    print(f"[fit] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
