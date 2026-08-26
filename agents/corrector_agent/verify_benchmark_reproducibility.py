"""Step 4A.8 — Metric Reproducibility Verification Script.

Independently loads `test_benchmark_report.json`, recomputes all aggregate metrics
from the raw per-sample evaluation records (`detailed_sample_results`), compares them
against the stored aggregate metrics, and asserts mathematical equivalence.

Fails with exit code 1 if any metric differs beyond tolerance (1e-4).
"""

import os
import sys
import json
import math
from typing import Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.corrector_agent.benchmark_engine import aggregate_metrics_from_results

REPORT_PATH = os.path.join(
    os.path.dirname(__file__),
    "training", "data", "contract_v4", "test_benchmark_report.json"
)
TOLERANCE = 1e-4


def verify_report(report_path: str = REPORT_PATH) -> bool:
    print("=" * 80)
    print("      STEP 4A.8 — BENCHMARK METRIC REPRODUCIBILITY VERIFICATION          ")
    print("=" * 80)
    print(f"Target Report: {report_path}")

    if not os.path.exists(report_path):
        print(f"[FAIL] Report file does not exist: {report_path}")
        return False

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    if "detailed_sample_results" not in report or not report["detailed_sample_results"]:
        print("[FAIL] 'detailed_sample_results' missing or empty in report.")
        return False

    results = report["detailed_sample_results"]
    stored_agg = report.get("aggregate_metrics", {})
    eval_count = len(results)
    total_samples = report.get("dataset_metadata", {}).get("total_records", eval_count)

    print(f"Loaded {eval_count} per-sample records.")
    print("Independently recomputing all 5 metric categories...\n")

    recomputed = aggregate_metrics_from_results(results, total_samples)

    all_passed = True
    mismatches = []

    def check_dict(stored: Dict[str, Any], recomputed_dict: Dict[str, Any], prefix: str = ""):
        nonlocal all_passed
        for k, v in recomputed_dict.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                if k not in stored:
                    mismatches.append(f"Missing section in stored report: {full_key}")
                    all_passed = False
                else:
                    check_dict(stored[k], v, full_key)
            else:
                stored_val = stored.get(k)
                if stored_val is None:
                    mismatches.append(f"Missing metric in stored report: {full_key}")
                    all_passed = False
                elif isinstance(v, (int, float)) and isinstance(stored_val, (int, float)):
                    diff = abs(float(v) - float(stored_val))
                    if diff > TOLERANCE:
                        mismatches.append(
                            f"Mismatch at {full_key}: stored={stored_val}, recomputed={v} (diff={diff:.6f})"
                        )
                        all_passed = False
                    else:
                        print(f"  [OK] {full_key:45s} = {v} (stored: {stored_val})")
                else:
                    if str(v) != str(stored_val):
                        mismatches.append(
                            f"Mismatch at {full_key}: stored={stored_val}, recomputed={v}"
                        )
                        all_passed = False
                    else:
                        print(f"  [OK] {full_key:45s} = {v}")

    check_dict(stored_agg, recomputed)

    print("-" * 80)
    if all_passed:
        print("REPRODUCIBILITY CHECK: PASS")
        print(f"All {eval_count} sample metrics independently verified with zero mathematical drift.")
        print("=" * 80)
        return True
    else:
        print("REPRODUCIBILITY CHECK: FAIL")
        print("Mismatches found:")
        for m in mismatches:
            print(f"  - {m}")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = verify_report()
    sys.exit(0 if success else 1)
