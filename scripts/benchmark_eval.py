"""
Benchmark Evaluation Script for HalluciGuard Phase 0.

Evaluates known-answer claims using the VerificationPipeline.
Calculates Accuracy, Precision, Recall, F1 score and Confusion Matrix.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import asyncio
from dotenv import load_dotenv

# Setup paths to ensure we can import the project modules
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
VERIFIER_DIR = PROJECT_ROOT / "agents" / "verifier_agent"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(VERIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFIER_DIR))

# Import pipeline and models
from api.pipeline import VerificationPipeline
from schemas.models import SuspiciousClaim, VerifierInputV2

BENCHMARK_CLAIMS = [
    # ── General Domain (8 claims) ──────────────────────────────
    {"claim": "Paris is the capital of France.", "domain": "general", "expected": "verified", "category": "direct_support"},
    {"claim": "The Eiffel Tower is located in London.", "domain": "general", "expected": "contradicted", "category": "direct_contradiction"},
    {"claim": "Amazon company was built by Sundar Pichai.", "domain": "general", "expected": "contradicted", "category": "entity_confusion"},
    {"claim": "The Moon is made of green cheese.", "domain": "general", "expected": "unverified", "category": "insufficient_evidence"},
    {"claim": "The Sun is at the center of the Solar System.", "domain": "general", "expected": "verified", "category": "direct_support"},
    {"claim": "Photosynthesis converts carbon dioxide and water into glucose.", "domain": "general", "expected": "verified", "category": "direct_support"},
    {"claim": "The Earth is flat.", "domain": "general", "expected": "unverified", "category": "adversarial_debunk"},
    {"claim": "Python was created by Guido van Rossum.", "domain": "general", "expected": "verified", "category": "direct_support"},

    # ── Healthcare Domain (8 claims) ───────────────────────────
    {"claim": "Aspirin is used to relieve mild to moderate pain.", "domain": "healthcare", "expected": "verified", "category": "medication_use"},
    {"claim": "This iron tablet cures the headache.", "domain": "healthcare", "expected": "unverified", "category": "unverified_remedy"},
    {"claim": "Paracetamol is commonly used to relieve fever.", "domain": "healthcare", "expected": "verified", "category": "medication_use"},
    {"claim": "Aspirin cures every type of cancer.", "domain": "healthcare", "expected": "unverified", "category": "exaggerated_claim"},
    {"claim": "XYZMED123 cures all bacterial infections.", "domain": "healthcare", "expected": "unverified", "category": "insufficient_evidence"},
    {"claim": "Metformin is used in the treatment of type 2 diabetes.", "domain": "healthcare", "expected": "verified", "category": "medication_use"},
    {"claim": "Ibuprofen is a nonsteroidal anti-inflammatory drug.", "domain": "healthcare", "expected": "verified", "category": "drug_class"},
    {"claim": "Water cure completely eliminates type 1 diabetes.", "domain": "healthcare", "expected": "unverified", "category": "unverified_remedy"},

    # ── Cybersecurity Domain (8 claims) ─────────────────────────
    {"claim": "CVE-2021-44228 is associated with Log4Shell.", "domain": "cybersecurity", "expected": "verified", "category": "cve_alias"},
    {"claim": "The Trojan virus is the most dangerous cyber attack.", "domain": "cybersecurity", "expected": "unverified", "category": "subjective_claim"},
    {"claim": "Log4Shell is a vulnerability in Apache Log4j.", "domain": "cybersecurity", "expected": "verified", "category": "vulnerability_fact"},
    {"claim": "CVE-2021-44228 affects Microsoft Word.", "domain": "cybersecurity", "expected": "unverified", "category": "wrong_product"},
    {"claim": "CVE-9999-99999 is a critical remote code execution vulnerability.", "domain": "cybersecurity", "expected": "unverified", "category": "nonsense_cve"},
    {"claim": "T1059 is a MITRE ATT&CK technique for Command and Scripting Interpreter.", "domain": "cybersecurity", "expected": "verified", "category": "mitre_technique"},
    {"claim": "CVE-2021-44228 is a known exploited vulnerability in CISA KEV catalog.", "domain": "cybersecurity", "expected": "verified", "category": "cisa_kev"},
    {"claim": "EternalBlue is an exploit developed by Microsoft for internal testing.", "domain": "cybersecurity", "expected": "unverified", "category": "threat_origin"},

    # ── AI Research Domain (6 claims) ───────────────────────────
    {"claim": "Transformers were introduced in the paper Attention Is All You Need.", "domain": "ai_research", "expected": "verified", "category": "paper_contribution"},
    {"claim": "The Attention Is All You Need paper was authored by Vaswani et al.", "domain": "ai_research", "expected": "verified", "category": "authorship"},
    {"claim": "ResNet was introduced in Attention Is All You Need.", "domain": "ai_research", "expected": "unverified", "category": "wrong_paper"},
    {"claim": "XYZABC999 was published at NeurIPS in 2024.", "domain": "ai_research", "expected": "unverified", "category": "fictitious_paper"},
    {"claim": "BERT stands for Bidirectional Encoder Representations from Transformers.", "domain": "ai_research", "expected": "verified", "category": "acronym_fact"},
    {"claim": "LoRA enables low-rank adaptation of large language models.", "domain": "ai_research", "expected": "verified", "category": "methodology"},

    # ── Finance Domain (5 claims) ───────────────────────────────
    {"claim": "The 1992 Indian securities scam involved stockbroker Harshad Mehta.", "domain": "finance", "expected": "verified", "category": "financial_history"},
    {"claim": "Apple Inc trades under the ticker symbol AAPL on NASDAQ.", "domain": "finance", "expected": "verified", "category": "ticker_symbol"},
    {"claim": "Tesla Inc was founded in 1850.", "domain": "finance", "expected": "contradicted", "category": "temporal_contradiction"},
    {"claim": "NonsenseCorp999 reported 500 billion revenue in 2023.", "domain": "finance", "expected": "unverified", "category": "fictitious_corporate"},
    {"claim": "Microsoft Corporation was co-founded by Bill Gates and Paul Allen.", "domain": "finance", "expected": "verified", "category": "corporate_founders"},
]

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    verdicts = ["verified", "contradicted", "unverified", "conflicted"]
    metrics = {}
    
    y_true = []
    y_pred = []
    latencies = []
    
    for r in results:
        if r.get("error"):
            continue
        y_true.append(r["expected"])
        y_pred.append(r["actual"])
        if "latency_ms" in r:
            latencies.append(r["latency_ms"])
        
    confusion_matrix = {v: {v2: 0 for v2 in verdicts} for v in verdicts}
    for t, p in zip(y_true, y_pred):
        if t in confusion_matrix and p in confusion_matrix[t]:
            confusion_matrix[t][p] += 1
            
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total > 0 else 0.0
    
    metrics["accuracy"] = accuracy
    metrics["total_evaluated"] = total
    metrics["errors"] = len(results) - total
    metrics["confusion_matrix"] = confusion_matrix
    
    per_class = {}
    for v in verdicts:
        tp = confusion_matrix[v][v]
        fp = sum(confusion_matrix[other][v] for other in verdicts if other != v)
        fn = sum(confusion_matrix[v][other] for other in verdicts if other != v)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[v] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        
    metrics["per_class"] = per_class
    
    # Macro avg
    active_classes = [v for v in verdicts if (per_class[v]["TP"] + per_class[v]["FN"]) > 0]
    if active_classes:
        metrics["macro_precision"] = round(sum(per_class[v]["precision"] for v in active_classes) / len(active_classes), 4)
        metrics["macro_recall"] = round(sum(per_class[v]["recall"] for v in active_classes) / len(active_classes), 4)
        metrics["macro_f1"] = round(sum(per_class[v]["f1"] for v in active_classes) / len(active_classes), 4)
    else:
        metrics["macro_precision"] = 0.0
        metrics["macro_recall"] = 0.0
        metrics["macro_f1"] = 0.0

    # Safety Metrics: False Verification Rate & False Contradiction Rate
    # False Verification = predicted 'verified' when actual was NOT 'verified'
    false_verified_count = sum(confusion_matrix[other]["verified"] for other in verdicts if other != "verified")
    non_verified_actual = sum(sum(confusion_matrix[other].values()) for other in verdicts if other != "verified")
    metrics["false_verification_rate"] = round(false_verified_count / non_verified_actual if non_verified_actual > 0 else 0.0, 4)

    # False Contradiction = predicted 'contradicted' when actual was NOT 'contradicted'
    false_contradicted_count = sum(confusion_matrix[other]["contradicted"] for other in verdicts if other != "contradicted")
    non_contradicted_actual = sum(sum(confusion_matrix[other].values()) for other in verdicts if other != "contradicted")
    metrics["false_contradiction_rate"] = round(false_contradicted_count / non_contradicted_actual if non_contradicted_actual > 0 else 0.0, 4)

    # Latency Percentiles
    if latencies:
        import numpy as np
        metrics["latency"] = {
            "avg_ms": int(np.mean(latencies)),
            "p50_ms": int(np.median(latencies)),
            "p95_ms": int(np.percentile(latencies, 95)),
            "min_ms": int(np.min(latencies)),
            "max_ms": int(np.max(latencies)),
        }

    # Domain Breakdown
    domain_results = defaultdict(lambda: {"total": 0, "correct": 0, "accuracy": 0.0})
    for r in results:
        if r.get("error"):
            continue
        d = r["domain"]
        domain_results[d]["total"] += 1
        if r["expected"] == r["actual"]:
            domain_results[d]["correct"] += 1
    for d, data in domain_results.items():
        data["accuracy"] = round(data["correct"] / data["total"] if data["total"] > 0 else 0.0, 4)
    metrics["domains"] = dict(domain_results)

    # Confidence Calibration Buckets
    buckets = {
        "0.0-0.2": {"count": 0, "correct": 0},
        "0.2-0.4": {"count": 0, "correct": 0},
        "0.4-0.6": {"count": 0, "correct": 0},
        "0.6-0.8": {"count": 0, "correct": 0},
        "0.8-1.0": {"count": 0, "correct": 0},
    }
    for r in results:
        if r.get("error"):
            continue
        conf = float(r.get("confidence", 0.0))
        is_corr = (r["expected"] == r["actual"])
        if conf <= 0.2: b_key = "0.0-0.2"
        elif conf <= 0.4: b_key = "0.2-0.4"
        elif conf <= 0.6: b_key = "0.4-0.6"
        elif conf <= 0.8: b_key = "0.6-0.8"
        else: b_key = "0.8-1.0"
        buckets[b_key]["count"] += 1
        if is_corr:
            buckets[b_key]["correct"] += 1
    for b_key, b_data in buckets.items():
        b_data["accuracy"] = round(b_data["correct"] / b_data["count"] if b_data["count"] > 0 else 0.0, 4)
    metrics["calibration_buckets"] = buckets

    return metrics

async def run_benchmark(limit: int = None, output_path: str = "benchmark_results.json"):
    import time
    load_dotenv()
    os.environ["VERIFIER_CACHE_ENABLED"] = "false"
    
    pipeline = VerificationPipeline()
    claims_to_run = BENCHMARK_CLAIMS[:limit] if limit else BENCHMARK_CLAIMS
    
    results = []
    
    print(f"\n========================================================================")
    print(f"  HALLUCIGUARD VERIFIER V1.7 — BENCHMARK EVALUATION SUITE")
    print(f"  Claims to evaluate: {len(claims_to_run)} across 5 domains")
    print(f"========================================================================\n")
    
    for i, claim_data in enumerate(claims_to_run):
        print(f"[{i+1}/{len(claims_to_run)}] Evaluating ({claim_data['domain'].upper()}): '{claim_data['claim']}'")
        payload = VerifierInputV2(
            query_id=f"bench-{i+1}",
            domain=claim_data["domain"],
            suspicious_claims=[SuspiciousClaim(claim_id=f"c-{i+1}", text=claim_data["claim"])],
        )
        
        result_entry = {
            "claim": claim_data["claim"],
            "domain": claim_data["domain"],
            "category": claim_data.get("category", "general"),
            "expected": claim_data["expected"],
        }
        
        t0 = time.time()
        try:
            res = await pipeline.verify(payload)
            latency_ms = int((time.time() - t0) * 1000)
            report = res.claim_evidence[0] if res.claim_evidence else None
            if report:
                result_entry["actual"] = report.verdict.value.lower()
                result_entry["confidence"] = round(report.confidence_score, 4)
                result_entry["trust_score"] = round(report.trust_score, 4)
                result_entry["explanation"] = report.explanation[:200]
                result_entry["evidence_count"] = len(report.evidence)
                result_entry["latency_ms"] = latency_ms
                if report.retrieval_trace:
                    result_entry["retrieval_method"] = "primary_only" if not report.retrieval_trace.get("tavily", {}).get("called") else "hybrid_tavily"
            else:
                result_entry["actual"] = "unverified"
                result_entry["error"] = "No claim report returned"
                result_entry["latency_ms"] = latency_ms
        except Exception as e:
            latency_ms = int((time.time() - t0) * 1000)
            print(f"  [ERROR] Evaluating claim: {e}")
            result_entry["error"] = str(e)
            result_entry["actual"] = "error"
            result_entry["latency_ms"] = latency_ms
            
        status_tag = "[MATCH]" if result_entry["expected"] == result_entry["actual"] else "[MISMATCH]"
        print(f"    Expected: {result_entry['expected'].upper():13} | Actual: {result_entry['actual'].upper():13} {status_tag} ({latency_ms}ms)")
        results.append(result_entry)
        
    metrics = calculate_metrics(results)
    
    output = {
        "metrics": metrics,
        "results": results
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        
    print("\n" + "=" * 72)
    print("  FINAL BENCHMARK EVALUATION SUMMARY")
    print("=" * 72)
    print(f"  Total Evaluated       : {metrics['total_evaluated']} (Errors: {metrics['errors']})")
    print(f"  Overall Accuracy      : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro Precision       : {metrics['macro_precision'] * 100:.2f}%")
    print(f"  Macro Recall          : {metrics['macro_recall'] * 100:.2f}%")
    print(f"  Macro F1 Score        : {metrics['macro_f1'] * 100:.2f}%")
    print(f"  False Verification    : {metrics['false_verification_rate'] * 100:.2f}%")
    print(f"  False Contradiction   : {metrics['false_contradiction_rate'] * 100:.2f}%")
    if "latency" in metrics:
        print(f"  Latency (P50 / P95)   : {metrics['latency']['p50_ms']} ms / {metrics['latency']['p95_ms']} ms")
    
    print("\n  Per-Domain Accuracy:")
    for d, data in metrics["domains"].items():
        print(f"    - {d.upper():14}: {data['correct']}/{data['total']} ({data['accuracy'] * 100:.1f}%)")

    print("\n  Per-Class Precision / Recall / F1:")
    for v, data in metrics["per_class"].items():
        print(f"    - {v.upper():13}: P={data['precision']*100:.1f}%  R={data['recall']*100:.1f}%  F1={data['f1']*100:.1f}%  (TP={data['TP']}, FP={data['FP']}, FN={data['FN']})")

    print("\n  Confusion Matrix:")
    for v in metrics["confusion_matrix"]:
        print(f"    {v:13}: {metrics['confusion_matrix'][v]}")
        
    print(f"\nDetailed results saved to: {output_path}\n" + "=" * 72 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run HalluciGuard Benchmark Evaluation")
    parser.add_argument("--output", type=str, default="benchmark_results_v1.7.json", help="Path to save results JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of claims to evaluate")
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(limit=args.limit, output_path=args.output))

if __name__ == "__main__":
    main()
