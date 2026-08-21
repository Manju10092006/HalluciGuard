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
from agents.verifier_agent.pipeline import VerificationPipeline
from agents.verifier_agent.schemas.models import VerificationRequest

BENCHMARK_CLAIMS = [
    {"claim": "Paris is the capital of France.", "domain": "general", "expected": "verified"},
    {"claim": "The Eiffel Tower is located in London.", "domain": "general", "expected": "contradicted"},
    {"claim": "Amazon company was built by Sundar Pichai.", "domain": "general", "expected": "contradicted"},
    {"claim": "AlphaFold was developed by DeepMind to predict protein structures.", "domain": "general", "expected": "verified"},
    {"claim": "Peter Shor developed Shor's algorithm for quantum factoring in 1994.", "domain": "general", "expected": "verified"},
    {"claim": "The 1992 Indian securities scam involved stockbroker Harshad Mehta.", "domain": "general", "expected": "verified"},
    {"claim": "The Moon is made of green cheese.", "domain": "general", "expected": "unverified"},
    {"claim": "Java was created by Gaurav.", "domain": "general", "expected": "contradicted"},
    {"claim": "The Sun is at the center of the Solar System.", "domain": "general", "expected": "verified"},
    {"claim": "Photosynthesis converts carbon dioxide and water into glucose.", "domain": "general", "expected": "verified"},
    {"claim": "Albert Einstein developed the theory of general relativity.", "domain": "general", "expected": "verified"},
    {"claim": "The Great Wall of China is visible from space with the naked eye.", "domain": "general", "expected": "contradicted"},
    {"claim": "Water boils at 100 degrees Celsius at sea level.", "domain": "general", "expected": "verified"},
    {"claim": "Aspirin is used to relieve mild to moderate pain.", "domain": "healthcare", "expected": "verified"},
    {"claim": "This iron tablet cures the headache.", "domain": "healthcare", "expected": "unverified"},
    {"claim": "CVE-2021-44228 is associated with Log4Shell.", "domain": "cybersecurity", "expected": "verified"},
    {"claim": "The Trojan virus is the most dangerous cyber attack.", "domain": "cybersecurity", "expected": "unverified"},
    {"claim": "Transformers were introduced in Attention Is All You Need.", "domain": "general", "expected": "verified"},
    {"claim": "XYZMED123 cures all bacterial infections.", "domain": "healthcare", "expected": "unverified"},
    {"claim": "The Earth is flat.", "domain": "general", "expected": "contradicted"},
    {"claim": "Python was created by Guido van Rossum.", "domain": "general", "expected": "verified"},
    {"claim": "HTML is a programming language.", "domain": "general", "expected": "contradicted"},
]

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    verdicts = ["verified", "contradicted", "unverified", "conflicted"]
    metrics = {}
    
    y_true = []
    y_pred = []
    
    for r in results:
        if r.get("error"):
            continue
        y_true.append(r["expected"])
        y_pred.append(r["actual"])
        
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
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
    metrics["per_class"] = per_class
    
    # Macro avg
    active_classes = [v for v in verdicts if (per_class[v]["TP"] + per_class[v]["FN"]) > 0]
    if active_classes:
        metrics["macro_precision"] = sum(per_class[v]["precision"] for v in active_classes) / len(active_classes)
        metrics["macro_recall"] = sum(per_class[v]["recall"] for v in active_classes) / len(active_classes)
        metrics["macro_f1"] = sum(per_class[v]["f1"] for v in active_classes) / len(active_classes)
    else:
        metrics["macro_precision"] = 0.0
        metrics["macro_recall"] = 0.0
        metrics["macro_f1"] = 0.0

    return metrics

async def run_benchmark(limit: int = None, output_path: str = "benchmark_results.json"):
    load_dotenv()
    os.environ["DISABLE_CACHE"] = "1"
    
    pipeline = VerificationPipeline()
    claims_to_run = BENCHMARK_CLAIMS[:limit] if limit else BENCHMARK_CLAIMS
    
    results = []
    
    print(f"Running benchmark on {len(claims_to_run)} claims...")
    
    for i, claim_data in enumerate(claims_to_run):
        print(f"[{i+1}/{len(claims_to_run)}] Evaluating claim: {claim_data['claim']}")
        req = VerificationRequest(claim=claim_data["claim"], domain=claim_data["domain"])
        
        result_entry = {
            "claim": claim_data["claim"],
            "domain": claim_data["domain"],
            "expected": claim_data["expected"]
        }
        
        try:
            res = await pipeline.verify_claim(req)
            result_entry["actual"] = res.verdict
            result_entry["confidence"] = res.confidence
            result_entry["reasoning"] = res.reasoning
        except Exception as e:
            print(f"Error evaluating claim: {e}")
            result_entry["error"] = str(e)
            result_entry["actual"] = "error"
            
        results.append(result_entry)
        
    metrics = calculate_metrics(results)
    
    output = {
        "metrics": metrics,
        "results": results
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
        
    print("\n--- Benchmark Complete ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro P: {metrics['macro_precision']:.4f}, R: {metrics['macro_recall']:.4f}, F1: {metrics['macro_f1']:.4f}")
    
    print("\nConfusion Matrix:")
    for v in metrics["confusion_matrix"]:
        print(f"  {v}: {metrics['confusion_matrix'][v]}")
        
    print(f"\nResults saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run HalluciGuard Benchmark Evaluation")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Path to save results JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of claims to evaluate")
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(limit=args.limit, output_path=args.output))

if __name__ == "__main__":
    main()
