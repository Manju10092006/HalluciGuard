from __future__ import annotations
import json
from typing import List, Dict, Any
from benchmarks.runner import BenchmarkResults
from api.pipeline import VerificationPipeline

class FEVERBenchmark:
    """Benchmark implementation for FEVER dataset."""
    
    def __init__(self):
        self.label_map = {
            "SUPPORTS": "verified",
            "REFUTES": "likely_hallucinated",
            "NOT ENOUGH INFO": "insufficient_evidence"
        }

    def load(self, path: str) -> List[Dict[str, Any]]:
        test_cases = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    test_cases.append({
                        "claim": data.get("claim", ""),
                        "label": self.label_map.get(data.get("label", "NOT ENOUGH INFO"), "insufficient_evidence"),
                        "evidence": data.get("evidence", [])
                    })
        except FileNotFoundError:
            pass
        return test_cases

    async def evaluate(self, pipeline: VerificationPipeline, test_cases: List[Dict[str, Any]]) -> BenchmarkResults:
        import time
        import logging
        
        if not getattr(pipeline, "model", None):
            logging.warning("FEVER: Pipeline model is not available. Mocking evaluation.")
            return BenchmarkResults(
                precision=0.0,
                recall=0.0,
                f1=0.0,
                mrr=0.0,
                avg_latency_ms=0.0,
                total_samples=len(test_cases),
                correct=0,
                api_success_rate=0.0,
                mock_mode=True
            )
            
        correct = 0
        total = len(test_cases)
        true_pos = 0
        false_pos = 0
        false_neg = 0
        
        start_time = time.time()
        for case in test_cases:
            result = await pipeline.verify(case["claim"])
            verdict = result.verdict if hasattr(result, "verdict") else "insufficient_evidence"
            
            if verdict == case["label"]:
                correct += 1
                
            # Treat verified as positive class for metrics
            if verdict == "verified" and case["label"] == "verified":
                true_pos += 1
            elif verdict == "verified" and case["label"] != "verified":
                false_pos += 1
            elif verdict != "verified" and case["label"] == "verified":
                false_neg += 1
                
        latency = ((time.time() - start_time) / total * 1000) if total > 0 else 0.0
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        mrr = correct / total if total > 0 else 0.0
        
        return BenchmarkResults(
            precision=precision,
            recall=recall,
            f1=f1,
            mrr=mrr,
            avg_latency_ms=latency,
            total_samples=total,
            correct=correct,
            api_success_rate=1.0,
            mock_mode=False
        )
