from __future__ import annotations
import json
import datetime
from typing import Dict, Any

class EvaluationReportGenerator:
    """Generates evaluation reports in JSON and Markdown formats."""
    
    def generate_json(self, benchmark_results: Dict[str, Any], output_path: str) -> None:
        """Writes evaluation results to a JSON file."""
        report = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "benchmark_name": "HalluciGuard_v2_evaluation",
            "metrics": benchmark_results,
            "per_domain_results": {},
            "api_success_rates": {}
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)

    def generate_markdown(self, benchmark_results: Dict[str, Any], output_path: str) -> None:
        """Writes a formatted markdown evaluation report."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# HalluciGuard Verifier Evaluation Report",
            f"**Generated:** {timestamp}",
            "",
            "## Summary of Benchmark Results",
            "",
            "| Benchmark | Precision | Recall | F1 Score | MRR | Avg Latency (ms) |",
            "|-----------|-----------|--------|----------|-----|------------------|"
        ]
        
        for name, res in benchmark_results.items():
            lines.append(f"| {name.capitalize()} | {res.precision:.2f} | {res.recall:.2f} | {res.f1:.2f} | {res.mrr:.2f} | {res.avg_latency_ms:.1f} |")
            
        lines.extend([
            "",
            "## Notes",
            "These metrics represent the performance of the latest iteration of the Verifier Agent Pipeline."
        ])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
