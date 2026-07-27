import csv
import os
from typing import Any, Dict, List


class CSVExporter:
    """Utilities for exporting evaluation results to CSV files."""

    @staticmethod
    def export_evaluation_results(records: List[Dict[str, Any]], filepath: str) -> str:
        """Exports detailed per-example evaluation results to CSV.
        
        Args:
            records: List of dictionary records containing evaluation details.
            filepath: Target absolute or relative CSV filepath.
            
        Returns:
            str: Path to generated CSV file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        if not records:
            return filepath

        fieldnames = [
            "id",
            "query",
            "response",
            "category",
            "expected_risk",
            "predicted_risk",
            "is_correct",
            "confidence_score",
            "hallucination_probability",
            "next_action",
            "token_prob_avg_logprob",
            "token_prob_low_conf_ratio",
            "entropy_normalized",
            "semantic_sim_normalized",
            "self_consistency_score",
            "self_consistency_status"
        ]

        with open(filepath, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)

        return filepath

    @staticmethod
    def export_category_summary(category_stats: Dict[str, Dict[str, Any]], filepath: str) -> str:
        """Exports aggregated category performance summary to CSV.
        
        Args:
            category_stats: Dictionary containing per-category summary statistics.
            filepath: Target absolute or relative CSV filepath.
            
        Returns:
            str: Path to generated CSV file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        fieldnames = [
            "category",
            "sample_count",
            "accuracy",
            "avg_confidence",
            "avg_hallucination_prob"
        ]

        with open(filepath, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for cat_name, stats in category_stats.items():
                row = {"category": cat_name, **stats}
                writer.writerow(row)

        return filepath
