from typing import Dict, Any, List, Optional
from .metrics import ClassificationMetrics


class ConsoleReporter:
    """Utilities for generating clean, formatted console evaluation reports."""

    @staticmethod
    def print_header(title: str) -> None:
        """Print a visually distinct header bar."""
        border = "=" * 65
        print(f"\n{border}")
        print(f" {title}")
        print(f"{border}\n")

    @staticmethod
    def print_overall_metrics(metrics: ClassificationMetrics) -> None:
        """Print summary table of overall classification metrics."""
        print("--- Overall Classification Performance ---")
        print(f"  Accuracy:          {metrics.accuracy * 100:.2f}%")
        print(f"  Macro Precision:   {metrics.precision_macro:.4f}")
        print(f"  Macro Recall:      {metrics.recall_macro:.4f}")
        print(f"  Macro F1-Score:    {metrics.f1_macro:.4f}")
        print(f"  Weighted Precision:{metrics.precision_weighted:.4f}")
        print(f"  Weighted Recall:   {metrics.recall_weighted:.4f}")
        print(f"  Weighted F1-Score: {metrics.f1_weighted:.4f}\n")

        print("--- Per-Class Breakdown ---")
        print(f"{'Class':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
        print("-" * 58)
        for cls_name, m in metrics.class_metrics.items():
            print(f"{cls_name:<10} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f} | {m['support']:<8}")
        print()

    @staticmethod
    def print_confusion_matrix(matrix: Dict[str, Dict[str, int]], labels: Optional[List[str]] = None) -> None:
        """Print formatted 2D confusion matrix table."""
        if labels is None:
            labels = ["LOW", "MEDIUM", "HIGH"]

        print("--- Confusion Matrix (True / Predicted) ---")
        header_title = "True / Pred"
        header = f"{header_title:<12} | " + " | ".join(f"{lbl:^8}" for lbl in labels)
        print(header)
        print("-" * len(header))
        for true_lbl in labels:
            row_str = f"{true_lbl:<12} | " + " | ".join(f"{matrix[true_lbl].get(p, 0):^8}" for p in labels)
            print(row_str)
        print()

    @staticmethod
    def print_category_breakdown(category_stats: Dict[str, Dict[str, Any]]) -> None:
        """Print summary table showing detector performance per prompt category."""
        print("--- Performance Breakdown by Query Category ---")
        header = f"{'Category':<22} | {'Count':<6} | {'Accuracy':<10} | {'Avg Conf':<10} | {'Avg Halluc':<10}"
        print(header)
        print("-" * len(header))
        for cat_name, stats in category_stats.items():
            count = stats["sample_count"]
            acc = f"{stats['accuracy'] * 100:.1f}%"
            avg_conf = f"{stats['avg_confidence']:.4f}"
            avg_halluc = f"{stats['avg_hallucination_prob']:.4f}"
            print(f"{cat_name:<22} | {count:<6} | {acc:<10} | {avg_conf:<10} | {avg_halluc:<10}")
        print()

    @staticmethod
    def print_signal_contributions(signal_stats: Dict[str, Dict[str, Any]]) -> None:
        """Print statistical contribution summary across active signal processors."""
        print("--- Detection Signal Summary ---")
        header = f"{'Signal Processor':<22} | {'Active Count':<12} | {'Mean Certainty Score':<20}"
        print(header)
        print("-" * len(header))
        for sig_name, stats in signal_stats.items():
            count = stats["count"]
            mean_score = f"{stats['mean_score']:.4f}" if stats["mean_score"] is not None else "N/A (Skipped)"
            print(f"{sig_name:<22} | {count:<12} | {mean_score:<20}")
        print()
