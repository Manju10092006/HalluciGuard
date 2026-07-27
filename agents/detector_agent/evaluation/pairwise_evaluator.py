"""Pairwise Ranking Evaluator for HalluciGuard Detector Agent.

Evaluates whether HalluciGuard consistently ranks truthful responses higher 
than hallucinated responses across paired benchmark examples sharing pair_id.
"""

import csv
import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt

from detector_agent.datasets import BenchmarkExample, HaluEvalLoader, TruthfulQALoader
from detector_agent.detector import DetectorAgent


class PairwiseEvaluator:
    """Evaluates HalluciGuard's pairwise ranking accuracy across paired benchmark samples."""

    def __init__(self, agent: Optional[DetectorAgent] = None) -> None:
        """Initialize PairwiseEvaluator with optional DetectorAgent instance."""
        self.agent = agent or DetectorAgent()

    def group_pairs(self, examples: List[BenchmarkExample]) -> Dict[str, Dict[str, BenchmarkExample]]:
        """Groups benchmark examples by pair_id into 'LOW' (correct) and 'HIGH' (hallucinated) pairs."""
        pairs: Dict[str, Dict[str, BenchmarkExample]] = {}

        for ex in examples:
            if not ex.pair_id:
                continue

            if ex.pair_id not in pairs:
                pairs[ex.pair_id] = {}

            if ex.expected_risk == "LOW":
                pairs[ex.pair_id]["correct"] = ex
            elif ex.expected_risk == "HIGH":
                pairs[ex.pair_id]["hallucinated"] = ex

        # Filter only complete pairs containing both correct and hallucinated examples
        complete_pairs = {k: v for k, v in pairs.items() if "correct" in v and "hallucinated" in v}
        return complete_pairs

    def evaluate_pairs(self, examples: List[BenchmarkExample]) -> List[Dict[str, Any]]:
        """Runs DetectorAgent on paired examples and calculates pairwise ranking metrics."""
        complete_pairs = self.group_pairs(examples)
        results: List[Dict[str, Any]] = []

        for pair_id, pair in complete_pairs.items():
            correct_ex = pair["correct"]
            halluc_ex = pair["hallucinated"]

            # Run detection on correct response
            res_correct = self.agent.detect(user_query=correct_ex.query, llm_response=correct_ex.response)

            # Run detection on hallucinated response
            res_halluc = self.agent.detect(user_query=halluc_ex.query, llm_response=halluc_ex.response)

            conf_correct = res_correct.confidence_score
            conf_halluc = res_halluc.confidence_score

            prob_correct = res_correct.hallucination_probability
            prob_halluc = res_halluc.hallucination_probability

            # Pairwise ranking rule: PASS if confidence(correct) > confidence(hallucinated)
            confidence_gap = conf_correct - conf_halluc
            hallucination_gap = prob_halluc - prob_correct
            ranking_result = "PASS" if conf_correct > conf_halluc else "FAIL"

            risk_correct_str = res_correct.risk_level.value if hasattr(res_correct.risk_level, 'value') else str(res_correct.risk_level)
            risk_halluc_str = res_halluc.risk_level.value if hasattr(res_halluc.risk_level, 'value') else str(res_halluc.risk_level)

            results.append(
                {
                    "pair_id": pair_id,
                    "dataset": correct_ex.source_dataset,
                    "category": correct_ex.category,
                    "query": correct_ex.query,
                    "correct_response": correct_ex.response,
                    "hallucinated_response": halluc_ex.response,
                    "correct_confidence": round(conf_correct, 4),
                    "hallucinated_confidence": round(conf_halluc, 4),
                    "confidence_gap": round(confidence_gap, 4),
                    "correct_probability": round(prob_correct, 4),
                    "hallucinated_probability": round(prob_halluc, 4),
                    "hallucination_gap": round(hallucination_gap, 4),
                    "correct_risk": risk_correct_str,
                    "hallucinated_risk": risk_halluc_str,
                    "ranking_result": ranking_result,
                }
            )

        return results

    def print_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prints a research-grade console report of pairwise ranking metrics."""
        total_pairs = len(results)
        if total_pairs == 0:
            print("No valid pairs found for evaluation.")
            return {}

        correct_rankings = sum(1 for r in results if r["ranking_result"] == "PASS")
        incorrect_rankings = total_pairs - correct_rankings
        accuracy = (correct_rankings / total_pairs) * 100.0

        avg_conf_gap = sum(r["confidence_gap"] for r in results) / total_pairs
        avg_halluc_gap = sum(r["hallucination_gap"] for r in results) / total_pairs

        print("\n================================================")
        print("         Pairwise Evaluation Report")
        print("================================================")
        print(f" Total Pairs:                {total_pairs}")
        print(f" Correct Rankings (PASS):    {correct_rankings}")
        print(f" Incorrect Rankings (FAIL):  {incorrect_rankings}")
        print(f" Pairwise Ranking Accuracy:  {accuracy:.2f}%")
        print(f" Average Confidence Gap:     {avg_conf_gap:+.4f}")
        print(f" Average Hallucination Gap:  {avg_halluc_gap:+.4f}")
        print("================================================\n")

        return {
            "total_pairs": total_pairs,
            "correct_rankings": correct_rankings,
            "incorrect_rankings": incorrect_rankings,
            "accuracy": accuracy,
            "avg_confidence_gap": avg_conf_gap,
            "avg_hallucination_gap": avg_halluc_gap,
        }

    def export_csv(self, results: List[Dict[str, Any]], filepath: Optional[str] = None) -> str:
        """Exports pairwise ranking evaluation results to CSV file."""
        if not filepath:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(dir_path, "pairwise_results.csv")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        fieldnames = [
            "pair_id",
            "dataset",
            "query",
            "correct_confidence",
            "hallucinated_confidence",
            "confidence_gap",
            "correct_probability",
            "hallucinated_probability",
            "ranking_result",
        ]

        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        abs_path = os.path.abspath(filepath)
        print(f"[PairwiseEvaluator] Exported {len(results)} rows to {abs_path}")
        return abs_path

    def plot_distributions(self, results: List[Dict[str, Any]], filepath: Optional[str] = None) -> str:
        """Generates and saves histograms of confidence gaps and hallucination probability gaps."""
        if not filepath:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(dir_path, "pairwise_gaps_distribution.png")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        conf_gaps = [r["confidence_gap"] for r in results]
        halluc_gaps = [r["hallucination_gap"] for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Confidence Gap Distribution
        axes[0].hist(conf_gaps, bins=15, color="#2b5c8f", edgecolor="black", alpha=0.85)
        axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Threshold")
        axes[0].set_title("Confidence Gap Distribution\n(Confidence_Correct - Confidence_Hallucinated)")
        axes[0].set_xlabel("Confidence Gap")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        axes[0].grid(True, linestyle=":", alpha=0.6)

        # Plot 2: Hallucination Probability Gap Distribution
        axes[1].hist(halluc_gaps, bins=15, color="#d95f02", edgecolor="black", alpha=0.85)
        axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Threshold")
        axes[1].set_title("Hallucination Probability Gap Distribution\n(Prob_Hallucinated - Prob_Correct)")
        axes[1].set_xlabel("Hallucination Probability Gap")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        axes[1].grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()

        abs_path = os.path.abspath(filepath)
        print(f"[PairwiseEvaluator] Saved gap distribution plot to {abs_path}")
        return abs_path


if __name__ == "__main__":
    evaluator = PairwiseEvaluator()

    halueval = HaluEvalLoader(subset="qa")
    tqa = TruthfulQALoader(subset="generation")

    examples = halueval.load_dataset(limit=10) + tqa.load_dataset(limit=10)
    results = evaluator.evaluate_pairs(examples)

    evaluator.print_report(results)
    evaluator.export_csv(results)
    evaluator.plot_distributions(results)
