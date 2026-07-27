"""Ablation Study Framework for HalluciGuard Detector Agent.

Evaluates the relative contribution of each detection signal (Token Probability,
Predictive Entropy, Semantic Similarity, and Self-Consistency) across 9 signal configurations.
"""

import csv
import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt

from detector_agent.config import DetectorConfig, SignalWeights
from detector_agent.datasets import BenchmarkExample, HaluEvalLoader, TruthfulQALoader
from detector_agent.detector import DetectorAgent
from detector_agent.evaluation.metrics import calculate_metrics
from detector_agent.evaluation.pairwise_evaluator import PairwiseEvaluator
from detector_agent.model_manager import ModelManager


class AblationEvaluator:
    """Orchestrates ablation experiments across 9 signal configurations."""

    CONFIGURATIONS = {
        "Full Detector": SignalWeights(token_probability=0.35, entropy=0.25, semantic_similarity=0.25, self_consistency=0.15),
        "Without Token Probability": SignalWeights(token_probability=0.00, entropy=0.40, semantic_similarity=0.40, self_consistency=0.20),
        "Without Entropy": SignalWeights(token_probability=0.50, entropy=0.00, semantic_similarity=0.30, self_consistency=0.20),
        "Without Semantic Similarity": SignalWeights(token_probability=0.50, entropy=0.30, semantic_similarity=0.00, self_consistency=0.20),
        "Without Self-Consistency": SignalWeights(token_probability=0.40, entropy=0.30, semantic_similarity=0.30, self_consistency=0.00),
        "Token Probability Only": SignalWeights(token_probability=1.00, entropy=0.00, semantic_similarity=0.00, self_consistency=0.00),
        "Entropy Only": SignalWeights(token_probability=0.00, entropy=1.00, semantic_similarity=0.00, self_consistency=0.00),
        "Semantic Similarity Only": SignalWeights(token_probability=0.00, entropy=0.00, semantic_similarity=1.00, self_consistency=0.00),
        "Self-Consistency Only": SignalWeights(token_probability=0.00, entropy=0.00, semantic_similarity=0.00, self_consistency=1.00),
    }

    def __init__(self, model_manager: Optional[ModelManager] = None) -> None:
        """Initialize with a shared ModelManager to prevent reloading models across runs."""
        self.model_manager = model_manager or ModelManager()

    def run_ablation_study(self, examples: List[BenchmarkExample]) -> List[Dict[str, Any]]:
        """Runs evaluation across all 9 signal configurations."""
        ablation_results: List[Dict[str, Any]] = []
        total_evaluations_per_config = len(examples)

        for config_name, weights in self.CONFIGURATIONS.items():
            print(f"\n[AblationEvaluator] Evaluating Configuration: '{config_name}'...")

            # Create DetectorConfig with target weights
            config = DetectorConfig(signal_weights=weights)
            agent = DetectorAgent(config=config, model_manager=self.model_manager)

            # 1. Classification Evaluation & Self-Consistency Tracking
            y_true = []
            y_pred = []
            
            sc_activated_count = 0
            sc_confidence_changes = []
            sc_samples_detail = []

            for ex in examples:
                res = agent.detect(user_query=ex.query, llm_response=ex.response)
                y_true.append(ex.expected_risk)

                risk_str = res.risk_level.value if hasattr(res.risk_level, 'value') else str(res.risk_level)
                y_pred.append(risk_str)

                # Check if Self-Consistency executed for this evaluation
                if res.metrics.self_consistency is not None:
                    sc_activated_count += 1
                    
                    # Calculate single-pass confidence before SC
                    single_pass_conf, _ = agent._aggregate_scores(
                        token_prob_metrics=res.metrics.token_probability,
                        entropy_metrics=res.metrics.entropy,
                        semantic_sim_metrics=res.metrics.semantic_similarity,
                        self_consist_metrics=None
                    )
                    
                    conf_change = res.confidence_score - single_pass_conf
                    sc_confidence_changes.append(conf_change)

                    sc_samples_detail.append({
                        "query": ex.query,
                        "response": ex.response,
                        "single_pass_conf": single_pass_conf,
                        "final_conf": res.confidence_score,
                        "conf_change": round(conf_change, 4),
                        "sc_score": res.metrics.self_consistency.consistency_score
                    })

            class_metrics = calculate_metrics(y_true, y_pred)

            # Calculate SC Metrics
            sc_activation_rate = round((sc_activated_count / total_evaluations_per_config) * 100.0, 2) if total_evaluations_per_config > 0 else 0.0
            avg_conf_change_sc = round(sum(sc_confidence_changes) / len(sc_confidence_changes), 4) if sc_confidence_changes else 0.0

            # 2. Pairwise Ranking Evaluation
            pairwise_eval = PairwiseEvaluator(agent=agent)
            pairwise_res = pairwise_eval.evaluate_pairs(examples)
            pairwise_report = pairwise_eval.print_report(pairwise_res)

            row = {
                "configuration": config_name,
                "accuracy": round(class_metrics.accuracy, 4),
                "precision": round(class_metrics.precision_macro, 4),
                "recall": round(class_metrics.recall_macro, 4),
                "f1": round(class_metrics.f1_macro, 4),
                "pairwise_accuracy": round(pairwise_report.get("accuracy", 0.0), 2),
                "avg_confidence_gap": round(pairwise_report.get("avg_confidence_gap", 0.0), 4),
                "avg_hallucination_gap": round(pairwise_report.get("avg_hallucination_gap", 0.0), 4),
                "sc_activation_count": sc_activated_count,
                "sc_activation_rate": sc_activation_rate,
                "avg_confidence_change_sc": avg_conf_change_sc,
                "sc_samples_detail": sc_samples_detail
            }
            ablation_results.append(row)

        return ablation_results

    def export_csv(self, results: List[Dict[str, Any]], filepath: Optional[str] = None) -> str:
        """Exports ablation results to CSV."""
        if not filepath:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(dir_path, "ablation_results.csv")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        fieldnames = [
            "configuration",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "pairwise_accuracy",
            "avg_confidence_gap",
            "avg_hallucination_gap",
            "sc_activation_rate",
            "avg_confidence_change_sc",
        ]

        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        abs_path = os.path.abspath(filepath)
        print(f"[AblationEvaluator] Exported CSV to {abs_path}")
        return abs_path

    def generate_markdown_report(self, results: List[Dict[str, Any]], filepath: Optional[str] = None) -> str:
        """Generates markdown table report summarizing ablation results and SC activation stats."""
        if not filepath:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(dir_path, "ablation_report.md")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        lines = [
            "# HalluciGuard - Signal Ablation Study Report",
            "",
            "## Summary Performance Table",
            "",
            "| Configuration | Accuracy | Precision | Recall | Macro F1 | Pairwise Acc | Avg Conf Gap | SC Activation Rate | Avg Conf Change (SC) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for r in results:
            lines.append(
                f"| **{r['configuration']}** | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['pairwise_accuracy']:.2f}% | {r['avg_confidence_gap']:+.4f} | {r['sc_activation_rate']:.2f}% | {r['avg_confidence_change_sc']:+.4f} |"
            )

        lines.extend([
            "",
            "## Self-Consistency Gating & Execution Analysis",
            "",
            "Self-Consistency is executed conditionally by `SelfConsistencyGate` ONLY when:",
            "1. Single-pass risk level evaluates to `MEDIUM`.",
            "2. User query belongs to an analytical category (`factual_question`, `explanation`, `summarization`, `reasoning`).",
            ""
        ])

        # Add SC detailed activation section for Full Detector
        full_det = next((r for r in results if r["configuration"] == "Full Detector"), None)
        if full_det and full_det["sc_samples_detail"]:
            lines.extend([
                "### Self-Consistency Execution Log (Full Detector)",
                "",
                f"- **Total Evaluations**: {len(full_det['sc_samples_detail'])} activated runs",
                f"- **Activation Rate**: {full_det['sc_activation_rate']:.2f}%",
                f"- **Average Confidence Impact**: {full_det['avg_confidence_change_sc']:+.4f}",
                "",
                "| Query | Response Snippet | Single-Pass Conf | Final Conf | Delta Conf | SC Consistency Score |",
                "| :--- | :--- | :---: | :---: | :---: | :---: |"
            ])
            for sample in full_det["sc_samples_detail"]:
                resp_snip = sample['response'][:40] + ("..." if len(sample['response']) > 40 else "")
                lines.append(
                    f"| {sample['query']} | {resp_snip} | {sample['single_pass_conf']:.4f} | {sample['final_conf']:.4f} | {sample['conf_change']:+.4f} | {sample['sc_score']:.4f} |"
                )
            lines.append("")
        else:
            lines.extend([
                "### Self-Consistency Execution Log",
                "",
                "No evaluations triggered the `SelfConsistencyGate` (all queries were classified as LOW or HIGH risk or non-analytical prompt types).",
                ""
            ])

        lines.extend([
            "## Findings & Signal Contribution Analysis",
            "",
            "- **Full Detector**: Allocates non-zero weight (`0.15`) to Self-Consistency, executing stochastic decoding on uncertain prompts while avoiding compute bloat on clear predictions.",
            "- **Without Self-Consistency**: Disables stochastic sampling completely (`self_consistency=0.00`), saving GPU compute while relying strictly on single-pass signals.",
            "- **Token Probability**: Serves as the primary token-level certainty anchor.",
            "- **Predictive Entropy**: Measures token uncertainty distribution under choice ambiguity.",
            "- **Semantic Similarity**: Essential for measuring semantic grounding between prompt context and candidate responses.",
            "",
        ])

        content = "\n".join(lines)
        with open(filepath, mode="w", encoding="utf-8") as f:
            f.write(content)

        abs_path = os.path.abspath(filepath)
        print(f"[AblationEvaluator] Generated markdown report at {abs_path}")
        return abs_path

    def generate_bar_charts(self, results: List[Dict[str, Any]], filepath: Optional[str] = None) -> str:
        """Generates bar charts comparing Accuracy, Macro F1, and Pairwise Ranking Accuracy."""
        if not filepath:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(dir_path, "ablation_comparison.png")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        configs = [r["configuration"] for r in results]
        accuracies = [r["accuracy"] * 100.0 for r in results]
        f1_scores = [r["f1"] * 100.0 for r in results]
        pairwise_accs = [r["pairwise_accuracy"] for r in results]

        fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

        # Chart 1: Classification Accuracy
        axes[0].bar(configs, accuracies, color="#1f77b4", edgecolor="black", alpha=0.85)
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_title("Ablation Study: Classification Accuracy by Signal Configuration")
        axes[0].grid(True, linestyle=":", alpha=0.6)

        # Chart 2: Macro F1 Score
        axes[1].bar(configs, f1_scores, color="#2ca02c", edgecolor="black", alpha=0.85)
        axes[1].set_ylabel("Macro F1 (%)")
        axes[1].set_title("Ablation Study: Macro F1 Score by Signal Configuration")
        axes[1].grid(True, linestyle=":", alpha=0.6)

        # Chart 3: Pairwise Ranking Accuracy
        axes[2].bar(configs, pairwise_accs, color="#d62728", edgecolor="black", alpha=0.85)
        axes[2].set_ylabel("Pairwise Accuracy (%)")
        axes[2].set_title("Ablation Study: Pairwise Ranking Accuracy by Signal Configuration")
        axes[2].grid(True, linestyle=":", alpha=0.6)
        plt.xticks(rotation=35, ha="right")

        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()

        abs_path = os.path.abspath(filepath)
        print(f"[AblationEvaluator] Saved comparison chart to {abs_path}")
        return abs_path


if __name__ == "__main__":
    evaluator = AblationEvaluator()

    halueval = HaluEvalLoader(subset="qa")
    tqa = TruthfulQALoader(subset="generation")

    examples = halueval.load_dataset(limit=10) + tqa.load_dataset(limit=10)
    results = evaluator.run_ablation_study(examples)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    evaluator.export_csv(results, os.path.join(out_dir, "ablation_results.csv"))
    evaluator.generate_markdown_report(results, os.path.join(out_dir, "ablation_report.md"))
    evaluator.generate_bar_charts(results, os.path.join(out_dir, "ablation_comparison.png"))
