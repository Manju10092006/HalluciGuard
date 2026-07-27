import os
from typing import Any, Dict, List, Optional
from ..config import DetectorConfig
from ..detector import DetectorAgent
from ..model_manager import ModelManager
from .csv_export import CSVExporter
from .metrics import ClassificationMetrics, calculate_metrics
from .report import ConsoleReporter


class Evaluator:
    """Core evaluation orchestrator for running benchmarks against the DetectorAgent."""

    MOCK_BENCHMARK_DATASET: List[Dict[str, str]] = [
        {
            "query": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "expected_risk": "LOW",
            "category": "factual_correct"
        },
        {
            "query": "Who wrote the play Romeo and Juliet?",
            "response": "Romeo and Juliet was written by William Shakespeare.",
            "expected_risk": "LOW",
            "category": "factual_correct"
        },
        {
            "query": "What is the chemical symbol for Gold?",
            "response": "The chemical symbol for Gold is Au.",
            "expected_risk": "LOW",
            "category": "factual_correct"
        },
        {
            "query": "When was Albert Einstein born?",
            "response": "Albert Einstein was born on March 14, 1879.",
            "expected_risk": "LOW",
            "category": "factual_correct"
        },
        {
            "query": "What is the capital of France?",
            "response": "The capital of France is Berlin.",
            "expected_risk": "HIGH",
            "category": "factual_hallucinated"
        },
        {
            "query": "Who invented the telephone?",
            "response": "The telephone was invented by Isaac Newton in 1995.",
            "expected_risk": "HIGH",
            "category": "factual_hallucinated"
        },
        {
            "query": "What is quantum mechanics?",
            "response": "Quantum mechanics is a branch of physics dealing with atomic scales.",
            "expected_risk": "MEDIUM",
            "category": "partially_correct"
        },
        {
            "query": "What causes rain?",
            "response": "Pizza is a delicious dish made with dough, cheese, and sauce.",
            "expected_risk": "HIGH",
            "category": "unrelated_response"
        },
        {
            "query": "Write a short creative story about a flying robot.",
            "response": "Once upon a time, a small robot named Aero unfolded silver wings and flew into the clouds.",
            "expected_risk": "LOW",
            "category": "creative_writing"
        },
        {
            "query": "Explain step-by-step how to calculate 12 squared.",
            "response": "To calculate 12 squared, multiply 12 by 12. 12 times 12 equals 144.",
            "expected_risk": "LOW",
            "category": "reasoning"
        }
    ]

    def __init__(
        self,
        agent: Optional[DetectorAgent] = None,
        dataset: Optional[List[Dict[str, str]]] = None
    ) -> None:
        """Initialize Evaluator with an optional DetectorAgent and benchmark dataset."""
        self.agent: DetectorAgent = agent or DetectorAgent()
        self.dataset: List[Dict[str, str]] = dataset or self.MOCK_BENCHMARK_DATASET

    def load_dataset(self, custom_dataset: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """Load benchmark dataset into evaluator."""
        if custom_dataset:
            self.dataset = custom_dataset
        return self.dataset

    def run_evaluation(self) -> Dict[str, Any]:
        """Execute all benchmark samples through DetectorAgent and aggregate metrics."""
        records = []
        y_true = []
        y_pred = []

        category_data = {}
        signal_data = {
            "token_probability": {"scores": []},
            "entropy": {"scores": []},
            "semantic_similarity": {"scores": []},
            "self_consistency": {"scores": []}
        }

        print(f"[Evaluator] Running benchmark evaluation across {len(self.dataset)} samples...")

        for idx, sample in enumerate(self.dataset, start=1):
            query = sample["query"]
            response = sample["response"]
            expected_risk = sample["expected_risk"]
            category = sample.get("category", "other")

            # Run DetectorAgent pipeline
            result = self.agent.detect(user_query=query, llm_response=response)
            predicted_risk = result.risk_level.value
            is_correct = (expected_risk == predicted_risk)

            y_true.append(expected_risk)
            y_pred.append(predicted_risk)

            # Record per-signal metrics breakdown
            metrics = result.metrics
            token_prob_logprob = metrics.token_probability.avg_logprob if metrics and metrics.token_probability else None
            token_prob_ratio = metrics.token_probability.low_confidence_ratio if metrics and metrics.token_probability else None
            entropy_norm = metrics.entropy.normalized_entropy if metrics and metrics.entropy else None
            sem_sim_norm = metrics.semantic_similarity.normalized_similarity if metrics and metrics.semantic_similarity else None
            self_consist_score = metrics.self_consistency.consistency_score if metrics and metrics.self_consistency else None
            self_consist_status = "Executed" if (metrics and metrics.self_consistency) else "Skipped"

            # Aggregate signal score contributions
            if metrics and metrics.token_probability:
                signal_data["token_probability"]["scores"].append(round(metrics.token_probability.avg_logprob, 4))
            if metrics and metrics.entropy:
                signal_data["entropy"]["scores"].append(round(1.0 - metrics.entropy.normalized_entropy, 4))
            if metrics and metrics.semantic_similarity:
                signal_data["semantic_similarity"]["scores"].append(round(metrics.semantic_similarity.normalized_similarity, 4))
            if metrics and metrics.self_consistency:
                signal_data["self_consistency"]["scores"].append(round(metrics.self_consistency.consistency_score, 4))

            # Record detailed row
            record = {
                "id": idx,
                "query": query,
                "response": response,
                "category": category,
                "expected_risk": expected_risk,
                "predicted_risk": predicted_risk,
                "is_correct": is_correct,
                "confidence_score": result.confidence_score,
                "hallucination_probability": result.hallucination_probability,
                "next_action": result.next_action.value,
                "token_prob_avg_logprob": token_prob_logprob,
                "token_prob_low_conf_ratio": token_prob_ratio,
                "entropy_normalized": entropy_norm,
                "semantic_sim_normalized": sem_sim_norm,
                "self_consistency_score": self_consist_score,
                "self_consistency_status": self_consist_status
            }
            records.append(record)

            # Aggregate category stats
            if category not in category_data:
                category_data[category] = {
                    "total": 0,
                    "correct": 0,
                    "conf_sum": 0.0,
                    "halluc_sum": 0.0
                }
            category_data[category]["total"] += 1
            if is_correct:
                category_data[category]["correct"] += 1
            category_data[category]["conf_sum"] += result.confidence_score
            category_data[category]["halluc_sum"] += result.hallucination_probability

        # Calculate Classification Metrics
        metrics_summary = calculate_metrics(y_true, y_pred, labels=["LOW", "MEDIUM", "HIGH"])

        # Format Category Breakdown
        category_stats = {}
        for cat, data in category_data.items():
            cnt = data["total"]
            category_stats[cat] = {
                "sample_count": cnt,
                "accuracy": round(data["correct"] / cnt, 4) if cnt > 0 else 0.0,
                "avg_confidence": round(data["conf_sum"] / cnt, 4) if cnt > 0 else 0.0,
                "avg_hallucination_prob": round(data["halluc_sum"] / cnt, 4) if cnt > 0 else 0.0
            }

        # Format Signal Contribution Summary
        signal_stats = {}
        for sig_name, data in signal_data.items():
            scores = data["scores"]
            count = len(scores)
            mean_val = round(sum(scores) / count, 4) if count > 0 else None
            signal_stats[sig_name] = {
                "count": count,
                "mean_score": mean_val
            }

        return {
            "records": records,
            "metrics": metrics_summary,
            "category_stats": category_stats,
            "signal_stats": signal_stats
        }


def main():
    """CLI entrypoint for running end-to-end evaluation using mock benchmark dataset."""
    ConsoleReporter.print_header("HalluciGuard - Detector Agent Evaluation Framework")

    # Use fast dev model for mock verification run
    model_name = "gpt2"
    model_manager = ModelManager(model_name=model_name)
    config = DetectorConfig(model_name=model_name)
    agent = DetectorAgent(config=config, model_manager=model_manager)

    evaluator = Evaluator(agent=agent)
    results = evaluator.run_evaluation()

    # Print Formatted Console Report
    ConsoleReporter.print_overall_metrics(results["metrics"])
    ConsoleReporter.print_confusion_matrix(results["metrics"].confusion_matrix)
    ConsoleReporter.print_category_breakdown(results["category_stats"])
    ConsoleReporter.print_signal_contributions(results["signal_stats"])

    # Export Results to CSV directly in detector_agent/evaluation/
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results_csv = os.path.join(output_dir, "evaluation_results.csv")
    category_csv = os.path.join(output_dir, "category_summary.csv")

    CSVExporter.export_evaluation_results(results["records"], results_csv)
    CSVExporter.export_category_summary(results["category_stats"], category_csv)

    results_url = results_csv.replace("\\", "/")
    category_url = category_csv.replace("\\", "/")

    print("--- Exported Artifacts ---")
    print(f"  Detailed Results CSV: [evaluation_results.csv](file:///{results_url})")
    print(f"  Category Summary CSV: [category_summary.csv](file:///{category_url})\n")

    ConsoleReporter.print_header("Evaluation Completed Successfully!")


if __name__ == "__main__":
    main()
