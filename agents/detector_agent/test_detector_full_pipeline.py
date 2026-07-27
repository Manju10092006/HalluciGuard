"""End-to-end integration test for full 4-signal HalluciGuard Detector Agent pipeline."""

from detector_agent import DetectorAgent, DetectorConfig, ModelManager
from detector_agent.config import SignalWeights


def main():
    print("=========================================================")
    print(" HalluciGuard - Full 4-Signal Detector Agent Pipeline Test")
    print("=========================================================\n")

    model_name = "gpt2"
    print(f"Initializing shared ModelManager with '{model_name}'...")
    model_manager = ModelManager(model_name=model_name)

    # Configure equal weights across all 4 active signals (25% each)
    custom_weights = SignalWeights(
        token_probability=0.25,
        entropy=0.25,
        semantic_similarity=0.25,
        self_consistency=0.25
    )

    config = DetectorConfig(
        model_name=model_name,
        signal_weights=custom_weights,
        num_samples=3,
        max_new_tokens=25
    )

    agent = DetectorAgent(config=config, model_manager=model_manager)

    test_cases = [
        {
            "query": "What is the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "query": "Explain quantum computing simply.",
            "response": "Quantum computing uses qubits, superposition, and entanglement to solve complex mathematical problems."
        }
    ]

    for idx, case in enumerate(test_cases, 1):
        query = case["query"]
        response = case["response"]

        print(f"---------------------------------------------------------")
        print(f" Test Case {idx}")
        print(f" Query:    '{query}'")
        print(f" Response: '{response}'")
        print(f"---------------------------------------------------------")

        # Execute full 4-signal detection pipeline
        result = agent.detect(user_query=query, llm_response=response)

        print("\n--- Per-Signal Metrics Breakdown ---")
        if result.metrics:
            if result.metrics.token_probability:
                print("1. Token Probability Metrics:")
                print(f"   Avg Logprob: {result.metrics.token_probability.avg_logprob}")
                print(f"   Min Logprob: {result.metrics.token_probability.min_logprob}")
                print(f"   Low Conf Ratio: {result.metrics.token_probability.low_confidence_ratio}")

            if result.metrics.entropy:
                print("2. Predictive Entropy Metrics:")
                print(f"   Avg Entropy: {result.metrics.entropy.average_entropy} nats")
                print(f"   Max Entropy: {result.metrics.entropy.maximum_entropy} nats")
                print(f"   Norm Entropy: {result.metrics.entropy.normalized_entropy}")

            if result.metrics.semantic_similarity:
                print("3. Semantic Similarity Metrics:")
                print(f"   Cosine Sim: {result.metrics.semantic_similarity.cosine_similarity}")
                print(f"   Semantic Dist: {result.metrics.semantic_similarity.semantic_distance}")
                print(f"   Norm Sim: {result.metrics.semantic_similarity.normalized_similarity}")

            if result.metrics.self_consistency:
                print("4. Self-Consistency Metrics:")
                print(f"   Pairwise Sim: {result.metrics.self_consistency.pairwise_similarity}")
                print(f"   Consistency Score: {result.metrics.self_consistency.consistency_score}")
                print(f"   Response Variance: {result.metrics.self_consistency.response_variance}")

        print("\n--- Aggregated Pipeline Decision ---")
        print(f" Confidence Score:          {result.confidence_score}")
        print(f" Hallucination Probability: {result.hallucination_probability}")
        print(f" Risk Level:               {result.risk_level.value}")
        print(f" Next Action:              {result.next_action.value}\n")

        # Assertions to guarantee completeness
        assert result.confidence_score >= 0.0 and result.confidence_score <= 1.0
        assert result.hallucination_probability >= 0.0 and result.hallucination_probability <= 1.0
        assert result.metrics is not None
        assert result.metrics.token_probability is not None
        assert result.metrics.entropy is not None
        assert result.metrics.semantic_similarity is not None
        assert result.metrics.self_consistency is not None

    print("=========================================================")
    print(" Full Pipeline Integration Test Completed Successfully!")
    print("=========================================================")


if __name__ == "__main__":
    main()
