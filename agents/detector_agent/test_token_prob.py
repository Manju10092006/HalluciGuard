"""Test script demonstrating shared ModelManager and structured TokenProbabilityMetrics."""

from detector_agent import DetectorAgent, DetectorConfig, ModelManager


def main():
    print("Initializing shared ModelManager...")
    # Initialize ModelManager once
    model_manager = ModelManager(model_name="gpt2")

    # Pass shared ModelManager to DetectorAgent
    config = DetectorConfig(model_name="gpt2")
    agent = DetectorAgent(config=config, model_manager=model_manager)

    user_query = "What is the capital of France?"
    llm_response = "The capital of France is Paris."

    print(f"\nUser Query: {user_query}")
    print(f"LLM Response: {llm_response}")

    print("\n--- 1. Testing _compute_token_probability() output ---")
    metrics = agent._compute_token_probability(user_query, llm_response)
    if metrics:
        print("TokenProbabilityMetrics object:")
        print(metrics.model_dump_json(indent=2))
    else:
        print("Could not compute token probability metrics.")

    print("\n--- 2. Testing full detect() API output ---")
    result = agent.detect(user_query=user_query, llm_response=llm_response)
    print("DetectionResult object:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
