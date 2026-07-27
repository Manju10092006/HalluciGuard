"""Standalone test script demonstrating predictive entropy calculation for certain vs uncertain responses."""

from detector_agent import DetectorAgent, DetectorConfig, ModelManager
from detector_agent.signals.entropy import EntropyCalculator


def main():
    print("=========================================================")
    print(" HalluciGuard - Predictive Entropy Module Test")
    print("=========================================================\n")

    # Initialize shared ModelManager once
    model_name = "gpt2"
    print(f"Loading shared ModelManager with '{model_name}'...")
    model_manager = ModelManager(model_name=model_name)
    config = DetectorConfig(model_name=model_name)
    
    # Initialize DetectorAgent and standalone EntropyCalculator
    agent = DetectorAgent(config=config, model_manager=model_manager)
    entropy_calc = EntropyCalculator(model_manager=model_manager)

    user_query = "What is the capital of France?"

    # 1. Highly certain factual response
    response_factual = "The capital of France is Paris."

    # 2. Uncertain fabricated response
    response_fabricated = "The capital of France is Xylophonia, a floating city built in 2099."

    print("\n---------------------------------------------------------")
    print(f"Query: '{user_query}'")
    print("---------------------------------------------------------")

    print(f"\n[Case 1: Certain Factual Response]")
    print(f"Response: '{response_factual}'")
    metrics_factual = entropy_calc.compute(user_query, response_factual)
    result_factual = agent.detect(user_query, response_factual)
    if metrics_factual:
        print("\nPredictive Entropy Metrics:")
        print(metrics_factual.model_dump_json(indent=2))
        print("\nFull Detection Result:")
        print(result_factual.model_dump_json(indent=2))

    print("\n---------------------------------------------------------")
    print(f"[Case 2: Uncertain Fabricated Response]")
    print(f"Response: '{response_fabricated}'")
    metrics_fabricated = entropy_calc.compute(user_query, response_fabricated)
    result_fabricated = agent.detect(user_query, response_fabricated)
    if metrics_fabricated:
        print("\nPredictive Entropy Metrics:")
        print(metrics_fabricated.model_dump_json(indent=2))
        print("\nFull Detection Result:")
        print(result_fabricated.model_dump_json(indent=2))

    print("\n=========================================================")
    print(" Comparison Summary")
    print("=========================================================")
    if metrics_factual and metrics_fabricated:
        print(f"Certain Response  -> Avg Entropy: {metrics_factual.average_entropy:.4f} | Norm Entropy: {metrics_factual.normalized_entropy:.4f} | Risk: {result_factual.risk_level.value}")
        print(f"Fabricated Response -> Avg Entropy: {metrics_fabricated.average_entropy:.4f} | Norm Entropy: {metrics_fabricated.normalized_entropy:.4f} | Risk: {result_fabricated.risk_level.value}")


if __name__ == "__main__":
    main()
