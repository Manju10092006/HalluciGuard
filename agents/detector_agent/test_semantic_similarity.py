"""Standalone test script evaluating Semantic Similarity across correct, partially relevant, and unrelated responses."""

from detector_agent import DetectorAgent, DetectorConfig, ModelManager
from detector_agent.signals.semantic_similarity import SemanticSimilarityCalculator


def main():
    print("=========================================================")
    print(" HalluciGuard - Semantic Similarity Signal Module Test")
    print("=========================================================\n")

    # Initialize shared ModelManager once
    model_name = "gpt2"
    print(f"Loading shared ModelManager with '{model_name}' and SentenceTransformer 'all-MiniLM-L6-v2'...")
    model_manager = ModelManager(model_name=model_name)
    config = DetectorConfig(model_name=model_name)
    
    agent = DetectorAgent(config=config, model_manager=model_manager)
    semantic_calc = SemanticSimilarityCalculator(model_manager=model_manager)

    user_query = "What is the capital of France?"

    # 3 Test Scenarios:
    responses = {
        "Correct Answer": "The capital of France is Paris.",
        "Partially Relevant Answer": "France is a European country known for its rich culture, history, and famous landmarks.",
        "Unrelated Answer": "Quantum computing leverages superposition and entanglement to perform complex computations."
    }

    print(f"\nUser Query: '{user_query}'\n")

    for label, response_text in responses.items():
        print("---------------------------------------------------------")
        print(f"[{label}]")
        print(f"Response: '{response_text}'")
        
        metrics = semantic_calc.compute(user_query, response_text)
        result = agent.detect(user_query, response_text)

        if metrics:
            print("\nSemantic Similarity Metrics:")
            print(metrics.model_dump_json(indent=2))
            print("\nFull Detection Result:")
            print(result.model_dump_json(indent=2))

    print("\n=========================================================")
    print(" Summary Comparison")
    print("=========================================================")
    for label, response_text in responses.items():
        m = semantic_calc.compute(user_query, response_text)
        res = agent.detect(user_query, response_text)
        if m and res:
            print(f"{label:27s} -> Cosine Sim: {m.cosine_similarity:6.4f} | Distance: {m.semantic_distance:6.4f} | Risk: {res.risk_level.value}")


if __name__ == "__main__":
    main()
