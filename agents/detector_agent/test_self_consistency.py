"""Standalone test script demonstrating the Self-Consistency signal module."""

from detector_agent import DetectorConfig, ModelManager
from detector_agent.signals.self_consistency import SelfConsistencyCalculator


def main():
    print("=========================================================")
    print(" HalluciGuard - Self-Consistency Signal Module Test")
    print("=========================================================\n")

    # Initialize shared ModelManager once
    model_name = "gpt2"
    print(f"Loading shared ModelManager with '{model_name}'...")
    model_manager = ModelManager(model_name=model_name)
    
    # Configure self-consistency sampling parameters via DetectorConfig
    config = DetectorConfig(
        model_name=model_name,
        num_samples=3,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=30
    )
    
    calculator = SelfConsistencyCalculator(model_manager=model_manager, config=config)

    test_prompts = [
        "What is the capital of France?",
        "Explain why the sky is blue in one sentence.",
        "Tell me a short rumor about an imaginary planet called Zalthor."
    ]

    for idx, prompt in enumerate(test_prompts, 1):
        print(f"\n---------------------------------------------------------")
        print(f" Prompt {idx}: '{prompt}'")
        print(f"---------------------------------------------------------")

        # Generate candidate responses and compute self-consistency metrics
        metrics, responses, matrix = calculator.compute(user_query=prompt)

        print("\nGenerated Candidate Responses:")
        for r_idx, resp in enumerate(responses, 1):
            print(f"  [{r_idx}] {resp}")

        print("\nPairwise Cosine Similarity Matrix:")
        n = len(responses)
        header = "       " + " ".join([f"  R{i+1}   " for i in range(n)])
        print(header)
        for i in range(n):
            row_str = f"  R{i+1}  |" + " ".join([f"{matrix[i][j]:8.4f}" for j in range(n)])
            print(row_str)

        if metrics:
            print("\nFinal Self-Consistency Metrics:")
            print(metrics.model_dump_json(indent=2))

    print("\n=========================================================")
    print(" Self-Consistency Test Completed Successfully!")
    print("=========================================================")


if __name__ == "__main__":
    main()
