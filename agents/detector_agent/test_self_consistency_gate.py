import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from detector_agent.utils import init_dll_paths
init_dll_paths()

from detector_agent import DetectorAgent, DetectorConfig, ModelManager, RiskLevel
from detector_agent.classifier import classify_query
from detector_agent.gate import SelfConsistencyGate


def main():
    print("=========================================================")
    print(" HalluciGuard - Intelligent Gating Mechanism Test")
    print("=========================================================\n")

    gate = SelfConsistencyGate()

    test_scenarios = [
        {
            "label": "Factual Medium-Risk Query",
            "query": "What is the capital of France?",
            "risk_level": RiskLevel.MEDIUM
        },
        {
            "label": "Factual Low-Risk Query",
            "query": "What is the capital of France?",
            "risk_level": RiskLevel.LOW
        },
        {
            "label": "Creative Prompt",
            "query": "Write a creative story about a magical flying car.",
            "risk_level": RiskLevel.MEDIUM
        },
        {
            "label": "Storytelling Prompt",
            "query": "Tell me a story about a brave knight in a golden castle.",
            "risk_level": RiskLevel.MEDIUM
        },
        {
            "label": "Reasoning Prompt",
            "query": "Explain step-by-step how to solve a quadratic equation.",
            "risk_level": RiskLevel.MEDIUM
        }
    ]

    print("--- 1. Testing Standalone Gate Decision Rules ---\n")

    for scenario in test_scenarios:
        label = scenario["label"]
        query = scenario["query"]
        risk = scenario["risk_level"]

        category = classify_query(query)
        should_run, log_reason, category = gate.should_run_self_consistency(query, risk)

        print(f"Scenario:      [{label}]")
        print(f"Query:         '{query}'")
        print(f"Category:      {category.value}")
        print(f"Initial Risk:  {risk.value}")
        print(f"Gate Decision: {'EXECUTE' if should_run else 'SKIP'}")
        print(f"Reason:        {log_reason}")
        print("-" * 55)

    print("\n--- 2. Testing End-to-End DetectorAgent Pipeline Gating ---\n")

    model_name = "gpt2"
    model_manager = ModelManager(model_name=model_name)
    config = DetectorConfig(model_name=model_name, num_samples=2, max_new_tokens=20)
    agent = DetectorAgent(config=config, model_manager=model_manager)

    agent_scenarios = [
        {
            "query": "What is the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "query": "Write a poem about a sleeping cat.",
            "response": "Soft purrs in the dark, curling tight in the warm park."
        }
    ]

    for case in agent_scenarios:
        q = case["query"]
        r = case["response"]

        print(f"Query: '{q}'")
        print(f"Response: '{r}'")
        result = agent.detect(q, r)
        self_consist_status = "Executed" if (result.metrics and result.metrics.self_consistency) else "Skipped"
        print(f"Self-Consistency Status: {self_consist_status}")
        print(f"Final Risk Level:        {result.risk_level.value}\n")

    print("=========================================================")
    print(" Gating Mechanism Tests Completed Successfully!")
    print("=========================================================")


if __name__ == "__main__":
    main()
