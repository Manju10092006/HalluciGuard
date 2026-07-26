"""
HalluciGuard - Judge Agent Input Simulator & Scenario Harness
Allows testing the Judge Agent using arbitrary ChatGPT/User queries and draft responses.
Simulates Detector & Verifier inputs across realistic Hallucination scenarios.
"""

import json
from typing import Dict, Any
from judge_agent import JudgeAgent

def run_chatgpt_simulation(
    user_query: str,
    chatgpt_draft_response: str,
    verifier_evidence: str,
    domain: str = "Healthcare",
    detector_hallucination_prob: float = 0.40
):
    """
    Simulates a full pipeline run taking user input / ChatGPT response + Verifier evidence.
    """
    print("=" * 80)
    print(f"  HALLUCIGUARD JUDGE AGENT - SIMULATION RUN ({domain.upper()})")
    print("=" * 80)
    print(f"USER QUERY     : {user_query}")
    print(f"DRAFT RESPONSE : {chatgpt_draft_response}")
    print(f"VERIFIER EVID. : {verifier_evidence}")
    print("-" * 80)

    # Format simulated Detector Output
    detector_output = {
        "hallucination_probability": detector_hallucination_prob,
        "confidence_score": 0.85,
        "risk_level": "HIGH" if detector_hallucination_prob > 0.5 else "LOW",
        "suspicious_claims": [chatgpt_draft_response]
    }

    # Format simulated Verifier Output
    verifier_output = {
        "domain": domain,
        "claim_evidence_pairs": [
            {
                "claim": chatgpt_draft_response,
                "evidence": verifier_evidence,
                "evidence_confidence": 0.90,
                "rank": 1,
                "source": f"Enterprise Verified KB ({domain})"
            }
        ]
    }

    agent = JudgeAgent()
    result = agent.evaluate(
        detector_output=detector_output,
        verifier_output=verifier_output,
        user_query=user_query,
        draft_response=chatgpt_draft_response
    )

    print("\n>>> JUDGE AGENT EVALUATION RESULT:")
    print(f"DECISION              : {result['decision']}")
    print(f"SEVERITY              : {result['severity']}")
    print(f"CALIBRATED CONFIDENCE : {result['metrics']['calibrated_confidence']:.2f}")
    print(f"CONTRADICTION INDEX   : {result['metrics']['overall_contradiction']:.2f}")
    print(f"REASON                : {result['reason']}")
    print(f"EXPLANATION           : {result['explanation']}")
    print(f"NEXT ACTION           : {result['next_action']}")

    print("\n>>> OUTPUT PAYLOAD PREPARED FOR DOWNSTREAM CORRECTOR AGENT:")
    print(json.dumps(result["corrector_payload"], indent=2))
    print("=" * 80 + "\n")
    return result


def run_prebuilt_test_scenarios():
    """
    Runs 4 standard enterprise scenarios:
    1. Factually Accurate Response (Accept)
    2. Medical Hallucination / Contradiction (Correct)
    3. Low Evidence / Borderline (Verify Again)
    4. Unsafe / Critical Contradiction (Reject)
    """
    scenarios = [
        {
            "name": "Scenario 1: Factually Accurate Financial Answer",
            "query": "What was Apple's total revenue in FY 2023?",
            "draft": "Apple Inc. reported a total net sales revenue of $383.29 billion for its fiscal year 2023.",
            "evidence": "Apple Inc. reported annual revenue of $383.29 billion in fiscal year 2023 according to SEC EDGAR filings.",
            "domain": "Finance",
            "prob": 0.10
        },
        {
            "name": "Scenario 2: Healthcare Claim Contradiction (Hallucination)",
            "query": "What is the recommended pediatric dosage for Aspirin in viral fever?",
            "draft": "Aspirin is the recommended first-line fever reducer for children under 12 with chickenpox or flu.",
            "evidence": "Aspirin is strictly contraindicated in children and teenagers recovering from chickenpox or flu-like symptoms due to the risk of Reye's syndrome.",
            "domain": "Healthcare",
            "prob": 0.85
        },
        {
            "name": "Scenario 3: Cybersecurity Low Evidence Baseline",
            "query": "Does CVE-2024-9999 affect Linux Kernel 6.5?",
            "draft": "CVE-2024-9999 allows remote code execution in Linux kernel version 6.5.",
            "evidence": "CVE-2024-9999 search returned no verified exploit reports in kernel version 6.5.",
            "domain": "Cybersecurity",
            "prob": 0.50
        }
    ]

    print("\n Running All Pre-built HalluciGuard Simulation Scenarios...\n")
    for sc in scenarios:
        print(f"--- {sc['name']} ---")
        run_chatgpt_simulation(
            user_query=sc["query"],
            chatgpt_draft_response=sc["draft"],
            verifier_evidence=sc["evidence"],
            domain=sc["domain"],
            detector_hallucination_prob=sc["prob"]
        )


if __name__ == "__main__":
    run_prebuilt_test_scenarios()
