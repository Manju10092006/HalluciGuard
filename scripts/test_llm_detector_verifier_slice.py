from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.test_openrouter_llm import load_local_dotenv
from services.llm_detector_verifier_service import BaseLLMDetectorVerifierService


async def run_test_case(title: str, query: str, force_verifier: bool = False) -> dict:
    print(f"\n============================================================")
    print(f"TEST CASE: {title}")
    print(f"Query: '{query}' | force_verifier: {force_verifier}")
    print(f"============================================================")

    service = BaseLLMDetectorVerifierService()
    result = await service.execute_slice(
        user_query=query,
        force_verifier=force_verifier,
    )
    dumped = result.model_dump()

    print("\n--- STEP 3 ACCEPTANCE CONTRACT OUTPUT ---")
    print(json.dumps(dumped, indent=2))

    # Print clean summary
    print("\n--- SUMMARY OF EXECUTION ---")
    print(f"Question:              {result.user_query}")
    print(f"Draft Response:        {result.draft_response[:100]}...")
    if result.detector:
        print(f"Detector Probability:  {result.detector.get('hallucination_probability')}")
        print(f"Risk Tier:             {result.detector.get('risk_tier')}")
        print(f"Detector Decision:     {result.detector.get('decision')}")
    else:
        print("Detector:              SKIPPED (LLM Generation Failed)")

    if result.verifier:
        print(f"Verifier Executed:     {result.verifier.get('executed')}")
        if result.verifier.get("executed"):
            print(f"Retrieved Sources:     {result.verifier.get('retrieved_sources')}")
            print(f"Verified Sources:      {result.verifier.get('verified_sources')}")
            print(f"Evidence Confidence:   {result.verifier.get('overall_evidence_confidence')}")
            claims = result.verifier.get("claim_evidence", [])
            print(f"Claims Evaluated:      {len(claims)}")
            for idx, c in enumerate(claims, 1):
                print(f"  Claim {idx} Verdict:  {c.get('verdict')}")
                print(f"  Evidence Count:     {len(c.get('evidence', []))}")
                for ev in c.get("evidence", []):
                    print(f"    - [{ev.get('source')}] {ev.get('title')} ({ev.get('url')}) -> NLI: {ev.get('entailment_label')} ({ev.get('entailment_score')})")
        else:
            print(f"Verifier Reason:       {result.verifier.get('reason')}")
    else:
        print("Verifier:              NONE")

    return dumped


async def main() -> int:
    load_local_dotenv()

    # Case 1: Normal Low-Risk Question (LOW -> ACCEPT -> Verifier SKIPPED)
    res1 = await run_test_case(
        title="1. Low-Risk Standard Question (Expected: Verifier SKIPPED)",
        query="What is the capital of France?",
        force_verifier=False,
    )
    assert res1["detector"]["risk_tier"] == "LOW" or res1["verifier"]["executed"] is False

    # Case 2: Forced Verifier / High Risk Question (Expected: Verifier EXECUTED)
    res2 = await run_test_case(
        title="2. High-Risk / Forced Verification Question (Expected: Verifier EXECUTED)",
        query="What are the key memory management functions of an operating system kernel?",
        force_verifier=True,
    )
    assert res2["verifier"]["executed"] is True, "Verifier should execute when force_verifier=True or HIGH risk"

    print("\n" + "=" * 60)
    print("ALL STEP 3 LIVE TEST CASES PASSED SUCCESSFULLY!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
