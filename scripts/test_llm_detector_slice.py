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
from services.llm_detector_service import BaseLLMDetectorService


async def main() -> int:
    load_local_dotenv()
    
    query = "Explain OS in one full paragraph"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    print(f"Executing Step 2 Slice for user_query: '{query}'")
    print("-" * 60)

    service = BaseLLMDetectorService()
    result = await service.execute_slice(user_query=query)
    dumped = result.model_dump()

    print("\n--- STEP 2 ACCEPTANCE CONTRACT OUTPUT ---")
    print(json.dumps(dumped, indent=2))
    print("-" * 60)

    # Verification assertions
    assert result.user_query == query
    assert result.generation["status"] in {"success", "failed"}

    if result.generation["status"] == "success":
        assert result.draft_response.strip(), "Draft response should not be empty on success"
        assert result.detector is not None, "Detector should have executed for successful generation"
        assert "hallucination_probability" in result.detector
        assert "risk_tier" in result.detector
        assert "decision" in result.detector
        print("\nSTEP 2 SLICE TEST PASSED: LLM -> DETECTOR CONNECTED SUCCESSFULLY.")
        return 0
    else:
        print(f"\nSTEP 2 SLICE: LLM Generation failed cleanly. Reason: {result.generation.get('error')}")
        assert result.detector is None, "Detector must be None when LLM generation fails"
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
