from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestration.graph import run_verification
from orchestration.runtime_validation import validate_orchestration_startup


async def main() -> None:
    """
    Run end-to-end verification tests for the orchestration pipeline.

    Tests two scenarios:
    1. A hallucinated response (Tokyo as capital of France)
    2. A correct response (Paris as capital of France)

    Outputs runtime validation, trace execution path, and agent results for each test case.
    """
    validation = validate_orchestration_startup()
    print("RUNTIME VALIDATION:", json.dumps(validation, indent=2, default=str))
    cases = [
        ("What is the capital of France?", "The capital of France is Tokyo."),
        ("What is the capital of France?", "The capital of France is Paris."),
    ]

    for query, response in cases:
        result = await run_verification(query, response, domain="general")
        trace = result.get("trace", [])
        nodes = [entry.get("node") for entry in trace]
        print("=" * 80)
        print(f"QUERY: {query}")
        print(f"DRAFT: {response}")
        print(f"FINAL: {result.get('final_response')}")
        print(f"RETRIES: {result.get('retry_count', 0)}")
        print("TRACE:", " -> ".join(str(node) for node in nodes))
        print(
            json.dumps(
                {
                    "detector": result.get("detector"),
                    "verifier_claims": len(
                        (result.get("verifier") or {}).get("claim_evidence", [])
                    ),
                    "judge_decision": (result.get("judge") or {}).get("decision"),
                    "corrector_called": bool(result.get("corrector")),
                    "memory_records": (result.get("memory") or {}).get("count", 0),
                    "errors": result.get("errors", []),
                },
                indent=2,
                default=str,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
