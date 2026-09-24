"""Prove the REAL orchestration graph runs judge -> corrector -> reverifier.

Runs the full production pipeline (run_verification: generate -> detector ->
claim_analyzer -> verifier -> judge -> corrector -> reverifier -> memory) on a
hallucinated draft and prints the execution TRACE plus the judge/corrector
outputs, so we can see the two agents actually fire in the live graph.

Run:  python scripts/prove_orchestration_judge_corrector.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from orchestration.graph import run_verification

QUERY = "Who is the founder of Microsoft?"
DRAFT = "Snehith is the founder of Microsoft."


async def main() -> None:
    result = await run_verification(QUERY, DRAFT, domain="general")

    trace = result.get("trace", [])
    nodes = [t.get("node") for t in trace]
    print("\n===== EXECUTION TRACE (real graph) =====")
    print(" -> ".join(str(n) for n in nodes))

    judge = result.get("judge") or result.get("judge_result") or {}
    corr = result.get("corrector") or result.get("correction_result") or {}
    rev = result.get("reverification_result") or {}

    print("\n===== JUDGE fired? =====")
    print("ran        :", "judge" in nodes)
    print("decision   :", judge.get("decision"))
    print("basis      :", judge.get("decision_basis"))
    print("metrics    :", json.dumps(judge.get("decision_metrics", {}), default=str))

    print("\n===== CORRECTOR fired? =====")
    print("ran        :", "corrector" in nodes)
    print("status     :", corr.get("status"))
    print("provider   :", corr.get("provider_used"))
    print("before     :", corr.get("original_text"))
    print("after      :", corr.get("corrected_text"))

    print("\n===== REVERIFIER =====")
    print("ran        :", "reverifier" in nodes)
    print("passed     :", rev.get("passed"), "| failure_category:", rev.get("failure_category"))

    print("\n===== FINAL =====")
    print("final_response  :", result.get("final_response"))
    print("verification    :", result.get("verification_status"))
    print("terminal_status :", result.get("terminal_status"))
    print("active_agents   :", result.get("active_agents"))
    print("disabled_agents :", result.get("disabled_agents"))


if __name__ == "__main__":
    asyncio.run(main())
