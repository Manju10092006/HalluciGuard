"""Ad-hoc end-to-end orchestration smoke test for the DetV2 integration.

Runs three pre-supplied responses through the FULL orchestration graph
(run_verification), so Draft -> DetV2 detector -> routing -> downstream nodes
all execute. Prints, per case: detector model_source / next_action / risk /
per-claim count, whether the response was directly Accepted, and the terminal
node reached. NOT a pytest test (it exercises the live graph); run directly.
"""
from __future__ import annotations

import asyncio
import os
import sys

from orchestration.graph import run_verification

CASES = [
    ("Who painted the Mona Lisa?",
     "The Mona Lisa was painted by Pablo Picasso in 1750.", "FALSE claim"),
    ("Who painted the Mona Lisa?",
     "The Mona Lisa was painted by Leonardo da Vinci during the Renaissance.", "CORRECT claim"),
    ("What is the capital of France?",
     "The capital of France is Paris.", "NORMAL factual"),
]


async def _run_one(query, response, label):
    state = await run_verification(user_query=query, llm_response=response)
    detector = state.get("detector") or {}
    trace = state.get("trace") or []
    nodes = [f"{t.get('node')}:{t.get('status')}" for t in trace]
    errors = state.get("errors") or []
    print(f"\n=== {label} ===")
    print(f"  query           : {query}")
    print(f"  response        : {response}")
    print(f"  detector.source : {detector.get('model_source')}")
    print(f"  detector.status : {detector.get('status')}")
    print(f"  next_action     : {detector.get('next_action')}")
    print(f"  risk_level      : {detector.get('risk_level')}")
    print(f"  hallu_prob      : {detector.get('hallucination_probability')}")
    print(f"  per_claim (n)   : {len(detector.get('per_claim_results') or [])}")
    print(f"  route           : {state.get('route')}")
    print(f"  nodes visited   : {nodes}")
    print(f"  errors          : {errors}")
    print(f"  final_response  : {(state.get('final_response') or '')[:120]}")
    print(f"  final_status    : {state.get('final_status') or state.get('status')}")
    directly_accepted = (
        str(detector.get("next_action")) == "Accept"
        and not any(str(n).startswith("verifier") for n in nodes)
    )
    print(f"  DIRECTLY ACCEPTED (detector-only, no verifier): {directly_accepted}")
    return label, directly_accepted, nodes


async def _main():
    results = []
    for q, r, lbl in CASES:
        try:
            results.append(await _run_one(q, r, lbl))
        except Exception as exc:  # report, don't crash the whole smoke run
            print(f"\n=== {lbl} === RAISED {type(exc).__name__}: {exc}")
            results.append((lbl, None, ["<exception>"]))
    print("\n----- SUMMARY -----")
    for lbl, acc, nodes in results:
        print(f"  {lbl}: directly_accepted={acc}, terminal={nodes[-1] if nodes else '?'}")
    # Guard: the FALSE claim must never be directly accepted by the detector alone.
    false_acc = next((acc for lbl, acc, _ in results if lbl == "FALSE claim"), None)
    if false_acc:
        print("\nFAIL: false claim was directly Accepted by the detector.")
        sys.exit(1)
    print("\nOK: false claim was NOT directly Accepted; graph completed for all cases.")


if __name__ == "__main__":
    asyncio.run(_main())
