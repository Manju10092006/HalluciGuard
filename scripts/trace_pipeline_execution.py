"""Bounded real-model pipeline diagnostic; no training or production changes.

external mode uses configured live retrieval. controlled mode supplies one
explicit, cited government excerpt (or no passages), not mocked NLI/verdicts.
Credentials are loaded in-process; output omits prompts/provider responses/errors.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import time


async def execute(args):
    from orchestration import graph
    Pipeline, Claim, Input = graph._get_verifier_imports()
    from schemas.models import Passage
    from schemas.retrieval_trace import RetrievalTrace
    government = Passage(title="Hanoi Investment Map", source="government",
        source_id="hanoi-government-capital", url="https://hanoiinvestment.hanoi.gov.vn/en",
        publication_date="unknown",
        snippet="Ha Noi is the capital of the Socialist Republic of Vietnam and the national political and administrative center.")
    class Adapter:
        name = "controlled-government-excerpt"
        last_retrieval_trace = None
        async def search(self, query, **kwargs):
            self.last_retrieval_trace = RetrievalTrace(primary_adapter=self.name)
            return [] if args.case == "insufficient" else [government]
    class ControlledPipeline(Pipeline):
        def __init__(self):
            super().__init__()
            self.settings = self.settings.model_copy(update={"n8n_retrieval_enabled": False})
            self.cache_enabled = False
    query = args.query or "What is the capital of Vietnam?"
    draft = {"supported": "The capital of Vietnam is Hanoi.",
             "contradicted": "The capital of Vietnam is Bangkok.",
             "insufficient": "Vietnam has a secret capital on Mars."}[args.case]
    if args.draft:
        draft = args.draft
    start = time.perf_counter()
    if args.mode == "controlled":
        adapter = Adapter()
        with patch("api.pipeline.get_registry", return_value=SimpleNamespace(get_adapter=lambda d: adapter)), \
             patch.object(graph, "_get_verifier_imports", return_value=(ControlledPipeline, Claim, Input)):
            result = await graph.run_verification(query, draft)
    else:
        result = await graph.run_verification(query, draft if args.draft else ("" if args.case == "supported" else draft))
    # Keep authoritative execution trace/counts, never raw provider payloads.
    verifier = result.get("verifier_result") or {}
    rev = result.get("reverification_result") or {}
    def reports(v):
        return [{"claim_id": r.get("claim_id"), "claim_text": r.get("claim_text"), "verdict": r.get("verdict"),
                 "evidence_count": len(r.get("evidence", [])),
                 "evidence": [{k: e.get(k) for k in ("source", "source_id", "url", "snippet", "entailment_label")}
                              for e in r.get("evidence", [])],
                 "retrieval_trace": r.get("retrieval_trace")} for r in v.get("claim_reports", [])]
    output = {"mode": args.mode, "case": args.case, "seconds": time.perf_counter()-start,
        "terminal_status": result.get("terminal_status"), "verification_status": result.get("verification_status"),
        "judge_decision": result.get("judge_decision"), "stage_trace": result.get("trace"),
        "original_reports": reports(verifier), "reverification_reports": reports(rev.get("verifier_result") or {}),
        "corrector_status": (result.get("correction_result") or {}).get("status"),
        "corrected_text": (result.get("correction_result") or {}).get("corrected_text"),
        "correction_attempts": result.get("correction_attempt_count"), "reverification_passed": rev.get("passed"),
        "reverification_failure_category": rev.get("failure_category"),
        "final_response": result.get("final_response"),
        "memory_status": (result.get("memory_result") or {}).get("status"),
        "memory_stored_count": (result.get("memory_result") or {}).get("stored_count")}
    detector = result.get("detector_result") or result.get("detector") or {}
    output["grounded_detector_execution"] = {
        k: detector.get(k) for k in ("status", "model_loaded", "inference_executed",
            "detector_degraded", "calibration_applied", "probability_semantics")}
    output["memory_isolation"] = str(args.output.parent / (args.output.stem + "-memory"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k:v for k,v in output.items() if k not in {"stage_trace", "original_reports", "reverification_reports"}}, indent=2))


def isolate_runtime(output: Path):
    """Use a fresh diagnostic store; never touch previously quarantined Memory."""
    directory = output.parent / (output.stem + "-memory")
    directory.mkdir(parents=True, exist_ok=False)
    for key, name in {
        "KG_PERSISTENCE_PATH": "knowledge_graph.json", "CACHE_DB_PATH": "memory-cache.db",
        "VECTOR_STORE_PATH": "vector_store", "PATTERN_DB_PATH": "patterns.db",
        "TRUST_DB_PATH": "source-trust.db",
    }.items():
        os.environ[key] = str((directory / name).resolve())
    os.environ["CACHE_ENABLED"] = "false"
    os.environ["VERIFIER_CACHE_ENABLED"] = "false"
    os.environ["ALWAYS_VERIFY"] = "true"
    os.environ["ALLOW_DETECTOR_FAST_PATH"] = "false"
    os.environ["ALLOW_MODEL_DOWNLOADS"] = "false"
    os.environ["MOCK_MODE"] = "false"
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--mode", choices=["external", "controlled"], required=True)
    parser.add_argument("--case", choices=["supported", "contradicted", "insufficient"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query")
    parser.add_argument("--draft")
    args = parser.parse_args()
    if args.mode == "controlled" and (args.query or args.draft):
        parser.error("custom queries require external mode; the controlled fixture is Vietnam-specific")
    if args.output.exists():
        raise FileExistsError("trace artifact already exists")
    from dotenv import load_dotenv
    load_dotenv(args.env_file)
    isolate_runtime(args.output)
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
