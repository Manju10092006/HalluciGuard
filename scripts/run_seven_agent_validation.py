"""Observe the unmodified production graph and export real seven-agent results.

No mocks, evidence injection, training or routing overrides. --draft bypasses
initial generation only. --suite runs the two requested cases serially. Output
directories must be new: checkpoints, prior reports and production Memory are
never overwritten. FIRED means an actual agent call, not configured membership.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import contextvars
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import functools
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AGENTS = (
    ("DETECTOR", ("detector", "grounded_detector"), "Risk triage; grounded inference requires evidence. Not factual authority."),
    ("CLAIM_ANALYZER", ("claim_analyzer",), "Decomposes the draft into factual claims with IDs and searches."),
    ("VERIFIER", ("verifier",), "Retrieves and ranks evidence; computes claim-level evidence verdicts."),
    ("JUDGE", ("judge",), "Chooses the action from actual verification results; may execute again after repair."),
    ("CORRECTOR", ("corrector",), "Produces an untrusted repair only when production routing requests one."),
    ("REVERIFIER", ("reverifier",), "Re-extracts and verifies corrected claims before release."),
    ("MEMORY", ("memory",), "Persists only eligible accepted verified facts; isolated diagnostic stores here."),
)
NODES = ("generate", "detector", "claim_analyzer", "verifier", "grounded_detector", "judge", "corrector", "reverifier", "memory", "accept", "reject", "human_escalation")
CURRENT_NODE = contextvars.ContextVar("validation_node", default="outside_graph")
OMIT_HANDOFF = {"trace", "inter_agent_bus", "audit", "active_agents", "disabled_agents", "agent_outcomes"}


def plain(value):
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return plain(value.model_dump())
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_value": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def scrub(value, secrets=()):
    """Redact credentials only; retain claims, evidence, model scores and errors."""
    value = plain(value)
    if isinstance(value, dict):
        return {k: "[REDACTED]" if re.search(r"api_key|password|authorization|access_token|webhook_secret|client_secret", k, re.I)
                else scrub(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in sorted(secrets, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)([?&](?:api_key|key|token|secret)=)[^&\s]+", r"\1[REDACTED]", value)
        return re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._-]+", r"\1[REDACTED]", value)
    return value


def pretty(value):
    return json.dumps(plain(value), indent=2, ensure_ascii=False, allow_nan=False)


class Recorder:
    def __init__(self, secrets=()):
        self.events = []
        self.secrets = secrets
        self.lock = threading.Lock()

    def start(self, kind, name, inputs):
        with self.lock:
            event = {"sequence": len(self.events) + 1, "kind": kind, "name": name,
                     "node": CURRENT_NODE.get(), "started_utc": datetime.now(timezone.utc).isoformat(),
                     "input": scrub(inputs, self.secrets), "status": "STARTED"}
            self.events.append(event)
        return event

    def finish(self, event, start, output=None, error=None, diagnostics=None):
        event["seconds"] = time.perf_counter() - start
        event["status"] = "FAILED" if error else "RETURNED"
        if error:
            event["error_type"] = type(error).__name__
        else:
            event["output"] = scrub(output, self.secrets)
        if diagnostics is not None:
            event["diagnostics"] = scrub(diagnostics, self.secrets)

    def wrap(self, name, original, *, kind="component", method=False, diagnostics=False):
        signature = inspect.signature(original)

        def begin(args, kwargs):
            try:
                inputs = dict(signature.bind(*args, **kwargs).arguments)
                if method:
                    inputs.pop("self", None)
                    inputs.pop("cls", None)
            except TypeError:
                inputs = {"args": args[1:] if method else args, "kwargs": kwargs}
            return self.start(kind, name, inputs)

        def diagnostic(args):
            if diagnostics and args and hasattr(args[0], "diagnostics"):
                try:
                    return args[0].diagnostics()
                except Exception as exc:
                    return {"observer_diagnostic_error_type": type(exc).__name__, "status": "unobserved"}
            return None

        if inspect.iscoroutinefunction(original):
            @functools.wraps(original)
            async def async_call(*args, **kwargs):
                event, start = begin(args, kwargs), time.perf_counter()
                try:
                    output = await original(*args, **kwargs)
                except BaseException as exc:
                    self.finish(event, start, error=exc)
                    raise
                self.finish(event, start, output=output, diagnostics=diagnostic(args))
                return output
            return async_call

        @functools.wraps(original)
        def sync_call(*args, **kwargs):
            event, start = begin(args, kwargs), time.perf_counter()
            try:
                output = original(*args, **kwargs)
            except BaseException as exc:
                self.finish(event, start, error=exc)
                raise
            self.finish(event, start, output=output, diagnostics=diagnostic(args))
            return output
        return sync_call

    def node_wrapper(self, name, original):
        wrapped = self.wrap(name, original, kind="node")

        async def invoke(state):
            token = CURRENT_NODE.set(name)
            try:
                output = wrapped(state)
                return await output if inspect.isawaitable(output) else output
            finally:
                CURRENT_NODE.reset(token)
        return invoke


def observe_optional_nli_input(stack, recorder, engine):
    """Capture a private model entry point only when this engine exposes it."""
    if hasattr(engine, "_invoke"):
        stack.enter_context(patch.object(engine, "_invoke", recorder.wrap(
            "NLIEngine._invoke", engine._invoke, method=True)))


def install_observers(stack, recorder, graph):
    """Transparent local wrappers: original methods called once, outputs unchanged."""
    Pipeline, _, _ = graph._get_verifier_imports()
    from services.base_llm_service import BaseLLMService
    from services.claim_analyzer import ClaimAnalyzer
    from services.character_regenerator import CharacterRegenerator
    from services.n8n_retrieval_client import N8NRetrievalClient
    from agents.memory_agent.memory.memory_agent import MemoryAgent
    from retrievers import HybridRetriever
    from retrievers.sparse import BM25Retriever
    from retrievers.dense import DenseRetriever
    from rerankers import CrossEncoderReranker
    from nli import NLIEngine
    from adapters.registry import get_registry
    from services import evidence_ranker
    targets = (
        (BaseLLMService, "generate", False), (ClaimAnalyzer, "analyze", False),
        (Pipeline, "verify", False), (HybridRetriever, "retrieve", True),
        (BM25Retriever, "retrieve", True), (DenseRetriever, "retrieve", True),
        (CrossEncoderReranker, "rerank", True), (CrossEncoderReranker, "score_gate_candidates", True),
        (NLIEngine, "batch_classify", True), (N8NRetrievalClient, "retrieve_evidence", False),
        (CharacterRegenerator, "regenerate", False), (MemoryAgent, "initialize", False),
        (MemoryAgent, "store_facts_batch", False),
    )
    for cls, method, diag in targets:
        stack.enter_context(patch.object(cls, method, recorder.wrap(f"{cls.__name__}.{method}", getattr(cls, method), method=True, diagnostics=diag)))
    # Older production NLI engines do not expose their private tokenizer entry
    # point. Observe it only when available; the public batch call remains
    # mandatory. An absent private hook is not an inference failure and cannot
    # establish token-level survival.
    observe_optional_nli_input(stack, recorder, NLIEngine)
    stack.enter_context(patch.object(evidence_ranker, "rank_and_select", recorder.wrap("evidence_ranker.rank_and_select", evidence_ranker.rank_and_select)))
    # Capture actual adapter queries/results, not only the last query's trace.
    registry = get_registry()
    original_get = registry.get_adapter
    wrapped_adapters = set()

    def get_adapter(*args, **kwargs):
        adapter = original_get(*args, **kwargs)
        if id(adapter) not in wrapped_adapters:
            wrapped_adapters.add(id(adapter))
            stack.enter_context(patch.object(adapter, "search", recorder.wrap(f"adapter:{adapter.name}.search", adapter.search)))
        return adapter
    stack.enter_context(patch.object(registry, "get_adapter", get_adapter))


def node_events(run, nodes):
    return [e for e in run["events"] if e["kind"] == "node" and e["name"] in nodes]


def agent_status(run, label, nodes):
    calls = node_events(run, nodes)
    if not calls:
        # Conditional stages are NOT_REQUIRED only if an actual Judge ran and
        # production chose a route that did not request this work.
        judges = [e for e in node_events(run, ("judge",)) if e["status"] == "RETURNED"
                  and not any(t.get("node") == "judge" and t.get("status") == "failed"
                              for t in e.get("output", {}).get("trace", []))]
        if label in {"CORRECTOR", "REVERIFIER"} and judges:
            requested = any(e.get("output", {}).get("correction_request") for e in judges)
            if not requested:
                return "NOT_REQUIRED"
        return "NOT_FIRED"
    for e in calls:
        if e["status"] != "RETURNED":
            return "FAILED"
        if label == "DETECTOR":
            detector = e.get("output", {}).get("detector_result") or e.get("output", {}).get("detector") or {}
            if (str(detector.get("status", "")).lower() in {"failed", "degraded", "unavailable", "error"}
                    or detector.get("detector_degraded") is True or detector.get("degraded") is True):
                return "FAILED"
        for trace in e.get("output", {}).get("trace", []):
            if trace.get("node") in (*nodes, "detector_grounded") and trace.get("status") == "failed":
                return "FAILED"
    # The Memory boundary may run but skip before instantiating the agent.
    if label == "MEMORY" and not any(e["name"] == "MemoryAgent.initialize" for e in run["events"]):
        return "NOT_REQUIRED"
    status_field = {"CORRECTOR": "correction_result", "REVERIFIER": "reverification_result", "MEMORY": "memory_result"}.get(label)
    for call in calls:
        out = call.get("output", {}).get(status_field, {}) if status_field else {}
        if str(out.get("status", "")).lower() in {"failed", "fallback"}:
            return "FAILED"
        if label == "REVERIFIER" and out.get("passed") is False:
            return "FAILED"
    return "FIRED"


def generator_status(run):
    calls = [e for e in run["events"] if e["name"] == "BaseLLMService.generate" and e["node"] == "generate"]
    if not calls:
        return "NOT_REQUIRED" if run["draft"] is not None else "NOT_FIRED"
    if any(e["status"] != "RETURNED" or str(e.get("output", {}).get("status", "")).lower() in {"failed", "fallback"} for e in calls):
        return "FAILED"
    return "FIRED"


def counts(run):
    labels = [agent_status(run, label, nodes) for label, nodes, _ in AGENTS]
    return {name: labels.count(name) for name in ("FIRED", "FAILED", "NOT_REQUIRED", "NOT_FIRED")}


def header(run):
    result = run["result"]
    return ["=" * 80, "HALLUCIGUARD - REAL PRODUCTION-GRAPH VALIDATION", "=" * 80,
            "CASE: " + run["case"], "Query: " + run["query"], "Initial draft mode: " + run["mode"],
            "Repository snapshot: " + pretty(run["git"]), "UTC: " + run["started_utc"],
            f"Wall time: {run['seconds']:.3f} seconds; runner status: {run['runner_status']}",
            "Seven downstream agents (Base LLM excluded): " + pretty(counts(run)),
            "Terminal status: " + str(result.get("terminal_status", "UNAVAILABLE")),
            "Verification status: " + str(result.get("verification_status", "UNAVAILABLE")),
            "Final Judge decision: " + str(result.get("judge_decision", "UNAVAILABLE")),
            "Original generated/supplied draft: " + str(result.get("draft_response", run.get("draft") or "UNAVAILABLE")),
            "FINAL ANSWER: " + str(result.get("final_response", "UNAVAILABLE")),
            "Production checkpoint identity: " + pretty(run.get("checkpoint_before", {})),
            "Checkpoint unchanged: " + str(run.get("checkpoint_unchanged", "NOT_CHECKED")),
            "Memory uses fresh diagnostic files, not the user's existing production records.",
            "Cache bypassed for this test; all evidence retrieval is live, not supplied.",
            "FIRED is execution coverage, NOT a claim that a verdict is factually correct.",
            "Missing probabilities remain null/N/A; genuine zeros are preserved.", ""]


def section(run, label, nodes, purpose):
    status = agent_status(run, label, nodes)
    calls = node_events(run, nodes)
    lines = ["-" * 80, "[AGENT] " + label, "PURPOSE: " + purpose, "STATUS: " + status,
             f"EXECUTION: {len(calls)} observed graph-boundary invocation(s)."]
    if label == "DETECTOR":
        lines += ["The initial pass is evidence-free triage; no neural probability is expected.",
                  "Grounded verification risk includes contradiction AND insufficient evidence; it is not proof of falsehood."]
    if not calls:
        lines += ["INPUT: N/A - not invoked", "ACTUAL OUTPUT: N/A - not invoked", "TIMING: N/A",
                  "EVIDENCE: N/A", "FAILURES: no successful execution demonstrated."]
    for call in calls:
        lines += ["", f"NODE {call['name']} (event {call['sequence']})", f"TIMING: {call['seconds']:.3f} seconds",
                  "EXECUTION STATUS: " + call["status"], "INPUT: exact state handoff, excluding repeated trace/bus",
                  pretty({k: v for k, v in call["input"].get("state", {}).items() if k not in OMIT_HANDOFF}),
                  "ACTUAL OUTPUT:", pretty({k: v for k, v in call.get("output", {}).items() if k not in OMIT_HANDOFF})]
        if call.get("error_type"):
            lines += ["FAILURES: " + call["error_type"]]
    components = [e for e in run["events"] if e["kind"] == "component" and e["node"] in nodes]
    lines += ["", "EVIDENCE: actual selected and submitted text is captured in handoffs/component calls; N/A if no evidence was supplied.",
              "MODEL / TOOL / EVIDENCE / DIAGNOSTIC CALLS (actual inputs and outputs):", pretty(components),
              "FAILURES: inspect returned status, degraded flags, graph error events and classified exceptions above.",
              "LIMITATIONS: selected or submitted text is not automatically tokenizer-surviving text; only explicit tokenizer diagnostics establish that.",
              "FINAL RESULT: " + status + "; production decision is " + str(run["result"].get("judge_decision", "UNAVAILABLE"))]
    return "\n".join(lines) + "\n"


def full_report(run):
    lines = header(run)
    generator = node_events(run, ("generate",))
    generated = [e for e in run["events"] if e["name"] == "BaseLLMService.generate" and e["node"] == "generate"]
    lines += ["[GENERATOR] BASE LLM - not one of the seven downstream agents",
              "STATUS: " + generator_status(run),
              "INPUT / EXECUTION / ACTUAL OUTPUT / TIMING:", pretty(generated or generator),
              "EVIDENCE: none supplied for generation.", "LIMITATIONS: generated text is untrusted until verified.", ""]
    for label, nodes, purpose in AGENTS:
        lines.append(section(run, label, nodes, purpose))
    lines += ["COMPLETE ORDERED EXECUTION TRACE (node and nested component calls):"]
    for event in run["events"]:
        lines.append(f"{event['sequence']:03d} {event['kind']:9} {event['name']} [{event['node']}] {event['status']} {event.get('seconds', 0):.3f}s")
    lines += ["AUTHORITATIVE GRAPH TRACE:", pretty(run["result"].get("trace", [])),
              "GRAPH FAILURES:", pretty(run["result"].get("errors", [])),
              "FINAL RESULT:", str(run["result"].get("final_response", "UNAVAILABLE"))]
    return "\n".join(lines) + "\n"


def export_reports(directory, runs):
    # Readable reports use the same narrated view as the CLI. Exact handoffs
    # and complete component inputs/outputs remain in raw/*_execution.json;
    # dumping the whole pipeline state into every agent's TXT obscures its work.
    from scripts.run_seven_agent_walkthrough import format_readable_walkthrough, STAGES, SUB

    directory = Path(directory)
    narrated = {id(run): format_readable_walkthrough(run) for run in runs}

    def readable(run):
        return narrated[id(run)]

    (directory / "01_ALL_7_AGENTS_END_TO_END.txt").write_text("\n".join(readable(r) for r in runs), encoding="utf-8")
    for index, (label, nodes, purpose) in enumerate(AGENTS, 2):
        stage_index = index - 1
        title = STAGES[stage_index - 1][1]
        marker = f"{stage_index}. [AGENT] {title}"
        excerpts = []
        for run in runs:
            narrative = readable(run)
            start = narrative.index(marker)
            if stage_index < len(STAGES):
                next_marker = f"{stage_index + 1}. [AGENT] {STAGES[stage_index][1]}"
                stop = narrative.index(next_marker, start)
            else:
                stop = narrative.index("  SUPERVISOR SUMMARY", start)
            excerpts.append("\n".join([
                "HALLUCIGUARD — " + label + " — RECORDED LIVE EXECUTION",
                "INPUT: " + run["query"], "PURPOSE: " + purpose,
                "STATUS: " + agent_status(run, label, nodes),
                "EXECUTION / ACTUAL OUTPUT / TIMING / EVIDENCE:", SUB,
                narrative[start:stop].rstrip(),
                "FAILURES: " + (pretty(run["result"].get("errors")) if run["result"].get("errors") else "None reported by graph"),
                "LIMITATIONS: selected passages do not establish tokenizer survival. Exact handoffs and full evidence are in raw/" + run["case"] + "_execution.json.",
                "FINAL RESULT: " + str(run["result"].get("final_response", "UNAVAILABLE")), "",
            ]))
        (directory / f"{index:02d}_{label}_AGENT.txt").write_text("\n".join(excerpts), encoding="utf-8")
    for name, mode in (("09_CONTRADICTION_CORRECTION_TEST.txt", "supplied_draft"), ("10_NORMAL_LIVE_QUERY.txt", "real_generation")):
        matching = [r for r in runs if r["mode"] == mode]
        content = "\n".join(readable(r) for r in matching) if matching else "INPUT: N/A\nPURPOSE: separate case\nEXECUTION: NOT_RUN\nACTUAL OUTPUT: N/A\nSTATUS: NOT_FIRED\nTIMING: N/A\nEVIDENCE: N/A\nFAILURES: not attempted in this invocation\nLIMITATIONS: no execution evidence\nFINAL RESULT: NOT_RUN\n"
        (directory / name).write_text(content, encoding="utf-8")
    dell_runs = [r for r in runs if r["mode"] == "real_generation" and "dell g15" in r["query"].lower()]
    if dell_runs:
        (directory / "10_DELL_G15_LIVE_QUERY.txt").write_text("\n".join(readable(r) for r in dell_runs), encoding="utf-8")
    summary = ["HALLUCIGUARD - VALIDATION SUMMARY", "This validation run does not train, override graph routing, commit, push or deploy.",
               "INPUT: " + "; ".join(r["case"] + ": " + r["query"] for r in runs),
               "PURPOSE: measure actual execution coverage and expose live failures, not claim detection accuracy.",
               "EXECUTION: transparent observers around unmodified production graph/functions."]
    for run in runs:
        summary += header(run)
        summary += ["STATUS: " + run["runner_status"], f"TIMING: {run['seconds']:.3f} seconds",
                    "ACTUAL OUTPUT: " + str(run["result"].get("final_response", "UNAVAILABLE")),
                    "EVIDENCE: actual retrieved/selected/model-submitted text in Verifier report and raw JSON; none injected."]
        for label, nodes, _ in AGENTS:
            summary.append(label + ": " + agent_status(run, label, nodes))
        summary += ["Node route: " + " -> ".join(e["name"] for e in run["events"] if e["kind"] == "node"),
                    "FAILURES: " + pretty(run["result"].get("errors", [])),
                    "Evidence, models and scores: see 04_VERIFIER_AGENT.txt and per-case execution JSON.",
                    "LIMITATIONS: two live requests cannot establish accuracy or production readiness; missing correction coverage is not PASS.",
                    "FINAL RESULT: " + str(run["result"].get("final_response", "UNAVAILABLE"))]
    summary += ["SEVEN-AGENT EXECUTION COVERAGE:"]
    for label, nodes, _ in AGENTS:
        summary.append(label + ": " + ", ".join(r["case"] + "=" + agent_status(r, label, nodes) for r in runs))
    (directory / "SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def git_snapshot():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {"branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"), "status": git("status", "--short")}


def isolate_runtime(directory):
    # The production Memory implementation still executes, but never mutates
    # pre-existing stores. No retrieval/verdict/correction is mocked or supplied.
    directory.mkdir(parents=True, exist_ok=False)
    for key, name in {"KG_PERSISTENCE_PATH": "knowledge_graph.json", "CACHE_DB_PATH": "memory-cache.db",
                      "VECTOR_STORE_PATH": "vector_store", "PATTERN_DB_PATH": "patterns.db", "TRUST_DB_PATH": "source-trust.db"}.items():
        os.environ[key] = str((directory / name).resolve())
    for key, value in {"CACHE_ENABLED": "false", "VERIFIER_CACHE_ENABLED": "false", "ALWAYS_VERIFY": "true",
                       "ALLOW_DETECTOR_FAST_PATH": "false", "ALLOW_MODEL_DOWNLOADS": "false", "MOCK_MODE": "false",
                       "HG_DETECTOR_TRACE_TOKENS": "true"}.items():
        os.environ[key] = value


def checkpoint_identity():
    configured = os.getenv("HALLUCIGUARD_DETECTOR_MODEL", "")
    path = Path(configured) if configured else ROOT / "artifacts" / "detector-best"
    if not path.is_absolute():
        path = ROOT / path
    files = {}
    for name in ("model.safetensors", "config.json", "tokenizer.json", "calibration.json", "model_metadata.json"):
        file = path / name
        if file.is_file():
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1048576), b""):
                    digest.update(block)
            files[name] = {"sha256": digest.hexdigest(), "bytes": file.stat().st_size}
    return {"path": str(path.resolve()), "files": files,
            "architecture": "existing production halluciguard_detector; no experimental V3 integration"}


async def execute_case(args, query, draft, case, secrets):
    from orchestration import graph
    recorder = Recorder(secrets)
    run = {"case": case, "query": query, "draft": draft, "mode": "supplied_draft" if draft is not None else "real_generation",
           "started_utc": datetime.now(timezone.utc).isoformat(), "git": git_snapshot(), "runner_status": "completed", "result": {}}
    run["checkpoint_before"] = checkpoint_identity()
    transcript = io.StringIO()
    start = time.perf_counter()
    with redirect_stdout(transcript), redirect_stderr(transcript), ExitStack() as stack:
        try:
            install_observers(stack, recorder, graph)
            overrides = {name: recorder.node_wrapper(name, getattr(graph, "_" + name + "_node")) for name in NODES}
            stack.enter_context(patch.object(graph, "_GRAPH", graph.build_verification_graph(overrides)))
            run["result"] = scrub(await asyncio.wait_for(graph.run_verification(query, draft or ""), args.timeout_seconds), secrets)
        except Exception as exc:
            run["runner_status"] = "failed:" + type(exc).__name__
            run["result"] = {"terminal_status": "runner_failed", "error_type": type(exc).__name__}
    run["seconds"] = time.perf_counter() - start
    run["checkpoint_after"] = checkpoint_identity()
    run["checkpoint_unchanged"] = run["checkpoint_before"] == run["checkpoint_after"]
    run["events"] = recorder.events
    (args.output_dir / "raw" / (case + "_console.txt")).write_text(scrub(transcript.getvalue(), secrets), encoding="utf-8")
    (args.output_dir / "raw" / (case + "_execution.json")).write_text(pretty(run), encoding="utf-8")
    for label, nodes, _ in AGENTS:
        (args.output_dir / "raw" / (case + "_" + label.lower() + ".json")).write_text(pretty(node_events(run, nodes)), encoding="utf-8")
    return run


def requested_cases(query, draft, suite=False):
    """Name generic runs honestly; named fixture cases belong only to --suite."""
    if suite:
        return [("What is the cost of Dell G15 laptop?", None, "TEST_A_DELL_G15"),
                ("Who founded Microsoft?", "Snehith is the founder of Microsoft.", "TEST_B_MICROSOFT")]
    return [(query, draft, "SUPPLIED_DRAFT_RUN" if draft is not None else "QUERY_RUN")]


def main():
    # Windows PowerShell/CMD may supply cp1252 pipes; real LLM output commonly
    # contains Unicode currencies and nonbreaking spaces. Preserve exact text.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query")
    parser.add_argument("--draft", help="Supplied initial draft; only intended for the deliberate correction case.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--checkpoint", type=Path, help="Existing production checkpoint; never written.")
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--suite", action="store_true", help="Run exact Dell and Snehith/Microsoft cases serially; combines reports.")
    args = parser.parse_args()
    if not args.suite and not args.query:
        parser.error("--query is required unless --suite is used")
    if args.suite and (args.query or args.draft is not None):
        parser.error("--suite uses the two fixed requested inputs, no additional --query/--draft")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("timeout must be positive and finite")
    if args.checkpoint and not args.checkpoint.is_dir():
        parser.error("checkpoint directory does not exist")
    from dotenv import load_dotenv
    load_dotenv(args.env_file)
    if args.checkpoint:
        os.environ["HALLUCIGUARD_DETECTOR_MODEL"] = str(args.checkpoint.resolve())
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "raw").mkdir()
    isolate_runtime(args.output_dir / "runtime")
    secrets = tuple(v for k, v in os.environ.items() if len(v) >= 8 and re.search(r"KEY|TOKEN|PASSWORD|SECRET|AUTH", k, re.I))
    cases = requested_cases(args.query, args.draft, args.suite)
    runs = []
    for query, draft, case in cases:
        print(f"Running {case}: {query}", flush=True)
        runs.append(asyncio.run(execute_case(args, query, draft, case, secrets)))
        export_reports(args.output_dir, runs)
        print("\n".join(header(runs[-1])), flush=True)
    print("Saved reports: " + str(args.output_dir))
    # Exit 0 = runner/export completed; it does not certify every agent or answer.
    return 0 if all(r["runner_status"] == "completed" for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
