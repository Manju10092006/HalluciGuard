"""Actual LangGraph/nodes with controlled provider, NLI and storage adapters.

These tests deliberately do not certify real models, retrieval or providers.
"""
from types import SimpleNamespace

import pytest

from orchestration import graph, detector_bridge
from services import claim_analyzer
from services.base_llm_service import BaseLLMService


@pytest.mark.asyncio
@pytest.mark.parametrize("corrected_verdict", ["verified", "unverified"])
async def test_real_nodes_independently_gate_corrected_answer(monkeypatch, corrected_verdict):
    original = "Paris is the capital of Germany."
    corrected = "Paris is the capital of France."
    provider_calls, verified_texts, stored = [], [], []

    async def provider(self, **kwargs):
        provider_calls.append(kwargs)
        text = original if len(provider_calls) == 1 else corrected
        fields = dict(status="success", draft_response=text, provider_used="fixture", model="fixture", error=None, error_code=None)
        return SimpleNamespace(**fields, model_dump=lambda: fields)

    def detection(query, draft, *args):
        return {"status":"completed", "next_action":"Verify", "risk_level":"HIGH",
                "hallucination_probability":0.9, "confidence_score":0.9,
                "per_claim_results":[{"claim_id":"c1", "text":draft}]}

    class Pipeline:
        async def verify(self, payload):
            reports = []
            for claim in payload.suspicious_claims:
                verified_texts.append(claim.text)
                after = "Germany" not in claim.text
                verdict = corrected_verdict if after else "contradicted"
                direction = "entailment" if verdict == "verified" else "contradiction" if verdict == "contradicted" else "neutral"
                reports.append({"claim_id":claim.claim_id, "claim_text":claim.text,
                    "verdict":verdict, "support_score":.98 if verdict == "verified" else .01,
                    "contradiction_score":.98 if verdict == "contradicted" else .01,
                    "confidence_score":.99,
                    "evidence":[{"evidence_id":"fixture", "source":"fixture", "snippet":corrected,
                                 "entailment_label":direction, "entailment_score":.99, "credibility_score":.99}]})
            return {"query_id":payload.query_id, "domain":"general", "claim_evidence":reports,
                    "overall_evidence_confidence":.99, "status":"completed"}

    class Storage:
        async def initialize(self): pass
        async def close(self): pass
        async def store_facts_batch(self, requests):
            stored.extend(requests)
            return SimpleNamespace(results=[{"fact_id":f"f{i}"} for i,_ in enumerate(requests)],
                                   stored=len(requests),duplicates=0,failed=0)

    monkeypatch.setattr(BaseLLMService, "generate", provider)
    monkeypatch.setattr(claim_analyzer, "ANALYZER_ENABLED", False)
    monkeypatch.setattr(detector_bridge, "run_detection", detection)
    monkeypatch.setattr(detector_bridge, "run_grounded_detection", lambda *args: {
        **detection(*args), "inference_executed":False, "model_loaded":False})
    monkeypatch.setattr(graph, "_get_verifier_imports", lambda: (Pipeline, SimpleNamespace, SimpleNamespace))
    monkeypatch.setattr("agents.memory_agent.memory.memory_agent.MemoryAgent", Storage)
    monkeypatch.setenv("HG_CORRECTOR_PROVIDER", "hosted")
    monkeypatch.setenv("ALWAYS_VERIFY", "true")
    monkeypatch.setenv("ALLOW_DETECTOR_FAST_PATH", "false")
    monkeypatch.setattr(graph, "_GRAPH", graph.build_verification_graph())
    result = await graph.run_verification("What is the capital of France?")
    nodes = [event["node"] for event in result["trace"]]
    for node in ("base_llm", "detector", "claim_analyzer", "verifier", "judge", "corrector", "reverifier", "memory"):
        assert node in nodes, (node, result)
    assert nodes.index("corrector") < nodes.index("reverifier") < len(nodes)-1
    assert len(provider_calls) == 2
    assert any("Germany" in text for text in verified_texts)
    assert any("France" in text and "Germany" not in text for text in verified_texts)
    if corrected_verdict == "verified":
        assert result["terminal_status"] == "accepted"
        assert result["reverification_result"]["passed"] is True
        assert result["final_response"] == corrected
        assert stored and all("Germany" not in request.claim_text for request in stored)
        assert all(request.evidence[0]["entailment_label"] == "entailment" for request in stored)
    else:
        assert result["terminal_status"] != "accepted"
        assert result["reverification_result"]["passed"] is False
        assert result["memory_result"]["stored_count"] == 0
        assert stored == []
