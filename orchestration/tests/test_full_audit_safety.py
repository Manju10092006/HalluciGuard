"""Network-free regressions for final factual safety gates (not accuracy tests)."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from orchestration import graph
from orchestration.schemas import ContractViolation


def evidence():
    return {"evidence_id": "long-source-id-12345678", "source": "wiki",
            "source_id": "document-A", "snippet": "Alpha exists.",
            "entailment_label": "entailment", "entailment_score": .9}


def report(claim_id="rev-1", text="Alpha exists.", with_evidence=True):
    return {"claim_id": claim_id, "claim_text": text, "verdict": "verified",
            "evidence": [evidence()] if with_evidence else []}


@pytest.mark.parametrize("verdict", ["not_contradicted", "not_conflicted", "hallucination_unknown"])
def test_canonical_rejects_substring_verdicts(verdict):
    r = report()
    r["verdict"] = verdict
    with pytest.raises(ContractViolation):
        graph._build_canonical_verifier_result({"claim_reports": [r]}, "q", "general")


def test_canonical_preserves_status_and_full_provenance():
    result = graph._build_canonical_verifier_result(
        {"status": "failed", "claim_reports": [report()]}, "q", "general")
    assert result.status == "failed"
    assert result.evidence[0].evidence_id == "long-source-id-12345678"
    assert result.evidence[0].source_id == "document-A"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["missing", "duplicate", "no_evidence", "failed", "complete"])
async def test_reverification_requires_every_claim_and_evidence(monkeypatch, case):
    texts = ["Alpha exists.", "Beta exists."]
    monkeypatch.setattr("agents.verifier_agent.claims.claim_decomposer.ClaimDecomposer.decompose",
                        lambda self, text: texts)
    reports = [report("rev-1", texts[0]), report("rev-2", texts[1])]
    if case == "missing":
        reports.pop()
    elif case == "duplicate":
        reports[1] = reports[0]
    elif case == "no_evidence":
        reports[1]["evidence"] = []
    verify = AsyncMock(return_value={"status": "failed" if case == "failed" else "completed",
                                    "claim_reports": reports})
    monkeypatch.setattr(graph, "_get_verifier_imports", lambda: (
        lambda: SimpleNamespace(verify=verify),
        lambda **kw: SimpleNamespace(**kw), lambda **kw: SimpleNamespace(**kw)))
    result = await graph._reverifier_node({"llm_response": " ".join(texts), "domain": "general"})
    verify.assert_awaited_once()
    assert result["reverification_result"]["passed"] is (case == "complete")


@pytest.mark.asyncio
@pytest.mark.parametrize("status,with_evidence", [("completed", False), ("failed", True), ("degraded", True)])
async def test_memory_rejects_ungrounded_or_failed_verification(status, with_evidence):
    result = await graph._memory_node({
        "judge_decision": "ACCEPT", "llm_response": "Alpha exists.",
        "verifier_result": {"status": status, "claim_reports": [report(with_evidence=with_evidence)]}})
    assert result["memory_result"]["stored_count"] == 0
    assert result["memory_result"]["status"] == "skipped"


def test_missing_judge_decision_escalates():
    assert graph._judge_route({}) == "human_escalation"


@pytest.mark.parametrize("route", [None, "unknown", "skipped"])
def test_unknown_detector_route_cannot_default_to_accept(route):
    assert graph._detector_route({"route": route, "llm_response": "Alpha exists."}) == "verifier"
    assert graph._detector_route({"route": route}) == "human_escalation"


@pytest.mark.parametrize("label", ["unsupported", "not_supported", "not_contradicted"])
def test_canonical_unknown_evidence_label_is_neutral(label):
    r = report()
    r["evidence"][0]["entailment_label"] = label
    result = graph._build_canonical_verifier_result({"claim_reports": [r]}, "q", "general")
    assert result.evidence[0].entailment_label == "neutral"


def test_graph_failure_does_not_expose_exception_content():
    result = graph._failure_update({}, "verifier", RuntimeError("dummy-secret provider prompt"))
    assert "dummy-secret" not in str(result)
    assert "RuntimeError" in str(result)


def test_runtime_health_supports_groq_without_openrouter(monkeypatch):
    from orchestration.runtime_validation import validate_base_llm_configuration
    monkeypatch.setenv("HALLUCIGUARD_LLM_PROVIDER_ORDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "dummy-secret")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = validate_base_llm_configuration()
    assert result.ok
    assert result.metadata["configured_providers"] == ["groq"]
    assert "dummy-secret" not in str(result)
    assert result.metadata["inference_tested"] is False


@pytest.mark.asyncio
async def test_deep_health_does_not_return_raw_exception(monkeypatch):
    from orchestration import api
    monkeypatch.setattr(api, "validate_orchestration_startup", lambda: {"ok": False})
    monkeypatch.setattr(api, "BaseLLMService", lambda: SimpleNamespace(
        health=AsyncMock(side_effect=RuntimeError("dummy-secret private prompt"))))
    result = await api.health(deep=True)
    assert result["status"] == "degraded"
    assert "dummy-secret" not in str(result)
    assert result["base_llm"]["error_type"] == "RuntimeError"


def test_shared_error_helper_sanitizes_all_node_errors():
    from orchestration.state import add_error
    assert "dummy-secret" not in str(add_error({}, "analyzer", ValueError("dummy-secret")))


@pytest.mark.parametrize("raw", [
    {"claim_evidence": [{"claim_text": "Alpha exists.", "verdict": "unverified",
                          "evidence": [evidence()]}]},
    {"claim_evidence_pairs": [{"claim": "Alpha exists.", "evidence": "Unrelated numbers 99.",
                                "source": "fixture"}]},
])
def test_judge_legacy_adapter_does_not_invent_grounding(raw):
    from agents.judge_agent.judge_agent import JudgeAgent
    result = JudgeAgent()._normalize_verifier_result(raw, fallback_domain="general")
    assert result.claim_reports[0].verdict == "unverified"
    assert result.claim_reports[0].support_score == 0
    assert result.claim_reports[0].confidence_score == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["missing", "duplicate", "wrong_text"])
async def test_initial_verifier_requires_requested_claim_coverage(monkeypatch, case):
    reports = [report("c1", "Alpha exists."), report("c2", "Beta exists.")]
    if case == "missing":
        reports.pop()
    elif case == "duplicate":
        reports[1] = reports[0]
    else:
        reports[1]["claim_text"] = "Different claim."
    verify = AsyncMock(return_value={"claim_evidence": reports})
    monkeypatch.setattr(graph, "_get_verifier_imports", lambda: (
        lambda: SimpleNamespace(verify=verify),
        lambda **kw: SimpleNamespace(**kw), lambda **kw: SimpleNamespace(**kw)))
    result = await graph._verifier_node({"llm_response": "Alpha exists. Beta exists.",
        "claims_gated": True, "detected_claims": [
            {"claim_id": "c1", "text": "Alpha exists."},
            {"claim_id": "c2", "text": "Beta exists."}]})
    verify.assert_awaited_once()
    assert result["verifier_result"]["status"] == "failed"
