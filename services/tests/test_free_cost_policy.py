"""Offline dispatch-boundary tests; no providers or ML models are invoked."""
import builtins
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from services.base_llm_service import BaseLLMConfig, BaseLLMService
from services.cost_policy import (
    CostPolicyError, free_only, guard_hosted_request, guard_public_retrieval_request,
)
from services.llm_providers import ProviderConfigError, ProviderSpec, resolve_provider_order


@pytest.fixture(autouse=True)
def explicit_free_profile(monkeypatch):
    monkeypatch.setenv("HALLUCIGUARD_FREE_ONLY", "true")
    monkeypatch.setenv("HALLUCIGUARD_LLM_PROVIDER_ORDER", "openrouter")
    monkeypatch.setenv("HALLUCIGUARD_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "fixture/model:free")
    monkeypatch.setenv("HALLUCIGUARD_LLM_MODEL", "fixture/model:free")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.delenv("HG_CORRECTOR_OPENROUTER_MODEL", raising=False)


@pytest.fixture(autouse=True)
def forbid_ml_imports(monkeypatch):
    original = builtins.__import__
    def checked_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {
            "torch", "transformers", "spacy", "sentence_transformers", "faiss", "tensorflow",
        }, "Boundary tests must not import ML libraries"
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked_import)


def boundary_module(monkeypatch, name, relative):
    # Adapter package initialization imports unrelated model infrastructure.
    # Execute the real boundary file directly; dependencies stay pure Python.
    from services import n8n_retrieval_client
    path = Path(__file__).resolve().parents[2] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("provider,url,model", [
    ("groq", "https://api.groq.com/openai/v1", "fixture/model:free"),
    ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "fixture/model:free"),
    ("openrouter", "https://openrouter.ai.evil.test/api/v1", "fixture/model:free"),
    ("openrouter", "https://openrouter.ai/api/v1", "paid/model"),
    ("openrouter", "https://openrouter.ai/api/v1", "openrouter/auto"),
])
async def test_unapproved_dispatch_never_constructs_http_client(monkeypatch, legacy, provider, url, model):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unapproved requests must not construct HTTP client")
    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    service = BaseLLMService(BaseLLMConfig(provider=provider, base_url=url, model=model))
    with pytest.raises(CostPolicyError):
        if legacy:
            await service._post_chat_completions({"model": model, "messages": []})
        else:
            await service._post_chat_completions_to(
                ProviderSpec(provider, url, "test-key", model), {"model": model, "messages": []})


@pytest.mark.parametrize("extra", [
    {"models": ["paid/model"]}, {"plugins": [{"id": "web"}]},
    {"provider": {"max_price": {"prompt": 100}}},
    {"messages": [{"content": [{"type": "image_url", "image_url": "https://example.test"}]}]},
    {"model": "fixture/model:online:free"},
])
def test_paid_addons_multimodal_and_search_routes_are_rejected(extra):
    with pytest.raises(CostPolicyError):
        guard_hosted_request("openrouter", "https://openrouter.ai/api/v1",
            {"model": "fixture/model:free", "messages": [], **extra})


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("status", [402, 404, 429])
async def test_free_provider_failure_never_changes_model_or_routes(monkeypatch, legacy, status):
    real_client = httpx.AsyncClient
    sent = []
    def respond(request):
        sent.append(json.loads(request.content))
        assert request.url.host == "openrouter.ai"
        return httpx.Response(status, json={"error": {"message": "unavailable"}})
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs:
        real_client(transport=httpx.MockTransport(respond), **kwargs))
    monkeypatch.setenv("OPENROUTER_MAX_RETRIES", "1")
    monkeypatch.setenv("GROQ_API_KEY", "present-but-not-authorized")
    monkeypatch.setenv("GEMINI_API_KEY", "present-but-not-authorized")
    service = BaseLLMService(BaseLLMConfig()) if legacy else BaseLLMService()
    service._retry_delay_seconds = lambda _: 0
    service._retry_after_or_backoff = lambda *_: 0
    result = await service.generate("public fixture")
    assert result.status == "failed"
    assert 1 <= len(sent) <= 2
    assert all(p["model"] == "fixture/model:free" for p in sent)
    assert all(p["provider"]["max_price"] == {"prompt": 0, "completion": 0, "request": 0} for p in sent)


@pytest.mark.parametrize("order", ["groq,openrouter", "", "groq", "gemini"])
def test_free_policy_never_silently_replaces_provider_order(order):
    with pytest.raises(ProviderConfigError):
        resolve_provider_order(order)


def test_free_policy_never_silently_replaces_paid_model(monkeypatch):
    monkeypatch.setenv("HALLUCIGUARD_LLM_MODEL", "paid/model")
    monkeypatch.setenv("OPENROUTER_MODEL", "paid/model")
    service = BaseLLMService()
    assert service._provider_specs["openrouter"].model == "paid/model"
    with pytest.raises(CostPolicyError):
        guard_hosted_request("openrouter", "https://openrouter.ai/api/v1",
                             {"model": service._provider_specs["openrouter"].model, "messages": []})


@pytest.mark.asyncio
async def test_health_does_not_probe_an_unapproved_configured_route(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network for unapproved health route")
    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    result = await BaseLLMService(BaseLLMConfig(model="paid/model")).health()
    assert not result.endpoint_reachable
    assert result.last_error == "COST_POLICY"


@pytest.mark.asyncio
async def test_corrector_explicit_paid_override_fails_without_network(monkeypatch):
    from services.character_regenerator import CharacterRegenerator
    monkeypatch.setenv("HG_CORRECTOR_OPENROUTER_MODEL", "paid/model")
    def forbidden(*args, **kwargs):
        raise AssertionError("No network for unapproved Corrector model")
    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    result = await CharacterRegenerator()._service.generate("fixture")
    assert result.status == "failed"
    assert result.error_code == "COST_POLICY"


def test_direct_groq_corrector_is_blocked():
    from agents.corrector_agent.corrector.groq_client import GroqGenerator
    with pytest.raises(CostPolicyError):
        GroqGenerator(api_key="fixture-key").generate("system", "fixture")


def test_tavily_is_blocked_even_with_a_cached_client(monkeypatch):
    TavilyWebRetriever = boundary_module(monkeypatch, "_free_web_retriever",
        "agents/verifier_agent/adapters/web_retriever.py").TavilyWebRetriever
    retriever = TavilyWebRetriever(api_key="fixture-key")
    retriever._client = object()
    assert not retriever.is_available
    with pytest.raises(CostPolicyError):
        retriever._get_client()


@pytest.mark.asyncio
async def test_blocked_tavily_trace_does_not_claim_a_call(monkeypatch):
    WebEnhancedAdapter = boundary_module(monkeypatch, "_free_web_enhanced",
        "agents/verifier_agent/adapters/web_enhanced.py").WebEnhancedAdapter
    from schemas.retrieval_trace import RetrievalTrace
    adapter = WebEnhancedAdapter(MagicMock(name="general"), tavily_api_key="fixture-key")
    trace = RetrievalTrace()
    assert await adapter._run_tavily("fixture", 3, trace, "test") == []
    assert not trace.tavily.called
    assert trace.tavily.reason == "FREE_ONLY_COST_POLICY"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["health", "single", "batch"])
@pytest.mark.parametrize("policy", ["true", "ture"])
async def test_n8n_all_entry_points_fail_closed_before_http(monkeypatch, operation, policy):
    from services.n8n_retrieval_client import N8NRetrievalClient
    monkeypatch.setenv("HALLUCIGUARD_FREE_ONLY", policy)
    def forbidden(*args, **kwargs):
        raise AssertionError("No unapproved workflow HTTP")
    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    client = N8NRetrievalClient(webhook_url="https://workflow.example.test/paid")
    async def invoke():
        if operation == "health":
            return await client.check_health()
        if operation == "single":
            return await client.retrieve_evidence("fixture", force_tavily=True)
        return await client.retrieve_claims_batch([{"claim_text": "fixture"}], force_tavily=True)
    if policy == "ture":
        with pytest.raises(CostPolicyError):
            await invoke()
    else:
        result = await invoke()
        assert not (result["healthy"] if isinstance(result, dict) else result.success)


@pytest.mark.parametrize("url,options", [
    ("https://www.alphavantage.co/query", {}),
    ("https://unknown.example.test/search", {}),
    ("https://www.courtlistener.com/api/rest/v4/search/", {"headers": {"Authorization": "Token fixture"}}),
    ("https://services.nvd.nist.gov/rest/json/cves/2.0", {"headers": {"apiKey": "fixture"}}),
    ("https://api.fda.gov/drug/label.json", {"params": {"api_key": "fixture"}}),
    ("https://en.wikipedia.org/w/api.php?access_token=fixture", {}),
])
def test_account_or_unapproved_retrieval_is_blocked(url, options):
    with pytest.raises(CostPolicyError):
        guard_public_retrieval_request("GET", url, options)


@pytest.mark.asyncio
async def test_public_retrieval_dispatch_rechecks_before_network(monkeypatch):
    from agents.verifier_agent.utils.http_client import ResilientHttpClient
    client = object.__new__(ResilientHttpClient)
    client.client = SimpleNamespace(request=AsyncMock(side_effect=AssertionError("No HTTP")))
    with pytest.raises(CostPolicyError):
        await client.get("https://api.fda.gov/drug/label.json", params={"api_key": "fixture"})
    client.client.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_anonymous_public_retrieval_is_allowed_with_mock_transport():
    from agents.verifier_agent.utils.http_client import ResilientHttpClient
    sent = []
    def respond(request):
        sent.append(request)
        return httpx.Response(200, json={"fixture": True})
    client = object.__new__(ResilientHttpClient)
    client.config = {"default": {"max_retries": 0, "base_delay_seconds": 0,
        "max_delay_seconds": 0, "retry_on_status": [], "timeout_seconds": 1}, "adapters": {}}
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client.client = http
        response = await client.get("https://en.wikipedia.org/w/api.php", params={"search": "fixture"})
    assert response.status_code == 200
    assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("inherited", [
    {"headers": {"Authorization": "Token fixture"}},
    {"params": {"api_key": "fixture"}},
    {"auth": ("fixture", "credential")},
    {"cookies": {"account": "fixture"}},
    {"follow_redirects": True},
])
async def test_public_retrieval_checks_inherited_client_options(inherited):
    from agents.verifier_agent.utils.http_client import ResilientHttpClient
    sent = []
    def respond(request):
        sent.append(request)
        raise AssertionError("Inherited account or redirect settings must not dispatch")
    client = object.__new__(ResilientHttpClient)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), **inherited) as http:
        client.client = http
        with pytest.raises(CostPolicyError):
            await client.get("https://en.wikipedia.org/w/api.php")
    assert sent == []


def test_public_retrieval_disallows_per_request_redirect_following():
    with pytest.raises(CostPolicyError, match="redirects"):
        guard_public_retrieval_request("GET", "https://en.wikipedia.org/w/api.php", {"follow_redirects": True})


@pytest.mark.asyncio
async def test_alphavantage_direct_adapter_is_blocked_before_client_call(monkeypatch):
    # Load the actual pure-Python resolver without executing claims/__init__.py,
    # which eagerly imports the unrelated spaCy claim decomposer.
    claims = ModuleType("claims")
    claims.__path__ = []
    monkeypatch.setitem(sys.modules, "claims", claims)
    boundary_module(monkeypatch, "claims.entity_resolver",
        "agents/verifier_agent/claims/entity_resolver.py")
    FinanceAdapter = boundary_module(monkeypatch, "_free_finance",
        "agents/verifier_agent/adapters/finance.py").FinanceAdapter
    adapter = object.__new__(FinanceAdapter)
    client = SimpleNamespace(get=AsyncMock(side_effect=AssertionError("No HTTP")))
    with pytest.raises(CostPolicyError):
        await adapter._search_alphavantage(client, "MSFT", 1)
    client.get.assert_not_awaited()


def test_invalid_policy_fails_closed(monkeypatch):
    monkeypatch.setenv("HALLUCIGUARD_FREE_ONLY", "ture")
    with pytest.raises(CostPolicyError):
        free_only()
    with pytest.raises(CostPolicyError):
        guard_public_retrieval_request("GET", "https://en.wikipedia.org/w/api.php", {})
    with pytest.raises(CostPolicyError):
        guard_hosted_request("openrouter", "https://openrouter.ai/api/v1", {"model": "fixture/model:free"})


def test_unset_flag_preserves_existing_provider_and_request_behavior(monkeypatch):
    monkeypatch.delenv("HALLUCIGUARD_FREE_ONLY")
    monkeypatch.delenv("HALLUCIGUARD_LLM_PROVIDER_ORDER")
    assert resolve_provider_order() == ["groq", "gemini", "openrouter"]
    payload = {"model": "paid/model"}
    assert guard_hosted_request("groq", "https://example.test", payload) is payload
    guard_public_retrieval_request("POST", "http://example.test", {"params": {"api_key": "fixture"}})
