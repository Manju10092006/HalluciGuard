"""Unit tests for the multi-provider LLM failover router (Groq -> Gemini -> OpenRouter).

All provider HTTP is mocked; no test performs a real network call.
"""

from __future__ import annotations

import httpx
import pytest

from services.base_llm_service import BaseLLMService, GenerationErrorCode
from services.llm_providers import ProviderConfigError, resolve_provider_order


def _ok(model: str, content: str = "hello") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 3},
        },
        request=httpx.Request("POST", "https://example.test"),
    )


def _status(code: int, text: str = "err") -> httpx.Response:
    return httpx.Response(
        code, text=text, request=httpx.Request("POST", "https://example.test")
    )


class RoutedService(BaseLLMService):
    """BaseLLMService whose per-provider HTTP is served from queued fakes."""

    def __init__(self, queues: dict[str, list]):
        super().__init__()
        # provider name -> list of httpx.Response | Exception, consumed in order
        self._queues = {k: list(v) for k, v in queues.items()}
        self.calls: dict[str, int] = {k: 0 for k in queues}

    async def _post_chat_completions_to(self, spec, payload):
        self.calls[spec.name] = self.calls.get(spec.name, 0) + 1
        queue = self._queues.get(spec.name) or []
        if not queue:
            raise AssertionError(f"unexpected extra call to provider {spec.name}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def _retry_delay_seconds(self, attempt: int) -> float:
        return 0

    def _retry_after_or_backoff(self, response, attempt: int) -> float:
        return 0


@pytest.fixture
def all_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gk")
    monkeypatch.setenv("GEMINI_API_KEY", "mk")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ok")
    monkeypatch.setenv("HALLUCIGUARD_LLM_PROVIDER_ORDER", "groq,gemini,openrouter")
    # One attempt per provider keeps failover assertions deterministic.
    monkeypatch.setenv("OPENROUTER_MAX_RETRIES", "0")


def test_order_resolution_and_validation():
    assert resolve_provider_order("groq, gemini ,openrouter") == [
        "groq",
        "gemini",
        "openrouter",
    ]
    assert resolve_provider_order(None) == ["groq", "gemini", "openrouter"]
    assert resolve_provider_order("openrouter,groq") == ["openrouter", "groq"]
    with pytest.raises(ProviderConfigError):
        resolve_provider_order("groq,mistral")


@pytest.mark.asyncio
async def test_groq_success(all_keys):
    svc = RoutedService({"groq": [_ok("llama")], "gemini": [], "openrouter": []})
    result = await svc.generate("hi")
    assert result.status == "success"
    assert result.provider_used == "groq"
    assert svc.calls["groq"] == 1
    assert svc.calls["gemini"] == 0


@pytest.mark.asyncio
async def test_groq_429_fails_over_to_gemini(all_keys):
    svc = RoutedService(
        {"groq": [_status(429, "rate limited")], "gemini": [_ok("gemini-2.0-flash")], "openrouter": []}
    )
    result = await svc.generate("hi")
    assert result.status == "success"
    assert result.provider_used == "gemini"
    assert svc.calls["groq"] == 1
    assert svc.calls["gemini"] == 1


@pytest.mark.asyncio
async def test_groq_timeout_fails_over_to_gemini(all_keys):
    svc = RoutedService(
        {
            "groq": [httpx.TimeoutException("timed out")],
            "gemini": [_ok("gemini-2.0-flash")],
            "openrouter": [],
        }
    )
    result = await svc.generate("hi")
    assert result.status == "success"
    assert result.provider_used == "gemini"


@pytest.mark.asyncio
async def test_gemini_failure_falls_over_to_openrouter(all_keys):
    svc = RoutedService(
        {
            "groq": [_status(500)],
            "gemini": [_status(503)],
            "openrouter": [_ok("qwen/qwen3-14b")],
        }
    )
    result = await svc.generate("hi")
    assert result.status == "success"
    assert result.provider_used == "openrouter"
    assert svc.calls["groq"] == 1
    assert svc.calls["gemini"] == 1
    assert svc.calls["openrouter"] == 1


@pytest.mark.asyncio
async def test_all_providers_fail(all_keys):
    svc = RoutedService(
        {"groq": [_status(500)], "gemini": [_status(500)], "openrouter": [_status(500)]}
    )
    result = await svc.generate("hi")
    assert result.status == "failed"
    assert result.provider_used is None
    # Attempt trail records every provider as exhausted.
    outcomes = {a["provider"]: a["outcome"] for a in result.provider_attempts}
    assert outcomes == {
        "groq": "exhausted",
        "gemini": "exhausted",
        "openrouter": "exhausted",
    }


@pytest.mark.asyncio
async def test_no_keys_fail_without_network(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("HALLUCIGUARD_LLM_PROVIDER_ORDER", "groq,gemini,openrouter")
    svc = RoutedService({"groq": [], "gemini": [], "openrouter": []})
    result = await svc.generate("hi")
    assert result.status == "failed"
    assert result.error_code == GenerationErrorCode.MISSING_API_KEY.value
    assert svc.calls == {"groq": 0, "gemini": 0, "openrouter": 0}


@pytest.mark.asyncio
async def test_missing_key_provider_is_skipped(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "mk")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ok")
    monkeypatch.setenv("HALLUCIGUARD_LLM_PROVIDER_ORDER", "groq,gemini,openrouter")
    svc = RoutedService({"groq": [], "gemini": [_ok("gemini-2.0-flash")], "openrouter": []})
    result = await svc.generate("hi")
    assert result.status == "success"
    assert result.provider_used == "gemini"
    assert svc.calls["groq"] == 0  # skipped, no network
