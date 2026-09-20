"""Unit tests for the OpenRouter Corrector Generator.

Exercises construction, config resolution, ``<think>`` stripping, content
extraction, HTTP error handling, and fail-closed behavior — all without
hitting the real OpenRouter API, using ``httpx.MockTransport`` injected
via the ``transport`` constructor param.
"""

from __future__ import annotations

import json
import os
import pytest

import httpx

from services.openrouter_corrector import (
    OpenRouterCorrectorError,
    OpenRouterCorrectorGenerator,
    _env,
    _extract_content,
    _resolve_max_tokens,
    _resolve_timeout,
    _strip_leading_think,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ok_response(content: str, status: int = 200) -> httpx.Response:
    """Build an httpx.Response with a valid OpenRouter chat-completion body."""
    body = {"choices": [{"message": {"content": content}}]}
    return httpx.Response(status, json=body)


def _transport_for(response: httpx.Response) -> httpx.MockTransport:
    """MockTransport returning the given response for any request."""
    return httpx.MockTransport(lambda request: response)


def _generator(
    content: str = '{"corrected": "ok"}',
    status: int = 200,
    **kwargs,
) -> OpenRouterCorrectorGenerator:
    """Build a generator wired to a mock transport returning `content`."""
    return OpenRouterCorrectorGenerator(
        api_key="test-key",
        transport=_transport_for(_ok_response(content, status)),
        **kwargs,
    )


# ── Construction Tests ───────────────────────────────────────────────────────

class TestConstruction:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(OpenRouterCorrectorError, match="OPENROUTER_API_KEY"):
            OpenRouterCorrectorGenerator()

    def test_explicit_api_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        gen = _generator()
        assert gen.api_key == "test-key"

    def test_env_api_key_used_when_no_explicit(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        gen = OpenRouterCorrectorGenerator(
            transport=_transport_for(_ok_response("hi"))
        )
        assert gen.api_key == "env-key"

    def test_blank_env_key_treated_as_missing(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
        with pytest.raises(OpenRouterCorrectorError):
            OpenRouterCorrectorGenerator()

    def test_kind_is_openrouter(self):
        gen = _generator()
        assert gen.kind == "openrouter"

    def test_defaults_applied(self, monkeypatch):
        monkeypatch.delenv("HG_CORRECTOR_OPENROUTER_MODEL", raising=False)
        monkeypatch.delenv("HG_CORRECTOR_OPENROUTER_MAX_TOKENS", raising=False)
        gen = _generator()
        assert gen.model == "qwen/qwen3-4b"
        assert gen.max_tokens == 1024
        assert gen.temperature == 0.0

    def test_explicit_overrides(self):
        gen = _generator(model="my/model", max_tokens=2048, temperature=0.7)
        assert gen.model == "my/model"
        assert gen.max_tokens == 2048
        assert gen.temperature == 0.7


# ── Config Resolution Tests ──────────────────────────────────────────────────

class TestConfigResolution:
    def test_env_helper_returns_default_for_missing(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        assert _env("NONEXISTENT_VAR", "fallback") == "fallback"

    def test_env_helper_returns_value_for_present(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "value")
        assert _env("TEST_VAR") == "value"

    def test_env_helper_treats_blank_as_unset(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "  ")
        assert _env("TEST_VAR", "default") == "default"

    def test_resolve_max_tokens_explicit(self):
        assert _resolve_max_tokens(512) == 512

    def test_resolve_max_tokens_env(self, monkeypatch):
        monkeypatch.setenv("HG_CORRECTOR_OPENROUTER_MAX_TOKENS", "4096")
        assert _resolve_max_tokens(None) == 4096

    def test_resolve_max_tokens_invalid_env_falls_to_default(self, monkeypatch):
        monkeypatch.setenv("HG_CORRECTOR_OPENROUTER_MAX_TOKENS", "not_a_number")
        assert _resolve_max_tokens(None) == 1024

    def test_resolve_timeout_explicit(self):
        assert _resolve_timeout(30.0) == 30.0

    def test_resolve_timeout_env(self, monkeypatch):
        monkeypatch.setenv("HG_CORRECTOR_OPENROUTER_TIMEOUT_SECONDS", "90")
        assert _resolve_timeout(None) == 90.0


# ── Think Block Stripping ────────────────────────────────────────────────────

class TestThinkStripping:
    def test_strips_well_formed_think_block(self):
        text = '<think>reasoning here</think>{"answer": "ok"}'
        assert _strip_leading_think(text) == '{"answer": "ok"}'

    def test_strips_with_leading_whitespace(self):
        text = '  \n<think>stuff</think>real content'
        assert _strip_leading_think(text) == "real content"

    def test_preserves_text_without_think(self):
        text = '{"answer": "already clean"}'
        assert _strip_leading_think(text) == text

    def test_preserves_unclosed_think_block(self):
        text = "<think>started but never closed..."
        assert _strip_leading_think(text) == text

    def test_preserves_think_not_at_start(self):
        text = 'prefix <think>not leading</think> suffix'
        assert _strip_leading_think(text) == text


# ── Content Extraction ───────────────────────────────────────────────────────

class TestContentExtraction:
    def test_extracts_valid_content(self):
        data = {"choices": [{"message": {"content": "hello"}}]}
        assert _extract_content(data) == "hello"

    def test_returns_empty_for_missing_choices(self):
        assert _extract_content({}) == ""

    def test_returns_empty_for_empty_choices(self):
        assert _extract_content({"choices": []}) == ""

    def test_returns_empty_for_non_string_content(self):
        data = {"choices": [{"message": {"content": 42}}]}
        assert _extract_content(data) == ""

    def test_returns_empty_for_none_data(self):
        assert _extract_content(None) == ""


# ── Generation Tests ─────────────────────────────────────────────────────────

class TestGeneration:
    def test_successful_generation(self):
        gen = _generator(content='{"corrected_text": "fixed"}')
        result = gen.generate("system", "prompt")
        assert result == '{"corrected_text": "fixed"}'

    def test_strips_think_block_from_response(self):
        gen = _generator(content='<think>reasoning</think>{"result": "ok"}')
        result = gen.generate("system", "prompt")
        assert result == '{"result": "ok"}'

    def test_http_error_raises(self):
        transport = httpx.MockTransport(
            lambda req: httpx.Response(500, json={"error": "boom"})
        )
        gen = OpenRouterCorrectorGenerator(api_key="k", transport=transport)
        with pytest.raises(OpenRouterCorrectorError, match="HTTP 500"):
            gen.generate("s", "p")

    def test_empty_content_raises(self):
        gen = _generator(content="   ")
        with pytest.raises(OpenRouterCorrectorError, match="empty content"):
            gen.generate("s", "p")

    def test_malformed_json_body_raises(self):
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, content=b"not json", headers={"content-type": "text/plain"})
        )
        gen = OpenRouterCorrectorGenerator(api_key="k", transport=transport)
        with pytest.raises(OpenRouterCorrectorError, match="malformed JSON"):
            gen.generate("s", "p")

    def test_connection_error_raises(self):
        def exploding_handler(request):
            raise httpx.ConnectError("connection refused")

        transport = httpx.MockTransport(exploding_handler)
        gen = OpenRouterCorrectorGenerator(api_key="k", transport=transport)
        with pytest.raises(OpenRouterCorrectorError, match="request failed"):
            gen.generate("s", "p")


# ── Headers ──────────────────────────────────────────────────────────────────

class TestHeaders:
    def test_includes_authorization(self):
        gen = _generator()
        headers = gen._headers()
        assert headers["Authorization"] == "Bearer test-key"

    def test_includes_x_title(self):
        gen = _generator(x_title="TestApp")
        headers = gen._headers()
        assert headers["X-Title"] == "TestApp"

    def test_omits_referer_when_none(self):
        gen = _generator(http_referer=None)
        headers = gen._headers()
        assert "HTTP-Referer" not in headers

    def test_includes_referer_when_set(self):
        gen = _generator(http_referer="https://example.com")
        headers = gen._headers()
        assert headers["HTTP-Referer"] == "https://example.com"
