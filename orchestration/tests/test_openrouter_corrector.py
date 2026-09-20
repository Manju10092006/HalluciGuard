"""Unit tests for the OpenRouter-backed Corrector generator (Fix 1).

These tests exercise :class:`services.openrouter_corrector.OpenRouterCorrectorGenerator`
in isolation using ``httpx.MockTransport`` (the generator's built-in ``transport``
injection seam) — no network, no API key from the real environment, no ML stack.

Coverage:
  * fail-closed at construction when ``OPENROUTER_API_KEY`` is absent (security),
  * exact request shape (endpoint, Bearer auth, deterministic temperature, message
    roles/content ordering) so the Corrector's strict parser receives what it expects,
  * lossless leading-``<think>`` stripping for the ``qwen/qwen3-4b`` default, including
    the verbatim (fail-closed) branches,
  * fail-closed generation errors (HTTP >= 400, malformed body, empty content),
  * env-driven configuration with keyword overrides (no hardcoded secrets/URLs).
"""

from __future__ import annotations

import json

import httpx
import pytest

from services.openrouter_corrector import (
    OpenRouterCorrectorError,
    OpenRouterCorrectorGenerator,
    _strip_leading_think,
)


def _make_generator(handler, **kwargs) -> OpenRouterCorrectorGenerator:
    """Build a generator wired to a MockTransport handler with a test key."""
    kwargs.setdefault("api_key", "test-key")
    return OpenRouterCorrectorGenerator(
        transport=httpx.MockTransport(handler), **kwargs
    )


def _content_response(content: str, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={"choices": [{"message": {"role": "assistant", "content": content}}]},
    )


# ---------------------------------------------------------------------------
# Construction / security
# ---------------------------------------------------------------------------

def test_missing_api_key_fails_closed(monkeypatch):
    """No key anywhere -> raise at construction (never a silent, keyless correction)."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(OpenRouterCorrectorError):
        OpenRouterCorrectorGenerator()


def test_explicit_api_key_overrides_missing_env(monkeypatch):
    """An explicit key lets construction succeed even with the env var cleared."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    gen = OpenRouterCorrectorGenerator(api_key="explicit-key")
    assert gen.api_key == "explicit-key"


def test_kind_is_informational_openrouter():
    """`kind` is the informational label surfaced in ModelStatus.kind."""
    assert OpenRouterCorrectorGenerator(api_key="k").kind == "openrouter"


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------

def test_request_shape_and_auth(monkeypatch):
    """The relayed request matches OpenRouter conventions the parser depends on."""
    monkeypatch.delenv("OPENROUTER_HTTP_REFERER", raising=False)
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return _content_response('{"sentence_id": 1, "corrected_sentence": "ok"}')

    gen = _make_generator(handler, model="qwen/qwen3-4b", max_tokens=777)
    out = gen.generate("SYSTEM PROMPT", "USER PROMPT")

    assert out == '{"sentence_id": 1, "corrected_sentence": "ok"}'
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["headers"]["Content-Type"] == "application/json"

    body = captured["body"]
    assert body["model"] == "qwen/qwen3-4b"
    assert body["temperature"] == 0.0  # deterministic
    assert body["max_tokens"] == 777
    assert body["messages"] == [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "USER PROMPT"},
    ]


def test_attribution_headers_sent_when_configured():
    """Optional HTTP-Referer / X-Title headers are forwarded when provided."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return _content_response("RESULT")

    gen = _make_generator(
        handler, http_referer="https://example.test", x_title="HalluciGuard"
    )
    gen.generate("s", "p")
    assert captured["headers"]["HTTP-Referer"] == "https://example.test"
    assert captured["headers"]["X-Title"] == "HalluciGuard"


# ---------------------------------------------------------------------------
# Lossless leading-<think> strip (qwen/qwen3-4b hybrid-reasoning default)
# ---------------------------------------------------------------------------

def test_generate_strips_leading_think_block():
    """A leading, well-formed <think>...</think> preamble is removed losslessly."""
    def handler(request: httpx.Request) -> httpx.Response:
        return _content_response(
            "<think>Elon Musk is wrong; Guido created Python.</think>\n"
            '{"sentence_id": 1, "corrected_sentence": "Guido created Python."}'
        )

    out = _make_generator(handler).generate("s", "p")
    assert out == '{"sentence_id": 1, "corrected_sentence": "Guido created Python."}'
    assert "<think>" not in out


@pytest.mark.parametrize(
    "content, expected",
    [
        # No think block -> verbatim.
        ('{"sentence_id": 1}', '{"sentence_id": 1}'),
        # Leading whitespace before a well-formed block -> stripped to remainder.
        ("  \n<think>reason</think>  RESULT  ", "RESULT"),
        # Unclosed think (e.g. truncated by max_tokens) -> verbatim (parser fails closed).
        ("<think>never closed", "<think>never closed"),
        # Think block not at the start -> verbatim (no greedy scavenging).
        ("prefix <think>reason</think> tail", "prefix <think>reason</think> tail"),
    ],
)
def test_strip_leading_think_branches(content, expected):
    assert _strip_leading_think(content) == expected


# ---------------------------------------------------------------------------
# Fail-closed generation errors
# ---------------------------------------------------------------------------

def test_http_error_status_fails_closed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream error")

    with pytest.raises(OpenRouterCorrectorError):
        _make_generator(handler).generate("s", "p")


def test_malformed_json_body_fails_closed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json{{{")

    with pytest.raises(OpenRouterCorrectorError):
        _make_generator(handler).generate("s", "p")


def test_empty_content_fails_closed():
    def handler(request: httpx.Request) -> httpx.Response:
        return _content_response("   ")

    with pytest.raises(OpenRouterCorrectorError):
        _make_generator(handler).generate("s", "p")


def test_transport_exception_normalized_to_corrector_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with pytest.raises(OpenRouterCorrectorError):
        _make_generator(handler).generate("s", "p")


# ---------------------------------------------------------------------------
# Env-driven configuration (no hardcoded secrets / URLs)
# ---------------------------------------------------------------------------

def test_config_defaults(monkeypatch):
    """With only the key set, config falls back to the documented defaults."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    for var in (
        "OPENROUTER_BASE_URL",
        "HG_CORRECTOR_OPENROUTER_MODEL",
        "HG_CORRECTOR_OPENROUTER_MAX_TOKENS",
        "HG_CORRECTOR_OPENROUTER_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(var, raising=False)

    gen = OpenRouterCorrectorGenerator()
    assert gen.api_key == "env-key"
    assert gen.base_url == "https://openrouter.ai/api/v1"
    assert gen.model == "qwen/qwen3-4b"  # approved default
    assert gen.max_tokens == 1024
    assert gen.temperature == 0.0
    assert gen.timeout_seconds == 60.0


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://proxy.internal/api/v1/")
    monkeypatch.setenv("HG_CORRECTOR_OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
    monkeypatch.setenv("HG_CORRECTOR_OPENROUTER_MAX_TOKENS", "512")
    monkeypatch.setenv("HG_CORRECTOR_OPENROUTER_TIMEOUT_SECONDS", "30")

    gen = OpenRouterCorrectorGenerator()
    assert gen.base_url == "https://proxy.internal/api/v1"  # trailing slash trimmed
    assert gen.model == "meta-llama/llama-3.1-8b-instruct"
    assert gen.max_tokens == 512
    assert gen.timeout_seconds == 30.0


def test_kwargs_override_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("HG_CORRECTOR_OPENROUTER_MODEL", "env-model")

    gen = OpenRouterCorrectorGenerator(api_key="kw-key", model="kw-model", max_tokens=99)
    assert gen.api_key == "kw-key"
    assert gen.model == "kw-model"
    assert gen.max_tokens == 99


def test_blank_env_values_treated_as_unset(monkeypatch):
    """Whitespace-only env values fall back to defaults, not empty strings."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("HG_CORRECTOR_OPENROUTER_MODEL", "   ")
    gen = OpenRouterCorrectorGenerator()
    assert gen.model == "qwen/qwen3-4b"
