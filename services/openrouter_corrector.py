"""OpenRouter-backed :class:`Generator` for the Corrector Agent.

This is the DEFAULT production seam that revives the correction loop. It supplies
a synchronous ``Generator`` (the contract in
``agents/corrector_agent/corrector/model_client.py``) that relays the Corrector's
per-target prompt bundles to the OpenRouter Chat Completions API and returns the
model's raw text content. Injecting this generator into ``CorrectorAgent`` /
``ModelClient`` bypasses on-disk fine-tuned-model loading entirely (availability
is detected by generator presence), so no local weights are required.

Design boundaries (deliberately narrow):
  * This module ONLY produces text. It never parses, repairs, or validates the
    correction JSON — the strict ``output_parser`` inside the Corrector remains
    the sole authority and fails closed on malformed output.
  * SECURITY: every secret and endpoint is read from the environment; nothing is
    hardcoded. A missing ``OPENROUTER_API_KEY`` fails closed by raising at
    construction, which the orchestration corrector node turns into human
    escalation rather than a silent, fabricated correction.
  * Determinism: temperature is pinned to ``0.0`` so the same prompt yields the
    same correction, matching the Corrector's strict, auditable contract.
  * A DEDICATED SYNCHRONOUS ``httpx`` client is used (not the async
    ``BaseLLMService``) because the corrector node invokes this via
    ``asyncio.to_thread`` — a sync client avoids sync/async friction while
    reusing the exact OpenRouter header/auth/endpoint conventions.

Thinking-model handling (per approved decision — keep ``qwen/qwen3-4b`` + strip):
    The documented default model ``qwen/qwen3-4b`` is a hybrid-reasoning model
    that may emit a leading ``<think>...</think>`` preamble before the JSON the
    strict parser expects. We strip a SINGLE, well-formed, leading think block
    LOSSLESSLY — only when the content starts (after optional whitespace) with
    ``<think>`` AND a matching ``</think>`` exists — then return everything after
    it. Anything else is returned verbatim: no greedy scavenging for embedded
    JSON, so a malformed or truncated response still fails closed at the parser.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx

__all__ = ["OpenRouterCorrectorGenerator", "OpenRouterCorrectorError"]


# ---------------------------------------------------------------------------
# Defaults (all overridable via environment; none are secrets)
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# Approved default: the hybrid-reasoning model whose <think> preamble the strip
# below handles. Overridable with HG_CORRECTOR_OPENROUTER_MODEL.
_DEFAULT_MODEL = "qwen/qwen3-4b"
# Generous headroom: a reasoning model spends tokens on the <think> block BEFORE
# emitting the JSON. Too small a budget truncates the answer and forces a
# fail-closed parse rejection. Overridable with HG_CORRECTOR_OPENROUTER_MAX_TOKENS.
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TIMEOUT_SECONDS = 60.0

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class OpenRouterCorrectorError(RuntimeError):
    """Raised when the OpenRouter corrector cannot produce usable output.

    Every raise is fail-closed: a raise at construction (missing key) escalates
    to human review via the corrector node's failure handling; a raise during
    :meth:`OpenRouterCorrectorGenerator.generate` is caught by the Corrector's
    ``generate_for_prompt`` and recorded as a rejected candidate (the original
    response is preserved — never a fabricated correction).
    """


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read an environment variable, treating blank/whitespace as unset."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw else default


def _resolve_max_tokens(explicit: Optional[int]) -> int:
    """Resolve the output token budget from an explicit value, env, or default."""
    if explicit is not None:
        return explicit
    raw = _env("HG_CORRECTOR_OPENROUTER_MAX_TOKENS")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    return _DEFAULT_MAX_TOKENS


def _resolve_timeout(explicit: Optional[float]) -> float:
    """Resolve the request timeout (seconds) from an explicit value, env, or default."""
    if explicit is not None:
        return explicit
    raw = _env("HG_CORRECTOR_OPENROUTER_TIMEOUT_SECONDS")
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        if value > 0:
            return value
    return _DEFAULT_TIMEOUT_SECONDS


def _strip_leading_think(text: str) -> str:
    """Losslessly remove a single well-formed leading ``<think>...</think>`` block.

    Returns everything after the first ``</think>`` (stripped) ONLY when the
    content begins — after optional leading whitespace — with ``<think>`` and a
    matching ``</think>`` is present. In every other case the text is returned
    verbatim so the strict parser stays the sole authority and fails closed on
    malformed/truncated output. No greedy scavenging for embedded JSON.
    """
    leading = text.lstrip()
    if not leading.startswith(_THINK_OPEN):
        return text
    close_idx = leading.find(_THINK_CLOSE)
    if close_idx == -1:
        # Opened a think block but never closed it (e.g. truncated by max_tokens).
        # Return verbatim; the parser rejects it -> fail closed.
        return text
    return leading[close_idx + len(_THINK_CLOSE):].strip()


def _extract_content(data: Any) -> str:
    """Pull ``choices[0].message.content`` out of an OpenRouter response.

    Returns an empty string on any structural surprise; the caller treats empty
    content as a fail-closed generation failure.
    """
    try:
        choices = data.get("choices") or []
        message = choices[0].get("message") or {}
        content = message.get("content")
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


class OpenRouterCorrectorGenerator:
    """Synchronous OpenRouter :class:`Generator` for the Corrector Agent.

    Implements the seam expected by ``ModelClient``: a ``kind`` attribute
    (informational — injection is detected by generator presence, and this value
    is reported in ``ModelStatus.kind``) and a synchronous
    ``generate(system_text, prompt_text) -> str``.

    Config is read from the environment at construction (overridable via keyword
    arguments for testing). ``OPENROUTER_API_KEY`` is required; its absence
    raises immediately so the pipeline fails closed instead of silently skipping
    correction. ``transport`` is an optional ``httpx`` transport injection point
    for unit tests — it is never set in production.
    """

    # Informational only. Not one of the on-disk ModelKind values by design: an
    # OpenRouter relay is neither the fine-tuned corrector nor a base-model
    # fallback (so ``is_finetuned`` is correctly False), and nothing coerces this
    # back into the ModelKind enum.
    kind: str = "openrouter"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        timeout_seconds: Optional[float] = None,
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        resolved_key = api_key if api_key is not None else _env("OPENROUTER_API_KEY")
        if not resolved_key:
            # Fail closed at construction: never attempt a correction without a key.
            raise OpenRouterCorrectorError("OPENROUTER_API_KEY is not configured")
        self.api_key = resolved_key

        base = base_url or _env("OPENROUTER_BASE_URL", _DEFAULT_BASE_URL) or _DEFAULT_BASE_URL
        self.base_url = base.rstrip("/")
        self.model = (
            model
            or _env("HG_CORRECTOR_OPENROUTER_MODEL", _DEFAULT_MODEL)
            or _DEFAULT_MODEL
        )
        self.max_tokens = _resolve_max_tokens(max_tokens)
        self.temperature = temperature
        self.timeout_seconds = _resolve_timeout(timeout_seconds)
        # Optional OpenRouter attribution headers (mirrors BaseLLMService).
        self.http_referer = (
            http_referer if http_referer is not None else _env("OPENROUTER_HTTP_REFERER")
        )
        self.x_title = (
            x_title if x_title is not None else _env("OPENROUTER_X_TITLE", "HalluciGuard")
        )
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        """Build request headers, including Bearer auth and optional attribution."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title
        return headers

    def generate(self, system_text: str, prompt_text: str) -> str:
        """Relay one prompt to OpenRouter and return the raw model text content.

        The returned string is passed verbatim (minus a single leading think
        block) to the Corrector's strict output parser; this method performs no
        JSON parsing or repair of its own. Any failure raises
        :class:`OpenRouterCorrectorError`, which the Corrector degrades to a
        rejected candidate (fail closed).
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt_text},
            ],
        }
        url = f"{self.base_url}/chat/completions"

        try:
            with httpx.Client(
                timeout=self.timeout_seconds, transport=self._transport
            ) as client:
                response = client.post(url, headers=self._headers(), json=payload)
        except Exception as exc:  # noqa: BLE001 - normalize to a fail-closed error
            raise OpenRouterCorrectorError(
                f"OpenRouter request failed: {type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise OpenRouterCorrectorError(
                f"OpenRouter returned HTTP {response.status_code}"
            )

        try:
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - malformed body -> fail closed
            raise OpenRouterCorrectorError(
                "OpenRouter returned a malformed JSON body"
            ) from exc

        content = _extract_content(data)
        if not content.strip():
            raise OpenRouterCorrectorError("OpenRouter returned empty content")

        return _strip_leading_think(content)
