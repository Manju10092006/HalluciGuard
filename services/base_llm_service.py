from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import random
import socket
import time
import uuid
from dataclasses import asdict, dataclass, field
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        """String-based enum for Python versions < 3.11 compatibility."""
        pass

from typing import Any, Literal

import httpx

from services.llm_providers import (
    ProviderConfigError,
    ProviderSpec,
    build_provider_specs,
    resolve_provider_order,
)

logger = logging.getLogger("services.base_llm_service")

GenerationMode = Literal["normal", "stress_test"]
GenerationStatus = Literal["success", "failed"]


class GenerationErrorCode(StrEnum):
    """Error codes for LLM generation failures, categorizing different failure modes."""
    MISSING_API_KEY = "MISSING_API_KEY"
    TIMEOUT = "TIMEOUT"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    DNS_ERROR = "DNS_ERROR"
    HTTP_400 = "HTTP_400"
    HTTP_401 = "HTTP_401"
    HTTP_402 = "HTTP_402"
    HTTP_403 = "HTTP_403"
    HTTP_404 = "HTTP_404"
    HTTP_408 = "HTTP_408"
    HTTP_409 = "HTTP_409"
    HTTP_429 = "HTTP_429"
    HTTP_500 = "HTTP_500"
    HTTP_502 = "HTTP_502"
    HTTP_503 = "HTTP_503"
    HTTP_504 = "HTTP_504"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    MALFORMED_JSON = "MALFORMED_JSON"
    EMPTY_CONTENT = "EMPTY_CONTENT"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    UNSUPPORTED_PROVIDER = "UNSUPPORTED_PROVIDER"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


def _env_str(name: str, default: str | None = None) -> str | None:
    """
    Retrieve a string environment variable with an optional default value.

    Args:
        name: The environment variable name.
        default: The default value if the environment variable is not set.

    Returns:
        The environment variable value or the default value.
    """
    return os.getenv(name, default)


def _env_float(name: str, default: str) -> float:
    """
    Retrieve a float environment variable with a default value.

    Args:
        name: The environment variable name.
        default: The default value as a string to parse as float.

    Returns:
        The environment variable value parsed as float, or the default value.
    """
    val = os.getenv(name)
    if val is not None and val.strip():
        return float(val.strip())
    return float(default)


def _env_int(name: str, default: str) -> int:
    """
    Retrieve an integer environment variable with a default value.

    Args:
        name: The environment variable name.
        default: The default value as a string to parse as int.

    Returns:
        The environment variable value parsed as int, or the default value.
    """
    val = os.getenv(name)
    if val is not None and val.strip():
        return int(val.strip())
    return int(default)


def _env_optional_int(name: str) -> int | None:
    """
    Retrieve an optional integer environment variable.

    Args:
        name: The environment variable name.

    Returns:
        The environment variable value parsed as int, or None if not set or empty.
    """
    val = os.getenv(name)
    return int(val.strip()) if val and val.strip() else None


@dataclass(frozen=True)
class BaseLLMConfig:
    """Configuration settings for the Base LLM service, sourced from environment variables."""
    provider: str = field(
        default_factory=lambda: _env_str("HALLUCIGUARD_LLM_PROVIDER", "openrouter")
        or "openrouter"
    )
    api_key: str | None = field(default_factory=lambda: _env_str("OPENROUTER_API_KEY"))
    base_url: str = field(
        default_factory=lambda: _env_str(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        or "https://openrouter.ai/api/v1"
    )
    model: str = field(
        default_factory=lambda: _env_str("HALLUCIGUARD_LLM_MODEL")
        or _env_str("OPENROUTER_MODEL", "qwen/qwen3-14b")
        or "qwen/qwen3-14b"
    )
    temperature: float = field(
        default_factory=lambda: _env_float("HALLUCIGUARD_LLM_TEMPERATURE", os.getenv("OPENROUTER_TEMPERATURE", "0.7"))
    )
    stress_temperature: float = field(
        default_factory=lambda: _env_float("OPENROUTER_STRESS_TEMPERATURE", "0.9")
    )
    max_tokens: int | None = field(
        default_factory=lambda: _env_optional_int("OPENROUTER_MAX_TOKENS")
    )
    timeout_seconds: float = field(
        default_factory=lambda: _env_float("HALLUCIGUARD_LLM_TIMEOUT", os.getenv("OPENROUTER_TIMEOUT_SECONDS", "30.0"))
    )
    max_retries: int = field(
        default_factory=lambda: _env_int("OPENROUTER_MAX_RETRIES", "3")
    )
    http_referer: str | None = field(
        default_factory=lambda: _env_str("OPENROUTER_HTTP_REFERER")
    )
    x_title: str | None = field(
        default_factory=lambda: _env_str("OPENROUTER_X_TITLE", "HalluciGuard")
    )


@dataclass(frozen=True)
class GenerationResult:
    """Result of an LLM generation request, containing the draft response or error details."""
    user_query: str
    draft_response: str
    model: str
    provider: str
    generation_mode: str
    mode: str
    temperature: float | None
    latency_ms: int
    finish_reason: str | None
    request_id: str
    status: GenerationStatus
    error: str | None = None
    error_code: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    # Which provider actually served the request (Groq/Gemini/OpenRouter), or
    # None when generation failed across every provider. Additive + defaulted
    # so existing construction sites remain valid.
    provider_used: str | None = None
    # Ordered, secret-free record of each provider attempt for observability.
    provider_attempts: list[dict[str, Any]] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        """Convert the generation result to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class BaseLLMHealth:
    """Health check result for the Base LLM service, including configuration and endpoint status."""
    provider_configured: bool
    provider: str
    model_configured: bool
    model: str
    key_configured: bool
    endpoint_reachable: bool
    latency_ms: int
    last_error: str | None = None

    def model_dump(self) -> dict[str, Any]:
        """Convert the health check result to a dictionary."""
        return asdict(self)


class BaseLLMService:
    """Generate draft responses through a multi-provider failover router.

    Two construction modes are supported for full backward compatibility:

    * **Multi-provider (default)** — ``BaseLLMService()`` with no explicit
      config routes generation through the configured provider order
      (Groq -> Gemini -> OpenRouter by default). Each provider gets its own
      retry budget; providers without a configured key are skipped. This is the
      path every live consumer (``BaseLLMService()``) now takes automatically.
    * **Legacy single-provider** — ``BaseLLMService(BaseLLMConfig(...))`` with an
      explicit config preserves the original single-endpoint behavior exactly,
      including the ``_post_chat_completions`` override seam used by tests.
    """

    RETRYABLE_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504}
    NON_RETRYABLE_HTTP_STATUS = {400, 401, 402, 403, 404}
    # Cap Retry-After honoring so a hostile/misconfigured header cannot stall
    # the request pipeline for minutes.
    MAX_RETRY_AFTER_SECONDS = 10.0

    def __init__(
        self,
        config: BaseLLMConfig | None = None,
        *,
        model_overrides: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the Base LLM service.

        Args:
            config: Optional legacy single-provider configuration. When provided,
                the service operates in single-provider mode (no failover) for
                backward compatibility. When omitted, the service builds the
                multi-provider failover router from the environment.
            model_overrides: Optional ``provider -> model`` overrides applied in
                multi-provider mode (e.g. the Corrector pinning a model).
        """
        self.config = config or BaseLLMConfig()
        self._multi = config is None
        # Shared generation settings (sourced from BaseLLMConfig so env tuning of
        # timeouts/retries/tokens continues to apply in multi-provider mode too).
        self._provider_order: list[str] = []
        self._provider_specs: dict[str, ProviderSpec] = {}
        if self._multi:
            self._provider_order = resolve_provider_order()
            self._provider_specs = build_provider_specs(model_overrides)

    def provider_status(self) -> dict[str, Any]:
        """Return a secret-free snapshot of provider configuration for startup logs."""
        if not self._multi:
            return {
                "mode": "single",
                "provider": self.config.provider,
                "model": self.config.model,
                "key_configured": bool(self.config.api_key),
            }
        return {
            "mode": "multi",
            "order": list(self._provider_order),
            "providers": {
                name: self._provider_specs[name].safe_summary()
                for name in self._provider_order
            },
        }


    async def health(self, check_network: bool = True) -> BaseLLMHealth:
        """
        Perform a health check on the Base LLM service.

        Args:
            check_network: Whether to perform a network connectivity check to the OpenRouter endpoint.

        Returns:
            A BaseLLMHealth instance with configuration and endpoint status information.
        """
        started = time.perf_counter()
        reachable = False
        last_error = None
        if check_network:
            try:
                async with httpx.AsyncClient(
                    timeout=min(self.config.timeout_seconds, 5)
                ) as client:
                    response = await client.get(
                        f"{self.config.base_url.rstrip('/')}/models"
                    )
                    reachable = response.status_code < 500
                    if not reachable:
                        last_error = f"HTTP_{response.status_code}"
            except Exception as exc:
                last_error = self._classify_exception(exc).value
        return BaseLLMHealth(
            provider_configured=bool(self.config.provider),
            provider=self.config.provider,
            model_configured=bool(self.config.model),
            model=self.config.model,
            key_configured=bool(self.config.api_key),
            endpoint_reachable=reachable,
            latency_ms=int((time.perf_counter() - started) * 1000),
            last_error=last_error,
        )

    async def generate(
        self,
        user_query: str,
        conversation_history: list[dict[str, str]] | None = None,
        generation_mode: GenerationMode = "normal",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """
        Generate a draft response for a user query via OpenRouter.

        Args:
            user_query: The user's question or prompt.
            conversation_history: Optional list of prior conversation messages.
            generation_mode: Either "normal" or "stress_test" (affects temperature).
            temperature: Optional temperature override (0.0 to 1.0).
            max_tokens: Optional maximum token limit for the response.

        Returns:
            A GenerationResult containing the draft response or error details.
        """
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        mode = (
            generation_mode
            if generation_mode in {"normal", "stress_test"}
            else "normal"
        )
        temp = temperature
        if temp is None:
            temp = (
                self.config.stress_temperature
                if mode == "stress_test"
                else self.config.temperature
            )

        if self._multi:
            return await self._generate_multi(
                user_query,
                conversation_history,
                mode,
                temp,
                max_tokens,
                request_id,
                started,
            )

        if self.config.provider.lower() != "openrouter":
            return self._failed(
                user_query,
                request_id,
                mode,
                temp,
                started,
                GenerationErrorCode.UNSUPPORTED_PROVIDER,
                f"Unsupported LLM provider: {self.config.provider}",
            )
        if not self.config.api_key:
            return self._failed(
                user_query,
                request_id,
                mode,
                temp,
                started,
                GenerationErrorCode.MISSING_API_KEY,
                "OPENROUTER_API_KEY is not configured",
            )

        payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": temp,
            "messages": [
                *list(conversation_history or []),
                {"role": "user", "content": user_query},
            ],
        }
        token_limit = max_tokens if max_tokens is not None else self.config.max_tokens
        if token_limit is not None:
            payload["max_tokens"] = token_limit

        last_code = GenerationErrorCode.UNKNOWN_ERROR
        last_error = "Unknown OpenRouter generation failure"
        attempts = max(1, self.config.max_retries + 1)
        for attempt in range(attempts):
            try:
                response = await self._post_chat_completions(payload)
                if response.status_code >= 400:
                    if response.status_code == 404 and payload.get("model") != "qwen/qwen-2.5-7b-instruct":
                        logging.getLogger("services.base_llm_service").warning(
                            "OpenRouter model '%s' returned HTTP 404; falling back to "
                            "'qwen/qwen-2.5-7b-instruct' and retrying.",
                            payload.get("model"),
                        )
                        payload["model"] = "qwen/qwen-2.5-7b-instruct"
                        continue
                    code = self._classify_http_status(
                        response.status_code, response.text
                    )
                    message = self._safe_http_error(response.status_code, response.text)
                    if (
                        not self._should_retry_status(response.status_code)
                        or attempt == attempts - 1
                    ):
                        return self._failed(
                            user_query, request_id, mode, temp, started, code, message
                        )
                    last_code, last_error = code, message
                else:
                    return self._parse_success(
                        user_query, response, request_id, mode, temp, started
                    )
            except Exception as exc:
                code = self._classify_exception(exc)
                message = f"{code.value}: {type(exc).__name__}"
                if (
                    code
                    not in {
                        GenerationErrorCode.TIMEOUT,
                        GenerationErrorCode.CONNECTION_ERROR,
                        GenerationErrorCode.DNS_ERROR,
                    }
                    or attempt == attempts - 1
                ):
                    return self._failed(user_query, request_id, mode, temp, started, code, message)
                last_code, last_error = code, message
            await asyncio.sleep(self._retry_delay_seconds(attempt))
        return self._failed(user_query, request_id, mode, temp, started, last_code, last_error)

    # ------------------------------------------------------------------
    # Multi-provider failover engine (Groq -> Gemini -> OpenRouter)
    # ------------------------------------------------------------------
    async def _generate_multi(
        self,
        user_query: str,
        conversation_history: list[dict[str, str]] | None,
        mode: str,
        temp: float,
        max_tokens: int | None,
        request_id: str,
        started: float,
    ) -> GenerationResult:
        """Attempt each configured provider in order, each with its own retry budget.

        Providers without a configured API key are skipped (no network, no wasted
        retry). The first provider to return usable content wins and its name is
        recorded in ``provider_used``. If every keyed provider is exhausted the
        result fails closed; if no provider had a key at all, the classic
        MISSING_API_KEY semantics are preserved.
        """
        messages = [
            *list(conversation_history or []),
            {"role": "user", "content": user_query},
        ]
        # NOTE: do not collapse to a single global token limit here. An explicit
        # per-call ``max_tokens`` wins everywhere; otherwise each provider uses
        # its own ``spec.max_tokens`` (resolved in build_provider_specs), so the
        # OpenRouter credit cap never throttles Groq/Gemini.
        attempts_trail: list[dict[str, Any]] = []
        any_key = False
        last_code = GenerationErrorCode.MISSING_API_KEY
        last_error = "No LLM provider has a configured API key"

        for name in self._provider_order:
            spec = self._provider_specs[name]
            if not spec.has_key:
                logger.info("LLM provider skipped (no key) provider=%s", name)
                attempts_trail.append(
                    {"provider": name, "outcome": "skipped_no_key"}
                )
                continue
            any_key = True
            logger.info("LLM generation started provider=%s model=%s", name, spec.model)
            outcome = await self._run_provider(
                spec, messages, temp, max_tokens, user_query, request_id, mode, started
            )
            if isinstance(outcome, GenerationResult):
                attempts_trail.append(
                    {"provider": name, "outcome": "success", "model": outcome.model}
                )
                logger.info("LLM generation succeeded provider=%s", name)
                # Re-emit the full attempt trail on the successful result.
                return dataclasses.replace(outcome, provider_attempts=attempts_trail)
            code, message = outcome
            last_code, last_error = code, message
            attempts_trail.append(
                {"provider": name, "outcome": "exhausted", "error_code": code.value}
            )
            logger.warning(
                "LLM provider exhausted provider=%s error=%s", name, code.value
            )

        if not any_key:
            logger.error("LLM generation failed: no provider key configured")
            return self._failed(
                user_query,
                request_id,
                mode,
                temp,
                started,
                GenerationErrorCode.MISSING_API_KEY,
                last_error,
                provider_attempts=attempts_trail,
            )

        logger.error("LLM generation failed across all providers last=%s", last_code.value)
        return self._failed(
            user_query,
            request_id,
            mode,
            temp,
            started,
            last_code,
            last_error,
            provider_attempts=attempts_trail,
        )

    async def _run_provider(
        self,
        spec: ProviderSpec,
        messages: list[dict[str, str]],
        temp: float,
        max_tokens: int | None,
        user_query: str,
        request_id: str,
        mode: str,
        started: float,
    ) -> GenerationResult | tuple[GenerationErrorCode, str]:
        """Run one provider with its own bounded retry budget.

        Returns a successful :class:`GenerationResult`, or a ``(code, message)``
        tuple describing why this provider was exhausted so the caller can fail
        over to the next provider.

        Token budget precedence: an explicit per-call ``max_tokens`` overrides
        everything; otherwise the provider's own ``spec.max_tokens`` is used
        (``None`` -> no ``max_tokens`` field sent). This keeps the credit-driven
        OpenRouter cap from starving reasoning models on Groq/Gemini.
        """
        token_limit = max_tokens if max_tokens is not None else spec.max_tokens
        payload: dict[str, Any] = {
            "model": spec.model,
            "temperature": temp,
            "messages": messages,
        }
        if token_limit is not None:
            payload["max_tokens"] = token_limit

        last_code = GenerationErrorCode.UNKNOWN_ERROR
        last_error = f"{spec.name} generation failure"
        attempts = max(1, self.config.max_retries + 1)
        for attempt in range(attempts):
            try:
                response = await self._post_chat_completions_to(spec, payload)
            except Exception as exc:  # noqa: BLE001 - classified below
                code = self._classify_exception(exc)
                last_code = code
                last_error = f"{code.value}: {type(exc).__name__}"
                transient = code in {
                    GenerationErrorCode.TIMEOUT,
                    GenerationErrorCode.CONNECTION_ERROR,
                    GenerationErrorCode.DNS_ERROR,
                }
                if not transient or attempt == attempts - 1:
                    return last_code, last_error
                await asyncio.sleep(self._retry_delay_seconds(attempt))
                continue

            if response.status_code < 400:
                parsed = self._parse_success(
                    user_query,
                    response,
                    request_id,
                    mode,
                    temp,
                    started,
                    provider=spec.name,
                    model_fallback=spec.model,
                )
                if parsed.status == "success":
                    return parsed
                # A 2xx that produced no usable content is not retryable here.
                try:
                    empty_code = GenerationErrorCode(parsed.error_code or "")
                except ValueError:
                    empty_code = GenerationErrorCode.EMPTY_CONTENT
                return empty_code, parsed.error or "empty content"

            # OpenRouter-specific: a 404 on the configured model falls back once
            # to a widely-available model before giving up on the provider.
            if (
                spec.name == "openrouter"
                and response.status_code == 404
                and payload.get("model") != "qwen/qwen-2.5-7b-instruct"
            ):
                logger.warning(
                    "OpenRouter model '%s' returned 404; falling back to "
                    "'qwen/qwen-2.5-7b-instruct'.",
                    payload.get("model"),
                )
                payload["model"] = "qwen/qwen-2.5-7b-instruct"
                continue

            code = self._classify_http_status(response.status_code, response.text)
            message = self._safe_http_error(response.status_code, response.text, spec)
            last_code, last_error = code, message
            if (
                not self._should_retry_status(response.status_code)
                or attempt == attempts - 1
            ):
                return last_code, last_error
            await asyncio.sleep(
                self._retry_after_or_backoff(response, attempt)
            )
        return last_code, last_error

    async def _post_chat_completions_to(
        self, spec: ProviderSpec, payload: dict[str, Any]
    ) -> httpx.Response:
        """Send a chat completion request to a specific provider endpoint."""
        headers = {
            "Authorization": f"Bearer {spec.api_key}",
            "Content-Type": "application/json",
            **dict(spec.extra_headers),
        }
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            return await client.post(
                f"{spec.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )

    def _retry_after_or_backoff(self, response: httpx.Response, attempt: int) -> float:
        """Honor a sane Retry-After header, else fall back to jittered backoff."""
        raw = response.headers.get("Retry-After") or response.headers.get("retry-after")
        if raw:
            try:
                seconds = float(raw.strip())
                if seconds >= 0:
                    return min(seconds, self.MAX_RETRY_AFTER_SECONDS)
            except (TypeError, ValueError):
                pass
        return self._retry_delay_seconds(attempt)

    async def _post_chat_completions(self, payload: dict[str, Any]) -> httpx.Response:
        """
        Send a chat completion request to the OpenRouter API.

        Args:
            payload: The JSON payload containing model, messages, and generation parameters.

        Returns:
            The HTTP response from OpenRouter.
        """
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.http_referer:
            headers["HTTP-Referer"] = self.config.http_referer
        if self.config.x_title:
            headers["X-Title"] = self.config.x_title
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            return await client.post(
                f"{self.config.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )

    def _parse_success(
        self,
        user_query: str,
        response: httpx.Response,
        request_id: str,
        mode: str,
        temperature: float,
        started: float,
        provider: str = "openrouter",
        model_fallback: str | None = None,
        provider_attempts: list[dict[str, Any]] | None = None,
    ) -> GenerationResult:
        """
        Parse a successful HTTP response into a GenerationResult.

        Args:
            user_query: The original user query.
            response: The HTTP response from the provider.
            request_id: The unique request identifier.
            mode: The generation mode ("normal" or "stress_test").
            temperature: The temperature used for generation.
            started: The start time of the request (from perf_counter).
            provider: The provider that served the request.
            model_fallback: Model id to record when the response omits one.
            provider_attempts: Secret-free per-provider attempt trail.

        Returns:
            A GenerationResult with status "success" or "failed" if parsing fails.
        """
        model_fallback = model_fallback or self.config.model
        if not response.content:
            return self._failed(
                user_query,
                request_id,
                mode,
                temperature,
                started,
                GenerationErrorCode.EMPTY_RESPONSE,
                f"{provider} returned an empty HTTP response",
            )
        try:
            data = response.json()
        except json.JSONDecodeError:
            return self._failed(
                user_query,
                request_id,
                mode,
                temperature,
                started,
                GenerationErrorCode.MALFORMED_JSON,
                f"{provider} returned malformed JSON",
            )
        choices = data.get("choices") or []
        content = ""
        finish_reason = None
        if choices:
            first = choices[0] or {}
            finish_reason = first.get("finish_reason")
            message = first.get("message") or {}
            content = str(message.get("content") or "").strip()
        if not content:
            return self._failed(
                user_query,
                request_id,
                mode,
                temperature,
                started,
                GenerationErrorCode.EMPTY_CONTENT,
                f"{provider} returned no assistant content",
            )
        return GenerationResult(
            user_query=user_query,
            draft_response=content,
            model=str(data.get("model") or model_fallback),
            provider=provider,
            generation_mode=mode,
            mode=mode,
            temperature=temperature,
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            request_id=request_id,
            status="success",
            usage=data.get("usage") or {},
            provider_used=provider,
            provider_attempts=provider_attempts or [],
        )

    def _failed(
        self,
        user_query: str,
        request_id: str,
        mode: str,
        temperature: float | None,
        started: float,
        error_code: GenerationErrorCode,
        error: str,
        provider_attempts: list[dict[str, Any]] | None = None,
    ) -> GenerationResult:
        """
        Create a failed GenerationResult with error details.

        Args:
            user_query: The original user query.
            request_id: The unique request identifier.
            mode: The generation mode.
            temperature: The temperature used (or None).
            started: The start time of the request.
            error_code: The error code enum value.
            error: The error message.
            provider_attempts: Secret-free per-provider attempt trail.

        Returns:
            A GenerationResult with status "failed" and error information.
        """
        return GenerationResult(
            user_query=user_query,
            draft_response="",
            model=self.config.model,
            provider=self.config.provider,
            generation_mode=mode,
            mode=mode,
            temperature=temperature,
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=None,
            request_id=request_id,
            status="failed",
            error=error,
            error_code=error_code.value,
            provider_used=None,
            provider_attempts=provider_attempts or [],
        )

    def _retry_delay_seconds(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay with jitter for retry attempts.

        Args:
            attempt: The current retry attempt number (0-indexed).

        Returns:
            The delay in seconds before the next retry attempt.
        """
        base = min(2.0, 0.25 * (2**attempt))
        return base + random.uniform(0, 0.1)

    def _should_retry_status(self, status_code: int) -> bool:
        """
        Determine if an HTTP status code is retryable.

        Args:
            status_code: The HTTP status code.

        Returns:
            True if the status code indicates a transient error that should be retried.
        """
        return status_code in self.RETRYABLE_HTTP_STATUS

    def _classify_http_status(self, status_code: int, body: str) -> GenerationErrorCode:
        """
        Classify an HTTP error status code into a GenerationErrorCode.

        Args:
            status_code: The HTTP status code.
            body: The response body text.

        Returns:
            The corresponding GenerationErrorCode enum value.
        """
        text = (body or "").lower()
        if status_code == 404 and "model" in text:
            return GenerationErrorCode.MODEL_UNAVAILABLE
        try:
            return GenerationErrorCode[f"HTTP_{status_code}"]
        except KeyError:
            return GenerationErrorCode.UNKNOWN_ERROR

    def _classify_exception(self, exc: Exception) -> GenerationErrorCode:
        """
        Classify a network or HTTP exception into a GenerationErrorCode.

        Args:
            exc: The exception that occurred during the request.

        Returns:
            The corresponding GenerationErrorCode enum value.
        """
        if isinstance(exc, httpx.TimeoutException):
            return GenerationErrorCode.TIMEOUT
        if isinstance(exc, httpx.ConnectError):
            if isinstance(exc.__cause__, socket.gaierror):
                return GenerationErrorCode.DNS_ERROR
            return GenerationErrorCode.CONNECTION_ERROR
        if isinstance(exc, httpx.NetworkError):
            return GenerationErrorCode.CONNECTION_ERROR
        return GenerationErrorCode.UNKNOWN_ERROR

    def _safe_http_error(
        self, status_code: int, body: str, spec: ProviderSpec | None = None
    ) -> str:
        """
        Create a safe error message from an HTTP response, redacting sensitive credentials.

        Args:
            status_code: The HTTP status code.
            body: The response body text.
            spec: The provider whose key should also be redacted (multi mode).

        Returns:
            A formatted error message with API keys redacted and long messages truncated.
        """
        redacted = (body or "").replace(self.config.api_key or "", "[REDACTED]")
        if spec is not None and spec.api_key:
            redacted = redacted.replace(spec.api_key, "[REDACTED]")
        if len(redacted) > 500:
            redacted = f"{redacted[:500]}..."
        label = spec.name if spec is not None else "provider"
        return f"HTTP_{status_code}: {redacted or f'{label} request failed'}"
