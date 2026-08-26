from __future__ import annotations

import asyncio
import json
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
        or _env_str("OPENROUTER_MODEL", "qwen/qwen3-4b")
        or "qwen/qwen3-4b"
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
    """Generate draft responses through OpenRouter without exposing secrets."""

    RETRYABLE_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504}
    NON_RETRYABLE_HTTP_STATUS = {400, 401, 402, 403, 404}

    def __init__(self, config: BaseLLMConfig | None = None) -> None:
        """
        Initialize the Base LLM service with optional configuration.

        Args:
            config: Optional configuration instance. If not provided, uses default environment-based config.
        """
        self.config = config or BaseLLMConfig()

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
    ) -> GenerationResult:
        """
        Parse a successful HTTP response from OpenRouter into a GenerationResult.

        Args:
            user_query: The original user query.
            response: The HTTP response from OpenRouter.
            request_id: The unique request identifier.
            mode: The generation mode ("normal" or "stress_test").
            temperature: The temperature used for generation.
            started: The start time of the request (from perf_counter).

        Returns:
            A GenerationResult with status "success" or "failed" if parsing fails.
        """
        if not response.content:
            return self._failed(
                user_query,
                request_id,
                mode,
                temperature,
                started,
                GenerationErrorCode.EMPTY_RESPONSE,
                "OpenRouter returned an empty HTTP response",
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
                "OpenRouter returned malformed JSON",
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
                "OpenRouter returned no assistant content",
            )
        return GenerationResult(
            user_query=user_query,
            draft_response=content,
            model=str(data.get("model") or self.config.model),
            provider="openrouter",
            generation_mode=mode,
            mode=mode,
            temperature=temperature,
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            request_id=request_id,
            status="success",
            usage=data.get("usage") or {},
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

    def _safe_http_error(self, status_code: int, body: str) -> str:
        """
        Create a safe error message from an HTTP response, redacting sensitive credentials.

        Args:
            status_code: The HTTP status code.
            body: The response body text.

        Returns:
            A formatted error message with API keys redacted and long messages truncated.
        """
        redacted = (body or "").replace(self.config.api_key or "", "[REDACTED]")
        if len(redacted) > 500:
            redacted = f"{redacted[:500]}..."
        return f"HTTP_{status_code}: {redacted or 'OpenRouter request failed'}"
