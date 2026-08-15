from __future__ import annotations

import asyncio
import json
import os
import random
import socket
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal

import httpx

GenerationMode = Literal["normal", "stress_test"]
GenerationStatus = Literal["success", "failed"]


class GenerationErrorCode(StrEnum):
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
    return os.getenv(name, default)


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _env_optional_int(name: str) -> int | None:
    return int(os.environ[name]) if os.getenv(name) else None


@dataclass(frozen=True)
class BaseLLMConfig:
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
        default_factory=lambda: _env_str("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct")
        or "qwen/qwen-2.5-7b-instruct"
    )
    temperature: float = field(
        default_factory=lambda: _env_float("OPENROUTER_TEMPERATURE", "0.7")
    )
    stress_temperature: float = field(
        default_factory=lambda: _env_float("OPENROUTER_STRESS_TEMPERATURE", "0.9")
    )
    max_tokens: int | None = field(
        default_factory=lambda: _env_optional_int("OPENROUTER_MAX_TOKENS")
    )
    timeout_seconds: float = field(
        default_factory=lambda: _env_float("OPENROUTER_TIMEOUT_SECONDS", "30")
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
        return asdict(self)


@dataclass(frozen=True)
class BaseLLMHealth:
    provider_configured: bool
    provider: str
    model_configured: bool
    model: str
    key_configured: bool
    endpoint_reachable: bool
    latency_ms: int
    last_error: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class BaseLLMService:
    """Generate draft responses through OpenRouter without exposing secrets."""

    RETRYABLE_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504}
    NON_RETRYABLE_HTTP_STATUS = {400, 401, 402, 403, 404}

    def __init__(self, config: BaseLLMConfig | None = None) -> None:
        self.config = config or BaseLLMConfig()

    async def health(self, check_network: bool = True) -> BaseLLMHealth:
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
                request_id,
                mode,
                temp,
                started,
                GenerationErrorCode.UNSUPPORTED_PROVIDER,
                f"Unsupported LLM provider: {self.config.provider}",
            )
        if not self.config.api_key:
            return self._failed(
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
                            request_id, mode, temp, started, code, message
                        )
                    last_code, last_error = code, message
                else:
                    return self._parse_success(
                        response, request_id, mode, temp, started
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
                    return self._failed(request_id, mode, temp, started, code, message)
                last_code, last_error = code, message
            await asyncio.sleep(self._retry_delay_seconds(attempt))
        return self._failed(request_id, mode, temp, started, last_code, last_error)

    async def _post_chat_completions(self, payload: dict[str, Any]) -> httpx.Response:
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
        response: httpx.Response,
        request_id: str,
        mode: str,
        temperature: float,
        started: float,
    ) -> GenerationResult:
        if not response.content:
            return self._failed(
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
                request_id,
                mode,
                temperature,
                started,
                GenerationErrorCode.EMPTY_CONTENT,
                "OpenRouter returned no assistant content",
            )
        return GenerationResult(
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
        request_id: str,
        mode: str,
        temperature: float | None,
        started: float,
        error_code: GenerationErrorCode,
        error: str,
    ) -> GenerationResult:
        return GenerationResult(
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
        base = min(2.0, 0.25 * (2**attempt))
        return base + random.uniform(0, 0.1)

    def _should_retry_status(self, status_code: int) -> bool:
        return status_code in self.RETRYABLE_HTTP_STATUS

    def _classify_http_status(self, status_code: int, body: str) -> GenerationErrorCode:
        text = (body or "").lower()
        if status_code == 404 and "model" in text:
            return GenerationErrorCode.MODEL_UNAVAILABLE
        try:
            return GenerationErrorCode[f"HTTP_{status_code}"]
        except KeyError:
            return GenerationErrorCode.UNKNOWN_ERROR

    def _classify_exception(self, exc: Exception) -> GenerationErrorCode:
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
        redacted = (body or "").replace(self.config.api_key or "", "[REDACTED]")
        if len(redacted) > 500:
            redacted = f"{redacted[:500]}..."
        return f"HTTP_{status_code}: {redacted or 'OpenRouter request failed'}"
