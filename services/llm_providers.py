"""Central multi-provider configuration for HalluciGuard's hosted LLM layer.

This module owns the *declaration* of the hosted LLM providers HalluciGuard can
route through (Groq primary, Gemini fallback, OpenRouter last resort) and the
resolution of the configured provider order. It deliberately holds no network
logic: :class:`services.base_llm_service.BaseLLMService` performs the actual
generation and failover so there is exactly one auditable request path.

All three providers speak the OpenAI-compatible ``/chat/completions`` contract,
so a single request/parse implementation in ``BaseLLMService`` serves them all;
only the base URL, credential, default model, and a few headers differ. Those
differences are captured here as immutable :class:`ProviderSpec` records.

Secrets are read from the environment and are never logged or serialized.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping


# Canonical provider identifiers. Order in this tuple is the documented default
# failover order (Groq -> Gemini -> OpenRouter) and is used when the
# HALLUCIGUARD_LLM_PROVIDER_ORDER environment variable is unset.
GROQ = "groq"
GEMINI = "gemini"
OPENROUTER = "openrouter"

KNOWN_PROVIDERS: tuple[str, ...] = (GROQ, GEMINI, OPENROUTER)
DEFAULT_PROVIDER_ORDER: tuple[str, ...] = (GROQ, GEMINI, OPENROUTER)

# Default models are configurable via *_MODEL env vars. These defaults target
# currently-supported, low-latency, general-purpose chat models for each
# provider. Override per deployment; never hardcode a model at a call site.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3-14b"


class ProviderConfigError(ValueError):
    """Raised when the configured provider order contains an unknown provider."""


@dataclass(frozen=True)
class ProviderSpec:
    """Immutable declaration of one hosted LLM provider endpoint.

    Attributes:
        name: Canonical provider id (one of :data:`KNOWN_PROVIDERS`).
        base_url: OpenAI-compatible API root (no trailing ``/chat/completions``).
        api_key: The credential, or ``None`` when unconfigured. Never logged.
        model: The default model id used for this provider.
        extra_headers: Additional request headers (e.g. OpenRouter attribution).
    """

    name: str
    base_url: str
    api_key: str | None
    model: str
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def has_key(self) -> bool:
        """Whether a non-empty API key is configured for this provider."""
        return bool(self.api_key and self.api_key.strip())

    def safe_summary(self) -> dict[str, object]:
        """Return a secret-free description suitable for logs and health checks."""
        return {
            "provider": self.name,
            "model": self.model,
            "key_configured": self.has_key,
            "base_url": self.base_url,
        }


def _env(name: str) -> str | None:
    val = os.getenv(name)
    return val.strip() if val and val.strip() else None


def resolve_provider_order(raw: str | None = None) -> list[str]:
    """Normalize and validate the configured provider failover order.

    Args:
        raw: Raw ``HALLUCIGUARD_LLM_PROVIDER_ORDER`` value (comma-separated).
            When ``None`` the environment variable is read; when that is also
            unset the documented default order is returned.

    Returns:
        A list of canonical provider ids in failover order, de-duplicated while
        preserving first-seen order.

    Raises:
        ProviderConfigError: If any entry is not a known provider.
    """
    if raw is None:
        raw = os.getenv("HALLUCIGUARD_LLM_PROVIDER_ORDER")
    if raw is None or not raw.strip():
        return list(DEFAULT_PROVIDER_ORDER)

    order: list[str] = []
    unknown: list[str] = []
    for token in raw.split(","):
        name = token.strip().lower()
        if not name:
            continue
        if name not in KNOWN_PROVIDERS:
            unknown.append(name)
            continue
        if name not in order:
            order.append(name)

    if unknown:
        raise ProviderConfigError(
            "Unknown LLM provider(s) in HALLUCIGUARD_LLM_PROVIDER_ORDER: "
            f"{', '.join(unknown)}. Valid providers: {', '.join(KNOWN_PROVIDERS)}."
        )
    if not order:
        return list(DEFAULT_PROVIDER_ORDER)
    return order


def build_provider_specs(
    model_overrides: Mapping[str, str] | None = None,
) -> dict[str, ProviderSpec]:
    """Construct all provider specs from the environment.

    Args:
        model_overrides: Optional mapping of ``provider name -> model id`` that
            overrides the per-provider default/env model. Used by the Corrector
            to pin a specific correction model without changing global config.

    Returns:
        A mapping of provider id to :class:`ProviderSpec` for every known
        provider (regardless of whether a key is configured; missing-key
        providers are skipped at generation time, not here).
    """
    overrides = {k.lower(): v for k, v in (model_overrides or {}).items() if v}

    openrouter_referer = _env("OPENROUTER_HTTP_REFERER")
    openrouter_title = _env("OPENROUTER_X_TITLE") or "HalluciGuard"
    openrouter_headers: dict[str, str] = {}
    if openrouter_referer:
        openrouter_headers["HTTP-Referer"] = openrouter_referer
    if openrouter_title:
        openrouter_headers["X-Title"] = openrouter_title

    specs = {
        GROQ: ProviderSpec(
            name=GROQ,
            base_url=_env("GROQ_BASE_URL") or "https://api.groq.com/openai/v1",
            api_key=_env("GROQ_API_KEY"),
            model=overrides.get(GROQ) or _env("GROQ_MODEL") or DEFAULT_GROQ_MODEL,
        ),
        GEMINI: ProviderSpec(
            name=GEMINI,
            base_url=(
                _env("GEMINI_BASE_URL")
                or "https://generativelanguage.googleapis.com/v1beta/openai"
            ),
            api_key=_env("GEMINI_API_KEY"),
            model=overrides.get(GEMINI) or _env("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
        ),
        OPENROUTER: ProviderSpec(
            name=OPENROUTER,
            base_url=_env("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
            api_key=_env("OPENROUTER_API_KEY"),
            model=(
                overrides.get(OPENROUTER)
                or _env("HALLUCIGUARD_LLM_MODEL")
                or _env("OPENROUTER_MODEL")
                or DEFAULT_OPENROUTER_MODEL
            ),
            extra_headers=openrouter_headers,
        ),
    }
    return specs


__all__ = [
    "GROQ",
    "GEMINI",
    "OPENROUTER",
    "KNOWN_PROVIDERS",
    "DEFAULT_PROVIDER_ORDER",
    "ProviderSpec",
    "ProviderConfigError",
    "resolve_provider_order",
    "build_provider_specs",
]
