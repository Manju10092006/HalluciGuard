"""Opt-in free hosted-inference guard. This does not cap cloud hosting bills."""
from __future__ import annotations

import os
from typing import Any
from urllib.parse import parse_qsl, urlsplit


class CostPolicyError(ValueError):
    """A request cannot use the approved free inference route."""


def free_only() -> bool:
    value = os.getenv("HALLUCIGUARD_FREE_ONLY", "").strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"", "0", "false", "no"}:
        return False
    raise CostPolicyError("Invalid HALLUCIGUARD_FREE_ONLY; use true or false")


def guard_hosted_request(provider: str, base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Reject unapproved endpoints, paid models and add-ons before HTTP dispatch."""
    if not free_only():
        return payload
    model = payload.get("model")
    if (
        provider != "openrouter"
        or base_url.rstrip("/") != "https://openrouter.ai/api/v1"
        or not isinstance(model, str)
        or not (model == "openrouter/free" or model.endswith(":free"))
        or ":online" in model
    ):
        raise CostPolicyError("Free-only policy requires the official OpenRouter endpoint and an explicitly free model")
    allowed = {"model", "messages", "temperature", "max_tokens", "response_format"}
    if set(payload) - allowed:
        raise CostPolicyError("Free-only policy rejected unapproved request options")
    messages = payload.get("messages", [])
    if not isinstance(messages, list) or any(
        not isinstance(message, dict) or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise CostPolicyError("Free-only policy permits text messages only")
    return {
        **payload,
        "provider": {"max_price": {"prompt": 0, "completion": 0, "request": 0}},
    }


def guard_unapproved_service(service: str) -> None:
    if free_only():
        raise CostPolicyError(f"Free-only policy blocks {service}: account and downstream costs are unverified")


_PUBLIC_RETRIEVAL_HOSTS = frozenset({
    "en.wikipedia.org", "services.nvd.nist.gov", "cve.circl.lu",
    "api.github.com", "raw.githubusercontent.com", "www.cisa.gov",
    "export.arxiv.org", "api.semanticscholar.org", "api.crossref.org",
    "efts.sec.gov", "api.worldbank.org", "eutils.ncbi.nlm.nih.gov",
    "api.fda.gov", "clinicaltrials.gov", "ghoapi.azureedge.net",
    "www.courtlistener.com",
})
_CREDENTIAL_KEYS = frozenset({
    "authorization", "apikey", "xapikey", "key", "token", "accesstoken",
    "accesskey", "subscriptionkey", "ocpapimsubscriptionkey", "cookie",
})


def guard_public_retrieval_request(method: str, url: str, options: dict[str, Any]) -> None:
    """Permit only the existing anonymous public GET retrieval routes.

    Account-backed retrieval is deliberately unavailable in this profile until
    that account's costs have been established. No credentials are serialized.
    """
    if not free_only():
        return
    target = urlsplit(url)
    if (method.upper() != "GET" or target.scheme != "https"
            or target.hostname not in _PUBLIC_RETRIEVAL_HOSTS
            or target.username or target.password or target.port not in (None, 443)):
        raise CostPolicyError("Free-only policy blocks unapproved public retrieval route")
    params = options.get("params") or {}
    param_items = params.items() if hasattr(params, "items") else params
    headers = options.get("headers") or {}
    header_items = headers.items() if hasattr(headers, "items") else headers
    for key, value in [*parse_qsl(target.query), *param_items, *header_items]:
        normalized = str(key).strip().lower().replace("-", "").replace("_", "")
        if value and normalized in _CREDENTIAL_KEYS:
            raise CostPolicyError("Free-only policy blocks account-dependent retrieval")
    if options.get("auth") or options.get("cookies"):
        raise CostPolicyError("Free-only policy blocks account-dependent retrieval")
    if options.get("follow_redirects"):
        raise CostPolicyError("Free-only policy requires retrieval redirects to remain disabled")
