"""HTTP client for the HalluciGuard Verifier Agent.

The Detector remains responsible for first-stage risk classification. This
client is invoked only when the Detector returns HIGH / Verify. The Verifier
is treated as an independent service so both agents keep their existing
interfaces and lifecycle.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import httpx


class VerifierUnavailableError(RuntimeError):
    """Raised when a HIGH-risk result cannot be handed to the Verifier."""


class VerifierClient:
    """Small, typed HTTP client for the Verifier Agent /verify endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("VERIFIER_AGENT_URL", "http://127.0.0.1:8001")).rstrip("/")
        self.timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv("VERIFIER_AGENT_TIMEOUT_SECONDS", "60")
        )

    async def health(self) -> Dict[str, Any]:
        """Return Verifier health payload or raise VerifierUnavailableError."""
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout_seconds, 10.0)) as client:
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise VerifierUnavailableError(
                f"Verifier Agent is unavailable at {self.base_url}: {exc}"
            ) from exc

    async def verify(
        self,
        *,
        query_id: str,
        domain: str,
        claim_text: str,
    ) -> Dict[str, Any]:
        """Send exactly one suspicious claim to the Verifier /verify endpoint."""
        payload = {
            "query_id": query_id,
            "domain": domain,
            "suspicious_claims": [
                {
                    "claim_id": f"{query_id}:claim-1",
                    "text": claim_text,
                }
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/verify",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("Verifier returned a non-object JSON response")
                return data
        except (httpx.HTTPError, ValueError) as exc:
            raise VerifierUnavailableError(
                f"Verifier Agent request failed at {self.base_url}/verify: {exc}"
            ) from exc


__all__ = ["VerifierClient", "VerifierUnavailableError"]
