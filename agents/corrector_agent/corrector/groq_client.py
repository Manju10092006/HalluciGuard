"""Groq-backed generator for the HalluciGuard Corrector.

Production correction path:
authorized target -> evidence-bound revision -> strict JSON candidate ->
HalluciGuard validation -> bounded retry -> deterministic reconstruction.

The implementation uses Groq's OpenAI-compatible Chat Completions HTTP API
directly, so no Groq SDK dependency is required.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


class GroqGenerator:
    """Small synchronous generator implementing the Corrector Generator seam."""

    kind = "groq_api"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "openai/gpt-oss-120b",
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
        reasoning_effort: str = "low",
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))
        self.reasoning_effort = reasoning_effort

    def generate(self, system_text: str, prompt_text: str) -> str:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt_text},
            ],
            "temperature": 0,
            "max_completion_tokens": 256,
            "reasoning_effort": self.reasoning_effort,
            "response_format": {"type": "json_object"},
        }

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    raw = response.read().decode("utf-8")
                data = json.loads(raw)
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError("Groq returned no choices")
                message = choices[0].get("message") or {}
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("Groq returned empty message content")
                return content
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = RuntimeError(f"Groq HTTP {exc.code}: {detail}")
                if exc.code == 429 and attempt < self.max_retries:
                    retry_after = exc.headers.get("retry-after")
                    try:
                        delay = max(0.5, min(float(retry_after), 15.0))
                    except (TypeError, ValueError):
                        delay = min(2.0 ** attempt, 8.0)
                    time.sleep(delay)
                    continue
                raise last_error from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = RuntimeError(f"Groq transport/JSON error: {exc}")
                if attempt < self.max_retries:
                    time.sleep(min(2.0 ** attempt, 8.0))
                    continue
                raise last_error from exc

        raise last_error or RuntimeError("Groq generation failed")
