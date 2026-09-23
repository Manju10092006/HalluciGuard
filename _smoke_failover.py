"""Forced-failover live test: simulate Groq 429, confirm Gemini serves it.

Does NOT change production config. Groq's transport is monkeypatched in-process
to raise a 429 the first time; Gemini and OpenRouter keep their real transport,
so the fallback actually hits the live Gemini endpoint. Prints NO secrets.
"""
import asyncio
import os

import httpx


def _load_env(path=".env"):
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


_load_env()

from services.base_llm_service import BaseLLMService  # noqa: E402
from services.llm_providers import GROQ  # noqa: E402


class GroqRateLimitedService(BaseLLMService):
    """Real router, except Groq always returns HTTP 429 (rate limited)."""

    async def _post_chat_completions_to(self, spec, payload):
        if spec.name == GROQ:
            return httpx.Response(
                429,
                text="simulated rate limit",
                request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
            )
        return await super()._post_chat_completions_to(spec, payload)


async def main():
    svc = GroqRateLimitedService()
    res = await svc.generate("Who created the Java programming language? One sentence.")
    print("status        :", res.status)
    print("provider_used :", res.provider_used)
    attempts = [(a.get("provider"), a.get("outcome")) for a in (res.provider_attempts or [])]
    print("attempts      :", attempts)
    print("response[:160]:", (res.draft_response or "")[:160])

    assert res.status == "success", "expected success via fallback"
    assert res.provider_used == "gemini", f"expected gemini, got {res.provider_used}"
    print("\nFORCED FAILOVER OK: groq 429 -> gemini served the request.")


if __name__ == "__main__":
    asyncio.run(main())
