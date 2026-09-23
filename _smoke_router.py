"""Live smoke test for the multi-provider LLM router. Prints NO secrets."""
import asyncio
import os
import sys

# Minimal .env loader (avoid extra deps); does not print values.
def _load_env(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

_load_env()

from services.base_llm_service import BaseLLMService  # noqa: E402


async def main():
    svc = BaseLLMService()  # no-arg => multi-provider router
    status = svc.provider_status()
    print("PROVIDER STATUS (no secrets):")
    print(status)

    print("\n--- Simple generation ---")
    res = await svc.generate("Who created the Java programming language? Answer in one sentence.")
    print("status        :", res.status)
    print("provider_used :", res.provider_used)
    print("error_code    :", getattr(res, "error_code", None))
    attempts = [(a.get("provider"), a.get("outcome")) for a in (res.provider_attempts or [])]
    print("attempts      :", attempts)
    text = (res.draft_response or "")[:200]
    print("response[:200]:", text)


if __name__ == "__main__":
    asyncio.run(main())
