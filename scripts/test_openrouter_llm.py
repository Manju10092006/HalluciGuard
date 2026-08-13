from __future__ import annotations

import asyncio
import os
import sys

from services.base_llm_service import BaseLLMConfig, BaseLLMService


def load_local_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


async def main() -> int:
    load_local_dotenv()
    config = BaseLLMConfig()
    if not config.api_key:
        print("OPENROUTER_API_KEY configured: no")
        return 2
    print("OPENROUTER_API_KEY configured: yes")
    result = await BaseLLMService(config).generate(
        user_query="Respond with the single word: Connected.",
        conversation_history=[],
        generation_mode="normal",
        temperature=config.temperature,
        max_tokens=16,
    )
    print(f"provider: {result.provider}")
    print(f"model: {result.model}")
    print(f"status: {result.status}")
    print(f"http_status: {'200' if result.status == 'success' else result.error_code}")
    print(f"latency_ms: {result.latency_ms}")
    print(f"content: {result.draft_response}")
    assert result.status == "success", result.error
    assert result.draft_response.strip(), "OpenRouter returned empty content"
    if "connected" not in result.draft_response.strip().lower():
        print("warning: returned content did not contain the word Connected")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
