from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
        print("SKIP: Real OpenRouter integration test skipped because OPENROUTER_API_KEY is not set.")
        return 2

    print("OPENROUTER_API_KEY configured: yes")
    
    # Message structure check
    messages = [
        {
            "role": "user",
            "content": "Respond with the single word: Connected."
        }
    ]
    user_query = messages[0]["content"]

    service = BaseLLMService(config)
    result = await service.generate(
        user_query=user_query,
        conversation_history=[],
        generation_mode="normal",
        temperature=config.temperature,
        max_tokens=16,
    )
    
    dumped = result.model_dump()
    print(f"user_query: {result.user_query}")
    print(f"provider: {result.provider}")
    print(f"model: {result.model}")
    print(f"status: {result.status}")
    print(f"http_status: {'200' if result.status == 'success' else result.error_code}")
    print(f"latency_ms: {result.latency_ms}")
    print(f"content: {result.draft_response}")

    # Verify security: Ensure API key is NOT present anywhere in dumped output
    dump_str = str(dumped)
    assert config.api_key not in dump_str, "SECURITY ERROR: OPENROUTER_API_KEY found in result dump!"

    # Verify Contract
    assert result.status == "success", f"Generation failed: {result.error}"
    assert result.draft_response.strip(), "OpenRouter returned empty content"
    assert result.provider == "openrouter", f"Expected provider 'openrouter', got '{result.provider}'"
    assert result.latency_ms > 0, "Expected positive latency_ms measurement"

    print("\nREAL OPENROUTER TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
