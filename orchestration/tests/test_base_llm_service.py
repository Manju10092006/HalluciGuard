from __future__ import annotations

import httpx
import pytest

from services.base_llm_service import BaseLLMConfig, BaseLLMService


class StubOpenRouterService(BaseLLMService):
    def __init__(self, responses):
        super().__init__(
            BaseLLMConfig(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="qwen/qwen3-4b",
                max_retries=3,
                timeout_seconds=1,
            )
        )
        self.responses = list(responses)
        self.calls = 0

    async def _post_chat_completions(self, payload):
        self.calls += 1
        next_item = self.responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item

    def _retry_delay_seconds(self, attempt: int) -> float:
        return 0


def json_response(status_code: int, payload: dict) -> httpx.Response:
    return httpx.Response(
        status_code, json=payload, request=httpx.Request("POST", "https://example.test")
    )


@pytest.mark.asyncio
async def test_openrouter_success_returns_draft_and_usage():
    service = StubOpenRouterService(
        [
            json_response(
                200,
                {
                    "model": "qwen/qwen3-4b",
                    "choices": [
                        {"message": {"content": "Connected."}, "finish_reason": "stop"}
                    ],
                    "usage": {"total_tokens": 3},
                },
            )
        ]
    )
    result = await service.generate("Respond with the single word: Connected.")
    assert result.status == "success"
    assert result.user_query == "Respond with the single word: Connected."
    assert result.provider == "openrouter"
    assert result.model == "qwen/qwen3-4b"
    assert result.draft_response == "Connected."
    assert result.usage["total_tokens"] == 3


@pytest.mark.asyncio
async def test_missing_key_fails_without_network_call():
    service = BaseLLMService(BaseLLMConfig(api_key=None))
    result = await service.generate("hello")
    assert result.status == "failed"
    assert result.error_code == "MISSING_API_KEY"
    assert result.draft_response == ""


@pytest.mark.asyncio
async def test_429_uses_bounded_retry_then_success():
    service = StubOpenRouterService(
        [
            httpx.Response(
                429,
                text="rate limited",
                request=httpx.Request("POST", "https://example.test"),
            ),
            json_response(
                200,
                {
                    "model": "qwen/qwen3-4b",
                    "choices": [{"message": {"content": "Connected"}}],
                },
            ),
        ]
    )
    result = await service.generate("Respond with the single word: Connected.")
    assert service.calls == 2
    assert result.status == "success"


@pytest.mark.asyncio
async def test_401_is_not_retried():
    service = StubOpenRouterService(
        [
            httpx.Response(
                401,
                text="bad auth",
                request=httpx.Request("POST", "https://example.test"),
            ),
            json_response(
                200, {"choices": [{"message": {"content": "should not happen"}}]}
            ),
        ]
    )
    result = await service.generate("hello")
    assert service.calls == 1
    assert result.status == "failed"
    assert result.error_code == "HTTP_401"


@pytest.mark.asyncio
async def test_empty_content_is_generation_failure():
    service = StubOpenRouterService(
        [json_response(200, {"choices": [{"message": {"content": ""}}]})]
    )
    result = await service.generate("hello")
    assert result.status == "failed"
    assert result.error_code == "EMPTY_CONTENT"
    assert result.draft_response == ""


@pytest.mark.asyncio
async def test_timeout_retries_are_bounded():
    service = StubOpenRouterService(
        [
            httpx.TimeoutException("timed out"),
            httpx.TimeoutException("timed out"),
            httpx.TimeoutException("timed out"),
            httpx.TimeoutException("timed out"),
        ]
    )
    result = await service.generate("hello")
    assert service.calls == 4
    assert result.status == "failed"
    assert result.error_code == "TIMEOUT"
