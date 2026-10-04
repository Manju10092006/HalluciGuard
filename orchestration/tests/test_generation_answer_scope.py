"""Question-focused generation and truthful provider metadata; no network."""
from copy import deepcopy

import pytest

from orchestration import graph


@pytest.mark.asyncio
async def test_normal_generation_uses_real_query_and_preserves_history(monkeypatch):
    observed = []

    class Service:
        async def generate(self, **kwargs):
            observed.append(kwargs)
            return {"status": "success", "draft_response": "A provider-generated answer.",
                    "provider": "groq", "provider_used": "groq", "model": "test-model",
                    "latency_ms": 12}

    monkeypatch.setattr(graph, "BaseLLMService", Service)
    state = {"user_query": "Who founded this company?", "conversation_history": [
        {"role": "user", "content": "Please explain if I ask for detail."}], "trace": []}
    before = deepcopy(state)
    out = await graph._generate_node(state)
    assert state == before
    assert observed[0]["user_query"] == state["user_query"]
    assert observed[0]["conversation_history"][0] == {
        "role": "system", "content": graph._ANSWER_SCOPE_INSTRUCTION}
    assert observed[0]["conversation_history"][1:] == state["conversation_history"]
    assert "Honor any depth" in graph._ANSWER_SCOPE_INSTRUCTION
    assert out["llm_response"] == "A provider-generated answer."
    assert out["trace"][-1]["details"]["provider"] == "groq"
    assert graph._generate_route(out) == "detector"


@pytest.mark.asyncio
async def test_stress_generation_keeps_its_existing_history(monkeypatch):
    observed = []

    class Service:
        async def generate(self, **kwargs):
            observed.append(kwargs)
            return {"status": "success", "draft_response": "Stress answer.",
                    "provider": "fixture", "model": "fixture"}

    monkeypatch.setattr(graph, "BaseLLMService", Service)
    history = [{"role": "user", "content": "Earlier context"}]
    await graph._generate_node({"user_query": "Question", "generation_mode": "stress_test",
                                "conversation_history": history})
    assert observed[0]["conversation_history"] == history
    assert observed[0]["generation_mode"] == "stress_test"


@pytest.mark.asyncio
async def test_supplied_answer_still_skips_initial_generator(monkeypatch):
    class Service:
        def __init__(self):
            raise AssertionError("Supplied-answer diagnostics must not call generation")

    monkeypatch.setattr(graph, "BaseLLMService", Service)
    out = await graph._generate_node({"user_query": "Question", "llm_response": "Supplied answer"})
    assert out["llm_response"] == "Supplied answer"
    assert out["base_llm"]["provider"] == "pre_supplied"
