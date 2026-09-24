"""Unit tests for the hosted Groq Corrector provider."""

from __future__ import annotations

import json

from agents.corrector_agent.corrector.groq_client import GroqGenerator


def test_groq_generator_uses_openai_compatible_endpoint(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "choices": [{
                    "message": {
                        "content": '{"sentence_id":"S1","corrected_sentence":"Java was created by James Gosling."}'
                    }
                }]
            }).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode())
        return FakeResponse()

    monkeypatch.setattr(
        "agents.corrector_agent.corrector.groq_client.urllib.request.urlopen",
        fake_urlopen,
    )

    result = GroqGenerator(
        api_key="test-key",
        model="openai/gpt-oss-120b",
        timeout_seconds=12,
        max_retries=0,
        reasoning_effort="low",
    ).generate("system", "data")

    assert result.startswith('{"sentence_id"')
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["body"]["model"] == "openai/gpt-oss-120b"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["reasoning_effort"] == "low"
    assert captured["timeout"] == 12


def test_groq_generator_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    generator = GroqGenerator(api_key="")

    try:
        generator.generate("system", "data")
    except RuntimeError as exc:
        assert "GROQ_API_KEY" in str(exc)
    else:
        raise AssertionError("expected missing-key failure")
