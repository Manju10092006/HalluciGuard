"""Selector fallbacks disclose error classes, not private requests/messages."""
from types import SimpleNamespace

from halluciguard_detector import evidence


def test_shared_selector_failure_is_sanitized(monkeypatch, caplog):
    monkeypatch.setattr(evidence, "_ensure_verifier", lambda: True)
    monkeypatch.setattr(evidence, "_to_passage", lambda text: text)
    def broken(*args, **kwargs):
        raise RuntimeError("dummy-secret-provider-response")
    monkeypatch.setattr(evidence, "_hybrid_retriever", SimpleNamespace(retrieve=broken))
    monkeypatch.setattr(evidence, "_reranker", object())
    trace = {}
    result = evidence.select_evidence("private-request Earth Sun", ["Earth orbits the Sun."], trace=trace)
    assert result == ["Earth orbits the Sun."]
    assert trace["degraded"] is True
    assert trace["reason"] == "selection failed: RuntimeError"
    assert "dummy-secret" not in str(trace) + caplog.text
    assert "private-request" not in caplog.text
