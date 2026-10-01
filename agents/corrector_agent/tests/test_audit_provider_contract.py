"""Provider contract and sanitized failure regression tests."""
import pytest
from agents.corrector_agent.corrector.config import CorrectorConfig
from agents.corrector_agent.corrector.model_client import ModelClient
from agents.corrector_agent.corrector.groq_client import GroqGenerator


def test_default_provider_is_local_and_env_is_explicit(monkeypatch):
    assert CorrectorConfig().provider == "local"
    monkeypatch.setenv("HG_CORRECTOR_PROVIDER", "groq")
    assert CorrectorConfig.from_env().provider == "groq"


@pytest.mark.parametrize("kwargs", [
    {"provider": "unknown"}, {"groq_timeout_seconds": float("nan")},
    {"groq_timeout_seconds": 0}, {"groq_max_retries": -1}, {"groq_max_retries": 11}])
def test_invalid_provider_settings_fail_before_io(kwargs):
    with pytest.raises(ValueError):
        CorrectorConfig(**kwargs)


def test_groq_construction_uses_transport_settings():
    generator = GroqGenerator.from_config(
        CorrectorConfig(provider="groq", groq_max_retries=3, groq_timeout_seconds=12),
        api_key="dummy-secret")
    assert generator.max_retries == 3
    assert generator.timeout_seconds == 12


def test_provider_initialization_exception_is_sanitized(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy-secret")
    def fail(*a, **kw):
        raise RuntimeError("dummy-secret private prompt")
    monkeypatch.setattr(GroqGenerator, "from_config", fail)
    status = ModelClient(CorrectorConfig(provider="groq")).ensure_loaded()
    assert not status.available
    assert "dummy-secret" not in status.detail
    assert "RuntimeError" in status.detail
