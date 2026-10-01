"""Regression tests for HG-006 / HG-M2: hosted Corrector provider contract.

render.yaml ships HG_CORRECTOR_PROVIDER=openrouter. Before the fix, CorrectorConfig
rejected any provider except local/groq (so from_env() raised in production) and the
startup validator then fell through to loading the local Qwen model for a hosted run.
These tests pin the corrected contract WITHOUT any network call or model load.
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.corrector_agent.corrector.config import (  # noqa: E402
    CorrectorConfig,
    VALID_CORRECTOR_PROVIDERS,
)
from orchestration.runtime_validation import validate_corrector_configuration  # noqa: E402

_PROVIDER_KEYS = ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")


@pytest.mark.parametrize("provider", sorted(VALID_CORRECTOR_PROVIDERS))
def test_config_accepts_every_valid_provider(provider):
    assert CorrectorConfig(provider=provider).provider == provider


def test_config_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unsupported Corrector provider"):
        CorrectorConfig(provider="bogus")


def test_from_env_openrouter_does_not_raise(monkeypatch):
    monkeypatch.setenv("HG_CORRECTOR_PROVIDER", "openrouter")
    assert CorrectorConfig.from_env().provider == "openrouter"


def test_validate_hosted_uses_router_not_local_model(monkeypatch):
    monkeypatch.setenv("HG_CORRECTOR_PROVIDER", "openrouter")
    for k in _PROVIDER_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "dummy-key-for-readiness-only")
    res = validate_corrector_configuration()
    assert res.metadata["provider"] == "openrouter"
    # Must NOT have taken the local-model path (no model load for a hosted run).
    assert "model_status" not in res.metadata
    assert res.ok is True


def test_validate_hosted_fails_closed_without_any_key(monkeypatch):
    monkeypatch.setenv("HG_CORRECTOR_PROVIDER", "hosted")
    for k in _PROVIDER_KEYS:
        monkeypatch.delenv(k, raising=False)
    res = validate_corrector_configuration()
    assert res.ok is False
    assert "No provider credential" in res.detail
