"""Configuration checks must agree with dispatch without loading ML or using HTTP."""
from __future__ import annotations

import builtins

import httpx
import pytest

from orchestration.runtime_validation import (
    validate_corrector_configuration,
    validate_openrouter_configuration,
)


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    for name in (
        "HALLUCIGUARD_FREE_ONLY", "HALLUCIGUARD_LLM_PROVIDER_ORDER",
        "HALLUCIGUARD_LLM_PROVIDER", "HALLUCIGUARD_LLM_MODEL",
        "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "OPENROUTER_BASE_URL",
        "OPENROUTER_MAX_TOKENS", "OPENROUTER_MAX_RETRIES",
        "OPENROUTER_TIMEOUT_SECONDS", "OPENROUTER_TEMPERATURE",
        "OPENROUTER_STRESS_TEMPERATURE", "HALLUCIGUARD_LLM_TEMPERATURE",
        "HALLUCIGUARD_LLM_TIMEOUT", "HG_CORRECTOR_OPENROUTER_MODEL",
        "HG_CORRECTOR_PROVIDER", "HG_CORRECTOR_MODEL_PATH",
        "HG_CORRECTOR_ALLOW_BASE_MODEL_FALLBACK", "GROQ_API_KEY", "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HALLUCIGUARD_FREE_ONLY", "true")
    monkeypatch.setenv("HALLUCIGUARD_LLM_PROVIDER_ORDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-not-a-real-key")
    monkeypatch.setenv("HALLUCIGUARD_LLM_MODEL", "fixture/model:free")

    def no_http(*args, **kwargs):
        raise AssertionError("Configuration validation must not construct an HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", no_http)
    monkeypatch.setattr(httpx, "Client", no_http)
    real_import = builtins.__import__

    def no_ml(name, *args, **kwargs):
        assert name.split(".")[0] not in {
            "torch", "transformers", "spacy", "sentence_transformers", "faiss", "tensorflow",
        }, "Configuration checks must not import ML libraries"
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ml)


@pytest.mark.parametrize("updates,expected", [
    ({}, True),
    ({"OPENROUTER_API_KEY": " "}, False),
    ({"HALLUCIGUARD_LLM_MODEL": "paid/model", "OPENROUTER_MODEL": "fixture/model:free"}, False),
    ({"HALLUCIGUARD_LLM_MODEL": "fixture/model:free", "OPENROUTER_MODEL": "paid/model"}, True),
    ({"OPENROUTER_BASE_URL": "https://unapproved.example/api/v1"}, False),
    ({"HALLUCIGUARD_LLM_PROVIDER_ORDER": "groq,openrouter"}, False),
    ({"HALLUCIGUARD_LLM_PROVIDER_ORDER": ""}, False),
    ({"HALLUCIGUARD_FREE_ONLY": "invalid"}, False),
])
def test_base_readiness_respects_runtime_model_and_cost_guard(monkeypatch, updates, expected):
    for name, value in updates.items():
        monkeypatch.setenv(name, value)
    result = validate_openrouter_configuration()
    assert result.ok is expected
    assert result.metadata["configuration_only"] is True
    assert result.metadata["network_checked"] is False
    assert "fixture-not-a-real-key" not in str(result.model_dump())


def test_base_reports_actual_multi_provider_model_precedence(monkeypatch):
    from services.base_llm_service import BaseLLMService
    monkeypatch.setenv("OPENROUTER_MODEL", "paid/ignored")
    configured = BaseLLMService()._provider_specs["openrouter"]
    result = validate_openrouter_configuration()
    assert result.ok is True
    assert result.metadata["providers"][0]["model"] == configured.model == "fixture/model:free"


def test_normal_router_can_be_configured_with_groq_and_no_openrouter_key(monkeypatch):
    monkeypatch.setenv("HALLUCIGUARD_FREE_ONLY", "false")
    monkeypatch.setenv("HALLUCIGUARD_LLM_PROVIDER_ORDER", "groq,openrouter")
    monkeypatch.setenv("GROQ_API_KEY", "fixture-groq-key")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    result = validate_openrouter_configuration()
    assert result.ok is True
    assert result.metadata["order"] == ["groq", "openrouter"]
    assert result.metadata["network_checked"] is False


@pytest.mark.parametrize("updates,expected", [
    ({}, True),
    ({"OPENROUTER_API_KEY": " "}, False),
    ({"HALLUCIGUARD_LLM_MODEL": "paid/model", "OPENROUTER_MODEL": "fixture/model:free"}, False),
    ({"HG_CORRECTOR_OPENROUTER_MODEL": "paid/model"}, False),
    ({"HALLUCIGUARD_LLM_MODEL": "paid/model", "HG_CORRECTOR_OPENROUTER_MODEL": "fixture/correction:free"}, True),
    ({"OPENROUTER_BASE_URL": "https://unapproved.example/api/v1"}, False),
    ({"HALLUCIGUARD_LLM_PROVIDER": "groq"}, False),
    ({"HALLUCIGUARD_FREE_ONLY": "invalid"}, False),
    # The current Corrector uses explicit single-provider BaseLLMConfig, so a
    # multi-provider order or hosted selector does not change its dispatch route.
    ({"HALLUCIGUARD_LLM_PROVIDER_ORDER": "groq", "HG_CORRECTOR_PROVIDER": "groq"}, True),
])
def test_corrector_readiness_respects_actual_dispatch_config(monkeypatch, updates, expected):
    for name, value in updates.items():
        monkeypatch.setenv(name, value)
    result = validate_corrector_configuration()
    assert result.ok is expected
    assert result.metadata["configuration_only"] is True
    assert result.metadata["network_checked"] is False
    assert "fixture-not-a-real-key" not in str(result.model_dump())


def test_corrector_metadata_matches_character_regenerator_override(monkeypatch):
    from services.character_regenerator import CharacterRegenerator
    monkeypatch.setenv("HG_CORRECTOR_OPENROUTER_MODEL", "fixture/correction:free")
    configured = CharacterRegenerator()._service.config
    result = validate_corrector_configuration()
    assert result.ok is True
    assert result.metadata["provider"] == configured.provider
    assert result.metadata["model"] == configured.model == "fixture/correction:free"


def test_local_corrector_resolution_does_not_certify_model_loading(monkeypatch, tmp_path):
    monkeypatch.setenv("HG_CORRECTOR_PROVIDER", "local")
    monkeypatch.setenv("HG_CORRECTOR_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("HG_CORRECTOR_ALLOW_BASE_MODEL_FALLBACK", "false")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").touch()
    result = validate_corrector_configuration()
    assert result.ok is True  # Filesystem resolution only; the fake weights are never loaded.
    assert result.metadata["configuration_only"] is True
    assert result.metadata["model_loaded"] is False
    assert "untested" in result.detail
