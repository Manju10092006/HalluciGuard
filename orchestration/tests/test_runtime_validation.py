from __future__ import annotations

import sys
from pathlib import Path

import pytest

VERIFIER_DIR = Path(__file__).resolve().parents[2] / "agents" / "verifier_agent"
if str(VERIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFIER_DIR))

from orchestration.runtime_validation import validate_detector_model_reference
from agents.verifier_agent.nli.entailment import NLIEngine
from agents.verifier_agent.api.pipeline import VerificationPipeline
from agents.verifier_agent.schemas.models import Passage


def test_detector_model_validation_ok_when_artifact_complete():
    """The package and every required local model artifact must be present."""
    result = validate_detector_model_reference()
    assert result.ok is True
    assert result.component == "detector"
    md = result.metadata
    assert md["detector"] == "halluciguard_detector"
    assert md["model_source"] == "local_safetensors"
    assert md["grounding_required"] is True
    assert md["missing_files"] == []


def test_detector_model_validation_fails_closed_when_artifact_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HALLUCIGUARD_DETECTOR_MODEL", str(tmp_path))
    result = validate_detector_model_reference()
    assert result.ok is False
    assert result.component == "detector"
    assert "incomplete" in result.detail.lower()
    assert "model.safetensors" in result.metadata["missing_files"]


def test_detector_model_validation_fails_closed_when_package_unimportable(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "halluciguard_detector" or name.startswith("halluciguard_detector."):
            raise ImportError("simulated missing halluciguard_detector package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    result = validate_detector_model_reference()
    assert result.ok is False
    assert result.component == "detector"
    assert "not importable" in result.detail.lower()


def test_nli_unavailable_is_degraded_not_uniform_success(monkeypatch):
    engine = NLIEngine()
    monkeypatch.setattr(
        engine, "_load_model", lambda: setattr(engine, "_is_available", False)
    )
    result = engine.classify(
        "Paris is in France", "Paris is the capital city of France."
    )
    assert result["degraded"] is True
    assert result["entailment_score"] == 0.0
    assert result["contradiction_score"] == 0.0
    assert result["neutral_score"] == 1.0


def test_degraded_nli_is_not_decision_grade_evidence():
    passage = Passage(
        title="t",
        source="s",
        url="https://example.com",
        publication_date="2026-01-01",
        snippet="evidence",
    )
    selected, selected_nli = VerificationPipeline._select_decision_grade_evidence(
        [passage],
        [
            {
                "degraded": True,
                "entailment_score": 0.99,
                "contradiction_score": 0.0,
                "neutral_score": 0.01,
            }
        ],
    )
    assert selected == []
    assert selected_nli == []
