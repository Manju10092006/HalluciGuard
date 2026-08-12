from __future__ import annotations

import sys
from pathlib import Path

import pytest

VERIFIER_DIR = Path(__file__).resolve().parents[2] / "agents" / "verifier_agent"
if str(VERIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFIER_DIR))

from agents.detector_agent.halueval_inference import validate_halueval_model_reference
from agents.verifier_agent.nli.entailment import NLIEngine
from agents.verifier_agent.api.pipeline import VerificationPipeline
from agents.verifier_agent.schemas.models import Passage


def test_detector_model_validation_rejects_missing_local_path(tmp_path):
    missing = tmp_path / "missing-detector"
    with pytest.raises(
        FileNotFoundError, match="HaluEval detector model directory not found"
    ):
        validate_halueval_model_reference(str(missing))


def test_detector_model_validation_rejects_incomplete_local_path(tmp_path):
    model_dir = tmp_path / "detector"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    with pytest.raises(
        FileNotFoundError, match="Missing required Hugging Face artifact"
    ):
        validate_halueval_model_reference(str(model_dir))


def test_detector_model_validation_accepts_huggingface_identifier():
    assert (
        validate_halueval_model_reference("hf://org/halluciguard-detector")
        == "org/halluciguard-detector"
    )


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
