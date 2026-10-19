"""The artifact must be self-describing and internally consistent.

These tests build a synthetic checkpoint rather than loading the real 283 MB
one, so they run in milliseconds while still pinning the cross-checks that
matter: a tokenizer truncating differently from calibration is a silent
train/serve skew, and it is exactly the drift that went unnoticed before.
"""
from __future__ import annotations

import json

import pytest

from halluciguard_detector.model_card import (
    build_metadata,
    recompute_evaluation,
)


def _checkpoint(tmp_path, *, tokenizer_max=256, calibration_max=256, extra_calibration=None):
    checkpoint = tmp_path / "detector"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"_name_or_path": "microsoft/deberta-v3-xsmall", "id2label": {"0": "SUPPORTED", "1": "CONTRADICTED", "2": "NOT_ENOUGH_INFO"}}),
        encoding="utf-8",
    )
    (checkpoint / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": tokenizer_max}), encoding="utf-8"
    )
    calibration = {
        "temperature": 0.7586,
        "contradiction_threshold": 0.46,
        "verification_risk_threshold": 0.625,
        "hallucination_threshold": 0.625,
        "max_length": calibration_max,
    }
    calibration.update(extra_calibration or {})
    (checkpoint / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    return checkpoint


def test_metadata_is_internally_consistent(tmp_path):
    metadata = build_metadata(_checkpoint(tmp_path))
    assert metadata["consistency"]["ok"] is True
    assert metadata["consistency"]["conflicts"] == []
    assert metadata["sequence_length"] == 256
    assert metadata["base_model"] == "microsoft/deberta-v3-xsmall"


def test_tokenizer_calibration_length_mismatch_is_reported_not_hidden(tmp_path):
    metadata = build_metadata(_checkpoint(tmp_path, tokenizer_max=384, calibration_max=256))
    assert metadata["consistency"]["ok"] is False
    assert any("model_max_length" in c for c in metadata["consistency"]["conflicts"])


def test_metadata_reports_the_real_base_model_from_config(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_name_or_path"] = "some/other-model"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert build_metadata(checkpoint)["base_model"] == "some/other-model"


def test_id2label_is_normalised_to_canonical_names(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["id2label"] = {"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    metadata = build_metadata(checkpoint)
    assert metadata["id2label"] == {0: "SUPPORTED", 1: "CONTRADICTED", 2: "NOT_ENOUGH_INFO"}


def test_recompute_evaluation_rebuilds_both_tasks_from_predictions(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    import numpy as np

    rng = np.random.default_rng(0)
    logits = np.vstack(
        [
            np.array([3.0, 0.0, 0.0]) + rng.normal(scale=0.2, size=(20, 3)),  # SUPPORTED
            np.array([0.0, 3.0, 0.0]) + rng.normal(scale=0.2, size=(20, 3)),  # CONTRADICTED
            np.array([0.0, 0.0, 3.0]) + rng.normal(scale=0.2, size=(20, 3)),  # NOT_ENOUGH_INFO
        ]
    )
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    np.savez(checkpoint / "test_predictions.npz", logits=logits, labels=labels)

    report = recompute_evaluation(checkpoint)
    assert report["samples"] == 60
    assert report["task_a_contradiction"]["score"] == "P(CONTRADICTED)"
    assert report["task_b_verification_needed"]["score"] == "P(CONTRADICTED)+P(NOT_ENOUGH_INFO)"
    # Both tasks are present, so neither can be silently conflated again.
    assert "task_a_contradiction" in report
    assert "task_b_verification_needed" in report


def test_recompute_evaluation_requires_predictions(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError):
        recompute_evaluation(checkpoint)
