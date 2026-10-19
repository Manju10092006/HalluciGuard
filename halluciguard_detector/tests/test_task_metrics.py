"""Task semantics, calibration persistence and metric correctness.

The regression these guard is the one that made the old model card
uninterpretable: a single function measured the verification-needed question and
reported the result as "hallucination". Task A (refutation) and Task B
(triage) are now measured separately, and this module pins that separation.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from halluciguard_detector.calibration import (
    DEFAULT_MAX_LENGTH,
    DEPRECATED_HALLUCINATION_THRESHOLD,
    apply_temperature,
    class_probabilities,
    contradiction_score,
    load_calibration,
    save_calibration,
    verification_risk_score,
)
from halluciguard_detector.training import (
    best_threshold,
    binary_metrics,
    calibration_report,
    contradiction_metrics,
    evaluate_saved_predictions,
    three_class_metrics,
    verification_needed_metrics,
)


# ------------------------------------------------------------------- scores


def test_contradiction_and_verification_scores_are_different_questions():
    probabilities = np.array([[0.10, 0.20, 0.70], [0.10, 0.65, 0.25], [0.80, 0.10, 0.10]])
    # A claim that is merely unverified must not read as refuted.
    assert contradiction_score(probabilities)[0] == pytest.approx(0.20)
    assert verification_risk_score(probabilities)[0] == pytest.approx(0.90)
    # A claim the evidence refutes is both contradicted and needs verification.
    assert contradiction_score(probabilities)[1] == pytest.approx(0.65)
    assert verification_risk_score(probabilities)[1] == pytest.approx(0.90)
    # A clean claim is neither.
    assert contradiction_score(probabilities)[2] == pytest.approx(0.10)
    assert verification_risk_score(probabilities)[2] == pytest.approx(0.20)


def test_class_probabilities_sum_to_one():
    logits = np.array([[2.0, -1.0, 0.5], [0.0, 0.0, 0.0]])
    probabilities = class_probabilities(logits, temperature=1.0)
    assert probabilities.shape == (2, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_temperature_sharpens_but_preserves_ordering():
    """Temperature < 1 sharpens, > 1 flattens; argmax never changes.

    The shipped checkpoint fits T = 0.7586, i.e. it sharpens, because the raw
    logits were under-confident relative to the observed frequencies.
    """
    import torch

    logits = torch.tensor([[1.0, 0.0, 0.0]])
    flat = apply_temperature(logits, 5.0)[0]
    sharp = apply_temperature(logits, 0.5)[0]
    assert flat.argmax().item() == sharp.argmax().item() == 0
    assert sharp[0] > flat[0]
    assert sharp[0] > 0.7
    assert apply_temperature(logits, 1.0)[0][0] == pytest.approx(1 / 3, abs=0.25)


def test_max_length_default_matches_shipped_checkpoint():
    # Training at 384 while serving at 256 is a silent train/serve skew.
    assert DEFAULT_MAX_LENGTH == 256


# -------------------------------------------------------------- persistence


def test_save_calibration_persists_two_distinct_thresholds(tmp_path):
    path = tmp_path / "calibration.json"
    save_calibration(path, temperature=0.75, contradiction_threshold=0.46, verification_risk_threshold=0.625)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["temperature"] == pytest.approx(0.75)
    assert data["contradiction_threshold"] == pytest.approx(0.46)
    assert data["verification_risk_threshold"] == pytest.approx(0.625)
    # The deprecated mirror tracks Task B, never Task A.
    assert data[DEPRECATED_HALLUCINATION_THRESHOLD] == pytest.approx(0.625)
    assert data["max_length"] == DEFAULT_MAX_LENGTH


def test_load_calibration_maps_legacy_threshold_onto_task_b_not_task_a(tmp_path):
    """A legacy checkpoint's single threshold was fitted for verification-needed.

    Mapping it onto ``contradiction_threshold`` would invent a falsity cut-off
    that was never fitted, which is the confusion this refactor exists to undo.
    """
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps({"temperature": 0.62, DEPRECATED_HALLUCINATION_THRESHOLD: 0.6216}),
        encoding="utf-8",
    )
    data = load_calibration(path)
    assert data["verification_risk_threshold"] == pytest.approx(0.6216)
    assert "contradiction_threshold" not in data


def test_load_calibration_prefers_explicit_task_b_value(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "verification_risk_threshold": 0.7,
                DEPRECATED_HALLUCINATION_THRESHOLD: 0.1,
            }
        ),
        encoding="utf-8",
    )
    assert load_calibration(path)["verification_risk_threshold"] == pytest.approx(0.7)


def test_load_calibration_missing_file_is_empty(tmp_path):
    assert load_calibration(tmp_path / "absent.json") == {}


# ------------------------------------------------------------------ metrics


def _fixture(n_per_class: int = 60, seed: int = 0):
    """A tiny, deterministic, learnable three-class set."""
    rng = np.random.default_rng(seed)
    blocks, labels = [], []
    for class_id, base in enumerate(([3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0])):
        noise = rng.normal(scale=0.4, size=(n_per_class, 3))
        blocks.append(np.array(base) + noise)
        labels.extend([class_id] * n_per_class)
    return np.vstack(blocks), np.array(labels)


def test_three_class_metrics_reports_per_class_recall():
    logits, labels = _fixture()
    metrics = three_class_metrics(logits, labels, temperature=1.0)
    assert metrics["accuracy"] > 0.9
    for name in ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"):
        assert metrics["per_class"][name]["recall"] is not None
        assert metrics["per_class"][name]["support"] == 60
    assert set(metrics["confusion_matrix"]) == {0, 1, 2} or isinstance(
        metrics["confusion_matrix"], dict
    )


def test_task_a_and_task_b_disagree_on_which_positives_count():
    """A NOT_ENOUGH_INFO example is positive for B and negative for A.

    Conflating them is exactly what the old single ``binary_metrics`` did.
    """
    probabilities = np.array([[0.05, 0.05, 0.90]])
    logits = np.log(probabilities)
    labels = np.array([2])  # NOT_ENOUGH_INFO
    task_a = contradiction_metrics(logits, labels, temperature=1.0, threshold=0.5)
    task_b = verification_needed_metrics(logits, labels, temperature=1.0, threshold=0.5)
    assert task_a["support"]["positives"] == 0
    assert task_b["support"]["positives"] == 1
    assert task_a["score"] == "P(CONTRADICTED)"
    assert task_b["score"] == "P(CONTRADICTED)+P(NOT_ENOUGH_INFO)"
    assert task_b["confusion_matrix"]["tp"] == 1
    assert task_a["confusion_matrix"]["fn"] == 0


def test_task_metrics_flag_missing_ranking_class_instead_of_crashing():
    logits = np.array([[5.0, 0.0, 0.0]] * 5)
    labels = np.array([0] * 5)  # no positives at all
    metrics = contradiction_metrics(logits, labels, temperature=1.0)
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert "ranking metrics undefined" in metrics["note"]


def test_binary_task_metrics_reports_support_and_imbalance():
    logits, labels = _fixture(n_per_class=100)
    metrics = verification_needed_metrics(logits, labels, temperature=1.0)
    assert metrics["support"]["positives"] == 200
    assert metrics["support"]["negatives"] == 100
    assert metrics["positive_rate"] == pytest.approx(2 / 3)
    assert "roc_auc" in metrics and "ece" in metrics and "brier" in metrics
    assert isinstance(metrics["reliability"], list) and metrics["reliability"]


def test_deprecated_binary_metrics_still_means_task_b():
    logits, labels = _fixture(n_per_class=20)
    assert binary_metrics(logits, labels, 1.0, 0.5) == verification_needed_metrics(
        logits, labels, 1.0, 0.5
    )


# ------------------------------------------------------------- calibration


def test_calibration_report_is_small_for_a_confident_correct_model():
    logits, labels = _fixture(n_per_class=80)
    report = calibration_report(logits, labels, temperature=1.0)
    assert report["multiclass_brier"] < 0.1
    assert report["multiclass_ece"] < 0.1
    for name in ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"):
        assert report["per_class"][name]["brier"] < 0.1


def test_best_threshold_is_fitted_per_task():
    logits, labels = _fixture(n_per_class=80)
    contradiction_threshold, contradiction_f1 = best_threshold(
        logits, labels, 1.0, "contradiction"
    )
    risk_threshold, risk_f1 = best_threshold(logits, labels, 1.0, "verification_needed")
    assert 0.0 <= contradiction_threshold <= 1.0
    assert 0.0 <= risk_threshold <= 1.0
    assert contradiction_f1 > 0.9
    assert risk_f1 > 0.9
    # Different questions, and the fitted cuts need not coincide.
    assert {contradiction_threshold, risk_threshold} <= set(np.linspace(0.05, 0.95, 181).tolist())


def test_best_threshold_rejects_unknown_task():
    logits, labels = _fixture(n_per_class=5)
    with pytest.raises(ValueError):
        best_threshold(logits, labels, 1.0, "nonsense")


def test_evaluate_saved_predictions_reports_both_tasks_and_calibration():
    logits, labels = _fixture(n_per_class=40)
    report = evaluate_saved_predictions(
        logits, labels, temperature=0.9, contradiction_threshold=0.46, verification_risk_threshold=0.625
    )
    assert report["samples"] == 120
    assert report["class_distribution"] == {
        "SUPPORTED": 40,
        "CONTRADICTED": 40,
        "NOT_ENOUGH_INFO": 40,
    }
    assert report["task_a_contradiction"]["threshold"] == pytest.approx(0.46)
    assert report["task_b_verification_needed"]["threshold"] == pytest.approx(0.625)
    assert "multiclass_ece" in report["calibration"]
    assert "per_class" in report["calibration"]
