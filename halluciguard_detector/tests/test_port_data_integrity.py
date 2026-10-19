import json

import numpy as np
import pytest

from halluciguard_detector.calibration import class_probabilities, load_calibration
from halluciguard_detector.data import prepare_ragtruth
from halluciguard_detector.training import evaluate, train


def test_one_source_split_fails_before_writing(tmp_path):
    dataset = tmp_path / "raw"
    dataset.mkdir()
    (dataset / "source_info.jsonl").write_text(json.dumps({"source_id": "s", "source_info": "Paris is in France"}) + "\n")
    rows = [{"id": i, "source_id": "s", "split": split, "quality": "good", "response": "Paris is in France.", "labels": []}
            for i, split in enumerate(("train", "test"))]
    (dataset / "response.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = tmp_path / "prepared"
    with pytest.raises(ValueError, match="source groups"):
        prepare_ragtruth(dataset, out, evidence_shape="lexical_top1")
    assert not out.exists()


@pytest.mark.parametrize("bad_row", ["not json", json.dumps({"id": 1, "source_id": "s", "split": "train", "quality": "good", "response": "Paris.", "labels": [{"start": 5, "end": 10, "label_type": "Evident Conflict"}]}), json.dumps({"id": 1, "source_id": "s", "split": "train", "quality": "good", "response": "Paris.", "labels": [{"start": 0, "end": 5, "label_type": "Unknown New Label"}]})])
def test_malformed_row_span_or_label_fails_without_artifact(tmp_path, bad_row):
    dataset = tmp_path / "raw"
    dataset.mkdir()
    (dataset / "source_info.jsonl").write_text(json.dumps({"source_id": "s", "source_info": "Paris is in France."}) + "\n")
    (dataset / "response.jsonl").write_text(bad_row + "\n")
    out = tmp_path / "prepared"
    with pytest.raises(ValueError, match="response.jsonl row"):
        prepare_ragtruth(dataset, out, evidence_shape="lexical_top1")
    assert not out.exists()


def test_training_protects_artifacts_before_model_load(tmp_path):
    out = tmp_path / "model"
    out.mkdir()
    (out / "model.safetensors").write_bytes(b"original")
    with pytest.raises(FileExistsError):
        train(tmp_path / "missing", out)
    assert (out / "model.safetensors").read_bytes() == b"original"


def test_training_rejects_overlap_and_empty_splits_before_model_load(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    row = {"claim": "Paris is in France.", "evidence": "Paris is in France.",
           "label": "SUPPORTED", "label_id": 0, "source_id": "same"}
    (data / "train.jsonl").write_text(json.dumps(row) + "\n")
    (data / "dev.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="overlap"):
        train(data, tmp_path / "model")
    (data / "dev.jsonl").write_text("")
    with pytest.raises(ValueError, match="nonempty"):
        train(data, tmp_path / "model")


def test_evaluation_length_and_artifact_guards(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "calibration.json").write_text(json.dumps({"max_length": 256, "temperature": 1.0}))
    with pytest.raises(ValueError, match="max_length"):
        evaluate(tmp_path, model, max_length=128)
    (model / "test_metrics.json").write_text("untouched")
    with pytest.raises(FileExistsError):
        evaluate(tmp_path, model)
    assert (model / "test_metrics.json").read_text() == "untouched"


def test_invalid_calibration_and_logits_rejected(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"temperature": float("nan")}))
    with pytest.raises(ValueError):
        load_calibration(path)
    with pytest.raises(ValueError):
        class_probabilities(np.array([[float("inf"), 0, 1]]))


@pytest.mark.parametrize("labels", [[.5], [3], [float("nan")], [], [[0]], [0, 1]])
def test_evaluation_rejects_invalid_or_misaligned_gold_labels(labels):
    from halluciguard_detector.training import evaluate_saved_predictions
    with pytest.raises(ValueError):
        evaluate_saved_predictions(np.array([[1., 0., 0.]]), np.asarray(labels), 1., .5, .5)


def test_evaluation_metadata_does_not_claim_test_is_dev():
    from halluciguard_detector.training import evaluate_saved_predictions
    result = evaluate_saved_predictions(np.array([[1., 0., 0.]]), np.array([0]), 1., .5, .5)
    assert "supplied evaluation labels" in result["calibration"]["interpretation"]
    assert result["task_a_contradiction"]["roc_auc"] is None
