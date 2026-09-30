import json

import numpy as np
import pytest

from halluciguard_detector.data import _sentence_label, iter_ragtruth_examples, prepare_ragtruth
from halluciguard_detector.training import binary_metrics, train, evaluate


def test_annotation_aliases_overlap_and_unknown_label():
    annotations = [
        {"start": 0, "end": 6, "label_type": "Evident Baseless Info"},
        {"start": 4, "end": 10, "label_type": "subtle_conflict"},
        {"start": 10, "end": 12, "label_type": "Subtle Baseless Info"},
    ]
    assert _sentence_label(0, 10, annotations) == "CONTRADICTED"
    assert _sentence_label(10, 12, annotations) == "NOT_ENOUGH_INFO"
    with pytest.raises(ValueError, match="unsupported RAGTruth label_type"):
        _sentence_label(0, 10, [{"start": 0, "end": 1, "label_type": "unknown"}])


def _dataset(path, duplicate=False):
    path.mkdir()
    sources = [{"source_id": str(i), "task_type": "Data2txt",
                "source_info": {"name": f"Cafe {i}", "rating": i + 1}}
               for i in range(5)]
    responses = [{"id": str(i), "source_id": str(i), "model": "fixture",
                  "split": "test" if i == 4 else "train", "quality": "good",
                  "response": f"Cafe {i} has rating {i + 1}.", "labels": []}
                 for i in range(5)]
    if duplicate:
        responses.append(dict(responses[0]))
    (path / "source_info.jsonl").write_text(
        "\n".join(json.dumps(row) for row in sources) + "\n", encoding="utf-8")
    (path / "response.jsonl").write_text(
        "\n".join(json.dumps(row) for row in responses) + "\n", encoding="utf-8")
    return path


def test_structured_source_preparation_group_split_and_artifact_preservation(tmp_path):
    source = _dataset(tmp_path / "source")
    rows = list(iter_ragtruth_examples(source, "train"))
    assert len(rows) == 4
    assert "rating:" in rows[0]["evidence"]
    first = prepare_ragtruth(source, tmp_path / "prepared-a", seed=7)
    second = prepare_ragtruth(source, tmp_path / "prepared-b", seed=7)
    assert first == second
    split_ids = {}
    for split in ("train", "dev", "test"):
        split_ids[split] = {json.loads(line)["source_id"] for line in
                            (tmp_path / "prepared-a" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()}
    assert not split_ids["train"] & split_ids["dev"]
    assert not split_ids["train"] & split_ids["test"]
    assert not split_ids["dev"] & split_ids["test"]
    assert first["test"]["source_groups"] == 1
    with pytest.raises(FileExistsError):
        prepare_ragtruth(source, tmp_path / "prepared-a", seed=7)


def test_duplicate_and_bad_annotation_fail_explicitly(tmp_path):
    duplicate = _dataset(tmp_path / "duplicate", duplicate=True)
    with pytest.raises(ValueError, match="duplicate response id"):
        list(iter_ragtruth_examples(duplicate, "train"))
    malformed = _dataset(tmp_path / "malformed")
    path = malformed / "response.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["labels"] = [{"start": 1, "end": 999, "label_type": "Evident Conflict"}]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="out-of-range annotation"):
        list(iter_ragtruth_examples(malformed, "train"))


def test_empty_source_files_do_not_create_prepared_artifacts(tmp_path):
    source = tmp_path / "empty"
    source.mkdir()
    (source / "source_info.jsonl").write_text("", encoding="utf-8")
    (source / "response.jsonl").write_text("", encoding="utf-8")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="nonempty train and test"):
        prepare_ragtruth(source, output)
    assert not output.exists()


def test_metrics_from_predictions_and_single_class_case():
    logits = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    metrics = binary_metrics(logits, np.array([0, 1, 2]))
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 0, "fn": 0, "tp": 2}
    assert metrics["f1"] == 1.0 and metrics["pr_auc"] == 1.0
    assert metrics["class_counts"] == {"SUPPORTED": 1, "CONTRADICTED": 1, "NOT_ENOUGH_INFO": 1}
    assert 0 <= metrics["brier"] < 0.01
    one_class = binary_metrics(logits[:1], np.array([0]))
    assert one_class["roc_auc"] is None and one_class["pr_auc"] is None


def test_training_and_evaluation_refuse_existing_artifacts(tmp_path):
    output = tmp_path / "model"
    output.mkdir()
    (output / "model.safetensors").write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        train(tmp_path / "missing-data", output)
    (output / "test_metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        evaluate(tmp_path / "missing-data", output)


def test_one_source_group_is_rejected_before_output_creation(tmp_path):
    source = _dataset(tmp_path / "source")
    path = source / "response.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["split"] == "train":
            row["source_id"] = "0"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="at least two train source groups"):
        prepare_ragtruth(source, output)
    assert not output.exists()


def test_pathological_dev_fraction_and_existing_output_are_rejected(tmp_path):
    source = _dataset(tmp_path / "source")
    with pytest.raises(ValueError, match="every train source group"):
        prepare_ragtruth(source, tmp_path / "prepared", dev_fraction=0.99)
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "precious.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_ragtruth(source, output)
    assert (output / "precious.txt").read_text(encoding="utf-8") == "preserve"


def test_malformed_jsonl_missing_labels_and_identifiers_are_explicit(tmp_path):
    source = _dataset(tmp_path / "source")
    path = source / "response.jsonl"
    original = path.read_text(encoding="utf-8")
    path.write_text("{bad json\n" + original, encoding="utf-8")
    with pytest.raises(ValueError, match="response.jsonl line 1: malformed JSON"):
        list(iter_ragtruth_examples(source, "train"))
    rows = [json.loads(line) for line in original.splitlines()]
    rows[0].pop("labels")
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or malformed labels"):
        list(iter_ragtruth_examples(source, "train"))
    rows[0]["labels"] = []
    rows[0]["id"] = None
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing id"):
        list(iter_ragtruth_examples(source, "train"))


def test_training_rejects_empty_or_malformed_data_before_model_load(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text("", encoding="utf-8")
    (data / "dev.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty train and dev"):
        train(data, tmp_path / "model")
    (data / "train.jsonl").write_text("{bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train.jsonl line 1: malformed JSON"):
        train(data, tmp_path / "model")


def test_evaluation_rejects_calibration_length_mismatch_before_model_load(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "calibration.json").write_text('{"max_length": 256}', encoding="utf-8")
    with pytest.raises(ValueError, match="must match saved calibration"):
        evaluate(tmp_path / "missing-data", model, max_length=384)


def test_metric_inputs_reject_fractional_labels_and_nonfinite_logits():
    with pytest.raises(ValueError, match="integer class IDs"):
        binary_metrics(np.array([[1.0, 0.0, 0.0]]), np.array([0.0]))
    with pytest.raises(ValueError, match="nonfinite logits"):
        binary_metrics(np.array([[float("nan"), 0.0, 0.0]]), np.array([0]))
    with pytest.raises(ValueError, match="invalid calibration temperature"):
        binary_metrics(np.array([[1.0, 0.0, 0.0]]), np.array([0]), temperature=0)
