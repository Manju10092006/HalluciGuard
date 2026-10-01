import json

import numpy as np
import pytest

from agents.detector_agent import halueval_dataset as dataset_module
from agents.detector_agent import halueval_trainer as trainer_module


def rows(count):
    return [{"user_query": f"Question {i}", "chatgpt_response": f"Answer {i}",
             "hallucination": "yes" if i % 2 else "no"} for i in range(count)]


def test_unknown_or_ambiguous_general_label_is_not_supported():
    for bad in ("unknown", "", 2, 0.5, None):
        example = rows(1)
        example[0]["hallucination"] = bad
        with pytest.raises(ValueError, match="hallucination label"):
            dataset_module._normalize_general(example)


def test_group_split_requires_three_sources_and_is_deterministic(monkeypatch):
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *args, **kwargs: rows(2))
    cfg = dataset_module.HaluEvalConfig(configs_to_load=["general"], general_upsample_factor=1)
    with pytest.raises(ValueError, match="source group"):
        dataset_module.load_halueval(cfg)
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *args, **kwargs: rows(10))
    first = dataset_module.load_halueval(cfg)
    second = dataset_module.load_halueval(cfg)
    assert {name: list(first[name]["text"]) for name in first} == {
        name: list(second[name]["text"]) for name in second}


def test_invalid_or_single_class_metric_fixture():
    with pytest.raises(ValueError, match="invalid HaluEval"):
        trainer_module.compute_metrics((np.array([[float("nan"), 0.0]]), np.array([0])))
    with pytest.raises(ValueError, match="invalid HaluEval"):
        trainer_module.compute_metrics((np.array([[0.0, 1.0]]), np.array([0.5])))
    measured = trainer_module.compute_metrics((np.array([[2.0, 0.0]]), np.array([0])))
    assert measured["class_counts"] == {"NO_HALLUCINATION": 1, "HALLUCINATION": 0}
    assert measured["confusion_matrix"] == [[1, 0], [0, 0]]


def test_existing_model_artifacts_block_training_before_dataset_load(tmp_path, monkeypatch):
    output = tmp_path / "model"
    output.mkdir()
    (output / "model.safetensors").write_bytes(b"preserve")
    monkeypatch.setattr(trainer_module.sys, "argv", ["trainer", "--output-dir", str(output)])
    monkeypatch.setattr(trainer_module, "load_halueval",
                        lambda *_: (_ for _ in ()).throw(AssertionError("dataset must not load")))
    with pytest.raises(FileExistsError, match="must be preserved"):
        trainer_module.main()
    assert (output / "model.safetensors").read_bytes() == b"preserve"
