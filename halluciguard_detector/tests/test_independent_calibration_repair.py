import json
import pytest
from halluciguard_detector.training import train


def row(source, label="SUPPORTED"):
    return {"source_id": source, "claim": "Fixture claim.", "evidence": "Fixture evidence.",
            "label": label, "label_id": {"SUPPORTED": 0, "CONTRADICTED": 1}[label]}


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.mark.parametrize("calibration", [[], [row("train")], [row("dev")],
    [row("cal", "CONTRADICTED") | {"label_id": 0}], [row("")]])
def test_bad_calibration_fails_before_expensive_model_loading(tmp_path, calibration):
    write(tmp_path / "train.jsonl", [row("train"), row("train", "CONTRADICTED")])
    write(tmp_path / "dev.jsonl", [row("dev")])
    write(tmp_path / "calibration.jsonl", calibration)
    with pytest.raises(ValueError):
        train(tmp_path, tmp_path / "model", base_model="must-not-load",
              calibration_data=tmp_path / "calibration.jsonl")
    assert not (tmp_path / "model").exists()


def test_unknown_selection_metric_fails_without_artifact(tmp_path):
    with pytest.raises(ValueError, match="selection metric"):
        train(tmp_path, tmp_path / "model", selection_metric="test_set_accuracy")
