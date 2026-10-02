import json
import pytest
from scripts.retrain_detector_pilot import prepare


def fixture(path, split, sources):
    rows = [{"id": f"{split}-{s}-{label}", "source_id": s,
        "label_id": label, "label": ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"][label],
        "claim": "Diagnostic fixture.", "evidence": "Diagnostic fixture evidence."}
        for s in sources for label in range(3)]
    (path / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_pilot_keeps_genuine_labels_and_group_separation(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    fixture(raw, "train", ["a", "b"])
    fixture(raw, "dev", ["c", "d"])
    fixture(raw, "test", ["e"])
    out = tmp_path / "experiment"
    summary = prepare(raw, out, per_class=1, limit=3, seed=42)
    assert summary["splits"]["train"]["labels"] == {
        "SUPPORTED": 1, "CONTRADICTED": 1, "NOT_ENOUGH_INFO": 1}
    val = {json.loads(r)["source_id"] for r in (out / "data/dev.jsonl").read_text().splitlines()}
    cal = {json.loads(r)["source_id"] for r in (out / "data/calibration.jsonl").read_text().splitlines()}
    assert not val & cal
    with pytest.raises(FileExistsError):
        prepare(raw, out, 1, 3, 42)
