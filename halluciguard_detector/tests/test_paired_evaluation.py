"""Small explicit fixtures check calculation/protection, not model accuracy."""
import numpy as np
import pytest

from halluciguard_detector.paired_evaluation import (
    candidate_rows, false_accept_diagnostic, run, validate_splits, wilson)


def row(name, source):
    return dict(id=name, source_id=source, evidence=f"City {source} is in France.",
                claim=f"City {source} is in France.", label="SUPPORTED", label_id=0)


def test_group_and_exact_pair_integrity():
    splits = {name: [row(name, name)] for name in ("train", "dev", "test")}
    assert validate_splits(splits)["source_disjoint"] is True
    splits["test"][0]["source_id"] = "train"
    with pytest.raises(ValueError, match="source overlap"):
        validate_splits(splits)


@pytest.mark.parametrize("change", [{"label_id": True}, {"label": "unknown"},
                                   {"id": ""}, {"source_id": ""}, {"evidence": ""}])
def test_invalid_gold_is_not_relabelled(change):
    record = row("1", "s")
    record.update(change)
    with pytest.raises(ValueError):
        validate_splits({"test": [record]})


def test_normalization_is_not_a_new_label_or_model():
    r = row("1", "s")
    converted, trace = candidate_rows([r])
    assert converted == [r]
    assert trace["changed_inputs"] == 0


def test_production_identity_and_actual_multi_record_selection():
    record = row("1", "s")
    record["evidence"] = "City s is in France. It has a cathedral."
    calls = []
    def selector(claim, candidates, trace):
        calls.append(candidates)
        trace["degraded"] = False
        return [candidates[-1]]
    normalized, stats = candidate_rows([record], selection="production", selector=selector)
    assert normalized == [record]
    assert calls == []
    record["evidence"] = '[{"city":"Paris"},{"city":"Lyon"}]'
    normalized, stats = candidate_rows([record], selection="production", selector=selector)
    assert normalized[0]["evidence"] == "city: Lyon"
    assert len(calls) == 1
    assert stats["actual_multi_candidate_selector_calls"] == 1


def test_exact_overlap_is_explicit_and_requires_opt_in():
    train, test = row("train", "one"), row("test", "two")
    test["evidence"], test["claim"] = train["evidence"], train["claim"]
    with pytest.raises(ValueError, match="exact"):
        validate_splits({"train": [train], "test": [test]})
    audit = validate_splits({"train": [train], "test": [test]}, allow_exact_overlap=True)
    assert audit["cross_split_exact_pairs"]["train/test"] == 1


def test_false_accept_count_and_interval():
    logits = np.asarray([[8., 0., 0.], [8., 0., 0.], [0., 8., 0.]])
    result = false_accept_diagnostic(logits, [0, 1, 1], 1., .625)
    assert result["eligible"] == 2
    assert result["contradicted_eligible"] == 1
    assert result["conditional_false_accept_rate"] == .5
    assert result["wilson_95_interval"][0] < .5 < result["wilson_95_interval"][1]
    assert wilson(0, 0) is None


def test_existing_report_is_protected_before_loading(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path / "data", tmp_path / "model", tmp_path)


def test_checkpoint_and_dataset_cannot_be_output(tmp_path):
    with pytest.raises(ValueError, match="inside"):
        run(tmp_path / "data", tmp_path / "model", tmp_path / "model" / "report")
