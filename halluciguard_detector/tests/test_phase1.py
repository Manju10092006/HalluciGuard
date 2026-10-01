import hashlib
import json
from pathlib import Path

import pytest

from halluciguard_detector.phase1 import GenerationTrace, Phase1Config, Phase1Service, claim_features
from halluciguard_detector.local_generation import _summarize_raw_logits
from halluciguard_detector.phase1_data import annotation_template, prepare, validate_record
from halluciguard_detector.phase1_train import _response_groups, _response_scores, _select_threshold


def _record(index: int):
    answer = f"Example {index} has a factual claim."
    return {
        "event_id": f"e{index}", "group_id": f"g{index}", "dataset_id": "human-study",
        "response": answer, "label_source": "human",
        "claims": [{"claim_id": "c1", "text": answer, "start": 0, "end": len(answer),
                    "label": "SUPPORTED" if index % 2 else "HALLUCINATION_RELATED"}],
        "trace": {"schema_version": "hg-token-logits-v2", "source": "local_transformers_generate_raw_logits",
                  "model_id": "m", "model_revision": "r", "tokenizer_id": "t", "tokenizer_revision": "r",
                  "text_sha256": hashlib.sha256(answer.encode()).hexdigest(), "complete": True,
                  "tokens": [{"start": 0, "end": len(answer), "nll": 0.3, "entropy": 1.2, "margin": 0.4}]},
    }


def test_trace_and_alignment_require_exact_response():
    record = _record(1)
    assert len(validate_record(record)[0]["features"]) == 5
    record["response"] += " extra"
    with pytest.raises(ValueError, match="exact generated response"):
        validate_record(record)


def test_raw_generation_logits_produce_finite_entropy_and_reject_warped_scores():
    import torch

    nll, entropy, margin = _summarize_raw_logits(torch.tensor([2.0, 1.0, 0.0]), 0)
    assert nll > 0 and entropy > 0 and 0 < margin < 1
    with pytest.raises(ValueError, match="finite vocabulary"):
        _summarize_raw_logits(torch.tensor([2.0, float("-inf"), float("-inf")]), 0)


def test_claim_alignment_rejects_uncovered_text():
    record = _record(1)
    trace = GenerationTrace.model_validate(record["trace"])
    with pytest.raises(ValueError, match="cuts through"):
        claim_features(trace, record["response"], 2, 8)


def test_missing_artifact_and_mismatched_generator_fail_closed(tmp_path: Path):
    record = _record(1)
    claims = [{"start": 0, "end": len(record["response"])}]
    config = Phase1Config(mode="shadow", head_dir=tmp_path, model_id="m", model_revision="r",
                          tokenizer_id="t", tokenizer_revision="r")
    service = Phase1Service(config)
    missing = service.score(record["response"], claims, record["trace"])
    assert missing.status == "unavailable" and not missing.bypass_eligible
    record["trace"]["model_id"] = "other"
    mismatched = service.score(record["response"], claims, record["trace"])
    assert mismatched.reason == "incompatible_generator" and not mismatched.bypass_eligible


def test_disabled_and_missing_trace_never_bypass():
    assert Phase1Service(Phase1Config()).score("answer", [], None).status == "disabled"
    assert not Phase1Service(Phase1Config(mode="shadow")).score("answer", [], None).bypass_eligible


def test_prepare_group_split_and_duplicate_rejection(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    records = [_record(i) for i in range(8)]
    records.append({**_record(1), "event_id": "duplicate-response"})
    source.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    output = tmp_path / "prepared"
    manifest = prepare(source, output)
    assert manifest["total_events"] == 8 and manifest["rejected_count"] == 1
    assert all(manifest["splits"][name]["groups"] for name in ("train", "validation", "calibration", "test"))


def test_annotation_template_requires_human_labels_before_prepare(tmp_path: Path):
    source = tmp_path / "events.jsonl"
    record = _record(1)
    record.pop("claims")
    record.pop("label_source")
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    destination = tmp_path / "annotation.jsonl"
    report = annotation_template(source, destination)
    assert report["events"] == 1
    task = json.loads(destination.read_text(encoding="utf-8"))
    assert task["claims"][0]["label"] is None
    with pytest.raises(ValueError, match="label_source"):
        validate_record(task)
    task["label_source"] = "human"
    task["claims"][0]["label"] = "SUPPORTED"
    assert validate_record(task)[0]["label_id"] == 0


def test_collect_resumes_only_matching_unlabelled_events(tmp_path: Path, monkeypatch):
    from halluciguard_detector.phase1_collect import collect

    record = _record(1)
    prompt_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "events.jsonl"
    prompt_path.write_text(json.dumps({"prompt_id": "p1", "query": "Question?"}) + "\n", encoding="utf-8")

    class FakeGenerator:
        def generate(self, _messages, *, max_new_tokens):
            return record["response"], record["trace"]

    monkeypatch.setattr("halluciguard_detector.phase1_collect.get_local_generator", lambda _ids: FakeGenerator())
    args = dict(model_id="m", model_revision="r", tokenizer_id="t", tokenizer_revision="r", dataset_id="study")
    assert collect(prompt_path, output_path, **args)["generated"] == 1
    event = json.loads(output_path.read_text(encoding="utf-8"))
    assert "claims" not in event and "label_source" not in event
    assert collect(prompt_path, output_path, **args)["already_present"] == 1
    prompt_path.write_text(json.dumps({"prompt_id": "p1", "query": "Changed question?"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match supplied prompts"):
        collect(prompt_path, output_path, **args)


def test_sentence_unit_requires_complete_sentence_coverage():
    record = _record(1)
    record["label_unit"] = "sentence"
    assert validate_record(record)[0]["label_unit"] == "sentence"
    record["claims"][0]["end"] -= 1
    record["claims"][0]["text"] = record["response"][:-1]
    with pytest.raises(ValueError, match="cover every sentence"):
        validate_record(record)


def test_threshold_uses_response_max_and_may_refuse_release():
    rows = [{"event_id": "a", "label_id": 0}, {"event_id": "a", "label_id": 1},
            {"event_id": "b", "label_id": 0}]
    risks, labels = _response_scores(rows, [0.1, 0.9, 0.2])
    assert risks.tolist() == [0.9, 0.2] and labels.tolist() == [1, 0]
    assert _select_threshold(risks, labels, max_false_accept=0.01, min_bypass=1) is None
    assert _response_groups([{"event_id": "a", "group_id": "g1"},
                             {"event_id": "a", "group_id": "g1"},
                             {"event_id": "b", "group_id": "g2"}]) == ["g1", "g2"]
    with pytest.raises(ValueError, match="multiple split groups"):
        _response_groups([{"event_id": "a", "group_id": "g1"},
                          {"event_id": "a", "group_id": "g2"}])


def test_checkpoint_fixture_scoring_shadow_and_fast_path(tmp_path: Path):
    """Tiny loadability fixture, not a trained model or benchmark result."""
    import torch
    from safetensors.torch import save_file

    record = _record(1)
    claims = [{"start": 0, "end": len(record["response"])}]
    head = torch.nn.Linear(5, 1)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(-10)
    path = tmp_path / "head.safetensors"
    save_file(head.state_dict(), str(path))
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {"schema": "hg-phase1-head-v1", "feature_schema": "hg-token-logits-v2",
                "label_unit": "sentence",
                "feature_names": ["mean_nll", "max_nll", "mean_entropy", "mean_margin", "log_token_count"],
                "feature_mean": [0] * 5, "feature_std": [1] * 5,
                "head_sha256": checksum, "model_id": "m", "model_revision": "r",
                "tokenizer_id": "t", "tokenizer_revision": "r"}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    calibration = {"schema": "hg-phase1-calibration-v1", "head_sha256": checksum, "temperature": 1.0,
                   "release_validated": True, "threshold": 0.01}
    calibration["calibration_id"] = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    (tmp_path / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    base = dict(head_dir=tmp_path, model_id="m", model_revision="r",
                tokenizer_id="t", tokenizer_revision="r")
    shadow = Phase1Service(Phase1Config(mode="shadow", **base)).score(record["response"], claims, record["trace"])
    assert shadow.status == "scored" and not shadow.bypass_eligible
    fast = Phase1Service(Phase1Config(mode="fast_path", **base)).score(record["response"], claims, record["trace"])
    assert fast.status == "unavailable" and not fast.bypass_eligible
    assert fast.reason == "fast_path_not_enabled"
    metadata["label_unit"] = "atomic_claim"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    atomic = Phase1Service(Phase1Config(mode="fast_path", **base)).score(record["response"], claims, record["trace"])
    assert atomic.status == "unavailable" and not atomic.bypass_eligible
    calibration["temperature"] = 0
    (tmp_path / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    invalid = Phase1Service(Phase1Config(mode="shadow", **base)).score(record["response"], claims, record["trace"])
    assert invalid.status == "unavailable" and not invalid.bypass_eligible


def test_training_lifecycle_fixture_only_not_model_quality(tmp_path: Path):
    """Exercise serialization/calibration/evaluation, never report fixture metrics."""
    from halluciguard_detector.phase1_train import evaluate, preflight, train

    source = tmp_path / "software-fixture.jsonl"
    source.write_text("\n".join(json.dumps(_record(i)) for i in range(40)), encoding="utf-8")
    data_dir, head_dir = tmp_path / "prepared", tmp_path / "head"
    prepare(source, data_dir)
    audit = preflight(data_dir, head_dir)
    assert audit["ready"]
    report = train(data_dir, head_dir, epochs=2, min_bypass=100)
    assert (head_dir / "head.safetensors").is_file()
    assert not report["release_validated"]
    test_report = evaluate(data_dir, head_dir)
    assert test_report["samples"] > 0
    assert "routing" not in test_report
