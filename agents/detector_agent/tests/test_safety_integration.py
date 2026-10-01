"""Offline, deterministic integration checks for the production HaluEval path."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from agents.detector_agent.config import DetectorConfig
from agents.detector_agent.detector import DetectorAgent
from agents.detector_agent.evidence import prepare_evidence
from agents.detector_agent.halueval_inference import HaluEvalInference, InferenceResult
from agents.detector_agent.models import NextAction


def test_nested_mixed_evidence_provenance_and_context_alignment():
    prepared = prepare_evidence({
        "source_id": "parent", "passages": [
            {"source_id": "child", "business": {"rating": 4.5, "opened": "2020-01-01"}},
            "Ordinary text evidence.",
            {"name": None, "address": {"city": "Paris"}},
        ], "publication_date": "2024-02-03",
    })
    assert "business.rating: 4.5" in prepared.context
    assert "business.opened: 2020-01-01" in prepared.context
    assert "Ordinary text evidence." in prepared.context
    assert "publication_date: 2024-02-03" in prepared.context
    assert "name:" not in prepared.context
    assert prepared.null_fields == 1
    assert {item["source_id"] for item in prepared.selected} == {"parent", "child"}
    assert prepared.context == "\n".join(item["text"] for item in prepared.selected)


def test_malformed_and_oversized_evidence_is_bounded_and_reported():
    prepared = prepare_evidence(["{bad json", "X" * 4000, {"missing": None}], max_context_chars=120)
    assert len(prepared.context) <= 120
    assert prepared.malformed_records == 1
    assert prepared.truncated_characters > 0
    assert prepared.null_fields == 1
    assert prepared.diagnostics()["degraded"] is True


def test_structured_truncation_does_not_emit_partial_field_value():
    prepared = prepare_evidence({"a": "short", "rating": "4.5 stars"}, max_context_chars=15)
    assert "a: short" in prepared.context
    assert "rating: 4" not in prepared.context
    assert prepared.truncated_fields > 0


def test_different_configurations_never_share_inference_instance():
    one = DetectorAgent(DetectorConfig(halueval_model_path="one", halueval_max_length=128), device="cpu")
    two = DetectorAgent(DetectorConfig(halueval_model_path="two", halueval_max_length=256), device="cuda")
    assert one._inference is not two._inference
    assert (one._inference.model_path, one._inference.max_length, one._inference.device) == ("one", 128, "cpu")
    assert (two._inference.model_path, two._inference.max_length, two._inference.device) == ("two", 256, "cuda")


def test_failed_load_can_retry_without_partial_healthy_state(monkeypatch):
    import agents.detector_agent.halueval_inference as module
    monkeypatch.setattr(module, "validate_halueval_model_reference", lambda value: value)
    calls = {"tokenizer": 0}

    def tokenizer(_):
        calls["tokenizer"] += 1
        if calls["tokenizer"] == 1:
            raise RuntimeError("dummy-secret")
        return object()

    class Model:
        def to(self, device):
            return self
        def eval(self):
            return self

    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(module.AutoModelForSequenceClassification, "from_pretrained", lambda _: Model())
    inference = HaluEvalInference("fixture", device="cpu")
    with pytest.raises(RuntimeError):
        inference.load()
    assert inference.is_loaded() is False and inference._tokenizer is None and inference._model is None
    inference.load()
    assert inference.is_loaded() is True and calls["tokenizer"] == 2


def test_concurrent_load_only_initializes_one_model(monkeypatch):
    import agents.detector_agent.halueval_inference as module
    monkeypatch.setattr(module, "validate_halueval_model_reference", lambda value: value)
    calls = {"tokenizer": 0, "model": 0}

    class Model:
        def to(self, device):
            return self
        def eval(self):
            return self

    def tokenizer(_):
        calls["tokenizer"] += 1
        return object()
    def model(_):
        calls["model"] += 1
        return Model()

    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(module.AutoModelForSequenceClassification, "from_pretrained", model)
    inference = HaluEvalInference("fixture", device="cpu")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: inference.load(), range(16)))
    assert calls == {"tokenizer": 1, "model": 1}


def test_detector_model_unavailable_is_uncertain_and_requires_verifier(monkeypatch):
    agent = DetectorAgent(DetectorConfig(halueval_model_path="missing"))
    monkeypatch.setattr(agent._inference, "load", lambda: (_ for _ in ()).throw(RuntimeError("dummy-secret")))
    result = agent.detect("Question", "Factual answer.")
    assert result.hallucination_probability is None
    assert result.probability_available is False
    assert result.next_action == NextAction.VERIFY
    assert result.detector_degraded is True
    assert "dummy-secret" not in str(result.model_dump())


def test_grounded_input_and_model_diagnostics_are_distinct(monkeypatch):
    agent = DetectorAgent(DetectorConfig(halueval_model_path="fixture"))
    monkeypatch.setattr(agent._inference, "load", lambda: setattr(agent._inference, "_loaded", True))
    observed = {}
    def predict(query, answer, context=None):
        observed["context"] = context
        return InferenceResult(0.1, 0.9, 0, "NO_HALLUCINATION",
                               {"tokenizer_observability": "measured", "model_input_tokens": 20,
                                "tokenizer_truncated": False})
    monkeypatch.setattr(agent._inference, "predict", predict)
    result = agent.detect("Question", "Answer", {"source_id": "s1", "rating": 4.5})
    assert observed["context"] == result.diagnostics["evidence"]["selected_context"]
    assert result.diagnostics["model_input"]["model_input_tokens"] == 20
    assert result.next_action == NextAction.VERIFY
    assert result.grounded is True and result.calibrated is False
