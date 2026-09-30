from types import SimpleNamespace

import torch

from halluciguard_detector.agent import DetectorAgent
from halluciguard_detector.detector import Detector
from halluciguard_detector.evidence_shapes import normalize_evidence
from halluciguard_detector.schemas import ClaimLabel, RiskLevel


def test_structured_values_preserve_paths_and_records():
    shaped = normalize_evidence([
        {"source_id": "a", "name": "Cafe North", "rating": 4.5,
         "hours": {"monday": "09:00"}, "opened": "2021-05-04", "closed": None},
        {"source_id": "b", "name": "Cafe South", "rating": 2},
    ])
    assert len(shaped.documents) == 2
    assert "hours.monday: 09:00" in shaped.documents[0]
    assert "rating: 4.5" in shaped.documents[0]
    assert "opened: 2021-05-04" in shaped.documents[0]
    assert "closed:" not in shaped.documents[0]
    assert shaped.source_ids == ["a", "b"]
    assert shaped.null_fields == 1
    assert shaped.diagnostics()["route"] == "lexical_only"
    assert shaped.diagnostics()["dense_executed"] is False


def test_json_string_wrapper_and_malformed_input():
    assert normalize_evidence('{"business":{"name":"Cafe A"}}').documents == ["business.name: Cafe A"]
    assert normalize_evidence({"source_id": "s1", "snippet": "Ordinary text."}).documents == ["Ordinary text."]
    malformed = normalize_evidence('{"business":')
    assert malformed.documents == []
    assert malformed.malformed_records == 1
    assert malformed.diagnostics()["degraded"] is True


def test_structured_title_and_publication_date_remain_factual_input():
    shaped = normalize_evidence({"source_id": "doc-7", "title": "Release notes",
                                 "publication_date": "2024-01-02", "version": 2})
    assert "title: Release notes" in shaped.documents[0]
    assert "publication_date: 2024-01-02" in shaped.documents[0]
    assert shaped.source_ids == ["doc-7"]


def test_oversize_stops_at_whole_field_and_reports_truncation():
    shaped = normalize_evidence({"a": "short", "b": "x" * 100}, max_chars=35)
    assert shaped.documents == ["a: short"]
    assert shaped.truncated_records == 1
    assert "b:" not in shaped.documents[0]
    assert normalize_evidence("A full sentence. " + "x" * 100, max_chars=25).documents == ["A full sentence."]


def test_long_punctuation_free_prose_keeps_bounded_prefix():
    shaped = normalize_evidence("one two three four five", max_chars=12)
    assert shaped.documents == ["one two"]
    assert shaped.character_truncations == 1
    assert normalize_evidence("x" * 50, max_chars=8).documents == ["x" * 8]


def test_wrapper_siblings_and_provenance_are_not_dropped():
    shaped = normalize_evidence({
        "source_id": "parent", "publication_date": "2024-01-02",
        "evidence": [
            {"source_id": "child", "snippet": "Cafe North has four stars."},
            {"snippet": "Cafe South has three stars."},
        ],
    })
    assert shaped.documents == ["Cafe North has four stars.",
                                "Cafe South has three stars.",
                                "publication_date: 2024-01-02"]
    assert shaped.document_sources == ["child", "parent", "parent"]
    assert shaped.source_ids == ["child", "parent", "parent"]
    assert normalize_evidence({"missing": None}).null_fields == 1


def test_truncation_counters_separate_fields_characters_and_documents():
    fields = normalize_evidence({"a": "short", "b": "other"}, max_fields=1)
    assert fields.field_truncations == 1
    chars = normalize_evidence({"a": "short", "b": "x" * 100}, max_chars=20)
    assert chars.character_truncations == 1
    docs = normalize_evidence(["first", "second"], max_documents=1)
    assert docs.document_truncations >= 1


class _Batch(dict):
    def to(self, _device):
        return self


def _fake_detector(logits):
    detector = object.__new__(Detector)
    seen = {}

    class Tokenizer:
        def __call__(self, contexts, claims, **kwargs):
            seen["contexts"] = contexts
            seen["claims"] = claims
            return _Batch()

    detector.tokenizer = Tokenizer()
    detector.model = lambda **kwargs: SimpleNamespace(logits=torch.tensor([logits], dtype=torch.float32))
    detector.device = torch.device("cpu")
    detector.temperature = 1.0
    detector.threshold = 0.5
    detector.max_length = 256
    detector.version = "fixture-only"
    return detector, seen


def test_normalized_model_input_matches_reported_evidence_and_guard_preserves_scores(monkeypatch):
    detector, seen = _fake_detector([5.0, -2.0, -2.0])
    agent = DetectorAgent(model_path="unused")
    monkeypatch.setattr(agent, "_get_detector", lambda: detector)
    result = agent.detect(
        "Who created Java?", "Java was created by Snehith.",
        evidence={"source_id": "s1", "language": "Java", "creator": "James Gosling"},
    )
    claim = result["claims"][0]
    assert seen["contexts"] == [" ".join(claim["evidence_snippets"])]
    assert "creator: James Gosling" in seen["contexts"][0]
    assert result["diagnostics"]["evidence_shape"]["normalization"]["source_ids"] == ["s1"]
    assert claim["probabilities"]["SUPPORTED"] > 0.99
    # An entity warning must not turn a neural support score into a fabricated
    # 0.90 contradiction probability.
    assert claim["probabilities"]["CONTRADICTED"] < 0.01


def test_structured_support_and_contradiction_are_model_outputs_not_rules(monkeypatch):
    agent = DetectorAgent(model_path="unused")
    for logits, expected in [([0.0, 4.0, 0.0], "CONTRADICTED"),
                             ([4.0, 0.0, 0.0], "SUPPORTED")]:
        detector, seen = _fake_detector(logits)
        monkeypatch.setattr(agent, "_get_detector", lambda detector=detector: detector)
        result = agent.detect("Cafe rating?", "Cafe North is rated 4.5.",
                              evidence={"name": "Cafe North", "rating": 4.5})
        assert result["claims"][0]["label"] == expected
        assert "rating: 4.5" in seen["contexts"][0]


def test_malformed_primary_evidence_context_fallback_is_reported(monkeypatch):
    detector, seen = _fake_detector([4.0, 0.0, 0.0])
    agent = DetectorAgent(model_path="unused")
    monkeypatch.setattr(agent, "_get_detector", lambda: detector)
    result = agent.detect("Who made Java?", "James Gosling made Java.",
                          evidence='{"broken":', context="James Gosling made Java.")
    assert seen["contexts"] == ["James Gosling made Java."]
    shape = result["diagnostics"]["evidence_shape"]
    assert shape["normalization"]["malformed_records"] == 1
    assert result["detector_degraded"] is True


def test_tokenizer_diagnostics_measure_pair_truncation_without_claiming_content(monkeypatch):
    detector, _seen = _fake_detector([4.0, 0.0, 0.0])

    class InspectableTokenizer:
        def __call__(self, contexts, claims, **kwargs):
            if kwargs.get("truncation") is False:
                return {"input_ids": [[1] * 15]}
            return _Batch({"input_ids": torch.ones((1, 8), dtype=torch.long)})

    detector.tokenizer = InspectableTokenizer()
    agent = DetectorAgent(model_path="unused")
    monkeypatch.setattr(agent, "_get_detector", lambda: detector)
    result = agent.detect("Rating?", "Cafe North is rated four stars.",
                          evidence={"source_id": "s1", "name": "Cafe North", "rating": 4})
    input_diag = result["diagnostics"]["model_input"]
    assert input_diag["tokenizer_observability"] == "measured"
    assert input_diag["untruncated_pair_tokens"] == [15]
    assert input_diag["model_input_pair_tokens"] == [8]
    assert input_diag["tokenizer_truncated"] == [True]
    assert result["claims"][0]["evidence_source_ids"] == ["s1"]
