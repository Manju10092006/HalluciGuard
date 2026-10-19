"""Regression checks for the current-main Detector port; no model download."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

import halluciguard_detector.agent as agent_module
from halluciguard_detector.agent import DetectorAgent
from halluciguard_detector.structured_evidence import normalize_evidence


def test_structured_fields_and_source_ids_survive_normalization():
    rows, meta = normalize_evidence({"source_id": "S1", "rating": 4,
        "documents": [{"text": "Shop closes on 2025-01-02", "price": 12.5},
                      {"details": {"city": "Paris", "open": False}, "missing": None}]})
    # The parent's own fields are now their OWN record, not glued onto every child
    # passage -- that gluing manufactured false lexical signal (audit M7).
    assert len(rows) == 3
    assert any("rating: 4" in r for r in rows)
    assert any("price: 12.5" in r and "Shop closes on 2025-01-02" in r for r in rows)
    assert any("details.city: Paris" in r and "details.open: False" in r for r in rows)
    # Child passages must NOT carry the parent's "rating: 4" metadata.
    child_rows = [r for r in rows if "price: 12.5" in r or "details.city" in r]
    assert child_rows and all("rating: 4" not in r for r in child_rows)
    assert meta["source_ids"] == ["S1", "S1", "S1"]
    assert meta["null_fields"] == 1


def test_sibling_fact_survives_empty_wrapper_collection():
    # #10: a usable sibling fact must not vanish (with degraded silently False)
    # when the wrapper list is empty -- the known n8n zero-passages payload shape.
    rows, meta = normalize_evidence(
        {"query": "q", "summary": "Paris is the capital of France", "passages": []}
    )
    assert any("Paris is the capital of France" in r for r in rows)
    assert meta["normalized_count"] >= 1


def test_normalization_preserves_prose_and_reports_all_truncation_modes():
    long = "x" * 9000  # no punctuation boundary
    rows, meta = normalize_evidence([long, "{bad json", {f"f{i}": i for i in range(80)}])
    assert rows[0] == long[:8000]
    assert rows[1] == "{bad json"
    assert meta["character_truncated"] and meta["field_truncated"]
    assert meta["malformed_structured"] == 1
    rows, meta = normalize_evidence(["plain"] * 70)
    assert len(rows) == 64 and meta["document_truncated"]


def test_duplicate_text_does_not_claim_ambiguous_model_input_source():
    rows, meta = normalize_evidence([
        {"source_id": "A", "text": "same factual passage"},
        {"source_id": "B", "text": "same factual passage"},
    ])
    ids, status = DetectorAgent._model_input_provenance(rows, meta["source_ids"], rows[0])
    assert ids == []
    assert status == "ambiguous"

    ids, status = DetectorAgent._model_input_provenance(
        ["different factual passage"], ["A"], "different factual passage"
    )
    assert ids == ["A"]
    assert status == "unique_source"


def test_model_instances_are_isolated_reused_and_retry(monkeypatch, tmp_path):
    calls = []
    fail = {"once": True}

    def constructor(path, device):
        if str(path).endswith("bad") and fail["once"]:
            fail["once"] = False
            raise RuntimeError("transient")
        obj = object()
        calls.append((str(path), device, obj))
        return obj

    monkeypatch.setattr(agent_module, "Detector", constructor)
    first = DetectorAgent(tmp_path / "bad", "cpu")
    second = DetectorAgent(tmp_path / "other", "cuda")
    with pytest.raises(RuntimeError):
        first._get_detector()
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda _: first._get_detector(), range(16)))
    assert all(v is values[0] for v in values)
    assert second._get_detector() is not values[0]
    assert first._get_detector() is values[0]
    assert len(calls) == 2
