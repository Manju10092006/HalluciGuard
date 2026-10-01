"""Backend execution telemetry uses deterministic retriever stubs, not models."""
from __future__ import annotations

from retrievers.hybrid import HybridRetriever
from rerankers.cross_encoder import CrossEncoderReranker
from schemas.models import Passage


def _passage():
    return Passage(title="Earth", source="fixture", url="", publication_date="",
                   snippet="The Earth orbits the Sun.", source_id="fixture")


class _Sparse:
    def build_index(self, passages):
        self.passages = passages

    def retrieve(self, query, k):
        return [(self.passages[0], 1.0)]


class _Dense:
    model_name = "fixture"

    def __init__(self, stage=None):
        self.stage = stage

    def build_index(self, passages):
        self.passages = passages
        if self.stage == "initialization":
            raise RuntimeError("dummy-private-secret")

    def retrieve(self, query, k):
        if self.stage == "inference":
            raise RuntimeError("dummy-private-secret")
        return [(self.passages[0], 1.0)]

    def diagnostics(self):
        return {"model_available": self.stage != "initialization",
                "inference_executed": self.stage is None,
                "failure_stage": self.stage,
                "error_type": "RuntimeError" if self.stage else None}


def test_hybrid_and_sparse_only_routes_without_score_changes():
    retriever = HybridRetriever()
    retriever.sparse = _Sparse()
    for stage, route in ((None, "hybrid"), ("initialization", "bm25_only"),
                         ("inference", "bm25_only")):
        retriever.dense = _Dense(stage)
        selected = retriever.retrieve("Earth Sun", [_passage()])
        trace = retriever.diagnostics()
        assert len(selected) == trace["selected_count"] == 1
        assert trace["route"] == route
        assert trace["dense_contributed"] is (stage is None)
        assert trace["degraded"] is (stage is not None)
        assert "dummy-private-secret" not in str(trace)


def test_reranker_initialization_and_inference_failures_are_explicit(monkeypatch):
    reranker = CrossEncoderReranker()
    monkeypatch.setattr(reranker, "_load_model", lambda: None)
    assert reranker.rerank("Earth", [_passage()], 1)
    assert reranker.diagnostics()["status"] == "unavailable"
    assert reranker.diagnostics()["inference_executed"] is False

    class Broken:
        def predict(self, pairs, batch_size):
            raise RuntimeError("dummy-private-secret")

    reranker.model = Broken()
    assert reranker.rerank("Earth", [_passage()], 1)
    diag = reranker.diagnostics()
    assert diag["failure_stage"] == "inference"
    assert diag["error_type"] == "RuntimeError"
    assert diag["degraded"] is True
    assert "dummy-private-secret" not in str(diag)
