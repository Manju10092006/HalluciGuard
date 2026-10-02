"""Failure paths use controlled stubs; no models downloaded or executed."""
import builtins

import pytest

from retrievers import hybrid as module
from retrievers.sparse import BM25Retriever
from retrievers.hybrid import HybridRetriever
from schemas.retrieval_trace import ModelExecutionTrace
from agents.verifier_agent.tests.test_port_backend_trace import _Sparse, _Dense, _passage


@pytest.mark.parametrize("stage,status,route", [
    (None, "succeeded", "hybrid"), ("initialization", "failed", "bm25_only"),
    ("inference", "failed", "bm25_only")])
def test_status_contributions_and_timings(stage, status, route):
    r = HybridRetriever()
    r.sparse, r.dense = _Sparse(), _Dense(stage)
    selected = r.retrieve("Earth Sun", [_passage()], k=1)
    trace = r.diagnostics()
    assert trace["route"] == route
    assert trace["dense_status"] == status
    assert trace["sparse_status"] == "succeeded"
    assert trace["fusion_attempted"] is True
    assert trace["fusion_status"] == "succeeded"
    assert trace["fusion_backend_count"] == (2 if stage is None else 1)
    assert trace["selected_count"] == len(selected)
    backends = trace["selected_contributors"][0]["backends"]
    assert ("dense" in backends) is (stage is None)
    for key in ("total_duration_ms", "sparse_duration_ms", "dense_duration_ms", "fusion_duration_ms"):
        assert trace[key] >= 0
    assert "dummy-private-secret" not in str(trace)


class EmptySparse(_Sparse):
    def retrieve(self, query, k):
        return []


class EmptyDense(_Dense):
    def retrieve(self, query, k):
        return []


def test_empty_backends_are_not_successful_hybrid():
    r = HybridRetriever()
    r.sparse, r.dense = EmptySparse(), EmptyDense()
    assert r.retrieve("Earth Sun", [_passage()])
    trace = r.diagnostics()
    assert trace["route"] == "lexical_fallback"
    assert trace["dense_status"] == trace["sparse_status"] == "empty"
    assert trace["fusion_status"] == "skipped"
    assert trace["fallback_reason"] == "backend_empty_results"
    assert trace["degraded"] is True
    assert trace["dense_result_count"] == trace["sparse_result_count"] == 0


def test_empty_input_performs_no_backend_work():
    r = HybridRetriever()
    assert r.retrieve("Earth Sun", []) == []
    trace = r.diagnostics()
    assert trace["route"] == "empty_input"
    assert not trace["dense_attempted"] and not trace["sparse_attempted"]
    assert trace["fusion_status"] == "not_run"


def test_timeout_has_own_status_and_safe_fallback():
    class TimeoutDense(_Dense):
        def retrieve(self, query, k):
            raise TimeoutError("dummy-private-secret")
        def diagnostics(self):
            return {"model_available": True, "inference_executed": False,
                    "failure_stage": "inference", "error_type": "TimeoutError"}
    r = HybridRetriever()
    r.sparse, r.dense = _Sparse(), TimeoutDense()
    assert r.retrieve("Earth Sun", [_passage()])
    trace = r.diagnostics()
    assert trace["route"] == "bm25_only" and trace["dense_status"] == "timeout"
    assert trace["dense_executed"] is False
    assert "dummy-private-secret" not in str(trace)


def test_missing_bm25_is_unavailable_not_executed(monkeypatch):
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "rank_bm25":
            raise ImportError("dummy-private-secret")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    r = HybridRetriever()
    r.dense = _Dense()
    assert r.retrieve("Earth Sun", [_passage()])
    trace = r.diagnostics()
    assert trace["route"] == "dense_only"
    assert trace["sparse_status"] == "unavailable"
    assert trace["sparse_executed"] is False
    assert trace["sparse_details"]["failure_stage"] == "initialization"
    assert trace["degraded"] is True


def test_fusion_exception_does_not_claim_success(monkeypatch):
    def broken(*args):
        raise ValueError("dummy-private-secret")
    monkeypatch.setattr(module, "_normalize_rank_fusion", broken)
    r = HybridRetriever()
    r.sparse, r.dense = _Sparse(), _Dense()
    assert r.retrieve("Earth Sun", [_passage()])
    trace = r.diagnostics()
    assert trace["route"] == "lexical_fallback"
    assert trace["fusion_status"] == "failed"
    assert trace["fusion_executed"] is False
    assert trace["fallback_reason"] == "fusion_failure"
    assert "dummy-private-secret" not in str(trace)


def test_reranker_total_duration_survives_response_schema():
    from rerankers.cross_encoder import CrossEncoderReranker
    r = CrossEncoderReranker()
    r._is_available = False
    assert r.rerank("Earth Sun", [_passage()], 1)
    trace = ModelExecutionTrace(**r.diagnostics()).model_dump()
    assert trace["requested"] is True
    assert trace["total_duration_ms"] >= 0
    assert trace["inference_executed"] is False


def test_selection_failure_preserves_actual_successful_fusion(monkeypatch):
    r = HybridRetriever()
    r.sparse, r.dense = _Sparse(), _Dense()
    def broken(*args):
        raise RuntimeError("dummy-private-secret")
    monkeypatch.setattr(r, "_compute_overlap", broken)
    pool = [_passage(), _passage().model_copy(update={"url": "https://second.test", "snippet": "The Sun is a star."})]
    assert r.retrieve("Earth Sun", pool, k=2)
    trace = r.diagnostics()
    assert trace["fusion_executed"] is True
    assert trace["fusion_status"] == "succeeded"
    assert trace["fallback_reason"] == "selection_failure"
    assert trace["errors"][-1]["component"] == "selection"


def test_repeated_reranker_load_failure_keeps_stage_without_new_attempt(monkeypatch):
    from types import SimpleNamespace
    from rerankers.cross_encoder import CrossEncoderReranker
    def fail(*args):
        raise ImportError("dummy-private-secret")
    monkeypatch.setattr("rerankers.cross_encoder.get_model_manager", lambda: SimpleNamespace(load_reranker_model=fail))
    r = CrossEncoderReranker()
    assert r.rerank("Earth", [_passage()], 1)
    assert r.diagnostics()["initialization_attempted"] is True
    assert r.rerank("Earth", [_passage()], 1)
    trace = r.diagnostics()
    assert trace["initialization_attempted"] is False
    assert trace["failure_stage"] == "initialization"
    assert trace["error_type"] == "ImportError"
    assert trace["inference_executed"] is False


def test_sparse_runtime_failure_is_not_model_unavailability(monkeypatch):
    from types import SimpleNamespace
    def broken(*args):
        raise TimeoutError("dummy-private-secret")
    r = HybridRetriever()
    sparse = BM25Retriever()
    sparse.passages = [_passage()]
    sparse.bm25 = SimpleNamespace(get_scores=broken)
    monkeypatch.setattr(sparse, "build_index", lambda passages: None)
    r.sparse, r.dense = sparse, _Dense()
    assert r.retrieve("Earth Sun", [_passage()])
    trace = r.diagnostics()
    assert trace["route"] == "dense_only"
    assert trace["sparse_status"] == "timeout"
    assert trace["sparse_available"] is True
    assert trace["sparse_executed"] is False
    assert trace["sparse_details"]["failure_stage"] == "inference"
    assert len([e for e in trace["errors"] if e["component"] == "sparse"]) == 1


def test_model_load_fallback_log_does_not_disclose_provider_message(monkeypatch, caplog):
    from models import model_manager as manager_module
    calls = []
    def constructor(name, **kwargs):
        calls.append(kwargs["device"])
        if kwargs["device"] == "cuda":
            raise RuntimeError("dummy-private-secret")
        return object()
    monkeypatch.setattr(manager_module, "SentenceTransformer", constructor)
    monkeypatch.setattr(manager_module.ModelManager, "_detect_device", staticmethod(lambda: "cuda"))
    manager = manager_module.ModelManager()
    assert manager.load_embedding_model("fixture") is not None
    assert calls == ["cuda", "cpu"]
    assert "dummy-private-secret" not in caplog.text
