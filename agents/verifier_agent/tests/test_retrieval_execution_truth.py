"""Execution telemetry against real BM25 and deterministic model/FAISS doubles.

No embedding or cross-encoder weights are loaded. The doubles exercise the
production build, search, fusion, reranking, and failure-handling code paths.
"""
from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

import rerankers.cross_encoder as reranker_module
import retrievers.dense as dense_module
from rerankers.cross_encoder import CrossEncoderReranker
from retrievers.dense import DenseRetriever
from retrievers.hybrid import HybridRetriever
from schemas.models import Passage
from schemas.retrieval_trace import GateRelevanceAuditTrace, ModelExecutionTrace, RetrievalTrace


class FakeEmbedding:
    def __init__(self):
        self.calls = []
        self.passage_error = None
        self.query_error = None

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        error = self.passage_error if "batch_size" in kwargs else self.query_error
        if error is not None:
            raise error
        return np.asarray([
            [float("mercury" in text.lower()), float("venus" in text.lower()),
             float("saturn" in text.lower())]
            for text in texts
        ], dtype=np.float32)


class FakeFaiss(ModuleType):
    def __init__(self):
        super().__init__("faiss")
        self.indexes = []
        self.search_error = None
        self.add_error = None

    def IndexFlatIP(self, dimension):
        owner = self

        class Index:
            def __init__(self):
                self.dimension = dimension
                self.vectors = None
                self.search_calls = []

            def add(self, vectors):
                if owner.add_error is not None:
                    raise owner.add_error
                self.vectors = np.asarray(vectors)

            def search(self, query, k):
                self.search_calls.append((np.asarray(query), k))
                if owner.search_error is not None:
                    raise owner.search_error
                scores = np.asarray(query) @ self.vectors.T
                indices = np.argsort(-scores, axis=1, kind="stable")[:, :k]
                return np.take_along_axis(scores, indices, axis=1), indices

        index = Index()
        self.indexes.append(index)
        return index


class FakeCrossEncoder:
    device = "cpu"

    def __init__(self):
        self.calls = []
        self.scores = [0.2, 0.9, -2.0]
        self.error = None

    def predict(self, pairs, **kwargs):
        self.calls.append((list(pairs), dict(kwargs)))
        if self.error is not None:
            raise self.error
        return np.asarray(self.scores)


class FakeManager:
    def __init__(self):
        self.embedding = FakeEmbedding()
        self.reranker = FakeCrossEncoder()
        self.embedding_error = None
        self.reranker_error = None
        self.embedding_loads = []
        self.reranker_loads = []

    def load_embedding_model(self, model_name):
        self.embedding_loads.append(model_name)
        if self.embedding_error is not None:
            raise self.embedding_error
        return self.embedding

    def load_reranker_model(self, model_name):
        self.reranker_loads.append(model_name)
        if self.reranker_error is not None:
            raise self.reranker_error
        return self.reranker


@pytest.fixture(autouse=True)
def fake_manager(monkeypatch):
    """Replace both manager entry points before any retriever can load weights."""
    manager = FakeManager()
    monkeypatch.setattr(dense_module, "get_model_manager", lambda: manager)
    monkeypatch.setattr(reranker_module, "get_model_manager", lambda: manager)
    return manager


@pytest.fixture
def fake_faiss(monkeypatch):
    backend = FakeFaiss()
    monkeypatch.setitem(sys.modules, "faiss", backend)
    return backend


@pytest.fixture
def passages():
    return [
        Passage(title="Mercury", source="fixture", url="https://example.test/mercury",
                publication_date="unknown", snippet="Mercury orbit is close to the Sun.",
                relevance_score=0.31),
        Passage(title="Venus", source="fixture", url="https://example.test/venus",
                publication_date="unknown", snippet="Venus clouds cover the planet.",
                relevance_score=0.42),
        Passage(title="Saturn", source="fixture", url="https://example.test/saturn",
                publication_date="unknown", snippet="Saturn rings contain ice.",
                relevance_score=0.13),
    ]


@pytest.fixture
def bm25_calls(monkeypatch):
    calls = []
    original = BM25Okapi.get_scores

    def record(self, tokens):
        calls.append(list(tokens))
        return original(self, tokens)

    monkeypatch.setattr(BM25Okapi, "get_scores", record)
    return calls


def assert_fallback_scores(output, inputs):
    assert [p.url for p in output] == [p.url for p in inputs]
    assert [p.relevance_score for p in output] == pytest.approx(
        [p.relevance_score for p in inputs]
    )


def test_hybrid_records_actual_bm25_and_dense_search(
    fake_manager, fake_faiss, passages, bm25_calls,
):
    retriever = HybridRetriever()
    output = retriever.retrieve("Mercury orbit", passages, k=2)
    proof = retriever.diagnostics()
    dense = proof["dense_execution"]
    assert bm25_calls == [["mercury", "orbit"]]
    assert len(fake_manager.embedding.calls) == 2
    assert len(fake_faiss.indexes[0].search_calls) == 1
    assert output[0].url == passages[0].url
    assert len(output) == proof["returned_count"] == 2
    assert proof["route"] == "hybrid_rrf"
    assert proof["sparse_available"] is proof["dense_available"] is True
    assert proof["sparse_executed"] is proof["dense_executed"] is True
    assert proof["sparse_contributed"] is proof["dense_contributed"] is True
    assert proof["rrf_executed"] is proof["fusion_executed"] is True
    assert proof["fusion_contributed"] is True
    assert proof["degraded"] is False
    assert dense["loaded"] is dense["index_built"] is True
    assert dense["passage_encoding_executed"] is dense["query_encoding_executed"] is True
    assert dense["search_attempted"] is dense["search_executed"] is True
    assert dense["status"] == "executed"
    assert dense["failure_stage"] is dense["fallback_reason"] is None
    # Serialization must preserve backend work, not just the configured route.
    assert RetrievalTrace(backend_execution=proof).model_dump()["backend_execution"] == proof


@pytest.mark.parametrize("error", [ImportError("embedding missing"), RuntimeError("cache missing")])
def test_dense_initialization_failure_uses_sparse_without_claiming_fusion(
    fake_manager, passages, bm25_calls, error,
):
    fake_manager.embedding_error = error
    retriever = HybridRetriever()
    output = retriever.retrieve("Mercury orbit", passages, k=2)
    proof = retriever.diagnostics()
    dense = proof["dense_execution"]
    assert len(output) == 2
    assert bm25_calls == [["mercury", "orbit"]]
    assert fake_manager.embedding.calls == []
    assert proof["route"] == "sparse_rrf_lexical"
    assert proof["sparse_executed"] is proof["sparse_contributed"] is True
    assert proof["dense_attempted"] is True
    assert proof["dense_executed"] is proof["dense_contributed"] is False
    assert proof["rrf_executed"] is True
    assert proof["fusion_executed"] is proof["fusion_contributed"] is False
    assert proof["degraded"] is True
    assert dense["load_attempted"] is True
    assert dense["loaded"] is dense["available"] is False
    assert dense["failure_stage"] == "model_load"
    assert dense["error_type"] == type(error).__name__
    assert dense["fallback_reason"] == f"model_load:{type(error).__name__}"


def test_missing_faiss_remembers_actual_reason_on_repeated_builds(
    monkeypatch, fake_manager, passages, bm25_calls, caplog,
):
    monkeypatch.setitem(sys.modules, "faiss", None)
    retriever = HybridRetriever()
    for _ in range(2):
        assert retriever.retrieve("Mercury orbit", passages, k=2)
        proof = retriever.diagnostics()
        dense = proof["dense_execution"]
        assert proof["route"] == "sparse_rrf_lexical"
        assert proof["dense_failure_stage"] == "faiss_import"
        assert dense["failure_stage"] == "faiss_import"
        assert dense["error_type"] == "ModuleNotFoundError"
        assert dense["fallback_reason"] == "faiss_import:ModuleNotFoundError"
        assert dense["faiss_available"] is False
        assert dense["loaded"] is True
        assert dense["available"] is dense["search_executed"] is False
        assert proof["fusion_executed"] is False
    assert len(fake_manager.embedding_loads) == 1
    assert fake_manager.embedding.calls == []
    assert len(bm25_calls) == 2
    assert "FAISS not available" in caplog.text
    assert "sentence_transformers not installed" not in caplog.text


@pytest.mark.parametrize("stage", ["passage_encoding", "query_encoding", "search", "index_build"])
def test_dense_runtime_failure_is_lexical_fallback_with_precise_stage(
    fake_manager, fake_faiss, passages, bm25_calls, stage,
):
    failure = RuntimeError(f"failed {stage}")
    if stage == "passage_encoding":
        fake_manager.embedding.passage_error = failure
    elif stage == "query_encoding":
        fake_manager.embedding.query_error = failure
    elif stage == "search":
        fake_faiss.search_error = failure
    else:
        fake_faiss.add_error = failure
    retriever = HybridRetriever()
    output = retriever.retrieve("Mercury orbit", passages, k=2)
    proof = retriever.diagnostics()
    dense = proof["dense_execution"]
    assert output[0].url == passages[0].url
    assert [p.relevance_score for p in output] == [1.0, 0.0]
    assert proof["route"] == "lexical_fallback"
    assert proof["degraded"] is True
    assert proof["failure_stage"] == (
        "dense_index" if stage in {"passage_encoding", "index_build"} else "dense_retrieval"
    )
    assert dense["failure_stage"] == stage
    assert dense["error_type"] == "RuntimeError"
    assert dense["fallback_reason"] == f"{stage}:RuntimeError"
    assert dense["search_executed"] is False
    assert proof["dense_executed"] is proof["dense_contributed"] is False
    assert proof["sparse_contributed"] is proof["fusion_contributed"] is False
    assert proof["fusion_executed"] is False
    assert proof["sparse_executed"] is (stage in {"query_encoding", "search"})
    assert len(bm25_calls) == int(stage in {"query_encoding", "search"})
    assert dense["passage_encoding_executed"] is (stage != "passage_encoding")
    assert dense["query_encoding_attempted"] is (stage in {"query_encoding", "search"})
    assert dense["query_encoding_executed"] is (stage == "search")
    assert dense["search_attempted"] is (stage == "search")


def test_passage_encoding_import_error_does_not_blame_faiss(
    fake_manager, fake_faiss, passages,
):
    fake_manager.embedding.passage_error = ImportError("optional encoder dependency")
    retriever = HybridRetriever()
    assert retriever.retrieve("Mercury orbit", passages, k=2)
    proof = retriever.diagnostics()
    assert proof["route"] == "sparse_rrf_lexical"
    assert proof["dense_execution"]["faiss_available"] is True
    assert proof["dense_failure_stage"] == "passage_encoding"
    assert proof["fallback_reason"] == "passage_encoding:ImportError"
    assert proof["dense_executed"] is False


@pytest.mark.parametrize("empty_case", ["passages", "k", "query"])
def test_empty_hybrid_call_clears_previous_execution_proof(
    fake_manager, fake_faiss, passages, empty_case,
):
    retriever = HybridRetriever()
    retriever.retrieve("Mercury orbit", passages, k=2)
    assert retriever.diagnostics()["dense_executed"] is True
    call_count = len(fake_manager.embedding.calls)
    output = retriever.retrieve(
        " " if empty_case == "query" else "Mercury orbit",
        [] if empty_case == "passages" else passages,
        k=0 if empty_case == "k" else 2,
    )
    proof = retriever.diagnostics()
    assert output == []
    assert len(fake_manager.embedding.calls) == call_count
    assert proof["route"] == "not_run"
    assert proof["returned_count"] == 0
    assert proof["sparse_attempted"] is proof["dense_attempted"] is False
    assert proof["sparse_executed"] is proof["dense_executed"] is False
    assert proof["rrf_executed"] is proof["fusion_executed"] is False
    assert "dense_execution" not in proof


def test_dense_rebuild_failure_clears_previous_index_and_search_proof(
    fake_manager, fake_faiss, passages,
):
    retriever = DenseRetriever()
    retriever.build_index(passages)
    assert retriever.retrieve("Mercury", 2)
    assert retriever.diagnostics()["search_executed"] is True
    fake_manager.embedding.passage_error = RuntimeError("next corpus failed")
    with pytest.raises(RuntimeError):
        retriever.build_index(passages[1:])
    proof = retriever.diagnostics()
    assert retriever.index is None
    assert proof["index_built"] is proof["search_executed"] is False
    assert proof["returned_count"] == 0
    assert proof["failure_stage"] == "passage_encoding"


@pytest.mark.parametrize("method", ["rerank", "gate"])
@pytest.mark.parametrize("error", [ImportError("model dependency missing"), RuntimeError("model load failed")])
def test_reranker_load_failure_is_available_as_a_precise_fallback(
    fake_manager, passages, method, error,
):
    fake_manager.reranker_error = error
    reranker = CrossEncoderReranker()
    if method == "rerank":
        output = reranker.rerank("claim", passages, k=2)
        assert_fallback_scores(output, passages[:2])
    else:
        scores, candidates = reranker.score_gate_candidates("claim", passages, max_candidates=2)
        assert candidates == passages[:2]
        assert scores == pytest.approx([p.relevance_score for p in passages[:2]])
    proof = reranker.diagnostics()
    assert proof["status"] == "unavailable"
    assert proof["load_attempted"] is True
    assert proof["loaded"] is proof["available"] is False
    assert proof["inference_attempted"] is proof["inference_completed"] is False
    assert proof["inference_executed"] is False
    assert proof["degraded"] is True
    assert proof["scored_count"] == 0
    assert proof["returned_count"] == 2
    assert proof["failure_stage"] == "model_load"
    assert proof["error_type"] == type(error).__name__
    assert proof["fallback_reason"] == f"model_load:{type(error).__name__}"
    assert fake_manager.reranker.calls == []
    assert ModelExecutionTrace(**proof).model_dump()["fallback_reason"] == proof["fallback_reason"]


def test_repeated_reranker_load_failure_remembers_reason_without_new_attempt(fake_manager, passages):
    fake_manager.reranker_error = RuntimeError("cache resolution failed")
    reranker = CrossEncoderReranker()
    for _ in range(3):
        assert_fallback_scores(reranker.rerank("claim", passages, k=1), passages[:1])
    proof = reranker.diagnostics()
    assert len(fake_manager.reranker_loads) == 2
    assert proof["load_attempted"] is False
    assert proof["failure_stage"] == "model_load"
    assert proof["error_type"] == "RuntimeError"
    assert proof["inference_executed"] is False


@pytest.mark.parametrize("method", ["rerank", "gate"])
def test_reranker_inference_failure_preserves_prior_scores_without_execution(
    fake_manager, passages, method,
):
    fake_manager.reranker.error = RuntimeError("prediction failed")
    reranker = CrossEncoderReranker()
    if method == "rerank":
        assert_fallback_scores(reranker.rerank("claim", passages, k=2), passages[:2])
    else:
        scores, output = reranker.score_gate_candidates("claim", passages, max_candidates=2)
        assert_fallback_scores(output, passages[:2])
        assert scores == pytest.approx([p.relevance_score for p in passages[:2]])
    proof = reranker.diagnostics()
    assert proof["loaded"] is proof["available"] is True
    assert proof["status"] == "degraded"
    assert proof["inference_attempted"] is True
    assert proof["inference_completed"] is proof["inference_executed"] is False
    assert proof["degraded"] is True
    assert proof["scored_count"] == 0
    assert proof["returned_count"] == 2
    assert proof["failure_stage"] == "inference"
    assert proof["fallback_reason"] == "inference:RuntimeError"


@pytest.mark.parametrize("method", ["rerank", "gate"])
@pytest.mark.parametrize("scores", [[], [0.8], [0.8, 0.6, 0.4, 0.2]])
def test_mismatched_reranker_scores_are_fallback_not_success(
    fake_manager, passages, method, scores,
):
    fake_manager.reranker.scores = scores
    reranker = CrossEncoderReranker()
    if method == "rerank":
        assert_fallback_scores(reranker.rerank("claim", passages, k=2), passages[:2])
    else:
        output_scores, output = reranker.score_gate_candidates("claim", passages, max_candidates=3)
        assert output == passages
        assert output_scores == pytest.approx([p.relevance_score for p in passages])
    proof = reranker.diagnostics()
    assert proof["inference_attempted"] is proof["inference_completed"] is True
    assert proof["inference_executed"] is False
    assert proof["status"] == "degraded"
    assert proof["failure_stage"] == "score_validation"
    assert proof["error_type"] == "ValueError"
    assert proof["scored_count"] == 0


def test_successful_rerank_records_all_scores_and_preserves_raw_ranking(fake_manager, passages):
    reranker = CrossEncoderReranker()
    output = reranker.rerank("claim", passages, k=2)
    proof = reranker.diagnostics()
    assert [p.url for p in output] == [passages[1].url, passages[0].url]
    assert [p.relevance_score for p in output] == pytest.approx([0.9, 0.2])
    assert [p.relevance_score for p in passages] == pytest.approx([0.31, 0.42, 0.13])
    assert proof["status"] == "executed"
    assert proof["load_attempted"] is True
    assert proof["inference_attempted"] is proof["inference_completed"] is True
    assert proof["inference_executed"] is True
    assert proof["degraded"] is False
    assert proof["device"] == "cpu"
    assert proof["scored_count"] == 3
    assert proof["returned_count"] == 2
    assert proof["failure_stage"] is proof["error_type"] is proof["fallback_reason"] is None
    assert fake_manager.reranker.calls[0][1] == {"batch_size": 16}


@pytest.mark.parametrize("method", ["rerank", "gate"])
@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), float("-inf")],
                         ids=["nan", "positive-infinity", "negative-infinity"])
def test_nonfinite_reranker_scores_fail_validation_without_neural_success(
    fake_manager, passages, method, invalid_score,
):
    fake_manager.reranker.scores = [0.2, invalid_score, 0.9]
    reranker = CrossEncoderReranker()
    if method == "rerank":
        output = reranker.rerank("claim", passages, k=2)
        assert_fallback_scores(output, passages[:2])
    else:
        scores, output = reranker.score_gate_candidates("claim", passages, max_candidates=3)
        assert output == passages
        assert scores == pytest.approx([passage.relevance_score for passage in passages])
        assert all(np.isfinite(score) for score in scores)
    proof = reranker.diagnostics()
    # predict() did return; its unusable output is not a successful neural rank.
    assert proof["inference_attempted"] is proof["inference_completed"] is True
    assert proof["inference_executed"] is False
    assert proof["degraded"] is True
    assert proof["status"] == "degraded"
    assert proof["failure_stage"] == "score_validation"
    assert proof["error_type"] == "ValueError"
    assert proof["fallback_reason"] == "score_validation:ValueError"
    assert proof["scored_count"] == 0
    assert proof["returned_count"] == (2 if method == "rerank" else 3)
    assert ModelExecutionTrace(**proof).inference_executed is False
    assert [passage.relevance_score for passage in passages] == pytest.approx([0.31, 0.42, 0.13])


def test_successful_gate_scores_are_bounded_aligned_and_clear_prior_failure(fake_manager, passages):
    reranker = CrossEncoderReranker()
    fake_manager.reranker.error = RuntimeError("temporary prediction failure")
    reranker.rerank("claim", passages, k=2)
    assert reranker.diagnostics()["degraded"] is True
    fake_manager.reranker.error = None
    fake_manager.reranker.scores = [-2.0, 2.0]
    scores, output = reranker.score_gate_candidates("claim", passages, max_candidates=2)
    proof = reranker.diagnostics()
    assert output == passages[:2]
    assert scores == pytest.approx([1.0 / (1.0 + np.exp(2.0)), 1.0])
    assert fake_manager.reranker.calls[-1][1] == {"batch_size": 2}
    assert proof["status"] == "executed"
    assert proof["inference_executed"] is True
    assert proof["load_attempted"] is False
    assert proof["degraded"] is False
    assert proof["scored_count"] == proof["returned_count"] == 2
    assert proof["failure_stage"] is proof["error_type"] is proof["fallback_reason"] is None


@pytest.mark.parametrize("method", ["rerank", "gate"])
@pytest.mark.parametrize("empty_case", ["passages", "limit"])
def test_empty_reranker_call_clears_prior_execution_without_inference(
    fake_manager, passages, method, empty_case,
):
    reranker = CrossEncoderReranker()
    reranker.rerank("claim", passages, k=2)
    assert reranker.diagnostics()["inference_executed"] is True
    call_count = len(fake_manager.reranker.calls)
    inputs = [] if empty_case == "passages" else passages
    limit = 0 if empty_case == "limit" else 2
    if method == "rerank":
        assert reranker.rerank("claim", inputs, k=limit) == []
    else:
        assert reranker.score_gate_candidates("claim", inputs, max_candidates=limit) == ([], [])
    proof = reranker.diagnostics()
    assert len(fake_manager.reranker.calls) == call_count
    assert proof["status"] == "not_run"
    assert proof["inference_attempted"] is proof["inference_completed"] is False
    assert proof["inference_executed"] is proof["degraded"] is False
    assert proof["load_attempted"] is False
    assert proof["scored_count"] == proof["returned_count"] == proof["latency_ms"] == 0
    assert proof["device"] == "unknown"
    assert proof["failure_stage"] is proof["error_type"] is proof["fallback_reason"] is None


def test_gate_audit_preserves_legacy_score_and_requires_explicit_execution_proof():
    audit = GateRelevanceAuditTrace()
    assert audit.final_bge_relevance_score == 0.0
    assert audit.final_bge_inference_executed is False
    executed = GateRelevanceAuditTrace(final_bge_relevance_score=0.8,
                                      final_bge_inference_executed=True)
    round_trip = RetrievalTrace(gate_relevance_audit=executed).model_dump()
    assert round_trip["gate_relevance_audit"]["final_bge_relevance_score"] == 0.8
    assert round_trip["gate_relevance_audit"]["final_bge_inference_executed"] is True
