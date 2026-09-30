"""Deterministic proof that retrieval routes name contributing backends."""

from retrievers.hybrid import HybridRetriever
from retrievers.dense import DenseRetriever
from rerankers.cross_encoder import CrossEncoderReranker
from schemas.models import Passage
from schemas.retrieval_trace import RetrievalTrace, ModelExecutionTrace
from api.pipeline import attach_retrieval_execution_trace


def _passage(name: str, snippet: str) -> Passage:
    return Passage(title=name, source=name, source_id=name,
                   url=f"https://example.test/{name}", publication_date="2024-01-01",
                   snippet=snippet)


class _Sparse:
    def __init__(self, results):
        self.results = results

    def build_index(self, passages):
        self.passages = passages

    def retrieve(self, query, k):
        return self.results[:k]


class _Dense:
    model_name = "fake"

    def __init__(self, results=(), *, fail=False):
        self.results = list(results)
        self.fail = fail

    def build_index(self, passages):
        self.passages = passages

    def retrieve(self, query, k):
        if self.fail:
            raise RuntimeError("dummy-sensitive-error")
        return self.results[:k]


def test_hybrid_and_sparse_only_routes_keep_ranking_formula():
    a = _passage("a", "Aspirin reduces pain.")
    b = _passage("b", "Aspirin treats fever.")
    retriever = HybridRetriever()
    retriever.sparse = _Sparse([(a, 3.0)])
    retriever.dense = _Dense([(b, 0.9)])
    results = retriever.retrieve("Aspirin pain", [a, b], k=2)
    diag = retriever.diagnostics()
    assert {item.source_id for item in results} == {"a", "b"}
    assert diag["route"] == "hybrid"
    assert diag["dense_contributed"] is True and diag["sparse_contributed"] is True
    assert diag["degraded"] is False

    retriever.dense = _Dense()
    sparse_results = retriever.retrieve("Aspirin pain", [a, b], k=1)
    assert sparse_results[0].source_id == "a"
    assert retriever.diagnostics()["route"] == "bm25_only"
    assert retriever.diagnostics()["degraded"] is True


def test_dense_initialization_failure_reports_sparse_only(monkeypatch):
    passage = _passage("a", "Python lists retain order.")

    class FailingManager:
        def load_embedding_model(self, name):
            raise RuntimeError("dummy-sensitive-error")

    monkeypatch.setattr("retrievers.dense.get_model_manager", lambda: FailingManager())
    retriever = HybridRetriever()
    retriever.sparse = _Sparse([(passage, 2.0)])
    result = retriever.retrieve("Python lists", [passage])
    diag = retriever.diagnostics()
    assert result and diag["route"] == "bm25_only"
    assert diag["dense_available"] is False
    assert diag["dense_executed"] is False
    assert diag["dense_failure_stage"] == "initialization"
    assert diag["dense_initialization_attempted"] is True
    assert "dummy-sensitive-error" not in str(diag)
    retriever.retrieve("Python lists", [passage])
    later = retriever.diagnostics()
    assert later["dense_failure_stage"] == "initialization"
    assert later["dense_initialization_attempted"] is False


def test_dense_inference_failure_keeps_sparse_results():
    passage = _passage("a", "Python lists retain order.")
    retriever = HybridRetriever()
    retriever.sparse = _Sparse([(passage, 2.0)])
    retriever.dense = _Dense(fail=True)
    result = retriever.retrieve("Python lists", [passage])
    diag = retriever.diagnostics()
    assert result and diag["route"] == "bm25_only"
    assert diag["dense_attempted"] is True
    assert diag["dense_executed"] is False
    assert {entry["error_type"] for entry in diag["errors"]} == {"RuntimeError"}
    assert "dummy-sensitive-error" not in str(diag)


def test_dense_retriever_identifies_inference_failure():
    passage = _passage("a", "Python lists retain order.")
    dense = DenseRetriever()

    class FailingModel:
        def encode(self, *args, **kwargs):
            raise RuntimeError("dummy-sensitive-error")

    dense.model = FailingModel()
    dense.index = object()
    dense.passages = [passage]
    try:
        dense.retrieve("Python lists", 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected controlled inference failure")
    assert dense.diagnostics()["failure_stage"] == "inference"
    assert dense.diagnostics()["inference_attempted"] is True
    assert dense.diagnostics()["inference_executed"] is False


def test_reranker_initialization_and_inference_failures_are_distinct(monkeypatch):
    passage = _passage("a", "Python lists retain order.")

    class FailingManager:
        def load_reranker_model(self, name):
            raise RuntimeError("dummy-sensitive-error")

    monkeypatch.setattr("rerankers.cross_encoder.get_model_manager", lambda: FailingManager())
    reranker = CrossEncoderReranker()
    assert reranker.rerank("Python lists", [passage], 1)
    init_diag = reranker.diagnostics()
    assert init_diag["attempted"] is True
    assert init_diag["failure_stage"] == "initialization"
    assert init_diag["inference_executed"] is False

    class FailingModel:
        def predict(self, *args, **kwargs):
            raise RuntimeError("dummy-sensitive-error")

    reranker.model = FailingModel()
    reranker._is_available = True
    assert reranker.rerank("Python lists", [passage], 1)
    inference_diag = reranker.diagnostics()
    assert inference_diag["failure_stage"] == "inference"
    assert inference_diag["inference_executed"] is False
    assert "dummy-sensitive-error" not in str(inference_diag)


def test_retrieval_trace_preserves_degraded_backend_and_reranker_status():
    trace = RetrievalTrace(hybrid_execution={"route": "bm25_only", "degraded": True},
                           evidence_degraded=True,
                           reranker_execution=ModelExecutionTrace(
                               component="bge_reranker", status="unavailable",
                               attempted=True, failure_stage="initialization", degraded=True))
    dumped = trace.model_dump()
    assert dumped["hybrid_execution"]["route"] == "bm25_only"
    assert dumped["evidence_degraded"] is True
    assert dumped["reranker_execution"]["failure_stage"] == "initialization"


def test_pipeline_attachment_propagates_degraded_execution_to_claim_trace():
    class Adapter:
        name = "fixture"
        last_retrieval_trace = None

    adapter = Adapter()
    trace = attach_retrieval_execution_trace(
        adapter, "general", hybrid={"route": "bm25_only", "degraded": True,
                                    "dense_executed": False, "selected_count": 1},
    )
    attach_retrieval_execution_trace(
        adapter, "general", reranker={"component": "bge_reranker",
                                    "status": "executed", "inference_executed": True,
                                    "degraded": False, "attempted": True},
    )
    assert adapter.last_retrieval_trace is trace
    assert trace.model_dump()["hybrid_execution"]["route"] == "bm25_only"
    assert trace.evidence_degraded is True
    assert trace.reranker_execution.inference_executed is True
