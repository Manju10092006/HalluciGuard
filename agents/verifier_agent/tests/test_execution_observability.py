"""Retrieval telemetry tests; all backends are controlled fixtures."""
import sys
from pathlib import Path

# The production Verifier still uses top-level retrievers/schemas imports.
# Keep this regression file collectable from the repository root as well.
verifier_root = str(Path(__file__).resolve().parents[1])
if verifier_root not in sys.path:
    sys.path.insert(0, verifier_root)

from retrievers.hybrid import HybridRetriever
from rerankers.cross_encoder import CrossEncoderReranker
from schemas.models import Passage, ClaimReport, VerdictLabel


def passage(source):
    return Passage(title=source, source=source, source_id=source,
                   url=f"https://example.test/{source}", publication_date="2024-01-01",
                   snippet=f"{source} supports the factual claim.")


class Sparse:
    def __init__(self, rows=(), fail=False):
        self.rows, self.fail = list(rows), fail
    def build_index(self, rows):
        if self.fail:
            raise RuntimeError("dummy-secret")
    def retrieve(self, query, k):
        return self.rows[:k]


class Dense:
    model_name = "fixture"
    def __init__(self, rows=(), fail=False):
        self.rows, self.fail = list(rows), fail
    def build_index(self, rows):
        if self.fail:
            raise RuntimeError("dummy-secret")
    def retrieve(self, query, k):
        return self.rows[:k]


def test_hybrid_and_sparse_fallback_report_actual_contributors():
    a, b = passage("a"), passage("b")
    retriever = HybridRetriever()
    retriever.sparse = Sparse([(a, 1.0)])
    retriever.dense = Dense([(b, 0.9)])
    selected = retriever.retrieve("factual claim", [a, b], k=2)
    assert retriever.diagnostics()["route"] == "hybrid"
    assert retriever.diagnostics()["dense_contributed"] is True
    assert retriever.diagnostics()["selected_source_ids"] == [p.source_id for p in selected]

    retriever.dense = Dense(fail=True)
    selected = retriever.retrieve("factual claim", [a, b], k=2)
    diag = retriever.diagnostics()
    assert diag["route"] == "bm25_only" and diag["degraded"] is True
    assert diag["dense_executed"] is False
    assert diag["selected_source_ids"] == [p.source_id for p in selected]
    assert "dummy-secret" not in str(diag)


def test_dense_initialization_failure_and_empty_retrieval(monkeypatch):
    import retrievers.dense as dense_module
    class Manager:
        def load_embedding_model(self, name):
            raise RuntimeError("dummy-secret")
    monkeypatch.setattr(dense_module, "get_model_manager", lambda: Manager())
    retriever = HybridRetriever()
    a = passage("a")
    retriever.sparse = Sparse([(a, 1.0)])
    assert retriever.retrieve("factual", [a])
    diag = retriever.diagnostics()
    assert diag["route"] == "bm25_only"
    assert diag["dense_failure_stage"] == "initialization"
    assert diag["dense_available"] is False
    assert retriever.retrieve("factual", []) == []
    assert retriever.diagnostics()["route"] == "empty_input"


def test_lexical_fallback_when_both_backends_fail():
    a = passage("a")
    retriever = HybridRetriever()
    retriever.sparse = Sparse(fail=True)
    retriever.dense = Dense(fail=True)
    assert retriever.retrieve("factual claim", [a])
    assert retriever.diagnostics()["route"] == "lexical_fallback"
    assert retriever.diagnostics()["degraded"] is True


def test_dense_inference_failure_retains_bm25_results():
    class InferenceFailure(Dense):
        def retrieve(self, query, k):
            raise RuntimeError("dummy-secret")
    a = passage("a")
    retriever = HybridRetriever()
    retriever.sparse = Sparse([(a, 1.0)])
    retriever.dense = InferenceFailure()
    assert retriever.retrieve("factual", [a])
    diag = retriever.diagnostics()
    assert diag["route"] == "bm25_only"
    assert diag["dense_executed"] is False
    assert "dummy-secret" not in str(diag)


def test_reranker_failure_preserves_order_and_reports_degradation(monkeypatch):
    import rerankers.cross_encoder as reranker_module
    class Manager:
        def load_reranker_model(self, name):
            raise RuntimeError("dummy-secret")
    monkeypatch.setattr(reranker_module, "get_model_manager", lambda: Manager())
    reranker = CrossEncoderReranker()
    a = passage("a")
    selected = reranker.rerank("factual", [a], 1)
    assert selected[0].source_id == "a"
    diag = reranker.diagnostics()
    assert diag["status"] == "unavailable" and diag["degraded"] is True
    assert diag["inference_executed"] is False


    assert diag["failure_stage"] == "initialization"
    assert "dummy-secret" not in str(diag)

    class FailingModel:
        def predict(self, *args, **kwargs):
            raise RuntimeError("dummy-secret")
    reranker.model = FailingModel()
    reranker._is_available = True
    reranker.rerank("factual", [a], 1)
    diag = reranker.diagnostics()
    assert diag["status"] == "failed"
    assert diag["failure_stage"] == "inference"
    assert diag["inference_executed"] is False


def test_claim_report_serializes_backend_and_selected_evidence_trace():
    trace = {"retrieval": {"route": "bm25_only", "degraded": True,
                            "dense_executed": False, "selected_source_ids": ["s1"]},
             "reranker": {"status": "unavailable", "inference_executed": False},
             "nli_input_source_ids": ["s1"], "nli_input_count": 1,
             "decision_source_ids": [], "decision_count": 0}
    report = ClaimReport(claim_id="c1", claim_text="Claim", evidence=[],
                         support_score=0, contradiction_score=0, trust_score=0,
                         verdict=VerdictLabel.INSUFFICIENT_EVIDENCE,
                         retrieval_execution=[trace])
    assert report.model_dump()["retrieval_execution"] == [trace]
