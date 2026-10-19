"""Constructed contract fixtures: context preservation and request isolation."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

from halluciguard_detector import evidence
from halluciguard_detector.text import lexical_documents
from halluciguard_detector.paired_evaluation import candidate_rows


def test_fallback_retains_negation_date_and_following_sentence(monkeypatch):
    monkeypatch.setattr(evidence, "_ensure_verifier", lambda: False)
    text = "Example City's capital was Alpha in 1900. It is not Alpha today. Its current capital is Beta."
    trace = {}
    assert evidence.select_evidence("Alpha is Example City's capital.", [text], trace=trace) == [text]
    assert trace["selected_evidence"] == [text]
    assert trace["context_policy"] == "whole_normalized_passages"
    assert trace["model_consumption_observed"] is False
    assert trace["tokenizer_truncation"] == "not_measured"
    assert trace["route"] == "lexical" and trace["degraded"]


def test_document_ranking_is_stable_bounded_and_does_not_erase_fields():
    docs = ['name: Alpha; founded: 1990; active: false', 'name: Beta; founded: 2000', 'unrelated']
    assert lexical_documents("Alpha founded 1990", docs, limit=1) == docs[:1]
    assert lexical_documents("absent", docs, limit=2) == docs[:2]
    assert lexical_documents("Alpha", docs, limit=0) == []


def test_fallback_evaluation_uses_the_runtime_context_policy():
    rows = [{"id": "1", "evidence": "Alpha was founded in 1990. It closed in 2000.", "claim": "Alpha was founded in 1990."}]
    result, _ = candidate_rows(rows, selection="context_preserving_fallback")
    assert result == rows
    old, _ = candidate_rows(rows, selection="lexical_fallback")
    assert old[0]["evidence"] != rows[0]["evidence"]


def test_concurrent_selection_cannot_replace_another_requests_index(monkeypatch):
    entered, release = Event(), Event()
    class Retriever:
        def retrieve(self, claim, passages, **kwargs):
            self.current = passages
            self.claim = claim
            if claim == "first":
                entered.set()
                assert release.wait(3)
            return self.current
        def diagnostics(self):
            return {"route": "hybrid", "degraded": False, "claim_marker": self.claim}
    monkeypatch.setattr(evidence, "_ensure_verifier", lambda: True)
    monkeypatch.setattr(evidence, "_to_passage", lambda t: SimpleNamespace(snippet=t))
    monkeypatch.setattr(evidence, "_hybrid_retriever", Retriever())
    monkeypatch.setattr(evidence, "_reranker", SimpleNamespace(
        rerank=lambda claim, pool, **kwargs: pool,
        diagnostics=lambda: {"inference_executed": True}))
    first_trace, second_trace = {}, {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(evidence.select_evidence, "first", ["first evidence"], trace=first_trace)
        assert entered.wait(3)
        second = pool.submit(evidence.select_evidence, "second", ["second evidence"], trace=second_trace)
        release.set()
        assert first.result(timeout=3) == ["first evidence"]
        assert second.result(timeout=3) == ["second evidence"]
    assert first_trace["retrieval"]["claim_marker"] == "first"
    assert second_trace["retrieval"]["claim_marker"] == "second"
