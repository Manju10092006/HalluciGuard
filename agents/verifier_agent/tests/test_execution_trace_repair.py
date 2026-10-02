"""Controlled backends: execution contracts, not measured model accuracy."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
import hashlib
import pytest
from schemas.models import Passage, SuspiciousClaim, VerifierInputV2
from retrievers.hybrid import HybridRetriever
from retrievers.dense import DenseRetriever
from rerankers.cross_encoder import CrossEncoderReranker
from nli.robust_entailment import NLIEngine
from scorers.relation_verifier import RelationVerifier


def passage(text="Hanoi is the capital of Vietnam."):
    return Passage(title="Hanoi", snippet=text, source="wikipedia", source_id="wiki-hanoi",
        url="https://en.wikipedia.org/wiki/Hanoi", publication_date="unknown", relevance_score=.9)


@pytest.mark.parametrize("mode,route,backends", [("both", "hybrid", 2),
    ("failure", "bm25_only", 1), ("empty", "lexical_fallback", 0), ("timeout", "bm25_only", 1)])
def test_backend_counts_execution_and_fallback(mode, route, backends):
    pool = [passage()]
    def retrieve(query, k):
        if mode in {"failure", "timeout"}:
            raise (TimeoutError if mode == "timeout" else RuntimeError)("dummy-secret")
        return [(pool[0], .9)] if mode == "both" else []
    retriever = HybridRetriever()
    retriever.sparse = SimpleNamespace(build_index=lambda p: None,
        retrieve=lambda q, k: [(pool[0], .8)] if mode != "empty" else [])
    retriever.dense = SimpleNamespace(model_name="fixture", build_index=lambda p: None, retrieve=retrieve,
        diagnostics=lambda: {"model_available": mode == "both", "inference_executed": mode == "both"})
    assert retriever.retrieve("capital Vietnam", pool)
    diag = retriever.diagnostics()
    assert diag["route"] == route
    assert diag["fusion_backend_count"] == backends
    assert diag["fusion_executed"] is (backends > 0)
    assert diag["degraded"] is (mode != "both")
    assert diag["dense_result_count"] == int(mode == "both")
    assert "dummy-secret" not in str(diag)


def test_empty_input_does_not_claim_backend_execution():
    r = HybridRetriever()
    assert r.retrieve("capital", []) == []
    d = r.diagnostics()
    assert not d["dense_attempted"] and not d["fusion_executed"] and d["selected_count"] == 0


def test_empty_gate_clears_previous_reranker_success():
    r = CrossEncoderReranker()
    r.last_inference_executed = True
    r.last_status = "executed"
    assert r.score_gate_candidates("claim", []) == ([], [])
    assert r.diagnostics()["status"] == "not_run"
    assert not r.diagnostics()["inference_executed"]


@pytest.mark.parametrize("scores", [[float("nan")], [], [float("inf")]])
def test_invalid_reranker_scores_are_explicit_fallback(scores):
    r = CrossEncoderReranker()
    r.model = SimpleNamespace(predict=lambda pairs, **kw: scores)
    assert r.rerank("claim", [passage()], 1)
    assert r.diagnostics()["degraded"]
    assert not r.diagnostics()["inference_executed"]
    assert r.diagnostics()["failure_stage"] == "inference"


def test_repeated_dense_initialization_failure_is_not_a_new_attempt(monkeypatch):
    calls = []
    def fail(name):
        calls.append(name)
        raise RuntimeError("dummy-secret")
    monkeypatch.setattr("retrievers.dense.get_model_manager", lambda: SimpleNamespace(load_embedding_model=fail))
    r = DenseRetriever()
    r.build_index([passage()])
    assert r.diagnostics()["initialization_attempted"]
    r.build_index([passage()])
    assert not r.diagnostics()["initialization_attempted"]
    assert r.diagnostics()["initialization_previously_failed"]
    assert len(calls) == 1


@pytest.mark.parametrize("claim,text", [
    ("Paul Allen is the founder of Microsoft.", "Microsoft was founded by Bill Gates."),
    ("Tim Cook is the CEO of Apple in 2026.", "Steve Jobs is the CEO of Apple in 2007."),
    ("Microsoft is headquartered in Redmond.", "Microsoft was started in Albuquerque."),
])
def test_incomplete_or_different_relationship_cannot_force_contradiction(claim, text):
    assert RelationVerifier().verify_relation(claim, [passage(text)]).status == "NO_TRIPLE_EXTRACTED"


def test_nli_trace_marks_submission_not_unknown_token_survival():
    e = NLIEngine()
    e.pipeline = lambda batch: [[{"label": "entailment", "score": .9},
        {"label": "contradiction", "score": .05}, {"label": "neutral", "score": .05}]]
    e.batch_classify("claim", ["evidence"])
    d = e.diagnostics()
    assert d["submitted_inputs"][0]["premise_sha256"] == hashlib.sha256(b"evidence").hexdigest()
    assert d["submitted_inputs"][0]["tokenizer_truncated"] is None
    e.batch_classify("claim", [])
    assert e.diagnostics()["submitted_inputs"] == []
    assert not e.diagnostics()["inference_executed"]


def test_formal_country_name_does_not_override_nli_with_contradiction():
    text = "Ha Noi is the capital of the Socialist Republic of Vietnam and the national political and administrative center."
    result = RelationVerifier().verify_relation("The capital of Vietnam is Ha Noi", [passage(text)])
    assert result.status == "NO_TRIPLE_EXTRACTED"
    assert "requires NLI" in result.mismatch_detail


def test_city_word_segmentation_defers_without_inventing_an_alias():
    result = RelationVerifier().verify_relation("The capital of Vietnam is Hanoi",
        [passage("Ha Noi is the capital of Vietnam.")])
    assert result.status == "NO_TRIPLE_EXTRACTED"


def test_nli_retry_reports_its_actual_character_bounded_inputs():
    e = NLIEngine()
    def predict(inputs, **kwargs):
        if isinstance(inputs, list):
            raise RuntimeError("dummy-secret")
        assert len(inputs["text"]) == 1500
        return [{"label": "entailment", "score": .9},
            {"label": "contradiction", "score": .05}, {"label": "neutral", "score": .05}]
    e.pipeline = predict
    e.batch_classify("claim", ["x" * 1700])
    d = e.diagnostics()
    assert d["degraded"] and d["inference_executed"] and d["attempted"]
    retry = d["submitted_inputs"][-1]
    assert retry["stage"] == "individual_retry"
    assert retry["character_truncated"] and retry["premise_characters"] == 1500
    assert retry["premise_sha256"] == hashlib.sha256(b"x" * 1500).hexdigest()
    assert "dummy-secret" not in str(d)


@pytest.mark.parametrize("text,selected", [("Hanoi is the capital of Vietnam.", True),
    ("Hanoi is a large city in Asia.", False), ("Paris is the capital of France.", False)])
def test_capital_counterevidence_anchors_country_not_false_city(text, selected):
    from api.pipeline import VerificationPipeline
    p = passage(text).model_copy(update={"title": "Source"})
    scores = {"entailment_score": .000047, "contradiction_score": .999843, "neutral_score": .000109}
    passages, results = VerificationPipeline._select_decision_grade_evidence(
        [p], [scores], claim="The capital of Vietnam is Bangkok.", relation_verifier=RelationVerifier())
    assert bool(passages) is selected
    if selected:
        assert results[0]["contradiction_score"] == scores["contradiction_score"]


@pytest.mark.asyncio
async def test_pipeline_empty_selection_resets_nli_and_retains_trace(monkeypatch):
    from api.pipeline import VerificationPipeline
    from schemas.retrieval_trace import RetrievalTrace
    p = VerificationPipeline()
    p.cache_enabled = False
    p.settings = SimpleNamespace(n8n_retrieval_enabled=False)
    p.claim_decomposer.decompose = lambda claim: [claim]
    p.query_expander.resolve_and_expand = lambda c, d: (c, None)
    p.query_expander.generate_search_queries = lambda c, d: [c]
    adapter = SimpleNamespace(name="controlled-empty", search=AsyncMock(return_value=[]),
        last_retrieval_trace=RetrievalTrace())
    monkeypatch.setattr("api.pipeline.get_registry", lambda: SimpleNamespace(get_adapter=lambda d: adapter))
    p.nli_engine.last_inference_executed = True
    p.nli_engine.last_status = "executed"
    result = await p.verify(VerifierInputV2(query_id="empty", domain="general",
        suspicious_claims=[SuspiciousClaim(claim_id="c", text="A fixture claim.")]))
    trace = result.claim_evidence[0].retrieval_trace
    assert trace["nli_execution"]["status"] == "not_run"
    assert trace["evidence_flow"]["nli_selected"] == 0
    assert trace["subclaim_executions"][0]["backend_execution"]["route"] == "empty_input"


@pytest.mark.asyncio
async def test_cache_origin_is_historical_and_never_claims_fresh_inference():
    from api.pipeline import VerificationPipeline
    from schemas.models import ClaimReport
    original = ClaimReport(claim_id="old", claim_text="A fixture claim.", evidence=[],
        support_score=0, contradiction_score=0, trust_score=0, verdict="unverified",
        retrieval_trace={"nli_execution": {"inference_executed": True},
            "provider_payload": "dummy-secret"}).model_dump()
    p = VerificationPipeline()
    p.cache_enabled = True
    p.cache.get = AsyncMock(return_value=original)
    result = await p.verify(VerifierInputV2(query_id="cache", domain="general",
        suspicious_claims=[SuspiciousClaim(claim_id="new", text="A fixture claim.")]))
    trace = result.claim_evidence[0].retrieval_trace
    assert trace["execution_origin"] == "cache"
    assert not trace["nli_execution"]["inference_executed"]
    assert trace["cached_execution"]["nli_execution"]["inference_executed"]
    assert "dummy-secret" not in str(trace["cached_execution"])
    assert "dummy-secret" not in str(trace)
    assert original["retrieval_trace"]["nli_execution"]["inference_executed"]


def test_dense_skip_reason_distinguishes_index_failure():
    r = DenseRetriever()
    r._is_available = False
    r._failure_stage = "indexing"
    r._reset_run()
    assert r.diagnostics()["skipped_reason"] == "previous_indexing_failure"


@pytest.mark.parametrize("name", ["John Smith Jr", "John Smith University", "John Smith Memorial"])
def test_extra_entity_qualifier_does_not_establish_identity(name):
    assert not RelationVerifier()._names_match("John Smith", name)


def test_non_exhaustive_founder_list_cannot_exclude_a_third_founder():
    result = RelationVerifier().verify_relation("Alice Smith founded ExampleCorp.",
        [passage("ExampleCorp was founded by Bob Jones and Carol White.")])
    assert result.status == "NO_TRIPLE_EXTRACTED"


@pytest.mark.parametrize("text,rejected", [
    ("Hoa Lu was the capital of Vietnam from 968 to 1010.", True),
    ("Hoa Lu is the ancient capital of Vietnam.", True),
    ("Hanoi is not the capital of Vietnam today.", False),
    ("Hoa Lu was the capital of Vietnam. Hanoi is the current capital of Vietnam.", False),
])
def test_historical_scope_is_not_current_counterevidence(text, rejected):
    from api.pipeline import VerificationPipeline
    assert VerificationPipeline._historical_capital_only("Hanoi is the capital of Vietnam.", passage(text)) is rejected


def test_capital_segmentation_changes_only_model_premise_and_records_transform():
    from api.pipeline import VerificationPipeline
    p = passage("Ha Noi is the capital of the Socialist Republic of Vietnam.")
    text, transforms = VerificationPipeline._capital_premise(
        "Hanoi is the capital of Vietnam.", p, RelationVerifier())
    assert text == "hanoi is the capital of the Socialist Republic of Vietnam."
    assert transforms == ["capital_name_whitespace_segmentation"]
    assert p.snippet.startswith("Ha Noi")
    assert VerificationPipeline._capital_premise("Hanoi is the capital of France.", p, RelationVerifier()) == (p.snippet, [])


def test_observed_tokenization_uses_local_pipeline_copy():
    class Tensor:
        def __init__(self, values): self.values = values
        def tolist(self): return [self.values]
    class Pipeline:
        def tokenizer(self, text, text_pair=None, **kwargs):
            return {"input_ids": list(range(len(text.split()) + len((text_pair or "").split()) + 2))}
        def preprocess(self, item, **params):
            return {"input_ids": Tensor([1, 2, 3, 4]), "attention_mask": Tensor([1, 1, 1, 1])}
        def __call__(self, inputs, **kwargs):
            for item in inputs: self.preprocess(item, **kwargs)
            return [[{"label": "entailment", "score": .9}, {"label": "contradiction", "score": .05},
                     {"label": "neutral", "score": .05}] for _ in inputs]
    shared = Pipeline()
    original = shared.preprocess
    e = NLIEngine()
    e.pipeline = shared
    e.batch_classify("a claim", ["one two three four five"])
    trace = e.diagnostics()["submitted_inputs"][-1]
    assert trace["stage"] == "tokenized" and trace["tokenizer_truncated"]
    assert trace["retained_pair_tokens"] == 4
    assert e.diagnostics()["tokenizer_observability"] == "observed_preprocess"
    assert shared.preprocess == original


def test_possessive_fragment_cannot_support_the_parent_entity_claim():
    from api.pipeline import VerificationPipeline
    p = passage("Earth&#039;s Moon. Areocentric orbit (named after Ares): An orbit around the planet Mars, such as that of its moons or artificial satellites.")
    nli = {"entailment_score": .582231, "contradiction_score": .197898, "neutral_score": .219872}
    selected, _ = VerificationPipeline._select_decision_grade_evidence([p], [nli],
        claim="Earth orbits Mars.", relation_verifier=RelationVerifier())
    assert selected == []
    assert not VerificationPipeline._subject_only_in_possessive_fragment("Earth orbits the Sun.",
        passage("Earth's Moon is a satellite. Earth orbits the Sun."))


def test_new_evidence_scope_policy_does_not_reuse_old_cached_decisions():
    from cache.sqlite_cache import SqliteCache
    import hashlib
    old = hashlib.sha256(b"verifier-v2.2:general:earth orbits mars.").hexdigest()
    assert SqliteCache()._normalize_key("general", "Earth orbits Mars.") != old
