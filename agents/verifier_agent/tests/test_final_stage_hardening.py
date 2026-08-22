from __future__ import annotations

from retrievers.hybrid import HybridRetriever
from nli.robust_entailment import NLIEngine, _decision, _normalize_scores
from api.pipeline import VerificationPipeline
from scorers.evidence_scorer import EvidenceScorer
from schemas.models import EntailmentLabel, Passage, VerdictLabel


def _passage(source_id: str, snippet: str) -> Passage:
    return Passage(
        title=source_id,
        source=source_id,
        source_id=source_id,
        url=f"https://example.com/{source_id}",
        publication_date="2026-01-01",
        snippet=snippet,
    )


def test_hybrid_retrieval_keeps_semantically_and_lexically_relevant_sources():
    relevant_a = _passage("source_a", "Aspirin reduces pain and fever.")
    relevant_b = _passage("source_b", "Aspirin is widely used as a pain reliever.")
    unrelated = _passage("source_c", "Mars has two small moons.")

    class FakeSparse:
        def build_index(self, passages):
            self.passages = passages

        def retrieve(self, query, k):
            return [(relevant_a, 8.0), (unrelated, 1.0)]

    class FakeDense:
        model_name = "fake"

        def build_index(self, passages):
            self.passages = passages

        def retrieve(self, query, k):
            return [(relevant_b, 0.95), (relevant_a, 0.85)]

    retriever = HybridRetriever()
    retriever.sparse = FakeSparse()
    retriever.dense = FakeDense()

    results = retriever.retrieve("Aspirin pain", [relevant_a, relevant_b, unrelated], k=2)
    result_ids = [item.source_id for item in results]

    assert set(result_ids) == {"source_a", "source_b"}
    assert all(0.0 <= item.relevance_score <= 1.0 for item in results)


def test_hybrid_retrieval_remains_useful_when_dense_backend_is_unavailable():
    relevant = _passage("source_a", "Python lists preserve insertion order.")
    unrelated = _passage("source_b", "Mars has two small moons.")

    class FakeSparse:
        def build_index(self, passages):
            self.passages = passages

        def retrieve(self, query, k):
            return [(relevant, 4.0), (unrelated, 0.0)]

    class DisabledDense:
        model_name = "fake"

        def build_index(self, passages):
            self.passages = passages

        def retrieve(self, query, k):
            return []

    retriever = HybridRetriever()
    retriever.sparse = FakeSparse()
    retriever.dense = DisabledDense()

    results = retriever.retrieve("Python lists insertion order", [relevant, unrelated], k=1)

    assert len(results) == 1
    assert results[0].source_id == "source_a"
    assert results[0].relevance_score > 0.0


def test_nli_label_mapping_handles_label_ids():
    scores = _normalize_scores(
        [
            {"label": "LABEL_0", "score": 0.05},
            {"label": "LABEL_1", "score": 0.90},
            {"label": "LABEL_2", "score": 0.05},
        ]
    )
    result = _decision(scores)

    assert scores["entailment"] == 0.9
    assert result["label"] == EntailmentLabel.ENTAILMENT


def test_nli_batch_preserves_input_alignment():
    class FakeConfig:
        id2label = {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}

    class FakeModel:
        config = FakeConfig()

    class FakePipeline:
        model = FakeModel()

        def __call__(self, batch):
            return [
                [
                    {"label": "ENTAILMENT", "score": 0.92},
                    {"label": "CONTRADICTION", "score": 0.03},
                    {"label": "NEUTRAL", "score": 0.05},
                ],
                [
                    {"label": "CONTRADICTION", "score": 0.91},
                    {"label": "ENTAILMENT", "score": 0.04},
                    {"label": "NEUTRAL", "score": 0.05},
                ],
            ]

    engine = NLIEngine()
    engine.pipeline = FakePipeline()
    engine._is_available = True

    results = engine.batch_classify("claim", ["supporting evidence", "contradicting evidence"])

    assert len(results) == 2
    assert results[0]["label"] == EntailmentLabel.ENTAILMENT
    assert results[1]["label"] == EntailmentLabel.CONTRADICTION


def test_evidence_scorer_accepts_one_strong_authoritative_source():
    scorer = EvidenceScorer()
    passage = _passage(
        "pubmed", "The study reports that the treatment reduced symptoms."
    ).model_copy(update={"relevance_score": 0.9})
    nli = [
        {
            "label": EntailmentLabel.ENTAILMENT,
            "entailment_score": 0.95,
            "contradiction_score": 0.02,
            "neutral_score": 0.03,
        }
    ]

    result = scorer.score_evidence(
        "The treatment reduced symptoms", [passage], nli, "healthcare"
    )

    assert result["support_score"] > 0.5
    assert result["trust_score"] > 0.5
    assert result["verdict"] == VerdictLabel.VERIFIED


def test_evidence_scorer_flags_strong_contradiction():
    scorer = EvidenceScorer()
    passage = _passage(
        "pubmed", "The study found no evidence that the treatment works."
    ).model_copy(update={"relevance_score": 0.9})
    nli = [
        {
            "label": EntailmentLabel.CONTRADICTION,
            "entailment_score": 0.02,
            "contradiction_score": 0.95,
            "neutral_score": 0.03,
        }
    ]

    result = scorer.score_evidence(
        "The treatment always works", [passage], nli, "healthcare"
    )

    assert result["contradiction_score"] > 0.5
    assert result["verdict"] == VerdictLabel.CONTRADICTED


def test_evidence_scorer_does_not_treat_strong_neutral_as_support():
    scorer = EvidenceScorer()
    passage = _passage(
        "general", "The article discusses treatment options without measuring outcomes."
    ).model_copy(update={"relevance_score": 0.9})
    nli = [
        {
            "label": EntailmentLabel.NEUTRAL,
            "entailment_score": 0.08,
            "contradiction_score": 0.07,
            "neutral_score": 0.85,
        }
    ]

    result = scorer.score_evidence(
        "The treatment reduces mortality", [passage], nli, "general"
    )

    assert result["support_score"] == 0.0
    assert result["contradiction_score"] == 0.0
    assert result["verdict"] == VerdictLabel.UNVERIFIED


def test_evidence_scorer_rejects_discussion_context_as_literal_support():
    scorer = EvidenceScorer()
    passage = _passage(
        "general",
        "The claim that Exampleton is made of crystal is a fictional legend.",
    ).model_copy(update={"relevance_score": 0.9})
    nli = [{
        "label": EntailmentLabel.ENTAILMENT,
        "entailment_score": 0.98,
        "contradiction_score": 0.01,
        "neutral_score": 0.01,
    }]

    result = scorer.score_evidence(
        "Exampleton is made of crystal", [passage], nli, "general"
    )

    assert result["support_score"] == 0.0
    assert result["verdict"] == VerdictLabel.UNVERIFIED


def test_pipeline_relevance_gate_filters_before_nli():
    passages = [
        _passage("general", "Relevant support").model_copy(update={"relevance_score": 0.8}),
        _passage("general", "Unrelated apparent contradiction").model_copy(update={"relevance_score": 0.1}),
    ]

    relevant = VerificationPipeline._select_relevant_passages(passages)

    assert [passage.snippet for passage in relevant] == ["Relevant support"]


def test_pipeline_relevance_gate_keeps_only_strongest_passage_per_source():
    passages = [
        _passage("general", "Direct statement").model_copy(update={"relevance_score": 0.9}),
        _passage("general", "Tangential page").model_copy(update={"relevance_score": 0.7}),
        _passage("second_source", "Independent statement").model_copy(update={"relevance_score": 0.8}),
    ]

    relevant = VerificationPipeline._select_relevant_passages(passages)

    assert [passage.snippet for passage in relevant] == [
        "Direct statement", "Independent statement"
    ]


def test_aspirin_like_authoritative_evidence_reaches_verified_aggregation():
    scorer = EvidenceScorer()
    passage = _passage(
        "openfda",
        "The official drug label states that the medicine provides temporary relief of minor aches and pains.",
    ).model_copy(update={"relevance_score": 0.85})
    nli = [{
        "label": EntailmentLabel.ENTAILMENT,
        "entailment_score": 0.96,
        "contradiction_score": 0.01,
        "neutral_score": 0.03,
        "degraded": False,
    }]

    result = scorer.score_evidence(
        "Aspirin is used to treat mild pain.", [passage], nli, "healthcare"
    )

    assert result["support_score"] >= 0.30
    assert result["verdict"] == VerdictLabel.VERIFIED
