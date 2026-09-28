"""Hermetic tests for the Detector Agent's claim-level hybrid evidence
verification (claims.py, evidence.py, nli.py) and the extended
DetectorAgent.detect() evidence path.

Everything here is deterministic and offline: retrieval backends are replaced
with fakes and the NLI model is a stub. No model weights are downloaded in
these tests.
"""

from __future__ import annotations

import pytest

from agents.detector_agent.claims import (
    ClaimType,
    classify_claim_type,
    extract_anchors,
    has_specific_anchor,
)
from agents.detector_agent.evidence import EvidenceItem, EvidenceResult, EvidenceRetriever
from agents.detector_agent.nli import (
    CONTRADICTED,
    NOT_ENOUGH_INFO,
    SUPPORTED,
    ClaimEvidenceClassifier,
    ClaimEvidenceResult,
)
from agents.verifier_agent.schemas.models import Passage


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeSparse:
    def __init__(self, hits=None):
        self.hits = hits or []
        self.passages = []

    def build_index(self, passages):
        self.passages = passages

    def retrieve(self, query, k):
        return [(p, float(score)) for (p, score) in self.hits[:k]]


class FakeDense:
    def __init__(self, hits=None):
        self.hits = hits or []
        self.passages = []

    def build_index(self, passages):
        self.passages = passages

    def retrieve(self, query, k):
        return [(p, float(score)) for (p, score) in self.hits[:k]]


class FakeBooster:  # sparse+dense that always raises -> lexical fallback
    def __init__(self, exc=RuntimeError("backend down")):
        self.exc = exc

    def __getattr__(self, name):
        def _boom(*a, **k):
            raise self.exc

        return _boom

    def build_index(self, passages):
        pass

    def retrieve(self, query, k):
        raise self.exc


class FakeReranker:
    def __init__(self, executed=True):
        self.last_status = "executed"
        self.last_inference_executed = executed

    def rerank(self, claim, pool, k):
        if not self.last_inference_executed:
            self.last_status = "degraded"
            return pool[:k]
        # emulate ascending scores by original order
        return list(reversed(pool[:k]))


class FakeDebertaNLI:
    """Stands in for NLIEngine.classify(...) -> dict payload."""

    def classify(self, claim, evidence):
        return {
            "label": "entailment",
            "entailment_score": 0.92,
            "contradiction_score": 0.03,
            "neutral_score": 0.05,
            "degraded": False,
        }


class BrokenNLI:
    def classify(self, claim, evidence):
        raise RuntimeError("model unavailable")


def passage(text, score=0.0) -> Passage:
    return Passage(
        title="",
        source="local",
        url="",
        publication_date="",
        snippet=text,
        source_id=f"doc-{abs(hash(text))}",
        relevance_score=score,
    )


class PrefabRetriever:
    """Returns a scripted EvidenceResult per claim (or empty = no evidence)."""

    def __init__(self, mapping, method="bm25+dense"):
        self.mapping = mapping
        self.method = method

    def retrieve_evidence(self, claim, documents):
        snippet = self.mapping.get(claim)
        if not snippet:
            return EvidenceResult()
        return EvidenceResult(
            items=[EvidenceItem(text=snippet, retrieval_method=self.method)],
            retrieved=True,
            methods_used=self.method.split("+"),
        )


class ScriptedNLI:
    def __init__(self, mapping):
        self.mapping = mapping

    def classify(self, claim, evidence):
        return self.mapping[claim]


def supported(score=0.9):
    return ClaimEvidenceResult(
        label=SUPPORTED, supported_probability=score, contradicted_probability=0.02, unknown_probability=0.08, degraded=False, model_source="deberta-nli"
    )


def contradicted(score=0.88):
    return ClaimEvidenceResult(
        label=CONTRADICTED, supported_probability=0.02, contradicted_probability=score, unknown_probability=0.10, degraded=False, model_source="deberta-nli"
    )


def insufficient():
    return ClaimEvidenceResult(
        label=NOT_ENOUGH_INFO, supported_probability=0.1, contradicted_probability=0.0, unknown_probability=0.9, degraded=False, model_source="deberta-nli"
    )


# ---------------------------------------------------------------------------
# Claim typing
# ---------------------------------------------------------------------------
class TestClaimTyping:
    def test_factual_default(self):
        assert classify_claim_type("The capital of France is Paris.") in (
            ClaimType.FACTUAL,
            ClaimType.ENTITY,
        )

    def test_numeric(self):
        assert classify_claim_type("Paris has a population of 2 million.") == ClaimType.NUMERICAL

    def test_percent(self):
        assert classify_claim_type("Inflation rose by 10 percent last year.") == ClaimType.NUMERICAL

    def test_temporal_year_beats_number(self):
        assert classify_claim_type("Google was founded in 1998.") == ClaimType.TEMPORAL

    def test_comparative(self):
        assert classify_claim_type("The Nile is longer than the Amazon.") == ClaimType.COMPARATIVE

    def test_opinion_explicit(self):
        assert classify_claim_type("Personally, I think Python is better.") == ClaimType.OPINION

    def test_opinion_superlative(self):
        assert classify_claim_type("This is the best programming language.") == ClaimType.OPINION

    def test_relational(self):
        assert classify_claim_type("Sun Microsystems was acquired by Oracle.") == ClaimType.RELATIONAL

    def test_anchors(self):
        assert has_specific_anchor("Paris has a population of 2 million.")
        assert has_specific_anchor("Google was founded in 1998.")
        assert has_specific_anchor("The Nile is longer than the Amazon.")
        assert not has_specific_anchor("The capital of France is Paris.")

    def test_extract_anchors(self):
        anchors = extract_anchors("Founded in 1998 with 2 million people")
        assert "1998" in anchors
        assert "2 million" in anchors


# ---------------------------------------------------------------------------
# Evidence retriever (offline via fakes)
# ---------------------------------------------------------------------------
class TestEvidenceRetriever:
    def test_no_documents_returns_empty(self):
        retriever = EvidenceRetriever(sparse=FakeSparse(), dense=FakeDense(), reranker=FakeReranker())
        assert not retriever.retrieve_evidence("Paris is the capital.", None).retrieved
        assert not retriever.retrieve_evidence("Paris is the capital.", []).retrieved

    def test_lexical_fallback_when_backends_raise(self):
        docs = ["Paris is the capital city of France.", "The Eiffel Tower is in Paris."]
        retriever = EvidenceRetriever(
            sparse=FakeBooster(), dense=FakeBooster(), reranker=FakeReranker()
        )
        result = retriever.retrieve_evidence("Paris is the capital of France.", docs)
        assert result.retrieved
        assert result.degraded
        assert "bm25" in result.methods_used
        assert any("capital" in item.text.lower() for item in result.items)

    def test_hybrid_merges_and_dedupes(self):
        p1 = passage("The capital of France is Paris.", 0.9)
        p2 = passage("Paris is in northern France.", 0.7)
        retriever = EvidenceRetriever(
            sparse=FakeSparse(hits=[(p1, 0.9), (p2, 0.7)]),
            dense=FakeDense(hits=[(p1, 0.85)]),
            reranker=FakeReranker(executed=True),
        )
        result = retriever.retrieve_evidence("What is the capital?", ["The capital of France is Paris.", "Paris is in northern France."])
        items = result.items
        assert {item.text for item in items} == {"The capital of France is Paris.", "Paris is in northern France."}
        methods = {item.retrieval_method for item in items}
        # p1 surfaced in BOTH backends (deduped into one item); p2 only in BM25.
        assert methods == {"bm25+dense+reranked", "bm25+reranked"}

    def test_rerank_order_applied(self):
        p1 = passage("unrelated filler text about weather.", 0.5)
        p2 = passage("The capital of France is Paris.", 0.6)
        retriever = EvidenceRetriever(
            sparse=FakeSparse(hits=[(p1, 0.6), (p2, 0.5)]),
            dense=FakeDense(hits=[]),
            reranker=FakeReranker(executed=True),
        )
        result = retriever.retrieve_evidence("What is the capital?", ["unrelated filler text about weather.", "The capital of France is Paris."])
        assert result.items[0].text == "The capital of France is Paris."


# ---------------------------------------------------------------------------
# Claim-evidence NLI classifier (deterministic / stubbed DeBERTa)
# ---------------------------------------------------------------------------
class TestClaimEvidenceClassifier:
    def _classifier(self, nli=None):
        return ClaimEvidenceClassifier(nli_engine=nli)

    def test_deberta_entailment_when_available(self):
        c = self._classifier(FakeDebertaNLI())
        r = c.classify("The capital of France is Paris.", "Paris is the capital of France.")
        assert r.label == SUPPORTED
        assert r.model_source == "deberta-nli"
        assert not r.degraded
        assert r.supported_probability == 0.92

    def test_sufficient_generic_evidence_supported(self):
        # No specific anchor -> generic related evidence may support.
        c = self._classifier(BrokenNLI())
        r = c.classify("The capital of France is Paris.", "Paris is the capital of France.")
        assert r.label == SUPPORTED

    def test_anchor_claim_not_supported_by_generic_evidence(self):
        # Claim has a numeric anchor; evidence is related but addresses NO number.
        c = self._classifier(BrokenNLI())
        r = c.classify("Paris has a population of 2 million.", "Paris is a beautiful city with rivers.")
        assert r.label == NOT_ENOUGH_INFO
        assert r.unknown_probability == 1.0

    def test_anchor_claim_sufficient_when_evidence_has_number(self):
        c = self._classifier(BrokenNLI())
        r = c.classify("Paris has a population of 2 million.", "The population of Paris is 2 million.")
        assert r.label in (SUPPORTED, NOT_ENOUGH_INFO)

    def test_numeric_conflict_is_contradiction(self):
        c = self._classifier(BrokenNLI())
        r = c.classify("Paris has a population of 2 million.", "The population of Paris is 12 million.")
        assert r.label == CONTRADICTED
        assert r.contradicted_probability == 1.0

    def test_not_enough_info_is_not_contradiction(self):
        c = self._classifier(BrokenNLI())
        r = c.classify("The capital of France is Paris.", "France is a country in Europe.")
        assert r.label == NOT_ENOUGH_INFO
        assert r.contradicted_probability == 0.0
        assert r.unknown_probability == 1.0

    def test_negated_evidence_is_contradiction(self):
        c = self._classifier(BrokenNLI())
        r = c.classify("Google was founded in 1998.", "Google was NOT founded in 1998; it was founded in 1995.")
        assert r.label == CONTRADICTED


# ---------------------------------------------------------------------------
# DetectorAgent evidence path (via scripted fakes)
# ---------------------------------------------------------------------------
BASE_DOCS = ["Paris is the capital city of France."]


def make_agent(retriever, nli):
    from agents.detector_agent.detector import DetectorAgent

    return DetectorAgent(evidence_retriever=retriever, nli_classifier=nli)


class TestDetectorEvidencePath:
    def test_1_supported_with_relevant_evidence(self):
        resp = "Paris is the capital of France."
        agent = make_agent(
            PrefabRetriever({"Paris is the capital of France": "Paris is the capital city of France."}),
            ScriptedNLI({"Paris is the capital of France": supported()}),
        )
        result = agent.detect("Where is the capital?", resp, documents=BASE_DOCS)
        assert result.next_action.value == "Accept"
        assert result.risk_level.value == "LOW"
        assert result.supported_count == 1
        claim = result.per_claim_results[0]
        assert claim.verification_status == "VERIFIED"
        assert claim.claim_type in (ClaimType.FACTUAL.value, ClaimType.ENTITY.value)
        assert not claim.requires_verification
        assert claim.supported_probability == 0.9

    def test_2_no_relevant_evidence_unverified(self):
        resp = "The Moon is made of green cheese."
        agent = make_agent(PrefabRetriever({}), ScriptedNLI({}))
        result = agent.detect("Tell me about cheese.", resp, documents=["Earth is a planet."])
        assert result.next_action.value == "Verify"
        assert result.unverified_count == 1
        assert result.requires_verification
        claim = result.per_claim_results[0]
        assert claim.verification_status == "UNVERIFIED"
        assert claim.requires_verification
        assert claim.hallucination_probability == 0.5

    def test_3_contradicting_evidence(self):
        resp = "Paris has a population of 2 million."
        agent = make_agent(
            PrefabRetriever({"Paris has a population of 2 million": "The population of Paris is 12 million."}),
            ScriptedNLI({"Paris has a population of 2 million": contradicted()}),
        )
        result = agent.detect("Tell me about Paris", resp, documents=BASE_DOCS)
        assert result.next_action.value == "Verify"
        assert result.contradicted_count == 1
        claim = result.per_claim_results[0]
        assert claim.verification_status == "CONTRADICTED"
        assert claim.contradicted_probability == 0.88

    def test_4_opinion_not_hallucination(self):
        resp = "This is the best programming language."
        agent = make_agent(PrefabRetriever({}), ScriptedNLI({}))
        result = agent.detect("Which language should I learn?", resp, documents=BASE_DOCS)
        assert result.opinion_count == 1
        assert result.contradicted_count == 0
        assert result.next_action.value == "Accept"
        assert result.per_claim_results[0].claim_type == ClaimType.OPINION.value

    def test_5_insufficient_evidence_distinct(self):
        resp = "The capital of France is Paris."
        agent = make_agent(
            PrefabRetriever({"The capital of France is Paris": "France is a country in Europe."}),
            ScriptedNLI({"The capital of France is Paris": insufficient()}),
        )
        result = agent.detect("Tell me about France", resp, documents=BASE_DOCS)
        assert result.unknown_count == 1
        claim = result.per_claim_results[0]
        assert claim.verification_status == "INSUFFICIENT"
        assert claim.unknown_probability == 0.9
        assert claim.contradicted_probability == 0.0

    def test_6_answer_level_aggregation(self):
        retriever = PrefabRetriever(
            {
                "Paris is the capital of France": "Paris is the capital of France.",
                "The Nile is longer than the Amazon": "The Amazon is the longest river.",
            }
        )
        nli = ScriptedNLI(
            {
                "Paris is the capital of France": supported(),
                "The Nile is longer than the Amazon": contradicted(),
            }
        )
        agent = make_agent(retriever, nli)
        result = agent.detect(
            "Geography",
            "Paris is the capital of France. The Nile is longer than the Amazon.",
            documents=BASE_DOCS,
        )
        assert result.claim_count == 2
        assert result.supported_count == 1
        assert result.contradicted_count == 1
        assert result.requires_verification

    def test_7_legacy_path_backward_compatible(self):
        # No documents -> the legacy triage path runs and the documented fields
        # (per-claim results, status, model_source, etc.) are preserved.
        agent = make_agent(PrefabRetriever({}), ScriptedNLI({}))
        result = agent.detect("Capital?", "Paris is the capital of France.")
        assert result.hallucination_probability is not None
        assert result.next_action in ("Accept", "Verify")
        assert result.claim_count == len(result.per_claim_results)
        assert result.evidence_available is False
        assert result.verification_risk is not None
        assert result.atomic_claims == ["Paris is the capital of France"]

    def test_7b_default_path_skips_opinions(self):
        agent = make_agent(PrefabRetriever({}), ScriptedNLI({}))
        result = agent.detect(
            "Opinion?", "Personally, I think this is the best pizza in town."
        )
        claim = result.per_claim_results[0]
        assert claim.claim_type == ClaimType.OPINION.value
        assert claim.hallucination_probability == 0.0
        assert not claim.requires_verification

    def test_8_fail_closed_when_nli_degraded(self):
        # Deterministic-fallback "support" must NOT authorize a fast path.
        degraded_support = ClaimEvidenceResult(
            label=SUPPORTED, supported_probability=0.71, contradicted_probability=0.1, unknown_probability=0.19, degraded=True, model_source="deterministic-fallback"
        )
        agent = make_agent(
            PrefabRetriever({"Paris is the capital of France": "Paris is the capital of France."}),
            ScriptedNLI({"Paris is the capital of France": degraded_support}),
        )
        result = agent.detect("Capital?", "Paris is the capital of France.", documents=BASE_DOCS)
        claim = result.per_claim_results[0]
        assert claim.verification_status == "VERIFIED"
        assert claim.nli_degraded
        assert claim.requires_verification
        assert result.status == "degraded"
        assert not result.nli_inference_executed

    def test_detection_input_context_schema(self):
        from agents.detector_agent.models import DetectionInput

        payload = DetectionInput(
            user_query="Capital?",
            llm_response="Paris is the capital of France.",
            context=["Paris is the capital city of France."],
        )
        assert payload.context == ["Paris is the capital city of France."]