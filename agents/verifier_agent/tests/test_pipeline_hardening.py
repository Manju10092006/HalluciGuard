"""
HalluciGuard Verification Pipeline Hardening Test Suite.

Contains:
1. Deterministic Unit Tests with Mocked Evidence (Tests 1-8)
2. Evidence Classification Invariant Tests (Section 32)
3. Calibrated Confidence Invariant Tests (Section 33)
4. Direct DeBERTa NLI Diagnostic Test (Section 20)
5. Direct BGE Reranker Diagnostic Test (Section 21)
"""
from __future__ import annotations

import os
import sys
import pytest
from typing import Dict, Any, List

# Ensure verifier_agent is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
VERIFIER_DIR = os.path.join(PROJECT_ROOT, "agents", "verifier_agent")
if VERIFIER_DIR not in sys.path:
    sys.path.insert(0, VERIFIER_DIR)

from schemas.models import Passage, VerdictLabel, EntailmentLabel
from scorers.evidence_scorer import EvidenceScorer
from nli import NLIEngine
from rerankers import CrossEncoderReranker


class TestDeterministicScoring:
    """Deterministic tests with mocked evidence evaluating EvidenceScorer & semantics."""

    def setup_method(self):
        self.scorer = EvidenceScorer()

    def test_1_paris_capital_of_france(self):
        """TEST 1: 'Paris is the capital of France.' with direct supporting evidence -> VERIFIED"""
        claim = "Paris is the capital of France."
        passages = [
            Passage(
                title="Wikipedia: Paris",
                source="wikipedia",
                source_id="wiki_paris",
                url="https://en.wikipedia.org/wiki/Paris",
                publication_date="2024-01-01",
                snippet="Paris is the capital and largest city of France.",
                relevance_score=0.98,
            )
        ]
        nli_results = [
            {
                "label": EntailmentLabel.ENTAILMENT,
                "entailment_score": 0.997,
                "contradiction_score": 0.001,
                "neutral_score": 0.002,
                "degraded": False,
            }
        ]
        res = self.scorer.score_evidence(claim, passages, nli_results, "general")
        assert res["verdict"] == VerdictLabel.VERIFIED
        assert res["support_score"] >= 0.50
        assert res["contradiction_score"] == 0.0
        assert res["confidence_score"] >= 0.50
        assert res["evidence_classification_counts"]["supporting"] == 1
        assert res["evidence_classification_counts"]["contradicting"] == 0

    def test_2_eiffel_tower_in_london(self):
        """TEST 2: 'The Eiffel Tower is located in London.' with refuting evidence -> CONTRADICTED"""
        claim = "The Eiffel Tower is located in London."
        passages = [
            Passage(
                title="Wikipedia: Eiffel Tower",
                source="wikipedia",
                source_id="wiki_eiffel",
                url="https://en.wikipedia.org/wiki/Eiffel_Tower",
                publication_date="2024-01-01",
                snippet="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
                relevance_score=0.85,
            )
        ]
        nli_results = [
            {
                "label": EntailmentLabel.CONTRADICTION,
                "entailment_score": 0.0001,
                "contradiction_score": 0.9995,
                "neutral_score": 0.0004,
                "degraded": False,
            }
        ]
        res = self.scorer.score_evidence(claim, passages, nli_results, "general")
        assert res["verdict"] == VerdictLabel.CONTRADICTED
        assert res["contradiction_score"] >= 0.50
        assert res["support_score"] == 0.0
        assert res["confidence_score"] >= 0.40
        assert res["evidence_classification_counts"]["contradicting"] == 1

    def test_3_moon_green_cheese_myth(self):
        """TEST 3: 'The Moon is made of green cheese.' with myth passage -> NEVER VERIFIED"""
        claim = "The Moon is made of green cheese."
        passages = [
            Passage(
                title="Wikipedia: The Moon is made of green cheese",
                source="wikipedia",
                source_id="wiki_moon_cheese",
                url="https://en.wikipedia.org/wiki/The_Moon_is_made_of_green_cheese",
                publication_date="2024-01-01",
                snippet="The Moon is made of green cheese is an English proverb and myth referring to a fanciful belief.",
                relevance_score=0.90,
            )
        ]
        nli_results = [
            {
                "label": EntailmentLabel.ENTAILMENT,  # raw NLI may falsely see lexical overlap
                "entailment_score": 0.85,
                "contradiction_score": 0.05,
                "neutral_score": 0.10,
                "degraded": False,
            }
        ]
        res = self.scorer.score_evidence(claim, passages, nli_results, "general")
        # Must NEVER be VERIFIED
        assert res["verdict"] != VerdictLabel.VERIFIED
        assert res["support_score"] == 0.0
        assert res["evidence_classification_counts"]["neutral"] == 1

    def test_4_xyzabc123_no_valid_evidence(self):
        """TEST 4: 'Xyzabc123 is the capital of France.' with no valid evidence -> UNVERIFIED"""
        claim = "Xyzabc123 is the capital of France."
        passages = []
        nli_results = []
        res = self.scorer.score_evidence(claim, passages, nli_results, "general")
        assert res["verdict"] == VerdictLabel.UNVERIFIED
        assert res["support_score"] == 0.0
        assert res["contradiction_score"] == 0.0
        assert res["confidence_score"] == 0.0

    def test_5_java_created_by_gaurav(self):
        """TEST 5: 'Java was created by Gaurav.' with James Gosling evidence -> CONTRADICTED"""
        claim = "Java was created by Gaurav."
        passages = [
            Passage(
                title="Wikipedia: Java (programming language)",
                source="wikipedia",
                source_id="wiki_java",
                url="https://en.wikipedia.org/wiki/Java_(programming_language)",
                publication_date="2024-01-01",
                snippet="Java was originally developed by James Gosling at Sun Microsystems and released in 1995.",
                relevance_score=0.65,
            )
        ]
        nli_results = [
            {
                "label": EntailmentLabel.CONTRADICTION,
                "entailment_score": 0.0001,
                "contradiction_score": 0.9998,
                "neutral_score": 0.0001,
                "degraded": False,
            }
        ]
        res = self.scorer.score_evidence(claim, passages, nli_results, "general")
        assert res["verdict"] == VerdictLabel.CONTRADICTED
        assert res["contradiction_score"] >= 0.25
        assert res["confidence_score"] > 0.0
        assert res["evidence_classification_counts"]["contradicting"] == 1

    def test_6_healthcare_aspirin(self):
        """TEST 6: Healthcare 'Aspirin is used to relieve mild to moderate pain.' -> VERIFIED"""
        claim = "Aspirin is used to relieve mild to moderate pain."
        passages = [
            Passage(
                title="MedlinePlus: Aspirin",
                source="pubmed",
                source_id="medline_aspirin",
                url="https://medlineplus.gov/druginfo/meds/a682878.html",
                publication_date="2024-01-01",
                snippet="Prescription aspirin is used to relieve the symptoms of rheumatoid arthritis, osteoarthritis, and to relieve mild to moderate pain.",
                relevance_score=0.95,
            )
        ]
        nli_results = [
            {
                "label": EntailmentLabel.ENTAILMENT,
                "entailment_score": 0.985,
                "contradiction_score": 0.001,
                "neutral_score": 0.014,
                "degraded": False,
            }
        ]
        res = self.scorer.score_evidence(claim, passages, nli_results, "healthcare")
        assert res["verdict"] == VerdictLabel.VERIFIED
        assert res["support_score"] >= 0.50
        assert res["evidence_classification_counts"]["supporting"] == 1

    def test_7_cybersecurity_log4shell(self):
        """TEST 7: Cybersecurity 'Log4Shell is associated with CVE-2021-44228.' -> VERIFIED"""
        claim = "Log4Shell is associated with CVE-2021-44228."
        passages = [
            Passage(
                title="NVD: CVE-2021-44228",
                source="nvd",
                source_id="nvd_cve_2021_44228",
                url="https://nvd.nist.gov/vuln/detail/CVE-2021-44228",
                publication_date="2021-12-10",
                snippet="Apache Log4j2 versions 2.0-beta9 through 2.15.0 JNDI features used in configuration, log messages, and parameters do not protect against attacker controlled LDAP and other JNDI related endpoints (Log4Shell).",
                relevance_score=0.99,
            )
        ]
        nli_results = [
            {
                "label": EntailmentLabel.ENTAILMENT,
                "entailment_score": 0.992,
                "contradiction_score": 0.002,
                "neutral_score": 0.006,
                "degraded": False,
            }
        ]
        res = self.scorer.score_evidence(claim, passages, nli_results, "cybersecurity")
        assert res["verdict"] == VerdictLabel.VERIFIED
        assert res["support_score"] >= 0.50
        assert res["evidence_classification_counts"]["supporting"] == 1

    def test_8_ai_research_transformers(self):
        """TEST 8: AI Research 'Transformers were introduced in the paper Attention Is All You Need.' -> VERIFIED"""
        claim = "Transformers were introduced in the paper Attention Is All You Need."
        passages = [
            Passage(
                title="arXiv: 1706.03762",
                source="arxiv",
                source_id="arxiv_1706_03762",
                url="https://arxiv.org/abs/1706.03762",
                publication_date="2017-06-12",
                snippet="We propose the Transformer, a model architecture eschewing recurrence and relying entirely on an attention mechanism to draw global dependencies.",
                relevance_score=0.97,
            )
        ]
        nli_results = [
            {
                "label": EntailmentLabel.ENTAILMENT,
                "entailment_score": 0.994,
                "contradiction_score": 0.001,
                "neutral_score": 0.005,
                "degraded": False,
            }
        ]
        res = self.scorer.score_evidence(claim, passages, nli_results, "ai_research")
        assert res["verdict"] == VerdictLabel.VERIFIED
        assert res["support_score"] >= 0.50
        assert res["evidence_classification_counts"]["supporting"] == 1


class TestClassificationInvariants:
    """Prove evidence classification invariants (Section 32)."""

    def test_invariant_sum_equals_total(self):
        scorer = EvidenceScorer()
        claim = "Paris is the capital of France."
        passages = [
            Passage(title="P1", source="s1", url="u1", publication_date="2024-01-01", snippet="Paris is capital of France", relevance_score=0.9),
            Passage(title="P2", source="s2", url="u2", publication_date="2024-01-01", snippet="Paris is not in France", relevance_score=0.8),
            Passage(title="P3", source="s3", url="u3", publication_date="2024-01-01", snippet="France has wine", relevance_score=0.5),
            Passage(title="P4", source="s4", url="u4", publication_date="2024-01-01", snippet="Unrelated noise", relevance_score=0.001),
        ]
        nli_results = [
            {"label": EntailmentLabel.ENTAILMENT, "entailment_score": 0.95, "contradiction_score": 0.02, "neutral_score": 0.03, "degraded": False},
            {"label": EntailmentLabel.CONTRADICTION, "entailment_score": 0.01, "contradiction_score": 0.98, "neutral_score": 0.01, "degraded": False},
            {"label": EntailmentLabel.NEUTRAL, "entailment_score": 0.10, "contradiction_score": 0.05, "neutral_score": 0.85, "degraded": False},
            {"label": EntailmentLabel.NEUTRAL, "entailment_score": 0.05, "contradiction_score": 0.05, "neutral_score": 0.90, "degraded": False},
        ]
        res = scorer.score_evidence(claim, passages, nli_results, "general")
        counts = res["evidence_classification_counts"]

        assert counts["supporting"] >= 0
        assert counts["contradicting"] >= 0
        assert counts["neutral"] >= 0
        assert counts["irrelevant"] >= 0
        assert counts["supporting"] + counts["contradicting"] + counts["neutral"] + counts["irrelevant"] == len(passages)


class TestConfidenceInvariants:
    """Prove confidence invariants across qualitative behaviors (Section 33)."""

    def setup_method(self):
        self.scorer = EvidenceScorer()

    def test_strong_support_high_confidence(self):
        claim = "Test claim"
        passages = [Passage(title="P1", source="s1", url="u1", publication_date="2024-01-01", snippet="Support test claim text", relevance_score=0.9)]
        nli = [{"label": EntailmentLabel.ENTAILMENT, "entailment_score": 0.99, "contradiction_score": 0.0, "neutral_score": 0.01, "degraded": False}]
        res = self.scorer.score_evidence(claim, passages, nli, "general")
        assert res["verdict"] == VerdictLabel.VERIFIED
        assert res["confidence_score"] >= 0.50

    def test_strong_contradiction_high_confidence(self):
        claim = "Test claim"
        passages = [Passage(title="P1", source="s1", url="u1", publication_date="2024-01-01", snippet="Test claim is false and refuted", relevance_score=0.9)]
        nli = [{"label": EntailmentLabel.CONTRADICTION, "entailment_score": 0.0, "contradiction_score": 0.99, "neutral_score": 0.01, "degraded": False}]
        res = self.scorer.score_evidence(claim, passages, nli, "general")
        assert res["verdict"] == VerdictLabel.CONTRADICTED
        assert res["confidence_score"] >= 0.50

    def test_conflict_reduced_confidence(self):
        claim = "Test claim"
        passages = [
            Passage(title="P1", source="s1", url="u1", publication_date="2024-01-01", snippet="Support test claim text", relevance_score=0.9),
            Passage(title="P2", source="s2", url="u2", publication_date="2024-01-01", snippet="Test claim is false and refuted", relevance_score=0.9),
        ]
        nli = [
            {"label": EntailmentLabel.ENTAILMENT, "entailment_score": 0.95, "contradiction_score": 0.02, "neutral_score": 0.03, "degraded": False},
            {"label": EntailmentLabel.CONTRADICTION, "entailment_score": 0.02, "contradiction_score": 0.95, "neutral_score": 0.03, "degraded": False},
        ]
        res = self.scorer.score_evidence(claim, passages, nli, "general")
        assert res["verdict"] == VerdictLabel.CONFLICTED
        # Confidence must be reduced due to low consensus
        assert res["confidence_score"] < 0.35

    def test_no_evidence_zero_confidence(self):
        res = self.scorer.score_evidence("Test claim", [], [], "general")
        assert res["verdict"] == VerdictLabel.UNVERIFIED
        assert res["confidence_score"] == 0.0


class TestDirectModelDiagnostics:
    """Direct ML model execution diagnostics (Sections 20 & 21)."""

    def test_direct_deberta_nli_execution(self):
        """Execute real cross-encoder/nli-deberta-v3-base inference."""
        nli = NLIEngine()
        res_entail = nli.classify(
            "Paris is the capital of France.",
            "Paris is the capital and largest city of France.",
        )
        assert res_entail["label"] == EntailmentLabel.ENTAILMENT
        assert res_entail["entailment_score"] > 0.95
        assert res_entail["entailment_score"] + res_entail["contradiction_score"] + res_entail["neutral_score"] == pytest.approx(1.0, abs=1e-3)

        res_contra = nli.classify(
            "The Eiffel Tower is located in London.",
            "The Eiffel Tower is located in Paris, France.",
        )
        assert res_contra["label"] == EntailmentLabel.CONTRADICTION
        assert res_contra["contradiction_score"] > 0.95

    def test_direct_bge_reranker_execution(self):
        """Execute real BAAI/bge-reranker-large inference."""
        reranker = CrossEncoderReranker()
        claim = "Paris is the capital of France."
        passages = [
            Passage(title="Paris city", source="s1", url="u1", publication_date="2024-01-01", snippet="Paris is the capital and largest city of France.", relevance_score=0.5),
            Passage(title="Man United", source="s2", url="u2", publication_date="2024-01-01", snippet="Manchester United is an English football club based in Old Trafford.", relevance_score=0.5),
            Passage(title="France info", source="s3", url="u3", publication_date="2024-01-01", snippet="France is a country in Western Europe.", relevance_score=0.5),
        ]
        reranked = reranker.rerank(claim, passages, k=3)
        assert len(reranked) == 3
        # Top passage must be the direct Paris definition
        assert reranked[0].title == "Paris city"
        assert reranked[0].relevance_score > reranked[1].relevance_score
        assert reranked[0].relevance_score > reranked[2].relevance_score

