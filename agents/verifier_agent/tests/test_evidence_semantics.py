import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from schemas.models import Passage, VerdictLabel, EntailmentLabel
from scorers.evidence_scorer import EvidenceScorer

def make_passage(
    title="Test Article",
    snippet="Test claim snippet with sufficient content for verification.",
    source="wikipedia",
    source_id="wiki_test",
    url="https://en.wikipedia.org/wiki/Test",
    relevance_score=0.75,
    publication_date="2024-01-01",
):
    return Passage(
        title=title,
        snippet=snippet,
        source=source,
        source_id=source_id,
        url=url,
        relevance_score=relevance_score,
        publication_date=publication_date,
    )

def make_nli(entailment=0.0, contradiction=0.0, neutral=0.0, label=None, nli_degraded=False, validity_factor=1.0):
    if label is None:
        if entailment > contradiction and entailment > neutral:
            label = "entailment"
        elif contradiction > entailment and contradiction > neutral:
            label = "contradiction"
        else:
            label = "neutral"
    return {
        "entailment_score": entailment,
        "contradiction_score": contradiction,
        "neutral_score": neutral,
        "label": label,
        "nli_degraded": nli_degraded,
        "validity_factor": validity_factor,
    }

def test_relevant_support_verified():
    scorer = EvidenceScorer()
    passages = [make_passage(relevance_score=0.8)]
    nli = [make_nli(entailment=0.9)]
    res = scorer.score_evidence("test claim", passages, nli, "test_domain")
    assert res["evidence_classification_counts"]["supporting"] == 1
    assert res["verdict"] == VerdictLabel.VERIFIED

def test_relevant_contradiction_contradicted():
    scorer = EvidenceScorer()
    passages = [make_passage(relevance_score=0.8)]
    nli = [make_nli(contradiction=0.9)]
    res = scorer.score_evidence("test claim", passages, nli, "test_domain")
    assert res["evidence_classification_counts"]["contradicting"] == 1
    assert res["verdict"] == VerdictLabel.CONTRADICTED

def test_neutral_evidence_unverified():
    scorer = EvidenceScorer()
    passages = [make_passage(relevance_score=0.8)]
    nli = [make_nli(neutral=0.95)]
    res = scorer.score_evidence("test claim", passages, nli, "test_domain")
    assert res["evidence_classification_counts"]["neutral"] == 1
    assert res["verdict"] == VerdictLabel.UNVERIFIED

def test_irrelevant_high_nli_contradiction_rejected():
    scorer = EvidenceScorer()
    passages = [make_passage(relevance_score=0.05)]
    nli = [make_nli(contradiction=0.99)]
    res = scorer.score_evidence("test claim", passages, nli, "test_domain")
    assert res["evidence_classification_counts"]["irrelevant"] == 1
    assert res["verdict"] == VerdictLabel.UNVERIFIED
    assert res["contradiction_score"] == 0.0

def test_genuine_conflict_conflicted():
    scorer = EvidenceScorer()
    passages = [
        make_passage(relevance_score=0.8, url="http://url1.com", source_id="s1"),
        make_passage(relevance_score=0.8, url="http://url2.com", source_id="s2")
    ]
    nli = [
        make_nli(entailment=0.8),
        make_nli(contradiction=0.8)
    ]
    res = scorer.score_evidence("test claim", passages, nli, "test_domain")
    assert res["evidence_classification_counts"]["supporting"] == 1
    assert res["evidence_classification_counts"]["contradicting"] == 1
    assert res["verdict"] == VerdictLabel.CONFLICTED

def test_duplicate_url_deduplication():
    scorer = EvidenceScorer()
    passages = [
        make_passage(relevance_score=0.8, url="http://same.com", source_id="s1"),
        make_passage(relevance_score=0.7, url="http://same.com", source_id="s1"),
        make_passage(relevance_score=0.9, url="http://same.com", source_id="s1")
    ]
    nli = [
        make_nli(entailment=0.9),
        make_nli(entailment=0.8),
        make_nli(entailment=0.95)
    ]
    res_multi = scorer.score_evidence("test claim", passages, nli, "test_domain")
    
    single_passage = [make_passage(relevance_score=0.9, url="http://same.com", source_id="s1")]
    single_nli = [make_nli(entailment=0.95)]
    res_single = scorer.score_evidence("test claim", single_passage, single_nli, "test_domain")
    
    assert res_multi["support_score"] == res_single["support_score"]
    assert res_multi["verdict"] == VerdictLabel.VERIFIED

def test_degraded_nli_cannot_verify():
    scorer = EvidenceScorer()
    passages = [make_passage(relevance_score=0.8)]
    nli = [make_nli(entailment=0.9, nli_degraded=True)]
    res = scorer.score_evidence("test claim", passages, nli, "test_domain")
    assert res["evidence_classification_counts"]["neutral"] == 1
    assert res["verdict"] == VerdictLabel.UNVERIFIED

def test_empty_evidence_unverified():
    scorer = EvidenceScorer()
    res = scorer.score_evidence("test claim", [], [], "test_domain")
    assert res["verdict"] == VerdictLabel.UNVERIFIED
    assert res["support_score"] == 0.0
    assert res["contradiction_score"] == 0.0

def test_myth_context_neutral():
    scorer = EvidenceScorer()
    claim = "The Moon is made of green cheese"
    passage = make_passage(snippet="It is a common myth and folklore that the Moon is made of green cheese")
    nli = make_nli(entailment=0.9)
    label = scorer.classify_evidence(claim, passage, nli)
    assert label == "NEUTRAL"

def test_classification_invariant():
    scorer = EvidenceScorer()
    passages = [
        make_passage(relevance_score=0.8, url="u1"),
        make_passage(relevance_score=0.8, url="u2"),
        make_passage(relevance_score=0.8, url="u3"),
        make_passage(relevance_score=0.8, url="u4"),
        make_passage(relevance_score=0.1, url="u5"),
        make_passage(relevance_score=0.1, url="u6")
    ]
    nli = [
        make_nli(entailment=0.9),
        make_nli(entailment=0.9),
        make_nli(contradiction=0.9),
        make_nli(neutral=0.9),
        make_nli(entailment=0.9),
        make_nli(contradiction=0.9)
    ]
    res = scorer.score_evidence("test claim", passages, nli, "test_domain")
    counts = res["evidence_classification_counts"]
    
    assert counts["supporting"] == 2
    assert counts["contradicting"] == 1
    assert counts["neutral"] == 1
    assert counts["irrelevant"] == 2
    
    total = counts["total"]
    assert total == 6
    assert counts["supporting"] + counts["contradicting"] + counts["neutral"] + counts["irrelevant"] == 6
    assert sum(counts.values()) - total == len(passages)

def test_confidence_nonzero_for_contradicted():
    scorer = EvidenceScorer()
    passages = [make_passage(relevance_score=0.8)]
    nli = [make_nli(contradiction=0.9)]
    res = scorer.score_evidence("test claim", passages, nli, "test_domain")
    assert res["verdict"] == VerdictLabel.CONTRADICTED
    assert res["contradiction_score"] >= 0.25
    assert res["confidence_score"] > 0.0

def test_adversarial_debunking_article():
    scorer = EvidenceScorer()
    claim = "vaccines cause illness"
    passage = make_passage(snippet="Scientists debunked the false belief that vaccines cause illness.")
    nli = make_nli(entailment=0.9)
    label = scorer.classify_evidence(claim, passage, nli)
    assert label in ("NEUTRAL", "CONTRADICTING")

