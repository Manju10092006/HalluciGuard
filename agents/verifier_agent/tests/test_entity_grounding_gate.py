"""Regression guard for the SUPPORTING entity/subject-grounding gate.

Root cause fixed here: a passage that merely entails the sentence FORM of a claim
was classified SUPPORTING even when it never mentioned the claim's discriminating
entities. That produced a false VERIFIED for unverifiable personal claims (e.g.
"Kushal is the topper of KMIT" grounded on a generic "a topper is..." passage),
which then poisoned memory.

The gate admits a passage as SUPPORTING only when it is lexically about the
claim's subject terms OR the entailment is near-certain (authoritative
paraphrase). These deterministic tests pin both directions.
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
VERIFIER_DIR = os.path.join(PROJECT_ROOT, "agents", "verifier_agent")
if VERIFIER_DIR not in sys.path:
    sys.path.insert(0, VERIFIER_DIR)

from schemas.models import Passage, VerdictLabel, EntailmentLabel
from scorers.evidence_scorer import EvidenceScorer


def _passage(snippet: str, relevance: float = 0.75) -> Passage:
    return Passage(
        title="",
        source="web",
        source_id="web_1",
        url="https://example.com/x",
        publication_date="2024-01-01",
        snippet=snippet,
        relevance_score=relevance,
    )


def test_generic_entity_mismatched_evidence_does_not_verify():
    """A generic definition entails the sentence form but names neither 'Kushal'
    nor 'KMIT' -> must NOT be treated as SUPPORTING, and the claim stays
    unverified rather than falsely verified."""
    scorer = EvidenceScorer()
    claim = "Kushal is the topper of KMIT hyderabad"
    passage = _passage("A topper is the highest-ranking student in a class or examination.")
    nli = [{
        "label": EntailmentLabel.ENTAILMENT,
        "entailment_score": 0.55,   # moderate: clears MIN_NLI_SIGNAL, below STRONG
        "contradiction_score": 0.10,
        "neutral_score": 0.35,
        "degraded": False,
    }]

    assert scorer.classify_evidence(claim, passage, nli[0]) == "NEUTRAL"
    result = scorer.score_evidence(claim, [passage], nli, "general")
    assert result["verdict"] != VerdictLabel.VERIFIED
    assert result["support_score"] < 0.30


def test_entity_grounded_evidence_verifies_at_moderate_entailment():
    """When the evidence actually mentions the claim's subject terms, a moderate
    entailment is enough to support it."""
    scorer = EvidenceScorer()
    claim = "Kushal is the topper of KMIT hyderabad"
    passage = _passage("Kushal secured the highest rank and is the topper at KMIT Hyderabad this year.")
    nli = [{
        "label": EntailmentLabel.ENTAILMENT,
        "entailment_score": 0.55,
        "contradiction_score": 0.05,
        "neutral_score": 0.40,
        "degraded": False,
    }]

    assert scorer.classify_evidence(claim, passage, nli[0]) == "SUPPORTING"
    result = scorer.score_evidence(claim, [passage], nli, "general")
    assert result["verdict"] == VerdictLabel.VERIFIED


def test_strong_entailment_paraphrase_still_supports_without_lexical_overlap():
    """Near-certain entailment from a relevance-gated passage supports even under
    heavy paraphrase / coreference (the authoritative-source case)."""
    scorer = EvidenceScorer()
    claim = "Aspirin is used to treat mild pain."
    passage = _passage(
        "The official drug label states that the medicine provides temporary relief of minor aches.",
        relevance=0.85,
    )
    nli = [{
        "label": EntailmentLabel.ENTAILMENT,
        "entailment_score": 0.96,   # near-certain
        "contradiction_score": 0.01,
        "neutral_score": 0.03,
        "degraded": False,
    }]

    assert scorer.classify_evidence(claim, passage, nli[0]) == "SUPPORTING"
    result = scorer.score_evidence(claim, [passage], nli, "healthcare")
    assert result["verdict"] == VerdictLabel.VERIFIED
