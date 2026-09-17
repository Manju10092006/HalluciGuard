"""
Phase-2 hardening tests: model loading, NLI token truncation, multi-signal
scoring, and one-time pipeline construction.

Defects covered:
  - #6  models must load once (process-wide pipeline singleton + ModelManager)
  - #8  claim+evidence pairs longer than 512 tokens must truncate, not error
  - #10 NLI must not be the sole truth: relevance/source/relevance/diversity
        gate the verdict and are exposed for observability
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas.models import Passage, VerdictLabel


def make_passage(
    source="wikipedia",
    source_id="wiki_test",
    url="https://en.wikipedia.org/wiki/Test",
    relevance_score=0.8,
    publication_date="2024-01-01",
):
    return Passage(
        title="Test Article",
        snippet="Test claim snippet with sufficient content for verification.",
        source=source,
        source_id=source_id,
        url=url,
        relevance_score=relevance_score,
        publication_date=publication_date,
    )


def make_nli(entailment=0.0, contradiction=0.0, neutral=0.0, nli_degraded=False, validity_factor=1.0):
    return {
        "entailment_score": entailment,
        "contradiction_score": contradiction,
        "neutral_score": neutral,
        "label": "entailment" if entailment >= max(contradiction, neutral) else "neutral",
        "nli_degraded": nli_degraded,
        "validity_factor": validity_factor,
    }


# ---------------------------------------------------------------------------
# #10 — multi-signal scoring
# ---------------------------------------------------------------------------

def test_multi_signal_breakdown_exposed_with_diversity_bonus():
    from scorers.evidence_scorer import EvidenceScorer

    scorer = EvidenceScorer()
    passages = [
        make_passage(source="wikipedia", source_id="wiki_a", url="https://a.example/x"),
        make_passage(source="britannica", source_id="brit_1", url="https://b.example/y"),
    ]
    nli = [make_nli(entailment=0.9), make_nli(entailment=0.8)]
    res = scorer.score_evidence("test claim here", passages, nli, "general")

    assert res["signals"]["diversity_supporting_sources"] >= 2
    assert res["signals"]["support_diversity_bonus"] > 0.0
    assert res["signals"]["evidence_quality"] > 0.0
    assert res["signals"]["source_credibility_floor"] > 0.0
    assert res["signals"]["relevance_floor"] > 0.0
    assert res["verdict"] == VerdictLabel.VERIFIED


def test_high_nli_with_irrelevant_evidence_is_not_verified():
    """NLI support alone cannot force a verdict past the relevance gate."""
    from scorers.evidence_scorer import EvidenceScorer

    scorer = EvidenceScorer()
    passages = [make_passage(relevance_score=0.05)]  # below the 0.20 gate
    nli = [make_nli(entailment=0.98)]
    res = scorer.score_evidence("test claim here", passages, nli, "general")

    assert res["evidence_classification_counts"]["supporting"] == 0
    assert res["verdict"] != VerdictLabel.VERIFIED


# ---------------------------------------------------------------------------
# #8 — NLI token truncation (518 > 512)
# ---------------------------------------------------------------------------

def test_nli_pipeline_built_with_truncation_kwargs():
    """claim+evidence pairs routinely exceed 512 tokens (e.g. 518 > 512);
    the loader must truncate — keeping the short claim intact — rather than
    crash with an indexing error."""
    from models.model_manager import _nli_pipeline_kwargs

    kwargs = _nli_pipeline_kwargs("some-model", -1)
    assert kwargs["tokenizer_kwargs"] == {
        "truncation": "only_first",
        "max_length": 512,
    }
    offline = _nli_pipeline_kwargs("some-model", -1, local_files_only=True)
    assert offline["model_kwargs"] == {"local_files_only": True}
    assert offline["tokenizer_kwargs"] == kwargs["tokenizer_kwargs"]


# ---------------------------------------------------------------------------
# #6 — process-wide pipeline singleton (load once)
# ---------------------------------------------------------------------------

def test_get_pipeline_returns_same_instance():
    from api.pipeline import get_pipeline

    first = get_pipeline()
    second = get_pipeline()
    assert first is second