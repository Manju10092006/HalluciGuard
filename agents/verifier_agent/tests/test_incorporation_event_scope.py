"""Reproduced live founding/incorporation false accept; no score synthesis."""
import pytest
from api.pipeline import VerificationPipeline
from schemas.models import Passage


def passage(text):
    return Passage(title="Microsoft", source="government", source_id="fixture",
                   url="https://www.microsoft.com/en-us/investor/faq", snippet=text,
                   publication_date="unknown")


@pytest.mark.parametrize("evidence", [
    "Gates and Allen established Microsoft on April 4, 1975, with Gates as CEO.",
    "Microsoft is a computer technology corporation founded on April 4, 1975, by Bill Gates and Paul Allen in Albuquerque.",
    "Established on April 4, 1975, to develop and sell BASIC interpreters, Microsoft rose to dominate the home computer.",
])
def test_real_distractor_snippets_cannot_verify_incorporation_date(evidence):
    scores = {"entailment_score": .99, "contradiction_score": .005, "neutral_score": .005}
    selected, output = VerificationPipeline._select_decision_grade_evidence(
        [passage(evidence)], [scores], claim="Microsoft was officially incorporated on April 4, 1975.")
    assert selected == output == []
    assert scores["entailment_score"] == .99


@pytest.mark.parametrize("claim,evidence", [
    ("Microsoft was incorporated in 1981.", "Microsoft was incorporated in the state of Washington on June 25, 1981."),
    ("Microsoft was incorporated in 1975.", "Microsoft became a privately held corporation in 1981."),
    ("Microsoft was founded in 1975.", "Microsoft was founded in 1975."),
])
def test_comparable_event_is_left_to_real_nli(claim, evidence):
    assert VerificationPipeline._incorporation_event_absent(claim, passage(evidence)) is False


def test_absent_event_does_not_manufacture_contradiction_even_for_negation():
    scores = {"entailment_score": .005, "contradiction_score": .99, "neutral_score": .005}
    selected, _ = VerificationPipeline._select_decision_grade_evidence(
        [passage("Microsoft was founded in 1975.")], [scores],
        claim="Microsoft was not incorporated in 1975.")
    assert selected == []


def test_old_false_accept_cache_key_cannot_survive_event_scope_upgrade():
    import hashlib
    from cache.sqlite_cache import SqliteCache
    claim = "Microsoft was officially incorporated on April 4, 1975."
    old = hashlib.sha256(f"verifier-v2.3-evidence-scope:general:{claim.lower()}".encode()).hexdigest()
    assert SqliteCache()._normalize_key("general", claim) != old
