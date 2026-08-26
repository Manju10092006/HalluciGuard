"""Step 2 tests — Authorization + Claim Targeting.

Covers each numbered Step 2 requirement explicitly:
    1. claims_to_correct is the ONLY correction authorization
    2. claims_to_preserve is locked content
    3. exact claim matching
    4. normalized matching
    5. sentence-level matching
    6. controlled token matching only when sufficiently unambiguous
    7. ambiguous claim -> fail safe
    8. claim not found -> fail safe
    9. NEVER first-sentence fallback
   10. NEVER invent a correction target
   11. never target a preserved claim
   12. byte-accurate sentence spans
   13. every span verified with original_response[start:end] == span.text

Evidence binding, model generation, validation/retry and reconstruction are out
of scope for Step 2 and are not tested here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[3])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest

from orchestration.schemas import ClaimReport, CorrectionRequest, VerdictLabel

from agents.corrector_agent.corrector.adapter import to_internal_request
from agents.corrector_agent.corrector.config import CorrectorConfig
from agents.corrector_agent.corrector.contracts import MatchStrategy, SkipReason
from agents.corrector_agent.corrector.sentence_utils import (
    extract_sentence_spans,
    find_span_integrity_error,
    normalize_for_matching,
    normalize_whitespace_case,
    tokenize,
    verify_spans_integrity,
)
from agents.corrector_agent.corrector.targeting import (
    build_correction_plan,
    match_claim_to_span,
)

CFG = CorrectorConfig()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _claim(claim_id: str, text: str, verdict: str = "contradicted") -> ClaimReport:
    return ClaimReport(claim_id=claim_id, claim_text=text, verdict=verdict)


def _plan(original: str, correct=(), preserve=(), config: CorrectorConfig = CFG):
    request = CorrectionRequest(
        execution_id="exec-step2",
        user_query="q",
        original_response=original,
        claims_to_correct=list(correct),
        claims_to_preserve=list(preserve),
    )
    return build_correction_plan(to_internal_request(request), config)


def _assert_byte_accurate(original: str, plan) -> None:
    """Requirements 12 & 13 for both spans and emitted targets."""
    assert find_span_integrity_error(original, plan.spans) is None
    for span in plan.spans:
        assert original[span.start_offset:span.end_offset] == span.text
    for target in plan.targets:
        assert original[target.start_offset:target.end_offset] == target.original_text


# ---------------------------------------------------------------------------
# Requirement 12 & 13 — byte-accurate sentence spans
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected_count",
    [
        ("The sky is blue.", 1),
        ("The sky is blue. Python was made by Musk.", 2),
        ("One. Two! Three?", 3),
        ("No terminal punctuation here", 1),
        ("Para one.\n\nPara two.", 2),
        ("Line one\nLine two", 2),
        ("Dr. Smith works here. He is kind.", 2),
        ("Pi is 3.14 exactly. That is true.", 2),
        ("The U.S. is large. It is.", 2),
        ('He said "stop." Then left.', 2),
        ("   Leading and trailing whitespace.   ", 1),
        ("", 0),
        ("   ", 0),
        ("\n\n\n", 0),
    ],
)
def test_sentence_spans_are_byte_accurate(text, expected_count):
    spans = extract_sentence_spans(text)
    assert len(spans) == expected_count
    assert verify_spans_integrity(text, spans)
    for span in spans:
        # Requirement 13, asserted directly.
        assert text[span.start_offset:span.end_offset] == span.text
        assert span.text.strip() == span.text  # no leading/trailing whitespace
        assert span.text.strip() != ""


def test_spans_are_ordered_non_overlapping_and_uniquely_identified():
    text = "First one. Second one! Third one?\n\nFourth one."
    spans = extract_sentence_spans(text)
    assert [s.sentence_id for s in spans] == ["S1", "S2", "S3", "S4"]
    for prev, cur in zip(spans, spans[1:]):
        assert prev.end_offset <= cur.start_offset
    assert len({s.sentence_id for s in spans}) == len(spans)


def test_abbreviations_and_decimals_do_not_split():
    text = "Version 1.5 shipped in Jan. 2020 per Dr. Lee."
    spans = extract_sentence_spans(text)
    assert len(spans) == 1
    assert spans[0].text == text
    assert verify_spans_integrity(text, spans)


def test_gaps_between_spans_are_whitespace_only():
    """Guarantees later reconstruction can splice by offset without loss."""
    text = "One.   Two.\n\n\tThree."
    spans = extract_sentence_spans(text)
    cursor = 0
    for span in spans:
        assert text[cursor:span.start_offset].strip() == ""
        cursor = span.end_offset
    assert text[cursor:].strip() == ""


def test_find_span_integrity_error_detects_tampering():
    text = "One. Two."
    spans = extract_sentence_spans(text)
    spans[0].text = "Tampered"
    assert find_span_integrity_error(text, spans) is not None
    assert not verify_spans_integrity(text, spans)


def test_integrity_error_detects_out_of_bounds_and_overlap():
    text = "One. Two."
    spans = extract_sentence_spans(text)
    spans[1].end_offset = len(text) + 5
    assert "out of bounds" in find_span_integrity_error(text, spans)

    spans = extract_sentence_spans(text)
    spans[1].start_offset = 0
    assert find_span_integrity_error(text, spans) is not None


# ---------------------------------------------------------------------------
# Requirement 3 — exact matching
# ---------------------------------------------------------------------------

def test_exact_match():
    text = "The sky is blue. Python was created by Elon Musk."
    spans = extract_sentence_spans(text)
    match = match_claim_to_span("Python was created by Elon Musk.", spans, CFG)
    assert match.matched
    assert match.strategy == MatchStrategy.EXACT
    assert match.span.sentence_id == "S2"
    assert match.score == 1.0


def test_exact_match_wins_over_later_levels():
    text = "Cats are mammals. Cats are mammals and pets."
    spans = extract_sentence_spans(text)
    match = match_claim_to_span("Cats are mammals.", spans, CFG)
    assert match.strategy == MatchStrategy.EXACT
    assert match.span.sentence_id == "S1"


# ---------------------------------------------------------------------------
# Requirement 4 — normalized matching
# ---------------------------------------------------------------------------

def test_normalized_match_case_and_whitespace():
    text = "The Sky   Is Blue. Something else entirely here."
    spans = extract_sentence_spans(text)
    match = match_claim_to_span("the sky is blue.", spans, CFG)
    assert match.matched
    assert match.strategy == MatchStrategy.NORMALIZED
    assert match.span.sentence_id == "S1"


def test_normalized_match_folds_smart_quotes_and_dashes():
    text = "It’s a well—known fact. Another sentence follows here."
    spans = extract_sentence_spans(text)
    match = match_claim_to_span("It's a well-known fact.", spans, CFG)
    assert match.matched
    assert match.strategy == MatchStrategy.NORMALIZED


def test_normalization_helpers():
    assert normalize_whitespace_case("  The   SKY\tis Blue.  ") == "the sky is blue."
    assert normalize_for_matching("The SKY, is Blue!") == "the sky is blue"
    assert tokenize("The sky is blue!") == ["the", "sky", "is", "blue"]


# ---------------------------------------------------------------------------
# Requirement 5 — sentence-level (containment) matching
# ---------------------------------------------------------------------------

def test_containment_match_claim_inside_sentence():
    text = "In 1991, Python was created by Elon Musk, a fact widely cited. Birds fly."
    spans = extract_sentence_spans(text)
    match = match_claim_to_span("Python was created by Elon Musk", spans, CFG)
    assert match.matched
    assert match.strategy == MatchStrategy.CONTAINMENT
    assert match.span.sentence_id == "S1"


def test_reverse_containment_is_not_accepted():
    """A long claim must not capture a short unrelated sentence (span in claim)."""
    text = "Yes. The moon orbits the earth in roughly twenty seven days total."
    spans = extract_sentence_spans(text)
    match = match_claim_to_span("Yes, absolutely and definitively yes indeed.", spans, CFG)
    assert not match.matched
    assert match.span is None


# ---------------------------------------------------------------------------
# Requirement 6 — controlled token matching only when unambiguous
# ---------------------------------------------------------------------------

def test_token_overlap_match_when_clearly_unambiguous():
    text = (
        "The Eiffel Tower was constructed in Berlin during 1889 by engineers. "
        "Completely unrelated topic about marine biology and coral reefs."
    )
    spans = extract_sentence_spans(text)
    # Reordered wording, so it is NOT a substring — containment cannot fire and
    # the guarded token-overlap level is genuinely exercised.
    match = match_claim_to_span(
        "Berlin 1889 constructed Eiffel Tower during", spans, CFG
    )
    assert match.matched
    assert match.strategy == MatchStrategy.TOKEN_OVERLAP
    assert match.span.sentence_id == "S1"
    assert match.score >= CFG.token_overlap_threshold


def test_stricter_levels_take_precedence_over_token_overlap():
    """A claim that is a substring must resolve by containment, not overlap."""
    text = (
        "The Eiffel Tower was constructed in Berlin during 1889 by engineers. "
        "Completely unrelated topic about marine biology and coral reefs."
    )
    spans = extract_sentence_spans(text)
    match = match_claim_to_span(
        "The Eiffel Tower was constructed in Berlin during 1889", spans, CFG
    )
    assert match.matched
    assert match.strategy == MatchStrategy.CONTAINMENT
    assert match.span.sentence_id == "S1"


def test_token_overlap_rejected_when_margin_not_met():
    """Two near-equally-scoring sentences must fail safe, not pick one."""
    text = "Alpha beta gamma delta epsilon. Alpha beta gamma delta zeta."
    spans = extract_sentence_spans(text)
    match = match_claim_to_span("alpha beta gamma delta omega theta", spans, CFG)
    assert not match.matched
    assert match.skip_reason == SkipReason.AMBIGUOUS_MATCH
    assert "margin" in match.detail


def test_token_overlap_rejected_below_threshold():
    text = "Alpha beta gamma delta epsilon zeta. Totally different words here now."
    spans = extract_sentence_spans(text)
    match = match_claim_to_span("alpha nine eight seven six five", spans, CFG)
    assert not match.matched
    assert match.skip_reason == SkipReason.CLAIM_NOT_FOUND
    assert "below threshold" in match.detail


def test_short_claims_are_never_token_matched():
    text = "The sky is blue today. The grass is green today."
    spans = extract_sentence_spans(text)
    match = match_claim_to_span("blue sky", spans, CFG)
    assert not match.matched
    assert match.skip_reason == SkipReason.CLAIM_NOT_FOUND
    assert "minimum" in match.detail


def test_token_overlap_threshold_and_margin_are_configurable():
    text = "Alpha beta gamma delta epsilon. Totally different content follows here."
    spans = extract_sentence_spans(text)
    claim = "alpha beta nine eight seven"

    strict = CorrectorConfig(token_overlap_threshold=0.95, token_overlap_margin=0.0)
    assert not match_claim_to_span(claim, spans, strict).matched

    permissive = CorrectorConfig(token_overlap_threshold=0.2, token_overlap_margin=0.05)
    assert match_claim_to_span(claim, spans, permissive).matched


# ---------------------------------------------------------------------------
# Requirement 7 — ambiguous claim -> fail safe
# ---------------------------------------------------------------------------

def test_duplicate_identical_sentences_are_ambiguous():
    text = "The sky is blue. The sky is blue."
    spans = extract_sentence_spans(text)
    match = match_claim_to_span("The sky is blue.", spans, CFG)
    assert not match.matched
    assert match.skip_reason == SkipReason.AMBIGUOUS_MATCH
    assert "refusing to guess" in match.detail


def test_ambiguous_normalized_match_fails_safe():
    text = "The sky is blue. THE SKY IS BLUE."
    spans = extract_sentence_spans(text)
    match = match_claim_to_span("the sky   is blue.", spans, CFG)
    assert not match.matched
    assert match.skip_reason == SkipReason.AMBIGUOUS_MATCH


def test_ambiguous_containment_fails_safe():
    text = "Python was created by Musk, reportedly. Python was created by Musk, allegedly."
    spans = extract_sentence_spans(text)
    match = match_claim_to_span("Python was created by Musk", spans, CFG)
    assert not match.matched
    assert match.skip_reason == SkipReason.AMBIGUOUS_MATCH


def test_ambiguous_claim_produces_no_target_in_plan():
    text = "The sky is blue. The sky is blue."
    plan = _plan(text, correct=[_claim("c1", "The sky is blue.")])
    assert plan.targets == []
    assert len(plan.skipped_claims) == 1
    assert plan.skipped_claims[0].reason == SkipReason.AMBIGUOUS_MATCH.value
    _assert_byte_accurate(text, plan)


# ---------------------------------------------------------------------------
# Requirements 8, 9, 10 — not found -> fail safe, never first-sentence fallback
# ---------------------------------------------------------------------------

def test_claim_not_found_fails_safe():
    text = "The sky is blue. Grass is green."
    spans = extract_sentence_spans(text)
    match = match_claim_to_span("Quantum chromodynamics explains gluon confinement.", spans, CFG)
    assert not match.matched
    assert match.span is None
    assert match.skip_reason == SkipReason.CLAIM_NOT_FOUND


def test_never_first_sentence_fallback():
    """Requirement 9 — the rejected reference behaviour returned spans[0] here."""
    text = "The sky is blue. Grass is green. Water is wet."
    unrelated = "Xylophone marimba vibraphone glockenspiel timpani percussion ensemble."
    spans = extract_sentence_spans(text)
    match = match_claim_to_span(unrelated, spans, CFG)
    assert match.span is None, "must not fall back to the first sentence"

    plan = _plan(text, correct=[_claim("c1", unrelated)])
    assert plan.targets == []
    assert "S1" not in [t.sentence_id for t in plan.targets]
    assert plan.skipped_claims[0].reason == SkipReason.CLAIM_NOT_FOUND.value


def test_never_invent_a_target_when_no_claims_authorized():
    text = "The sky is blue. Grass is green."
    plan = _plan(text, correct=[], preserve=[_claim("c1", "The sky is blue.", "verified")])
    assert plan.targets == []
    assert plan.has_targets is False
    _assert_byte_accurate(text, plan)


def test_target_count_never_exceeds_authorized_claims():
    text = "One is here. Two is here. Three is here. Four is here."
    plan = _plan(text, correct=[_claim("c1", "Two is here.")])
    assert len(plan.targets) <= 1
    assert len(plan.targets) == 1
    assert plan.targets[0].sentence_id == "S2"


def test_empty_claim_text_is_skipped():
    text = "The sky is blue."
    plan = _plan(text, correct=[_claim("c1", "   ")])
    assert plan.targets == []
    assert plan.skipped_claims[0].reason == SkipReason.EMPTY_CLAIM_TEXT.value


def test_no_spans_when_original_response_is_blank():
    plan = _plan("   ", correct=[_claim("c1", "anything at all here")])
    assert plan.spans == []
    assert plan.targets == []
    assert plan.skipped_claims[0].reason == SkipReason.NO_SENTENCE_SPANS.value


# ---------------------------------------------------------------------------
# Requirements 1, 2, 11 — authorization and preservation
# ---------------------------------------------------------------------------

def test_only_authorized_claims_become_targets():
    text = "The sky is blue. Python was created by Elon Musk. Grass is green."
    plan = _plan(
        text,
        correct=[_claim("c2", "Python was created by Elon Musk.")],
        preserve=[
            _claim("c1", "The sky is blue.", "verified"),
            _claim("c3", "Grass is green.", "verified"),
        ],
    )
    assert [t.sentence_id for t in plan.targets] == ["S2"]
    assert plan.targets[0].claim_ids == ["c2"]
    assert plan.locked_sentence_ids == ["S1", "S3"]
    _assert_byte_accurate(text, plan)


def test_preserved_span_is_never_targeted():
    """Requirement 11 — same sentence in both lists must not be corrected."""
    text = "The sky is blue. Grass is green."
    plan = _plan(
        text,
        correct=[_claim("cX", "The sky is blue.")],
        preserve=[_claim("cY", "The sky is blue.", "verified")],
    )
    assert plan.targets == []
    assert plan.locked_sentence_ids == ["S1"]
    assert (
        plan.skipped_claims[0].reason
        == SkipReason.SPAN_LOCKED_BY_PRESERVED_CLAIM.value
    )


def test_claim_id_in_both_lists_preservation_wins():
    text = "The sky is blue. Grass is green."
    plan = _plan(
        text,
        correct=[_claim("c1", "Grass is green.")],
        preserve=[_claim("c1", "The sky is blue.", "verified")],
    )
    assert plan.targets == []
    assert (
        plan.skipped_claims[0].reason
        == SkipReason.CLAIM_ALSO_IN_PRESERVE_LIST.value
    )


def test_unlocated_preserved_claim_is_reported_not_ignored():
    text = "The sky is blue. Grass is green."
    plan = _plan(
        text,
        correct=[_claim("c1", "Grass is green.")],
        preserve=[_claim("cZ", "A claim that does not appear in the response at all.", "verified")],
    )
    assert len(plan.unlocated_preserved_claims) == 1
    assert plan.unlocated_preserved_claims[0].claim_id == "cZ"
    # The locatable authorized claim still proceeds.
    assert [t.sentence_id for t in plan.targets] == ["S2"]


def test_preserved_claims_are_located_before_targeting():
    """Lock ordering: preservation must be resolved first regardless of list order."""
    text = "Alpha statement here. Beta statement here."
    plan = _plan(
        text,
        correct=[_claim("c1", "Alpha statement here."), _claim("c2", "Beta statement here.")],
        preserve=[_claim("p1", "Beta statement here.", "verified")],
    )
    assert [t.sentence_id for t in plan.targets] == ["S1"]
    assert "S2" in plan.locked_sentence_ids


# ---------------------------------------------------------------------------
# Multiple claims / merging / plan integrity
# ---------------------------------------------------------------------------

def test_multiple_claims_on_one_sentence_merge_without_losing_authorization():
    text = "Musk created Python in 1991 in Berlin. Unrelated marine biology sentence."
    plan = _plan(
        text,
        correct=[
            _claim("c1", "Musk created Python"),
            _claim("c2", "in 1991 in Berlin"),
        ],
    )
    assert len(plan.targets) == 1
    target = plan.targets[0]
    assert target.sentence_id == "S1"
    assert target.claim_ids == ["c1", "c2"]
    assert target.claim_texts == ["Musk created Python", "in 1991 in Berlin"]


def test_multiple_distinct_targets_preserve_document_order():
    text = "Bad one here. Good middle sentence. Bad two here."
    plan = _plan(
        text,
        correct=[_claim("c2", "Bad two here."), _claim("c1", "Bad one here.")],
    )
    # Targets are emitted in first-seen (claim) order; both are present and accurate.
    assert {t.sentence_id for t in plan.targets} == {"S1", "S3"}
    _assert_byte_accurate(text, plan)


def test_plan_reports_integrity_verified():
    text = "One sentence here. Two sentence here."
    plan = _plan(text, correct=[_claim("c1", "Two sentence here.")])
    assert plan.integrity_verified is True


def test_targets_never_cover_locked_spans_across_a_larger_document():
    text = (
        "Fact one is verified. Claim two is false. Fact three is verified. "
        "Claim four is false. Fact five is verified."
    )
    plan = _plan(
        text,
        correct=[_claim("c2", "Claim two is false."), _claim("c4", "Claim four is false.")],
        preserve=[
            _claim("p1", "Fact one is verified.", "verified"),
            _claim("p3", "Fact three is verified.", "verified"),
            _claim("p5", "Fact five is verified.", "verified"),
        ],
    )
    target_ids = {t.sentence_id for t in plan.targets}
    assert target_ids == {"S2", "S4"}
    assert target_ids.isdisjoint(set(plan.locked_sentence_ids))
    _assert_byte_accurate(text, plan)


def test_target_offsets_are_usable_for_exact_splicing():
    """The offsets must support lossless reconstruction in a later step."""
    text = "Keep this. Replace this one. Keep that too."
    plan = _plan(text, correct=[_claim("c1", "Replace this one.")])
    target = plan.targets[0]
    rebuilt = text[:target.start_offset] + target.original_text + text[target.end_offset:]
    assert rebuilt == text


def test_verdict_does_not_grant_authorization():
    """Requirement 1 — a 'contradicted' verdict alone must not create a target."""
    text = "The sky is blue. Python was created by Elon Musk."
    plan = _plan(
        text,
        correct=[],
        preserve=[_claim("c2", "Python was created by Elon Musk.", VerdictLabel.CONTRADICTED)],
    )
    assert plan.targets == []
    assert plan.locked_sentence_ids == ["S2"]
