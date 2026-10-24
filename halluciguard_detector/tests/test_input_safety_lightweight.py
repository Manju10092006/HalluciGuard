"""Source input safety using the actual filters, without any model imports."""
import pytest
import subprocess
import sys

from agents.verifier_agent.claims.claim_decomposer import ClaimDecomposer
from halluciguard_detector import evidence
from halluciguard_detector.text import complete_claim_units, sentence_spans


def test_input_helpers_do_not_import_the_ml_runtime():
    result = subprocess.run([sys.executable, '-S', '-c',
        "import halluciguard_detector.evidence, halluciguard_detector.nli_input, sys; "
        "assert not {'torch', 'transformers', 'sentence_transformers'} & set(sys.modules)"],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('text', [
    'Note that Paris is the capital of Germany.',
    'Keep in mind that Paris is the capital of Germany.',
    'You can see Mount Everest from space.',
])
def test_assertions_remain_checkable_behind_framing(monkeypatch, text):
    monkeypatch.setattr(evidence, '_ensure_verifier', lambda: True)
    monkeypatch.setattr(evidence, '_decomposer', ClaimDecomposer())
    assert evidence.is_checkable(text)


@pytest.mark.parametrize('text', ['Sure!', 'I am not sure whether this is true.', 'What is the capital of France?'])
def test_conversation_and_questions_still_have_no_factual_exemption_error(monkeypatch, text):
    monkeypatch.setattr(evidence, '_ensure_verifier', lambda: True)
    monkeypatch.setattr(evidence, '_decomposer', ClaimDecomposer())
    assert not evidence.is_checkable(text)


def test_cap_cannot_hide_more_assertions_in_an_already_aligned_sentence():
    draft = 'Aster opened in 2001 and closed in 2002 and moved in 2003 and reopened in 2004 and sold in 2005 and failed in 2006.'
    spans = sentence_spans(draft)
    offsets = (spans[0].start, spans[0].end)
    partial = [(f'Aster event {year}', offsets) for year in range(2001, 2006)]
    completed = complete_claim_units(spans, partial, decomposition_capped=True)
    assert completed[:5] == partial
    assert completed[-1] == (draft, offsets)
    assert '2006' in completed[-1][0]


def test_full_sentence_is_not_duplicated_at_the_cap():
    draft = 'Paris is in France.'
    spans = sentence_spans(draft)
    aligned = [(draft, (spans[0].start, spans[0].end))]
    assert complete_claim_units(spans, aligned, decomposition_capped=True) == aligned


def test_uncovered_source_sentence_is_retained():
    spans = sentence_spans('Paris is in France. Berlin is in Germany.')
    aligned = [(spans[0].text, (spans[0].start, spans[0].end))]
    assert [text for text, _ in complete_claim_units(spans, aligned)] == [span.text for span in spans]
