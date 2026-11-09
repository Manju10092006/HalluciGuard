from types import SimpleNamespace

import pytest
import torch

from halluciguard_detector.detector import Detector, LABELS


def fake_detector(logits):
    detector = Detector.__new__(Detector)
    detector.device = torch.device('cpu')
    detector.labels = LABELS
    detector.temperature = 1.0
    detector.threshold = 0.62
    detector.max_length = 256
    detector.version = 'unit-test'
    calls = []

    class Tokens(dict):
        def to(self, device):
            return self

    def tokenize(contexts, claims, **kwargs):
        calls.append((len(claims), kwargs['truncation']))
        return Tokens(input_ids=torch.ones((len(claims), 4), dtype=torch.long))

    detector.tokenizer = tokenize
    detector.model = lambda input_ids: SimpleNamespace(logits=torch.tensor(logits).repeat(len(input_ids), 1))
    return detector, calls


def test_entity_differences_cannot_overwrite_calibrated_probabilities():
    detector, _ = fake_detector([4.0, 0.0, 0.0])
    result = detector.detect('Java was designed by James Gosling in Canada.', ['Java was designed by James Gosling at Sun Microsystems.'])
    expected = torch.softmax(torch.tensor([4.0, 0.0, 0.0]), -1)
    assert result.sentences[0].probabilities[LABELS[0]] == pytest.approx(float(expected[0]))
    assert result.sentences[0].probabilities[LABELS[1]] != 0.90
    assert result.warnings
    assert result.requires_verification is True
    assert result.label == 'REVIEW_REQUIRED'


def test_long_answers_use_bounded_batches_and_preserve_claims():
    detector, calls = fake_detector([0.0, 4.0, 0.0])
    result = detector.detect(' '.join(f'Claim {i} is factual.' for i in range(19)), ['Claim evidence.'])
    assert len(result.sentences) == 19
    assert calls == [(8, 'only_first'), (8, 'only_first'), (3, 'only_first')]


def test_nonfinite_predictions_fail_closed():
    detector, _ = fake_detector([float('nan'), 0.0, 0.0])
    with pytest.raises(ValueError, match='non-finite'):
        detector.detect('A factual claim.', ['Evidence for a claim.'])
