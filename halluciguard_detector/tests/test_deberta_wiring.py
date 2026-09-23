"""
Tests that detector.py actually invokes the M2 DeBERTa model (not a 0.5 stub),
and fails closed when the model is unavailable.

These use a mocked classifier interface so they run without the real (large)
checkpoint. The PRODUCTION code path still constructs and calls the real
M2DebertaClassifier — see test_production_uses_real_classifier_class.
"""
import sys, os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from halluciguard_detector.detector import StandaloneDetector
from halluciguard_detector.models.deberta import M2DebertaClassifier, ModelUnavailableError
from halluciguard_detector.schemas import RiskLevel, RoutingDecision, VerificationHint, StatusEnum
from halluciguard_detector.calibration import Calibrator


class FakeClassifier:
    """Records the claims it was asked to score and returns fixed scores."""
    def __init__(self, scores=None, fail=False):
        self._scores = scores
        self._fail = fail
        self.calls = []

    def predict_claim_risk(self, user_query, draft_answer, claim_texts):
        self.calls.append(list(claim_texts))
        if self._fail:
            raise ModelUnavailableError("simulated missing checkpoint")
        if self._scores is not None:
            return list(self._scores[: len(claim_texts)])
        return [0.8] * len(claim_texts)  # non-0.5 to prove real invocation


def _supplied(n):
    return [{"claim_id": f"claim_{i:03d}", "text": f"Claim number {i} about a topic."} for i in range(1, n + 1)]


# A. Real model invocation is used instead of the 0.5 stub.
def test_model_is_actually_invoked():
    clf = FakeClassifier(scores=[0.91])
    det = StandaloneDetector(classifier=clf)
    resp = det.detect("Who created Java?", "Java was created by James Gosling.",
                       supplied_claims=_supplied(1))
    assert clf.calls, "classifier.predict_claim_risk was never called"
    assert resp.claims[0].raw_score == 0.91  # real score, not 0.5


# B. One claim -> one prediction.
def test_one_claim_one_prediction():
    clf = FakeClassifier(scores=[0.7])
    det = StandaloneDetector(classifier=clf)
    resp = det.detect("q", "a", supplied_claims=_supplied(1))
    assert len(resp.claims) == 1
    assert len(clf.calls[0]) == 1


# C. Multiple claims -> matching claim IDs preserved in order.
def test_multiple_claims_preserve_ids():
    clf = FakeClassifier(scores=[0.2, 0.5, 0.9])
    det = StandaloneDetector(classifier=clf)
    resp = det.detect("q", "a", supplied_claims=_supplied(3))
    assert [c.claim_id for c in resp.claims] == ["claim_001", "claim_002", "claim_003"]


# D. Scores are in [0, 1].
def test_scores_bounded():
    clf = FakeClassifier(scores=[0.05, 0.99, 0.42])
    det = StandaloneDetector(classifier=clf)
    resp = det.detect("q", "a", supplied_claims=_supplied(3))
    assert all(0.0 <= c.raw_score <= 1.0 for c in resp.claims)


# E. Model unavailable -> UNKNOWN / DEEP / VERIFY.
def test_model_unavailable_fails_closed():
    clf = FakeClassifier(fail=True)
    det = StandaloneDetector(classifier=clf)
    resp = det.detect("q", "a", supplied_claims=_supplied(2))
    assert resp.status == StatusEnum.DEGRADED
    assert "MODEL_UNAVAILABLE" in resp.degraded_reasons
    assert all(c.risk_level == RiskLevel.UNKNOWN for c in resp.claims)
    assert all(c.verification_hint == VerificationHint.DEEP for c in resp.claims)
    assert resp.routing == RoutingDecision.VERIFY


# F. Calibration unavailable -> no fabricated calibrated probability.
def test_no_calibration_means_null_calibrated_prob():
    clf = FakeClassifier(scores=[0.8, 0.3])
    det = StandaloneDetector(classifier=clf)  # no calibrator
    resp = det.detect("q", "a", supplied_claims=_supplied(2))
    assert all(c.calibrated_probability is None for c in resp.claims)
    assert "NO_CALIBRATION" in resp.degraded_reasons
    # Uncalibrated real scores must not be promoted to a trusted band.
    assert all(c.risk_level == RiskLevel.UNKNOWN for c in resp.claims)


# F2. With a fitted calibrator, calibrated probabilities ARE produced and banded.
def test_with_calibration_produces_bands():
    import numpy as np
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, size=300)
    raw = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, 0.15, size=300), 0.01, 0.99)
    cal = Calibrator(method="isotonic").fit(raw.tolist(), y.tolist(), dataset_name="DEV")

    clf = FakeClassifier(scores=[0.95, 0.05])
    det = StandaloneDetector(classifier=clf, calibrator=cal, t_low=0.25, t_high=0.65)
    resp = det.detect("q", "a", supplied_claims=_supplied(2))
    assert all(c.calibrated_probability is not None for c in resp.claims)
    # High raw -> not LOW; low raw -> not HIGH.
    assert resp.claims[0].risk_level != RiskLevel.LOW_RISK
    assert resp.claims[1].risk_level != RiskLevel.HIGH_RISK
    assert resp.calibration.method == "isotonic"


# G. Claim IDs unchanged end-to-end (custom IDs).
def test_custom_claim_ids_unchanged():
    clf = FakeClassifier(scores=[0.4, 0.6])
    det = StandaloneDetector(classifier=clf)
    supplied = [{"claim_id": "x_777", "text": "t1"}, {"claim_id": "y_888", "text": "t2"}]
    resp = det.detect("q", "a", supplied_claims=supplied)
    assert [c.claim_id for c in resp.claims] == ["x_777", "y_888"]


# I. ALWAYS_VERIFY: routing is always VERIFY regardless of risk.
def test_always_verify():
    for scores in ([0.99], [0.01], [0.5]):
        clf = FakeClassifier(scores=scores)
        det = StandaloneDetector(classifier=clf)
        resp = det.detect("q", "a", supplied_claims=_supplied(1))
        assert resp.routing == RoutingDecision.VERIFY


# J. LLM Judge remains disabled by default.
def test_judge_disabled_by_default():
    clf = FakeClassifier(scores=[0.5])
    det = StandaloneDetector(classifier=clf)
    resp = det.detect("q", "a", supplied_claims=_supplied(1))
    assert resp.judge.enabled is False
    assert resp.judge.used is False


# Production wiring: the default detector constructs the REAL classifier class.
def test_production_uses_real_classifier_class():
    det = StandaloneDetector()
    assert isinstance(det._classifier, M2DebertaClassifier)


# H (structural): inference receives only query/answer/claim text, never labels.
def test_inference_receives_no_labels():
    clf = FakeClassifier(scores=[0.5, 0.5])
    det = StandaloneDetector(classifier=clf)
    det.detect("q", "a", supplied_claims=_supplied(2))
    # The only thing passed to the model is claim text; assert no label-like keys leak.
    assert clf.calls == [["Claim number 1 about a topic.", "Claim number 2 about a topic."]]
