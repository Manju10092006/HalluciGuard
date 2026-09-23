"""
Tests for calibration, threshold selection, and the (disabled) LLM Judge.
"""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from halluciguard_detector.calibration import (
    Calibrator,
    expected_calibration_error,
    select_best_calibrator,
)
from halluciguard_detector.thresholds import select_thresholds
from halluciguard_detector.llm_judge import LLMJudge, JudgeConfig


def _synthetic(n=400, seed=0):
    """Well-separated raw scores so calibration/thresholds have signal."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, size=n)
    raw = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, 0.15, size=n), 0.01, 0.99)
    return raw.tolist(), y.tolist()


# ----------------------------------------------------------------- calibration
def test_isotonic_does_not_crash_and_returns_probabilities():
    # Regression test for the old predict_proba bug on IsotonicRegression.
    raw, y = _synthetic()
    cal = Calibrator(method="isotonic").fit(raw, y)
    out = cal.calibrate(raw)
    assert len(out) == len(raw)
    assert all(0.0 <= p <= 1.0 for p in out)


@pytest.mark.parametrize("method", ["temperature", "platt", "isotonic", "none"])
def test_all_methods_return_bounded_probs(method):
    raw, y = _synthetic()
    cal = Calibrator(method=method).fit(raw, y)
    out = cal.calibrate(raw)
    assert len(out) == len(raw)
    assert all(0.0 <= p <= 1.0 for p in out)


def test_calibration_improves_or_matches_ece():
    raw, y = _synthetic()
    base_ece = expected_calibration_error(raw, y)
    cal = Calibrator(method="temperature").fit(raw, y)
    cal_ece = expected_calibration_error(cal.calibrate(raw), y)
    # Temperature scaling should not make calibration worse on the fit set.
    assert cal_ece <= base_ece + 1e-6


def test_single_class_downgrades_to_identity():
    raw = [0.3, 0.4, 0.5, 0.6]
    y = [0, 0, 0, 0]
    cal = Calibrator(method="platt").fit(raw, y)
    assert cal.method == "none"  # refused to fit a misleading calibrator
    assert cal.calibrate(raw) == [round(x, 4) for x in raw]


def test_select_best_calibrator_reports_all_methods():
    raw, y = _synthetic()
    best, report = select_best_calibrator(raw, y, dataset_name="DEV")
    for m in ("temperature", "platt", "isotonic"):
        assert m in report
    assert report["selected"] in ("temperature", "platt", "isotonic")
    assert best.fitted


def test_empty_calibrate_returns_empty():
    cal = Calibrator(method="temperature")
    assert cal.calibrate([]) == []


# ------------------------------------------------------------------ thresholds
def test_low_band_respects_miss_target():
    raw, y = _synthetic(n=600)
    cal = Calibrator(method="isotonic").fit(raw, y)
    probs = cal.calibrate(raw)
    sel = select_thresholds(probs, y, low_miss_target=0.10, high_precision_target=0.60)
    assert 0.0 <= sel.t_low < sel.t_high <= 1.0
    # If a LOW band exists, its miss rate must honour the cap.
    if sel.low_coverage > 0:
        assert sel.low_miss_rate <= 0.10 + 1e-9


def test_high_band_respects_precision_target():
    raw, y = _synthetic(n=600)
    cal = Calibrator(method="isotonic").fit(raw, y)
    probs = cal.calibrate(raw)
    sel = select_thresholds(probs, y, low_miss_target=0.10, high_precision_target=0.60)
    if sel.high_coverage > 0 and sel.high_target_met:
        assert sel.high_precision >= 0.60 - 1e-9


def test_thresholds_never_inverted():
    raw, y = _synthetic()
    sel = select_thresholds(raw, y)
    assert sel.t_high > sel.t_low


# ------------------------------------------------------------------- llm judge
def test_judge_disabled_by_default():
    judge = LLMJudge(JudgeConfig(), completion_fn=lambda s, u: "{}")
    assert not judge.available  # enabled=False by default
    assert judge.judge_uncertain_claims("q", "a", [{"claim_id": "c1", "text": "x", "raw_score": 0.5}]) == []


def test_judge_without_client_abstains():
    judge = LLMJudge(JudgeConfig(enabled=True), completion_fn=None)
    assert not judge.available
    assert judge.judge_claim("q", "a", "c1", "x", 0.5) is None


def test_judge_parses_valid_verdict_and_respects_budget():
    calls = {"n": 0}

    def fake(system, user):
        calls["n"] += 1
        return '{"risk": "HIGH_RISK", "confidence": 0.8, "reason": "specific number"}'

    judge = LLMJudge(JudgeConfig(enabled=True, max_calls_per_request=2), completion_fn=fake)
    claims = [{"claim_id": f"c{i}", "text": "t", "raw_score": 0.5} for i in range(5)]
    verdicts = judge.judge_uncertain_claims("q", "a", claims)
    assert calls["n"] == 2  # budget enforced
    assert all(v.risk in ("LOW_RISK", "UNCERTAIN", "HIGH_RISK") for v in verdicts)


def test_judge_rejects_factual_truth_verdict():
    judge = LLMJudge(
        JudgeConfig(enabled=True),
        completion_fn=lambda s, u: '{"risk": "HIGH_RISK", "confidence": 0.9, "reason": "this is VERIFIED as FALSE"}',
    )
    # Smuggled factual terms -> abstain.
    assert judge.judge_claim("q", "a", "c1", "x", 0.5) is None


def test_judge_rejects_illegal_risk_label():
    judge = LLMJudge(
        JudgeConfig(enabled=True),
        completion_fn=lambda s, u: '{"risk": "HALLUCINATED", "confidence": 0.9, "reason": "x"}',
    )
    assert judge.judge_claim("q", "a", "c1", "x", 0.5) is None


def test_judge_handles_garbage_output():
    judge = LLMJudge(JudgeConfig(enabled=True), completion_fn=lambda s, u: "not json at all")
    assert judge.judge_claim("q", "a", "c1", "x", 0.5) is None
