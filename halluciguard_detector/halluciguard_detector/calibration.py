"""
calibration.py
──────────────
Probability calibration for the standalone HalluciGuard Detector.

Implements three calibrators:
    - temperature : single-scalar temperature scaling on logits (NLL-fit)
    - platt       : logistic regression on the raw score
    - isotonic    : monotonic isotonic regression

Plus:
    - prior-shift logit adjustment (train prevalence -> production prevalence)
    - select_best_calibrator(): fits all methods on DEV ONLY and picks the
      lowest-ECE method. Never fit or select on the frozen test split.

Contract:
    All calibrators map a raw risk score in [0, 1] to a calibrated
    probability in [0, 1]. Fitting requires binary labels (0 = low risk /
    supported, 1 = high risk / contradicted-or-unsupported).
"""

from __future__ import annotations
import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

_EPS = 1e-7
VALID_METHODS = ("temperature", "platt", "isotonic", "none")


def _to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Calibrator:
    """
    Fits and applies probability calibration on raw model risk scores.

    Parameters
    ----------
    method : one of {"temperature", "platt", "isotonic", "none"}
    version : free-form version tag recorded in the detector response.
    """

    def __init__(self, method: str = "temperature", version: str = "cal-001") -> None:
        if method not in VALID_METHODS:
            raise ValueError(f"Unknown calibration method '{method}'. Valid: {VALID_METHODS}")
        self.method = method
        self.version = version
        self.fitted = False
        self.fitted_on = "none"
        self._model = None            # sklearn estimator for platt / isotonic
        self._temperature: float = 1.0  # scalar for temperature scaling

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        raw_scores: Sequence[float],
        labels: Sequence[int],
        dataset_name: str = "P0-val",
    ) -> "Calibrator":
        raw = np.asarray(raw_scores, dtype=float).ravel()
        y = np.asarray(labels, dtype=int).ravel()
        if raw.shape[0] != y.shape[0]:
            raise ValueError("raw_scores and labels must have equal length")
        if raw.shape[0] == 0:
            raise ValueError("cannot fit calibrator on empty data")
        self.fitted_on = dataset_name

        if self.method == "none":
            self.fitted = True
            return self

        if len(np.unique(y)) < 2:
            # Degenerate: only one class present on dev. Refuse to fit a
            # misleading calibrator; fall back to identity and warn.
            logger.warning(
                "[Calibrator] Only one label class present in '%s' -> "
                "identity calibration (method downgraded to none).",
                dataset_name,
            )
            self.method = "none"
            self.fitted = True
            return self

        if self.method == "platt":
            self._model = LogisticRegression()
            self._model.fit(raw.reshape(-1, 1), y)
        elif self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._model.fit(raw, y)
        elif self.method == "temperature":
            self._temperature = self._fit_temperature(raw, y)

        self.fitted = True
        return self

    def _fit_temperature(self, raw: np.ndarray, y: np.ndarray) -> float:
        """Grid + local search for the temperature minimizing NLL on logits."""
        logits = _to_logit(raw)

        def nll(temp: float) -> float:
            p = np.clip(_sigmoid(logits / temp), _EPS, 1.0 - _EPS)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p)))

        grid = np.linspace(0.05, 10.0, 200)
        best_t = float(min(grid, key=nll))
        # local refinement around the grid winner
        lo, hi = max(0.01, best_t - 0.1), best_t + 0.1
        fine = np.linspace(lo, hi, 100)
        best_t = float(min(fine, key=nll))
        return best_t

    # -------------------------------------------------------------- calibrate
    def calibrate(self, raw_scores: Sequence[float]) -> List[float]:
        raw = np.asarray(raw_scores, dtype=float).ravel()
        if raw.size == 0:
            return []
        if not self.fitted or self.method == "none":
            return [float(round(s, 4)) for s in raw]

        if self.method == "platt":
            probs = self._model.predict_proba(raw.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            # IsotonicRegression exposes predict(), NOT predict_proba().
            probs = self._model.predict(raw)
        elif self.method == "temperature":
            probs = _sigmoid(_to_logit(raw) / self._temperature)
        else:
            probs = raw

        return [float(round(float(p), 4)) for p in probs]

    # ------------------------------------------------------------ prior shift
    @staticmethod
    def adjust_prior_shift(
        probs: Sequence[float], pi_train: float = 0.5, pi_prod: float = 0.2
    ) -> List[float]:
        """Logit adjustment for a train/production base-rate mismatch."""
        adjusted = []
        offset = np.log(pi_prod / (1 - pi_prod)) - np.log(pi_train / (1 - pi_train))
        for p in probs:
            p_c = np.clip(p, _EPS, 1 - _EPS)
            logit = np.log(p_c / (1 - p_c))
            adjusted.append(float(round(float(_sigmoid(logit + offset)), 4)))
        return adjusted

    def describe(self) -> dict:
        info = {"method": self.method, "version": self.version, "fitted_on": self.fitted_on}
        if self.method == "temperature":
            info["temperature"] = round(self._temperature, 4)
        return info


def expected_calibration_error(
    probs: Sequence[float], labels: Sequence[int], n_bins: int = 10
) -> float:
    """Standard equal-width ECE over [0, 1]."""
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(labels, dtype=int).ravel()
    if p.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return float(ece / len(y))


def _cv_ece(
    method: str,
    raw: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    n_bins: int,
    seed: int = 0,
) -> float:
    """
    Cross-validated ECE for one calibration method.

    Selection MUST NOT use in-sample ECE: isotonic (and to a lesser extent
    platt) can fit the fitting data almost perfectly, reporting a near-zero
    ECE that does not survive on unseen claims. We therefore fit on k-1 folds
    and score the held-out fold, averaging the held-out ECE (weighted by fold
    size). Falls back to in-sample only when there is too little data to fold.
    """
    n = raw.shape[0]
    # Need at least 2 samples per fold and both classes to fit a real calibrator.
    if n < n_splits * 2 or len(np.unique(y)) < 2:
        cal = Calibrator(method=method).fit(raw, y)
        return expected_calibration_error(cal.calibrate(list(raw)), y, n_bins=n_bins)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, n_splits)

    total_err = 0.0
    total_cnt = 0
    for i in range(n_splits):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        if len(np.unique(y[train_idx])) < 2 or test_idx.size == 0:
            continue
        cal = Calibrator(method=method).fit(raw[train_idx], y[train_idx])
        preds = cal.calibrate(list(raw[test_idx]))
        fold_ece = expected_calibration_error(preds, y[test_idx], n_bins=n_bins)
        total_err += fold_ece * test_idx.size
        total_cnt += test_idx.size

    if total_cnt == 0:  # every fold degenerate -> fall back to in-sample
        cal = Calibrator(method=method).fit(raw, y)
        return expected_calibration_error(cal.calibrate(list(raw)), y, n_bins=n_bins)
    return total_err / total_cnt


def select_best_calibrator(
    raw_scores: Sequence[float],
    labels: Sequence[int],
    dataset_name: str = "P4-dev",
    methods: Sequence[str] = ("temperature", "platt", "isotonic"),
    n_bins: int = 10,
    n_splits: int = 5,
) -> Tuple[Calibrator, dict]:
    """
    Select the best calibration method on DEV data by CROSS-VALIDATED ECE,
    then refit the winner on all of dev. MUST be called on dev/validation
    only, never on the frozen test split.

    Selection uses held-out (k-fold) ECE rather than in-sample ECE, so a
    method that merely memorises the dev split (e.g. isotonic) cannot win on
    an artificially perfect in-sample score. The final refit-on-all-dev ECE
    is reported separately as `selected_insample_ece` and should be read as
    optimistic, not as the expected test ECE.

    Returns
    -------
    (best_calibrator, report) where report maps each method -> its CV ECE.
    """
    raw = np.asarray(raw_scores, dtype=float).ravel()
    y = np.asarray(labels, dtype=int).ravel()
    report: dict = {}
    best_method: Optional[str] = None
    best_cv_ece = float("inf")

    for m in methods:
        cv_ece = _cv_ece(m, raw, y, n_splits=n_splits, n_bins=n_bins)
        report[m] = round(cv_ece, 4)
        if cv_ece < best_cv_ece:
            best_cv_ece = cv_ece
            best_method = m

    if best_method is None:  # pragma: no cover - methods is never empty in practice
        best_method = "none"

    # Refit the winner on ALL dev data for deployment.
    best = Calibrator(method=best_method).fit(raw, y, dataset_name=dataset_name)
    insample = expected_calibration_error(best.calibrate(list(raw)), y, n_bins=n_bins)

    report["selected"] = best.method
    report["selected_cv_ece"] = round(best_cv_ece, 4)
    report["selected_insample_ece"] = round(insample, 4)
    report["selection_metric"] = f"{n_splits}-fold CV ECE"
    logger.info(
        "[Calibrator] Selected '%s' on %s (CV ECE=%.4f, in-sample ECE=%.4f)",
        best.method, dataset_name, best_cv_ece, insample,
    )
    return best, report
