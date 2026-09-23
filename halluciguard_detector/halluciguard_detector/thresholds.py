"""
thresholds.py
─────────────
Dev-only selection of the LOW / UNCERTAIN / HIGH routing thresholds for the
standalone HalluciGuard Detector.

Design (per handoff spec §21-23):
    - Calibration and threshold selection are SEPARATE steps.
    - Thresholds are chosen on calibrated probabilities using DEV data only,
      then frozen and applied unchanged to the frozen test split.

Bands (on calibrated probability p):
    p <= t_low               -> LOW_RISK   (verification hint STANDARD)
    t_low < p < t_high       -> UNCERTAIN  (DEEP)
    p >= t_high              -> HIGH_RISK  (DEEP)

Selection targets:
    LOW band  : among claims called LOW_RISK, miss rate (fraction actually
                positive) <= low_miss_target, while maximising LOW coverage.
    HIGH band : among claims called HIGH_RISK, precision >= high_precision_target,
                while maximising HIGH coverage.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ThresholdSelection:
    t_low: float
    t_high: float
    low_miss_rate: float
    low_coverage: float
    high_precision: float
    high_coverage: float
    low_target_met: bool
    high_target_met: bool
    provisional: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _low_band_stats(probs: np.ndarray, labels: np.ndarray, t_low: float):
    """Miss rate and coverage for claims classified LOW_RISK (p <= t_low)."""
    mask = probs <= t_low
    n = int(mask.sum())
    coverage = n / len(probs) if len(probs) else 0.0
    miss = float(labels[mask].mean()) if n > 0 else 0.0  # positives wrongly called low
    return miss, coverage, n


def _high_band_stats(probs: np.ndarray, labels: np.ndarray, t_high: float):
    """Precision and coverage for claims classified HIGH_RISK (p >= t_high)."""
    mask = probs >= t_high
    n = int(mask.sum())
    coverage = n / len(probs) if len(probs) else 0.0
    precision = float(labels[mask].mean()) if n > 0 else 0.0  # positives among flagged
    return precision, coverage, n


def select_thresholds(
    probs: Sequence[float],
    labels: Sequence[int],
    low_miss_target: float = 0.10,
    low_coverage_min: float = 0.20,
    high_precision_target: float = 0.60,
    grid_step: float = 0.01,
) -> ThresholdSelection:
    """
    Choose (t_low, t_high) on DEV calibrated probabilities.

    t_low  : the LARGEST threshold whose LOW-band miss rate <= low_miss_target
             (maximises coverage while respecting the safety cap).
    t_high : the SMALLEST threshold whose HIGH-band precision >= high_precision_target
             (maximises coverage while respecting the precision floor).

    If a target cannot be met anywhere on the grid, the selection is marked
    provisional and the safest available threshold is returned.
    """
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(labels, dtype=int).ravel()
    if p.size == 0:
        raise ValueError("cannot select thresholds on empty data")

    grid = np.round(np.arange(0.0, 1.0 + grid_step, grid_step), 4)

    # ---- t_low: largest t with miss <= target (and coverage >= min if possible)
    best_t_low = 0.0
    best_low_cov = -1.0
    low_target_met = False
    for t in grid:
        miss, cov, n = _low_band_stats(p, y, t)
        if n == 0:
            continue
        if miss <= low_miss_target:
            low_target_met = True
            # prefer higher coverage; ties -> higher threshold (already ascending)
            if cov >= best_low_cov:
                best_low_cov = cov
                best_t_low = float(t)
    if not low_target_met:
        # Nothing meets the miss cap; be conservative -> tiny low band.
        best_t_low = 0.0

    # ---- t_high: smallest t with precision >= target
    best_t_high = 1.0
    best_high_cov = -1.0
    high_target_met = False
    for t in grid:
        prec, cov, n = _high_band_stats(p, y, t)
        if n == 0:
            continue
        if prec >= high_precision_target:
            high_target_met = True
            if cov > best_high_cov:
                best_high_cov = cov
                best_t_high = float(t)
    if not high_target_met:
        best_t_high = 1.0  # nothing precise enough -> effectively no HIGH band

    # Guard against inverted band.
    if best_t_high <= best_t_low:
        best_t_high = min(1.0, best_t_low + grid_step)

    low_miss, low_cov, _ = _low_band_stats(p, y, best_t_low)
    high_prec, high_cov, _ = _high_band_stats(p, y, best_t_high)

    provisional = not (low_target_met and high_target_met)
    if provisional:
        logger.warning(
            "[thresholds] Targets not fully met (low_met=%s, high_met=%s) -> provisional.",
            low_target_met, high_target_met,
        )

    return ThresholdSelection(
        t_low=round(best_t_low, 4),
        t_high=round(best_t_high, 4),
        low_miss_rate=round(low_miss, 4),
        low_coverage=round(low_cov, 4),
        high_precision=round(high_prec, 4),
        high_coverage=round(high_cov, 4),
        low_target_met=bool(low_target_met and low_cov >= low_coverage_min),
        high_target_met=bool(high_target_met),
        provisional=provisional,
    )
