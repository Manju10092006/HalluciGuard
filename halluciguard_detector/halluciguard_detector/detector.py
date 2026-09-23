"""
halluciguard_detector / detector.py
──────────────────────────────────
Main standalone Detector class following the Master Specification.

Routing in v1 is ALWAYS VERIFY.
Verification hints:
    STANDARD  -> for LOW_RISK claims
    DEEP      -> for UNCERTAIN / HIGH_RISK / UNKNOWN claims
"""

from __future__ import annotations
import logging
import os
import time
from typing import List, Optional, Dict, Any, Tuple

from .schemas import (
    CalibrationInfo,
    ClaimResult,
    DetectorRequest,
    DetectorResponse,
    ExtractionMode,
    JudgeInfo,
    LatencyInfo,
    ModelInfo,
    OverallRisk,
    RiskLevel,
    RoutingDecision,
    StatusEnum,
    ThresholdsInfo,
    VerificationHint,
    ExtractedClaim,
)
from .claim_extraction import ClaimExtractor
from .models.deberta import M2DebertaClassifier, ModelUnavailableError

logger = logging.getLogger(__name__)


class StandaloneDetector:
    """
    Standalone Hallucination Detector (Detector V2).
    Completely independent of agents/detector_agent/.

    Scoring path:
        claim -> M2 DeBERTa-v3-base -> raw risk score
              -> calibration (only if a fitted calibrator is supplied)
              -> LOW/UNCERTAIN/HIGH routing
    If the model cannot produce a real score, the detector fails closed
    (UNKNOWN / DEEP / VERIFY) rather than fabricating a score.
    """

    def __init__(
        self,
        model_name: str = "deberta-v3-base",
        checkpoint_path: Optional[str] = None,
        t_low: float = 0.25,
        t_high: float = 0.65,
        always_verify: bool = True,
        classifier: Optional[M2DebertaClassifier] = None,
        calibrator: Optional[object] = None,
    ) -> None:
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path or "halluciguard_detector/data/checkpoints/deberta-v3-base"
        self.t_low = t_low
        self.t_high = t_high
        self.always_verify = always_verify
        self.extractor = ClaimExtractor(max_claims=100)
        # Real M2 model. Injectable for testing; production uses the real class.
        self._classifier = classifier or M2DebertaClassifier(checkpoint_path=self.checkpoint_path)
        # Optional fitted Calibrator (with a .calibrate([...]) method). None => uncalibrated.
        self._calibrator = calibrator
        self._has_calibration = calibrator is not None

    def detect(
        self,
        user_query: str,
        draft_answer: str,
        request_id: str = "HG-0001",
        supplied_claims: Optional[List[Dict[str, Any]]] = None,
    ) -> DetectorResponse:
        """
        Main entry point for hallucination detection.
        """
        t_start = time.time()
        degraded_reasons: List[str] = []
        status = StatusEnum.COMPLETED

        # Fail closed check 1: Empty or whitespace answer
        if not draft_answer or not draft_answer.strip():
            logger.warning("[Detector] Empty draft_answer received -> FAILED EMPTY_ANSWER.")
            return self._fail_output(
                request_id=request_id,
                reason="EMPTY_ANSWER",
                user_query=user_query,
                t_start=t_start,
            )

        # Stage 1: Claim Extraction
        t_ext_start = time.time()
        extracted_claims, ext_mode, ext_reasons = self.extractor.extract(
            user_query=user_query,
            draft_answer=draft_answer,
            supplied_claims=supplied_claims,
        )
        t_ext_ms = (time.time() - t_ext_start) * 1000.0
        degraded_reasons.extend(ext_reasons)

        if not extracted_claims:
            logger.warning("[Detector] Zero claims extracted -> FAILED NO_CLAIMS.")
            return self._fail_output(
                request_id=request_id,
                reason="NO_CLAIMS",
                user_query=user_query,
                t_start=t_start,
            )

        # Check claim cap overflow (spec: cap at 40 claims per answer; overflow claims get risk_level="UNKNOWN" with flag CLAIM_CAP)
        if len(extracted_claims) > 40:
            if "CLAIM_CAP" not in degraded_reasons:
                degraded_reasons.append("CLAIM_CAP")
            status = StatusEnum.DEGRADED

        # Stage 2: Scoring
        t_score_start = time.time()
        claim_results, score_status, score_reasons = self._score_claims(
            user_query=user_query,
            draft_answer=draft_answer,
            claims=extracted_claims,
        )
        t_score_ms = (time.time() - t_score_start) * 1000.0
        degraded_reasons.extend(score_reasons)

        if score_status == StatusEnum.DEGRADED and status != StatusEnum.FAILED:
            status = StatusEnum.DEGRADED
        elif score_status == StatusEnum.FAILED:
            status = StatusEnum.FAILED

        # Compute overall risk (max aggregation).
        # When calibrated, aggregate over calibrated probabilities and map to a
        # band. When uncalibrated OR any claim is UNKNOWN, the whole answer is
        # UNKNOWN (we cannot trust a band) and routing stays DEEP/VERIFY.
        if claim_results:
            n_high = sum(1 for c in claim_results if c.risk_level == RiskLevel.HIGH_RISK)
            n_unc = sum(1 for c in claim_results if c.risk_level == RiskLevel.UNCERTAIN)
            has_unknown = any(c.risk_level == RiskLevel.UNKNOWN for c in claim_results)
            calibrated_vals = [
                c.calibrated_probability for c in claim_results if c.calibrated_probability is not None
            ]
            if self._has_calibration and calibrated_vals and not has_unknown:
                max_score = max(calibrated_vals)
                overall_risk_level = self._compute_risk_level(max_score)
            else:
                max_score = max(c.raw_score for c in claim_results)
                overall_risk_level = RiskLevel.UNKNOWN
        else:
            max_score = 0.5
            n_high = 0
            n_unc = 0
            overall_risk_level = RiskLevel.UNKNOWN

        overall = OverallRisk(
            risk_score=round(max_score, 4),
            risk_level=overall_risk_level,
            aggregation="max",
            n_claims=len(claim_results),
            n_high=n_high,
            n_uncertain=n_unc,
        )

        t_total_ms = (time.time() - t_start) * 1000.0

        return DetectorResponse(
            request_id=request_id,
            status=status,
            degraded_reasons=degraded_reasons,
            extraction_mode=ext_mode,
            claims=claim_results,
            overall=overall,
            routing=RoutingDecision.VERIFY,
            calibration=self._calibration_info(),
            thresholds=ThresholdsInfo(
                low=self.t_low,
                high=self.t_high,
                provisional=True,
            ),
            judge=JudgeInfo(enabled=False, used=False),
            model=ModelInfo(name=self.model_name, checkpoint_id=self.checkpoint_path),
            latency_ms=LatencyInfo(
                extraction_ms=round(t_ext_ms, 2),
                scoring_ms=round(t_score_ms, 2),
                total_ms=round(t_total_ms, 2),
            ),
        )

    def _score_claims(
        self,
        user_query: str,
        draft_answer: str,
        claims: List[ExtractedClaim],
    ) -> Tuple[List[ClaimResult], StatusEnum, List[str]]:
        """Score claims with the real M2 DeBERTa model.

        - Claims within the cap are scored by DeBERTa (one prediction each).
        - Overflow claims (index >= 40) are marked UNKNOWN/DEEP with CLAIM_CAP.
        - If the model is unavailable, ALL scored claims fail closed to
          UNKNOWN/DEEP and the status becomes DEGRADED. Scores are never
          fabricated.
        - Calibration is applied only when a fitted calibrator was supplied;
          otherwise calibrated_probability stays None.
        """
        results: List[ClaimResult] = []
        reasons: List[str] = []
        status = StatusEnum.COMPLETED

        if not self._has_calibration:
            reasons.append("NO_CALIBRATION")
            status = StatusEnum.DEGRADED

        scored_claims = [c for i, c in enumerate(claims) if i < 40]
        overflow_claims = [c for i, c in enumerate(claims) if i >= 40]

        # --- Real model inference for in-cap claims (batched by the model class).
        raw_scores: Optional[List[float]] = None
        model_available = True
        if scored_claims:
            try:
                raw_scores = self._classifier.predict_claim_risk(
                    user_query=user_query,
                    draft_answer=draft_answer,
                    claim_texts=[c.text for c in scored_claims],
                )
                if raw_scores is None or len(raw_scores) != len(scored_claims):
                    raise ModelUnavailableError(
                        "model returned wrong number of scores "
                        f"({0 if raw_scores is None else len(raw_scores)} for {len(scored_claims)} claims)"
                    )
            except ModelUnavailableError as exc:
                logger.warning("[Detector] Model unavailable -> fail closed: %s", exc)
                model_available = False
                if "MODEL_UNAVAILABLE" not in reasons:
                    reasons.append("MODEL_UNAVAILABLE")
                status = StatusEnum.DEGRADED

        # --- Optional calibration (dev-fitted). Only when model produced scores.
        calibrated: Optional[List[float]] = None
        if model_available and raw_scores is not None and self._has_calibration:
            try:
                calibrated = self._calibrator.calibrate(raw_scores)  # type: ignore[attr-defined]
            except Exception as exc:  # calibrator misbehaves -> treat as uncalibrated
                logger.warning("[Detector] Calibration failed -> uncalibrated: %s", exc)
                calibrated = None
                if "CALIBRATION_FAILED" not in reasons:
                    reasons.append("CALIBRATION_FAILED")

        # --- Build per-claim results.
        for idx, claim in enumerate(scored_claims):
            if not model_available or raw_scores is None:
                # Fail closed: no real score -> UNKNOWN/DEEP. raw_score is a
                # sentinel and MUST NOT be interpreted as a calibrated risk.
                results.append(
                    ClaimResult(
                        claim_id=claim.claim_id,
                        text=claim.text,
                        source_sentence_idx=claim.source_sentence_idx,
                        raw_score=0.5,
                        calibrated_probability=None,
                        risk_level=RiskLevel.UNKNOWN,
                        verification_hint=VerificationHint.DEEP,
                        flags=["MODEL_UNAVAILABLE"],
                    )
                )
                continue

            raw = float(raw_scores[idx])
            cal_prob = float(calibrated[idx]) if calibrated is not None else None

            if self._has_calibration and cal_prob is not None:
                r_level = self._compute_risk_level(cal_prob)
            else:
                # Uncalibrated: we have a real raw score but cannot map it to a
                # trustworthy probability band. Stay UNKNOWN per the contract.
                r_level = RiskLevel.UNKNOWN

            v_hint = (
                VerificationHint.STANDARD
                if r_level == RiskLevel.LOW_RISK
                else VerificationHint.DEEP
            )
            results.append(
                ClaimResult(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    source_sentence_idx=claim.source_sentence_idx,
                    raw_score=round(raw, 4),
                    calibrated_probability=(round(cal_prob, 4) if cal_prob is not None else None),
                    risk_level=r_level,
                    verification_hint=v_hint,
                    flags=[],
                )
            )

        # --- Overflow claims (beyond the cap).
        for claim in overflow_claims:
            results.append(
                ClaimResult(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    source_sentence_idx=claim.source_sentence_idx,
                    raw_score=0.5,
                    calibrated_probability=None,
                    risk_level=RiskLevel.UNKNOWN,
                    verification_hint=VerificationHint.DEEP,
                    flags=["CLAIM_CAP"],
                )
            )

        return results, status, reasons

    def _calibration_info(self) -> CalibrationInfo:
        """Report the actual calibrator state, not a hardcoded 'none'."""
        if not self._has_calibration or self._calibrator is None:
            return CalibrationInfo(method="none", fitted_on="none", version="v1")
        describe = getattr(self._calibrator, "describe", None)
        if callable(describe):
            d = describe()
            return CalibrationInfo(
                method=d.get("method", "unknown"),
                fitted_on=d.get("fitted_on", "unknown"),
                version=d.get("version", "v1"),
            )
        return CalibrationInfo(
            method=getattr(self._calibrator, "method", "unknown"),
            fitted_on=getattr(self._calibrator, "fitted_on", "unknown"),
            version=getattr(self._calibrator, "version", "v1"),
        )

    def _compute_risk_level(self, score: float) -> RiskLevel:
        """Map score to RiskLevel using provisional thresholds."""
        if score <= self.t_low:
            return RiskLevel.LOW_RISK
        if score >= self.t_high:
            return RiskLevel.HIGH_RISK
        return RiskLevel.UNCERTAIN

    def _fail_output(
        self,
        request_id: str,
        reason: str,
        user_query: str,
        t_start: float,
    ) -> DetectorResponse:
        """Fail-closed output generation."""
        t_total_ms = (time.time() - t_start) * 1000.0
        return DetectorResponse(
            request_id=request_id,
            status=StatusEnum.FAILED,
            degraded_reasons=[reason],
            extraction_mode=ExtractionMode.SENTENCE_FALLBACK,
            claims=[],
            overall=OverallRisk(
                risk_score=0.5,
                risk_level=RiskLevel.UNKNOWN,
                aggregation="max",
                n_claims=0,
                n_high=0,
                n_uncertain=0,
            ),
            routing=RoutingDecision.VERIFY,
            latency_ms=LatencyInfo(total_ms=round(t_total_ms, 2)),
        )
