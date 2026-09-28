"""
halluciguard_judge / detector.py
─────────────────────────────────
Main orchestrator: JudgeDetector

Pipeline (exactly as specified):
    LLM Response
         |
         v
    Claim Extractor      (claim_extractor.py)
         |
    [Claim, ...]
         |
         v
    DeBERTa Classifier   (classifier.py)
         |
         v
    ┌────┴────────┐
    |             |
  LOW/HIGH    MEDIUM (uncertain)
    |             |
    |             v
    |         LLM Judge  (llm_judge.py)
    |             |
    └──────┬──────┘
           |
           v
      DetectorOutput
           |
           v
       VERIFIER

Key design decisions (from spec):
    - Claim extraction FIRST (before classifier)
    - Classifier is the primary detector, NOT NLI
    - LLM judge is OPTIONAL, only for uncertain cases
    - No web search, Tavily, Wikipedia in this module
    - Degraded mode: return 0.5 probability, route to VERIFY (fail-closed)
    - hallucination_probability is a TRIAGE SIGNAL, not a factual verdict
"""

from __future__ import annotations

import logging
from typing import Optional

from .claim_extractor import ClaimExtractor
from .classifier import HallucinationClassifier
from .config import JudgeDetectorConfig
from .llm_judge import LLMJudge
from .models import (
    Claim,
    ClaimDetectionResult,
    DetectorInput,
    DetectorOutput,
    RiskLevel,
    RoutingDecision,
)

logger = logging.getLogger(__name__)


class JudgeDetector:
    """New standalone hallucination detector for HalluciGuard.

    This is completely independent of agents/detector_agent/.
    It will replace the old detector only AFTER passing benchmarks.

    Usage
    -----
        detector = JudgeDetector()
        result = detector.detect(
            user_query="Who created Java?",
            llm_response="Java was created by Dennis Ritchie in 1972. "
                         "It was developed at Sun Microsystems.",
        )

        # result.overall_risk   -> RiskLevel.HIGH
        # result.routing        -> RoutingDecision.VERIFY
        # result.claims[0].text -> "Java was created by Dennis Ritchie in 1972."
        # result.claims[0].hallucination_probability -> 0.94

    Per-claim output example (from spec):
        {
          "claims": [
            {"claim_id": "C001", "text": "Java was created by Dennis Ritchie.",
             "hallucination_probability": 0.94, "risk": "HIGH"},
            {"claim_id": "C002", "text": "Java was developed at Sun Microsystems.",
             "hallucination_probability": 0.08, "risk": "LOW"},
          ],
          "overall_risk": "HIGH",
          "routing": "VERIFY"
        }
    """

    # Shared instances across requests (loaded once)
    _shared_classifier: Optional[HallucinationClassifier] = None
    _shared_extractor: Optional[ClaimExtractor] = None
    _classifier_loaded: bool = False

    def __init__(self, config: Optional[JudgeDetectorConfig] = None) -> None:
        self.config = config or JudgeDetectorConfig()

        # Initialise Claim Extractor (fast, no model)
        if JudgeDetector._shared_extractor is None:
            JudgeDetector._shared_extractor = ClaimExtractor(
                max_claims=self.config.max_claims_per_response,
                min_claim_length=self.config.min_claim_length,
            )
        self._extractor = JudgeDetector._shared_extractor

        # Initialise Classifier (loads model lazily on first detect() call)
        if JudgeDetector._shared_classifier is None:
            JudgeDetector._shared_classifier = HallucinationClassifier(
                model_path=self.config.classifier_model_path,
                max_length=self.config.classifier_max_length,
            )
        self._classifier = JudgeDetector._shared_classifier

        # LLM Judge (only created if enabled and API key present)
        self._llm_judge: Optional[LLMJudge] = None
        if self.config.llm_judge_enabled and self.config.openrouter_api_key:
            self._llm_judge = LLMJudge(
                api_key=self.config.openrouter_api_key,
                base_url=self.config.openrouter_base_url,
                model=self.config.judge_llm_model,
                timeout=self.config.judge_llm_timeout,
            )
            logger.info("[JudgeDetector] LLM judge enabled: %s", self.config.judge_llm_model)
        else:
            logger.info("[JudgeDetector] LLM judge disabled (no API key or disabled in config).")

    def _ensure_classifier_loaded(self) -> bool:
        """Load the classifier on first call. Returns True if loaded OK."""
        if not JudgeDetector._classifier_loaded:
            success = self._classifier.load()
            JudgeDetector._classifier_loaded = True
            if not success:
                logger.warning("[JudgeDetector] Classifier load failed — degraded mode.")
            return success
        return getattr(self._classifier, "_is_finetuned", False)

    def _risk_level(self, prob: float) -> RiskLevel:
        """Map probability to risk level using configured thresholds."""
        if prob <= self.config.low_risk_threshold:
            return RiskLevel.LOW
        if prob >= self.config.high_risk_threshold:
            return RiskLevel.HIGH
        return RiskLevel.MEDIUM

    def _routing(self, risk: RiskLevel) -> RoutingDecision:
        """Determine routing based on risk level."""
        if self.config.always_verify:
            return RoutingDecision.VERIFY
        if risk == RiskLevel.LOW:
            return RoutingDecision.ACCEPT
        return RoutingDecision.VERIFY

    def _process_claim(
        self,
        claim: Claim,
        user_query: str,
        classifier_prob: float,
        context: Optional[str],
    ) -> ClaimDetectionResult:
        """Run classifier + optional LLM judge for one claim."""
        risk = self._risk_level(classifier_prob)
        final_prob = classifier_prob
        judge_prob = None
        judge_reasoning = None
        method = "classifier"
        llm_judge_invoked = False

        # LLM Judge: only for MEDIUM (uncertain) risk
        if (
            risk == RiskLevel.MEDIUM
            and self._llm_judge is not None
        ):
            logger.info(
                "[JudgeDetector] MEDIUM risk (%.3f) — invoking LLM judge for: %r",
                classifier_prob,
                claim.text[:80],
            )
            judge_result = self._llm_judge.evaluate(
                user_query=user_query,
                claim_text=claim.text,
                context=context,
                classifier_probability=classifier_prob,
            )
            judge_prob = judge_result.probability
            judge_reasoning = judge_result.reasoning
            # Weighted blend: 40% classifier + 60% LLM judge
            final_prob = round(0.4 * classifier_prob + 0.6 * judge_prob, 4)
            risk = self._risk_level(final_prob)
            method = "classifier+llm_judge"
            llm_judge_invoked = True
            logger.info(
                "[JudgeDetector] LLM judge done. classifier=%.3f judge=%.3f final=%.3f risk=%s",
                classifier_prob, judge_prob, final_prob, risk.value,
            )

        return ClaimDetectionResult(
            claim_id=claim.claim_id,
            text=claim.text,
            hallucination_probability=round(final_prob, 4),
            confidence=round(abs(final_prob - 0.5) * 2, 4),  # 0=uncertain, 1=confident
            risk_level=risk,
            requires_verification=(risk != RiskLevel.LOW),
            classifier_probability=round(classifier_prob, 4),
            llm_judge_probability=round(judge_prob, 4) if judge_prob is not None else None,
            llm_judge_reasoning=judge_reasoning,
            method_used=method,
        ), llm_judge_invoked

    def detect(
        self,
        user_query: str,
        llm_response: str,
        context: Optional[str] = None,
    ) -> DetectorOutput:
        """Run the full detection pipeline on one LLM response.

        Pipeline:
            1. Validate inputs
            2. Extract atomic claims (ClaimExtractor)
            3. Batch-predict hallucination probability (DeBERTa Classifier)
            4. For MEDIUM-risk claims: invoke optional LLM Judge
            5. Aggregate into DetectorOutput

        Args:
            user_query:   The original user question.
            llm_response: The draft answer from the Base LLM.
            context:      Optional RAG context for faithfulness checks.

        Returns:
            DetectorOutput — the pipeline state contract for the Verifier.
        """
        # ── Input validation ──────────────────────────────────────────────────
        if not user_query or not user_query.strip():
            logger.warning("[JudgeDetector] Empty user_query — degraded output.")
            return self._degraded_output("Empty user_query")

        if not llm_response or not llm_response.strip():
            logger.warning("[JudgeDetector] Empty llm_response — degraded output.")
            return self._degraded_output("Empty llm_response")

        # ── Step 1: Claim Extraction ──────────────────────────────────────────
        claims = self._extractor.extract(llm_response, user_query)

        if not claims:
            logger.warning("[JudgeDetector] No claims extracted — single-claim fallback.")
            from .models import Claim, ClaimType
            claims = [Claim(
                claim_id="C001",
                text=llm_response.strip()[:512],
                claim_type=ClaimType.GENERAL,
                importance="MEDIUM",
            )]

        # ── Step 2: Load Classifier ───────────────────────────────────────────
        is_finetuned = self._ensure_classifier_loaded()
        degraded = not is_finetuned

        # ── Step 3: Batch-predict per claim ───────────────────────────────────
        if is_finetuned:
            claim_texts = [c.text for c in claims]
            classifier_probs = self._classifier.predict_batch(user_query, claim_texts)
        else:
            # Degraded: use 0.5 for all (fail-closed)
            classifier_probs = [0.5] * len(claims)

        # ── Step 4: Per-claim processing (+ optional LLM judge) ───────────────
        claim_results = []
        llm_judge_invoked_any = False

        for claim, prob in zip(claims, classifier_probs):
            result, judge_used = self._process_claim(
                claim, user_query, prob, context
            )
            claim_results.append(result)
            if judge_used:
                llm_judge_invoked_any = True

        # ── Step 5: Aggregate ─────────────────────────────────────────────────
        max_prob = max(r.hallucination_probability for r in claim_results)
        overall_risk = self._risk_level(max_prob)
        routing = self._routing(overall_risk)

        num_high = sum(1 for r in claim_results if r.risk_level == RiskLevel.HIGH)
        num_med  = sum(1 for r in claim_results if r.risk_level == RiskLevel.MEDIUM)
        num_low  = sum(1 for r in claim_results if r.risk_level == RiskLevel.LOW)

        logger.info(
            "[JudgeDetector] Done. claims=%d high=%d medium=%d low=%d "
            "max_prob=%.4f overall_risk=%s routing=%s degraded=%s judge=%s",
            len(claim_results), num_high, num_med, num_low,
            max_prob, overall_risk.value, routing.value,
            degraded, llm_judge_invoked_any,
        )

        return DetectorOutput(
            claims=claim_results,
            overall_hallucination_probability=round(max_prob, 4),
            overall_risk=overall_risk,
            routing=routing,
            model_source=self._classifier.model_name,
            status="degraded" if degraded else "completed",
            degraded=degraded,
            num_claims=len(claim_results),
            num_high_risk=num_high,
            num_medium_risk=num_med,
            num_low_risk=num_low,
            llm_judge_invoked=llm_judge_invoked_any,
        )

    def _degraded_output(self, reason: str) -> DetectorOutput:
        """Fail-closed degraded output. Always routes to VERIFY."""
        logger.warning("[JudgeDetector] Degraded output: %s", reason)
        degraded_claim = ClaimDetectionResult(
            claim_id="C001",
            text=reason,
            hallucination_probability=0.5,
            confidence=0.0,
            risk_level=RiskLevel.HIGH,
            requires_verification=True,
            method_used="fallback",
        )
        return DetectorOutput(
            claims=[degraded_claim],
            overall_hallucination_probability=0.5,
            overall_risk=RiskLevel.HIGH,
            routing=RoutingDecision.VERIFY,
            model_source="degraded",
            status="degraded",
            degraded=True,
            num_claims=1,
            num_high_risk=1,
            num_medium_risk=0,
            num_low_risk=0,
        )


__all__ = ["JudgeDetector"]
