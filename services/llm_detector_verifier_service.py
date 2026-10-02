from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Optional

from halluciguard_detector import DetectorAgent
from halluciguard_detector.phase1 import FAST_PATH_RELEASE_ENABLED
from services.base_llm_service import BaseLLMConfig, BaseLLMService, GenerationResult

logger = logging.getLogger(__name__)


def _valid_risk_score(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _source_identifier(value: Any) -> str | None:
    # Providers may omit this optional field. Never serialize a proxy/custom
    # object or invent an identifier from its repr; preserve real scalar IDs.
    return str(value) if type(value) in (str, int) else None


def _load_verifier_imports():
    verifier_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "agents", "verifier_agent")
    )
    if verifier_dir not in sys.path:
        sys.path.insert(0, verifier_dir)
    from api.pipeline import VerificationPipeline
    from schemas.models import SuspiciousClaim, VerifierInputV2

    return VerificationPipeline, SuspiciousClaim, VerifierInputV2


def _load_certification():
    """Import the (lightweight) certification helpers without importing the
    heavy verifier pipeline. Safe to call before deciding whether to verify."""
    verifier_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "agents", "verifier_agent")
    )
    if verifier_dir not in sys.path:
        sys.path.insert(0, verifier_dir)
    from api.certification import (
        enforce_detector,
        certification_enabled_from_env,
        CertificationError,
    )

    return enforce_detector, certification_enabled_from_env, CertificationError


@dataclass(frozen=True)
class LLMDetectorVerifierSliceResult:
    """Acceptance contract structure for Step 3: Base LLM -> Detector -> conditional Verifier."""

    user_query: str
    draft_response: str
    generation: dict[str, Any]
    detector: dict[str, Any] | None
    verifier: dict[str, Any] | None
    grounded_detector: dict[str, Any] | None = None
    certification: dict[str, Any] | None = None

    def model_dump(self) -> dict[str, Any]:
        """Convert the slice result to a dictionary."""
        return asdict(self)


class BaseLLMDetectorVerifierService:
    """Orchestrates Step 3 vertical slice: Base LLM -> Detector -> conditional Verifier.
    
    Data Flow:
        user_query -> BaseLLMService -> draft_response -> DetectorAgent -> Risk Level
                                                               │
                                                     ┌─────────┴─────────┐
                                                     │                   │
                                              Evidence-free triage
                                                     │
                                                     ▼
                                              Verifier -> Evidence & NLI
    """

    def __init__(
        self,
        llm_service: Optional[BaseLLMService] = None,
        detector_agent: Optional[DetectorAgent] = None,
        verifier_pipeline: Optional[Any] = None,
    ) -> None:
        self.llm_service = llm_service or BaseLLMService()
        self.detector_agent = detector_agent or DetectorAgent()
        self._verifier_pipeline = verifier_pipeline

    async def execute_slice(
        self,
        user_query: str,
        domain: str = "general",
        force_verifier: bool = False,
        conversation_history: list[dict[str, str]] | None = None,
        generation_mode: str = "normal",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMDetectorVerifierSliceResult:
        """
        Execute the Base LLM -> Detector -> conditional Verifier vertical slice.

        Args:
            user_query: The user's question or prompt.
            domain: The verification domain (e.g., general, biomedical).
            force_verifier: If True, skip detector gating and always run the verifier.
            conversation_history: Optional conversation context.
            generation_mode: Either "normal" or "stress_test".
            temperature: Optional temperature override.
            max_tokens: Optional token limit.

        Returns:
            A LLMDetectorVerifierSliceResult containing generation, detection, and optional verification outputs.
        """
        # Step 1: Base LLM Generation
        gen_result: GenerationResult = await self.llm_service.generate(
            user_query=user_query,
            conversation_history=conversation_history,
            generation_mode=generation_mode,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        gen_dict: dict[str, Any] = {
            "status": gen_result.status,
            "provider": gen_result.provider,
            "model": gen_result.model,
            "latency_ms": gen_result.latency_ms,
            "request_id": gen_result.request_id,
            "finish_reason": gen_result.finish_reason,
        }
        if gen_result.error:
            gen_dict["error"] = gen_result.error
            gen_dict["error_code"] = gen_result.error_code

        draft_response = gen_result.draft_response or ""

        # Guard: If LLM generation failed or produced empty draft_response, skip Detector & Verifier
        if gen_result.status != "success" or not draft_response.strip():
            logger.warning(
                "[LLMDetectorVerifierService] Skipping detector & verifier: LLM generation failed or returned empty response."
            )
            return LLMDetectorVerifierSliceResult(
                user_query=user_query,
                draft_response=draft_response,
                generation=gen_dict,
                detector=None,
                verifier=None,
            )

        # Step 2: pre-retrieval claim triage. The Verifier obtains evidence;
        # no truth probability is invented before that evidence exists.
        try:
            detection_result: dict[str, Any] = self.detector_agent.detect(
                user_query=user_query,
                llm_response=draft_response,
            )

            if not isinstance(detection_result, dict):
                raise ValueError("invalid_detector_result")
            for key in ("hallucination_probability", "confidence_score"):
                value = detection_result.get(key)
                if value is not None and not _valid_risk_score(value):
                    raise ValueError("invalid_detector_score")
            status = str(detection_result.get("status", "unknown")).lower()
            completed = status == "completed"
            risk_tier = str(detection_result.get("risk_level", "HIGH")).upper()
            next_act_str = str(detection_result.get("next_action", "Verify")).upper()
            decision = "ACCEPT" if next_act_str == "ACCEPT" else "VERIFY"
            model_source = str(detection_result.get("model_source", "halluciguard_detector"))
            # Prefer the explicit degraded flag and preserve execution diagnostics.
            degraded = detection_result.get("detector_degraded") is not False or not completed

            detector_dict: dict[str, Any] = {
                "confidence_score": detection_result.get("confidence_score"),
                "hallucination_probability": detection_result.get("hallucination_probability"),
                "risk_tier": risk_tier,
                "decision": decision,
                "status": status,
                "model_source": model_source,
                # §6 execution diagnostics — prove real inference vs. degraded fallback.
                "detector_model_loaded": detection_result.get("model_loaded") is True,
                "detector_inference_executed": detection_result.get("inference_executed") is True,
                "detector_degraded": degraded,
                "calibrated": detection_result.get("calibration_applied") is True,
                "detector_model_source": model_source,
                "probability_available": detection_result.get("hallucination_probability") is not None,
                "grounded": detection_result.get("grounded") is True,
                "claims": detection_result.get("claims", []),
                "verification_reason": detection_result.get("verification_reason"),
            }
        except Exception as exc:
            logger.error("[LLMDetectorVerifierService] Detector failed (%s)", type(exc).__name__)
            detector_dict = {
                "status": "failed",
                "error": f"Detector error: {type(exc).__name__}",
                "risk_tier": "UNKNOWN",
                "decision": "VERIFY",
                "hallucination_probability": None,
                "confidence_score": None,
                "probability_available": False,
                "detector_degraded": True,
            }
            decision, risk_tier = "VERIFY", "UNKNOWN"

        # §28 certification: a degraded detector (baseline heuristic) must not be
        # silently certified. Initial triage cannot be certified as grounded. Import is
        # guarded so normal (non-certification) runs are never affected.
        certification_requested = os.environ.get("CERTIFICATION_MODE", "false").strip().lower() in {"true", "1", "yes", "on"}
        certification_helpers_available = False
        try:
            _enforce_detector, _cert_from_env, _CertErr = _load_certification()
            certification_helpers_available = True
            certification_requested = certification_requested or bool(_cert_from_env())
        except Exception as _imp_exc:  # pragma: no cover - defensive
            _enforce_detector = None
            logger.debug("Certification helpers unavailable (%s)", type(_imp_exc).__name__)
        # This is evidence-free triage, not the grounded Detector stage. A
        # certification failure here must not prevent evidence retrieval.
        detector_dict["certification_status"] = (
            "pending_grounded_detection" if certification_requested else "not_requested"
        )
        if certification_requested:
            # Initial triage is never certified; final stage certification is
            # separate and runs only after independent evidence verification.
            detector_dict["certification_completed"] = False
            detector_dict["certification_reason"] = (
                "grounded_detection_not_executed" if certification_helpers_available
                else "certification_helpers_unavailable"
            )

        # Step 3: Conditional Verifier Routing
        should_verify = (
            force_verifier
            or not FAST_PATH_RELEASE_ENABLED
            or certification_requested
            or decision == "VERIFY"
            or risk_tier in {"MEDIUM", "HIGH"}
            or risk_tier not in {"LOW", "MEDIUM", "HIGH"}
            or detector_dict.get("status") != "completed"
            or detector_dict.get("detector_degraded", True)
            or not detector_dict.get("grounded", False)
            or not detector_dict.get("detector_inference_executed", False)
            or not detector_dict.get("probability_available", False)
            or not detector_dict.get("calibrated", False)
            or not _valid_risk_score(detector_dict.get("hallucination_probability"))
            or os.environ.get("ALWAYS_VERIFY", "true").lower() in {"true", "1"}
            or os.environ.get("ALLOW_DETECTOR_FAST_PATH", "false").lower() not in {"true", "1"}
        )
        if not should_verify:
            logger.info(
                f"[LLMDetectorVerifierService] Detector risk_tier={risk_tier} (decision={decision}). Skipping Verifier."
            )
            verifier_dict: dict[str, Any] = {
                "executed": False,
                "reason": f"Detector classified risk as {risk_tier} ({decision}). Verifier skipped.",
            }
            return LLMDetectorVerifierSliceResult(
                user_query=user_query,
                draft_response=draft_response,
                generation=gen_dict,
                detector=detector_dict,
                verifier=verifier_dict,
            )

        # Execute Verifier Pipeline
        logger.info(
            "[LLMDetectorVerifierService] Invoking Verifier Pipeline"
        )
        try:
            VerificationPipelineClass, SuspiciousClaim, VerifierInputV2 = (
                _load_verifier_imports()
            )
            pipeline = self._verifier_pipeline or VerificationPipelineClass()

            from agents.verifier_agent.claims.claim_decomposer import ClaimDecomposer
            decomposer = ClaimDecomposer()
            sub_claims = decomposer.decompose(draft_response)
            if not sub_claims:
                sub_claims = [draft_response]

            suspicious_claims = [
                SuspiciousClaim(claim_id=f"c{idx+1}", text=sc)
                for idx, sc in enumerate(sub_claims[:5])
            ]

            payload = VerifierInputV2(
                query_id=gen_result.request_id or str(uuid.uuid4()),
                domain=domain,
                suspicious_claims=suspicious_claims,
            )

            verifier_output = await pipeline.verify(payload)

            def _safe_float(val, default=0.0):
                try:
                    if hasattr(val, "_mock_name") or "MagicMock" in str(type(val)):
                        return default
                    return float(val)
                except (TypeError, ValueError):
                    return default

            claims_data = []
            for claim_rep in getattr(verifier_output, "claim_evidence", []):
                ev_items = []
                for ev in getattr(claim_rep, "evidence", []):
                    ev_items.append(
                        {
                            "source_id": _source_identifier(getattr(ev, "source_id", None)),
                            "title": str(getattr(ev, "title", "")),
                            "source": str(getattr(ev, "source", "")),
                            "url": str(getattr(ev, "url", "")),
                            "publication_date": str(getattr(ev, "publication_date", "")),
                            "snippet": str(getattr(ev, "snippet", "")),
                            "entailment_label": getattr(
                                getattr(ev, "entailment_label", "neutral"),
                                "value",
                                str(getattr(ev, "entailment_label", "neutral")),
                            ),
                            "entailment_score": _safe_float(
                                getattr(ev, "entailment_score", 0.0)
                            ),
                            "credibility_score": _safe_float(
                                getattr(ev, "credibility_score", 0.0)
                            ),
                            "adapter_score": _safe_float(
                                getattr(ev, "adapter_score", 0.0)
                            ),
                            "bge_score": _safe_float(
                                getattr(ev, "bge_score", 0.0)
                            ),
                            "nli_entailment": _safe_float(
                                getattr(ev, "nli_entailment", getattr(ev, "entailment_score", 0.0))
                            ),
                            "nli_contradiction": _safe_float(
                                getattr(ev, "nli_contradiction", 0.0)
                            ),
                            "nli_neutral": _safe_float(
                                getattr(ev, "nli_neutral", 0.0)
                            ),
                            "classification": str(
                                getattr(ev, "classification", "")
                            ),
                        }
                    )
                verdict_obj = getattr(claim_rep, "verdict", "")
                verdict_str = (
                    getattr(verdict_obj, "value", str(verdict_obj))
                    if verdict_obj
                    else ""
                )

                trace_raw = getattr(claim_rep, "retrieval_trace", None)
                if trace_raw is not None and not hasattr(trace_raw, "_mock_name") and "MagicMock" not in str(type(trace_raw)):
                    if hasattr(trace_raw, "model_dump"):
                        trace_dict = trace_raw.model_dump()
                    elif isinstance(trace_raw, dict):
                        trace_dict = trace_raw
                    else:
                        trace_dict = None
                else:
                    trace_dict = None

                claims_data.append(
                    {
                        "claim_id": str(getattr(claim_rep, "claim_id", "c1")),
                        "claim_text": str(getattr(claim_rep, "claim_text", draft_response)),
                        "verdict": verdict_str,
                        "support_score": _safe_float(
                            getattr(claim_rep, "support_score", 0.0)
                        ),
                        "contradiction_score": _safe_float(
                            getattr(claim_rep, "contradiction_score", 0.0)
                        ),
                        "trust_score": _safe_float(
                            getattr(claim_rep, "trust_score", 0.0)
                        ),
                        "confidence_score": _safe_float(
                            getattr(claim_rep, "confidence_score", 0.0)
                        ),
                        "explanation": str(getattr(claim_rep, "explanation", "")),
                        "evidence": ev_items,
                        "retrieval_trace": trace_dict,
                    }
                )

            verifier_dict = {
                "executed": True,
                "status": "degraded" if len(sub_claims) > len(suspicious_claims) else "completed",
                "candidate_claim_count": len(sub_claims),
                "submitted_claim_count": len(suspicious_claims),
                "claim_limit_reached": len(sub_claims) > len(suspicious_claims),
                "domain": str(getattr(verifier_output, "domain", domain)),
                "domain_validated": bool(getattr(verifier_output, "domain_validated", True)),
                "retrieved_sources": int(_safe_float(getattr(verifier_output, "retrieved_sources", 0))),
                "verified_sources": int(_safe_float(getattr(verifier_output, "verified_sources", 0))),
                "overall_evidence_confidence": _safe_float(
                    getattr(verifier_output, "overall_evidence_confidence", 0.0)
                ),
                "latency_ms": int(_safe_float(getattr(verifier_output, "latency_ms", 0))),
                "cache_hit": bool(getattr(verifier_output, "cache_hit", False)),
                "claim_evidence": claims_data,
            }
        except Exception as exc:
            logger.error("[LLMDetectorVerifierService] Verifier failed (%s)", type(exc).__name__)
            verifier_dict = {
                "executed": True,
                "status": "failed",
                "error": f"Verifier error: {type(exc).__name__}",
                "claim_evidence": [],
            }
            if certification_requested:
                detector_dict["certification_reason"] = "verifier_failed_before_grounded_detection"

        grounded_detector = None
        certification = None
        if certification_requested:
            certification = {"requested": True, "completed": False, "status": "failed",
                             "initial_triage_certified": False}
            if verifier_dict.get("status") != "completed":
                certification["reason"] = "verifier_failed_or_incomplete"
            elif not certification_helpers_available:
                certification["reason"] = "certification_helpers_unavailable"
            else:
                evidence = [
                    {"text": ev["snippet"], "source_id": ev.get("source_id"),
                     "source": ev.get("source"), "url": ev.get("url")}
                    for report in verifier_dict["claim_evidence"]
                    for ev in report["evidence"]
                    if isinstance(ev.get("snippet"), str) and ev["snippet"].strip()
                ]
                if not evidence:
                    certification["reason"] = "no_grounding_evidence"
                else:
                    try:
                        grounded_detector = await asyncio.to_thread(
                            self.detector_agent.detect, user_query=user_query,
                            llm_response=draft_response, evidence=evidence, domain=domain,
                        )
                        _enforce_detector(grounded_detector, enabled=True)
                        certification.update(status="passed", completed=True,
                                             reason="grounded_detector_executed")
                    except Exception as exc:
                        # Never expose provider messages or certify an invalid
                        # result. The independent Verifier result remains intact.
                        grounded_detector = {"status": "failed", "detector_degraded": True,
                                             "hallucination_probability": None,
                                             "error_type": type(exc).__name__}
                        certification.update(reason="grounded_detector_not_certified",
                                             error_type=type(exc).__name__)

        return LLMDetectorVerifierSliceResult(
            user_query=user_query,
            draft_response=draft_response,
            generation=gen_dict,
            detector=detector_dict,
            verifier=verifier_dict,
            grounded_detector=grounded_detector,
            certification=certification,
        )
