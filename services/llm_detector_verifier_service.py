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
from services.base_llm_service import BaseLLMConfig, BaseLLMService, GenerationResult

logger = logging.getLogger(__name__)


def _detector_record(result: Any) -> dict[str, Any]:
    """Normalize triage diagnostics without treating missing output as success."""
    if not isinstance(result, dict):
        raise ValueError("Detector output must be a dictionary")
    for field in ("hallucination_probability", "confidence_score"):
        raw = result.get(field)
        if raw is not None and (
            isinstance(raw, bool) or not math.isfinite(float(raw))
            or not 0.0 <= float(raw) <= 1.0
        ):
            raise ValueError("Invalid detector numeric output")
    status = str(result.get("status", "unavailable")).lower()
    degraded = bool(result.get("detector_degraded")) or status != "completed"
    risk_tier = str(result.get("risk_level", "UNKNOWN")).upper()
    next_action = str(result.get("next_action", "Verify")).upper()
    decision = "ACCEPT" if (
        not degraded and risk_tier == "LOW" and next_action == "ACCEPT"
    ) else "VERIFY"
    model_source = str(result.get("model_source", "halluciguard_detector"))
    claims = result.get("claims") or []
    if not isinstance(claims, list) or any(not isinstance(c, dict) for c in claims):
        raise ValueError("Invalid detector claims")
    return {
        "status": status,
        "confidence_score": result.get("confidence_score"),
        "hallucination_probability": result.get("hallucination_probability"),
        "risk_tier": risk_tier,
        "decision": decision,
        "model_source": model_source,
        "detector_model_loaded": result.get("model_loaded") is True,
        "detector_inference_executed": result.get("inference_executed") is True,
        "detector_degraded": degraded,
        "detector_model_source": model_source,
        "probability_available": result.get("hallucination_probability") is not None,
        "grounded": result.get("grounded") is True,
        "claims": claims,
        "phase1": result.get("phase1"),
        "verification_reason": result.get("verification_reason"),
    }


def _valid_phase1_bypass(record: Any, claims: list[dict[str, Any]]) -> bool:
    """Only the calibrated Phase 1 fast path can authorize evidence-free bypass."""
    if not claims or any(not str(c.get("text") or "").strip() for c in claims):
        return False
    if not isinstance(record, dict) or not (
        record.get("bypass_eligible") is True and record.get("status") == "scored"
        and record.get("mode") == "fast_path"
        and isinstance(record.get("head_id"), str) and record["head_id"].strip()
        and isinstance(record.get("calibration_id"), str) and record["calibration_id"].strip()
    ):
        return False
    risks = record.get("calibrated_risks")
    if not isinstance(risks, (list, tuple)) or not risks or len(risks) != len(claims):
        return False
    values = [*risks, record.get("response_risk")]
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in values):
        return False
    return math.isclose(max(risks), record["response_risk"], rel_tol=0.0, abs_tol=1e-9)


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

    def model_dump(self) -> dict[str, Any]:
        """Convert the slice result to a dictionary."""
        return asdict(self)


class BaseLLMDetectorVerifierService:
    """Generate, triage, and verify a draft before returning evidence and NLI.

    Verification is the default, including when initial triage is unavailable.
    Bypass requires an explicitly enabled, validated Phase 1 calibrated fast
    path. Optional detector certification describes actual execution only.
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
            force_verifier: If True, always run the verifier for a valid draft.
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

            detector_dict = _detector_record(detection_result)
        except Exception as exc:
            logger.warning("Detector unavailable (%s); continuing with evidence verification.",
                           type(exc).__name__)
            detector_dict = _detector_record({
                "status": "failed",
                "risk_level": "HIGH",
                "next_action": "Verify",
                "detector_degraded": True,
                "model_source": "halluciguard_detector_unavailable",
                "verification_reason": "detector_failure",
            })
            detector_dict["error"] = (
                f"Detector error: {type(exc).__name__}; continuing with evidence verification."
            )

        # Certification needs retrieved evidence. Initial triage cannot satisfy
        # the grounded-inference proof, but must not prevent the Verifier running.
        certification_enabled = os.getenv("CERTIFICATION_MODE", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        try:
            _enforce_detector, _cert_from_env, _CertErr = _load_certification()
            certification_enabled = _cert_from_env()
        except Exception as _imp_exc:  # pragma: no cover - defensive
            _enforce_detector = None
            logger.debug("Certification helpers unavailable (%s).", type(_imp_exc).__name__)

        # Step 3: Conditional Verifier Routing
        risk_tier = detector_dict["risk_tier"]
        decision = detector_dict["decision"]
        always_verify = os.getenv("ALWAYS_VERIFY", "true").lower() in {"true", "1"}
        allow_fast_path = os.getenv("ALLOW_DETECTOR_FAST_PATH", "false").lower() in {"true", "1"}
        should_verify = (
            force_verifier
            or always_verify
            or not allow_fast_path
            or certification_enabled
            or generation_mode == "stress_test"
            or detector_dict["detector_degraded"]
            or detector_dict["status"] != "completed"
            or decision != "ACCEPT"
            or risk_tier != "LOW"
            or not _valid_phase1_bypass(detector_dict["phase1"], detector_dict["claims"])
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
            f"[LLMDetectorVerifierService] Invoking Verifier Pipeline for query: '{user_query[:60]}...'"
        )
        try:
            VerificationPipelineClass, SuspiciousClaim, VerifierInputV2 = (
                _load_verifier_imports()
            )
            pipeline = self._verifier_pipeline or VerificationPipelineClass()

            from claims.claim_decomposer import ClaimDecomposer
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
            logger.error("Verifier execution failed (%s).", type(exc).__name__)
            verifier_dict = {
                "executed": True,
                "status": "failed",
                "error": f"Verifier error: {type(exc).__name__}",
                "claim_evidence": [],
            }

        if certification_enabled:
            certificate = {"status": "not_certified", "stage": "detector"}
            detector_dict["certification"] = certificate
            if _enforce_detector is None:
                certificate["reason"] = "Certification helpers unavailable."
            elif verifier_dict.get("status") == "failed":
                certificate["reason"] = "Verifier execution failed; no grounded detector proof."
            else:
                try:
                    # This slice has no post-retrieval detector stage. Report
                    # certification of its actual execution, without inventing
                    # an additional model call or certifying skipped inference.
                    _enforce_detector(detector_dict, True)
                    if not (detector_dict["detector_model_loaded"] and detector_dict["grounded"]):
                        raise _CertErr("detector", "Grounded model execution proof is incomplete.")
                    certificate["status"] = "certified"
                except Exception as exc:
                    certificate["reason"] = (
                        "This slice did not demonstrate grounded detector inference "
                        f"({type(exc).__name__}); pre-retrieval triage is not a truth verdict."
                    )

        return LLMDetectorVerifierSliceResult(
            user_query=user_query,
            draft_response=draft_response,
            generation=gen_dict,
            detector=detector_dict,
            verifier=verifier_dict,
        )
