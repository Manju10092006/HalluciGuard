"""
HalluciGuard - Enterprise Judge Agent Orchestrator
Main orchestrator class tying together Domain Policy Registry, Evidence Intelligence,
Source Consensus Engine, Contradiction Taxonomy, Dynamic 11-Signal Calibrator, Decision Intelligence Engine,
Memory Intelligence, Multi-Agent Negotiation Protocol, and Reproducible Audit Logging.
Includes Circuit Breaker, Graceful Degradation, and Fallback Policies.
"""

import time
import logging
import asyncio
from typing import Dict, List, Any, Optional

from config import JudgeConfig, DEFAULT_CONFIG
from domain_policies import DomainPolicyRegistry, DEFAULT_DOMAIN_REGISTRY
from evidence_intelligence import EvidenceIntelligenceEngine
from source_consensus import SourceConsensusEngine
from contradiction_analyzer import ContradictionTaxonomyAnalyzer
from claim_criticality import ClaimCriticalityAssessor
from memory_integration import MemoryIntelligenceEngine
from nli_engine import NLIEngine
from confidence_calibrator import DynamicConfidenceCalibrator
from decision_engine import DecisionIntelligenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HalluciGuard.JudgeAgent")


class JudgeAgent:
    def __init__(self, config: Optional[JudgeConfig] = None):
        self.config = config or DEFAULT_CONFIG
        logger.info("Initializing HalluciGuard Enterprise Judge Agent Engine...")

        self.domain_registry = DEFAULT_DOMAIN_REGISTRY
        self.evidence_intel_engine = EvidenceIntelligenceEngine(config=self.config)
        self.consensus_engine = SourceConsensusEngine()
        self.contradiction_analyzer = ContradictionTaxonomyAnalyzer()
        self.criticality_assessor = ClaimCriticalityAssessor()
        self.memory_engine = MemoryIntelligenceEngine()

        self.nli_engine = NLIEngine(
            model_name=self.config.default_nli_model,
            use_hf=self.config.use_huggingface,
        )
        self.calibrator = DynamicConfidenceCalibrator(config=self.config)
        self.decision_engine = DecisionIntelligenceEngine(config=self.config)

        # Circuit breaker state
        self._consecutive_errors = 0
        self._circuit_open = False

        logger.info("HalluciGuard Enterprise Judge Agent initialized and ready.")

    @staticmethod
    def _normalize_verifier_output(verifier_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert the Verifier Agent v2 response contract into the Judge Agent's
        internal claim-evidence pair contract.

        Verifier v2 emits:
            claim_evidence -> ClaimReport -> evidence -> EvidenceItem

        Judge internally consumes:
            [{claim, evidence, source, publication_date, evidence_confidence, ...}]

        Keeping this translation at the boundary prevents both agents from
        depending on each other's internal Python models.
        """
        raw_pairs = verifier_output.get("claim_evidence_pairs", [])
        if raw_pairs:
            return list(raw_pairs)

        normalized: List[Dict[str, Any]] = []
        claim_reports = verifier_output.get("claim_evidence", [])

        for claim_report in claim_reports:
            if not isinstance(claim_report, dict):
                continue

            claim_text = str(claim_report.get("claim_text", "")).strip()
            if not claim_text:
                continue

            evidence_items = claim_report.get("evidence", []) or []
            for rank, evidence_item in enumerate(evidence_items, start=1):
                if not isinstance(evidence_item, dict):
                    continue

                evidence_text = str(evidence_item.get("snippet", "")).strip()
                if not evidence_text:
                    continue

                entailment_score = float(evidence_item.get("entailment_score", 0.0) or 0.0)
                credibility_score = float(evidence_item.get("credibility_score", 0.0) or 0.0)

                normalized.append(
                    {
                        "claim": claim_text,
                        "evidence": evidence_text,
                        "evidence_source": evidence_item.get("source", "Unknown"),
                        "source": evidence_item.get("source", "Unknown"),
                        "url": evidence_item.get("url", ""),
                        "publication_date": evidence_item.get("publication_date", ""),
                        "evidence_confidence": round(
                            max(0.0, min(1.0, entailment_score * credibility_score)), 4
                        ),
                        "verifier_entailment": entailment_score,
                        "verifier_credibility": credibility_score,
                        "verifier_verdict": claim_report.get("verdict", ""),
                        "verifier_trust_score": claim_report.get("trust_score", 0.0),
                        "rank": rank,
                    }
                )

        return normalized

    def evaluate(
        self,
        detector_output: Dict[str, Any],
        verifier_output: Dict[str, Any],
        user_query: str = "",
        draft_response: str = "",
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the complete Enterprise Decision Intelligence Pipeline."""
        start_time = time.time()

        if self._circuit_open:
            logger.warning("Circuit Breaker is OPEN due to past errors. Returning graceful fallback decision.")
            return self._graceful_fallback(user_query, draft_response, "CIRCUIT_BREAKER_ACTIVE")

        try:
            # 1. Normalize Inputs
            detector_prob = detector_output.get("hallucination_probability", 0.3)
            detector_conf = detector_output.get("confidence_score", 0.8)
            domain_name = verifier_output.get("domain", "General Knowledge")

            # 2. Domain Policy Lookup
            domain_policy = self.domain_registry.get_policy(domain_name)

            # 3. Normalize Verifier v2 -> Judge claim/evidence pairs
            raw_pairs = self._normalize_verifier_output(verifier_output)
            logger.info(
                "Judge received %d claim-evidence pairs from Verifier Agent",
                len(raw_pairs),
            )

            # 4. NLI Inference
            evaluated_pairs = self.nli_engine.batch_predict(raw_pairs)

            # 5. Evidence Intelligence (Authority, Freshness, Diversity, Graph)
            evidence_intel = self.evidence_intel_engine.analyze_evidence_set(
                evaluated_pairs, domain_name
            )

            # 6. Source Consensus Matrix
            consensus_data = self.consensus_engine.evaluate_consensus(evaluated_pairs)

            # 7. Contradiction Taxonomy & Primary Contradiction
            contradiction_data = {"has_contradiction": False, "risk_weight": 0.0}
            for pair in evaluated_pairs:
                contradiction = self.contradiction_analyzer.classify_contradiction(
                    pair.get("claim", ""),
                    pair.get("evidence", ""),
                    pair.get("nli_scores", {}),
                )
                if (
                    contradiction["has_contradiction"]
                    and contradiction["risk_weight"] > contradiction_data["risk_weight"]
                ):
                    contradiction_data = contradiction

            # 8. Claim Criticality Assessment
            criticality_data = self.criticality_assessor.evaluate_criticality(
                draft_response, domain_name
            )

            # 9. Memory Signals Integration
            sources = [pair.get("source", "Unknown") for pair in evaluated_pairs]
            memory_data = self.memory_engine.evaluate_memory_signals(
                draft_response, sources, memory_context
            )

            # 10. Dynamic 11-Signal Calibration
            calibration_results = self.calibrator.calibrate_11_signal(
                detector_prob=detector_prob,
                detector_confidence=detector_conf,
                evidence_intel=evidence_intel,
                consensus_data=consensus_data,
                memory_data=memory_data,
                contradiction_data=contradiction_data,
                criticality_data=criticality_data,
                domain_policy=domain_policy,
            )

            # 11. Decision Intelligence Engine
            decision_results = self.decision_engine.evaluate_decision(
                calibration_results=calibration_results,
                evidence_intel=evidence_intel,
                consensus_data=consensus_data,
                memory_data=memory_data,
                criticality_data=criticality_data,
                domain_policy=domain_policy,
                user_query=user_query,
                draft_response=draft_response,
            )

            execution_latency_ms = round((time.time() - start_time) * 1000, 2)
            self._consecutive_errors = 0

            return {
                "agent": "JUDGE_AGENT",
                "status": "SUCCESS",
                "decision": decision_results["judge_decision"],
                "severity": decision_results["severity"],
                "reason": decision_results["reason"],
                "explanation": decision_results["explanation"],
                "next_action": decision_results["next_action"],
                "domain_policy_enforced": domain_policy.domain_name,
                "metrics": {
                    "calibrated_confidence": calibration_results["calibrated_confidence"],
                    "risk_score": calibration_results["risk_score"],
                    "overall_entailment": calibration_results["overall_entailment"],
                    "overall_contradiction": calibration_results["overall_contradiction"],
                    "source_authority": evidence_intel["overall_authority"],
                    "evidence_freshness": evidence_intel["overall_freshness"],
                    "diversity_index": evidence_intel["diversity_index"],
                    "consensus_score": consensus_data["consensus_score"],
                    "conflict_index": consensus_data["conflict_index"],
                    "latency_ms": execution_latency_ms,
                },
                "evidence_intelligence": evidence_intel,
                "source_consensus": consensus_data,
                "contradiction_taxonomy": decision_results["contradiction_taxonomy"],
                "claim_criticality": criticality_data,
                "corrector_payload": decision_results["corrector_payload"],
                "negotiation_protocol": decision_results["negotiation_protocol"],
                "audit_record": decision_results["audit_record"],
                "observability_bus_signal": {
                    "event": "ENTERPRISE_JUDGE_DECISION_EMITTED",
                    "decision": decision_results["judge_decision"],
                    "severity": decision_results["severity"],
                    "domain": domain_policy.domain_name,
                    "calibrated_conf": calibration_results["calibrated_confidence"],
                    "timestamp_ms": time.time(),
                },
            }

        except Exception as exc:
            logger.error("Error during Judge Agent evaluation: %s", exc, exc_info=True)
            self._consecutive_errors += 1
            if self._consecutive_errors >= self.config.circuit_breaker_error_threshold:
                self._circuit_open = True
                logger.error("Circuit breaker triggered! Opening circuit.")
            return self._graceful_fallback(user_query, draft_response, str(exc))

    def _graceful_fallback(
        self,
        user_query: str,
        draft_response: str,
        error_reason: str,
    ) -> Dict[str, Any]:
        """Provide a safe fallback decision during runtime failures."""
        return {
            "agent": "JUDGE_AGENT",
            "status": "FALLBACK_MODE",
            "decision": "ABSTAIN",
            "severity": "HIGH",
            "reason": f"System in fallback state ({error_reason}).",
            "explanation": "Runtime exception occurred; returning safe fallback abstain decision.",
            "next_action": "Retry verification via fallback agent",
            "metrics": {"calibrated_confidence": 0.0, "latency_ms": 0.0},
            "corrector_payload": {
                "judge_decision": "ABSTAIN",
                "severity": "HIGH",
                "reason": "System fallback mode",
                "hallucinated_claims": [],
                "trusted_evidence": [],
                "user_query": user_query,
                "original_draft_response": draft_response,
            },
        }

    async def evaluate_async(
        self,
        detector_output: Dict[str, Any],
        verifier_output: Dict[str, Any],
        user_query: str = "",
        draft_response: str = "",
    ) -> Dict[str, Any]:
        """Async thread-safe execution interface."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.evaluate,
            detector_output,
            verifier_output,
            user_query,
            draft_response,
        )
