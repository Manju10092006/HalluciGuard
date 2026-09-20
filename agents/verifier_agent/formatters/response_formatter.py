from __future__ import annotations
from typing import List, Dict, Any, Optional
from schemas.models import (
    ClaimReport, VerifierOutputV2, EvidenceItem,
    PipelineStageStatus, VerdictLabel, RuntimeModelInfo
)


class ResponseFormatter:
    """Formats the final API response according to the schema."""

    def format_claim_report(
        self,
        claim_id: str,
        claim_text: str,
        evidence_items: List[EvidenceItem],
        scores: Dict[str, float],
        explanation: str,
        verdict: VerdictLabel,
        confidence_score: float = 0.0,
        supporting_sources: List[str] = None,
        contradicting_sources: List[str] = None,
        retrieved_documents: int = 0,
        reranked_documents: int = 0,
        verified_evidence: int = 0,
    ) -> ClaimReport:
        """Format a single claim report with enhanced metadata."""
        return ClaimReport(
            claim_id=claim_id,
            claim_text=claim_text,
            verdict=verdict,
            support_score=float(scores.get('support_score', 0.0)),
            contradiction_score=float(scores.get('contradiction_score', 0.0)),
            trust_score=float(scores.get('trust_score', 0.0)),
            confidence_score=confidence_score,
            evidence=evidence_items,
            explanation=explanation,
            supporting_sources=supporting_sources or [],
            contradicting_sources=contradicting_sources or [],
            retrieved_documents=retrieved_documents,
            reranked_documents=reranked_documents,
            verified_evidence=verified_evidence,
        )

    def format_full_response(
        self,
        query_id: str,
        domain: str,
        domain_validated: bool,
        claim_reports: List[ClaimReport],
        latency_ms: int,
        pipeline_stages: List[PipelineStageStatus] = None,
        retrieved_sources: int = 0,
        verified_sources: int = 0,
        runtime_models: Optional[RuntimeModelInfo] = None,
        cache_hit: bool = False,
    ) -> VerifierOutputV2:
        """Format the full VerifierOutputV2 response."""
        pipeline_stages = pipeline_stages or []

        # Calculate overall evidence confidence.
        #
        # This is the verifier's certainty in its per-claim ASSESSMENTS, which the
        # Judge consumes as the grounding confidence for its decision. It must be
        # the calibrated per-claim ``confidence_score`` (evidence certainty &
        # consensus, non-zero for BOTH verified and contradicted verdicts) — NOT
        # ``trust_score``. Trust reflects only *supporting*-source reliability, so
        # it collapses to ~0 on any contradiction and is structurally low even for
        # strongly-entailed claims; averaging it made well-grounded answers report
        # ~0.2 confidence and get starved into human review. Averaging the
        # calibrated confidence restores a faithful, entailment-driven signal.
        if claim_reports:
            avg_confidence = sum(cr.confidence_score for cr in claim_reports) / len(claim_reports)
        else:
            avg_confidence = 0.0

        return VerifierOutputV2(
            query_id=query_id,
            domain=domain,
            domain_validated=domain_validated,
            retrieved_sources=retrieved_sources,
            verified_sources=verified_sources or len(claim_reports),
            claim_evidence=claim_reports,
            overall_evidence_confidence=round(avg_confidence, 4),
            latency_ms=latency_ms,
            pipeline_stages=pipeline_stages,
            runtime_models=runtime_models,
            cache_hit=cache_hit,
        )
