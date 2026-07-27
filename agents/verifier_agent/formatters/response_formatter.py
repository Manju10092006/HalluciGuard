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

        # Calculate overall evidence confidence
        if claim_reports:
            avg_trust = sum(cr.trust_score for cr in claim_reports) / len(claim_reports)
        else:
            avg_trust = 0.0

        return VerifierOutputV2(
            query_id=query_id,
            domain=domain,
            domain_validated=domain_validated,
            retrieved_sources=retrieved_sources,
            verified_sources=verified_sources or len(claim_reports),
            claim_evidence=claim_reports,
            overall_evidence_confidence=round(avg_trust, 4),
            latency_ms=latency_ms,
            pipeline_stages=pipeline_stages,
            runtime_models=runtime_models,
            cache_hit=cache_hit,
        )
