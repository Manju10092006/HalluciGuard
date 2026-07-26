from __future__ import annotations
from typing import List, Dict, Any
from schemas.models import ClaimReport, VerifierOutputV2, EvidenceItem, PipelineStageStatus, VerdictLabel

class ResponseFormatter:
    """Formats the final API response according to the schema."""

    def format_claim_report(
        self, 
        claim_id: str, 
        claim_text: str, 
        evidence_items: List[EvidenceItem], 
        scores: Dict[str, float], 
        explanation: str, 
        verdict: VerdictLabel
    ) -> ClaimReport:
        """
        Format a single claim report.
        """
        return ClaimReport(
            claim_id=claim_id,
            claim_text=claim_text,
            verdict=verdict,
            support_score=scores.get('support_score', 0.0),
            contradiction_score=scores.get('contradiction_score', 0.0),
            trust_score=scores.get('trust_score', 0.0),
            evidence=evidence_items,
            explanation=explanation
        )

    def format_full_response(
        self, 
        query_id: str, 
        domain: str, 
        domain_validated: bool, 
        claim_reports: List[ClaimReport], 
        latency_ms: int, 
        pipeline_stages: List[PipelineStageStatus]
    ) -> VerifierOutputV2:
        """
        Format the full VerifierOutputV2 response.
        """
        # Determine overall verdict based on the worst claim verdict
        # or an aggregate logic
        has_hallucinated = any(cr.verdict == VerdictLabel.LIKELY_HALLUCINATED for cr in claim_reports)
        has_mixed = any(cr.verdict == VerdictLabel.MIXED_EVIDENCE for cr in claim_reports)
        
        if has_hallucinated:
            overall_verdict = VerdictLabel.LIKELY_HALLUCINATED
        elif has_mixed:
            overall_verdict = VerdictLabel.MIXED_EVIDENCE
        else:
            overall_verdict = VerdictLabel.VERIFIED

        return VerifierOutputV2(
            query_id=query_id,
            overall_verdict=overall_verdict,
            domain=domain,
            domain_validated=domain_validated,
            claims=claim_reports,
            latency_ms=latency_ms,
            pipeline_stages=pipeline_stages
        )
