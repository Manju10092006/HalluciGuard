"""
HalluciGuard - Decision Explainability & Audit Trail Engine
Generates human-understandable reasoning chains, details why alternative decisions were rejected,
and records reproducible audit logs for enterprise compliance.
"""

import time
import json
import uuid
from typing import Dict, List, Any

class DecisionAuditEngine:
    """Engine for creating audit trails of judge decisions."""
    def __init__(self):
        pass

    def build_audit_record(
        self,
        decision: str,
        severity: str,
        calibrated_conf: float,
        domain_policy: Any,
        evidence_intel: Dict[str, Any],
        consensus_data: Dict[str, Any],
        contradiction_data: Dict[str, Any],
        detector_output: Dict[str, Any],
        user_query: str,
        draft_response: str
    ) -> Dict[str, Any]:
        """
        Constructs a complete, reproducible enterprise audit trail & explanation chain.
        """
        audit_id = f"AUDIT_JUDGE_{uuid.uuid4().hex[:10].upper()}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Generate Reasoning Chain
        reasoning_chain = [
            f"1. Domain Context: Evaluated under policy '{domain_policy.domain_name}' (Strictness: {domain_policy.strictness_level}).",
            f"2. Evidence Quality: Evaluated {len(evidence_intel.get('processed_pairs', []))} evidence item(s). Source Authority: {evidence_intel.get('overall_authority', 0.0):.2f}, Freshness: {evidence_intel.get('overall_freshness', 0.0):.2f}, Diversity Index: {evidence_intel.get('diversity_index', 0.0):.2f}.",
            f"3. Source Consensus: {consensus_data.get('consensus_level', 'UNKNOWN')} with score {consensus_data.get('consensus_score', 0.0):.2f} across {consensus_data.get('independent_source_count', 0)} independent provider(s).",
            f"4. Contradiction Taxonomy: {contradiction_data.get('taxonomy_type', 'NONE')} (Risk Weight: {contradiction_data.get('risk_weight', 0.0):.2f}).",
            f"5. Confidence Calibration: Final calibrated confidence = {calibrated_conf:.2f} (Required for ACCEPT: {domain_policy.accept_confidence_threshold:.2f})."
        ]

        # Alternative Decision Rejection Rationale
        alternatives_rejected = {}
        if decision != "ACCEPT":
            alternatives_rejected["ACCEPT"] = (
                f"Rejected because calibrated confidence ({calibrated_conf:.2f}) was below domain threshold "
                f"({domain_policy.accept_confidence_threshold:.2f}) or contradiction was present."
            )
        if decision != "CORRECT":
            alternatives_rejected["CORRECT"] = "Rejected because no fixable, non-critical hallucinated claims were identified."
        if decision != "REJECT":
            alternatives_rejected["REJECT"] = "Rejected because contradiction index did not reach critical domain rejection threshold."
        if decision != "VERIFY_AGAIN":
            alternatives_rejected["VERIFY_AGAIN"] = "Rejected because evidence was either conclusive or entirely absent."

        return {
            "audit_id": audit_id,
            "timestamp": timestamp,
            "user_query": user_query,
            "draft_response": draft_response,
            "final_decision": decision,
            "severity": severity,
            "calibrated_confidence": calibrated_conf,
            "domain_policy": {
                "name": domain_policy.domain_name,
                "strictness": domain_policy.strictness_level,
                "accept_threshold": domain_policy.accept_confidence_threshold
            },
            "evidence_metrics": {
                "authority_score": evidence_intel.get("overall_authority", 0.0),
                "freshness_score": evidence_intel.get("overall_freshness", 0.0),
                "diversity_index": evidence_intel.get("diversity_index", 0.0),
                "completeness": evidence_intel.get("evidence_completeness", "N/A")
            },
            "consensus_metrics": {
                "consensus_score": consensus_data.get("consensus_score", 0.0),
                "conflict_index": consensus_data.get("conflict_index", 0.0),
                "level": consensus_data.get("consensus_level", "N/A")
            },
            "contradiction_taxonomy": contradiction_data.get("taxonomy_type", "NONE"),
            "reasoning_chain": reasoning_chain,
            "alternatives_rejected_rationale": alternatives_rejected
        }
