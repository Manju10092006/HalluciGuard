"""
HalluciGuard - Runtime Inspector
Inspects pipeline health BEFORE trusting verification results.
Checks if retrieval worked, APIs responded, NLI executed, evidence returned.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from config import SystemHealth


@dataclass
class RuntimeInspectionReport:
    system_health: SystemHealth
    retrieval_executed: bool
    nli_executed: bool
    evidence_returned: bool
    api_failures: List[str]
    degradation_warnings: List[str]
    is_pipeline_trustworthy: bool
    reasoning: str


class RuntimeInspector:
    def inspect_pipeline(
        self,
        detector_output: Dict[str, Any],
        verifier_output: Dict[str, Any],
        nli_results: List[Dict[str, Any]],
        execution_context: Optional[Dict[str, Any]] = None
    ) -> RuntimeInspectionReport:
        warnings = []
        failures = []

        # Check Detector output integrity
        det_prob = detector_output.get("hallucination_probability")
        det_conf = detector_output.get("confidence_score")
        if det_prob is None:
            warnings.append("Detector output missing 'hallucination_probability' field.")
        if det_conf is None:
            warnings.append("Detector output missing 'confidence_score' field.")

        # Check Verifier output
        pairs = verifier_output.get("claim_evidence_pairs", [])
        evidence_returned = False
        retrieval_executed = bool(pairs)

        for pair in pairs:
            ev = pair.get("evidence", "")
            if ev and ev.strip():
                evidence_returned = True
                break

        if not retrieval_executed:
            warnings.append("Verifier returned no claim-evidence pairs. Retrieval may have failed.")
        elif not evidence_returned:
            warnings.append("Verifier returned pairs but all evidence fields are empty.")

        # Check NLI results
        nli_executed = bool(nli_results)
        if not nli_executed:
            warnings.append("NLI inference produced no results.")

        # Check execution context for API failures
        if execution_context:
            for api, status in execution_context.get("api_statuses", {}).items():
                if status != "OK":
                    failures.append(f"External API '{api}' returned status '{status}'.")

        # Determine system health
        if failures:
            health = SystemHealth.PARTIAL_FAILURE
        elif len(warnings) >= 3:
            health = SystemHealth.DEGRADED
        elif not evidence_returned or not nli_executed:
            health = SystemHealth.DEGRADED
        else:
            health = SystemHealth.HEALTHY

        trustworthy = health in (SystemHealth.HEALTHY,)

        # Build reasoning
        if health == SystemHealth.HEALTHY:
            reasoning = "All pipeline components executed successfully. Detector, Verifier, and NLI produced valid outputs."
        elif health == SystemHealth.DEGRADED:
            reasoning = f"Pipeline is degraded: {'; '.join(warnings)}. Results should be treated with caution."
        else:
            reasoning = f"Pipeline experienced failures: {'; '.join(failures + warnings)}. Decision reliability is compromised."

        return RuntimeInspectionReport(
            system_health=health, retrieval_executed=retrieval_executed,
            nli_executed=nli_executed, evidence_returned=evidence_returned,
            api_failures=failures, degradation_warnings=warnings,
            is_pipeline_trustworthy=trustworthy, reasoning=reasoning
        )
