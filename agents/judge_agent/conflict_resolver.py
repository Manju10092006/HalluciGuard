"""
HalluciGuard - Conflict Resolution Engine
Detects and classifies conflicts between claims and evidence.
Identifies specific conflict TYPES and their governance implications.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any
from config import ConflictType
from decision_policies import DomainPolicy


@dataclass
class ConflictReport:
    """Report of detected conflicts between claims and evidence."""
    has_conflict: bool
    conflict_type: ConflictType
    affected_claim: str
    conflicting_evidence: str
    implication: str
    is_safety_critical: bool
    requires_immediate_action: bool


class ConflictResolver:
    """Analyzes and resolves conflicts between claims and evidence."""
    def analyze_conflicts(
        self,
        evaluated_pairs: List[Dict[str, Any]],
        domain_policy: DomainPolicy
    ) -> List[ConflictReport]:
        """Analyze evidence-claim pairs for conflicts and return conflict reports."""
        reports = []
        for pair in evaluated_pairs:
            claim = pair.get("claim", "")
            evidence = pair.get("evidence", "")
            nli = pair.get("nli_scores", {})
            contra = nli.get("contradiction", 0.0)

            if contra < 0.30 or not evidence:
                reports.append(ConflictReport(
                    has_conflict=False, conflict_type=ConflictType.NO_CONFLICT,
                    affected_claim=claim, conflicting_evidence=evidence,
                    implication="No significant conflict detected.",
                    is_safety_critical=False, requires_immediate_action=False
                ))
                continue

            cl, ev = claim.lower(), evidence.lower()
            conflict_type, implication, safety, immediate = self._classify(claim, evidence, cl, ev, domain_policy)

            reports.append(ConflictReport(
                has_conflict=True, conflict_type=conflict_type,
                affected_claim=claim, conflicting_evidence=evidence,
                implication=implication, is_safety_critical=safety,
                requires_immediate_action=immediate
            ))
        return reports

    def _classify(self, claim, evidence, cl, ev, policy):
        # Safety violation check (drug contraindication, exploit)
        safety_kw = {"contraindicated", "prohibited", "fatal", "banned", "not recommended",
                      "denied", "unauthorized", "do not use", "dangerous"}
        critical_kw = set(policy.critical_claim_categories)

        ev_has_safety = any(w in ev for w in safety_kw)
        cl_has_critical = any(w in cl for w in critical_kw) if critical_kw else False

        if ev_has_safety and cl_has_critical:
            return (ConflictType.SAFETY_VIOLATION,
                    f"SAFETY: Evidence explicitly prohibits or contraindicates what the claim recommends. Domain: {policy.domain_name}.",
                    True, True)
        if ev_has_safety:
            return (ConflictType.DIRECT_REFUTATION,
                    f"Evidence contains explicit refutation language that directly contradicts the claim.",
                    policy.escalate_on_safety_conflict, policy.escalate_on_safety_conflict)

        # Numeric mismatch
        cl_nums = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', claim))
        ev_nums = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', evidence))
        if cl_nums and ev_nums and not cl_nums & ev_nums:
            return (ConflictType.NUMERIC_MISMATCH,
                    f"Numeric values conflict: claim states {cl_nums} but evidence states {ev_nums}.",
                    False, False)

        # Temporal mismatch
        cl_years = set(re.findall(r'\b(19|20)\d{2}\b', claim))
        ev_years = set(re.findall(r'\b(19|20)\d{2}\b', evidence))
        if cl_years and ev_years and not cl_years & ev_years:
            return (ConflictType.TEMPORAL_MISMATCH,
                    f"Date/year conflict: claim references {cl_years} but evidence references {ev_years}.",
                    False, False)

        # Negation mismatch
        neg = {"not", "never", "no", "neither", "nor", "cannot"}
        ev_neg = any(w in ev.split() for w in neg)
        cl_neg = any(w in cl.split() for w in neg)
        if ev_neg != cl_neg:
            return (ConflictType.DIRECT_REFUTATION,
                    "Evidence and claim have opposing polarity (negation mismatch).",
                    False, False)

        return (ConflictType.PARTIAL_DISAGREEMENT,
                "Evidence partially disagrees with or does not fully support the claim.",
                False, False)
