"""
HalluciGuard - Claim Criticality Assessor
Evaluates claim importance and domain risk to assign criticality weights (0.0 to 1.0).
Prioritizes hallucination corrections based on real-world impact.
"""

import re
from typing import Dict, Any

class ClaimCriticalityAssessor:
    """Assesses the criticality level of claims."""
    def __init__(self):
        pass

    def evaluate_criticality(self, claim: str, domain: str = "General Knowledge") -> Dict[str, Any]:
        """
        Assesses criticality weight and category for a claim.
        """
        cl_lower = claim.lower()

        # 1. Critical Domain Keywords (Dosage, Safety, Security Exploit, Financial Revenue, Legal Liability)
        critical_keywords = [
            "dosage", "mg", "pediatric", "contraindicated", "side effect", "fatal", "treatment", "prescribed",
            "cve-", "remote code execution", "rce", "zero-day", "exploit", "vulnerability",
            "revenue", "net income", "sec 10-k", "bankruptcy", "ebitda",
            "statute", "unlawful", "penalty", "indictment"
        ]

        if any(kw in cl_lower for kw in critical_keywords) or domain in ["Healthcare", "Cybersecurity"]:
            if any(kw in cl_lower for kw in ["dosage", "contraindicated", "remote code execution", "cve-", "revenue"]):
                return {
                    "criticality_score": 0.95,
                    "criticality_tier": "CRITICAL_IMPACT",
                    "reason": "High-risk domain keyword detected (medical dosage, security exploit, or core financial metric)."
                }
            return {
                "criticality_score": 0.85,
                "criticality_tier": "HIGH_IMPACT",
                "reason": "Regulated domain claim with potential safety or financial consequences."
            }

        # 2. Medium Importance (Dates, Specifications, Technical parameters)
        medium_keywords = ["year", "version", "released", "percent", "growth", "founder", "author"]
        if any(kw in cl_lower for kw in medium_keywords):
            return {
                "criticality_score": 0.50,
                "criticality_tier": "MEDIUM_IMPACT",
                "reason": "Factual parameter (date, release version, or specification)."
            }

        # 3. Low / Minor Importance (Trivia, Birthdays, Casual facts)
        return {
            "criticality_score": 0.25,
            "criticality_tier": "LOW_IMPACT",
            "reason": "Informational or low-risk factual claim."
        }
