"""
HalluciGuard - Coverage Analyzer
Analyzes whether evidence adequately covers the claims. Identifies gaps.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class CoverageReport:
    claims_covered: int
    claims_uncovered: int
    claims_partially_covered: int
    total_claims: int
    coverage_status: str  # FULLY_COVERED, PARTIALLY_COVERED, INSUFFICIENTLY_COVERED, NO_COVERAGE
    uncovered_claims: List[str]
    coverage_concerns: List[str]


class CoverageAnalyzer:
    def analyze_coverage(self, evaluated_pairs: List[Dict[str, Any]]) -> CoverageReport:
        if not evaluated_pairs:
            return CoverageReport(0, 0, 0, 0, "NO_COVERAGE", [], ["No claim-evidence pairs to analyze."])

        covered = uncovered = partial = 0
        uncovered_claims = []
        concerns = []

        for pair in evaluated_pairs:
            claim = pair.get("claim", "")
            evidence = pair.get("evidence", "")
            rel = pair.get("top_relation", "neutral")

            if not evidence or not evidence.strip():
                uncovered += 1
                uncovered_claims.append(claim)
            elif rel == "entailment":
                covered += 1
            elif rel == "contradiction":
                uncovered += 1
                uncovered_claims.append(claim)
                concerns.append(f"Claim contradicted by evidence: '{claim[:80]}...'")
            else:
                partial += 1
                concerns.append(f"Claim has ambiguous evidence alignment: '{claim[:80]}...'")

        total = covered + uncovered + partial
        if total == 0:
            status = "NO_COVERAGE"
        elif uncovered == 0 and partial == 0:
            status = "FULLY_COVERED"
        elif covered > 0 and (uncovered + partial) <= covered:
            status = "PARTIALLY_COVERED"
        else:
            status = "INSUFFICIENTLY_COVERED"

        return CoverageReport(
            claims_covered=covered, claims_uncovered=uncovered,
            claims_partially_covered=partial, total_claims=total,
            coverage_status=status, uncovered_claims=uncovered_claims,
            coverage_concerns=concerns
        )
