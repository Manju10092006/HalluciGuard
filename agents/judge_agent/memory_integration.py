"""
HalluciGuard - Memory Intelligence Integration Module
Interfaces with Memory Agent signals to retrieve entity accuracy records, historical source reliability,
and past hallucination patterns to adjust calibration scores.
"""

from typing import Dict, List, Any, Optional

class MemoryIntelligenceEngine:
    def __init__(self):
        # In-memory historical cache for demonstration & production lookup
        self._source_reliability_history: Dict[str, float] = {
            "PubMed": 0.98,
            "NIH": 0.98,
            "FDA": 0.96,
            "SEC EDGAR": 0.97,
            "MITRE": 0.95,
            "Wikipedia": 0.70,
            "Unverified Blog": 0.35
        }

    def evaluate_memory_signals(
        self,
        claim: str,
        sources: List[str],
        memory_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates memory context signals and computes memory reliability multiplier.
        """
        if not memory_context:
            memory_context = {}

        # 1. Source Reliability Lookup from Memory
        reliabilities = []
        for s in sources:
            rel = memory_context.get("source_reliability", {}).get(s, self._source_reliability_history.get(s, 0.75))
            reliabilities.append(rel)

        avg_source_reliability = sum(reliabilities) / len(reliabilities) if reliabilities else 0.75

        # 2. Historical Hallucination Pattern Match
        past_patterns = memory_context.get("historical_patterns", [])
        has_recurring_hallucination = any(p.lower() in claim.lower() for p in past_patterns)

        # 3. Overall Memory Trust Index
        if has_recurring_hallucination:
            memory_trust_score = round(avg_source_reliability * 0.50, 4)
            note = "Known recurring hallucination pattern matched from Memory Agent history."
        else:
            memory_trust_score = round(avg_source_reliability, 4)
            note = "Historical memory signals indicate consistent entity accuracy."

        return {
            "memory_trust_score": memory_trust_score,
            "has_recurring_hallucination_pattern": has_recurring_hallucination,
            "historical_source_reliability": round(avg_source_reliability, 4),
            "memory_note": note
        }
