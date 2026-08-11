"""
HalluciGuard - Memory Intelligence Layer
Integration point for Memory Agent signals. Checks historical patterns,
source reliability history, recurring hallucination patterns.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class MemoryInsight:
    has_historical_context: bool
    known_hallucination_pattern: bool
    source_historically_reliable: bool
    previous_verification_outcome: Optional[str]
    memory_concerns: List[str]
    memory_recommendations: List[str]


class MemoryIntelligenceLayer:
    def query_memory(
        self,
        claim: str,
        sources: List[str],
        memory_context: Optional[Dict[str, Any]] = None
    ) -> MemoryInsight:
        if not memory_context:
            return MemoryInsight(
                has_historical_context=False, known_hallucination_pattern=False,
                source_historically_reliable=True, previous_verification_outcome=None,
                memory_concerns=[], memory_recommendations=["No Memory Agent context available; proceeding with current evidence only."]
            )

        concerns = []
        recommendations = []

        # Check for known hallucination patterns
        patterns = memory_context.get("known_hallucination_patterns", [])
        known_pattern = any(p.lower() in claim.lower() for p in patterns)
        if known_pattern:
            concerns.append("This claim matches a previously identified hallucination pattern from Memory Agent history.")
            recommendations.append("Apply heightened scrutiny. Consider requesting additional verification sources.")

        # Check source reliability history
        source_history = memory_context.get("source_reliability", {})
        unreliable = [s for s in sources if source_history.get(s, 1.0) < 0.5]
        if unreliable:
            concerns.append(f"Sources {unreliable} have been historically unreliable per Memory Agent records.")
            recommendations.append("Discount evidence from historically unreliable sources.")

        previous = memory_context.get("previous_outcome")

        return MemoryInsight(
            has_historical_context=True,
            known_hallucination_pattern=known_pattern,
            source_historically_reliable=len(unreliable) == 0,
            previous_verification_outcome=previous,
            memory_concerns=concerns,
            memory_recommendations=recommendations
        )
