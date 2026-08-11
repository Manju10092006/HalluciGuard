"""
HalluciGuard - Consensus Analyzer
Evaluates whether multiple independent sources agree or disagree about claims.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ConsensusReport:
    has_consensus: bool
    agreeing_sources: int
    disagreeing_sources: int
    neutral_sources: int
    independent_source_count: int
    consensus_status: str  # STRONG_AGREEMENT, MAJORITY_AGREEMENT, MIXED_SIGNALS, DISAGREEMENT, INSUFFICIENT_SOURCES
    reasoning: str


class ConsensusAnalyzer:
    def evaluate_consensus(self, evaluated_pairs: List[Dict[str, Any]]) -> ConsensusReport:
        if not evaluated_pairs:
            return ConsensusReport(False, 0, 0, 0, 0, "INSUFFICIENT_SOURCES",
                                  "No evidence items to evaluate consensus.")

        agree = disagree = neutral = 0
        unique_sources = set()

        for pair in evaluated_pairs:
            src = pair.get("source", pair.get("evidence_source", "Unknown"))
            unique_sources.add(src.strip().lower())
            rel = pair.get("top_relation", "neutral")
            nli = pair.get("nli_scores", {})

            if rel == "entailment" or nli.get("entailment", 0) >= 0.55:
                agree += 1
            elif rel == "contradiction" or nli.get("contradiction", 0) >= 0.45:
                disagree += 1
            else:
                neutral += 1

        total = agree + disagree + neutral
        ind_count = len(unique_sources)

        if ind_count < 2:
            status = "INSUFFICIENT_SOURCES"
            reasoning = f"Only {ind_count} independent source(s) available. Consensus requires multiple independent providers."
            has = False
        elif disagree == 0 and agree >= 2:
            status = "STRONG_AGREEMENT"
            reasoning = f"All {agree} source(s) agree with the claim. No contradictions found across {ind_count} independent provider(s)."
            has = True
        elif agree > disagree and agree >= 1:
            status = "MAJORITY_AGREEMENT"
            reasoning = f"Majority agreement: {agree} supporting vs {disagree} contradicting across {ind_count} provider(s)."
            has = True
        elif disagree > agree:
            status = "DISAGREEMENT"
            reasoning = f"Sources disagree: {disagree} contradicting vs {agree} supporting. Evidence conflict detected."
            has = False
        else:
            status = "MIXED_SIGNALS"
            reasoning = f"Mixed signals: {agree} agree, {disagree} disagree, {neutral} neutral across {ind_count} provider(s)."
            has = False

        return ConsensusReport(
            has_consensus=has, agreeing_sources=agree, disagreeing_sources=disagree,
            neutral_sources=neutral, independent_source_count=ind_count,
            consensus_status=status, reasoning=reasoning
        )
