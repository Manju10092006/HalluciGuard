"""
HalluciGuard - Source Consensus Engine
Measures independent source agreement, calculates consensus matrix, and determines conflict indices.
"""

from typing import Dict, List, Any

class SourceConsensusEngine:
    """Engine for evaluating consensus across multiple sources."""
    def __init__(self):
        pass

    def evaluate_consensus(self, evaluated_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates agreement across independent evidence items.
        Returns consensus_score, conflict_index, and consensus_level.
        """
        if not evaluated_pairs:
            return {
                "consensus_score": 0.0,
                "conflict_index": 0.0,
                "consensus_level": "NO_EVIDENCE",
                "independent_source_count": 0,
                "consensus_matrix": []
            }

        entail_count = 0
        contradict_count = 0
        neutral_count = 0
        unique_sources = set()

        matrix = []

        for pair in evaluated_pairs:
            source = pair.get("source", pair.get("evidence_source", "Unknown"))
            unique_sources.add(source)
            
            top_rel = pair.get("top_relation", "neutral")
            nli = pair.get("nli_scores", {})
            entail_prob = nli.get("entailment", 0.0)
            contra_prob = nli.get("contradiction", 0.0)

            if top_rel == "entailment" or entail_prob >= 0.55:
                entail_count += 1
                status = "AGREES"
            elif top_rel == "contradiction" or contra_prob >= 0.45:
                contradict_count += 1
                status = "CONTRADICTS"
            else:
                neutral_count += 1
                status = "NEUTRAL"

            matrix.append({
                "source": source,
                "claim": pair.get("claim", ""),
                "status": status,
                "nli_scores": nli
            })

        total = len(evaluated_pairs)
        independent_count = len(unique_sources)

        if total == 0:
            consensus_score = 0.0
            conflict_index = 0.0
        else:
            agreement_ratio = entail_count / total
            conflict_index = round(contradict_count / total, 4)

            # Consensus incorporates multi-source amplification
            multi_source_bonus = min(0.20, (independent_count - 1) * 0.05) if agreement_ratio > 0.5 else 0.0
            consensus_score = round(min(1.0, agreement_ratio + multi_source_bonus - conflict_index * 0.3), 4)

        # Categorize consensus level
        if consensus_score >= 0.80 and independent_count >= 2:
            level = "VERY_HIGH_CONSENSUS"
        elif consensus_score >= 0.60:
            level = "HIGH_CONSENSUS"
        elif conflict_index >= 0.40:
            level = "CRITICAL_SOURCE_CONFLICT"
        elif consensus_score >= 0.40:
            level = "MODERATE_CONSENSUS"
        else:
            level = "WEAK_CONSENSUS"

        return {
            "consensus_score": max(0.0, consensus_score),
            "conflict_index": max(0.0, conflict_index),
            "consensus_level": level,
            "independent_source_count": independent_count,
            "agreeing_source_count": entail_count,
            "contradicting_source_count": contradict_count,
            "consensus_matrix": matrix
        }
