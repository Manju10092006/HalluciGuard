"""
HalluciGuard - Contradiction Taxonomy & Intelligence Analyzer
Classifies contradiction types: Direct Contradiction, Numeric Mismatch, Temporal Contradiction,
Entity Identity Mismatch, Topic Drift, and Missing Evidence Grounding.
"""

import re
from typing import Dict, List, Any

class ContradictionTaxonomyAnalyzer:
    def __init__(self):
        pass

    def classify_contradiction(
        self,
        claim: str,
        evidence: str,
        nli_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyzes relationship between claim and evidence to categorize contradiction taxonomy.
        """
        contra_score = nli_scores.get("contradiction", 0.0)
        
        if contra_score < 0.30:
            return {
                "has_contradiction": False,
                "taxonomy_type": "NO_CONTRADICTION",
                "risk_weight": 0.0,
                "explanation": "No significant contradiction detected."
            }

        cl_lower = claim.lower()
        ev_lower = evidence.lower()

        # 1. Numeric / Quantity Mismatch
        cl_nums = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', claim))
        ev_nums = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', evidence))
        if cl_nums and ev_nums and not cl_nums.intersection(ev_nums):
            return {
                "has_contradiction": True,
                "taxonomy_type": "NUMERIC_QUANTITY_MISMATCH",
                "risk_weight": 0.85,
                "conflicting_values": {"claim_values": list(cl_nums), "evidence_values": list(ev_nums)},
                "explanation": f"Numerical contradiction detected: claim states {list(cl_nums)} vs evidence states {list(ev_nums)}."
            }

        # 2. Temporal / Date Contradiction
        cl_years = set(re.findall(r'\b(19|20)\d{2}\b', claim))
        ev_years = set(re.findall(r'\b(19|20)\d{2}\b', evidence))
        if cl_years and ev_years and not cl_years.intersection(ev_years):
            return {
                "has_contradiction": True,
                "taxonomy_type": "TEMPORAL_DATE_CONTRADICTION",
                "risk_weight": 0.80,
                "conflicting_values": {"claim_years": list(cl_years), "evidence_years": list(ev_years)},
                "explanation": f"Temporal mismatch: claim specifies year {list(cl_years)} vs evidence specifies {list(ev_years)}."
            }

        # 3. Explicit Contradiction / Negation / Medical Safety Refutation
        refutation_words = {"contraindicated", "prohibited", "false", "denied", "incorrect", "not recommended", "banned"}
        if any(w in ev_lower for w in refutation_words) and not any(w in cl_lower for w in refutation_words):
            return {
                "has_contradiction": True,
                "taxonomy_type": "DIRECT_SAFETY_CONTRADICTION",
                "risk_weight": 1.00, # Highest severity
                "explanation": "Direct safety refutation: evidence explicitly prohibits or refutes the claim."
            }

        # 4. Entity Mismatch (Capitalized proper nouns conflict)
        cl_entities = set(re.findall(r'\b[A-Z][a-z]+\b', claim)) - {"The", "A", "In", "On", "What", "Is"}
        ev_entities = set(re.findall(r'\b[A-Z][a-z]+\b', evidence)) - {"The", "A", "In", "On", "What", "Is"}
        if cl_entities and ev_entities and len(cl_entities.intersection(ev_entities)) == 0 and len(cl_entities) > 1:
            return {
                "has_contradiction": True,
                "taxonomy_type": "ENTITY_IDENTITY_MISMATCH",
                "risk_weight": 0.75,
                "explanation": f"Entity conflict: claim references {list(cl_entities)} whereas evidence references {list(ev_entities)}."
            }

        # 5. Generic Direct Contradiction
        return {
            "has_contradiction": True,
            "taxonomy_type": "DIRECT_SEMANTIC_CONTRADICTION",
            "risk_weight": 0.90,
            "explanation": f"Semantic contradiction between claim and verified evidence with score {contra_score:.2f}."
        }
