"""
HalluciGuard - Enterprise Evidence Intelligence Engine
Performs multi-dimensional analysis on evidence items:
1. Source Authority Tier Rating
2. Domain-Aware Evidence Freshness Decay
3. Evidence Diversity Index across independent providers
4. Evidence Relationship Graph & Clustering (deduplication, primary vs secondary evidence)
"""

import math
import time
import re
from typing import Dict, List, Any, Tuple
from config import JudgeConfig, DEFAULT_CONFIG

class EvidenceIntelligenceEngine:
    """Intelligence engine for evidence analysis and scoring."""
    def __init__(self, config: JudgeConfig = DEFAULT_CONFIG):
        self.config = config

    def analyze_evidence_set(
        self,
        claim_evidence_pairs: List[Dict[str, Any]],
        domain: str = "General Knowledge"
    ) -> Dict[str, Any]:
        """
        Runs full Evidence Intelligence analysis over all claim-evidence pairs.
        Returns composite metrics for Authority, Freshness, Diversity, and Graph Topology.
        """
        if not claim_evidence_pairs:
            return {
                "overall_authority": 0.0,
                "overall_freshness": 0.0,
                "diversity_index": 0.0,
                "evidence_graph": {"nodes": [], "edges": [], "clusters": []},
                "evidence_completeness": "MISSING_ALL_EVIDENCE",
                "processed_pairs": []
            }

        analyzed_pairs = []
        authorities = []
        freshnesses = []
        sources = []

        for rank, pair in enumerate(claim_evidence_pairs, start=1):
            source_name = pair.get("source", pair.get("evidence_source", "Unknown"))
            pub_date = pair.get("publication_date", pair.get("date", None))
            evidence_text = pair.get("evidence", pair.get("evidence_snippet", ""))
            claim_text = pair.get("claim", "")

            # 1. Authority Rating
            auth_score, auth_tier = self.evaluate_source_authority(source_name)
            authorities.append(auth_score)
            sources.append(source_name)

            # 2. Freshness Decay
            fresh_score = self.evaluate_evidence_freshness(pub_date, domain)
            freshnesses.append(fresh_score)

            analyzed_pairs.append({
                **pair,
                "rank": rank,
                "authority_score": auth_score,
                "authority_tier": auth_tier,
                "freshness_score": fresh_score
            })

        # 3. Overall composite scores
        avg_authority = sum(authorities) / len(authorities) if authorities else 0.0
        avg_freshness = sum(freshnesses) / len(freshnesses) if freshnesses else 0.0
        diversity_idx = self.calculate_evidence_diversity(sources)

        # 4. Evidence Relationship Graph & Clustering
        graph_data = self.build_evidence_graph(analyzed_pairs)

        # 5. Evidence Completeness Rating
        completeness = self._assess_completeness(analyzed_pairs, avg_authority, diversity_idx)

        return {
            "overall_authority": round(avg_authority, 4),
            "overall_freshness": round(avg_freshness, 4),
            "diversity_index": round(diversity_idx, 4),
            "evidence_graph": graph_data,
            "evidence_completeness": completeness,
            "processed_pairs": analyzed_pairs
        }

    def evaluate_source_authority(self, source_name: str) -> Tuple[float, str]:
        """
        Determines authority score (0.0 to 1.0) and tier based on source name keywords.
        """
        src = source_name.lower()
        
        # Tier 1: Official / Government / Standard Organizations (1.0)
        t1_keywords = ["pubmed", "nih", "fda", "sec edgar", "mitre", "cve", "who", "cdc", "gov", "cochrane", "court acts", "statute"]
        if any(k in src for k in t1_keywords):
            return self.config.authority_tiers.OFFICIAL_GOVT, "OFFICIAL_GOVERNMENT"

        # Tier 2: Peer-Reviewed / Academic (0.90)
        t2_keywords = ["nature", "the lancet", "ieee", "arxiv", "journal", "academic", "university", "doi", "researchgate"]
        if any(k in src for k in t2_keywords):
            return self.config.authority_tiers.PEER_REVIEWED, "PEER_REVIEWED"

        # Tier 3: Enterprise Docs / Vendor Documentation (0.80)
        t3_keywords = ["enterprise pdf", "official documentation", "microsoft docs", "aws docs", "google Cloud docs", "vendor specification", "10-k", "annual report"]
        if any(k in src for k in t3_keywords):
            return self.config.authority_tiers.ENTERPRISE_DOCS, "ENTERPRISE_DOCUMENTATION"

        # Tier 4: Reputable News (0.65)
        t4_keywords = ["reuters", "bloomberg", "wall street journal", "bbc", "associated press", "financial times"]
        if any(k in src for k in t4_keywords):
            return self.config.authority_tiers.NEWS_REPUTABLE, "REPUTABLE_NEWS"

        # Tier 5: Community Content (0.40)
        t5_keywords = ["wikipedia", "stackoverflow", "github", "reddit", "forum", "wiki"]
        if any(k in src for k in t5_keywords):
            return self.config.authority_tiers.COMMUNITY_CONTENT, "COMMUNITY_CONTENT"

        # Tier 6: Unverified Blog / Unknown (0.20)
        return self.config.authority_tiers.UNVERIFIED_BLOG, "UNVERIFIED_SOURCE"

    def evaluate_evidence_freshness(self, pub_date: Any, domain: str) -> float:
        """
        Computes freshness decay using exponential half-life formula:
        S_fresh = exp(- ln(2) * age_in_days / half_life_days)
        """
        if not pub_date:
            return 0.85 # Neutral baseline when publication date is unspecified

        try:
            # Parse year/date if string
            current_year = 2026 # Reference current system year
            if isinstance(pub_date, int):
                age_days = max(0, (current_year - pub_date) * 365)
            elif isinstance(pub_date, str):
                match = re.search(r'\b(19|20)\d{2}\b', pub_date)
                if match:
                    year = int(match.group(0))
                    age_days = max(0, (current_year - year) * 365)
                else:
                    age_days = 180 # Default ~6 months
            else:
                age_days = 180

            half_life = self.config.freshness_decay.decay_half_life_days.get(domain, 365.0)
            decay_factor = math.exp(- (math.log(2) * age_days) / half_life)
            return round(max(0.10, min(1.0, decay_factor)), 4)
        except Exception:
            return 0.80

    def calculate_evidence_diversity(self, sources: List[str]) -> float:
        """
        Calculates Shannon Diversity Entropy normalized between 0.0 and 1.0
        based on distinct independent source domains.
        """
        if not sources:
            return 0.0
        
        # Extract unique domain categories
        categories = [self.evaluate_source_authority(s)[1] for s in sources]
        counts = {}
        for c in categories:
            counts[c] = counts.get(c, 0) + 1
        
        num_items = len(categories)
        if num_items <= 1 or len(counts) <= 1:
            return 0.25 if num_items > 0 else 0.0

        entropy = 0.0
        for count in counts.values():
            p = count / num_items
            entropy -= p * math.log2(p)
        
        max_entropy = math.log2(len(counts))
        normalized_diversity = entropy / max_entropy if max_entropy > 0 else 0.5
        
        # Reward multi-source coverage
        bonus = min(0.3, len(counts) * 0.1)
        return min(1.0, round(normalized_diversity + bonus, 4))

    def build_evidence_graph(self, analyzed_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Constructs an Evidence Relationship Graph clustering duplicate, primary vs secondary evidence.
        """
        nodes = []
        edges = []
        clusters = []

        for idx, pair in enumerate(analyzed_pairs):
            node_id = f"EVID_{idx + 1}"
            nodes.append({
                "id": node_id,
                "source": pair.get("source", "Unknown"),
                "authority_score": pair.get("authority_score", 0.5),
                "freshness_score": pair.get("freshness_score", 0.8),
                "snippet": pair.get("evidence", "")[:100] + "..."
            })

            # Check relationships with previous nodes
            for prev_idx, prev_node in enumerate(nodes[:-1]):
                prev_text = analyzed_pairs[prev_idx].get("evidence", "")
                curr_text = pair.get("evidence", "")
                
                # Jaccard text similarity
                s1 = set(prev_text.lower().split())
                s2 = set(curr_text.lower().split())
                overlap = len(s1.intersection(s2)) / max(1, len(s1.union(s2)))

                if overlap > 0.80:
                    edges.append({"source": prev_node["id"], "target": node_id, "relation": "DUPLICATE_DERIVED"})
                elif overlap > 0.40:
                    edges.append({"source": prev_node["id"], "target": node_id, "relation": "CORROBORATING"})

        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges
        }

    def _assess_completeness(self, analyzed_pairs: List[Dict[str, Any]], avg_auth: float, diversity: float) -> str:
        """
        Rates evidence completeness.
        """
        count = len(analyzed_pairs)
        if count >= 3 and avg_auth >= 0.75 and diversity >= 0.60:
            return "STRONG_MULTI_SOURCE_GROUNDING"
        elif count >= 2 and avg_auth >= 0.60:
            return "SUFFICIENT_GROUNDING"
        elif count >= 1:
            return "PARTIAL_INCOMPLETE_EVIDENCE"
        else:
            return "MISSING_CRITICAL_SOURCES"
