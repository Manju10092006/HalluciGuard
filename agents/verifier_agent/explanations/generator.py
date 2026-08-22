from __future__ import annotations
from typing import Dict, Any, List
from schemas.models import EntailmentLabel, VerdictLabel, EvidenceItem


class ExplanationGenerator:
    """Generates faithful human-readable natural language explanations for verification results."""

    @staticmethod
    def _format_source_name(raw_source: str) -> str:
        """Format raw source strings (like tavily:domain.com or wikipedia) into clean display names."""
        if not raw_source:
            return "Unknown"
        if raw_source.startswith("tavily:"):
            domain = raw_source.split(":", 1)[1]
            return f"web source ({domain})"
        if raw_source == "wikipedia":
            return "Wikipedia"
        if raw_source == "pubmed":
            return "PubMed"
        if raw_source == "openfda":
            return "OpenFDA"
        if raw_source == "nvd":
            return "NVD"
        if raw_source == "sec_edgar":
            return "SEC EDGAR"
        if raw_source == "clinical_trials":
            return "ClinicalTrials.gov"
        return raw_source

    def generate(
        self,
        claim_text: str,
        evidence_items: List[Any],
        verdict: str | VerdictLabel,
        scores: Dict[str, float],
        conflict_resolution: Dict[str, Any],
    ) -> str:
        """
        Generate a human-readable explanation paragraph based on observable reasoning.
        """
        if not evidence_items:
            return "No supporting or contradicting evidence was found from any authoritative source."

        total_evidence = len(evidence_items)

        supports = []
        contradicts = []
        neutrals = []

        for e in evidence_items:
            label = getattr(e, "entailment_label", None) or (
                e.get("entailment_label") if isinstance(e, dict) else None
            )
            if label in (EntailmentLabel.ENTAILMENT, "entailment", "supports"):
                supports.append(e)
            elif label in (EntailmentLabel.CONTRADICTION, "contradiction", "contradicts"):
                contradicts.append(e)
            else:
                neutrals.append(e)

        def get_cred(item: Any) -> float:
            if isinstance(item, dict):
                return float(
                    item.get("credibility_score", item.get("source_credibility", 0.5))
                )
            return float(getattr(item, "credibility_score", 0.5))

        most_credible = max(evidence_items, key=get_cred)

        if isinstance(most_credible, dict):
            raw_source = most_credible.get(
                "source", most_credible.get("source_name", "Unknown")
            )
            credibility = get_cred(most_credible)
            snippet = most_credible.get("snippet", "")
            pub_date = most_credible.get("publication_date", "Unknown date")
        else:
            raw_source = getattr(most_credible, "source", "Unknown")
            credibility = get_cred(most_credible)
            snippet = getattr(most_credible, "snippet", "")
            pub_date = getattr(most_credible, "publication_date", "Unknown date")

        source_name = self._format_source_name(raw_source)

        if len(snippet) > 180:
            snippet = snippet[:177] + "..."

        verdict_str = str(
            verdict.value if isinstance(verdict, VerdictLabel) else verdict
        )
        trust_sc = float(scores.get("trust_score", 0.0))

        if verdict_str == VerdictLabel.VERIFIED.value:
            src_count_str = f"{len(supports)} out of {total_evidence} sources support" if total_evidence > 1 else "1 source supports"
            explanation = (
                f"Verified ({trust_sc * 100:.1f}% trust score): {src_count_str} this claim. "
                f"The primary source ({source_name}, authority: {credibility:.2f}) "
                f'states: "{snippet}" Published {pub_date}.'
            )
        elif verdict_str == VerdictLabel.CONTRADICTED.value:
            contra_count_str = f"{len(contradicts)} contradicting sources" if len(contradicts) != 1 else "1 contradicting source"
            explanation = (
                f"Contradicted ({contra_count_str}): Authoritative evidence contradicts this claim. "
                f'The primary source ({source_name}, authority: {credibility:.2f}) states: "{snippet}" Published {pub_date}.'
            )
            if not supports:
                explanation += " No supporting evidence was found from any authoritative database."
        elif verdict_str == VerdictLabel.CONFLICTED.value:
            explanation = (
                f"Conflicted: Available evidence shows conflicting findings ({len(supports)} supporting vs {len(contradicts)} contradicting). "
                f'A primary source ({source_name}, authority: {credibility:.2f}) states: "{snippet}".'
            )
        else:
            explanation = (
                f"Unverified: Evaluated {total_evidence} evidence item{'s' if total_evidence != 1 else ''} from {source_name}, "
                f"but current evidence remains inconclusive or neutral regarding the specific claim."
            )

        # Include conflict resolution explanation if a genuine conflict was detected
        if (
            conflict_resolution
            and conflict_resolution.get("resolution_type") == "genuine_conflict"
        ):
            conflict_msg = conflict_resolution.get("explanation", "")
            if conflict_msg:
                explanation += f" Note: {conflict_msg}"

        return explanation.strip()
