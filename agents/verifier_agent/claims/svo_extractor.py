"""
Structured Subject–Relation–Object extraction for query expansion.

Generalizes across phrasing variants without claim-specific hard-coding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ClaimTriple:
    subject: str
    relation: str
    object: str
    qualifiers: List[str] = field(default_factory=list)


_CREATION_VERBS = (
    "created", "developed", "invented", "founded", "built", "designed",
    "written", "authored", "discovered", "introduced", "co-founded",
)
_KINSHIP_RELS = ("father", "mother", "parent", "son", "daughter", "uncle", "brother", "sister")
_STAR_VERBS = ("starred in", "acted in", "appeared in", "played in")


class SVOExtractor:
    """Extract structured triples from natural-language factual claims."""

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join((text or "").split())

    def extract(self, claim: str) -> Optional[ClaimTriple]:
        clean = self._clean(claim)
        if len(clean) < 5:
            return None

        triple = (
            self._passive_creation(clean)
            or self._active_creation(clean)
            or self._wh_question_creation(clean)
            or self._kinship(clean)
            or self._starring(clean)
            or self._capital(clean)
            or self._location(clean)
            or self._company_founder(clean)
        )
        return triple

    def _passive_creation(self, text: str) -> Optional[ClaimTriple]:
        m = re.search(
            rf"^(.+?)\s+(?:was|is|were)?\s*(?:originally\s+)?({'|'.join(_CREATION_VERBS)})\s+by\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if not m:
            return None
        return ClaimTriple(
            subject=self._clean(m.group(1)),
            relation="created_by",
            object=self._clean(m.group(3)),
            qualifiers=[m.group(2).lower()],
        )

    def _active_creation(self, text: str) -> Optional[ClaimTriple]:
        m = re.search(
            rf"^([A-Za-z0-9\s\-\.]+?)\s+({'|'.join(_CREATION_VERBS)})\s+(?:the\s+)?(.+)$",
            text,
            re.IGNORECASE,
        )
        if not m or any(w in m.group(1).lower() for w in ("was", "is", "were", "that", "which")):
            return None
        return ClaimTriple(
            subject=self._clean(m.group(3)),
            relation="created_by",
            object=self._clean(m.group(1)),
            qualifiers=[m.group(2).lower()],
        )

    def _wh_question_creation(self, text: str) -> Optional[ClaimTriple]:
        m = re.search(
            r"^who\s+(created|developed|invented|founded|built|designed)\s+(?:the\s+)?(.+)\??$",
            text,
            re.IGNORECASE,
        )
        if not m:
            return None
        return ClaimTriple(
            subject=self._clean(m.group(2)),
            relation="created_by",
            object="",
            qualifiers=["wh_question", m.group(1).lower()],
        )

    def _kinship(self, text: str) -> Optional[ClaimTriple]:
        m = re.search(
            rf"^(.+?)\s+is\s+(?:the\s+)?(?:maternal\s+|paternal\s+)?({'|'.join(_KINSHIP_RELS)})\s+of\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if not m:
            return None
        return ClaimTriple(
            subject=self._clean(m.group(1)),
            relation=f"{m.group(2).lower()}_of",
            object=self._clean(m.group(3)),
        )

    def _starring(self, text: str) -> Optional[ClaimTriple]:
        m = re.search(
            rf"^(.+?)\s+({'|'.join(_STAR_VERBS)})\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if not m:
            return None
        return ClaimTriple(
            subject=self._clean(m.group(1)),
            relation="starred_in",
            object=self._clean(m.group(3)),
        )

    def _capital(self, text: str) -> Optional[ClaimTriple]:
        m = re.search(
            r"^(.+?)\s+is\s+the\s+capital\s+of\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if not m:
            return None
        return ClaimTriple(
            subject=self._clean(m.group(1)),
            relation="capital_of",
            object=self._clean(m.group(2)),
        )

    def _location(self, text: str) -> Optional[ClaimTriple]:
        m = re.search(
            r"^(.+?)\s+is\s+(?:located\s+in|in)\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if not m:
            return None
        return ClaimTriple(
            subject=self._clean(m.group(1)),
            relation="location_of",
            object=self._clean(m.group(2)),
        )

    def _company_founder(self, text: str) -> Optional[ClaimTriple]:
        m = re.search(
            r"^(.+?)\s+(?:company|corporation|inc\.?|corp\.?)?\s*(?:was|is)\s+(?:built|founded|started|established)\s+by\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if not m:
            return None
        return ClaimTriple(
            subject=self._clean(m.group(1)),
            relation="created_by",
            object=self._clean(m.group(2)),
            qualifiers=["company"],
        )

    def generate_queries(self, claim: str) -> List[str]:
        """Generate retrieval queries from structured triple extraction."""
        clean = self._clean(claim)
        if not clean:
            return []

        queries: List[str] = [clean]
        triple = self.extract(clean)

        if triple is None:
            return queries

        subj, rel, obj = triple.subject, triple.relation, triple.object

        if rel == "created_by":
            if subj and obj:
                queries.extend([
                    f"{subj} creator",
                    f"who created {subj}",
                    f"{subj} created by",
                    f"{obj} {subj}",
                ])
            elif subj and "wh_question" in triple.qualifiers:
                queries.extend([
                    f"{subj} creator",
                    f"who created {subj}",
                    f"history of {subj} creators",
                ])
            if obj and subj:
                verb = triple.qualifiers[0] if triple.qualifiers else "created"
                queries.append(f"{obj} {verb} {subj}")

        elif rel.endswith("_of") and "starred_in" not in rel:
            if subj and obj:
                rel_word = rel.replace("_of", "")
                queries.extend([
                    f"{obj} {rel_word}",
                    f"{obj} family",
                    f"{obj} parents",
                ])

        elif rel == "starred_in":
            if subj and obj:
                queries.extend([
                    f"{obj} cast",
                    f"{obj} starring",
                    f"{obj} {subj}",
                    f"{subj} filmography",
                ])

        elif rel == "capital_of":
            if obj:
                queries.append(f"capital of {obj}")
            if subj:
                queries.append(f"{subj} capital city")

        elif rel == "location_of":
            if subj:
                queries.append(f"{subj} location")

        # Deduplicate preserving order
        seen: set[str] = set()
        unique: List[str] = []
        for q in queries:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(q.strip())
        return unique[:6]
