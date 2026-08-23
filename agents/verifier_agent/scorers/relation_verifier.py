"""
HalluciGuard Verifier Agent — Relation Verification Layer.

Extracts structured (subject, relation, object) triples from claims and evidence passages,
and compares them to detect direct relational matches, object mismatches, or relation mismatches.

Supported relation types:
  - capital_of (e.g. Hyderabad / Telangana vs India)
  - location_of / located_in (e.g. Eiffel Tower / Paris vs London)
  - parent_of / father_of / mother_of / kinship (e.g. Allu Arjun / Allu Aravind vs Chiranjeevi; uncle vs father)
  - created_by / invented_by / founded_by / developed_by (e.g. Java / James Gosling vs Gaurav; Amazon / Bezos vs Pichai)
  - vulnerability_of / associated_with (e.g. CVE-2021-44228 / Log4Shell)
"""
from __future__ import annotations

import re
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field


class Triple(BaseModel):
    subject: str
    relation: str
    object: str
    qualifiers: List[str] = Field(default_factory=list)
    negated: bool = False
    raw_text: str = ""


class RelationCheckResult(BaseModel):
    claim_triple: Optional[Triple] = None
    evidence_triples: List[Triple] = Field(default_factory=list)
    status: str = "NO_TRIPLE_EXTRACTED"  # MATCH, OBJECT_MISMATCH, RELATION_MISMATCH, NO_TRIPLE_EXTRACTED
    mismatch_detail: str = ""
    combination_rule_applied: str = ""


class RelationVerifier:
    """
    Lightweight, deterministic relation extraction and verification engine.
    Runs in 0ms with zero network dependencies.
    """

    KINSHIP_HIERARCHY = {
        "father_of": "parent",
        "mother_of": "parent",
        "parent_of": "parent",
        "uncle_of": "uncle",
        "brother_of": "sibling",
        "sister_of": "sibling",
    }

    @staticmethod
    def _clean_str(s: str) -> str:
        s = re.sub(r"[^\w\s\-]", " ", s)
        return " ".join(s.lower().split())

    def _normalize_name(self, name: str) -> str:
        clean = self._clean_str(name)
        # Remove common descriptors, articles, professions
        stopwords = {
            "the", "a", "an", "mr", "mrs", "dr", "sir", "actor", "actress",
            "producer", "director", "film", "engineer", "city", "state",
            "company", "corporation", "inc", "ltd", "tech", "technology",
            "country", "nation", "indian", "french", "american"
        }
        words = [w for w in clean.split() if w not in stopwords]
        return " ".join(words) if words else clean

    def extract_triples(self, text: str) -> List[Triple]:
        """Extract all candidate (subject, relation, object) triples from a text."""
        triples: List[Triple] = []
        if not text or len(text.strip()) < 5:
            return triples

        # Normalize sentence breaks
        sentences = re.split(r"[.!?\n]+", text)
        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue

            # ── 1. Capital Relations ──────────────────────────────────
            # e.g., "Hyderabad is the capital ... of the Indian state of Telangana"
            # e.g., "Paris is the capital of France"
            cap_match = re.search(
                r"([A-Za-z\s\-]+?)\s+(?:is|was|serves as|became)\s+(?:the\s+)?(?:state\s+|national\s+)?capital(?:\s+and\s+[\w\s]+?)?\s+(?:city\s+)?(?:of|for)\s+(?:the\s+)?(?:([a-z\s]+)\s+state\s+of\s+)?([A-Za-z\s\-]+)",
                sent_clean,
                re.IGNORECASE,
            )
            if cap_match:
                subj = cap_match.group(1).strip()
                qualifier_state = (cap_match.group(2) or "").strip()
                obj = cap_match.group(3).strip()
                obj = re.split(r"\b(occupies|in|along|with|and|on|which|where|predating)\b", obj, flags=re.IGNORECASE)[0].strip()
                qualifiers = [qualifier_state] if qualifier_state else []
                if "state of" in sent_clean.lower():
                    qualifiers.append("state_level")
                if "country" in sent_clean.lower() or "national capital" in sent_clean.lower():
                    qualifiers.append("national_level")

                triples.append(
                    Triple(
                        subject=self._normalize_name(subj),
                        relation="capital_of",
                        object=self._normalize_name(obj),
                        qualifiers=qualifiers,
                        raw_text=sent_clean,
                    )
                )

            # Inverted capital pattern, e.g. "Kingdom of France: The traditional capital was Paris..."
            cap_colon = re.search(
                r"([A-Za-z\s\-]+?)\s*:\s*(?:the\s+)?(?:traditional\s+|official\s+|national\s+|administrative\s+)?capital\s+(?:is|was|became|remains)\s+([A-Za-z\s\-]+)",
                sent_clean,
                re.IGNORECASE,
            )
            cap_inverted = re.search(
                r"(?:the\s+)?(?:traditional\s+|official\s+|national\s+|administrative\s+)?capital\s+(?:(?:city\s+)?(?:of|for)\s+([A-Za-z\s\-]+?)\s+)?(?:is|was|became|remains)\s+([A-Za-z\s\-]+)",
                sent_clean,
                re.IGNORECASE,
            )
            if cap_colon and not cap_match:
                country_entity = cap_colon.group(1).strip()
                city_entity = cap_colon.group(2).strip()
                city_entity = re.split(r"\b(though|from|in|along|with|and|on|which|where|predating)\b", city_entity, flags=re.IGNORECASE)[0].strip()
                if city_entity and len(city_entity) > 2 and country_entity:
                    triples.append(
                        Triple(
                            subject=self._normalize_name(city_entity),
                            relation="capital_of",
                            object=self._normalize_name(country_entity),
                            qualifiers=["historical_or_traditional"],
                            raw_text=sent_clean,
                        )
                    )
            elif cap_inverted and not cap_match:
                country_entity = (cap_inverted.group(1) or "").strip()
                city_entity = cap_inverted.group(2).strip()
                city_entity = re.split(r"\b(though|from|in|along|with|and|on|which|where|predating)\b", city_entity, flags=re.IGNORECASE)[0].strip()
                if city_entity and len(city_entity) > 2 and country_entity:
                    triples.append(
                        Triple(
                            subject=self._normalize_name(city_entity),
                            relation="capital_of",
                            object=self._normalize_name(country_entity),
                            qualifiers=["historical_or_traditional"],
                            raw_text=sent_clean,
                        )
                    )



            # ── 2. Location Relations ─────────────────────────────────
            # e.g., "The Eiffel Tower is located in London"
            # e.g., "The Eiffel Tower is a lattice tower on the Champ de Mars in Paris, France"
            # e.g., "Texas's Eiffel Tower is a landmark in the city of Paris, Texas"
            loc_match = re.search(
                r"([A-Za-z0-9\s\'\-]+?)\s+(?:is|was|are|stands)?\s*(?:located\s+in|situated\s+in|a\s+landmark\s+in|on\s+the\s+[\w\s]+\s+in)\s+(?:the\s+(?:city|heart|centre|center)\s+of\s+)?([A-Za-z\s\-]+)",
                sent_clean,
                re.IGNORECASE,
            )
            if loc_match:
                subj = loc_match.group(1).strip()
                obj = loc_match.group(2).strip()
                obj = re.split(r"\b(and|with|which|where|at|from|is|whose)\b", obj, flags=re.IGNORECASE)[0].strip()
                triples.append(
                    Triple(
                        subject=self._normalize_name(subj),
                        relation="location_of",
                        object=self._normalize_name(obj),
                        raw_text=sent_clean,
                    )
                )

            # ── 3. Kinship / Family Relations ─────────────────────────
            # e.g. "Chiranjeevi is the father of Allu Arjun"
            # e.g. "Allu Arjun was born ... to film producer Allu Aravind and Nirmala"
            # e.g. "Chiranjeevi is the maternal/paternal uncle of Allu Arjun"
            kin_direct = re.search(
                r"([A-Za-z\s\-]+?)\s+(?:is|was)\s+(?:the\s+)?(?:maternal\s+|paternal\s+)?(father|mother|parent|uncle|brother|sister|son|daughter)\s+of\s+([A-Za-z\s\-]+)",
                sent_clean,
                re.IGNORECASE,
            )
            if kin_direct:
                subj = kin_direct.group(1).strip()
                rel_word = kin_direct.group(2).lower().strip() + "_of"
                obj = kin_direct.group(3).strip()
                obj = re.split(r"\b(and|with|which|where|who|in|born)\b", obj, flags=re.IGNORECASE)[0].strip()
                triples.append(
                    Triple(
                        subject=self._normalize_name(subj),
                        relation=rel_word,
                        object=self._normalize_name(obj),
                        raw_text=sent_clean,
                    )
                )

            born_match = re.search(
                r"([A-Za-z\s\-]+?)\s+was\s+born\s+(?:on\s+[\w\s\d,]+\s+)?(?:in\s+(?:a\s+)?[\w\s]+\s+family\s+)?to\s+(?:(?:film\s+)?(?:producer|actor|director|musician|doctor|writer|engineer)\s+)?([A-Za-z\s\-]+?)(?:\s+and\s+([A-Za-z\s\-]+?))?(?:\s+in\b|\.|\,|$)",
                sent_clean,
                re.IGNORECASE,
            )
            if born_match:
                child = born_match.group(1).strip()
                parent1 = born_match.group(2).strip()
                parent2 = (born_match.group(3) or "").strip()
                if parent1:
                    triples.append(
                        Triple(
                            subject=self._normalize_name(parent1),
                            relation="father_of",
                            object=self._normalize_name(child),
                            qualifiers=["parent"],
                            raw_text=sent_clean,
                        )
                    )
                if parent2:
                    triples.append(
                        Triple(
                            subject=self._normalize_name(parent2),
                            relation="mother_of",
                            object=self._normalize_name(child),
                            qualifiers=["parent"],
                            raw_text=sent_clean,
                        )
                    )

            # ── 4. Creation / Invention / Authorship Relations ────────
            # e.g., "Java was created by James Gosling"
            # e.g., "Amazon company was built by Sundar Pichai"
            # e.g., "Python was created by Guido van Rossum"
            create_match = re.search(
                r"([A-Za-z0-9\s\-]+?)\s+(?:was|is|were)?\s*(?:originally\s+)?(created|developed|invented|built|designed|founded|introduced)\s+by\s+([A-Za-z0-9\s\-]+)",
                sent_clean,
                re.IGNORECASE,
            )
            if create_match:
                subj = create_match.group(1).strip()
                verb = create_match.group(2).lower()
                obj = create_match.group(3).strip()
                obj = re.split(r"\b(in|at|and|with|which|for|as|on)\b", obj, flags=re.IGNORECASE)[0].strip()
                triples.append(
                    Triple(
                        subject=self._normalize_name(subj),
                        relation="created_by",
                        object=self._normalize_name(obj),
                        qualifiers=[verb],
                        raw_text=sent_clean,
                    )
                )

            # Direct subject-verb-object: "James Gosling created Java"
            svo_match = re.search(
                r"([A-Za-z\s\-]+?)\s+(created|developed|invented|founded|built)\s+([A-Za-z0-9\s\-]+)",
                sent_clean,
                re.IGNORECASE,
            )
            if svo_match and not any(w in svo_match.group(1).lower() for w in ("was", "is", "were", "that", "which")):
                creator = svo_match.group(1).strip()
                verb = svo_match.group(2).lower()
                creation = svo_match.group(3).strip()
                creation = re.split(r"\b(in|at|and|with|which|for|as|on)\b", creation, flags=re.IGNORECASE)[0].strip()
                triples.append(
                    Triple(
                        subject=self._normalize_name(creation),
                        relation="created_by",
                        object=self._normalize_name(creator),
                        qualifiers=[verb],
                        raw_text=sent_clean,
                    )
                )

            # ── 5. Cybersecurity Vulnerability / Association ──────────
            # e.g., "CVE-2021-44228 is associated with Log4Shell"
            cve_match = re.search(
                r"(cve-\d{4}-\d{4,8})\s+(?:is\s+)?(?:associated\s+with|related\s+to|known\s+as|alias\s+for|affects)\s+([A-Za-z0-9\s\-]+)",
                sent_clean,
                re.IGNORECASE,
            )
            if cve_match:
                cve_id = cve_match.group(1).upper()
                vuln_name = cve_match.group(2).strip()
                vuln_name = re.split(r"\b(in|and|which|reported|published)\b", vuln_name, flags=re.IGNORECASE)[0].strip()
                triples.append(
                    Triple(
                        subject=cve_id.lower(),
                        relation="associated_with",
                        object=self._normalize_name(vuln_name),
                        raw_text=sent_clean,
                    )
                )

            # ── 6. Entity Classification / Type Relation ("X is a Y") ──
            # e.g., "HTML is a programming language"
            # e.g., "HTML is the standard markup language"
            isa_match = re.search(
                r"^([A-Za-z0-9\s\-]+?)\s+(?:is|was)\s+(?:the\s+|a\s+|an\s+)?(?:standard\s+)?([A-Za-z0-9\s\-]+?\s+(?:language|protocol|database|operating system|framework|library|algorithm|disease|medication|drug|planet|element))\b",
                sent_clean,
                re.IGNORECASE,
            )
            if isa_match:
                subj = isa_match.group(1).strip()
                obj = isa_match.group(2).strip()
                triples.append(
                    Triple(
                        subject=self._normalize_name(subj),
                        relation="is_a",
                        object=self._normalize_name(obj),
                        raw_text=sent_clean,
                    )
                )

        return triples

    def _names_match(self, name1: str, name2: str) -> bool:
        """Check if two entity names match (exact, substring, or token overlap)."""
        n1 = self._normalize_name(name1)
        n2 = self._normalize_name(name2)
        if not n1 or not n2:
            return False
        if n1 == n2 or n1 in n2 or n2 in n1:
            return True
        t1 = set(n1.split())
        t2 = set(n2.split())
        if t1 and t2 and (t1.issubset(t2) or t2.issubset(t1)):
            return True
        if len(t1) >= 2 and len(t2) >= 2 and len(t1.intersection(t2)) >= min(len(t1), len(t2)):
            return True
        return False

    def verify_relation(
        self,
        claim_text: str,
        evidence_passages: List[Any],
    ) -> RelationCheckResult:
        """
        Extract triples from the claim and all evidence passages,
        and perform relational consistency checks.
        """
        claim_triples = self.extract_triples(claim_text)
        if not claim_triples:
            return RelationCheckResult(
                status="NO_TRIPLE_EXTRACTED",
                mismatch_detail="No structured relation triple recognized in claim",
            )

        c_triple = claim_triples[0]
        all_evidence_triples: List[Triple] = []

        for p in evidence_passages:
            text = f"{getattr(p, 'title', '')} {getattr(p, 'snippet', '')}"
            e_triples = self.extract_triples(text)
            all_evidence_triples.extend(e_triples)

        if not all_evidence_triples:
            return RelationCheckResult(
                claim_triple=c_triple,
                evidence_triples=[],
                status="NO_TRIPLE_EXTRACTED",
                mismatch_detail="No structured relation triples recognized in retrieved evidence",
            )

        # Compare claim triple against candidate evidence triples
        for e_triple in all_evidence_triples:
            # Check for subject alignment or reverse alignment
            subj_match = self._names_match(c_triple.subject, e_triple.subject)
            obj_match = self._names_match(c_triple.object, e_triple.object)

            # ── 1. Capital Relation Check ─────────────────────────────
            if c_triple.relation == "capital_of" and e_triple.relation == "capital_of":
                if subj_match:
                    if "state_level" in e_triple.qualifiers and not self._names_match(c_triple.object, e_triple.object):
                        return RelationCheckResult(
                            claim_triple=c_triple,
                            evidence_triples=all_evidence_triples,
                            status="OBJECT_MISMATCH",
                            mismatch_detail=f"Claim asserts '{c_triple.subject}' is capital of '{c_triple.object}', but authoritative evidence proves it is the capital of the state of '{e_triple.object}'",
                            combination_rule_applied="BYPASS_SUPPRESSION_FORCE_CONTRADICTION",
                        )
                    elif self._names_match(c_triple.object, e_triple.object):
                        return RelationCheckResult(
                            claim_triple=c_triple,
                            evidence_triples=all_evidence_triples,
                            status="MATCH",
                            combination_rule_applied="CONFIRM_ENTAILMENT",
                        )
                    else:
                        return RelationCheckResult(
                            claim_triple=c_triple,
                            evidence_triples=all_evidence_triples,
                            status="OBJECT_MISMATCH",
                            mismatch_detail=f"Claim asserts capital of '{c_triple.object}', but evidence proves capital of '{e_triple.object}'",
                            combination_rule_applied="BYPASS_SUPPRESSION_FORCE_CONTRADICTION",
                        )

            # ── 2. Location Relation Check ────────────────────────────
            if c_triple.relation == "location_of" and e_triple.relation == "location_of":
                if subj_match:
                    if self._names_match(c_triple.object, e_triple.object):
                        return RelationCheckResult(
                            claim_triple=c_triple,
                            evidence_triples=all_evidence_triples,
                            status="MATCH",
                            combination_rule_applied="CONFIRM_ENTAILMENT",
                        )
                    else:
                        return RelationCheckResult(
                            claim_triple=c_triple,
                            evidence_triples=all_evidence_triples,
                            status="OBJECT_MISMATCH",
                            mismatch_detail=f"Claim asserts location in '{c_triple.object}', but authoritative evidence proves location in '{e_triple.object}'",
                            combination_rule_applied="BYPASS_SUPPRESSION_FORCE_CONTRADICTION",
                        )

            # ── 3. Kinship / Family Relation Check ────────────────────
            if "of" in c_triple.relation and "of" in e_triple.relation:
                c_kin = self.KINSHIP_HIERARCHY.get(c_triple.relation)
                e_kin = self.KINSHIP_HIERARCHY.get(e_triple.relation)

                if c_kin and e_kin:
                    # e.g. claim: Chiranjeevi father_of Allu Arjun
                    # e_triple: Allu Aravind father_of Allu Arjun (same object 'allu arjun')
                    if self._names_match(c_triple.object, e_triple.object):
                        if c_kin == e_kin:
                            if not self._names_match(c_triple.subject, e_triple.subject):
                                return RelationCheckResult(
                                    claim_triple=c_triple,
                                    evidence_triples=all_evidence_triples,
                                    status="OBJECT_MISMATCH",
                                    mismatch_detail=f"Claim asserts father/parent is '{c_triple.subject}', but evidence proves father/parent is '{e_triple.subject}'",
                                    combination_rule_applied="BYPASS_SUPPRESSION_FORCE_CONTRADICTION",
                                )
                            else:
                                return RelationCheckResult(
                                    claim_triple=c_triple,
                                    evidence_triples=all_evidence_triples,
                                    status="MATCH",
                                    combination_rule_applied="CONFIRM_ENTAILMENT",
                                )
                        else:
                            # e.g. Chiranjeevi uncle_of Allu Arjun vs father_of
                            if self._names_match(c_triple.subject, e_triple.subject):
                                return RelationCheckResult(
                                    claim_triple=c_triple,
                                    evidence_triples=all_evidence_triples,
                                    status="RELATION_MISMATCH",
                                    mismatch_detail=f"Claim asserts '{c_triple.relation}', but evidence proves '{e_triple.relation}' (uncle != father)",
                                    combination_rule_applied="BYPASS_SUPPRESSION_FORCE_CONTRADICTION",
                                )

            # ── 4. Creation / Invention Relation Check ────────────────
            if c_triple.relation == "created_by" and e_triple.relation == "created_by":
                if subj_match:
                    if self._names_match(c_triple.object, e_triple.object):
                        return RelationCheckResult(
                            claim_triple=c_triple,
                            evidence_triples=all_evidence_triples,
                            status="MATCH",
                            combination_rule_applied="CONFIRM_ENTAILMENT",
                        )
                    else:
                        return RelationCheckResult(
                            claim_triple=c_triple,
                            evidence_triples=all_evidence_triples,
                            status="OBJECT_MISMATCH",
                            mismatch_detail=f"Claim asserts creator is '{c_triple.object}', but authoritative evidence proves creator is '{e_triple.object}'",
                            combination_rule_applied="BYPASS_SUPPRESSION_FORCE_CONTRADICTION",
                        )

            # ── 5. Cybersecurity Vulnerability Check ──────────────────
            if c_triple.relation == "associated_with" and e_triple.relation == "associated_with":
                if subj_match and self._names_match(c_triple.object, e_triple.object):
                    return RelationCheckResult(
                        claim_triple=c_triple,
                        evidence_triples=all_evidence_triples,
                        status="MATCH",
                        combination_rule_applied="CONFIRM_ENTAILMENT",
                    )

        return RelationCheckResult(
            claim_triple=c_triple,
            evidence_triples=all_evidence_triples,
            status="NO_TRIPLE_EXTRACTED",
            mismatch_detail="No conclusive direct relational match or contradiction found among triples",
        )