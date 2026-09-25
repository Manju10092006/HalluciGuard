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

    # Negation cues used to mark a triple's polarity. The extractor is otherwise
    # polarity-blind (it collapses "X was NOT created by Y" into the positive triple
    # (X, created_by, Y)), so relation checks must consult Triple.negated before
    # forcing a contradiction. Matches "not"/"never"/"cannot"/"no longer" and any
    # "n't" contraction (isn't, wasn't, didn't, doesn't, hasn't, won't, can't, ...).
    _NEGATION_CUE = re.compile(
        r"\b(?:not|never|cannot|no|none|neither|nor|unrelated|incorrect|false|unassociated|no\s+longer)\b|n['’]t\b",
        re.IGNORECASE,
    )

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

            # Record where this sentence's triples begin, and detect negation once
            # so we can stamp every triple from this sentence with its polarity below.
            _triple_base = len(triples)
            sent_negated = bool(self._NEGATION_CUE.search(sent_clean))

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

            # ── 4b. Predicate-noun role: "X is the ROLE of Y" ─────────
            # The dominant hallucination shape. The verb patterns above only
            # catch "X founded Y" / "Y was founded by X"; a claim phrased as
            # "Snehith is the founder of Microsoft" (or CEO/president/…) carries
            # no verb, so without this it yields NO triple and the relation layer
            # goes silent — letting a same-name distractor drive the verdict.
            #   creation roles -> created_by, subject=ORG, object=PERSON
            #     "Snehith is the founder of Microsoft" -> (microsoft, created_by, snehith)
            #   leadership roles -> leads, subject=ORG, object=PERSON
            #     "Tim Cook is the CEO of Apple"        -> (apple, leads, tim cook)
            role_np = re.search(
                r"([A-Za-z0-9\s\.\-]+?)\s+(?:is|was)\s+(?:the\s+|a\s+|an\s+|one\s+of\s+the\s+)?"
                r"(?:co[\s-]?)?(founder|cofounder|creator|inventor|author|developer|"
                r"designer|maker|discoverer|architect|writer|builder|"
                r"ceo|c\.e\.o|president|chairman|chairwoman|owner|director|head|"
                r"chief\s+executive)s?\s+of\s+([A-Za-z0-9\s\.\-]+)",
                sent_clean,
                re.IGNORECASE,
            )
            if role_np:
                person = role_np.group(1).strip()
                role_word = re.sub(r"\s+", " ", role_np.group(2).strip().lower())
                org = role_np.group(3).strip()
                org = re.split(
                    r"\b(and|which|who|in|since|from|located|headquartered|based)\b",
                    org,
                    flags=re.IGNORECASE,
                )[0].strip()
                creation_roles = {
                    "founder", "cofounder", "creator", "inventor", "author",
                    "developer", "designer", "maker", "discoverer", "architect",
                    "writer", "builder",
                }
                if person and org and len(org) > 1:
                    triples.append(
                        Triple(
                            subject=self._normalize_name(org),
                            relation="created_by" if role_word in creation_roles else "leads",
                            object=self._normalize_name(person),
                            qualifiers=[role_word, "predicate_noun"],
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

            # Stamp sentence-level polarity onto every triple extracted from this
            # sentence. Without this a negated claim ("X was NOT created by Y") yields
            # a positive triple and can be force-contradicted downstream.
            # verify_relation() consults Triple.negated to stay fail-safe.
            if sent_negated:
                for _t in triples[_triple_base:]:
                    _t.negated = True

        return triples

    def _names_match(self, name1: str, name2: str) -> bool:
        """Return True only when two entity names denote the SAME entity.

        Fail-closed matching ladder (F-2 fix — false-VERIFIED vector):
          1. Exact normalized equality. ``_normalize_name`` already strips
             articles, honorifics, professions and corporate suffixes, so this
             covers "Paris"=="Paris", "James A. Gosling"=="James Gosling" and
             "Amazon Inc"=="Amazon".
          2. Token-SET equality. The extractor concatenates ``title + snippet``
             before extraction, so a subject is routinely duplicated
             ("Hyderabad Hyderabad ...") or reordered. Comparing the sets of
             distinct tokens treats "hyderabad" and "hyderabad hyderabad" as the
             same entity while keeping "india" vs "indiana" ({india} != {indiana})
             and "paris" vs "paris texas" ({paris} != {paris, texas}) distinct.
          3. Guarded multi-token containment: the SMALLER name must have >= 2
             tokens and be a full token-subset of the larger. This admits
             "Eiffel Tower" vs "Eiffel Tower landmark" and "James Gosling" vs
             "Sir James Gosling", while rejecting the qualifier trap where a
             lone token is swallowed by a longer, DIFFERENT entity.

        A single-token name therefore matches ONLY by exact / set equality. This
        kills the prior substring / single-token-subset bugs where "Paris"
        matched "Paris, Texas" and "India" matched "Indiana" — each forced a
        spurious 0.95 entailment and a false VERIFIED downstream. When names do
        not match here the caller degrades to the polarity-aware NLI stage
        instead of asserting a relation, so a miss is fail-safe (defers), never a
        fabricated match.
        """
        n1 = self._normalize_name(name1)
        n2 = self._normalize_name(name2)
        if not n1 or not n2:
            return False
        if n1 == n2:
            return True
        t1 = set(n1.split())
        t2 = set(n2.split())
        if not t1 or not t2:
            return False
        # Same distinct tokens (handles the title+snippet duplication and any
        # word reordering) => same entity.
        if t1 == t2:
            return True
        smaller, larger = (t1, t2) if len(t1) <= len(t2) else (t2, t1)
        # Require the contained name to be multi-token: a bare token (Paris,
        # India, Washington) can never be engulfed by a longer distinct name.
        if len(smaller) >= 2 and smaller.issubset(larger):
            return True
        return False

    def _contextual_creation_triples(self, passage: Any) -> List[Triple]:
        """Recover passive creation facts whose subject is supplied by the page.

        Encyclopedic leads commonly say ``Founded in 1975 by ...`` instead of
        repeating the article title.  The generic sentence regex cannot infer
        that omitted subject and may accidentally consume page-label text as
        the entity.  Use the passage title as the subject only for this explicit
        passive construction; otherwise leave the result to NLI.
        """
        raw_title = str(getattr(passage, "title", "") or "")
        snippet = str(getattr(passage, "snippet", "") or "")
        if not raw_title or not snippet:
            return []

        title = re.sub(r"^Wikipedia:\s*", "", raw_title, flags=re.IGNORECASE)
        title = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
        if not title:
            return []

        passive = re.search(
            r"\b(created|developed|invented|built|designed|founded|introduced)"
            r"(?:\s+(?:originally|initially))?"
            r"(?:\s+in\s+(?:the\s+)?(?:year\s+)?\d{3,4})?"
            r"\s+by\s+([A-Za-z0-9][A-Za-z0-9 .,'&\-]+?)"
            r"(?=\s+(?:to|for|at|in|as|who|which|where)\b|[.;:]|$)",
            snippet,
            re.IGNORECASE,
        )
        if not passive:
            return []

        creator = passive.group(2).strip(" ,")
        if not creator:
            return []
        return [
            Triple(
                subject=self._normalize_name(title),
                relation="created_by",
                object=self._normalize_name(creator),
                qualifiers=[passive.group(1).lower(), "title_anchored"],
                negated=bool(self._NEGATION_CUE.search(passive.group(0))),
                raw_text=passive.group(0),
            )
        ]

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

        # Polarity gate (fail-safe). The extractor is polarity-blind: it collapses
        # "X was NOT created by Y" into the positive triple (X, created_by, Y). If a
        # negated claim runs the structural comparison, a TRUE refutation such as
        # "Python was not created by Elon Musk" matches truth evidence ("created by
        # Guido van Rossum"), fires OBJECT_MISMATCH, and manufactures a false 0.95
        # contradiction over ~0 NLI. When the claim is negated we cannot trust the
        # (polarity-blind) triple, so we abstain here and defer to the polarity-aware
        # NLI stage by reporting NO_TRIPLE_EXTRACTED.
        if c_triple.negated:
            return RelationCheckResult(
                claim_triple=c_triple,
                evidence_triples=[],
                status="NO_TRIPLE_EXTRACTED",
                mismatch_detail="Claim is negated; relation comparison is polarity-blind, deferring to NLI",
            )

        all_evidence_triples: List[Triple] = []

        for p in evidence_passages:
            text = f"{getattr(p, 'title', '')} {getattr(p, 'snippet', '')}"
            e_triples = self.extract_triples(text)
            e_triples.extend(self._contextual_creation_triples(p))
            all_evidence_triples.extend(e_triples)

        if not all_evidence_triples:
            return RelationCheckResult(
                claim_triple=c_triple,
                evidence_triples=[],
                status="NO_TRIPLE_EXTRACTED",
                mismatch_detail="No structured relation triples recognized in retrieved evidence",
            )

        # ── Creation / leadership pre-scan (match-first) ──────────────
        # "X is the founder of Y" / "X is the CEO of Y" claims frequently meet
        # evidence naming MULTIPLE valid people (Bill Gates AND Paul Allen).
        # The generic loop below returns on the FIRST subject-matching evidence
        # triple, so a true co-founder claim could be falsely contradicted just
        # because a different founder's triple was encountered first. Resolve
        # these relations up front: only declare OBJECT_MISMATCH when NO
        # subject-matching evidence triple confirms the claimed person.
        for rel, noun in (("created_by", "creator"), ("leads", "leader")):
            if c_triple.relation != rel:
                continue
            subj_aligned = [
                e for e in all_evidence_triples
                if e.relation == rel and self._names_match(c_triple.subject, e.subject)
            ]
            if not subj_aligned:
                continue
            if any(self._names_match(c_triple.object, e.object) for e in subj_aligned):
                return RelationCheckResult(
                    claim_triple=c_triple,
                    evidence_triples=all_evidence_triples,
                    status="MATCH",
                    combination_rule_applied="CONFIRM_ENTAILMENT",
                )
            proven = subj_aligned[0].object
            return RelationCheckResult(
                claim_triple=c_triple,
                evidence_triples=all_evidence_triples,
                status="OBJECT_MISMATCH",
                mismatch_detail=f"Claim asserts {noun} is '{c_triple.object}', but authoritative evidence proves {noun} is '{proven}'",
                combination_rule_applied="BYPASS_SUPPRESSION_FORCE_CONTRADICTION",
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
