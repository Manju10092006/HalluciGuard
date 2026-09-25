from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy, cached spaCy model loader.
#
# The decomposer prefers a POS/dependency-aware split (accurate on the hard
# cases — compound modifiers vs. distributive objects, pronoun subjects,
# sentence-fragment rejection). If spaCy or its English model is unavailable,
# it degrades gracefully to a conservative regex path that never emits the
# fragments this module exists to prevent.
# ---------------------------------------------------------------------------
_NLP = None                 # cached spaCy Language object once loaded
_NLP_LOAD_ATTEMPTED = False  # guards against repeated load attempts on failure


def _load_nlp():
    """Load and cache the spaCy English model once. Returns None if unavailable."""
    global _NLP, _NLP_LOAD_ATTEMPTED
    if _NLP is not None or _NLP_LOAD_ATTEMPTED:
        return _NLP
    _NLP_LOAD_ATTEMPTED = True
    try:
        import spacy  # type: ignore

        # NER and lemmatizer are not needed for decomposition — disable for speed.
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        logger.debug("ClaimDecomposer: loaded spaCy model 'en_core_web_sm'.")
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning(
            "ClaimDecomposer: spaCy unavailable (%s); using regex fallback decomposition.",
            exc,
        )
        _NLP = None
    return _NLP


# Dependency labels used to recognise clause structure.
_SUBJ_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
# Object-position dependencies eligible for distributive ("A verbs X and Y") splitting.
_OBJ_DEPS = {"dobj", "obj", "dative", "attr", "oprd"}
# Complement dependencies that make a copula ("X is ...") a complete proposition.
_COMPLEMENT_DEPS = {
    "acomp", "attr", "dobj", "obj", "oprd", "dative", "prep",
    "advcl", "xcomp", "ccomp", "npadvmod", "acl", "relcl",
}
# Subject pronouns resolved against the most recent concrete subject.
_RESOLVABLE_PRONOUNS = {"it", "this", "that", "they", "these", "those", "he", "she"}
# Copular surface forms (lemmatizer is disabled, so match on text).
_COPULA = {"is", "are", "was", "were", "be", "been", "being", "am", "'s", "’s", "'re", "'m"}

# Whole-utterance conversational acknowledgements that carry no factual content.
_FILLER_EXACT = {
    "that's correct", "that is correct", "that's right", "that is right",
    "correct", "sure", "yes", "yeah", "yep", "no", "nope", "absolutely",
    "of course", "indeed", "right", "you're right", "you are right",
    "exactly", "i agree", "good question", "great question",
    "that's a good question", "well", "ok", "okay", "certainly",
    "actually", "actually no",
    # negated acknowledgements — assert nothing internet-checkable on their own
    "that's not correct", "that is not correct", "that's incorrect",
    "that is incorrect", "that's wrong", "that is wrong", "no that's not correct",
    "actually that's not correct", "actually that is not correct", "actually that isn't correct",
    "actually that's incorrect", "actually that is incorrect", "actually that's wrong",
    "that isn't correct", "that isn't right", "that's not right", "that is not right",
    "no that isn't correct", "no that is not correct", "no that's not right",
}

# Punctuation-insensitive form of the filler set, so "No, that's not correct"
# (with comma/apostrophe) matches "that's not correct".
_FILLER_STRIPPED = {
    re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", f.lower())).strip()
    for f in _FILLER_EXACT
}

# Non-factual / non-checkable sentence patterns. These are grammatically valid
# (subject + predicate) so _is_valid() lets them through, but they assert nothing
# a search engine can confirm or refute — they are conversational meta, offers to
# help, or direct address. Sending them to retrieval wastes Tavily/n8n calls and
# credits, so they are filtered BEFORE verification.
_NONFACTUAL_RE = re.compile(
    r"^\s*(?:"
    r"if\s+you\b|feel\s+free\b|let\s+me\s+know\b|please\s+(?:let|feel|note|check)\b|"
    r"for\s+the\s+most\b|for\s+more\s+(?:accurate|information|details)\b|"
    r"i\s+hope\s+this\b|hope\s+(?:this|that)\s+helps\b|"
    r"i(?:'m| am)\s+(?:sorry|not\s+sure|happy\s+to|unable)\b|"
    r"i\s+(?:can(?:not|'t)?|could\s+not|do\s+not|don't)\b|"
    r"you\s+(?:can|could|should|may|might|will|would)\b|"
    r"as\s+an\s+ai\b|note\s+that\b|keep\s+in\s+mind\b|"
    r"checking\b.*\bwould\s+be\s+best\b|"
    r".*\bwould\s+be\s+(?:best|advisable|recommended)\s*\.?\s*$|"
    r".*\bfeel\s+free\s+to\b|.*\bclarify\b\s*!?\s*$"
    r")",
    re.IGNORECASE,
)

# Markdown constructs stripped before decomposition (drafts are markdown; without
# this the decomposer treats "**bold**", "# Heading", "- bullet" as claim text).
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)]*)\)")   # [text](url) -> text
_MD_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)  # # Heading
_MD_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)  # - / 1. bullets
_MD_EMPHASIS_RE = re.compile(r"(\*\*|__|\*|_|`)")        # **bold** _italic_ `code`
_MD_LABEL_RE = re.compile(r"^\s*[A-Z][A-Za-z /]{1,40}:\s+(?=\S)")  # "Key points: ..."

# Leading conversational framing stripped from the front of a sentence.
_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"actually,\s*(?:that's|that\s+is|that\s+isn't)\s+(?:not\s+)?(?:correct|right|true|accurate|wrong|incorrect)|"
    r"actually,\s*no|actually|"
    r"no,\s*(?:that's|that\s+is|that\s+isn't)\s+(?:not\s+)?(?:correct|right|true|accurate|wrong|incorrect)|"
    r"that's\s+(?:not\s+)?(?:correct|right|true|accurate|wrong|incorrect)|"
    r"that\s+is\s+(?:not\s+)?(?:correct|right|true|accurate|wrong|incorrect)|"
    r"that\s+isn't\s+(?:correct|right|true|accurate)|"
    r"sure|yes|yeah|well|of\s+course|certainly|absolutely|indeed|"
    r"according\s+to\s+me|in\s+my\s+opinion|to\s+answer\s+your\s+question|"
    r"i\s+think|i\s+believe|as\s+an\s+ai(?: language model)?"
    r")\b[\s,:;!.\-]*",
    re.IGNORECASE,
)

# Coarse segmentation on semicolons, newlines, and numbered-list markers.
_COARSE_SPLIT_RE = re.compile(r"[;\n]+|(?:^|\s)\d+\.\s+")

_MAX_CLAIMS = 5


class ClaimDecomposer:
    """Decomposes an LLM response into atomic, complete factual propositions.

    A valid output claim is a grammatically complete proposition (has a subject
    and a predicate), independently understandable, and faithful to the source
    text. Conversational framing is removed, subject pronouns are resolved from
    immediate context, compound predicates/modifiers are preserved as one claim,
    and coordinated objects ("cures X and Y") are distributed into separate
    claims. Sentence fragments (bare pronouns, noun/adjective phrases, dangling
    conjunction clauses) are never emitted.
    """

    def __init__(self) -> None:
        # Model is loaded lazily/shared at module level; construction stays cheap.
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decompose(self, text: str) -> List[str]:
        """Decompose ``text`` into a list of atomic factual claims.

        Args:
            text: The claim / response text to decompose.

        Returns:
            A list of complete, atomic sub-claims (capped at 5). Returns an
            empty list for empty input.
        """
        if not text or not text.strip():
            return []

        # Strip markdown so headers/bullets/bold/links don't become claim text.
        text = self._clean_markdown(text)
        if not text.strip():
            return []

        nlp = _load_nlp()
        if nlp is None:
            claims = self._decompose_regex(text)
        else:
            claims = self._decompose_spacy(nlp, text)

        # Drop non-factual / non-checkable sentences (offers, meta, questions) so
        # retrieval isn't wasted on them. Applied to every path's output.
        claims = [c for c in claims if self._is_checkable(c)]

        if not claims:
            # Nothing survived. Only fall back to the raw text if it is itself a
            # checkable factual sentence; otherwise emit nothing (don't send junk
            # like "feel free to clarify!" to the verifier).
            raw = text.strip()
            return [raw] if self._is_checkable(raw) else []

        logger.debug("Decomposed '%s' into %d sub-claims", text[:80], len(claims))
        return claims[:_MAX_CLAIMS]

    # ------------------------------------------------------------------
    # Pre-processing / filtering helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_markdown(text: str) -> str:
        """Strip markdown formatting so it isn't parsed as claim content."""
        text = _MD_LINK_RE.sub(r"\1", text)      # [label](url) -> label
        text = _MD_HEADER_RE.sub("", text)       # drop "# " heading markers
        text = _MD_BULLET_RE.sub("", text)       # drop "- " / "1. " bullet markers
        text = _MD_EMPHASIS_RE.sub("", text)     # drop **, __, *, _, `
        text = _MD_LABEL_RE.sub("", text)        # drop lead-in "Key points: " labels
        return text

    @classmethod
    def _is_checkable(cls, text: str) -> bool:
        """True if ``text`` is a factual sentence worth sending to retrieval.

        Rejects questions, conversational meta, offers to help and direct address
        — grammatically valid but not confirmable/refutable by evidence.
        """
        s = text.strip()
        if not s:
            return False
        if s.endswith("?"):
            return False
        # Punctuation-insensitive filler match ("No, that's not correct" -> "no thats not correct").
        filler_key = re.sub(r"[^a-z0-9\s]", "", s.lower())
        filler_key = re.sub(r"\s+", " ", filler_key).strip()
        if filler_key in _FILLER_STRIPPED or cls._normalize_key(s) in _FILLER_EXACT:
            return False
        if _NONFACTUAL_RE.match(s):
            return False
        # Require at least 3 word-tokens of substance.
        if len(re.findall(r"[A-Za-z0-9]+", s)) < 3:
            return False
        return True

    # ------------------------------------------------------------------
    # spaCy-powered path
    # ------------------------------------------------------------------
    def _decompose_spacy(self, nlp, text: str) -> List[str]:
        claims: List[str] = []
        seen: set[str] = set()
        last_subject: Optional[str] = None

        for segment in _COARSE_SPLIT_RE.split(text):
            segment = segment.strip()
            if not segment:
                continue

            for sent in nlp(segment).sents:
                stripped = self._strip_prefix(sent.text)
                if not stripped or self._normalize_key(stripped) in _FILLER_EXACT:
                    continue

                # Re-parse only when prefix stripping changed the string.
                sent_span = sent if stripped == sent.text.strip() else self._first_span(nlp, stripped)
                if sent_span is None:
                    continue

                resolved_text = self._resolve_pronoun(sent_span, last_subject)
                resolved_span = (
                    sent_span if resolved_text == sent_span.text.strip()
                    else self._first_span(nlp, resolved_text)
                )
                if resolved_span is None:
                    continue

                # Track the subject for resolving pronouns in following sentences.
                subject = self._subject_phrase(resolved_span)
                if subject:
                    last_subject = subject

                candidates = self._distribute(resolved_span) or [resolved_span.text]
                for candidate in candidates:
                    cand_span = self._first_span(nlp, candidate)
                    if cand_span is None or not self._is_valid(cand_span):
                        continue
                    cleaned = candidate.strip().rstrip(".").strip()
                    key = self._normalize_key(cleaned)
                    if not cleaned or key in seen:
                        continue
                    seen.add(key)
                    claims.append(cleaned)

        return claims

    @staticmethod
    def _first_span(nlp, text: str):
        """Parse ``text`` and return its first sentence as a Span (or None)."""
        doc = nlp(text)
        sents = list(doc.sents)
        if sents:
            return sents[0]
        return doc[:] if len(doc) else None

    @staticmethod
    def _content_tokens(span):
        return [t for t in span if not t.is_punct and not t.is_space]

    def _is_valid(self, span) -> bool:
        """A complete proposition: >=3 content tokens, a subject, and a predicate."""
        content = self._content_tokens(span)
        if len(content) < 3:
            return False
        if not any(t.dep_ in _SUBJ_DEPS for t in span):
            return False
        if not any(t.pos_ in ("VERB", "AUX") for t in span):
            return False
        # Reject a bare copula with no complement ("it is", "this is also").
        root = span.root
        if root is not None and root.lower_ in _COPULA:
            if not any(child.dep_ in _COMPLEMENT_DEPS for child in root.children):
                return False
        return True

    @staticmethod
    def _subject_phrase(span) -> Optional[str]:
        """Return the concrete (non-pronoun) subject noun phrase of ``span``."""
        for tok in span:
            if tok.dep_ in _SUBJ_DEPS and tok.pos_ != "PRON":
                sub = span.doc[tok.left_edge.i: tok.right_edge.i + 1]
                return sub.text.strip()
        return None

    @staticmethod
    def _resolve_pronoun(span, last_subject: Optional[str]) -> str:
        """Replace a leading subject pronoun with the most recent concrete subject."""
        if not last_subject:
            return span.text.strip()
        subjects = [t for t in span if t.dep_ in _SUBJ_DEPS]
        if not subjects:
            return span.text.strip()
        subj = subjects[0]
        if subj.text.lower() not in _RESOLVABLE_PRONOUNS:
            return span.text.strip()
        rebuilt = []
        for tok in span:
            if tok.i == subj.i:
                rebuilt.append(last_subject + tok.whitespace_)
            else:
                rebuilt.append(tok.text_with_ws)
        return "".join(rebuilt).strip()

    def _distribute(self, span) -> Optional[List[str]]:
        """Split coordinated *object* nouns into separate claims.

        "Vitamin C cures cancer and diabetes" -> two claims. Compound modifiers
        ("technological and commercial hub") and coordinated prepositional
        objects are left as a single claim.
        """
        doc = span.doc
        for tok in span:
            conjuncts = list(tok.conjuncts)
            if not conjuncts:
                continue
            if tok.dep_ not in _OBJ_DEPS or tok.pos_ not in ("NOUN", "PROPN"):
                continue
            if not all(c.pos_ in ("NOUN", "PROPN") for c in conjuncts):
                continue

            # Collect the coordination chain rooted at ``tok``.
            group = [tok]
            stack = [tok]
            while stack:
                node = stack.pop()
                for child in node.children:
                    if child.dep_ == "conj":
                        group.append(child)
                        stack.append(child)
            group = sorted(set(group), key=lambda x: x.i)
            if len(group) < 2:
                continue

            owns = {c: self._own_tokens(c) for c in group}
            prefix = doc[span.start: tok.left_edge.i].text
            max_i = max(t.i for toks in owns.values() for t in toks)
            suffix = doc[max_i + 1: span.end].text

            claims: List[str] = []
            for c in group:
                phrase = " ".join(t.text for t in sorted(owns[c], key=lambda x: x.i))
                parts = [p for p in (prefix.strip(), phrase.strip(), suffix.strip()) if p]
                claims.append(" ".join(parts))
            return claims
        return None

    @staticmethod
    def _own_tokens(node):
        """Subtree of ``node`` excluding its coordinated siblings and connectors."""
        excluded = set()
        for child in node.children:
            if child.dep_ in ("conj", "cc", "preconj"):
                excluded |= set(child.subtree)
        return [t for t in node.subtree if t not in excluded]

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_prefix(text: str) -> str:
        prev = None
        out = text
        while prev != out:
            prev = out
            out = _PREFIX_RE.sub("", out, count=1)
        return out.strip()

    @staticmethod
    def _normalize_key(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().strip('."\'!?,;:').lower())

    # ------------------------------------------------------------------
    # Regex fallback path (spaCy unavailable)
    # ------------------------------------------------------------------
    def _decompose_regex(self, text: str) -> List[str]:
        """Conservative sentence-level split with pronoun resolution and object distribution."""
        claims: List[str] = []
        seen: set[str] = set()
        last_subject: Optional[str] = None

        for segment in _COARSE_SPLIT_RE.split(text):
            for raw in re.split(r"(?<=[.!?])\s+", segment.strip()):
                stripped = self._strip_prefix(raw)
                if not stripped:
                    continue
                key = self._normalize_key(stripped)
                if key in _FILLER_EXACT:
                    continue

                # Pronoun resolution: e.g. "It is also a major city" -> "Hyderabad is also a major city"
                pronoun_match = re.match(
                    r"^(it|this|that|they|he|she)\b(\s+(?:is|was|are|were|also|has|have|can|will)\b.*)$",
                    stripped,
                    re.IGNORECASE,
                )
                if pronoun_match and last_subject:
                    stripped = f"{last_subject}{pronoun_match.group(2)}"
                    key = self._normalize_key(stripped)
                else:
                    # Extract candidate subject phrase before verb / copula
                    subj_match = re.match(
                        r"^([A-Z][a-zA-Z0-9\s'-]+?)\s+(?:is|are|was|were|cures|treats|causes|has|have|contains|located)\b",
                        stripped,
                    )
                    if subj_match:
                        cand_subj = subj_match.group(1).strip()
                        if cand_subj.lower() not in _RESOLVABLE_PRONOUNS and len(cand_subj.split()) <= 4:
                            last_subject = cand_subj

                # Object distribution: e.g. "Vitamin C cures cancer and diabetes" -> "Vitamin C cures cancer", "Vitamin C cures diabetes"
                dist_match = re.match(
                    r"^(.+?\b(?:cures|treats|causes|prevents|produces|contains|includes|is responsible for)\b)\s+([a-zA-Z0-9\s]+?)\s+and\s+([a-zA-Z0-9\s]+)$",
                    stripped,
                    re.IGNORECASE,
                )
                candidates: List[str] = []
                if dist_match:
                    prefix_part = dist_match.group(1).strip()
                    first_obj = dist_match.group(2).strip()
                    second_obj = dist_match.group(3).strip()
                    if len(first_obj.split()) <= 4 and len(second_obj.split()) <= 4:
                        candidates = [f"{prefix_part} {first_obj}", f"{prefix_part} {second_obj}"]

                if not candidates:
                    candidates = [stripped]

                for cand in candidates:
                    cand_clean = cand.strip().rstrip(".").strip()
                    cand_key = self._normalize_key(cand_clean)
                    if not cand_clean or cand_key in _FILLER_EXACT or cand_key in seen:
                        continue
                    if len(cand_key.split()) < 3:
                        continue
                    seen.add(cand_key)
                    claims.append(cand_clean)

        return claims
