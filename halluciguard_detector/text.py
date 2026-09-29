import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    text: str
    start: int
    end: int


_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+(?=[\"'(\[]*[A-Z0-9])")
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_ENTITY = re.compile(r"\b(?:[A-Z][A-Za-z0-9'-]*)(?:\s+[A-Z][A-Za-z0-9'-]*)*\b")
#: Words that genuinely start lowercase; used to tell a real proper noun from a
#: sentence-initial capitalised common word (see ``shared_relation``).
_LOWERCASE_WORD = re.compile(r"(?<![A-Za-z'])([a-z][A-Za-z0-9'-]*)")
_ENTITY_STOP = {"The", "A", "An", "It", "This", "That", "In", "On", "At", "By"}
_NON_TERMINAL = re.compile(
    r"^(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|[A-Z]|\d+)\.$",
    re.IGNORECASE,
)


def sentence_spans(text: str) -> list[Span]:
    """Split prose while retaining exact offsets in the original string."""
    spans: list[Span] = []
    cursor = 0
    for match in _BOUNDARY.finditer(text):
        end = match.start()
        raw = text[cursor:end]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            spans.append(Span(raw[left:right], cursor + left, cursor + right))
        cursor = match.end()
    raw = text[cursor:]
    left = len(raw) - len(raw.lstrip())
    right = len(raw.rstrip())
    if right > left:
        spans.append(Span(raw[left:right], cursor + left, cursor + right))
    merged: list[Span] = []
    for span in spans:
        if merged and _NON_TERMINAL.match(merged[-1].text):
            previous = merged.pop()
            merged.append(Span(text[previous.start : span.end], previous.start, span.end))
        else:
            merged.append(span)
    return merged


def lexical_evidence(claim: str, documents: list[str], limit: int = 6) -> list[str]:
    """Return compact evidence fragments using deterministic lexical retrieval."""
    query = set(_TOKEN.findall(claim.lower()))
    candidates: list[tuple[float, int, str]] = []
    order = 0
    for document in documents:
        for span in sentence_spans(document):
            terms = set(_TOKEN.findall(span.text.lower()))
            overlap = len(query & terms)
            score = overlap / max(1.0, len(query) ** 0.5 * len(terms) ** 0.5)
            candidates.append((score, -order, span.text))
            order += 1
    candidates.sort(reverse=True)
    selected = [text for score, _, text in candidates if score > 0][:limit]
    if not selected:
        selected = [text for _, _, text in candidates[:limit]]
    return selected


_RELATION_STOP = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "by", "to", "for",
    "from", "with", "is", "are", "was", "were", "be", "been", "it", "its",
    "this", "that", "as", "into", "about", "over", "under", "during",
}


def _entity_tokens(text: str) -> set[str]:
    """Lowercased tokens that belong to a detected named entity."""
    tokens: set[str] = set()
    for entity in _ENTITY.findall(text):
        if entity in _ENTITY_STOP:
            continue
        tokens.update(_TOKEN.findall(entity.lower()))
    return tokens


def shared_relation(claim: str, evidence: str) -> set[str]:
    """Return content words shared by both texts, ignoring named entities.

    Used to tell a *direct* entity contradiction (same predicate, e.g.
    "Apple acquired Company A" vs "Apple acquired Company B", which shares
    ``acquired``) from a *mere* entity mismatch (different predicate, e.g.
    "Apple works with Company A" vs "Apple acquired Company B", which shares
    no content word once the entities are removed). The entity guard uses this
    only to modulate the secondary contradiction signal; it never decides truth
    on its own.
    """
    # Entity words are excluded: they are the thing that *differs*, so counting
    # them would make every entity conflict look like a shared relation. Bare
    # numbers are excluded too: those are quantities, handled by
    # ``numeric_consistency``, not relations.
    #
    # A capitalised word is only a proper noun if it never occurs in lowercase.
    # The entity vocabulary is shared across both texts, so one spurious hit
    # erases the word from *both* sides: "Released in 1995." reads "Released" as
    # a name, and the shared predicate "released" then vanishes, hiding a real
    # 1995-vs-1996 conflict. "Company A" / "Company B" stay excluded because
    # "company" is only ever capitalised.
    lowercase_seen: set[str] = set()
    for text in (claim, evidence):
        lowercase_seen.update(_LOWERCASE_WORD.findall(text))
    entity_words = (_entity_tokens(claim) | _entity_tokens(evidence)) - lowercase_seen
    claim_words = {
        w
        for w in _TOKEN.findall(claim.lower())
        if w and w not in _RELATION_STOP and w not in entity_words and not w.isdigit()
    }
    evidence_words = {
        w
        for w in _TOKEN.findall(evidence.lower())
        if w and w not in _RELATION_STOP and w not in entity_words and not w.isdigit()
    }
    return claim_words & evidence_words


def has_entity_conflict(claim: str, evidence: str) -> bool:
    """Conservative guard for incompatible named entities around a shared anchor.

    It fires only when both texts share a named anchor and both also contain a
    different named entity. This avoids treating a merely absent entity as false.
    """
    claim_entities = {x for x in _ENTITY.findall(claim) if x not in _ENTITY_STOP}
    evidence_entities = {x for x in _ENTITY.findall(evidence) if x not in _ENTITY_STOP}
    shared = claim_entities & evidence_entities
    return bool(
        shared
        and (claim_entities - evidence_entities)
        and (evidence_entities - claim_entities)
    )


_QUANTITY = re.compile(r"(\d[\d,.]*)\s*(%|percent|million|billion|trillion|m|bn|k)?\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")


def numeric_consistency(claim: str, evidence: str) -> list[str]:
    """Report lightweight number/date/percent mismatches between claim and evidence.

    Conservative by design (no giant rule engine). Every reported mismatch must
    clear two bars:

    1. Both texts must share a relation word once named entities and bare
       numbers are removed, so the two statements are about a comparable
       predicate. Without this, "Tesla was founded in 2003" against "Tesla
       went public in 2018" looks like a year conflict even though they are
       different facts about different events, and a plain year disagreement
       would manufacture contradiction mass.
    2. For quantities, both sides must use the same unit, so a percentage is
       never compared against a raw count.

    Raw numbers with no mismatching unit/context are ignored, so dates, IDs and
    naturally different statistics are not treated as contradictions. This is a
    secondary signal only: it may resolve a near-tie and never overturn a
    decisive model call.
    """
    conflicts: list[str] = []

    # Comparability gate. A disagreement about numbers or years only speaks to
    # the claim's truth when both texts are talking about the same relation.
    if not shared_relation(claim, evidence):
        return conflicts

    def quantities(text: str) -> list[tuple[str, str]]:
        # A unit is required. Unitless numbers are deliberately ignored: giving
        # them a placeholder unit made *any* two differing bare integers in a
        # comparable sentence a "conflict", so identifiers ("record id 12345"
        # vs "98765") and plain counts were reported as contradictions. Years
        # are handled separately below, where 1995-vs-1996 really is a clash.
        lowered = text.lower()
        return [
            (num.replace(",", ""), (unit or "").lower())
            for num, unit in _QUANTITY.findall(lowered)
            if unit and not (len(num) == 4 and num.isdigit())
        ]

    claim_q = quantities(claim)
    evidence_q = quantities(evidence)
    units_in_claim = {unit for _, unit in claim_q}
    for num, unit in evidence_q:
        if unit in units_in_claim:
            claim_nums = {n for n, u in claim_q if u == unit}
            if claim_nums and num not in claim_nums:
                conflicts.append(f"conflicting {unit}: '{num} {unit}' vs claim")

    claim_years = set(_YEAR.findall(claim))
    evidence_years = set(_YEAR.findall(evidence))
    if claim_years and evidence_years and not (claim_years & evidence_years):
        conflicts.append(f"conflicting years: {','.join(sorted(evidence_years))} vs claim")

    return conflicts
