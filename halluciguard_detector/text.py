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
