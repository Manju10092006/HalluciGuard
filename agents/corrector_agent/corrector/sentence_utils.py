"""Byte-accurate sentence segmentation and text normalization (Phase 2, Step 2).

Adapted from the read-only reference implementation's span extractor, which is
reimplemented here (never imported) and hardened.

GUARANTEES for every span returned by ``extract_sentence_spans(text)``:
    1. ``text[span.start_offset:span.end_offset] == span.text``  (byte-accurate)
    2. spans are strictly ordered and non-overlapping
    3. no span is empty or pure whitespace
    4. inter-sentence whitespace/newlines are NOT part of any span, so the
       original text can later be reconstructed by offset splicing alone

This module performs NO matching decisions and NO correction logic. It is pure,
deterministic, and side-effect free.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Set, Tuple

from .contracts import SentenceSpan

__all__ = [
    "COMMON_ABBREVIATIONS",
    "extract_sentence_spans",
    "verify_spans_integrity",
    "find_span_integrity_error",
    "normalize_whitespace_case",
    "normalize_for_matching",
    "tokenize",
    "token_coverage",
]

# Abbreviations that must not be treated as sentence boundaries.
COMMON_ABBREVIATIONS: Set[str] = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "eg", "ie",
    "approx", "apt", "dept", "est", "inc", "ltd", "corp", "co", "jan", "feb",
    "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "u.s", "u.k", "u.n", "e.u", "st", "ave", "rd", "blvd", "no", "nos",
    "fig", "figs", "al", "cf", "vol", "pp",
}

# Sentence terminators (with trailing closing quotes/brackets) or paragraph breaks.
_BOUNDARY = re.compile(r"([.!?]+[\'\"’”\)\]]*|\n+)")

_QUOTE_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
}


def _skip_ws(text: str, pos: int) -> int:
    n = len(text)
    while pos < n and text[pos].isspace():
        pos += 1
    return pos


def _rstrip_pos(text: str, low: int, high: int) -> int:
    while high > low and text[high - 1].isspace():
        high -= 1
    return high


def _is_abbreviation(token: str) -> bool:
    return token.rstrip(".").lower() in COMMON_ABBREVIATIONS


def _is_initial(token: str) -> bool:
    """Single-letter initial such as the 'R' in 'Guido van R. Smith'."""
    stripped = token.rstrip(".")
    return len(stripped) == 1 and stripped.isalpha()


def _preceding_word(text: str, low: int, high: int) -> str:
    chunk = text[low:high].split()
    return chunk[-1] if chunk else ""


def _is_inside_number(text: str, match_start: int, match_end: int) -> bool:
    """True for decimals/versions like 3.14 or 1.5 where '.' is not a boundary."""
    if match_start == 0 or match_end >= len(text):
        return False
    return text[match_start - 1].isdigit() and text[match_end].isdigit()


def _make_span(index: int, text: str, start: int, end: int) -> SentenceSpan:
    return SentenceSpan(
        sentence_id=f"S{index}",
        text=text[start:end],
        start_offset=start,
        end_offset=end,
    )


def extract_sentence_spans(text: str) -> List[SentenceSpan]:
    """Split ``text`` into byte-accurate, ordered, non-overlapping sentence spans."""
    if not text or not text.strip():
        return []

    spans: List[SentenceSpan] = []
    n = len(text)
    start = _skip_ws(text, 0)
    idx = start
    sid = 1

    while idx < n:
        match = _BOUNDARY.search(text, idx)
        if match is None:
            end = _rstrip_pos(text, start, n)
            if end > start:
                spans.append(_make_span(sid, text, start, end))
            break

        match_start, match_end = match.span()
        token = match.group(1)

        # Paragraph break: the boundary sits BEFORE the newline run.
        if "\n" in token:
            end = _rstrip_pos(text, start, match_start)
            if end > start:
                spans.append(_make_span(sid, text, start, end))
                sid += 1
            start = _skip_ws(text, match_end)
            idx = start
            continue

        if "." in token:
            if _is_inside_number(text, match_start, match_end):
                idx = match_end
                continue
            preceding = _preceding_word(text, start, match_start)
            if _is_abbreviation(preceding) or _is_initial(preceding):
                idx = match_end
                continue

        # A terminator only ends a sentence when followed by whitespace or EOF.
        if match_end < n and not text[match_end].isspace():
            idx = match_end
            continue

        end = _rstrip_pos(text, start, match_end)
        if end > start and text[start:end].strip():
            spans.append(_make_span(sid, text, start, end))
            sid += 1
        start = _skip_ws(text, match_end)
        idx = start

    # Text with no recognizable terminator is a single sentence.
    if not spans:
        s = _skip_ws(text, 0)
        e = _rstrip_pos(text, s, n)
        if e > s:
            spans.append(_make_span(1, text, s, e))

    return spans


def find_span_integrity_error(text: str, spans: List[SentenceSpan]) -> Optional[str]:
    """Return a human-readable reason if the span set violates any guarantee."""
    last_end = 0
    for i, span in enumerate(spans):
        if span.start_offset < 0 or span.end_offset > len(text):
            return f"{span.sentence_id}: offsets out of bounds"
        if span.end_offset <= span.start_offset:
            return f"{span.sentence_id}: empty or inverted span"
        if span.start_offset < last_end:
            return f"{span.sentence_id}: overlaps previous span"
        # Requirement 13 — the core byte-accuracy check.
        if text[span.start_offset:span.end_offset] != span.text:
            return (
                f"{span.sentence_id}: text[{span.start_offset}:{span.end_offset}] "
                f"!= span.text"
            )
        if i and spans[i - 1].sentence_id == span.sentence_id:
            return f"{span.sentence_id}: duplicate sentence_id"
        last_end = span.end_offset
    return None


def verify_spans_integrity(text: str, spans: List[SentenceSpan]) -> bool:
    """True when every span satisfies ``text[start:end] == span.text`` and ordering."""
    return find_span_integrity_error(text, spans) is None


# ---------------------------------------------------------------------------
# Normalization (matching support only — never applied to emitted text)
# ---------------------------------------------------------------------------

def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(_QUOTE_FOLD.get(ch, ch) for ch in text)


def normalize_whitespace_case(text: str) -> str:
    """Level 2 normalization: unicode/quote fold, casefold, collapse whitespace."""
    return re.sub(r"\s+", " ", _fold(text).casefold()).strip()


def normalize_for_matching(text: str) -> str:
    """Level 3 normalization: additionally drop punctuation.

    Note: this maps '3.14' to '3 14'. That is acceptable for locating a claim
    but makes it unsuitable for anything other than comparison.
    """
    folded = re.sub(r"[^\w\s]", " ", _fold(text).casefold())
    return re.sub(r"\s+", " ", folded).strip()


def tokenize(text: str) -> List[str]:
    """Word tokens of the fully normalized text."""
    return re.findall(r"\w+", normalize_for_matching(text))


def token_coverage(claim_tokens: Set[str], span_tokens: Set[str]) -> float:
    """Fraction of the claim's distinct tokens present in the span."""
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & span_tokens) / len(claim_tokens)
