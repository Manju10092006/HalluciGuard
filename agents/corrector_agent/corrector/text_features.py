"""Step 5 — deterministic text feature extraction for candidate validation.

PURE EXTRACTION, ZERO DECISIONS
-------------------------------
Every function here answers "what is in this text?" and never "is this text
acceptable?". All acceptance logic lives in :mod:`validators`. Nothing here
raises on ordinary input, imports a model, touches the filesystem, or depends on
the ML stack. Given identical input every function returns identical output —
there is no randomness, no clock, and no I/O.

Normalization is deliberately delegated to :mod:`sentence_utils` rather than
reimplemented, so Steps 2, 3 and 5 share one normalization vocabulary.

WORD-BOUNDARY DISCIPLINE
------------------------
Number extraction is value-level and boundary-anchored, so ``"48"`` is NOT found
inside ``"1948"`` or ``"480"``. This matters: the reference implementation used
substring containment (``num not in combined_source``) and therefore treated a
hallucinated ``48`` as grounded whenever an unrelated ``1948`` appeared in the
evidence.

COMPARISON, NOT REPAIR
----------------------
No function returns a "cleaned" version of the caller's text for emission.
Normalized forms are produced for comparison only. Candidate text is carried
verbatim everywhere else in the pipeline.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .sentence_utils import (
    COMMON_ABBREVIATIONS,
    normalize_for_matching,
    normalize_whitespace_case,
    tokenize,
)

__all__ = [
    "STOPWORDS",
    "ABSTENTION_VOCABULARY",
    "UNIT_ALIASES",
    "extract_numbers",
    "extract_dates",
    "extract_quantities",
    "extract_entities",
    "content_tokens",
    "abstention_phrase",
    "is_abstention_shaped",
    "leaked_scaffolding",
    "control_characters",
    "looks_multi_sentence",
    "edit_distance",
    "similarity_ratio",
    "length_ratio",
    "salient_values",
]


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

# Function words and common sentence starters. Applied SYMMETRICALLY to
# candidate, original and evidence, so any systematic bias cancels out.
STOPWORDS: FrozenSet[str] = frozenset(
    """
    a an the this that these those it its itself they them their there here
    is are was were be been being am do does did doing done have has had having
    of in on at to for from by with about into over under between during before
    after above below up down out off again further then once
    and or but if while although though because so than as
    i you he she we who whom whose which what when where why how
    not no nor only own same too very can will just should now
    you're your yours our ours my mine his hers
    may might must shall would could
    all any both each few more most other some such
    per via using used use also however therefore thus hence
    """.split()
)

# The trained abstention shape (edge_cases.jsonl #4:
# "Current evidence is insufficient to support this claim."). These tokens are
# the model's SAFETY vocabulary, not hallucinated content, so they are excluded
# from the unsupported-addition comparison when a candidate is a legitimate
# abstention. benchmark_engine.py hardcodes the same words into its stopword
# list for the same reason.
ABSTENTION_VOCABULARY: FrozenSet[str] = frozenset(
    """
    current currently evidence evidences insufficient sufficient support
    supported supports supporting claim claims verify verified verifiable
    unverified available unable cannot could not establish established
    confirm confirmed determine determined data information source sources
    this the is are to be for of no none any at present time
    """.split()
)

# Sentence-initial words that are capitalized by grammar rather than because
# they name something. Dropping these prevents "You"/"Current" being read as
# entities. Applied symmetrically.
_CAPITALIZED_NON_ENTITIES: FrozenSet[str] = frozenset(
    """
    the a an this that these those it they them there here you your we our i he
    she his her its and or but if when while although though because so however
    therefore thus also current currently no not none all any some each both
    every many most more less few several such other another next previous
    according based given despite before after during since until unless
    yes maybe perhaps possibly likely unlikely
    """.split()
)

# Unit aliases collapse surface variants onto one canonical unit, so "48 hours"
# and "48 h" produce the SAME quantity and a retained falsehood cannot be hidden
# behind an abbreviation.
UNIT_ALIASES: Dict[str, str] = {
    "h": "hour", "hr": "hour", "hrs": "hour", "hour": "hour", "hours": "hour",
    "min": "minute", "mins": "minute", "minute": "minute", "minutes": "minute",
    "s": "second", "sec": "second", "secs": "second",
    "second": "second", "seconds": "second",
    "d": "day", "day": "day", "days": "day",
    "wk": "week", "wks": "week", "week": "week", "weeks": "week",
    "mo": "month", "month": "month", "months": "month",
    "y": "year", "yr": "year", "yrs": "year", "year": "year", "years": "year",
    "pct": "percent", "percent": "percent", "percents": "percent", "%": "percent",
    "km": "kilometre", "kilometre": "kilometre", "kilometres": "kilometre",
    "kilometer": "kilometre", "kilometers": "kilometre",
    "m": "metre", "metre": "metre", "metres": "metre",
    "meter": "metre", "meters": "metre",
    "cm": "centimetre", "centimetre": "centimetre", "centimetres": "centimetre",
    "mm": "millimetre", "millimetre": "millimetre", "millimetres": "millimetre",
    "mi": "mile", "mile": "mile", "miles": "mile",
    "ft": "foot", "foot": "foot", "feet": "foot",
    "in": "inch", "inch": "inch", "inches": "inch",
    "kg": "kilogram", "kilogram": "kilogram", "kilograms": "kilogram",
    "g": "gram", "gram": "gram", "grams": "gram",
    "mg": "milligram", "milligram": "milligram", "milligrams": "milligram",
    "lb": "pound", "lbs": "pound", "pound": "pound", "pounds": "pound",
    "l": "litre", "litre": "litre", "litres": "litre",
    "liter": "litre", "liters": "litre",
    "ml": "millilitre", "millilitre": "millilitre", "millilitres": "millilitre",
    "usd": "usd", "dollar": "usd", "dollars": "usd", "$": "usd",
    "eur": "eur", "euro": "eur", "euros": "eur", "€": "eur",
    "gbp": "gbp", "pound_sterling": "gbp", "£": "gbp",
    "k": "thousand", "bn": "billion", "b": "billion", "mn": "million",
}

_MONTHS: Dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

_NUMBER_WORDS: Dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_SCALES: Dict[str, int] = {
    "hundred": 100, "thousand": 1000, "million": 1000000,
    "billion": 1000000000, "trillion": 1000000000000,
}


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

# Boundary-anchored: the lookbehind rejects a digit or '.' before the match and
# the lookahead rejects a word character after it, so "48" cannot be found
# inside "1948", "480" or "v1.48".
_NUMERIC_RE = re.compile(
    r"(?<![\w.])"
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?![\w])"
)


def _canonical_number(raw: str) -> str:
    """Canonical string for a numeric literal: '1,000' and '1000.0' -> '1000'."""
    cleaned = raw.replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return cleaned
    if value.is_integer():
        return str(int(value))
    # repr-free fixed formatting keeps 0.50 and 0.5 identical.
    return ("%.10f" % value).rstrip("0").rstrip(".")


def _word_number_values(text: str) -> Set[str]:
    """Canonical values for spelled-out numbers ('forty-eight' -> '48').

    Needed because a model can hide a retained falsehood by spelling a number
    out. Handles units, teens, tens, hyphenated compounds and the common scales.
    """
    words = re.findall(r"[a-z]+", normalize_for_matching(text).replace("-", " "))
    found: Set[str] = set()
    current = 0
    running = 0
    active = False

    for word in words:
        if word in _NUMBER_WORDS:
            current += _NUMBER_WORDS[word]
            active = True
        elif word in _NUMBER_SCALES:
            scale = _NUMBER_SCALES[word]
            if scale == 100:
                current = (current or 1) * 100
            else:
                running += (current or 1) * scale
                current = 0
            active = True
        elif word == "and" and active:
            continue
        else:
            if active:
                total = running + current
                if total:
                    found.add(str(total))
            current = 0
            running = 0
            active = False

    if active:
        total = running + current
        if total:
            found.add(str(total))
    return found


def extract_numbers(text: str) -> Set[str]:
    """Value-normalized numbers, digit and word forms alike.

    Word-boundary anchored, so no number is discovered inside a longer number.
    """
    if not text:
        return set()
    values = {_canonical_number(m.group(1)) for m in _NUMERIC_RE.finditer(text)}
    values |= _word_number_values(text)
    return {v for v in values if v}


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_ISO_RE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_SLASH_RE = re.compile(r"(?<!\d)(\d{1,2})[/.](\d{1,2})[/.](\d{2,4})(?!\d)")
_DASH_DMY_RE = re.compile(r"(?<!\d)(\d{1,2})-(\d{1,2})-(\d{4})(?!\d)")
_MONTH_NAME = "|".join(sorted(_MONTHS, key=len, reverse=True))
# The (?<!\d)/(?!\d) guards around the day group stop a 4-digit year being read
# as a 2-digit day: without them "May 2024" would parse as day=20.
_MONTH_FIRST_RE = re.compile(
    rf"\b({_MONTH_NAME})\b\.?\s+(?<!\d)(\d{{1,2}})(?!\d)(?:st|nd|rd|th)?"
    rf"\s*,?\s*(?:(\d{{4}})(?!\d))?",
    re.IGNORECASE,
)
_DAY_FIRST_RE = re.compile(
    rf"\b(?<!\d)(\d{{1,2}})(?!\d)(?:st|nd|rd|th)?\s+({_MONTH_NAME})\b\.?"
    rf"\s*,?\s*(?:(\d{{4}})(?!\d))?",
    re.IGNORECASE,
)
_BARE_YEAR_RE = re.compile(r"(?<![\w.])(1\d{3}|2\d{3})(?![\w])")


def _date_key(parts: Sequence[int]) -> str:
    """Order-insensitive component key.

    ``2024-05-03``, ``May 3, 2024`` and ``03/05/2024`` all collapse to the same
    key. Day/month ORDER is deliberately not distinguished: slash dates are
    genuinely ambiguous across locales, and treating 03/05 and 05/03 as one
    value avoids inventing a locale the request never specified. Documented as a
    known bound.
    """
    return "|".join(str(p) for p in sorted(int(p) for p in parts if p))


def extract_dates(text: str) -> Set[str]:
    """Component-normalized dates across ISO, slash, dash and month-name forms."""
    if not text:
        return set()
    keys: Set[str] = set()

    for match in _ISO_RE.finditer(text):
        keys.add(_date_key([int(match.group(1)), int(match.group(2)), int(match.group(3))]))
    for match in _DASH_DMY_RE.finditer(text):
        keys.add(_date_key([int(match.group(1)), int(match.group(2)), int(match.group(3))]))
    for match in _SLASH_RE.finditer(text):
        year = int(match.group(3))
        if year < 100:
            year += 2000 if year < 50 else 1900
        keys.add(_date_key([int(match.group(1)), int(match.group(2)), year]))

    for regex, day_group, month_group in (
        (_MONTH_FIRST_RE, 2, 1),
        (_DAY_FIRST_RE, 1, 2),
    ):
        for match in regex.finditer(text):
            month = _MONTHS.get(match.group(month_group).lower().rstrip("."))
            if month is None:
                continue
            day = int(match.group(day_group))
            year_raw = match.group(3)
            parts = [day, month] + ([int(year_raw)] if year_raw else [])
            keys.add(_date_key(parts))

    # Bare years are dates too, but only when not already part of a full date.
    consumed = set()
    for regex in (_ISO_RE, _SLASH_RE, _DASH_DMY_RE, _MONTH_FIRST_RE, _DAY_FIRST_RE):
        for match in regex.finditer(text):
            consumed.update(range(match.start(), match.end()))
    for match in _BARE_YEAR_RE.finditer(text):
        if match.start() not in consumed:
            keys.add(match.group(1))

    return {k for k in keys if k}


# ---------------------------------------------------------------------------
# Quantities (number + unit)
# ---------------------------------------------------------------------------

_UNIT_PATTERN = "|".join(
    sorted((re.escape(u) for u in UNIT_ALIASES), key=len, reverse=True)
)
_QUANTITY_RE = re.compile(
    rf"(?<![\w.])(\d+(?:[,.]\d+)*)\s*({_UNIT_PATTERN})(?![A-Za-z])",
    re.IGNORECASE,
)
_WORD_QUANTITY_RE = re.compile(
    rf"\b([a-z\-]+)\s+({_UNIT_PATTERN})(?![A-Za-z])", re.IGNORECASE
)


def extract_quantities(text: str) -> Set[Tuple[str, str]]:
    """``(canonical_value, canonical_unit)`` pairs.

    Unit aliasing means ``48 hours`` and ``48 h`` yield the identical pair, so a
    retained unsupported quantity cannot be disguised by abbreviating the unit.
    """
    if not text:
        return set()
    pairs: Set[Tuple[str, str]] = set()

    for match in _QUANTITY_RE.finditer(text):
        unit = UNIT_ALIASES.get(match.group(2).lower())
        if unit:
            pairs.add((_canonical_number(match.group(1)), unit))

    for match in _WORD_QUANTITY_RE.finditer(text):
        values = _word_number_values(match.group(1))
        unit = UNIT_ALIASES.get(match.group(2).lower())
        if unit and len(values) == 1:
            pairs.add((next(iter(values)), unit))

    return pairs


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

_CAP_SEQUENCE_RE = re.compile(
    r"\b([A-Z][\w'’]*(?:[\s\-](?:of|the|and|de|van|von|da|di)?[\s\-]?[A-Z][\w'’]*)*)"
)
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,})\b")


def extract_entities(text: str) -> Set[str]:
    """Casefolded proper-noun-ish spans and acronyms.

    Regex-based by design: spaCy is not installed and adding it would introduce
    a runtime dependency. Extraction is applied symmetrically to candidate,
    original and evidence, so systematic bias cancels. Residual error surfaces
    as extra rejections (the fail-safe direction), never as extra acceptances.
    """
    if not text:
        return set()
    found: Set[str] = set()

    for match in _CAP_SEQUENCE_RE.finditer(text):
        phrase = match.group(1).strip()
        words = [w for w in re.split(r"[\s\-]+", phrase) if w]
        keep = [w for w in words if w.lower() not in _CAPITALIZED_NON_ENTITIES]
        if not keep:
            continue
        # A lone capitalized function-ish word is not an entity.
        if len(keep) == 1:
            single = keep[0]
            lowered = single.lower()
            if lowered in _CAPITALIZED_NON_ENTITIES or lowered in STOPWORDS:
                continue
            if lowered in ABSTENTION_VOCABULARY:
                continue
            if lowered in _MONTHS:
                continue
            if single.isdigit():
                continue
        normalized = " ".join(keep).casefold()
        normalized = re.sub(r"[^\w\s'’]", "", normalized).strip()
        if normalized:
            found.add(normalized)

    for match in _ACRONYM_RE.finditer(text):
        token = match.group(1)
        if token.lower() not in _CAPITALIZED_NON_ENTITIES:
            found.add(token.casefold())

    return found


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

def content_tokens(text: str, extra_allowed: FrozenSet[str] = frozenset()) -> Set[str]:
    """Distinct meaning-bearing tokens: normalized, stopword- and digit-free."""
    if not text:
        return set()
    return {
        token
        for token in tokenize(text)
        if token not in STOPWORDS
        and token not in extra_allowed
        and not token.isdigit()
        and len(token) > 1
    }


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

_ABSTENTION_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("insufficient_evidence", re.compile(
        r"\b(?:current(?:ly)?\s+)?evidence\s+(?:is|was|remains)\s+insufficient\b")),
    # Bounded-gap variant for "evidence from 2024 is insufficient". The gap
    # excludes '.' so it cannot span a sentence boundary. Broadening SHAPE
    # detection is safe because legitimacy is decided separately: an abstention
    # that smuggles in an ungrounded value is rejected by `validators`
    # regardless of how confidently its shape matched.
    ("insufficient_evidence_gapped", re.compile(
        r"\bevidence\b[^.]{0,40}?\b(?:is|was|remains)\s+insufficient\b")),
    ("insufficient_evidence_to", re.compile(
        r"\binsufficient\s+(?:\w+\s+){0,2}evidence\b")),
    ("cannot_be_verified", re.compile(
        r"\b(?:can\s?not|cannot|could\s+not|couldn't)\s+be\s+verified\b")),
    ("unable_to_verify", re.compile(
        r"\b(?:unable|not\s+able)\s+to\s+(?:verify|confirm|establish|determine)\b")),
    ("no_evidence", re.compile(
        r"\bno\s+(?:available\s+|supporting\s+|reliable\s+)?evidence\b")),
    ("not_supported", re.compile(
        r"\bnot\s+supported\s+by\s+(?:the\s+)?(?:available\s+)?evidence\b")),
    ("does_not_establish", re.compile(
        r"\b(?:does|do|did)\s+not\s+(?:establish|support|confirm)\b")),
    ("not_verifiable", re.compile(r"\bnot\s+verifiable\b")),
)


def abstention_phrase(text: str) -> Optional[str]:
    """Name of the matched abstention pattern, or None.

    Shape detection ONLY. Whether an abstention is *legitimate* additionally
    requires that it asserts no new specific value; that combination is decided
    in :mod:`validators`, not here.
    """
    if not text:
        return None
    normalized = normalize_whitespace_case(text)
    for name, pattern in _ABSTENTION_PATTERNS:
        if pattern.search(normalized):
            return name
    return None


def is_abstention_shaped(text: str) -> bool:
    """True when the text matches the trained abstention vocabulary."""
    return abstention_phrase(text) is not None


# ---------------------------------------------------------------------------
# Structural hazards
# ---------------------------------------------------------------------------

_SCAFFOLD_MARKERS: Tuple[str, ...] = (
    "<<<", ">>>", "HG-DATA-", ":BEGIN ", ":END ",
    "=== ", "OUTPUT MANDATE", "(DATA)", "```",
    '"sentence_id"', '"corrected_sentence"',
    "sentence_id:", "ABSOLUTE RULES",
)


def leaked_scaffolding(text: str) -> List[str]:
    """Prompt-scaffolding markers echoed back inside a candidate sentence.

    A corrected sentence containing fence tags, section headers or the JSON
    envelope means the model reproduced its own prompt rather than rewriting the
    target. Returned in stable order so failure detail is deterministic.
    """
    if not text:
        return []
    return [marker for marker in _SCAFFOLD_MARKERS if marker in text]


def control_characters(text: str) -> List[str]:
    """Unicode escapes for control/format characters present in ``text``.

    Newlines and tabs count: a single corrected sentence must not carry line
    structure. Deterministic order (first appearance).
    """
    if not text:
        return []
    seen: List[str] = []
    for char in text:
        if char in ("\n", "\r", "\t") or unicodedata.category(char) in ("Cc", "Cf"):
            escaped = "\\u%04x" % ord(char)
            if escaped not in seen:
                seen.append(escaped)
    return seen


_TERMINATOR_RE = re.compile(r"([.!?]+)[\'\"’”\)\]]*\s+(?=[A-Z0-9“\"'])")


def looks_multi_sentence(text: str) -> bool:
    """True when ``text`` carries more than one sentence.

    Abbreviations and initials are excluded using the same abbreviation set as
    Step 2's segmenter, so "Dr. Smith founded it." is one sentence. Any newline
    is treated as multi-sentence structure.
    """
    if not text or not text.strip():
        return False
    if "\n" in text or "\r" in text:
        return True

    body = text.strip()
    for match in _TERMINATOR_RE.finditer(body):
        preceding = body[: match.start(1)].split()
        last = preceding[-1] if preceding else ""
        stripped = last.rstrip(".")
        if stripped.lower() in COMMON_ABBREVIATIONS:
            continue
        if len(stripped) == 1 and stripped.isalpha():
            continue
        # Decimal like "3.14 Something" is not a boundary.
        if match.start(1) > 0 and body[match.start(1) - 1].isdigit():
            following = body[match.end(1):].lstrip()
            if following[:1].isdigit():
                continue
        return True
    return False


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def edit_distance(left: str, right: str) -> int:
    """Levenshtein distance, pure Python, O(n*m) time and O(min) space."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def similarity_ratio(left: str, right: str) -> float:
    """Normalized similarity in [0, 1] over whitespace/case-folded text.

    NAMING: this is similarity of the candidate to the ORIGINAL sentence, which
    is computable at runtime. It is NOT ``benchmark_engine.minimal_edit_score``,
    which measures similarity to the GOLD sentence and needs a label the
    Corrector does not have in production. Same idea, different reference point;
    the two must never be conflated.
    """
    left_norm = normalize_whitespace_case(left)
    right_norm = normalize_whitespace_case(right)
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    return difflib.SequenceMatcher(None, left_norm, right_norm).ratio()


def length_ratio(candidate: str, original: str) -> float:
    """Candidate length as a multiple of the original's. 0.0 guards empty input."""
    base = len(normalize_whitespace_case(original))
    if base == 0:
        return 0.0
    return len(normalize_whitespace_case(candidate)) / base


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def salient_values(text: str) -> Dict[str, Set[str]]:
    """All specific, checkable values in one pass.

    "Salient" means a value a reader would treat as a fact: a number, a date, a
    quantity with a unit, or a named entity. These are what a correction must
    either ground in evidence or remove — vague prose carries no such burden.
    """
    return {
        "numbers": extract_numbers(text),
        "dates": extract_dates(text),
        "quantities": {f"{value} {unit}" for value, unit in extract_quantities(text)},
        "entities": extract_entities(text),
    }
