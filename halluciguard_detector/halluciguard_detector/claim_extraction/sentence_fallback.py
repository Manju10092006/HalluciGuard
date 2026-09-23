"""
Sentence-based fallback claim extractor.
Runs deterministically when LLM extraction fails or is unavailable.
"""

import re
from typing import List, Tuple

NON_FACTUAL_PATTERNS = [
    r"^(sure|certainly|of course|here is|here's|let me|i will|thank|please|happy to)",
    r"^\s*(yes|no|ok|okay)\s*[.,!]?\s*$",
    r"^(in (summary|conclusion|short)|to summarize|overall)",
    r"^(note|disclaimer|important|warning)\s*:",
]
NON_FACTUAL_RE = re.compile("|".join(NON_FACTUAL_PATTERNS), re.IGNORECASE)


def sentence_split(text: str) -> List[Tuple[int, str]]:
    """Split text into sentences with sentence index."""
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    results = []
    idx = 0
    for s in raw_sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        if len(s_clean) < 10 or NON_FACTUAL_RE.match(s_clean):
            continue
        results.append((idx, s_clean))
        idx += 1
    return results
