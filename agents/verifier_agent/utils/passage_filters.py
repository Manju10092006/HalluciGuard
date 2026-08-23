"""Passage filtering utilities for retrieval quality control."""
from __future__ import annotations

import re
from typing import List

from schemas.models import Passage


def is_disambiguation_passage(passage: Passage) -> bool:
    """
    Detect Wikipedia disambiguation pages that are not useful factual evidence.

    Examples: "Paris (disambiguation)", URLs containing disambiguation markers.
    """
    title = (getattr(passage, "title", "") or "").lower()
    url = (getattr(passage, "url", "") or "").lower()
    snippet = (getattr(passage, "snippet", "") or "").lower()

    disambig_markers = (
        "(disambiguation)",
        "disambiguation page",
        "may refer to:",
        "can refer to:",
    )
    if any(m in title for m in disambig_markers):
        return True
    if "disambiguation" in url or "/disambiguation" in url:
        return True
    if title.endswith("disambiguation)") or re.search(r"\bdisambiguation\b", title):
        return True
    if snippet.startswith("article [") and "(disambiguation)" in snippet[:120]:
        return True
    return False


def filter_disambiguation_passages(
    passages: List[Passage],
    *,
    keep_if_only: bool = True,
) -> List[Passage]:
    """
    Remove disambiguation pages when factual alternatives exist.

    If every passage is a disambiguation page, retain them (``keep_if_only``)
    so the pipeline can still return UNVERIFIED rather than empty evidence.
    """
    if not passages:
        return []
    factual = [p for p in passages if not is_disambiguation_passage(p)]
    if factual:
        return factual
    return passages if keep_if_only else []
