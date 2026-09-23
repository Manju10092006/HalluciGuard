"""
Deterministic claim validator.
Catches hallucinated entities/numbers added by the extractor, validates sentence index, etc.
"""

import re
from typing import List, Tuple

NUM_RE = re.compile(r"\b\d+[\.,]?\d*\b")
ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


def validate_extracted_claim(claim_text: str, source_text: str, query_text: str) -> Tuple[bool, List[str]]:
    """
    Validate that numbers and entities in claim text exist in source_text or query_text.
    Returns (is_valid, list_of_violation_flags).
    """
    combined_source = (source_text + " " + query_text).lower()
    flags = []

    # Check numbers
    claim_nums = NUM_RE.findall(claim_text)
    for num in claim_nums:
        if num.lower() not in combined_source:
            flags.append(f"EXTRACTOR_HALLUCINATED_NUM_{num}")

    # Check capitalized entities
    claim_entities = ENTITY_RE.findall(claim_text)
    for ent in claim_entities:
        if ent.lower() not in combined_source:
            flags.append(f"EXTRACTOR_HALLUCINATED_ENT_{ent}")

    is_valid = len(flags) == 0
    return is_valid, flags
