from __future__ import annotations
import re
import unicodedata

import logging

class ClaimNormalizer:
    """Normalizes claim text for consistent processing."""

    def __init__(self) -> None:
        self.filler_words = re.compile(r'(?i)\b(?:basically|essentially|actually|literally)\b')
        self.logger = logging.getLogger(__name__)

    def normalize(self, claim: str) -> str:
        """
        Normalize the claim text by stripping whitespace, normalizing unicode,
        lowercasing, and removing filler words.
        
        Args:
            claim: The claim text to normalize.
            
        Returns:
            The cleaned claim text.
        """
        if not claim:
            return ""

        original_claim = claim

        # Strip leading/trailing quotes
        claim = claim.strip("'\"")

        # Normalize unicode
        normalized = unicodedata.normalize('NFKC', claim)
        
        # Lowercase
        normalized = normalized.lower()
        
        # Remove filler words
        normalized = self.filler_words.sub('', normalized)
        
        # Strip extra whitespace
        normalized = ' '.join(normalized.split())
        
        if len(original_claim) > 0 and len(normalized) / len(original_claim) < 0.7:
            self.logger.debug("Normalization significantly changed text: '%s' -> '%s'", original_claim[:50], normalized[:50])

        return normalized
