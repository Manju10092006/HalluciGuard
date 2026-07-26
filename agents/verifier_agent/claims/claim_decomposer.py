from __future__ import annotations
import re

class ClaimDecomposer:
    """Decomposes complex claims into atomic sub-claims."""

    def __init__(self) -> None:
        # Strategy: Split compound claims on conjunctions and punctuation
        self.split_pattern = re.compile(
            r'(?i)\b(?:and|but|also|additionally|furthermore|moreover|as well as|plus|in addition)\b|[;.]'
        )
        self.list_pattern = re.compile(r'\d+\.\s+')

    def decompose(self, text: str) -> list[str]:
        """
        Decompose a compound claim into a list of atomic sub-claims.
        
        Args:
            text: The claim text to decompose.
            
        Returns:
            A list of atomic sub-claims.
        """
        if not text:
            return []
            
        # Handle numbered lists
        if self.list_pattern.search(text):
            parts = self.list_pattern.split(text)
            claims = [p.strip() for p in parts if p.strip()]
            return claims if claims else [text]

        # Split on conjunctions and punctuation
        parts = self.split_pattern.split(text)
        claims = [p.strip() for p in parts if p.strip()]
        
        if not claims:
            return [text]
            
        return claims
