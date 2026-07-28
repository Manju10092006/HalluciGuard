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
        import logging
        logger = logging.getLogger(__name__)
        
        if not text:
            return []
        
        # Handle numbered lists
        if self.list_pattern.search(text):
            parts = self.list_pattern.split(text)
            claims = [p.strip() for p in parts if p.strip() and len(p.strip()) >= 10]
            if claims:
                logger.debug("Decomposed '%s' into %d sub-claims (numbered list)", text[:80], len(claims))
                return claims[:5]
        
        # Don't split short claims
        if len(text) < 80:
            return [text]
        
        # Split on conjunctions and punctuation, but not within quotes or parens
        # Remove content in quotes and parens temporarily to avoid splitting inside them
        protected = re.sub(r'"[^"]*"', lambda m: '□' * len(m.group()), text)
        protected = re.sub(r'\([^)]*\)', lambda m: '□' * len(m.group()), protected)
        
        parts = self.split_pattern.split(protected)
        
        # Map back to original text positions
        claims = []
        pos = 0
        for part in parts:
            part_stripped = part.strip()
            if not part_stripped:
                pos += len(part) + 1  # +1 for the split character
                continue
            # Find the corresponding substring in the original text
            start = text.find(part_stripped.replace('□', ''), pos)
            if start == -1:
                start = pos
            end = start + len(part_stripped)
            original_part = text[start:end].strip()
            if len(original_part) >= 10:
                claims.append(original_part)
            pos = end
        
        if not claims:
            return [text]
        
        logger.debug("Decomposed '%s' into %d sub-claims", text[:80], len(claims))
        return claims[:5]  # Cap at 5 sub-claims
