from __future__ import annotations
import logging
from typing import List

from schemas.models import Passage
from .duplicate_remover import DuplicateRemover

logger = logging.getLogger(__name__)

class EvidenceAggregator:
    """Aggregates and deduplicates passages from multiple sources."""

    def __init__(self, overlap_threshold: float = 0.85) -> None:
        self.duplicate_remover = DuplicateRemover(overlap_threshold=overlap_threshold)

    def aggregate(self, all_passages: List[List[Passage]]) -> List[Passage]:
        """
        Flatten, deduplicate, and sort passages by relevance score.
        
        Args:
            all_passages: List of passage lists from different sources.
            
        Returns:
            Merged, deduplicated, and sorted list of passages.
        """
        # Flatten
        flat_list = [passage for sublist in all_passages for passage in sublist]
        
        # Deduplicate
        deduped = self.duplicate_remover.remove_duplicates(flat_list)
        
        logger.info("Aggregation deduped passages from %d down to %d", len(flat_list), len(deduped))
        
        # Sort by relevance_score descending
        deduped.sort(key=lambda p: p.relevance_score if p.relevance_score is not None else 0.0, reverse=True)
        
        return deduped
