from __future__ import annotations
from typing import Dict, List

from schemas.models import Passage
from .duplicate_remover import DuplicateRemover

class EvidenceMerger:
    """Merges evidence from multiple sources with round-robin diversity."""

    def __init__(self) -> None:
        self.duplicate_remover = DuplicateRemover()

    def merge_from_sources(self, source_results: Dict[str, List[Passage]]) -> List[Passage]:
        """
        Merge results from different sources using round-robin interleaving.
        
        Args:
            source_results: Dictionary of source name to list of passages.
            
        Returns:
            Merged and deduplicated list of passages.
        """
        merged = []
        max_len = max((len(results) for results in source_results.values()), default=0)
        
        for i in range(max_len):
            for source, passages in source_results.items():
                if i < len(passages):
                    passage = passages[i]
                    # Tag with source_id if not already set
                    # Note: Using source_name as a fallback
                    if hasattr(passage, 'metadata'):
                        passage.metadata = passage.metadata or {}
                        passage.metadata['source_id'] = source
                    merged.append(passage)
                    
        return self.duplicate_remover.remove_duplicates(merged)
