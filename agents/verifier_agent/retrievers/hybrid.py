from __future__ import annotations
from typing import List

from schemas.models import Passage
from .sparse import BM25Retriever
from .dense import DenseRetriever

class HybridRetriever:
    """Combines BM25 and Dense retrievers using RRF."""

    def __init__(self) -> None:
        self.sparse = BM25Retriever()
        self.dense = DenseRetriever()
        
    def retrieve(self, query: str, passages: List[Passage], k: int = 5) -> List[Passage]:
        """
        Retrieve and fuse results using Reciprocal Rank Fusion (RRF).
        
        Args:
            query: The search query.
            passages: The pool of passages to retrieve from.
            k: The number of top passages to return.
            
        Returns:
            List of top-k Passages sorted by fused score.
        """
        if not passages:
            return []
            
        self.sparse.build_index(passages)
        self.dense.build_index(passages)
        
        sparse_results = self.sparse.retrieve(query, k * 2)
        dense_results = self.dense.retrieve(query, k * 2)
        
        # RRF Fusion
        # rrf_score = sum(1 / (rank + 60) for each system)
        rrf_scores: dict[str, float] = {}
        passage_map: dict[str, Passage] = {}
        
        for rank, (passage, _) in enumerate(sparse_results):
            passage_id = passage.source_id or passage.snippet
            passage_map[passage_id] = passage
            rrf_scores[passage_id] = rrf_scores.get(passage_id, 0.0) + 1.0 / (rank + 1 + 60)
            
        if self.dense._is_available:
            for rank, (passage, _) in enumerate(dense_results):
                passage_id = passage.source_id or passage.snippet
                passage_map[passage_id] = passage
                rrf_scores[passage_id] = rrf_scores.get(passage_id, 0.0) + 1.0 / (rank + 1 + 60)

                
        # Sort by fused score
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Deduplication (token overlap > 90%)
        final_results = []
        for passage_id, score in sorted_items:
            passage = passage_map[passage_id]
            
            # Check overlap
            is_duplicate = False
            for selected in final_results:
                if self._compute_overlap(passage.snippet, selected.snippet) > 0.90:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                final_results.append(passage)
                
            if len(final_results) >= k:
                break
                
        return final_results
        
    def _compute_overlap(self, text1: str, text2: str) -> float:
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        if not set1 or not set2:
            return 0.0
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)
