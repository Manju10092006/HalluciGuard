from __future__ import annotations
from typing import List, Tuple
from schemas.models import Passage

class BM25Retriever:
    """Sparse retriever using BM25."""
    
    def __init__(self) -> None:
        self.bm25 = None
        self.passages: List[Passage] = []
        
    def build_index(self, passages: List[Passage]) -> None:
        """
        Tokenize snippets and build BM25Okapi index.
        
        Args:
            passages: List of Passage objects to index.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            import logging
            logging.warning("rank_bm25 not installed. BM25Retriever will not function properly.")
            return

        self.passages = passages
        tokenized_corpus = [p.snippet.lower().split() for p in passages]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int) -> List[Tuple[Passage, float]]:
        """
        Retrieve top k passages with BM25 scores.
        
        Args:
            query: The search query.
            k: The number of passages to retrieve.
            
        Returns:
            List of tuples containing (Passage, score).
        """
        if not self.bm25 or not self.passages:
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        scored_passages = [(self.passages[i], float(score)) for i, score in enumerate(scores)]
        scored_passages.sort(key=lambda x: x[1], reverse=True)
        
        return scored_passages[:k]
