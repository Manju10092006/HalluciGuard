from __future__ import annotations
from typing import List, Any

def precision_at_k(relevant: set, retrieved: List[Any], k: int) -> float:
    if not retrieved or k <= 0:
        return 0.0
    k = min(k, len(retrieved))
    retrieved_k = set(retrieved[:k])
    intersection = relevant.intersection(retrieved_k)
    return len(intersection) / k

def recall_at_k(relevant: set, retrieved: List[Any], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    k = min(k, len(retrieved))
    retrieved_k = set(retrieved[:k])
    intersection = relevant.intersection(retrieved_k)
    return len(intersection) / len(relevant)

def mean_reciprocal_rank(rankings: List[List[Any]], relevant_sets: List[set]) -> float:
    if not rankings or len(rankings) != len(relevant_sets):
        return 0.0
    
    total_rr = 0.0
    for ranking, relevant in zip(rankings, relevant_sets):
        for i, item in enumerate(ranking):
            if item in relevant:
                total_rr += 1.0 / (i + 1)
                break
    return total_rr / len(rankings)

def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def accuracy(correct: int, total: int) -> float:
    if total == 0:
        return 0.0
    return correct / total
