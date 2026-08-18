"""Prometheus instrumentation for the Memory Agent."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

store_total = Counter(
    "memory_store_total",
    "Total facts stored (or deduped)",
    ["verdict", "duplicate"],
)

recall_total = Counter(
    "memory_recall_total",
    "Total recall queries",
    ["cached", "reranked"],
)

cache_hit_total = Counter(
    "memory_cache_hit_total",
    "Exact cache hits",
    ["domain"],
)

cache_miss_total = Counter(
    "memory_cache_miss_total",
    "Exact cache misses",
    ["domain"],
)

contradiction_total = Counter(
    "memory_contradiction_total",
    "Contradictions detected on store",
    ["domain"],
)

request_duration = Histogram(
    "memory_request_duration_seconds",
    "HTTP request latency",
    ["endpoint"],
)

store_duration = Histogram(
    "memory_store_duration_seconds",
    "Fact store latency",
)

recall_duration = Histogram(
    "memory_recall_duration_seconds",
    "Recall query latency",
)
