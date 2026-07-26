from __future__ import annotations
import threading
from typing import Dict, Any

class MetricsCollector:
    """Thread-safe collector for pipeline performance metrics."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._reset_internal()

    def _reset_internal(self) -> None:
        self.total_requests = 0
        self.total_latency_ms = 0
        
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.nli_calls = 0
        self.nli_total_latency = 0
        
        self.retrieval_calls = 0
        self.retrieval_total_latency = 0
        self.retrieval_total_results = 0
        
        self.adapters: Dict[str, Dict[str, int]] = {}

    def record_request(self, domain: str, latency_ms: int, success: bool) -> None:
        """Record a top-level verification request."""
        with self.lock:
            self.total_requests += 1
            if success:
                self.total_latency_ms += latency_ms

    def record_adapter_call(self, adapter_name: str, latency_ms: int, success: bool) -> None:
        """Record a call to an external retriever adapter."""
        with self.lock:
            if adapter_name not in self.adapters:
                self.adapters[adapter_name] = {'calls': 0, 'failures': 0, 'total_latency': 0}
            
            self.adapters[adapter_name]['calls'] += 1
            if not success:
                self.adapters[adapter_name]['failures'] += 1
            else:
                self.adapters[adapter_name]['total_latency'] += latency_ms

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self.lock:
            self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self.lock:
            self.cache_misses += 1

    def record_nli_inference(self, latency_ms: int) -> None:
        """Record an NLI model inference call."""
        with self.lock:
            self.nli_calls += 1
            self.nli_total_latency += latency_ms

    def record_retrieval(self, latency_ms: int, num_results: int) -> None:
        """Record a local retrieval operation."""
        with self.lock:
            self.retrieval_calls += 1
            self.retrieval_total_latency += latency_ms
            self.retrieval_total_results += num_results

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected metrics."""
        with self.lock:
            avg_latency = self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0
            
            total_cache = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_cache if total_cache > 0 else 0
            
            nli_avg = self.nli_total_latency / self.nli_calls if self.nli_calls > 0 else 0
            retrieval_avg = self.retrieval_total_latency / self.retrieval_calls if self.retrieval_calls > 0 else 0
            
            adapter_stats = {}
            for name, stats in self.adapters.items():
                successful_calls = stats['calls'] - stats['failures']
                avg_lat = stats['total_latency'] / successful_calls if successful_calls > 0 else 0
                adapter_stats[name] = {
                    'calls': stats['calls'],
                    'failures': stats['failures'],
                    'avg_latency': avg_lat
                }
                
            return {
                'total_requests': self.total_requests,
                'avg_latency_ms': avg_latency,
                'adapter_stats': adapter_stats,
                'cache_hit_rate': hit_rate,
                'nli_avg_latency': nli_avg,
                'retrieval_avg_latency': retrieval_avg
            }

    def reset(self) -> None:
        """Reset all counters to zero."""
        with self.lock:
            self._reset_internal()
