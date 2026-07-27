from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BenchmarkExample:
    """Standardized research-grade benchmark example data structure across HalluciGuard agents."""

    query: str
    response: str
    expected_risk: str  # "LOW", "MEDIUM", or "HIGH"
    category: str
    context: Optional[str] = None
    source_dataset: str = "HaluEval"
    source_model: Optional[str] = "ChatGPT"
    pair_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert BenchmarkExample instance to dictionary representation."""
        return {
            "query": self.query,
            "response": self.response,
            "expected_risk": self.expected_risk,
            "category": self.category,
            "context": self.context,
            "source_dataset": self.source_dataset,
            "source_model": self.source_model,
            "pair_id": self.pair_id,
            "metadata": self.metadata or {}
        }
