from abc import ABC, abstractmethod
from typing import List, Optional
from .benchmark_example import BenchmarkExample


class BaseDatasetLoader(ABC):
    """Abstract base class for all benchmark dataset loaders following SOLID principles."""

    @abstractmethod
    def load_dataset(self, limit: Optional[int] = None) -> List[BenchmarkExample]:
        """Loads and standardizes a benchmark dataset into a list of BenchmarkExample objects.
        
        Args:
            limit: Optional maximum number of examples to load.
            
        Returns:
            List[BenchmarkExample]: Standardized list of benchmark items.
        """
        pass
