from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Dict, List, Generator

from schemas.models import PipelineStageStatus

class PerformanceTracker:
    """Tracks execution time for different pipeline stages."""

    def __init__(self) -> None:
        self.timings: Dict[str, int] = {}
        self.start_time = time.time()

    @contextmanager
    def track(self, stage_name: str) -> Generator[None, None, None]:
        """Context manager to time a specific stage."""
        stage_start = time.time()
        try:
            yield
        finally:
            stage_end = time.time()
            duration_ms = int((stage_end - stage_start) * 1000)
            
            if stage_name in self.timings:
                self.timings[stage_name] += duration_ms
            else:
                self.timings[stage_name] = duration_ms

    def get_stage_timings(self) -> Dict[str, int]:
        """Get raw timings dictionary."""
        return dict(self.timings)

    def get_total_ms(self) -> int:
        """Get total milliseconds since tracker creation."""
        return int((time.time() - self.start_time) * 1000)

    def to_pipeline_stages(self) -> List[PipelineStageStatus]:
        """Convert recorded timings to PipelineStageStatus schema objects."""
        return [
            PipelineStageStatus(stage_name=name, duration_ms=duration)
            for name, duration in self.timings.items()
        ]
