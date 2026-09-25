"""HalluciGuard reference-grounded detector and orchestration adapter."""

from .agent import DetectorAgent
from .detector import Detector

__all__ = ["Detector", "DetectorAgent"]
__version__ = "0.1.0"
