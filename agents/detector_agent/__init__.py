"""
HalluciGuard Detector Agent Package — HaluEval Classifier Edition.

The Detector Agent estimates hallucination probability using a fine-tuned
DistilBERT classifier trained on the HaluEval dataset.
"""

import os
import sys

# Critical Windows Environment & DLL Initialization for PyTorch / C++ extensions
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Derive torch native lib dir dynamically instead of hardcoding machine paths.
if sys.platform == "win32":
    try:
        from .utils import init_dll_paths

        init_dll_paths()
    except Exception:
        pass

from .config import DetectorConfig
from .detector import DetectorAgent
from .model_manager import ModelManager
from .models import (
    DetectionInput,
    DetectionResult,
    NextAction,
    RiskLevel,
)

__all__ = [
    "DetectorAgent",
    "DetectorConfig",
    "ModelManager",
    "DetectionInput",
    "DetectionResult",
    "RiskLevel",
    "NextAction",
]
