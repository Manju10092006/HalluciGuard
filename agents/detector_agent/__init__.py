"""
HalluciGuard Detector Agent Package — HaluEval Classifier Edition.

The Detector Agent estimates hallucination probability using a fine-tuned
DistilBERT classifier trained on the HaluEval dataset.
"""

import os
import sys

# Critical Windows Environment & DLL Initialization for PyTorch / C++ extensions
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    known_dll_dirs = [
        r"C:\ProgramData\anaconda3\Library\bin",
        r"C:\ProgramData\anaconda3\DLLs",
    ]
    for d in known_dll_dirs:
        if os.path.exists(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass

from .config import DetectorConfig
from .detector import DetectorAgent
from .models import (
    DetectionInput,
    DetectionResult,
    NextAction,
    RiskLevel,
)

__all__ = [
    "DetectorAgent",
    "DetectorConfig",
    "DetectionInput",
    "DetectionResult",
    "RiskLevel",
    "NextAction",
]
