"""
HalluciGuard Detector Agent — Staging Package.

Provides fine-tuned HaluEval hallucination detection model inference
for staging deployment.
"""

import os
import sys

# Windows Environment & DLL Initialization for PyTorch / C++ extensions
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Derive torch native lib dir dynamically instead of hardcoding machine paths.
if sys.platform == "win32":
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and os.path.exists(os.path.dirname(torch.__file__)):
        try:
            os.add_dll_directory(os.path.dirname(torch.__file__))
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
