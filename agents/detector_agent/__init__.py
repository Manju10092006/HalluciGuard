import os
import sys

# Critical Windows Environment & DLL Initialization for PyTorch / C++ extensions
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    known_dll_dirs = [
        r"C:\ProgramData\anaconda3\Library\bin",
        r"C:\ProgramData\anaconda3\DLLs",
        r"C:\Users\prane\AppData\Roaming\Python\Python310\site-packages\torch\lib",
    ]
    for d in known_dll_dirs:
        if os.path.exists(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass

from .classifier import QueryCategory, classify_query
from .config import DetectorConfig
from .detector import DetectorAgent
from .gate import SelfConsistencyGate
from .model_manager import ModelManager
from .models import (
    DetectionInput,
    DetectionResult,
    EntropyMetrics,
    NextAction,
    RiskLevel,
    SemanticSimilarityMetrics,
    SelfConsistencyMetrics,
    SignalMetricsDetail,
    TokenProbabilityMetrics,
)
from .signals.entropy import EntropyCalculator
from .signals.semantic_similarity import SemanticSimilarityCalculator
from .signals.self_consistency import SelfConsistencyCalculator

__all__ = [
    "DetectorAgent",
    "ModelManager",
    "EntropyCalculator",
    "SemanticSimilarityCalculator",
    "SelfConsistencyCalculator",
    "SelfConsistencyGate",
    "QueryCategory",
    "classify_query",
    "DetectionInput",
    "DetectionResult",
    "TokenProbabilityMetrics",
    "EntropyMetrics",
    "SemanticSimilarityMetrics",
    "SelfConsistencyMetrics",
    "SignalMetricsDetail",
    "RiskLevel",
    "NextAction",
    "DetectorConfig",
]
