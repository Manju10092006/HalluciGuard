"""LangGraph-based HalluciGuard five-agent orchestration layer."""

from .schemas import (
    ClaimReport,
    CorrectionRequest,
    CorrectionResult,
    DetectorResult,
    Evidence,
    ExecutionStatus,
    JudgeDecision,
    JudgeResult,
    MemoryResult,
    MemoryStatus,
    NextAction,
    ReverificationResult,
    RiskLevel,
    SeverityLevel,
    ValidationStatus,
    VerdictLabel,
    VerifierResult,
)
from .state import HalluciGuardState


def __getattr__(name: str):
    """
    Lazy-load graph functions to avoid circular imports and improve startup time.

    Args:
        name: The attribute name being accessed.

    Returns:
        The requested graph function if it exists in the lazy-loading set.

    Raises:
        AttributeError: If the requested attribute is not available in this module.
    """
    if name in {"build_verification_graph", "get_verification_graph", "run_verification"}:
        from .graph import (
            build_verification_graph,
            get_verification_graph,
            run_verification,
        )

        mapping = {
            "build_verification_graph": build_verification_graph,
            "get_verification_graph": get_verification_graph,
            "run_verification": run_verification,
        }
        return mapping[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "build_verification_graph",
    "get_verification_graph",
    "run_verification",
    "HalluciGuardState",
    "DetectorResult",
    "Evidence",
    "ClaimReport",
    "VerifierResult",
    "CorrectionRequest",
    "JudgeResult",
    "CorrectionResult",
    "ReverificationResult",
    "MemoryResult",
    "RiskLevel",
    "NextAction",
    "VerdictLabel",
    "JudgeDecision",
    "SeverityLevel",
    "ExecutionStatus",
    "MemoryStatus",
    "ValidationStatus",
]
