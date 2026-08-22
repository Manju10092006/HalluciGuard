from .base_llm_service import (
    BaseLLMConfig,
    BaseLLMHealth,
    BaseLLMService,
    GenerationResult,
)


def __getattr__(name: str):
    if name in ("BaseLLMDetectorService", "LLMDetectorSliceResult"):
        from .llm_detector_service import BaseLLMDetectorService, LLMDetectorSliceResult
        return locals()[name]
    if name in ("BaseLLMDetectorVerifierService", "LLMDetectorVerifierSliceResult"):
        from .llm_detector_verifier_service import (
            BaseLLMDetectorVerifierService,
            LLMDetectorVerifierSliceResult,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseLLMConfig",
    "BaseLLMHealth",
    "BaseLLMService",
    "GenerationResult",
    "BaseLLMDetectorService",
    "LLMDetectorSliceResult",
    "BaseLLMDetectorVerifierService",
    "LLMDetectorVerifierSliceResult",
]

