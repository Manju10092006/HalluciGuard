from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeValidationResult:
    ok: bool
    component: str
    detail: str
    metadata: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "component": self.component,
            "detail": self.detail,
            "metadata": self.metadata,
        }


def validate_detector_model_reference() -> RuntimeValidationResult:
    """Validate that Detector configuration points to a loadable real model artifact."""
    from agents.detector_agent.config import DetectorConfig
    from agents.detector_agent.halueval_inference import (
        validate_halueval_model_reference,
    )

    cfg = DetectorConfig()
    try:
        resolved = validate_halueval_model_reference(cfg.halueval_model_path)
    except Exception as exc:
        return RuntimeValidationResult(
            ok=False,
            component="detector",
            detail=f"{type(exc).__name__}: {exc}",
            metadata={"configured_model_path": cfg.halueval_model_path},
        )
    return RuntimeValidationResult(
        ok=True,
        component="detector",
        detail="Detector model reference is structurally valid.",
        metadata={
            "configured_model_path": cfg.halueval_model_path,
            "resolved_model_reference": resolved,
        },
    )


def validate_orchestration_startup() -> dict[str, Any]:
    detector = validate_detector_model_reference()
    return {
        "ok": detector.ok,
        "checks": [detector.model_dump()],
        "action_required": (
            None
            if detector.ok
            else "Provide HALUEVAL_MODEL_PATH or train the detector artifact before running real E2E."
        ),
    }
