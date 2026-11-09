from __future__ import annotations

import os
import sys
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ComponentCheckResult:
    """Result of a runtime component validation check."""
    ok: bool
    component: str
    detail: str
    metadata: Dict[str, Any]

    def model_dump(self) -> Dict[str, Any]:
        """Convert the check result to a dictionary."""
        return {
            "ok": self.ok,
            "component": self.component,
            "detail": self.detail,
            "metadata": self.metadata,
        }


def validate_openrouter_configuration() -> ComponentCheckResult:
    """Validate the configured provider router (legacy function name retained)."""
    from services.llm_providers import build_provider_specs, resolve_provider_order
    try:
        order = resolve_provider_order()
        specs = build_provider_specs()
    except ValueError as exc:
        return ComponentCheckResult(False, "base_llm", str(exc), {})
    ready = [name for name in order if specs[name].has_key and specs[name].model]
    return ComponentCheckResult(
        ok=bool(ready),
        component="base_llm",
        detail="LLM provider configuration is ready." if ready else "No configured LLM provider has a key and model.",
        metadata={
            "provider": ready[0] if ready else None,
            "order": order,
            "providers": {name: specs[name].safe_summary() for name in order},
            "key_configured": bool(ready),
            "validation_scope": "configuration_only",
        },
    )


def validate_detector_model_reference() -> ComponentCheckResult:
    """Validate the local reference-grounded detector package and artifacts."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    configured = os.environ.get("HALLUCIGUARD_DETECTOR_MODEL", "").strip()
    model_dir = Path(configured) if configured else repo_root / "artifacts" / "detector-best"
    if not model_dir.is_absolute():
        model_dir = repo_root / model_dir
    required = (
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "calibration.json",
    )
    try:
        from halluciguard_detector import DetectorAgent  # noqa: F401
    except Exception as exc:
        return ComponentCheckResult(
            ok=False,
            component="detector",
            detail=f"HalluciGuard detector package is not importable: {type(exc).__name__}: {exc}",
            metadata={"detector": "halluciguard_detector", "model_dir": str(model_dir)},
        )

    missing = [name for name in required if not (model_dir / name).is_file()]
    metadata: Dict[str, Any] = {
        "detector": "halluciguard_detector",
        "model_dir": str(model_dir),
        "model_source": "local_safetensors",
        "required_files": list(required),
        "missing_files": missing,
        "grounding_required": True,
    }
    if missing:
        return ComponentCheckResult(
            ok=False,
            component="detector",
            detail=f"Detector artifact directory is incomplete; missing: {', '.join(missing)}",
            metadata=metadata,
        )
    try:
        with (model_dir / "model.safetensors").open("rb") as weight:
            if weight.read(128).startswith(b"version https://git-lfs.github.com/spec"):
                raise ValueError("Model weights are a Git LFS pointer; run git lfs pull")
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        calibration = json.loads((model_dir / "calibration.json").read_text(encoding="utf-8"))
        temperature = float(calibration["temperature"])
        threshold = float(calibration["hallucination_threshold"])
        max_length = int(calibration["max_length"])
        if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Invalid detector calibration")
        if not 8 <= max_length <= config["max_position_embeddings"]:
            raise ValueError("Invalid detector context length")
        if set(config["id2label"].values()) != {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"}:
            raise ValueError("Detector labels do not match the grounded contract")
    except (ValueError, KeyError, OSError, TypeError) as exc:
        return ComponentCheckResult(False, "detector", f"Invalid detector artifact: {exc}", metadata)
    metadata["validation_scope"] = "local_files_and_configuration_only"
    return ComponentCheckResult(
        ok=True,
        component="detector",
        detail="Reference-grounded detector package and local artifacts are ready.",
        metadata=metadata,
    )


def validate_verifier_configuration() -> ComponentCheckResult:
    """Validate Verifier pipeline and NLI configuration."""
    try:
        verifier_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "agents", "verifier_agent")
        )
        if verifier_dir not in sys.path:
            sys.path.insert(0, verifier_dir)
        from config.settings import get_settings
        settings = get_settings()
        nli_model = os.environ.get("HALLUCIGUARD_NLI_MODEL_PATH") or settings.nli_model
        return ComponentCheckResult(
            ok=True,
            component="verifier",
            detail="Verifier agent pipeline and NLI settings are valid.",
            metadata={
                "nli_model": nli_model,
                "allow_model_downloads": settings.allow_model_downloads,
            },
        )
    except Exception as exc:
        return ComponentCheckResult(
            ok=False,
            component="verifier",
            detail=f"{type(exc).__name__}: {exc}",
            metadata={"error": str(exc)},
        )


def validate_memory_configuration() -> ComponentCheckResult:
    """Validate Memory agent dependencies and vector store accessibility."""
    try:
        from agents.memory_agent.memory.memory_agent import MemoryAgent
        return ComponentCheckResult(
            ok=True,
            component="memory",
            detail="Memory agent dependencies and FAISS storage are available.",
            metadata={"persistence": "local_or_ephemeral"},
        )
    except Exception as exc:
        return ComponentCheckResult(
            ok=False,
            component="memory",
            detail=f"{type(exc).__name__}: {exc}",
            metadata={"error": str(exc)},
        )


def validate_judge_configuration() -> ComponentCheckResult:
    """Validate that the canonical Judge is importable."""
    try:
        from agents.judge_agent.judge_agent import JudgeAgent
        return ComponentCheckResult(
            ok=callable(JudgeAgent),
            component="judge",
            detail="Canonical Judge agent is available.",
            metadata={"implementation": "agents.judge_agent.judge_agent.JudgeAgent"},
        )
    except Exception as exc:
        return ComponentCheckResult(
            ok=False,
            component="judge",
            detail=f"{type(exc).__name__}: {exc}",
            metadata={"error": str(exc)},
        )


def validate_corrector_configuration() -> ComponentCheckResult:
    """Validate the configured correction generator without loading model weights."""
    provider = os.environ.get("HG_CORRECTOR_PROVIDER", "openrouter").strip().lower()
    try:
        from agents.corrector_agent.corrector import CorrectorAgent
        if provider == "openrouter":
            key_ready = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
            return ComponentCheckResult(
                ok=key_ready and callable(CorrectorAgent),
                component="corrector",
                detail="OpenRouter-backed Corrector is ready." if key_ready else "OPENROUTER_API_KEY is required by the configured Corrector.",
                metadata={"provider": provider, "key_configured": key_ready},
            )
        from agents.corrector_agent.corrector.config import CorrectorConfig
        from agents.corrector_agent.corrector.model_client import resolve_model
        status = resolve_model(CorrectorConfig.from_env())
        return ComponentCheckResult(
            ok=bool(status.available),
            component="corrector",
            detail=status.detail or "Local Corrector model is available.",
            metadata={"provider": "local", "model_status": status.model_dump()},
        )
    except Exception as exc:
        return ComponentCheckResult(
            ok=False,
            component="corrector",
            detail=f"{type(exc).__name__}: {exc}",
            metadata={"provider": provider, "error": str(exc)},
        )
def validate_orchestration_startup() -> Dict[str, Any]:
    """Run comprehensive validation checks for all production components."""
    openrouter = validate_openrouter_configuration()
    detector = validate_detector_model_reference()
    verifier = validate_verifier_configuration()
    judge = validate_judge_configuration()
    corrector = validate_corrector_configuration()
    memory = validate_memory_configuration()

    checks = [openrouter, detector, verifier, judge, corrector, memory]
    all_ok = all(c.ok for c in checks)

    active_agents = ["base_llm", "detector", "verifier", "judge", "corrector", "reverifier", "memory"]
    disabled_agents: List[str] = []

    return {
        "ok": all_ok,
        "validation_scope": "configuration_only_not_inference_or_retrieval",
        "backend_status": "healthy" if all_ok else "degraded",
        "environment": os.environ.get("HALLUCIGUARD_ENV", "production"),
        "active_agents": active_agents,
        "disabled_agents": disabled_agents,
        "checks": [c.model_dump() for c in checks],
        "base_llm": openrouter.metadata,
        "detector": detector.metadata,
        "verifier": verifier.metadata,
        "judge": judge.metadata,
        "corrector": corrector.metadata,
        "reverifier": verifier.metadata,
        "memory": memory.metadata,
    }
