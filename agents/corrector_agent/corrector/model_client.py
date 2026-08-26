"""Step 4 — production model client for the Corrector Agent.

THE NO-SILENT-FALLBACK RULE
---------------------------
If the fine-tuned corrector model (merged weights, or base + LoRA adapter)
cannot be loaded, this client:

* does NOT load an unrelated base model,
* does NOT produce a correction,
* does NOT report success,

and instead returns an explicit :class:`ModelStatus` with ``available=False`` and
a machine-readable :class:`ModelUnavailableReason` for Step 5 to map onto the
canonical degraded/fallback outcome.

Both existing clients in this repository violate this rule and are therefore NOT
adapted for their resolution logic:

* ``agents/corrector_agent/app/model_client.py`` (legacy) — when the merged path
  is absent it assigns the base model and continues, attaching the adapter only
  if it coincidentally exists.
* ``Corrector_agent-reference/app/model_client.py`` (read-only donor) — same
  shape, with a chain of guessed adapter directories.

Base-model use is reachable ONLY when an operator explicitly sets
``CorrectorConfig.allow_base_model_fallback=True`` (env
``HG_CORRECTOR_ALLOW_BASE_MODEL_FALLBACK``). It is then reported as
``ModelKind.BASE_MODEL_OPTIN`` — never as fine-tuned — so the distinction
survives into the result.

IMPORT SAFETY
-------------
``torch`` / ``transformers`` / ``peft`` are imported lazily inside
:meth:`ModelClient.ensure_loaded`. Importing this module never imports the ML
stack, so the Corrector package (and its test suite) works in environments where
those packages are not installed; a missing dependency becomes
``DEPENDENCY_MISSING``, not an ImportError at package import time.
"""

from __future__ import annotations

import json
import os
from typing import Callable, List, Optional, Sequence

from .config import CorrectorConfig
from .contracts import (
    CorrectionCandidate,
    GenerationOutcome,
    GroundedPlan,
    InternalCorrectionRequest,
    ModelKind,
    ModelStatus,
    ModelUnavailableReason,
    PromptBundle,
    RejectionReason,
)
from .output_parser import parse_candidate, rejected_candidate
from .prompt_builder import build_prompts

__all__ = [
    "REQUIRED_ML_PACKAGES",
    "Generator",
    "ModelClient",
    "generate_corrections",
]

REQUIRED_ML_PACKAGES = ("torch", "transformers")

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")
_ADAPTER_CONFIG = "adapter_config.json"
_MODEL_CONFIG = "config.json"


class Generator:
    """Seam for a text generator: ``generate(system_text, prompt_text) -> str``.

    Subclassed by :class:`_TransformersGenerator` in production and injected
    directly in tests, so Step 4 is testable without the ML stack installed.
    """

    kind: str = ModelKind.INJECTED_TEST_GENERATOR.value

    def generate(self, system_text: str, prompt_text: str) -> str:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Filesystem resolution (no imports, no side effects)
# ---------------------------------------------------------------------------

def _has_weights(path: str) -> bool:
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    return any(name.endswith(_WEIGHT_SUFFIXES) for name in entries)


def _is_adapter_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, _ADAPTER_CONFIG))


def _read_adapter_base(path: str) -> Optional[str]:
    """Return the adapter's declared base model, or None if unreadable."""
    try:
        with open(os.path.join(path, _ADAPTER_CONFIG), "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict):
        return None
    base = cfg.get("base_model_name_or_path")
    if isinstance(base, str) and base.strip():
        return base.strip()
    return None


def _unavailable(
    reason: ModelUnavailableReason,
    detail: str,
    tried: Sequence[str] = (),
    base_model: str = "",
) -> ModelStatus:
    return ModelStatus(
        available=False,
        kind="",
        reason=reason.value,
        detail=detail,
        tried_paths=list(tried),
        base_model=base_model,
    )


def resolve_model(config: CorrectorConfig) -> ModelStatus:
    """Resolve what is on disk WITHOUT importing or loading anything.

    Distinguishes, explicitly:
      * path missing entirely            -> MODEL_PATH_MISSING
      * merged dir present but no weights-> MODEL_ARTIFACTS_INCOMPLETE
      * adapter dir with unreadable cfg  -> ADAPTER_CONFIG_INVALID
      * adapter dir with no weight files -> ADAPTER_WEIGHTS_MISSING
    """
    path = config.model_path
    tried = [path]

    if not path or not os.path.isdir(path):
        if config.allow_base_model_fallback:
            return ModelStatus(
                available=True,
                kind=ModelKind.BASE_MODEL_OPTIN.value,
                resolved_path=config.base_model_name,
                base_model=config.base_model_name,
                reason="",
                detail=(
                    "fine-tuned corrector weights not found; using the base model "
                    "because allow_base_model_fallback was explicitly enabled"
                ),
                tried_paths=tried,
            )
        return _unavailable(
            ModelUnavailableReason.MODEL_PATH_MISSING,
            f"fine-tuned corrector model not found at {path!r}; refusing to "
            "substitute the base model (allow_base_model_fallback is False)",
            tried,
            config.base_model_name,
        )

    if _is_adapter_dir(path):
        base = _read_adapter_base(path)
        if base is None:
            return _unavailable(
                ModelUnavailableReason.ADAPTER_CONFIG_INVALID,
                f"{_ADAPTER_CONFIG} at {path!r} is unreadable or declares no "
                "base_model_name_or_path",
                tried,
                config.base_model_name,
            )
        if not _has_weights(path):
            return _unavailable(
                ModelUnavailableReason.ADAPTER_WEIGHTS_MISSING,
                f"adapter directory {path!r} has {_ADAPTER_CONFIG} but no weight "
                "files; refusing to run the bare base model",
                tried,
                base,
            )
        return ModelStatus(
            available=True,
            kind=ModelKind.BASE_PLUS_ADAPTER.value,
            resolved_path=base,
            adapter_path=path,
            base_model=base,
            tried_paths=tried,
        )

    if not _has_weights(path):
        return _unavailable(
            ModelUnavailableReason.MODEL_ARTIFACTS_INCOMPLETE,
            f"{path!r} exists but contains no model weight files "
            f"({', '.join(_WEIGHT_SUFFIXES)}) and no {_ADAPTER_CONFIG}",
            tried,
            config.base_model_name,
        )

    if not os.path.isfile(os.path.join(path, _MODEL_CONFIG)):
        return _unavailable(
            ModelUnavailableReason.MODEL_ARTIFACTS_INCOMPLETE,
            f"{path!r} has weight files but no {_MODEL_CONFIG}",
            tried,
            config.base_model_name,
        )

    return ModelStatus(
        available=True,
        kind=ModelKind.MERGED_FINETUNED.value,
        resolved_path=path,
        base_model=config.base_model_name,
        tried_paths=tried,
    )


def missing_dependencies() -> List[str]:
    """Return the required ML packages that are not importable."""
    import importlib.util

    missing: List[str] = []
    for name in REQUIRED_ML_PACKAGES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


# ---------------------------------------------------------------------------
# Production generator (lazy ML imports)
# ---------------------------------------------------------------------------

class _TransformersGenerator(Generator):
    """Wraps a loaded HF model + tokenizer behind the :class:`Generator` seam."""

    def __init__(self, model, tokenizer, config: CorrectorConfig, kind: str) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self.kind = kind

    def generate(self, system_text: str, prompt_text: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt_text},
        ]
        encoded = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(encoded, "keys"):
            inputs = {k: v.to(self._model.device) for k, v in encoded.items()}
            input_len = inputs["input_ids"].shape[-1]
            kwargs = dict(inputs)
        else:
            tensor = encoded.to(self._model.device)
            input_len = tensor.shape[-1]
            kwargs = {"input_ids": tensor}

        # Deterministic by default: greedy decoding. Sampling is only used when
        # the operator explicitly turns determinism off.
        if self._config.deterministic:
            gen_kwargs = {"do_sample": False}
        else:
            gen_kwargs = {"do_sample": True, "temperature": 0.1, "top_p": 0.9}

        with torch.no_grad():
            out = self._model.generate(
                **kwargs,
                max_new_tokens=self._config.max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                **gen_kwargs,
            )
        return self._tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ModelClient:
    """Loads the fine-tuned corrector model and generates per-target output.

    The client is inert until :meth:`ensure_loaded` is called; construction
    performs no filesystem or import work.
    """

    def __init__(
        self,
        config: Optional[CorrectorConfig] = None,
        generator: Optional[Generator] = None,
    ) -> None:
        self.config = config or CorrectorConfig()
        self._generator = generator
        self._status: Optional[ModelStatus] = None
        if generator is not None:
            self._status = ModelStatus(
                available=True,
                kind=getattr(generator, "kind", ModelKind.INJECTED_TEST_GENERATOR.value),
                resolved_path="<injected>",
                base_model=self.config.base_model_name,
                detail="using an injected generator; no model was loaded from disk",
            )

    @property
    def status(self) -> ModelStatus:
        return self._status or ModelStatus()

    def ensure_loaded(self) -> ModelStatus:
        """Resolve, then load. Returns the resolved status; never raises."""
        if self._status is not None:
            return self._status

        status = resolve_model(self.config)
        if not status.available:
            self._status = status
            return status

        missing = missing_dependencies()
        if missing:
            self._status = _unavailable(
                ModelUnavailableReason.DEPENDENCY_MISSING,
                "required package(s) not installed: " + ", ".join(missing),
                status.tried_paths,
                status.base_model,
            )
            return self._status

        try:
            self._generator = self._load(status)
        except Exception as exc:  # noqa: BLE001 - must degrade, never propagate
            reason = (
                ModelUnavailableReason.ADAPTER_LOAD_FAILED
                if status.kind == ModelKind.BASE_PLUS_ADAPTER.value
                else ModelUnavailableReason.MODEL_LOAD_FAILED
            )
            self._status = _unavailable(
                reason,
                f"{type(exc).__name__}: {exc}",
                status.tried_paths,
                status.base_model,
            )
            return self._status

        self._status = status
        return status

    def _load(self, status: ModelStatus) -> Generator:
        """Load tokenizer + model. Raises on failure; caller degrades."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer_src = status.adapter_path or status.resolved_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
        except Exception as exc:
            raise RuntimeError(
                f"tokenizer load failed from {tokenizer_src!r}: {exc}"
            ) from exc
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if torch.cuda.is_available():
            device_map = "auto"
            dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            device_map = None
            dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            status.resolved_path, torch_dtype=dtype, device_map=device_map
        )

        if status.kind == ModelKind.BASE_PLUS_ADAPTER.value:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, status.adapter_path)

        model.eval()
        return _TransformersGenerator(model, tokenizer, self.config, status.kind)

    def generate_for_prompt(self, bundle: PromptBundle) -> CorrectionCandidate:
        """Generate and parse one candidate for one prompt. Never raises."""
        if not bundle.build_ok:
            return rejected_candidate(
                bundle,
                RejectionReason.PROMPT_BUILD_FAILED,
                f"{bundle.build_error}: {bundle.build_detail}",
            )

        status = self.ensure_loaded()
        if not status.available or self._generator is None:
            return rejected_candidate(
                bundle,
                RejectionReason.GENERATION_FAILED,
                f"model unavailable ({status.reason}): {status.detail}",
            )

        try:
            raw = self._generator.generate(bundle.system_text, bundle.prompt_text)
        except Exception as exc:  # noqa: BLE001 - degrade, never propagate
            return rejected_candidate(
                bundle,
                RejectionReason.GENERATION_FAILED,
                f"{type(exc).__name__}: {exc}",
            )

        return parse_candidate(raw, bundle)


def generate_corrections(
    request: InternalCorrectionRequest,
    plan: GroundedPlan,
    config: Optional[CorrectorConfig] = None,
    generator: Optional[Generator] = None,
    client: Optional[ModelClient] = None,
    token_counter: Optional[Callable[[str], int]] = None,
) -> GenerationOutcome:
    """Step 4 entry point: build prompts, generate, parse. No validation, no splice.

    When the model is unavailable the prompts are still returned (they are
    inspectable diagnostics) but ``candidates`` is EMPTY — Step 4 never
    manufactures a correction it did not obtain from a real model.
    """
    cfg = config or CorrectorConfig()
    active = client or ModelClient(cfg, generator)

    prompts = build_prompts(request, plan, cfg, token_counter)
    warnings: List[str] = [
        f"prompt build failed for {b.sentence_id}: {b.build_error} ({b.build_detail})"
        for b in prompts
        if not b.build_ok
    ]

    status = active.ensure_loaded()
    if not status.available:
        warnings.append(
            f"correction model unavailable ({status.reason}): {status.detail}"
        )
        return GenerationOutcome(
            model_status=status, prompts=prompts, candidates=[], warnings=warnings
        )

    if status.kind == ModelKind.BASE_MODEL_OPTIN.value:
        warnings.append(
            "generating with the BASE model (operator opt-in); output is not "
            "from the fine-tuned corrector"
        )

    candidates = [active.generate_for_prompt(bundle) for bundle in prompts]
    return GenerationOutcome(
        model_status=status,
        prompts=prompts,
        candidates=candidates,
        warnings=warnings,
    )
