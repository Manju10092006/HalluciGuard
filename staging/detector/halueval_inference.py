"""
HaluEval Inference Module — Staging Runtime.

Loads the fine-tuned HaluEval detector model from Hugging Face or disk
and provides high-performance inference with structured logging.
"""

import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    HAS_TORCH = False

from .halueval_dataset import format_detector_input

logger = logging.getLogger("halluciguard.detector")

# Default Hugging Face repository
DEFAULT_MODEL_PATH = "Manjunath2000006/halluciguard-detector"


def _looks_like_local_path(value: str) -> bool:
    """Return True when a model reference should be validated as a local filesystem path."""
    path = Path(value).expanduser()
    if path.exists():
        return True
    if value.startswith((".", "/", "~", "\\")):
        return True
    if os.name == "nt" and len(value) > 1 and value[1] == ":":
        return True
    return False


def validate_halueval_model_reference(model_ref: str) -> str:
    """Validate and normalize a HaluEval model reference.

    Local model directories must contain standard Hugging Face files.
    Non-path values are treated as Hugging Face Hub identifiers.
    """
    if not model_ref or not model_ref.strip():
        raise ValueError(
            "HALUEVAL_MODEL_PATH is empty; set it to a local model directory or Hugging Face model id."
        )

    if model_ref.startswith("hf://"):
        return model_ref.removeprefix("hf://")

    if not _looks_like_local_path(model_ref):
        return model_ref

    resolved = Path(model_ref).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"HaluEval detector model directory not found at: {resolved}\n"
            "Set HALUEVAL_MODEL_PATH to a valid fine-tuned model directory or Hugging Face repo."
        )
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"HaluEval detector model path is not a directory: {resolved}"
        )

    required_any_weight = ["model.safetensors", "pytorch_model.bin"]
    missing = [name for name in ("config.json",) if not (resolved / name).exists()]
    if not any((resolved / name).exists() for name in required_any_weight):
        missing.append("model.safetensors or pytorch_model.bin")
    if not any((resolved / name).exists() for name in ("tokenizer.json", "vocab.txt")):
        missing.append("tokenizer.json or vocab.txt")
    if missing:
        raise FileNotFoundError(
            f"Invalid HaluEval detector model directory: {resolved}\n"
            f"Missing required Hugging Face artifact(s): {', '.join(missing)}"
        )
    return str(resolved)


@dataclass
class InferenceResult:
    """Raw inference output from the HaluEval model."""
    hallucination_probability: float
    confidence_score: float
    predicted_label: int  # 0 = NO_HALLUCINATION, 1 = HALLUCINATION
    predicted_label_name: str


class HaluEvalInference:
    """Loads and runs inference using the fine-tuned HaluEval detector model."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_length: int = 384,
        device: Optional[str] = None,
    ):
        self.model_path = model_path or self._resolve_model_path()
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            self.device = "cuda" if (HAS_TORCH and torch and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._loaded = False

    def _resolve_model_path(self) -> str:
        """Resolve model repository from environment or default."""
        env_path = os.environ.get("HALUEVAL_MODEL_PATH")
        if env_path and env_path.strip():
            resolved = validate_halueval_model_reference(env_path.strip())
            return resolved
        return DEFAULT_MODEL_PATH

    def load(self) -> None:
        """Load model and tokenizer from Hugging Face Hub or disk."""
        if self._loaded:
            return

        if not HAS_TORCH or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise RuntimeError(
                "torch and transformers are required to load HaluEval model. "
                "Please install torch and transformers."
            )

        self.model_path = validate_halueval_model_reference(self.model_path)

        print("[STARTUP] HalluciGuard Detector starting")
        print(f"[MODEL] Model reference: {self.model_path}")
        logger.info(f"[MODEL] Model reference: {self.model_path}")

        print("[MODEL] Loading tokenizer...")
        logger.info("[MODEL] Loading tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        print("[MODEL] Loading model...")
        logger.info("[MODEL] Loading model...")
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self._model.to(self.device)
        self._model.eval()

        print(f"[MODEL] Device: {self.device}")
        logger.info(f"[MODEL] Device: {self.device}")

        print("[MODEL] Model loaded successfully")
        logger.info("[MODEL] Model loaded successfully")
        self._loaded = True

    def predict(
        self,
        user_query: str,
        llm_response: str,
        context: Optional[str] = None,
    ) -> InferenceResult:
        """Run inference on a query-response pair with latency profiling."""
        if not self._loaded:
            self.load()

        start_time = time.perf_counter()

        logger.info("[REQUEST] Detection request received")
        logger.info(f"[INPUT] Query length: {len(user_query)}")
        logger.info(f"[INPUT] Response length: {len(llm_response)}")
        print(f"[REQUEST] Detection request received | Query len: {len(user_query)} | Response len: {len(llm_response)}")

        # Step 1: Formatting
        logger.info("[INFERENCE] Formatting detector input")
        text = format_detector_input(user_query, llm_response, context)

        # Step 2: Tokenizing & Device alignment (ZeroGPU support)
        logger.info("[INFERENCE] Tokenizing")
        current_device = "cuda" if (HAS_TORCH and torch and torch.cuda.is_available()) else self.device
        if self._model is not None and current_device != self.device:
            try:
                self._model.to(current_device)
                self.device = current_device
            except Exception as e:
                logger.warning(f"Could not transfer model to {current_device}: {e}")

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Step 3: Forward pass
        logger.info("[INFERENCE] Running HaluEval classifier")
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(f"[INFERENCE] Inference completed in {elapsed_ms:.2f} ms")
        print(f"[INFERENCE] Inference completed in {elapsed_ms:.2f} ms")

        no_halluc_prob = probs[0][0].item()
        halluc_prob = probs[0][1].item()
        predicted_label = 1 if halluc_prob > no_halluc_prob else 0
        label_name = "HALLUCINATION" if predicted_label == 1 else "NO_HALLUCINATION"

        result = InferenceResult(
            hallucination_probability=round(halluc_prob, 6),
            confidence_score=round(1.0 - halluc_prob, 6),
            predicted_label=predicted_label,
            predicted_label_name=label_name,
        )

        logger.info(
            f"[RESULT]\n"
            f"hallucination_probability={result.hallucination_probability:.4f}\n"
            f"confidence_score={result.confidence_score:.4f}\n"
            f"predicted_label={result.predicted_label}\n"
            f"predicted_label_name={result.predicted_label_name}"
        )
        print(
            f"[RESULT] hallucination_probability={result.hallucination_probability:.4f} "
            f"confidence_score={result.confidence_score:.4f} "
            f"label={result.predicted_label_name}"
        )

        return result

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded in memory."""
        return self._loaded
