"""
HaluEval Inference Module.

Loads the fine-tuned HaluEval detector model from disk and provides
a simple predict() interface for the DetectorAgent.
"""

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# Default model artifact path — updated to final versioned directory
DEFAULT_MODEL_PATH = "artifacts/halueval-detector-final"
FALLBACK_MODEL_PATH = "artifacts/halueval-detector"

# Import the canonical formatter — single source of truth
from agents.detector_agent.halueval_dataset import format_detector_input



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
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._loaded = False

    def _resolve_model_path(self) -> str:
        """Try to find the model artifacts directory."""
        candidates = [
            DEFAULT_MODEL_PATH,
            FALLBACK_MODEL_PATH,
            os.path.join(os.path.dirname(__file__), "..", "..", DEFAULT_MODEL_PATH),
            os.path.join(os.path.dirname(__file__), "..", "..", FALLBACK_MODEL_PATH),
        ]

        env_path = os.environ.get("HALUEVAL_MODEL_PATH")
        if env_path:
            candidates.insert(0, env_path)

        for candidate in candidates:
            resolved = os.path.abspath(candidate)
            if os.path.exists(resolved) and os.path.isdir(resolved):
                config_file = os.path.join(resolved, "config.json")
                if os.path.exists(config_file):
                    logger.info(f"Resolved model path: {resolved}")
                    return resolved

        return os.path.abspath(DEFAULT_MODEL_PATH)


    def load(self) -> None:
        """Load model and tokenizer from disk."""
        if self._loaded:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"HaluEval detector model not found at: {self.model_path}\n"
                f"Run the HaluEval training command first:\n"
                f"  python -m agents.detector_agent.halueval_trainer\n"
                f"Or set HALUEVAL_MODEL_PATH environment variable."
            )

        config_file = os.path.join(self.model_path, "config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"Model config.json not found in: {self.model_path}\n"
                f"The directory exists but does not contain a valid model.\n"
                f"Run the training command to generate the model."
            )

        logger.info(f"Loading HaluEval model from: {self.model_path}")
        print(f"[HaluEval] Loading model from: {self.model_path}")
        print(f"[HaluEval] Device: {self.device}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self._model.to(self.device)
        self._model.eval()
        self._loaded = True

        print(f"[HaluEval] Model loaded successfully.")

    def predict(
        self,
        user_query: str,
        llm_response: str,
        context: Optional[str] = None,
    ) -> InferenceResult:
        """Run inference on a query-response pair.

        Args:
            user_query:   The user's original question.
            llm_response: The LLM's generated response.
            context:      Optional reference context. If provided, the model
                          uses it to ground the hallucination detection.
                          Production path (no context) is fully supported.

        Returns:
            InferenceResult with hallucination_probability, confidence_score,
            predicted_label, and predicted_label_name.
        """
        if not self._loaded:
            self.load()

        # Use the canonical formatter — same as training
        text = format_detector_input(user_query, llm_response, context)

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        no_halluc_prob = probs[0][0].item()
        halluc_prob = probs[0][1].item()
        predicted_label = 1 if halluc_prob > no_halluc_prob else 0
        label_name = "HALLUCINATION" if predicted_label == 1 else "NO_HALLUCINATION"

        return InferenceResult(
            hallucination_probability=round(halluc_prob, 6),
            confidence_score=round(1.0 - halluc_prob, 6),
            predicted_label=predicted_label,
            predicted_label_name=label_name,
        )


    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._loaded


if __name__ == "__main__":
    """Quick standalone test."""
    engine = HaluEvalInference()
    engine.load()

    test_cases = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the capital of France?", "The capital of France is Tokyo, Japan."),
        ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
        ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by Albert Einstein in 1920."),
    ]

    for query, response in test_cases:
        result = engine.predict(query, response)
        print(f"\nQuery: {query}")
        print(f"Response: {response}")
        print(f"  hallucination_probability = {result.hallucination_probability:.4f}")
        print(f"  confidence_score          = {result.confidence_score:.4f}")
        print(f"  prediction                = {result.predicted_label_name}")
