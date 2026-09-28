"""
halluciguard_judge / classifier.py
────────────────────────────────────
DeBERTa-v3-base hallucination classifier.

This is the PRIMARY detection layer.

Architecture
------------
    Input: "Query: {user_query}\nClaim: {claim_text}"
    Model: microsoft/deberta-v3-base  (fine-tuned for binary hallucination)
    Output: P(hallucination) in [0.0, 1.0]

Training
--------
    Fine-tuned on HaluBench + RAGTruth + custom HalluciGuard data.
    See trainer.py for the training script.

Fallback
--------
    If the fine-tuned checkpoint is not available, the classifier loads
    a zero-shot NLI-style baseline using the raw DeBERTa model with a
    degraded probability of 0.5 (fail-closed -> VERIFY).

Design principles
-----------------
    - CPU-feasible (DeBERTa-v3-base ~184M params runs on CPU in ~150ms/claim)
    - No GPU required (though GPU will make it faster)
    - Calibrated probabilities (temperature scaling post-training)
    - NOT the factual truth authority -- that is the Verifier's job
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Default HuggingFace base model for fine-tuning / fallback
_DEBERTA_BASE = "microsoft/deberta-v3-base"
_DEBERTA_SMALL = "microsoft/deberta-v3-small"  # lighter alternative


class HallucinationClassifier:
    """DeBERTa-v3-base binary hallucination classifier.

    Inputs
    ------
    user_query : str
        The original user question.
    claim_text : str
        One atomic factual claim extracted from the LLM response.

    Output
    ------
    float : P(hallucination) in [0.0, 1.0]
        Calibrated probability that this claim is a hallucination.
        This is a TRIAGE signal, not a factual verdict.

    Example
    -------
        classifier = HallucinationClassifier(model_path="path/to/checkpoint")
        prob = classifier.predict("Who created Java?", "Java was created by Dennis Ritchie.")
        # prob ~ 0.94
    """

    def __init__(
        self,
        model_path: str = _DEBERTA_BASE,
        max_length: int = 512,
        temperature: float = 1.0,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.max_length = max_length
        self.temperature = temperature  # calibration temperature (1.0 = uncalibrated)
        self._loaded = False
        self._model = None
        self._tokenizer = None
        self._is_finetuned = False

        if device:
            self._device = torch.device(device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            "[Classifier] Initialised. model_path=%s device=%s",
            model_path,
            self._device,
        )

    def load(self) -> bool:
        """Load the model and tokenizer. Returns True on success."""
        if self._loaded:
            return True

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # Try fine-tuned checkpoint first
            checkpoint_path = Path(self.model_path)
            if checkpoint_path.exists() and (checkpoint_path / "config.json").exists():
                logger.info("[Classifier] Loading fine-tuned checkpoint: %s", self.model_path)
                self._tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    str(checkpoint_path),
                    num_labels=2,
                )
                self._is_finetuned = True
            else:
                # Try as HuggingFace model ID (e.g. "Manjunath2000006/halluciguard-judge")
                try:
                    logger.info("[Classifier] Loading from HF hub: %s", self.model_path)
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    self._model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_path,
                        num_labels=2,
                    )
                    self._is_finetuned = True
                except Exception:
                    # Fallback: load raw DeBERTa (no classification head trained)
                    logger.warning(
                        "[Classifier] Fine-tuned checkpoint not found at '%s'. "
                        "Loading raw DeBERTa base — DEGRADED MODE (prob=0.5).",
                        self.model_path,
                    )
                    self._tokenizer = AutoTokenizer.from_pretrained(_DEBERTA_BASE)
                    self._model = AutoModelForSequenceClassification.from_pretrained(
                        _DEBERTA_BASE,
                        num_labels=2,
                    )
                    self._is_finetuned = False

            self._model.eval()
            self._model.to(self._device)
            self._loaded = True
            logger.info(
                "[Classifier] Model loaded. is_finetuned=%s device=%s",
                self._is_finetuned,
                self._device,
            )
            return True

        except Exception as exc:
            logger.error("[Classifier] Failed to load model: %s", exc)
            return False

    def predict(self, user_query: str, claim_text: str) -> float:
        """Predict P(hallucination) for a single claim.

        Args:
            user_query:  The original user question.
            claim_text:  One atomic factual claim.

        Returns:
            float in [0.0, 1.0] — calibrated hallucination probability.
            Returns 0.5 in degraded mode (fail-closed: send to Verifier).
        """
        if not self._loaded:
            success = self.load()
            if not success:
                logger.error("[Classifier] Model not loaded — returning 0.5 (degraded).")
                return 0.5

        if not self._is_finetuned:
            # Raw model has random weights for the classification head
            logger.debug("[Classifier] Degraded mode — returning 0.5.")
            return 0.5

        try:
            # Format: "Query: {q}\nClaim: {c}"
            text = f"Query: {user_query.strip()}\nClaim: {claim_text.strip()}"

            inputs = self._tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**inputs).logits  # shape [1, 2]

            # Apply temperature scaling (calibration)
            scaled_logits = logits / self.temperature
            probs = torch.softmax(scaled_logits, dim=-1)

            # Label 1 = HALLUCINATION
            hallucination_prob = float(probs[0, 1].item())
            return round(hallucination_prob, 4)

        except Exception as exc:
            logger.error("[Classifier] Prediction failed: %s", exc)
            return 0.5  # fail-closed

    def predict_batch(
        self,
        user_query: str,
        claim_texts: list[str],
        batch_size: int = 8,
    ) -> list[float]:
        """Predict P(hallucination) for multiple claims efficiently.

        Args:
            user_query:   The original user question (same for all claims).
            claim_texts:  List of atomic claims.
            batch_size:   Tokenization batch size.

        Returns:
            List of floats in [0.0, 1.0].
        """
        if not claim_texts:
            return []

        if not self._loaded:
            self.load()

        if not self._is_finetuned:
            return [0.5] * len(claim_texts)

        results = []
        texts = [
            f"Query: {user_query.strip()}\nClaim: {c.strip()}"
            for c in claim_texts
        ]

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                inputs = self._tokenizer(
                    batch,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(self._device)

                with torch.no_grad():
                    logits = self._model(**inputs).logits

                scaled = logits / self.temperature
                probs = torch.softmax(scaled, dim=-1)
                hallucination_probs = probs[:, 1].tolist()
                results.extend([round(p, 4) for p in hallucination_probs])

            except Exception as exc:
                logger.error("[Classifier] Batch prediction failed: %s", exc)
                results.extend([0.5] * len(batch))

        return results

    @property
    def model_name(self) -> str:
        if self._is_finetuned:
            return self.model_path
        return f"{_DEBERTA_BASE}:degraded"


__all__ = ["HallucinationClassifier"]
