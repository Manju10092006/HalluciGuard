"""
models / deberta.py
───────────────────
Primary Production Candidate M2: DeBERTa-v3-base fine-tuned per claim.

Input Format:
  text_a = claim
  text_b = "[QUESTION] {user_query} [ANSWER] {draft_answer}"
  truncation = "only_second"
  max_length = 384
"""

from __future__ import annotations
import logging
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

DEBERTA_BASE = "microsoft/deberta-v3-base"


class ModelUnavailableError(RuntimeError):
    """Raised when the DeBERTa model cannot produce a real score.

    The detector catches this and fails closed (UNKNOWN / DEEP / VERIFY)
    instead of fabricating a score.
    """


class M2DebertaClassifier:
    """Primary candidate M2: DeBERTa-v3-base fine-tuned for claim risk classification."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        max_length: int = 384,
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path or DEBERTA_BASE
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._tokenizer = None
        self._model = None
        self.fitted = False

    def load_or_init(self) -> None:
        if self._model is not None:
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.checkpoint_path, num_labels=2
            )
            self._model.to(self.device)
            self._model.eval()
            self.fitted = True
        except Exception as exc:
            logger.warning("[M2Deberta] Could not load checkpoint %s: %s", self.checkpoint_path, exc)
            self.fitted = False

    def is_available(self) -> bool:
        """True only if a real model is loaded and ready for inference.

        Attempts a lazy load on first call. Callers MUST check this before
        trusting scores; an unavailable model must be treated as UNKNOWN by
        the pipeline, never silently scored as 0.5.
        """
        if self._model is None:
            self.load_or_init()
        return bool(self.fitted and self._model is not None)

    def predict_claim_risk(
        self,
        user_query: str,
        draft_answer: str,
        claim_texts: List[str],
    ) -> List[float]:
        """
        Score a list of claims given the user query and draft answer context.
        Format: text_a = claim, text_b = "[QUESTION] {query} [ANSWER] {answer}"
        """
        if not claim_texts:
            return []

        if not self.fitted:
            self.load_or_init()

        if not self.fitted or self._model is None:
            # NEVER fabricate a score. Signal unavailability so the detector
            # can fail closed (UNKNOWN / DEEP / VERIFY).
            raise ModelUnavailableError(
                f"DeBERTa checkpoint '{self.checkpoint_path}' is not loaded"
            )

        context_str = f"[QUESTION] {user_query} [ANSWER] {draft_answer}"
        scores = []

        try:
            for claim in claim_texts:
                inputs = self._tokenizer(
                    claim,
                    context_str,
                    truncation="only_second",
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    logits = self._model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)
                    risk_prob = float(probs[0, 1].item())
                    scores.append(round(risk_prob, 4))
        except Exception as exc:
            logger.error("[M2Deberta] Inference failed: %s", exc)
            # Propagate rather than fabricate; detector fails closed.
            raise ModelUnavailableError(f"DeBERTa inference failed: {exc}") from exc

        return scores
