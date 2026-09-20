"""HalluDetect-style conditional token-probability risk signal.

The reference HalluDetect method derives hallucination features from evaluator
token probabilities.  This production adaptation uses batched masked-token
pseudo likelihood so it works with a compact bidirectional evaluator and does
not require access to the draft model's generation logits.
"""

from __future__ import annotations

import math
import os
from typing import Optional


class TokenSurprisalEvaluator:
    _tokenizer = None
    _model = None
    _device = None
    _load_attempted = False

    def __init__(self, model_name: Optional[str] = None, max_length: int = 128) -> None:
        self.model_name = model_name or os.environ.get(
            "HG_DETECTOR_EVALUATOR_MODEL", "distilbert-base-uncased"
        )
        self.max_length = max_length

    def _load(self) -> bool:
        if self.__class__._model is not None:
            return True
        if self.__class__._load_attempted:
            return False
        self.__class__._load_attempted = True
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            allow_downloads = os.environ.get("ALLOW_MODEL_DOWNLOADS", "true").lower() in {
                "1",
                "true",
                "yes",
            }
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=not allow_downloads
            )
            model = AutoModelForMaskedLM.from_pretrained(
                self.model_name, local_files_only=not allow_downloads
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            self.__class__._tokenizer = tokenizer
            self.__class__._model = model
            self.__class__._device = device
            return True
        except Exception:
            return False

    def score(self, user_query: str, claim: str) -> Optional[float]:
        """Return probability-like surprisal in [0, 1], or ``None`` on failure."""
        if not self._load():
            return None
        try:
            import torch

            tokenizer = self.__class__._tokenizer
            model = self.__class__._model
            device = self.__class__._device
            encoded = tokenizer(
                claim,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = encoded["input_ids"][0]
            sep_id = tokenizer.sep_token_id
            sep_positions = (input_ids == sep_id).nonzero(as_tuple=False).flatten().tolist()
            first_sep = 0
            last_sep = sep_positions[-1] if sep_positions else len(input_ids)
            positions = [
                index
                for index in range(first_sep + 1, last_sep)
                if int(input_ids[index])
                not in set(tokenizer.all_special_ids)
            ]
            if not positions:
                return None

            total_log_probability = 0.0
            total_tokens = 0
            batch_size = 32
            base_mask = encoded.get("attention_mask")
            for offset in range(0, len(positions), batch_size):
                chunk = positions[offset : offset + batch_size]
                batch_ids = input_ids.unsqueeze(0).repeat(len(chunk), 1)
                targets = []
                for row, position in enumerate(chunk):
                    targets.append(int(batch_ids[row, position]))
                    batch_ids[row, position] = tokenizer.mask_token_id
                kwargs = {"input_ids": batch_ids.to(device)}
                if base_mask is not None:
                    kwargs["attention_mask"] = base_mask.repeat(len(chunk), 1).to(device)
                with torch.no_grad():
                    logits = model(**kwargs).logits
                for row, (position, target) in enumerate(zip(chunk, targets)):
                    log_probability = torch.log_softmax(
                        logits[row, position], dim=-1
                    )[target]
                    total_log_probability += float(log_probability.item())
                    total_tokens += 1

            average_log_probability = total_log_probability / max(1, total_tokens)
            # HalluDetect intuition: low evaluator probability -> high risk.
            return max(0.0, min(1.0, 1.0 - math.exp(average_log_probability)))
        except Exception:
            return None


__all__ = ["TokenSurprisalEvaluator"]
