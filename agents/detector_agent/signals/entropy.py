from ..utils import init_dll_paths
init_dll_paths()

import math
from typing import Optional
from pydantic import BaseModel, Field
import torch
from ..model_manager import ModelManager


class EntropyMetrics(BaseModel):
    """Structured output containing predictive entropy metrics for an LLM response sequence."""

    average_entropy: float = Field(
        ...,
        description="Average predictive token entropy across the response sequence (in nats)."
    )
    maximum_entropy: float = Field(
        ...,
        description="Maximum predictive token entropy observed across response sequence (in nats)."
    )
    normalized_entropy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average entropy normalized by maximum theoretical vocabulary entropy log(V)."
    )


class EntropyCalculator:
    """Computes predictive token entropy from raw LLM logit distributions."""

    def __init__(self, model_manager: ModelManager) -> None:
        """Initialize EntropyCalculator with a shared ModelManager instance."""
        self.model_manager: ModelManager = model_manager

    def compute(self, user_query: str, llm_response: str) -> Optional[EntropyMetrics]:
        """Compute predictive entropy metrics for the response conditioned on the user query."""
        tokenizer, model = self.model_manager.get_model_and_tokenizer()
        if tokenizer is None or model is None:
            return None

        try:
            prompt_text = f"User: {user_query}\nAssistant:"
            full_text = f"{prompt_text} {llm_response}"

            prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
            full_inputs = tokenizer(full_text, return_tensors="pt")
            full_ids = full_inputs["input_ids"]

            prompt_len = prompt_ids.shape[1]
            seq_len = full_ids.shape[1]

            if seq_len <= prompt_len:
                return None

            with torch.no_grad():
                outputs = model(**full_inputs)
                logits = outputs.logits

            vocab_size = logits.shape[-1]
            max_theoretical_entropy = math.log(vocab_size) if vocab_size > 1 else 1.0

            token_entropies = []
            for i in range(prompt_len, seq_len):
                step_logits = logits[0, i - 1, :]
                log_probs = torch.log_softmax(step_logits, dim=-1)
                probs = torch.softmax(step_logits, dim=-1)
                entropy_tensor = -torch.sum(probs * log_probs)
                token_entropies.append(entropy_tensor.item())

            if not token_entropies:
                return None

            avg_entropy = sum(token_entropies) / len(token_entropies)
            max_entropy = max(token_entropies)
            normalized_entropy = avg_entropy / max_theoretical_entropy
            normalized_entropy = max(0.0, min(1.0, normalized_entropy))

            return EntropyMetrics(
                average_entropy=round(avg_entropy, 6),
                maximum_entropy=round(max_entropy, 6),
                normalized_entropy=round(normalized_entropy, 6)
            )

        except Exception as e:
            print(f"[EntropyCalculator] Warning: Error computing predictive entropy: {e}")
            return None
