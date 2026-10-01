"""Explicit local generator that captures logits from its own generation.

No model is loaded at import time. Downloads are disabled unless explicitly
enabled by the operator. The normal hosted-provider path never imports this.
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
from typing import Any

from .phase1 import FEATURE_SCHEMA, GenerationTrace, TokenFeature


def _summarize_raw_logits(logits, token_id: int) -> tuple[float, float, float]:
    """Summarize one unprocessed generation step without 0 * -inf NaNs."""
    import torch

    values = logits.float()
    if values.ndim != 1 or values.numel() < 2 or not torch.isfinite(values).all():
        raise ValueError("raw generation logits must be a finite vocabulary vector")
    log_probs = torch.log_softmax(values, dim=-1)
    probs = log_probs.exp()
    top = torch.topk(probs, k=2).values
    nll = -float(log_probs[token_id])
    entropy = float(torch.special.entr(probs).sum())
    margin = float(top[0] - top[1])
    if not all(math.isfinite(x) for x in (nll, entropy, margin)):
        raise ValueError("non-finite raw-logit token feature")
    return nll, entropy, margin


class LocalTraceGenerator:
    def __init__(self, model_id: str, revision: str, tokenizer_id: str, tokenizer_revision: str):
        if not all((model_id, revision, tokenizer_id, tokenizer_revision)):
            raise ValueError("local UQ generation requires exact model and tokenizer IDs and revisions")
        self.identifiers = (model_id, revision, tokenizer_id, tokenizer_revision)
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id, revision, tokenizer_id, tokenizer_revision = self.identifiers
            allow_download = os.getenv("HALLUCIGUARD_PHASE1_ALLOW_DOWNLOAD", "false").lower() in ("1", "true")
            kwargs = {"local_files_only": not allow_download}
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=tokenizer_revision, **kwargs)
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, **kwargs)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device).eval()
            self._tokenizer, self._model, self._device = tokenizer, model, device

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 128, temperature: float = 0.7) -> tuple[str, dict[str, Any] | None]:
        """Return one response plus signals from exactly that decoding event."""
        if not 1 <= max_new_tokens <= 256:
            raise ValueError("local UQ max_new_tokens must be between 1 and 256")
        self._load()
        import torch

        tokenizer, model, device = self._tokenizer, self._model, self._device
        assert tokenizer is not None and model is not None and device is not None
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True, output_scores=True,
                output_logits=True,
            )
        ids = output.sequences[0, prompt_len:].tolist()
        raw = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        answer = raw.strip()
        if not answer:
            raise ValueError("local generator returned empty answer")
        leading = len(raw) - len(raw.lstrip())
        visible_end = leading + len(answer)
        pieces: list[TokenFeature] = []
        previous = ""
        raw_logits = getattr(output, "logits", None)
        valid = raw_logits is not None and len(ids) == len(raw_logits)
        if not valid:
            return answer, None
        for step, token_id in enumerate(ids):
            prefix = tokenizer.decode(ids[: step + 1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if not prefix.startswith(previous):
                valid = False
                break
            left, right = len(previous) - leading, len(prefix) - leading
            previous = prefix
            if right <= 0 or left >= len(answer):
                continue  # EOS or stripped outer whitespace
            if left < 0 or right > len(answer):
                valid = False
                break
            if right <= left:
                valid = False
                break
            # Sampling scores may be post-warper and -inf for nearly every
            # token. Use genuine pre-warper logits from this decoding event.
            try:
                nll, entropy, margin = _summarize_raw_logits(raw_logits[step][0], token_id)
            except ValueError:
                valid = False
                break
            pieces.append(TokenFeature(start=left, end=right, nll=nll, entropy=entropy, margin=margin))
        if not valid or previous[leading:visible_end] != answer or not pieces:
            return answer, None
        model_id, revision, tokenizer_id, tokenizer_revision = self.identifiers
        trace = GenerationTrace(
            schema_version=FEATURE_SCHEMA,
            source="local_transformers_generate_raw_logits",
            model_id=model_id, model_revision=revision,
            tokenizer_id=tokenizer_id, tokenizer_revision=tokenizer_revision,
            text_sha256=hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            tokens=pieces,
        )
        return answer, trace.model_dump()


_LOCAL_GENERATORS: dict[tuple[str, str, str, str], LocalTraceGenerator] = {}
_LOCAL_LOCK = threading.Lock()


def get_local_generator(identifiers: tuple[str, str, str, str]) -> LocalTraceGenerator:
    with _LOCAL_LOCK:
        if identifiers not in _LOCAL_GENERATORS:
            _LOCAL_GENERATORS[identifiers] = LocalTraceGenerator(*identifiers)
        return _LOCAL_GENERATORS[identifiers]
