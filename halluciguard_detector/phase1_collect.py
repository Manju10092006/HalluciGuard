"""Collect genuine local generation events for later independent human labelling."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .local_generation import get_local_generator
from .phase1 import GenerationTrace


def collect(prompt_jsonl: Path, output_jsonl: Path, *, model_id: str,
            model_revision: str, tokenizer_id: str, tokenizer_revision: str,
            dataset_id: str, max_new_tokens: int = 128, seed: int = 42) -> dict[str, Any]:
    """Generate once per prompt, persisting only complete same-event traces.

    Existing prompt_ids are skipped on rerun. This is an *unlabelled* collection,
    not a training run or a claim of annotation quality.
    """
    import torch

    if not prompt_jsonl.is_file():
        raise FileNotFoundError(f"prompt file missing: {prompt_jsonl}")
    if prompt_jsonl.resolve() == output_jsonl.resolve():
        raise ValueError("output must not overwrite prompts")
    if not dataset_id.strip():
        raise ValueError("dataset_id is required")
    prompts = [json.loads(line) for line in prompt_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError("no prompts supplied")
    prompt_ids: set[str] = set()
    for prompt in prompts:
        if not isinstance(prompt, dict) or not isinstance(prompt.get("prompt_id"), str) or not prompt["prompt_id"].strip():
            raise ValueError("every prompt needs a nonempty prompt_id")
        if not isinstance(prompt.get("query"), str) or not prompt["query"].strip():
            raise ValueError("every prompt needs a nonempty query")
        if prompt["prompt_id"] in prompt_ids:
            raise ValueError("duplicate prompt_id")
        prompt_ids.add(prompt["prompt_id"])
    identifiers = (model_id, model_revision, tokenizer_id, tokenizer_revision)
    if not all(identifiers):
        raise ValueError("exact model and tokenizer IDs/revisions are required")
    prompt_by_id = {prompt["prompt_id"]: prompt for prompt in prompts}
    completed: set[str] = set()
    if output_jsonl.exists():
        for line in output_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            trace = GenerationTrace.model_validate(record["trace"])
            if tuple(getattr(trace, key) for key in ("model_id", "model_revision", "tokenizer_id", "tokenizer_revision")) != identifiers:
                raise ValueError("existing output uses a different generator")
            if trace.text_sha256 != hashlib.sha256(record["response"].encode()).hexdigest():
                raise ValueError("existing output has a mismatched response trace")
            prompt_id = record["prompt_id"]
            if prompt_id in completed:
                raise ValueError("existing output has duplicate prompt_id")
            if prompt_id not in prompt_by_id or record.get("query") != prompt_by_id[prompt_id]["query"]:
                raise ValueError("existing output does not match supplied prompts")
            if record.get("dataset_id") != dataset_id:
                raise ValueError("existing output uses a different dataset_id")
            completed.add(prompt_id)
    generator = get_local_generator(identifiers)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    generated = 0
    with output_jsonl.open("a", encoding="utf-8") as target:
        for index, prompt in enumerate(prompts):
            if prompt["prompt_id"] in completed:
                continue
            random.seed(seed + index)
            torch.manual_seed(seed + index)
            answer, trace_data = generator.generate(
                [{"role": "user", "content": prompt["query"]}], max_new_tokens=max_new_tokens
            )
            if trace_data is None:
                raise ValueError(f"generation trace unavailable for prompt_id={prompt['prompt_id']}")
            trace = GenerationTrace.model_validate(trace_data)
            if trace.text_sha256 != hashlib.sha256(answer.encode()).hexdigest():
                raise ValueError("generated trace does not match response")
            record = {
                "event_id": hashlib.sha256(f"{dataset_id}:{prompt['prompt_id']}:{model_revision}:{seed}".encode()).hexdigest(),
                "prompt_id": prompt["prompt_id"],
                "group_id": str(prompt.get("group_id") or prompt["prompt_id"]),
                "dataset_id": dataset_id,
                "query": prompt["query"], "response": answer,
                "trace": trace.model_dump(),
            }
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            target.flush()
            completed.add(prompt["prompt_id"])
            generated += 1
    return {"generated": generated, "already_present": len(completed) - generated,
            "status": "unlabelled_requires_human_review", "output": str(output_jsonl)}
