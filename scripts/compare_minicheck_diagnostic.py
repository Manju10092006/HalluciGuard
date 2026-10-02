"""Offline, short-input MiniCheck diagnostic. Not a production replacement.

Matches the upstream encoder's EOS-joined document/claim input for one chunk:
https://github.com/Liyan06/MiniCheck/blob/main/minicheck/inference.py
Long documents are rejected, not silently truncated or scored as full documents.
The six constructed examples are diagnostic fixtures, not a held-out benchmark.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


CASES = [
    ("supported_capital", "Hanoi is the capital of Vietnam.", "The capital of Vietnam is Hanoi.", 1),
    ("wrong_capital", "Hanoi is the capital of Vietnam.", "The capital of Vietnam is Bangkok.", 0),
    ("supported_cofounder", "Microsoft was founded by Bill Gates and Paul Allen in 1975.", "Paul Allen co-founded Microsoft in 1975.", 1),
    ("wrong_founder", "Microsoft was founded by Bill Gates and Paul Allen in 1975.", "Steve Jobs founded Microsoft in 1975.", 0),
    ("absent_attribute", "The restaurant serves Vietnamese food.", "The restaurant has a five-star rating.", 0),
    ("temporal_insufficiency", "Alex was the CEO of ExampleCo in 2010.", "Alex is the CEO of ExampleCo in 2026.", 0),
]


def encode_short(tokenizer, document: str, claim: str):
    if not tokenizer.eos_token:
        raise ValueError("MiniCheck requires an EOS separator")
    if len(tokenizer(document, add_special_tokens=False)["input_ids"]) > 400:
        raise ValueError("diagnostic only supports one short document chunk")
    encoded = tokenizer(tokenizer.eos_token.join([document, claim]), return_tensors="pt", truncation=False)
    if encoded["input_ids"].shape[-1] > 2048:
        raise ValueError("diagnostic input exceeds model limit")
    return encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("diagnostic output already exists")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, local_files_only=True)
    if model.config.num_labels != 2:
        raise ValueError("expected official binary MiniCheck checkpoint")
    model.to(args.device).eval()
    records = []
    with torch.inference_mode():
        for name, doc, claim, expected in CASES:
            encoded = encode_short(tokenizer, doc, claim)
            start = time.perf_counter()
            logits = model(**{key: value.to(args.device) for key, value in encoded.items()}).logits
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite model logits")
            support = float(torch.softmax(logits.float(), dim=-1)[0, 1])
            records.append({"case": name, "document": doc, "claim": claim,
                "expected_supported": expected, "predicted_supported": int(support > .5),
                "raw_support_probability": support, "input_tokens": encoded["input_ids"].shape[-1],
                "seconds": time.perf_counter() - start, "tokenizer_truncated": False})
    report = {"checkpoint": str(args.checkpoint.resolve()), "device": args.device,
        "label_mapping": {"0": "unsupported", "1": "supported"},
        "probability_calibrated_for_halluciguard": False,
        "production_changed": False, "held_out_benchmark": False,
        "correct_fixture_predictions": sum(r["expected_supported"] == r["predicted_supported"] for r in records),
        "records": records,
        "limitations": ["six constructed diagnostic cases only", "binary unsupported is not proof of contradiction",
            "short single-chunk inputs only; no upstream long-document aggregation"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
