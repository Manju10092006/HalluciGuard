import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .calibration import apply_temperature
from .schemas import ClaimLabel, DetectResponse, RiskLevel, SentenceResult
from .text import has_entity_conflict, lexical_evidence, sentence_spans


LABELS = [ClaimLabel.SUPPORTED, ClaimLabel.CONTRADICTED, ClaimLabel.NOT_ENOUGH_INFO]


class Detector:
    """Reference-grounded sentence detector.

    Evidence is mandatory by design. Without evidence, a detector can estimate
    plausibility but cannot determine factual support.
    """

    def __init__(self, model_path: str | Path, device: str | None = None):
        model_path = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected)
        self.model.to(self.device).eval()
        calibration_path = model_path / "calibration.json"
        calibration = (
            json.loads(calibration_path.read_text(encoding="utf-8"))
            if calibration_path.exists()
            else {}
        )
        self.temperature = float(calibration.get("temperature", 1.0))
        self.threshold = float(calibration.get("hallucination_threshold", 0.5))
        self.max_length = int(calibration.get("max_length", 384))
        self.version = model_path.name

    @torch.inference_mode()
    def detect(self, draft_answer: str, evidence: list[str], user_query: str = "") -> DetectResponse:
        if not evidence or not any(x.strip() for x in evidence):
            raise ValueError("evidence is required for grounded hallucination detection")
        spans = sentence_spans(draft_answer)
        if not spans:
            raise ValueError("draft_answer contains no detectable sentence")
        snippets = [lexical_evidence(s.text, evidence) for s in spans]
        contexts = [" ".join(parts) for parts in snippets]
        token_diagnostics: dict = {
            "selected_snippet_counts": [len(parts) for parts in snippets],
            "selected_character_counts": [len(context) for context in contexts],
            "tokenizer_observability": "unknown",
            "tokenizer_truncated": None,
            "maximum_length": self.max_length,
        }
        # Measure actual pair-tokenization lengths without changing the inputs
        # used for inference. A tokenizer without inspectable IDs stays unknown.
        try:
            untruncated = self.tokenizer(
                contexts, [span.text for span in spans], padding=False,
                truncation=False, return_tensors=None,
            )
            raw_lengths = [len(row) for row in untruncated["input_ids"]]
        except (KeyError, TypeError, ValueError, AttributeError):
            raw_lengths = None
        encoded = self.tokenizer(
            contexts,
            [s.text for s in spans],
            padding=True,
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        if raw_lengths is not None and "input_ids" in encoded:
            consumed_lengths = (
                [int(mask.sum()) for mask in encoded["attention_mask"]]
                if "attention_mask" in encoded
                else [len(row) for row in encoded["input_ids"]]
            )
            if len(raw_lengths) == len(consumed_lengths) == len(spans):
                token_diagnostics.update({
                    "tokenizer_observability": "measured",
                    "untruncated_pair_tokens": raw_lengths,
                    "model_input_pair_tokens": consumed_lengths,
                    "tokenizer_truncated": [before > after for before, after in zip(raw_lengths, consumed_lengths)],
                })
        probabilities = apply_temperature(self.model(**encoded).logits, self.temperature).cpu()
        results: list[SentenceResult] = []
        guard_fired = False
        for index, (span, probs, evidence_parts) in enumerate(zip(spans, probabilities, snippets)):
            mapping = {label: float(probs[i]) for i, label in enumerate(LABELS)}
            entity_guard = has_entity_conflict(span.text, " ".join(evidence_parts))
            guard_fired = guard_fired or entity_guard
            label = max(mapping, key=mapping.get)
            hallucination_probability = mapping[ClaimLabel.CONTRADICTED] + mapping[ClaimLabel.NOT_ENOUGH_INFO]
            risk = RiskLevel.HIGH if entity_guard else self._risk(hallucination_probability)
            results.append(
                SentenceResult(
                    sentence_id=f"S{index + 1:03d}",
                    text=span.text,
                    start=span.start,
                    end=span.end,
                    label=label,
                    probabilities=mapping,
                    hallucination_probability=hallucination_probability,
                    risk=risk,
                    evidence_snippets=evidence_parts,
                )
            )
        overall = max(item.hallucination_probability for item in results)
        warnings = []
        if guard_fired:
            warnings.append("Named-entity mismatch raised verification risk; model probabilities were not changed.")
        if all(item.label == ClaimLabel.NOT_ENOUGH_INFO for item in results):
            warnings.append("Evidence may be incomplete; NOT_ENOUGH_INFO is not proof of falsehood.")
        return DetectResponse(
            label="HALLUCINATION" if overall >= self.threshold else "NO_HALLUCINATION",
            probability=overall,
            risk=RiskLevel.HIGH if guard_fired else self._risk(overall),
            requires_verification=guard_fired or overall >= self.threshold,
            sentences=results,
            model_version=self.version,
            warnings=warnings,
            input_diagnostics=token_diagnostics,
        )

    def _risk(self, probability: float) -> RiskLevel:
        if probability >= max(0.75, self.threshold):
            return RiskLevel.HIGH
        if probability >= min(0.35, self.threshold):
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
