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
        encoded = self.tokenizer(
            contexts,
            [s.text for s in spans],
            padding=True,
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        probabilities = apply_temperature(self.model(**encoded).logits, self.temperature).cpu()
        results: list[SentenceResult] = []
        guard_fired = False
        for index, (span, probs, evidence_parts) in enumerate(zip(spans, probabilities, snippets)):
            mapping = {label: float(probs[i]) for i, label in enumerate(LABELS)}
            if has_entity_conflict(span.text, " ".join(evidence_parts)):
                mapping = {
                    ClaimLabel.SUPPORTED: 0.05,
                    ClaimLabel.CONTRADICTED: 0.90,
                    ClaimLabel.NOT_ENOUGH_INFO: 0.05,
                }
                guard_fired = True
            label = max(mapping, key=mapping.get)
            hallucination_probability = mapping[ClaimLabel.CONTRADICTED] + mapping[ClaimLabel.NOT_ENOUGH_INFO]
            risk = self._risk(hallucination_probability)
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
            warnings.append("Named-entity conflict guard triggered for at least one sentence.")
        if all(item.label == ClaimLabel.NOT_ENOUGH_INFO for item in results):
            warnings.append("Evidence may be incomplete; NOT_ENOUGH_INFO is not proof of falsehood.")
        return DetectResponse(
            label="HALLUCINATION" if overall >= self.threshold else "NO_HALLUCINATION",
            probability=overall,
            risk=self._risk(overall),
            requires_verification=overall >= self.threshold,
            sentences=results,
            model_version=self.version,
            warnings=warnings,
        )

    def _risk(self, probability: float) -> RiskLevel:
        if probability >= max(0.75, self.threshold):
            return RiskLevel.HIGH
        if probability >= min(0.35, self.threshold):
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
