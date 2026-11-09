import json
import math
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        # Read the checkpoint's class order instead of silently assuming it.
        self.labels = [ClaimLabel(self.model.config.id2label[i]) for i in range(self.model.config.num_labels)]
        if len(self.labels) != 3 or set(self.labels) != set(LABELS):
            raise ValueError("Detector checkpoint must declare the three grounded labels")
        selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected)
        self.model.to(self.device).eval()
        calibration_path = model_path / "calibration.json"
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        self.temperature = float(calibration["temperature"])
        self.threshold = float(calibration["hallucination_threshold"])
        self.max_length = int(calibration["max_length"])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Detector calibration temperature must be finite and positive")
        if not math.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("Detector threshold must be between zero and one")
        if not 8 <= self.max_length <= self.model.config.max_position_embeddings:
            raise ValueError("Detector max_length exceeds the checkpoint context window")
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
        # Bound peak memory on long answers. Preserve the claim; truncate only
        # evidence rather than silently judging an incomplete claim.
        batches = []
        for start in range(0, len(spans), 8):
            encoded = self.tokenizer(
                contexts[start:start + 8],
                [s.text for s in spans[start:start + 8]],
                padding=True,
                truncation="only_first",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            probs = apply_temperature(self.model(**encoded).logits, self.temperature).cpu()
            if not torch.isfinite(probs).all():
                raise ValueError("Detector produced non-finite probabilities")
            batches.append(probs)
        probabilities = torch.cat(batches)
        results: list[SentenceResult] = []
        guard_fired = False
        for index, (span, probs, evidence_parts) in enumerate(zip(spans, probabilities, snippets)):
            mapping = {label: float(probs[i]) for i, label in enumerate(self.labels)}
            entity_review_required = has_entity_conflict(span.text, " ".join(evidence_parts))
            if entity_review_required:
                # Different entities may express different relations. This is
                # diagnostic only, never synthetic calibrated model probability.
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
                    entity_review_required=entity_review_required,
                )
            )
        overall = max(item.hallucination_probability for item in results)
        warnings = []
        if guard_fired:
            warnings.append("Different named entities detected; model probabilities are unchanged and Verifier must assess the relation.")
        if all(item.label == ClaimLabel.NOT_ENOUGH_INFO for item in results):
            warnings.append("Evidence may be incomplete; NOT_ENOUGH_INFO is not proof of falsehood.")
        return DetectResponse(
            label="HALLUCINATION" if overall >= self.threshold else "REVIEW_REQUIRED" if guard_fired else "NO_HALLUCINATION",
            probability=overall,
            risk=self._risk(overall),
            requires_verification=overall >= self.threshold or guard_fired,
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
