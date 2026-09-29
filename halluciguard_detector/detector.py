import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .calibration import apply_temperature
from .evidence import ClaimEvidenceEngine
from .schemas import ClaimLabel, DetectResponse, RiskLevel, SentenceResult
from .text import has_entity_conflict, numeric_consistency, sentence_spans


LABELS = [ClaimLabel.SUPPORTED, ClaimLabel.CONTRADICTED, ClaimLabel.NOT_ENOUGH_INFO]


class Detector:
    """Reference-grounded, claim-level hallucination detector.

    Evidence is mandatory by design. Without evidence, a detector can estimate
    plausibility but cannot determine factual support.

    Pipeline (claim-level evidence verification):
      1. Decompose the answer into atomic claims (Verifier's ClaimDecomposer);
         fall back to sentence spans when decomposition is unavailable/empty.
      2. For each claim, hybrid-retrieve + cross-encoder rerank evidence from
         the supplied passages (Verifier's shared stack), reranking on the REAL
         claim. Falls back to deterministic lexical selection offline.
      3. Classify ``(best_evidence, claim)`` with the trained DeBERTa model.
      4. Emit separate SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO probabilities
         and an answer-level aggregation of claim counts for the Judge.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        evidence: ClaimEvidenceEngine | None = None,
    ):
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
        self.evidence = evidence or ClaimEvidenceEngine()

    # ------------------------------------------------------------------
    # Claim-level helpers
    # ------------------------------------------------------------------
    def _atomic_claims(self, draft_answer: str) -> list[tuple[str, tuple[int, int]]]:
        """Return ``(claim_text, span)`` units preserving original offsets."""
        spans = sentence_spans(draft_answer)
        if not spans:
            return []
        decomposed = self.evidence.decompose(draft_answer)
        # Prefer atomic claims when the shared decomposer produced any. Each
        # atomic claim is aligned back to the sentence whose text it matches
        # (exact, prefix, or best token overlap), defaulting to the full answer
        # span so offsets always stay inside the original string.
        if decomposed:
            return self._align_claims(draft_answer, spans, decomposed)
        return [(span.text, (span.start, span.end)) for span in spans]

    @staticmethod
    def _token_words(text: str) -> set[str]:
        return {t.lower().strip(".,!?;:") for t in text.split() if t.strip()}

    def _align_claims(
        self,
        draft_answer: str,
        spans,
        claims: list[str],
    ) -> list[tuple[str, tuple[int, int]]]:
        aligned: list[tuple[str, tuple[int, int]]] = []
        candidate_spans = list(spans)
        for claim in claims:
            claim_key = self._token_words(claim)
            best, best_score = None, -1.0
            for span in candidate_spans:
                span_words = self._token_words(span.text)
                if not span_words:
                    continue
                overlap = len(claim_key & span_words) / len(claim_key or {""})
                if claim.strip().lower() in span.text.lower():
                    overlap = 1.0
                if overlap > best_score:
                    best, best_score = span, overlap
            if best is None:
                aligned.append((claim, (0, len(draft_answer))))
            else:
                # Prefer the highest-scoring span for each claim; allow the same
                # span to back multiple atomic claims (compound predicates).
                aligned.append((claim, (best.start, best.end)))
        return aligned

    @torch.inference_mode()
    def _classify(
        self,
        claim: str,
        evidence_text: str,
    ) -> dict[ClaimLabel, float]:
        """Classify a single (evidence, claim) pair with the trained model."""
        encoded = self.tokenizer(
            [evidence_text],
            [claim],
            padding=True,
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**encoded).logits
        probs: list[float] = apply_temperature(logits, self.temperature)[0].cpu().tolist()
        return {label: float(probs[i]) for i, label in enumerate(LABELS)}

    def _guard_signals(
        self,
        claim: str,
        evidence_text: str,
        mapping: dict[ClaimLabel, float],
    ) -> tuple[dict[ClaimLabel, float], list[str]]:
        """Apply conservative secondary signals without inventing probabilities.

        - Named-entity conflict: reinforce CONTRADICTED modestly only when the
          raw model already doubts SUPPORTED; never overwrites to 0.05/0.90/0.05.
        - Number/date/percent mismatch: emit a warning; the numeric mismatch is
          a secondary signal for the Judge, not an automatic contradiction.
        """
        warnings: list[str] = []
        guard = mapping.copy()

        if has_entity_conflict(claim, evidence_text):
            if mapping[ClaimLabel.CONTRADICTED] < mapping[ClaimLabel.SUPPORTED]:
                delta = 0.25 * (mapping[ClaimLabel.SUPPORTED] - mapping[ClaimLabel.CONTRADICTED])
                guard[ClaimLabel.SUPPORTED] = mapping[ClaimLabel.SUPPORTED] - delta
                guard[ClaimLabel.CONTRADICTED] = mapping[ClaimLabel.CONTRADICTED] + delta
                guard[ClaimLabel.NOT_ENOUGH_INFO] = mapping[ClaimLabel.NOT_ENOUGH_INFO]
            warnings.append("Named-entity conflict raised contradiction confidence for one claim.")

        numeric = numeric_consistency(claim, evidence_text)
        if numeric:
            warnings.append(
                "Number/date mismatch flagged between claim and evidence: "
                + "; ".join(numeric)
            )

        return guard, warnings

    @torch.inference_mode()
    def detect(self, draft_answer: str, evidence: list[str], user_query: str = "") -> DetectResponse:
        if not evidence or not any(x.strip() for x in evidence):
            raise ValueError("evidence is required for grounded hallucination detection")

        claim_units = self._atomic_claims(draft_answer)
        if not claim_units:
            raise ValueError("draft_answer contains no detectable sentence")

        results: list[SentenceResult] = []
        warnings: list[str] = []
        for index, (claim_text, (start, end)) in enumerate(claim_units, start=1):
            # Opinion / non-factual claims are not evidence-verifiable facts:
            # they must not be counted as hallucination and never reach NLI.
            if not self.evidence.checkable(claim_text):
                results.append(
                    SentenceResult(
                        sentence_id=f"C{index:03d}",
                        text=claim_text,
                        start=start,
                        end=end,
                        label=ClaimLabel.NOT_ENOUGH_INFO,
                        probabilities={},
                        hallucination_probability=0.0,
                        risk=RiskLevel.LOW,
                        evidence_snippets=[],
                        supported_probability=0.0,
                        contradicted_probability=0.0,
                        unknown_probability=0.0,
                        requires_verification=False,
                        non_factual=True,
                    )
                )
                continue

            snippets = self.evidence.select(claim_text, evidence)
            if not snippets:
                mapping = {
                    ClaimLabel.SUPPORTED: 0.0,
                    ClaimLabel.CONTRADICTED: 0.0,
                    ClaimLabel.NOT_ENOUGH_INFO: 1.0,
                }
                item_warnings = ["No relevant evidence found; claim marked NOT_ENOUGH_INFO."]
            else:
                # DeBERTa input = claim + best reranked evidence only, so the
                # pair stays within max_length; the fuller snippet list is kept
                # on the result for inspection.
                best = snippets[0]
                mapping = self._classify(claim_text, best)
                mapping, item_warnings = self._guard_signals(claim_text, best, mapping)

            label = max(mapping, key=mapping.get) if mapping else ClaimLabel.NOT_ENOUGH_INFO
            hallucination_probability = mapping[ClaimLabel.CONTRADICTED] + mapping[ClaimLabel.NOT_ENOUGH_INFO]
            requires_verification = label != ClaimLabel.SUPPORTED
            results.append(
                SentenceResult(
                    sentence_id=f"C{index:03d}",
                    text=claim_text,
                    start=start,
                    end=end,
                    label=label,
                    probabilities=mapping,
                    hallucination_probability=hallucination_probability,
                    risk=self._risk(hallucination_probability),
                    evidence_snippets=snippets,
                    supported_probability=mapping.get(ClaimLabel.SUPPORTED, 0.0),
                    contradicted_probability=mapping.get(ClaimLabel.CONTRADICTED, 0.0),
                    unknown_probability=mapping.get(ClaimLabel.NOT_ENOUGH_INFO, 0.0),
                    requires_verification=requires_verification,
                    non_factual=False,
                )
            )
            warnings.extend(item_warnings)

        overall = max((item.hallucination_probability for item in results), default=0.0)
        claim_count = sum(1 for item in results if not item.non_factual)
        supported_count = sum(1 for item in results if item.label == ClaimLabel.SUPPORTED)
        contradicted_count = sum(1 for item in results if item.label == ClaimLabel.CONTRADICTED)
        unknown_count = sum(1 for item in results if item.label == ClaimLabel.NOT_ENOUGH_INFO and not item.non_factual)
        non_factual_count = sum(1 for item in results if item.non_factual)

        if contradicted_count:
            warnings.append("At least one atomic claim is CONTRADICTED by the supplied evidence.")
        if unknown_count and not contradicted_count:
            warnings.append("Evidence may be incomplete; NOT_ENOUGH_INFO is not proof of falsehood.")

        return DetectResponse(
            label="HALLUCINATION" if overall >= self.threshold else "NO_HALLUCINATION",
            probability=overall,
            risk=self._risk(overall),
            requires_verification=overall >= self.threshold,
            sentences=results,
            model_version=self.version,
            warnings=warnings,
            claim_count=claim_count,
            supported_count=supported_count,
            contradicted_count=contradicted_count,
            unknown_count=unknown_count,
            non_factual_count=non_factual_count,
        )

    def _risk(self, probability: float) -> RiskLevel:
        if probability >= max(0.75, self.threshold):
            return RiskLevel.HIGH
        if probability >= min(0.35, self.threshold):
            return RiskLevel.MEDIUM
        return RiskLevel.LOW