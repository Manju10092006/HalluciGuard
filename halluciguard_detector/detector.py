import inspect
import math
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .calibration import DEFAULT_MAX_LENGTH, apply_temperature, load_calibration
from .evidence import ClaimEvidenceEngine
from .nli_input import encode_nli_pair
from .schemas import ClaimLabel, DetectResponse, RiskLevel, SentenceResult
from .text import (
    has_entity_conflict,
    numeric_consistency,
    sentence_spans,
    shared_relation,
)


LABELS = [ClaimLabel.SUPPORTED, ClaimLabel.CONTRADICTED, ClaimLabel.NOT_ENOUGH_INFO]


def _supports_trace(select) -> bool:
    """Whether an evidence selector accepts the ``trace`` out-parameter.

    Checked by signature rather than by catching ``TypeError`` so a genuine
    error raised *inside* selection is never swallowed and retried.
    """
    try:
        return "trace" in inspect.signature(select).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False


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

    Semantics:
      - SUPPORTED: the evidence supports the claim.
      - CONTRADICTED: the evidence conflicts with the claim.
      - NOT_ENOUGH_INFO: the evidence is insufficient to tell. It is NOT a
        contradiction and is never folded into ``contradicted_probability``.
      - verification_risk (aliased by the legacy ``probability`` /
        ``hallucination_probability``): an operational triage score equal to
        P(CONTRADICTED) + P(NOT_ENOUGH_INFO); how likely a claim still needs
        checking. It is not the probability the claim is false, and a softmax
        output is not a calibrated probability of real-world truth. The
        answer-level HALLUCINATION label is driven by contradiction mass only,
        so an all-NOT_ENOUGH_INFO answer is never reported as hallucinated.
        The Judge makes the final decision.

    Secondary checks (a same-slot number/date clash, or a named entity swapped on
    an otherwise identical relation) never change any neural probability or class;
    they only append a diagnostic warning and set an explicit requires_verification
    flag so the claim is routed to the Verifier. They never fabricate probabilities
    and never convert NOT_ENOUGH_INFO into a contradiction.
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
        config = getattr(self.model, "config", None)
        if config is not None and getattr(config, "id2label", None) != {i: label.value for i, label in enumerate(LABELS)}:
            raise ValueError("incompatible_detector_label_mapping")
        selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected)
        self.model.to(self.device).eval()
        calibration = load_calibration(model_path / "calibration.json")
        self.calibration_available = bool(calibration)
        self.temperature = float(calibration.get("temperature", 1.0))
        # Two thresholds for two different questions. The near-tie guard below
        # resolves SUPPORTED-vs-CONTRADICTED disagreement, so it must be driven
        # by the contradiction threshold. Reading the verification-risk value
        # here is what previously let a triage score act as a falsity cut-off.
        self.contradiction_threshold = float(calibration.get("contradiction_threshold", 0.5))
        self.verification_risk_threshold = float(calibration.get("verification_risk_threshold", 0.5))
        # Deprecated alias retained for any caller still reading ``threshold``.
        self.threshold = self.verification_risk_threshold
        self.max_length = int(calibration.get("max_length", DEFAULT_MAX_LENGTH))
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
        trace: dict | None = None,
    ) -> dict[ClaimLabel, float]:
        """Classify a single (evidence, claim) pair with the trained model."""
        encoded = encode_nli_pair(
            self.tokenizer,
            claim,
            evidence_text,
            max_length=self.max_length,
            trace=trace,
        ).to(self.device)
        logits = self.model(**encoded).logits
        if logits.shape != (1, len(LABELS)) or not torch.isfinite(logits).all():
            raise ValueError("invalid_detector_logits")
        probs: list[float] = apply_temperature(logits, self.temperature)[0].cpu().tolist()
        return {label: float(probs[i]) for i, label in enumerate(LABELS)}


    @staticmethod
    def _normalize_class_mapping(mapping: dict[ClaimLabel, float]) -> dict[ClaimLabel, float]:
        """Coerce a classifier output into a complete, well-formed distribution.

        Every downstream score assumes three explicit class probabilities that
        sum to 1. An absent class is never a soft signal to be guessed: a missing
        CONTRADICTED probability means *no refutation was found*, so it must read
        as exactly 0.0 rather than as a weak contradiction. Renormalizing across
        the classes that are present preserves the sum-to-1 invariant without
        inventing any probability.
        """
        values: dict[ClaimLabel, float] = {}
        if not mapping or any(label not in mapping for label in LABELS):
            raise ValueError("incomplete_detector_probabilities")
        for label in LABELS:
            raw = mapping.get(label) if mapping else None
            try:
                value = float(raw) if raw is not None else 0.0
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid_detector_probabilities") from exc
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid_detector_probabilities")
            values[label] = value
        total = sum(values.values())
        if not math.isclose(total, 1.0, abs_tol=1e-5):
            raise ValueError("invalid_detector_probability_sum")
        return {label: value / total for label, value in values.items()}

    def _guard_signals(
        self,
        claim: str,
        evidence_text: str,
        mapping: dict[ClaimLabel, float],
    ) -> tuple[dict[ClaimLabel, float], list[str], bool]:
        """Apply conservative secondary signals without inventing probabilities.

        - Named-entity conflict: fires on a swapped named entity. When the two
          texts also share the same relation (e.g. "Apple acquired A" vs
          "Apple acquired B") verification is requested; a mere
          mismatch with a different relation is only noted, never treated as a
          contradiction.
        - Number/date/percent mismatch: a same-slot quantity or year clash is a
          potential conflict and requests verification the same way.
        - These diagnostics never change any neural probability or class. The
          verification request is returned as an explicit boolean (not inferred
          from warning prose), so rewording a message can never silently drop it.
        """
        warnings: list[str] = []
        guard = mapping.copy()
        requires_verification = False

        if has_entity_conflict(claim, evidence_text):
            if not shared_relation(claim, evidence_text):
                warnings.append(
                    "Named-entity mismatch noted; relation differs, so it is not treated "
                    "as an automatic contradiction."
                )
            else:
                warnings.append(
                    "Named-entity conflict on a shared relation noted; requires verification. "
                    "Model probabilities are unchanged."
                )
                requires_verification = True

        numeric = numeric_consistency(claim, evidence_text)
        if numeric:
            # A same-unit quantity clash or a conflicting year is a concrete,
            # evidence-grounded conflict (e.g. "population is 10 million" vs
            # "12 million"). It does NOT change any neural probability or class;
            # it only requests verification so a numerically-wrong but
            # topic-similar claim is not silently accepted on soft NLI alone.
            warnings.append(
                "Number/date mismatch flagged between claim and evidence: "
                    + "; ".join(numeric) + "; requires verification; model probabilities unchanged."
            )
            requires_verification = True

        return guard, warnings, requires_verification

    def _risk(self, probability: float) -> RiskLevel:
        if probability >= max(0.75, self.threshold):
            return RiskLevel.HIGH
        if probability >= min(0.35, self.threshold):
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    @torch.inference_mode()
    def detect(self, draft_answer: str, evidence: list[str], user_query: str = "") -> DetectResponse:
        if not evidence or not any(x.strip() for x in evidence):
            raise ValueError("evidence is required for grounded hallucination detection")

        claim_units = self._atomic_claims(draft_answer)
        if not claim_units:
            raise ValueError("draft_answer contains no detectable sentence")

        results: list[SentenceResult] = []
        warnings: list[str] = []
        select_supports_trace = _supports_trace(self.evidence.select)
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
                        model_input_evidence=None,
                        evidence_selection_trace={},
                        supported_probability=0.0,
                        contradicted_probability=0.0,
                        unknown_probability=0.0,
                        requires_verification=False,
                        non_factual=True,
                        verification_risk=0.0,
                    )
                )
                continue

            trace: dict[str, object] = {}
            if select_supports_trace:
                snippets = self.evidence.select(claim_text, evidence, trace=trace)
            else:
                # An evidence adapter predating the trace seam: treat the
                # selection as untraceable rather than pretending it was clean.
                snippets = self.evidence.select(claim_text, evidence)
                trace = {"route": "untraced", "degraded": True}
            trace["selected_count"] = len(snippets)
            trace["model_input_evidence"] = snippets[0] if snippets else None
            trace.setdefault("tokenizer_truncation", "not_measured")
            evidence_degraded = bool(trace.get("degraded"))
            evidence_route = str(trace.get("route", "unknown"))
            if not snippets:
                mapping = {
                    ClaimLabel.SUPPORTED: 0.0,
                    ClaimLabel.CONTRADICTED: 0.0,
                    ClaimLabel.NOT_ENOUGH_INFO: 1.0,
                }
                item_warnings = ["No relevant evidence found; claim marked NOT_ENOUGH_INFO."]
                guard_requires = False
            else:
                # DeBERTa input = claim + best reranked evidence only, so the
                # pair stays within max_length; the fuller snippet list is kept
                # on the result for inspection.
                best = snippets[0]
                # The tokenization diagnostic re-tokenizes the full (untruncated)
                # pair a SECOND time per claim; keep it off the hot path unless
                # explicitly enabled for debugging (audit #26).
                import os
                _trace_tokens = os.environ.get("HG_DETECTOR_TRACE_TOKENS", "").strip().lower() in ("1", "true", "yes", "on")
                tokenization_trace: dict = {}
                if _trace_tokens and _supports_trace(self._classify):
                    raw_mapping = self._classify(claim_text, best, trace=tokenization_trace)
                else:
                    raw_mapping = self._classify(claim_text, best)
                mapping = self._normalize_class_mapping(raw_mapping)
                trace["tokenization"] = tokenization_trace
                trace["tokenizer_truncation"] = tokenization_trace.get("truncated", "not_measured")
                trace["model_consumption_observed"] = tokenization_trace.get("status") == "measured"
                mapping, item_warnings, guard_requires = self._guard_signals(claim_text, best, mapping)

            label = max(mapping, key=mapping.get) if mapping else ClaimLabel.NOT_ENOUGH_INFO
            # Operational verification risk = P(CONTRADICTED) + P(NOT_ENOUGH_INFO).
            # It is the probability this claim still needs checking; it is NOT
            # the probability the claim is false (that is contradicted only).
            verification_risk = mapping[ClaimLabel.CONTRADICTED] + mapping[ClaimLabel.NOT_ENOUGH_INFO]
            requires_verification = label != ClaimLabel.SUPPORTED or guard_requires
            results.append(
                SentenceResult(
                    sentence_id=f"C{index:03d}",
                    text=claim_text,
                    start=start,
                    end=end,
                    label=label,
                    probabilities=mapping,
                    # Backward-compatible alias of verification_risk.
                    hallucination_probability=verification_risk,
                    risk=self._risk(verification_risk),
                    evidence_snippets=snippets,
                    model_input_evidence=snippets[0] if snippets else None,
                    evidence_selection_trace=trace,
                    supported_probability=mapping.get(ClaimLabel.SUPPORTED, 0.0),
                    contradicted_probability=mapping.get(ClaimLabel.CONTRADICTED, 0.0),
                    unknown_probability=mapping.get(ClaimLabel.NOT_ENOUGH_INFO, 0.0),
                    requires_verification=requires_verification,
                    non_factual=False,
                    verification_risk=verification_risk,
                    evidence_degraded=evidence_degraded,
                    evidence_route=evidence_route,
                )
            )
            if evidence_degraded:
                reason = trace.get("reason")
                warnings.append(
                    f"Evidence selection degraded for a claim (route={evidence_route}"
                    + (f": {reason}" if reason else "")
                    + "); using deterministic lexical or pre-rerank order."
                )
            warnings.extend(item_warnings)

        assessed = [item for item in results if not item.non_factual]
        # Operational risk: max over assessed claims of (contradicted+unknown).
        verification_risk = max((item.verification_risk for item in assessed), default=0.0)
        # Contradiction mass: the actual "this is refuted" evidence. Kept
        # separate so NOT_ENOUGH_INFO can never alone trip the HALLUCINATION
        # label (Issue: unknown must not equal false).
        contradiction_mass = max((item.contradicted_probability for item in assessed), default=0.0)
        claim_count = len(assessed)
        supported_count = sum(1 for item in assessed if item.label == ClaimLabel.SUPPORTED)
        contradicted_count = sum(1 for item in assessed if item.label == ClaimLabel.CONTRADICTED)
        unknown_count = sum(1 for item in assessed if item.label == ClaimLabel.NOT_ENOUGH_INFO)
        non_factual_count = sum(1 for item in results if item.non_factual)
        # Any claim that is not SUPPORTED (contradicted OR unknown) still needs
        # verification by the Judge.
        requires_verification = any(item.requires_verification for item in assessed)

        if contradicted_count:
            warnings.append("At least one atomic claim is CONTRADICTED by the supplied evidence.")
        if unknown_count and not contradicted_count:
            warnings.append("Evidence may be incomplete; NOT_ENOUGH_INFO is not proof of falsehood.")

        return DetectResponse(
            # HALLUCINATION is driven by contradiction alone, never by
            # NOT_ENOUGH_INFO. It follows the claim-level CONTRADICTED decisions
            # so the answer verdict can never disagree with the claims that
            # produced it, and an all-NOT_ENOUGH_INFO answer is never reported
            # as hallucinated.
            label="HALLUCINATION" if contradicted_count else "NO_HALLUCINATION",
            # Deprecated alias of verification_risk (unchanged for compatibility).
            probability=verification_risk,
            verification_risk=verification_risk,
            contradiction_mass=contradiction_mass,
            risk=self._risk(verification_risk),
            requires_verification=requires_verification,
            sentences=results,
            model_version=self.version,
            warnings=warnings,
            claim_count=claim_count,
            supported_count=supported_count,
            contradicted_count=contradicted_count,
            unknown_count=unknown_count,
            non_factual_count=non_factual_count,
            evidence_degraded=any(item.evidence_degraded for item in results),
        )
