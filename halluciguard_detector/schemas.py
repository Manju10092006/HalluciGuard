from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ClaimLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class DetectRequest(BaseModel):
    user_query: str = ""
    draft_answer: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def evidence_is_required(self):
        if not any(item.strip() for item in self.evidence):
            raise ValueError(
                "evidence is required: truth cannot be established from an answer alone"
            )
        return self


class SentenceResult(BaseModel):
    sentence_id: str
    text: str
    start: int
    end: int
    label: ClaimLabel
    probabilities: dict[ClaimLabel, float]
    # DEPRECATED (kept for backward compatibility): equal to
    # ``contradicted_probability + unknown_probability``. Read it as an
    # operational *verification risk* (how likely this claim needs checking),
    # NOT as the probability that the claim is objectively false. Only
    # ``label``/``contradicted_probability`` speak to contradiction.
    hallucination_probability: float
    risk: RiskLevel
    evidence_snippets: list[str] = Field(default_factory=list)
    # ------------------------------------------------------------------
    # Claim-level evidence verification fields (additive, backward compatible).
    # ``SUPPORTED`` = evidence supports the claim, ``CONTRADICTED`` = evidence
    # conflicts with it, ``NOT_ENOUGH_INFO`` = evidence is insufficient.
    # NOT_ENOUGH_INFO is NOT a contradiction and is never folded into
    # ``contradicted_probability``. A softmax output is model confidence about
    # the evidence relation, not a calibrated probability about real-world truth.
    # ------------------------------------------------------------------
    supported_probability: float = 0.0
    contradicted_probability: float = 0.0
    unknown_probability: float = 0.0
    requires_verification: bool = True
    non_factual: bool = False
    # Canonical name for the operational triage score (same value as the
    # deprecated ``hallucination_probability``); kept separate from truth.
    verification_risk: float = 0.0
    # True when evidence selection for this claim could not use the full shared
    # hybrid retrieval + reranking stack and had to fall back to deterministic
    # lexical selection (or kept the pre-rerank hybrid order). The label and
    # probabilities above are still real model output; this only signals that
    # retrieval was degraded, so consumers can weight it accordingly.
    evidence_degraded: bool = False
    # Which evidence-selection route produced ``evidence_snippets``:
    # "hybrid", "hybrid_order" (reranker produced nothing), or "lexical".
    evidence_route: str = "hybrid"


class DetectResponse(BaseModel):
    label: str
    # DEPRECATED alias of ``verification_risk`` (kept for backward
    # compatibility). Operational triage score, NOT a calibrated probability
    # that the answer is hallucinated. The final decision is the Judge's.
    probability: float
    risk: RiskLevel
    requires_verification: bool
    sentences: list[SentenceResult]
    model_version: str
    warnings: list[str] = Field(default_factory=list)
    # ------------------------------------------------------------------
    # Answer-level aggregation. ``verification_risk`` is max over assessed
    # claims of (P(CONTRADICTED) + P(NOT_ENOUGH_INFO)): an operational score
    # that rises for both contradicted and unverified claims. The binary
    # ``label`` is driven by contradiction mass only, so a NOT_ENOUGH_INFO
    # answer is never reported as HALLUCINATION on its own. The Detector is a
    # triage layer; the Judge makes the final decision from these counts
    # alongside independent Verifier verdicts.
    # ------------------------------------------------------------------
    verification_risk: float = 0.0
    # Max P(CONTRADICTED) over assessed claims: the strongest "this is refuted"
    # signal, kept separate from ``verification_risk`` (which also rises for
    # merely unverified claims). NOT_ENOUGH_INFO never contributes to it.
    contradiction_mass: float = 0.0
    claim_count: int = 0
    supported_count: int = 0
    contradicted_count: int = 0
    unknown_count: int = 0
    non_factual_count: int = 0
    # True when at least one claim had to select evidence through a degraded
    # route (lexical fallback or pre-rerank hybrid order) because the shared
    # hybrid retrieval / reranking stack was unavailable or failed.
    evidence_degraded: bool = False
