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
    hallucination_probability: float
    risk: RiskLevel
    evidence_snippets: list[str]


class DetectResponse(BaseModel):
    label: str
    probability: float
    risk: RiskLevel
    requires_verification: bool
    sentences: list[SentenceResult]
    model_version: str
    warnings: list[str] = Field(default_factory=list)
