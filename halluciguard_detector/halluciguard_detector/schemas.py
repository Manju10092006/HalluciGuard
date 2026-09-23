"""
halluciguard_detector / schemas.py
──────────────────────────────────
Pydantic schemas and enumerations for halluciguard_detector.

TERMINOLOGY:
    Risk levels: LOW_RISK, UNCERTAIN, HIGH_RISK, UNKNOWN
    Routing: VERIFY (always in v1)
    Verification hint: STANDARD (for LOW_RISK), DEEP (for UNCERTAIN/HIGH_RISK/UNKNOWN)
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW_RISK = "LOW_RISK"
    UNCERTAIN = "UNCERTAIN"
    HIGH_RISK = "HIGH_RISK"
    UNKNOWN = "UNKNOWN"


class RoutingDecision(str, Enum):
    VERIFY = "VERIFY"


class VerificationHint(str, Enum):
    STANDARD = "STANDARD"
    DEEP = "DEEP"


class ExtractionMode(str, Enum):
    LLM = "LLM"
    SENTENCE_FALLBACK = "SENTENCE_FALLBACK"
    PASSTHROUGH = "PASSTHROUGH"


class StatusEnum(str, Enum):
    COMPLETED = "COMPLETED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class ExtractedClaim(BaseModel):
    """Canonical atomic claim structure."""
    claim_id: str = Field(..., description="Canonical immutable ID e.g. claim_001")
    text: str = Field(..., description="Atomic factual claim text")
    source_sentence_idx: int = Field(default=0, description="Sentence index in draft answer")
    char_span: Optional[List[int]] = Field(default=None, description="[start, end] char offset if available")


class ClaimResult(BaseModel):
    """Per-claim risk evaluation output."""
    claim_id: str
    text: str
    source_sentence_idx: int = 0
    raw_score: float = Field(..., description="Raw model risk score")
    calibrated_probability: Optional[float] = Field(default=None, description="Calibrated risk probability [0, 1]")
    risk_level: RiskLevel
    verification_hint: VerificationHint
    flags: List[str] = Field(default_factory=list)


class OverallRisk(BaseModel):
    """Response-level risk aggregation."""
    risk_score: float
    risk_level: RiskLevel
    aggregation: str = "max"
    n_claims: int = 0
    n_high: int = 0
    n_uncertain: int = 0


class CalibrationInfo(BaseModel):
    method: str = "none"
    fitted_on: str = "none"
    version: str = "v1"


class ThresholdsInfo(BaseModel):
    low: float = 0.25
    high: float = 0.65
    provisional: bool = True


class JudgeInfo(BaseModel):
    enabled: bool = False
    used: bool = False


class ModelInfo(BaseModel):
    name: str = "deberta-v3-base"
    checkpoint_id: str = "none"


class LatencyInfo(BaseModel):
    extraction_ms: float = 0.0
    scoring_ms: float = 0.0
    total_ms: float = 0.0


class DetectorResponse(BaseModel):
    """Full pipeline output contract for halluciguard_detector."""
    detector_version: str = "1.0.0"
    request_id: str
    status: StatusEnum = StatusEnum.COMPLETED
    degraded_reasons: List[str] = Field(default_factory=list)
    extraction_mode: ExtractionMode = ExtractionMode.LLM
    claims: List[ClaimResult] = Field(default_factory=list)
    overall: OverallRisk
    routing: RoutingDecision = RoutingDecision.VERIFY
    calibration: CalibrationInfo = Field(default_factory=CalibrationInfo)
    thresholds: ThresholdsInfo = Field(default_factory=ThresholdsInfo)
    judge: JudgeInfo = Field(default_factory=JudgeInfo)
    model: ModelInfo = Field(default_factory=ModelInfo)
    latency_ms: LatencyInfo = Field(default_factory=LatencyInfo)

    def to_legacy(self) -> Dict[str, Any]:
        """
        Temporary compatibility adapter for legacy consumers.
        NEVER translates UNCERTAIN or UNKNOWN into 'NO_HALLUCINATION'.
        """
        if self.overall.risk_level == RiskLevel.HIGH_RISK:
            label = "HIGH_RISK_REVERIFY"
        elif self.overall.risk_level == RiskLevel.LOW_RISK:
            label = "LOW_RISK_TENTATIVE"
        else:
            label = "UNKNOWN_UNCERTAIN"

        return {
            "label": label,
            "probability": self.overall.risk_score,
            "risk": self.overall.risk_level.value,
        }


class DetectorRequest(BaseModel):
    request_id: str
    user_query: str
    draft_answer: str
    claims: Optional[List[Dict[str, Any]]] = None
