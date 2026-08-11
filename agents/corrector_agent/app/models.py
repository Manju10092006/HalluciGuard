from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
import time

class ClaimStatus(str, Enum):
    VERIFIED = "VERIFIED"
    HALLUCINATED = "HALLUCINATED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNCERTAIN = "UNCERTAIN"

class AtomicClaim(BaseModel):
    id: str
    text: str
    status: ClaimStatus
    confidenceScore: float
    evidenceIds: List[str] = Field(default_factory=list)

class EvidencePassage(BaseModel):
    id: str
    sourceTitle: str
    passageText: str
    url: Optional[str] = None
    relevanceScore: float = 0.95

class SourceMetadata(BaseModel):
    id: str
    title: str
    authorOrOrg: Optional[str] = None
    url: Optional[str] = None
    publishDate: Optional[str] = None

class JudgeVerificationPayload(BaseModel):
    query: str
    originalResponse: str
    claims: List[AtomicClaim]
    supportingEvidence: List[EvidencePassage]
    contradictionEvidence: List[EvidencePassage] = Field(default_factory=list)
    trustScore: float
    sourceMetadata: List[SourceMetadata] = Field(default_factory=list)
    correctionInstructions: str

class ClaimToEdit(BaseModel):
    claim: AtomicClaim
    reason: str
    matchedEvidence: List[EvidencePassage]
    matchedContradictions: List[EvidencePassage]

class CorrectionPlan(BaseModel):
    preservedClaims: List[AtomicClaim]
    claimsToRewrite: List[ClaimToEdit]
    unsupportedClaims: List[AtomicClaim]
    strategyNotes: str

class DiffAction(str, Enum):
    PRESERVED_EXACT = "PRESERVED_EXACT"
    REWRITTEN_WITH_EVIDENCE = "REWRITTEN_WITH_EVIDENCE"
    REPLACED_UNSUPPORTED_DISCLAIMER = "REPLACED_UNSUPPORTED_DISCLAIMER"
    REMOVED = "REMOVED"

class ClaimDiff(BaseModel):
    originalClaimId: Optional[str]
    originalText: str
    status: ClaimStatus
    actionTaken: DiffAction
    correctedText: str
    explanation: str

def current_time_ms():
    return int(time.time() * 1000)

class TraceLogEntry(BaseModel):
    timestamp: int = Field(default_factory=current_time_ms)
    stage: str
    title: str
    message: str
    detailJson: Optional[str] = None
    durationMs: int = 0

class JudgeVerificationResult(BaseModel):
    isApproved: bool
    trustScore: float
    verifiedClaimsCount: int
    remainingHallucinationsCount: int
    feedback: str
    rejectionReasons: List[str] = Field(default_factory=list)

class LlmObservabilityMetrics(BaseModel):
    provider: str
    model: str
    promptTokens: int
    completionTokens: int
    totalTokens: int
    generationLatencyMs: int
    retryCount: int
    temperature: float = 0.1
    topP: float = 0.9
    finishReason: str = "stop"
    estimatedCostUsd: float = 0.00012
    cacheHit: bool = False
    circuitBreakerStatus: str = "HEALTHY_CLOSED"

class EvaluationMetrics(BaseModel):
    hallucinationsRemovedRate: float
    verifiedClaimsPreservationRate: float
    newHallucinationsIntroduced: int = 0
    factualPrecision: float
    factualRecall: float
    factualF1Score: float
    judgePassRate: float
    averageRetries: int
    factualAlignmentScore: float

class PipelineVersionInfo(BaseModel):
    promptVersion: str = "v2.4.0-enterprise"
    modelSchemaVersion: str = "v1.2-json"
    experimentId: str = "EXP-HG-QWEN3-8B-PROD"

class CorrectionAttemptResult(BaseModel):
    attemptNumber: int
    promptUsed: str
    rawLlmResponse: str
    processedResponse: str
    judgeResult: JudgeVerificationResult
    latencyMs: int
    diffs: List[ClaimDiff]
    telemetry: LlmObservabilityMetrics

class CorrectorExecutionResult(BaseModel):
    finalResponse: str
    isFullyApproved: bool
    attemptsCount: int
    totalLatencyMs: int
    initialTrustScore: float
    finalTrustScore: float
    providerUsed: str
    modelUsed: str
    attempts: List[CorrectionAttemptResult]
    diffs: List[ClaimDiff]
    traceLogs: List[TraceLogEntry]
    observability: LlmObservabilityMetrics
    evaluation: EvaluationMetrics
    versionInfo: PipelineVersionInfo = Field(default_factory=PipelineVersionInfo)
    isTerminatedUnresolved: bool = False
    baseLlmRegenerationPrompt: Optional[str] = None

class AuditLogEntity(BaseModel):
    id: Optional[int] = None
    timestamp: int = Field(default_factory=current_time_ms)
    query: str
    originalResponse: str
    finalResponse: str
    initialTrustScore: float
    finalTrustScore: float
    isApproved: bool
    attemptsCount: int
    totalLatencyMs: int
    providerUsed: str
    modelUsed: str
    promptTokens: int = 0
    completionTokens: int = 0
    totalTokens: int = 0
    estimatedCostUsd: float = 0.0
    claimsSummary: str
    fullTraceJson: str
