package com.example.halluciguard.model

import androidx.room.Entity
import androidx.room.PrimaryKey
import com.squareup.moshi.JsonClass

enum class ClaimStatus {
    VERIFIED,
    HALLUCINATED,
    CONTRADICTED,
    INSUFFICIENT_EVIDENCE,
    UNCERTAIN
}

@JsonClass(generateAdapter = true)
data class AtomicClaim(
    val id: String,
    val text: String,
    val status: ClaimStatus,
    val confidenceScore: Double,
    val evidenceIds: List<String> = emptyList()
)

@JsonClass(generateAdapter = true)
data class EvidencePassage(
    val id: String,
    val sourceTitle: String,
    val passageText: String,
    val url: String? = null,
    val relevanceScore: Double = 0.95
)

@JsonClass(generateAdapter = true)
data class SourceMetadata(
    val id: String,
    val title: String,
    val authorOrOrg: String? = null,
    val url: String? = null,
    val publishDate: String? = null
)

@JsonClass(generateAdapter = true)
data class JudgeVerificationPayload(
    val query: String,
    val originalResponse: String,
    val claims: List<AtomicClaim>,
    val supportingEvidence: List<EvidencePassage>,
    val contradictionEvidence: List<EvidencePassage> = emptyList(),
    val trustScore: Double,
    val sourceMetadata: List<SourceMetadata> = emptyList(),
    val correctionInstructions: String
)

data class ClaimToEdit(
    val claim: AtomicClaim,
    val reason: String,
    val matchedEvidence: List<EvidencePassage>,
    val matchedContradictions: List<EvidencePassage>
)

data class CorrectionPlan(
    val preservedClaims: List<AtomicClaim>,
    val claimsToRewrite: List<ClaimToEdit>,
    val unsupportedClaims: List<AtomicClaim>,
    val strategyNotes: String
)

data class ClaimDiff(
    val originalClaimId: String?,
    val originalText: String,
    val status: ClaimStatus,
    val actionTaken: DiffAction,
    val correctedText: String,
    val explanation: String
)

enum class DiffAction {
    PRESERVED_EXACT,
    REWRITTEN_WITH_EVIDENCE,
    REPLACED_UNSUPPORTED_DISCLAIMER,
    REMOVED
}

data class TraceLogEntry(
    val timestamp: Long = System.currentTimeMillis(),
    val stage: String,
    val title: String,
    val message: String,
    val detailJson: String? = null,
    val durationMs: Long = 0
)

data class JudgeVerificationResult(
    val isApproved: Boolean,
    val trustScore: Double,
    val verifiedClaimsCount: Int,
    val remainingHallucinationsCount: Int,
    val feedback: String,
    val rejectionReasons: List<String> = emptyList()
)

/**
 * Enterprise LLM Observability telemetry output for every inference call
 */
data class LlmObservabilityMetrics(
    val provider: String,
    val model: String,
    val promptTokens: Int,
    val completionTokens: Int,
    val totalTokens: Int,
    val generationLatencyMs: Long,
    val retryCount: Int,
    val temperature: Double = 0.1,
    val topP: Double = 0.9,
    val finishReason: String = "stop",
    val estimatedCostUsd: Double = 0.00012,
    val cacheHit: Boolean = false,
    val circuitBreakerStatus: String = "HEALTHY_CLOSED"
)

/**
 * Enterprise Scientific Evaluation Framework metrics calculated on pipeline completion
 */
data class EvaluationMetrics(
    val hallucinationsRemovedRate: Double, // e.g. 100.0%
    val verifiedClaimsPreservationRate: Double, // e.g. 100.0%
    val newHallucinationsIntroduced: Int = 0,
    val factualPrecision: Double,
    val factualRecall: Double,
    val factualF1Score: Double,
    val judgePassRate: Double,
    val averageRetries: Int,
    val factualAlignmentScore: Double // Simulates ROUGE/BLEU ground-truth overlap
)

/**
 * System and Pipeline Versioning
 */
data class PipelineVersionInfo(
    val promptVersion: String = "v2.4.0-enterprise",
    val modelSchemaVersion: String = "v1.2-json",
    val experimentId: String = "EXP-HG-QWEN3-8B-PROD"
)

data class CorrectionAttemptResult(
    val attemptNumber: Int,
    val promptUsed: String,
    val rawLlmResponse: String,
    val processedResponse: String,
    val judgeResult: JudgeVerificationResult,
    val latencyMs: Long,
    val diffs: List<ClaimDiff>,
    val telemetry: LlmObservabilityMetrics
)

data class CorrectorExecutionResult(
    val finalResponse: String,
    val isFullyApproved: Boolean,
    val attemptsCount: Int,
    val totalLatencyMs: Long,
    val initialTrustScore: Double,
    val finalTrustScore: Double,
    val providerUsed: String,
    val modelUsed: String,
    val attempts: List<CorrectionAttemptResult>,
    val diffs: List<ClaimDiff>,
    val traceLogs: List<TraceLogEntry>,
    val observability: LlmObservabilityMetrics,
    val evaluation: EvaluationMetrics,
    val versionInfo: PipelineVersionInfo = PipelineVersionInfo(),
    val isTerminatedUnresolved: Boolean = false
)

@Entity(tableName = "audit_logs")
data class AuditLogEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val timestamp: Long = System.currentTimeMillis(),
    val query: String,
    val originalResponse: String,
    val finalResponse: String,
    val initialTrustScore: Double,
    val finalTrustScore: Double,
    val isApproved: Boolean,
    val attemptsCount: Int,
    val totalLatencyMs: Long,
    val providerUsed: String,
    val modelUsed: String,
    val promptTokens: Int = 0,
    val completionTokens: Int = 0,
    val totalTokens: Int = 0,
    val estimatedCostUsd: Double = 0.0,
    val claimsSummary: String,
    val fullTraceJson: String
)
