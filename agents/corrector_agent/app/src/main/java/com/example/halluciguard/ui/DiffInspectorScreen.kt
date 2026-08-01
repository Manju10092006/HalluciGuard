package com.example.halluciguard.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Analytics
import androidx.compose.material.icons.filled.ArrowForward
import androidx.compose.material.icons.filled.CompareArrows
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Science
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Verified
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.halluciguard.model.ClaimDiff
import com.example.halluciguard.model.DiffAction
import com.example.halluciguard.viewmodel.CorrectorUiState
import com.example.ui.theme.Amber500
import com.example.ui.theme.Coral500
import com.example.ui.theme.Emerald500
import com.example.ui.theme.Indigo500

@Composable
fun DiffInspectorScreen(state: CorrectorUiState) {
    val result = state.executionResult

    if (result == null) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Icon(
                    Icons.Default.CompareArrows,
                    contentDescription = null,
                    modifier = Modifier.size(48.dp),
                    tint = MaterialTheme.colorScheme.primary
                )
                Text(
                    "No Pipeline Execution Results Yet",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    "Run the Corrector Agent in the Test Bench tab to inspect claim diffs.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
        return
    }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Status & Trust Score Gauge Banner
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = if (result.isFullyApproved) {
                        Emerald500.copy(alpha = 0.12f)
                    } else {
                        Coral500.copy(alpha = 0.12f)
                    }
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = if (result.isFullyApproved) Icons.Default.Verified else Icons.Default.Error,
                                contentDescription = null,
                                tint = if (result.isFullyApproved) Emerald500 else Coral500,
                                modifier = Modifier.size(24.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = if (result.isFullyApproved) "JUDGE VERIFIED & APPROVED" else "TERMINATED UNRESOLVED",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                                color = if (result.isFullyApproved) Emerald500 else Coral500
                            )
                        }

                        Surface(
                            shape = RoundedCornerShape(8.dp),
                            color = MaterialTheme.colorScheme.surface
                        ) {
                            Row(
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(
                                    Icons.Default.Speed,
                                    contentDescription = null,
                                    modifier = Modifier.size(14.dp),
                                    tint = MaterialTheme.colorScheme.primary
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Text(
                                    "${result.totalLatencyMs} ms",
                                    style = MaterialTheme.typography.labelSmall,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    // Trust Score Progression
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center
                    ) {
                        TrustScoreGaugeItem(
                            label = "Initial Trust",
                            score = result.initialTrustScore
                        )
                        Spacer(modifier = Modifier.width(16.dp))
                        Icon(
                            Icons.Default.ArrowForward,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(16.dp))
                        TrustScoreGaugeItem(
                            label = "Refined Trust",
                            score = result.finalTrustScore
                        )
                    }

                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Provider: ${result.providerUsed} (${result.modelUsed}) • Retries: ${result.attemptsCount - 1}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.align(Alignment.CenterHorizontally)
                    )
                }
            }
        }

        // Enterprise LLM Observability Metrics
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            Icons.Default.Analytics,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            "LLM OBSERVABILITY TELEMETRY",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary,
                            fontWeight = FontWeight.Bold
                        )
                    }

                    Spacer(modifier = Modifier.height(12.dp))
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        MetricBox("Prompt Tokens", "${result.observability.promptTokens}")
                        MetricBox("Completion Tokens", "${result.observability.completionTokens}")
                        MetricBox("Total Tokens", "${result.observability.totalTokens}")
                        MetricBox("Cost (USD)", "\$${String.format("%.6f", result.observability.estimatedCostUsd)}")
                    }

                    Spacer(modifier = Modifier.height(8.dp))
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        MetricBox("Generation Latency", "${result.observability.generationLatencyMs} ms")
                        MetricBox("Temperature / TopP", "${result.observability.temperature} / ${result.observability.topP}")
                        MetricBox("Circuit Breaker", result.observability.circuitBreakerStatus)
                        MetricBox("Cache Status", if (result.observability.cacheHit) "HIT" else "MISS")
                    }
                }
            }
        }

        // Scientific Evaluation Framework Results
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            Icons.Default.Science,
                            contentDescription = null,
                            tint = Emerald500
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            "SCIENTIFIC EVALUATION BENCHMARK METRICS",
                            style = MaterialTheme.typography.labelSmall,
                            color = Emerald500,
                            fontWeight = FontWeight.Bold
                        )
                    }

                    Spacer(modifier = Modifier.height(12.dp))
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        MetricBox("Hallucinations Removed", "${String.format("%.1f", result.evaluation.hallucinationsRemovedRate)}%", isHighlight = true)
                        MetricBox("Claims Preserved", "${String.format("%.1f", result.evaluation.verifiedClaimsPreservationRate)}%", isHighlight = true)
                        MetricBox("Factual F1 Score", String.format("%.2f", result.evaluation.factualF1Score), isHighlight = true)
                    }

                    Spacer(modifier = Modifier.height(8.dp))
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        MetricBox("Precision / Recall", "${String.format("%.2f", result.evaluation.factualPrecision)} / ${String.format("%.2f", result.evaluation.factualRecall)}")
                        MetricBox("New Hallucinations", "${result.evaluation.newHallucinationsIntroduced}")
                        MetricBox("Factual Alignment", String.format("%.3f", result.evaluation.factualAlignmentScore))
                    }
                }
            }
        }

        // Side-by-Side Comparison: Original vs Refined Output
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        "UNVERIFIED ORIGINAL LLM RESPONSE",
                        style = MaterialTheme.typography.labelSmall,
                        color = Coral500,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        state.currentPayload.originalResponse,
                        style = MaterialTheme.typography.bodyMedium
                    )

                    Spacer(modifier = Modifier.height(12.dp))
                    HorizontalDivider(color = MaterialTheme.colorScheme.surfaceVariant)
                    Spacer(modifier = Modifier.height(12.dp))

                    Text(
                        "CORRECTED EVIDENCE-GROUNDED RESPONSE",
                        style = MaterialTheme.typography.labelSmall,
                        color = Emerald500,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        result.finalResponse,
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }
        }

        // Claim-by-Claim Diff Inspector Section
        item {
            Text(
                "CLAIM-LEVEL EDITING & PRESERVATION AUDIT",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold
            )
        }

        items(result.diffs) { diff ->
            ClaimDiffCard(diff)
        }
    }
}

@Composable
fun MetricBox(title: String, value: String, isHighlight: Boolean = false) {
    Surface(
        shape = RoundedCornerShape(8.dp),
        color = if (isHighlight) Emerald500.copy(alpha = 0.12f) else MaterialTheme.colorScheme.surfaceVariant
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                title,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                value,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Bold,
                color = if (isHighlight) Emerald500 else MaterialTheme.colorScheme.onSurface
            )
        }
    }
}

@Composable
fun TrustScoreGaugeItem(label: String, score: Double) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            "${(score * 100).toInt()}%",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.ExtraBold,
            color = if (score >= 0.85) Emerald500 else Coral500
        )
    }
}

@Composable
fun ClaimDiffCard(diff: ClaimDiff) {
    val (badgeColor, badgeTitle) = when (diff.actionTaken) {
        DiffAction.PRESERVED_EXACT -> Pair(Emerald500, "VERIFIED CLAIM PRESERVED")
        DiffAction.REWRITTEN_WITH_EVIDENCE -> Pair(Indigo500, "HALLUCINATION REWRITTEN")
        DiffAction.REPLACED_UNSUPPORTED_DISCLAIMER -> Pair(Amber500, "DISCLAIMER REPLACED")
        DiffAction.REMOVED -> Pair(Coral500, "REMOVED")
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Surface(
                    shape = RoundedCornerShape(6.dp),
                    color = badgeColor.copy(alpha = 0.15f)
                ) {
                    Text(
                        text = badgeTitle,
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = badgeColor,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp)
                    )
                }
                Text(
                    text = "ID: ${diff.originalClaimId ?: "N/A"}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Original: \"${diff.originalText}\"",
                style = MaterialTheme.typography.bodySmall,
                color = Coral500
            )

            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Corrected: \"${diff.correctedText}\"",
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold,
                color = Emerald500
            )

            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = "Reason: ${diff.explanation}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}
