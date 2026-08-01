package com.example.halluciguard.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.halluciguard.db.AuditDatabase
import com.example.halluciguard.db.AuditRepository
import com.example.halluciguard.model.AuditLogEntity
import com.example.halluciguard.model.CorrectorExecutionResult
import com.example.halluciguard.model.JudgeVerificationPayload
import com.example.halluciguard.modelmanager.ModelManager
import com.example.halluciguard.modelmanager.ProviderType
import com.example.halluciguard.orchestrator.CorrectorOrchestrator
import com.example.halluciguard.samples.SampleData
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class CorrectorUiState(
    val selectedScenarioId: String = SampleData.scenarios.first().id,
    val currentPayload: JudgeVerificationPayload = SampleData.scenarios.first().payload,
    val payloadJsonText: String = "",
    val activeProvider: ProviderType = ProviderType.HUGGING_FACE_QWEN,
    val maxRetries: Int = 3,
    val apiKeyInput: String = "",
    val hfTokenInput: String = "",
    val isExecuting: Boolean = false,
    val currentPipelineStep: String = "Idle",
    val executionResult: CorrectorExecutionResult? = null,
    val auditLogs: List<AuditLogEntity> = emptyList(),
    val activeTab: Int = 0,
    val errorMessage: String? = null,
    val successSnackbarMessage: String? = null
)

class CorrectorViewModel(application: Application) : AndroidViewModel(application) {

    private val db = AuditDatabase.getDatabase(application)
    private val repository = AuditRepository(db.auditLogDao())
    private val orchestrator = CorrectorOrchestrator()
    private val modelManager = ModelManager.instance

    private val moshi = Moshi.Builder()
        .addLast(KotlinJsonAdapterFactory())
        .build()
    private val payloadAdapter = moshi.adapter(JudgeVerificationPayload::class.java)

    private val _uiState = MutableStateFlow(CorrectorUiState())
    val uiState: StateFlow<CorrectorUiState> = _uiState.asStateFlow()

    init {
        val initialPayload = SampleData.scenarios.first().payload
        val initialJson = try {
            payloadAdapter.indent("  ").toJson(initialPayload)
        } catch (e: Exception) {
            ""
        }

        _uiState.update {
            it.copy(
                payloadJsonText = initialJson,
                activeProvider = modelManager.getActiveProviderType()
            )
        }

        viewModelScope.launch {
            repository.allLogs.collect { logs ->
                _uiState.update { it.copy(auditLogs = logs) }
            }
        }
    }

    fun selectScenario(scenarioId: String) {
        val item = SampleData.scenarios.find { it.id == scenarioId } ?: return
        val json = try {
            payloadAdapter.indent("  ").toJson(item.payload)
        } catch (e: Exception) {
            ""
        }

        _uiState.update {
            it.copy(
                selectedScenarioId = scenarioId,
                currentPayload = item.payload,
                payloadJsonText = json,
                executionResult = null,
                errorMessage = null
            )
        }
    }

    fun updatePayloadJson(json: String) {
        _uiState.update { it.copy(payloadJsonText = json) }
        try {
            val parsed = payloadAdapter.fromJson(json)
            if (parsed != null) {
                _uiState.update { it.copy(currentPayload = parsed, errorMessage = null) }
            }
        } catch (e: Exception) {
            // Keep text, user editing in progress
        }
    }

    fun setProvider(providerType: ProviderType) {
        modelManager.setActiveProvider(providerType)
        _uiState.update { it.copy(activeProvider = providerType) }
    }

    fun setApiKey(key: String) {
        modelManager.setApiKey(key)
        _uiState.update { it.copy(apiKeyInput = key, successSnackbarMessage = "Gemini API Key updated successfully!") }
    }

    fun setHfToken(token: String) {
        modelManager.setHfToken(token)
        _uiState.update { it.copy(hfTokenInput = token, successSnackbarMessage = "Hugging Face Token updated successfully!") }
    }

    fun setMaxRetries(retries: Int) {
        _uiState.update { it.copy(maxRetries = retries) }
    }

    fun setActiveTab(tabIndex: Int) {
        _uiState.update { it.copy(activeTab = tabIndex) }
    }

    fun clearSnackbar() {
        _uiState.update { it.copy(successSnackbarMessage = null, errorMessage = null) }
    }

    fun runCorrectionPipeline() {
        val state = _uiState.value
        val payload = state.currentPayload

        _uiState.update {
            it.copy(
                isExecuting = true,
                currentPipelineStep = "1. Analyzing Claims & Evidence (Planner)",
                errorMessage = null,
                executionResult = null
            )
        }

        viewModelScope.launch {
            try {
                val result = orchestrator.executeCorrectionPipeline(
                    payload = payload,
                    maxRetries = state.maxRetries
                )

                // Save to Room Audit Database with Observability Telemetry
                val auditLog = AuditLogEntity(
                    query = payload.query,
                    originalResponse = payload.originalResponse,
                    finalResponse = result.finalResponse,
                    initialTrustScore = result.initialTrustScore,
                    finalTrustScore = result.finalTrustScore,
                    isApproved = result.isFullyApproved,
                    attemptsCount = result.attemptsCount,
                    totalLatencyMs = result.totalLatencyMs,
                    providerUsed = result.providerUsed,
                    modelUsed = result.modelUsed,
                    promptTokens = result.observability.promptTokens,
                    completionTokens = result.observability.completionTokens,
                    totalTokens = result.observability.totalTokens,
                    estimatedCostUsd = result.observability.estimatedCostUsd,
                    claimsSummary = "Total Claims: ${payload.claims.size}, Preserved: ${result.diffs.count { it.actionTaken == com.example.halluciguard.model.DiffAction.PRESERVED_EXACT }}, Rewritten: ${result.diffs.count { it.actionTaken == com.example.halluciguard.model.DiffAction.REWRITTEN_WITH_EVIDENCE }}",
                    fullTraceJson = result.traceLogs.joinToString("\n") { "[${it.stage}] ${it.title}: ${it.message}" }
                )
                repository.saveAuditLog(auditLog)

                _uiState.update {
                    it.copy(
                        isExecuting = false,
                        currentPipelineStep = "Completed",
                        executionResult = result,
                        activeTab = 1 // Auto switch to Diff Inspector on completion!
                    )
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isExecuting = false,
                        currentPipelineStep = "Failed",
                        errorMessage = "Error during execution: ${e.localizedMessage}"
                    )
                }
            }
        }
    }

    fun clearAuditHistory() {
        viewModelScope.launch {
            repository.clearLogs()
            _uiState.update { it.copy(successSnackbarMessage = "Audit log history cleared.") }
        }
    }
}
