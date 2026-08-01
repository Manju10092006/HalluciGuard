package com.example.halluciguard.modelmanager

class ModelManager {

    private val clients = mutableMapOf<ProviderType, LlmClient>()
    private var activeProviderType: ProviderType = ProviderType.HUGGING_FACE_QWEN
    private var customApiKey: String? = null
    private var customHfToken: String? = null

    init {
        // Pre-initialize provider registry
        clients[ProviderType.HUGGING_FACE_QWEN] = HuggingFaceRouterClient(customHfToken)
        clients[ProviderType.GEMINI_CLOUD] = GeminiCloudClient(customApiKey)
        clients[ProviderType.LOCAL_ON_DEVICE] = LocalOnDeviceClient()
        clients[ProviderType.MOCK_ENTERPRISE] = MockEnterpriseClient()
    }

    fun setApiKey(apiKey: String) {
        this.customApiKey = apiKey
        clients[ProviderType.GEMINI_CLOUD] = GeminiCloudClient(apiKey)
    }

    fun setHfToken(hfToken: String) {
        this.customHfToken = hfToken
        clients[ProviderType.HUGGING_FACE_QWEN] = HuggingFaceRouterClient(hfToken)
    }

    fun setActiveProvider(providerType: ProviderType) {
        this.activeProviderType = providerType
    }

    fun getActiveProviderType(): ProviderType = activeProviderType

    fun getActiveClient(): LlmClient {
        val client = clients[activeProviderType]
        // Automatic Provider Failover Policy:
        // If active client circuit breaker is tripped, failover to Local On-Device Engine
        if (client != null && client.isCircuitBreakerTripped()) {
            return clients[ProviderType.LOCAL_ON_DEVICE]!!
        }
        return client ?: clients[ProviderType.LOCAL_ON_DEVICE]!!
    }

    fun getProviderMetricsSummary(): String {
        val client = getActiveClient()
        return "Provider: ${client.getProviderName()} | Model: ${client.getModelName()} | CircuitBreaker: ${if (client.isCircuitBreakerTripped()) "TRIPPED_FAILOVER" else "CLOSED_HEALTHY"}"
    }

    companion object {
        val instance: ModelManager by lazy { ModelManager() }
    }
}
