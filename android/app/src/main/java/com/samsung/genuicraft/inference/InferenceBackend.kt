package com.samsung.genuicraft.inference

/**
 * Abstraction for LLM inference backends (Gemini API, local vLLM server, etc.).
 */
interface InferenceBackend {

    fun generate(request: GenerateRequest): GenerateResponse

    fun checkHealth(): HealthCheckResult

    /** Classifies an error string as retryable with a specific fallback strategy. */
    fun classifyError(error: String): ErrorClass

    data class GenerateRequest(
        val prompt: String,
        val systemPrompt: String?,
        val temperature: Double,
        val maxOutputTokens: Int,
        val jsonMode: Boolean,
        val enableGoogleSearch: Boolean = false,
        val cachedContentName: String? = null,
        val structuredOutput: Boolean = false,
        val localSystemPromptCacheKey: String? = null,
        val localSendSystemPrompt: Boolean = true,
        val onStreamUpdate: ((StreamUpdate) -> Unit)? = null
    )

    data class StreamUpdate(
        val text: String,
        val outputTokens: Int?,
        val outputTokensPerSecond: Double?,
        val runtimeBackend: String?,
        val metricsAreEstimated: Boolean,
        val complete: Boolean
    )

    data class GenerateResponse(
        val text: String,
        val rawResponse: String?,
        val error: String?,
        val streamDurationMs: Long?,
        val inputTokens: Int? = null,
        val outputTokens: Int? = null,
        val outputTokensPerSecond: Double? = null,
        val runtimeBackend: String? = null
    )

    data class HealthCheckResult(
        val healthy: Boolean,
        val errorMessage: String? = null
    )

    enum class ErrorClass {
        /** Not a recognized recoverable error. */
        UNKNOWN,
        /** Gemini google_search tool configuration rejected. */
        SEARCH_TOOL_CONFIG,
        /** Gemini responseSchema/structured output rejected. */
        STRUCTURED_OUTPUT_CONFIG,
        /** Gemini cached content not found / expired. */
        CACHED_CONTENT_MISSING,
        /** Local server system prompt cache miss. */
        LOCAL_CACHE_MISS,
        /** Transient error (timeout, 429, 503). */
        TRANSIENT
    }
}
