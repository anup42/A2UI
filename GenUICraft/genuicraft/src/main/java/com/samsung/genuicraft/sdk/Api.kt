package com.samsung.genuicraft.sdk

data class GenUiRequest(
    val text: String,
    /** Optional host context, excluded from formatting prompts to avoid re-answering its instructions. */
    val query: String? = null,
    val sources: List<GenUiSource> = emptyList(),
)

data class GenUiSource(val id: String, val url: String, val title: String? = null)

data class GenUiDocument(
    val express: String,
    val a2uiJson: String,
    val profile: String = "genuicraft_express_v1",
    val schemaVersion: String = "v0.9",
)

data class GenUiAction(val name: String, val parameters: Map<String, String> = emptyMap())

data class GenUiPrompt(
    val system: String,
    val user: String,
    val maxOutputTokens: Int = 8192,
    val temperature: Double = 0.0,
    /** Prior turns, for profiles whose training contract includes a worked example. */
    val initialMessages: List<GenUiPromptMessage> = emptyList(),
    /** Optional native template adapter; scoped to this conversation only. */
    val chatTemplateOverride: String? = null,
    /** When supplied, native rendering must match this exact training prefix before inference. */
    val expectedRenderedPrompt: String? = null,
)

/** How [GenUiCompiler.compileWithRepair] produced an accepted document. */
enum class GenUiRepairKind {
    /** The supplied program passed the normal strict compiler unchanged. */
    NONE,
    /** Only bounded syntax normalization was needed; visible values and graph structure were not rewritten. */
    STRUCTURAL,
    /** General model-output-only DSL salvage; source text is never imported into this document. */
    GENERATED_DSL_REPAIR,
    /** The generated program was rejected and a source-bound document was rebuilt from source text. */
    SOURCE_TEXT_FALLBACK,
}

/** Auditable result of opt-in generated-output recovery. */
data class GenUiCompileOutcome(
    val document: GenUiDocument,
    val repairKind: GenUiRepairKind,
    val diagnostics: List<String> = emptyList(),
)

enum class GenUiPromptRole { USER, MODEL }

data class GenUiPromptMessage(val role: GenUiPromptRole, val text: String)

/** Actual native measurements or server-reported usage; unavailable values are null, never estimated. */
data class GenUiGenerationMetrics(
    /** Native tokens for the last prefill, or server-reported prompt tokens. */
    val inputTokens: Int?,
    /** Native tokens for the last decode (including reasoning), or server-reported completion tokens. */
    val outputTokens: Int?,
    /** Native decode throughput; excludes model initialization and prompt prefill. */
    val decodeTokensPerSecond: Double?,
    /** Native prompt-prefill throughput. */
    val prefillTokensPerSecond: Double? = null,
    /** Native latency from generation start to the first decoded token. */
    val timeToFirstTokenSeconds: Double? = null,
    /** Measured engine initialization wall time when initialization occurred in this request. */
    val engineInitializationSeconds: Double? = null,
    /** True for the request that initialized the native engine, false when it reused the engine. */
    val engineInitializedForRequest: Boolean? = null,
    /** LiteRT's sum of native init phases; phases can overlap, so this is not additive wall time. */
    val nativeInitializationPhaseSeconds: Double? = null,
)

/** Metrics that LiteRT can publish only when one complete native engine session ends. */
data class GenUiGenerationSessionMetrics(
    /** Whether the session actually used the MTP/speculative drafter. */
    val speculativeDecodingEnabled: Boolean,
    /** Verified draft tokens divided by all proposed draft tokens, in the range 0.0..1.0. */
    val drafterAcceptanceRate: Double? = null,
)

/** Why native generation ended. A repetition cutoff still returns its accumulated text for repair. */
enum class GenUiGenerationFinishReason {
    COMPLETED,
    REPETITION_LIMIT,
}

data class GenUiModelOutput(
    val text: String,
    val runtime: String,
    val outputTokens: Int? = null,
    val metrics: GenUiGenerationMetrics? = null,
    val renderedPromptSha256: String? = null,
    val finishReason: GenUiGenerationFinishReason = GenUiGenerationFinishReason.COMPLETED,
    val finishDetail: String? = null,
)

interface GenUiProvider : AutoCloseable {
    val id: String
    suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput
    /**
     * Reports cumulative raw output, potentially on a native/provider worker thread. The observer
     * must return promptly and dispatch UI work to main. Providers without streaming support report
     * one final snapshot. Repair/compilation never modifies these snapshots.
     */
    suspend fun generate(prompt: GenUiPrompt, onPartialText: (String) -> Unit): GenUiModelOutput {
        return generate(prompt).also { onPartialText(it.text) }
    }
    /**
     * Ends an opt-in metrics session after all conversion attempts. Providers return null when
     * they have no session-level telemetry. Normal generation must not depend on this hook.
     */
    suspend fun finishGenerationMetrics(): GenUiGenerationSessionMetrics? = null
    suspend fun closeAndAwait() {
        close()
    }
    override fun close() {}
}

sealed interface GenUiConversionResult {
    data class Success(
        val document: GenUiDocument,
        val provider: String,
        val elapsedMs: Long,
        val attempts: Int,
        val warnings: List<String> = emptyList(),
        /** Stable recovery classification; hosts do not need to parse warning text. */
        val repairKind: GenUiRepairKind = GenUiRepairKind.NONE,
    ) : GenUiConversionResult

    data class Failure(
        val message: String,
        val provider: String,
        val elapsedMs: Long,
        val attempts: Int,
        val rawOutput: String? = null,
    ) : GenUiConversionResult
}

data class ConversionOptions(
    val maxOutputTokens: Int = 8192,
    val maxRepairAttempts: Int = 1,
    val temperature: Double = 0.0,
)
