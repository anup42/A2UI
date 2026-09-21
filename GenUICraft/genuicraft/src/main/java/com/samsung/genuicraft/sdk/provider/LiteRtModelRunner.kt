package com.samsung.genuicraft.sdk.provider

import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Capabilities
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig
import java.io.File
import java.util.Locale
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** Accelerators supported by the reusable LiteRT-LM model runner. */
enum class LiteRtAccelerator {
    /** Prefer GPU and use CPU only when the selected model policy allows it. */
    AUTO,
    GPU,
    CPU,
    NPU,
}

/** Role for an initial conversation turn supplied to [LiteRtGenerationRequest]. */
enum class LiteRtConversationRole {
    USER,
    MODEL,
}

/** Initial conversation turn for model profiles that include worked examples. */
data class LiteRtConversationMessage(
    val role: LiteRtConversationRole,
    val text: String,
)

/**
 * Model and runtime policy for [LiteRtModelRunner].
 *
 * Hosts remain responsible for model discovery and user preferences. Once resolved, those values
 * are passed here so every UI surface uses the same validation and native inference behavior.
 */
data class LiteRtModelConfig(
    val modelPath: String,
    val accelerator: LiteRtAccelerator = LiteRtAccelerator.AUTO,
    val npuNativeLibraryDir: String = "",
    val maxContextTokens: Int = 12_288,
    val maxOutputTokens: Int = 4_096,
    val minimumFileSizeBytes: Long = 1L,
    val modelDisplayName: String? = null,
    val requireGpu: Boolean = false,
    val enableSpeculativeDecoding: Boolean = false,
    val useRawTrainingWrapper: Boolean = false,
    val allowGpuQualityFallback: Boolean = false,
)

/** A synchronous LiteRT-LM request. Callers should invoke it away from their UI thread. */
data class LiteRtGenerationRequest(
    val prompt: String,
    val systemPrompt: String? = null,
    val temperature: Double = 0.0,
    val maxOutputTokens: Int = 4_096,
    val sendSystemPrompt: Boolean = true,
    val initialMessages: List<LiteRtConversationMessage> = emptyList(),
    val onStreamUpdate: ((LiteRtStreamUpdate) -> Unit)? = null,
)

/** Generation progress. The current native sync path emits an initial and a final snapshot. */
data class LiteRtStreamUpdate(
    val text: String,
    val outputTokens: Int?,
    val outputTokensPerSecond: Double?,
    val runtimeBackend: String?,
    val metricsAreEstimated: Boolean,
    val complete: Boolean,
)

/** Text and performance data returned by [LiteRtModelRunner.generate]. */
data class LiteRtGenerationResult(
    val text: String,
    val inputTokens: Int?,
    val outputTokens: Int,
    val outputTokensPerSecond: Double,
    val runtimeBackend: String,
    val metricsAreEstimated: Boolean,
)

/** Non-throwing model-file readiness result. */
data class LiteRtModelHealth(
    val healthy: Boolean,
    val errorMessage: String? = null,
)

/**
 * Reusable LiteRT-LM engine and conversation runner.
 *
 * The native engine cache is process-wide so different host views can reuse the same loaded model.
 * Requests are serialized because LiteRT-LM engines must not run overlapping conversations.
 */
class LiteRtModelRunner(
    private val config: LiteRtModelConfig,
) {
    fun checkHealth(): LiteRtModelHealth = LiteRtModelPolicy.checkModelHealth(config)

    fun generate(request: LiteRtGenerationRequest): LiteRtGenerationResult {
        checkHealth().errorMessage?.let { error(it) }
        return LiteRtSharedEngineCache.withGenerationLock {
            generateLocked(request)
        }
    }

    private fun generateLocked(request: LiteRtGenerationRequest): LiteRtGenerationResult {
        val promptParts = LiteRtModelPolicy.preparePrompt(
            request = request,
            useRawTrainingWrapper = config.useRawTrainingWrapper,
        )
        val modelFile = File(config.modelPath.trim())
        val maxContextTokens = LiteRtModelPolicy.maxContextTokens(
            prompt = promptParts.combinedForEstimates,
            requestedMaxOutputTokens = request.maxOutputTokens,
            modelMaxContextTokens = config.maxContextTokens,
            modelMaxOutputTokens = config.maxOutputTokens,
        )
        val requireGpuForInitialization = config.requireGpu && !(
            config.allowGpuQualityFallback && config.accelerator == LiteRtAccelerator.AUTO
        )
        val holder = LiteRtSharedEngineCache.getOrCreateEngine(
            modelFile = modelFile,
            maxContextTokens = maxContextTokens,
            forceCpu = false,
            requireGpu = requireGpuForInitialization,
            enableSpeculativeDecoding = config.enableSpeculativeDecoding,
            accelerator = config.accelerator,
            npuNativeLibraryDir = config.npuNativeLibraryDir,
        )
        val maxOutputTokens = min(request.maxOutputTokens, config.maxOutputTokens)
        val generation = try {
            generateWithEngine(holder, promptParts, request, maxOutputTokens)
        } catch (gpuFailure: Throwable) {
            if (
                holder.backendName != BACKEND_GPU ||
                holder.speculativeDecodingEnabled ||
                config.accelerator != LiteRtAccelerator.AUTO ||
                (!config.allowGpuQualityFallback && config.requireGpu)
            ) {
                throw gpuFailure
            }
            Log.w(LOG_TAG, "LiteRT GPU generation failed; retrying on CPU: ${gpuFailure.message}")
            LiteRtSharedEngineCache.release()
            generateWithEngine(
                holder = cpuFallbackHolder(modelFile, maxContextTokens),
                promptParts = promptParts,
                request = request,
                maxOutputTokens = maxOutputTokens,
            )
        }
        return if (
            generation.runtimeBackend == BACKEND_GPU &&
            config.accelerator == LiteRtAccelerator.AUTO &&
            config.allowGpuQualityFallback &&
            LiteRtModelPolicy.isInvalidGpuOutput(generation.text)
        ) {
            Log.w(
                LOG_TAG,
                "LiteRT GPU returned invalid special-token output; retrying on CPU. " +
                    "gpuText=${generation.text.take(160)}",
            )
            LiteRtSharedEngineCache.release()
            generateWithEngine(
                holder = cpuFallbackHolder(modelFile, maxContextTokens),
                promptParts = promptParts,
                request = request,
                maxOutputTokens = maxOutputTokens,
            )
        } else {
            generation
        }
    }

    private fun cpuFallbackHolder(modelFile: File, maxContextTokens: Int): EngineHolder {
        return LiteRtSharedEngineCache.getOrCreateEngine(
            modelFile = modelFile,
            maxContextTokens = maxContextTokens,
            forceCpu = true,
            requireGpu = false,
            enableSpeculativeDecoding = config.enableSpeculativeDecoding,
            accelerator = LiteRtAccelerator.CPU,
            npuNativeLibraryDir = config.npuNativeLibraryDir,
        )
    }

    private fun generateWithEngine(
        holder: EngineHolder,
        promptParts: LiteRtModelPolicy.PreparedPrompt,
        request: LiteRtGenerationRequest,
        maxOutputTokens: Int,
    ): LiteRtGenerationResult {
        val startedAt = System.currentTimeMillis()
        val approximateInputTokens = LiteRtModelPolicy.estimateTokens(promptParts.combinedForEstimates)
        val runtimeBackend = LiteRtModelPolicy.runtimeBackendLabel(
            backendName = holder.backendName,
            speculativeDecodingEnabled = holder.speculativeDecodingEnabled,
        )
        Log.i(
            LOG_TAG,
            "LiteRT IR generation start backend=${holder.backendName} context=${holder.maxContextTokens} " +
                "mtp=${holder.speculativeDecodingEnabled} inputTokensApprox=$approximateInputTokens",
        )
        val deterministic = request.temperature <= 0.0
        val conversationConfig = ConversationConfig(
            systemInstruction = promptParts.system?.let(Contents::of),
            initialMessages = promptParts.initialMessages.map { message ->
                when (message.role) {
                    LiteRtConversationRole.USER -> Message.user(message.text)
                    LiteRtConversationRole.MODEL -> Message.model(message.text)
                }
            },
            samplerConfig = SamplerConfig(
                temperature = request.temperature,
                topK = if (deterministic) 1 else 32,
                topP = if (deterministic) 1.0 else 0.9,
            ),
            maxOutputToken = maxOutputTokens.coerceAtLeast(1),
        )
        request.onStreamUpdate?.invoke(
            LiteRtStreamUpdate(
                text = "",
                outputTokens = 0,
                outputTokensPerSecond = null,
                runtimeBackend = runtimeBackend,
                metricsAreEstimated = true,
                complete = false,
            ),
        )
        // A different SDK provider may temporarily install a trained chat template. Explicitly
        // select the model package's default while creating this conversation, then restore it.
        val conversation = LiteRtRuntimeFlags.withPromptTemplate(null) {
            holder.engine.createConversation(conversationConfig)
        }
        val generation = conversation.use { conversation ->
            // The LiteRT-LM callback path can finish without exposing final GPU text. The sync API
            // is also used by the official GPU probe and reliably returns the final Content.Text.
            val responseMessage = conversation.sendMessage(promptParts.user)
            val text = responseMessage.textContent().trim()
            Log.i(
                LOG_TAG,
                "LiteRT sync response role=${responseMessage.role} " +
                    "contentTypes=${responseMessage.contents.contents.map { it.javaClass.name }} " +
                    "textChars=${text.length}",
            )
            val benchmark = readBenchmarkSnapshot(conversation)
            val nativeOutputTokens = benchmark?.outputTokens?.takeIf { it > 0 }
            val outputTokens = nativeOutputTokens ?: LiteRtModelPolicy.estimateTokens(text)
            val nativeRate = benchmark?.outputTokensPerSecond
                ?.takeIf { it.isFinite() && it > 0.0 }
            val elapsedMs = (System.currentTimeMillis() - startedAt).coerceAtLeast(1L)
            val outputTokensPerSecond = nativeRate ?: outputTokens * 1_000.0 / elapsedMs
            val metricsAreEstimated = nativeOutputTokens == null || nativeRate == null
            request.onStreamUpdate?.invoke(
                LiteRtStreamUpdate(
                    text = text,
                    outputTokens = outputTokens,
                    outputTokensPerSecond = outputTokensPerSecond,
                    runtimeBackend = runtimeBackend,
                    metricsAreEstimated = metricsAreEstimated,
                    complete = true,
                ),
            )
            LiteRtGenerationResult(
                text = text,
                inputTokens = benchmark?.inputTokens?.takeIf { it > 0 } ?: approximateInputTokens,
                outputTokens = outputTokens,
                outputTokensPerSecond = outputTokensPerSecond,
                runtimeBackend = runtimeBackend,
                metricsAreEstimated = metricsAreEstimated,
            )
        }
        Log.i(
            LOG_TAG,
            "LiteRT IR generation complete backend=$runtimeBackend " +
                "elapsedMs=${System.currentTimeMillis() - startedAt} " +
                "outputTokens=${generation.outputTokens} " +
                "decodeTokensPerSecond=${generation.outputTokensPerSecond}",
        )
        return generation
    }

    companion object {
        /** Releases the process-wide cached engine after any active generation finishes. */
        @JvmStatic
        fun releaseCachedEngine() {
            LiteRtSharedEngineCache.withGenerationLock {
                LiteRtSharedEngineCache.release()
            }
        }
    }
}

/** Pure LiteRT policy shared by Android hosts and covered without loading a native engine. */
object LiteRtModelPolicy {
    internal data class PreparedPrompt(
        val system: String?,
        val user: String,
        val initialMessages: List<LiteRtConversationMessage>,
    ) {
        val combinedForEstimates: String
            get() = buildList {
                if (!system.isNullOrBlank()) add(system)
                initialMessages.forEach { message ->
                    if (message.text.isNotBlank()) add(message.text)
                }
                add(user)
            }.joinToString("\n\n")
    }

    internal fun preparePrompt(
        request: LiteRtGenerationRequest,
        useRawTrainingWrapper: Boolean,
    ): PreparedPrompt {
        val system = request.systemPrompt?.trim().orEmpty()
        val user = request.prompt.trim()
        if (useRawTrainingWrapper) {
            val combined = if (request.sendSystemPrompt && system.isNotBlank()) {
                "$system\n\n$user"
            } else {
                user
            }
            return PreparedPrompt(
                system = null,
                user = "<|im_start|>user\n$combined\n<|im_start|>assistant\n",
                initialMessages = emptyList(),
            )
        }
        return PreparedPrompt(
            system = system.takeIf { request.sendSystemPrompt && it.isNotBlank() },
            user = user,
            initialMessages = request.initialMessages,
        )
    }

    @JvmStatic
    fun mtpEnabledForBackend(backendName: String, requested: Boolean): Boolean =
        requested && backendName == BACKEND_GPU

    @JvmStatic
    fun runtimeBackendLabel(
        backendName: String,
        speculativeDecodingEnabled: Boolean,
    ): String = if (speculativeDecodingEnabled) "$backendName+MTP" else backendName

    @JvmStatic
    fun backendOrder(
        forceCpu: Boolean,
        requireGpu: Boolean,
        accelerator: LiteRtAccelerator = LiteRtAccelerator.AUTO,
    ): List<String> {
        if (forceCpu || accelerator == LiteRtAccelerator.CPU) return listOf(BACKEND_CPU)
        return when (accelerator) {
            LiteRtAccelerator.GPU -> listOf(BACKEND_GPU)
            LiteRtAccelerator.NPU -> listOf(BACKEND_NPU)
            LiteRtAccelerator.AUTO -> if (requireGpu) {
                listOf(BACKEND_GPU)
            } else {
                listOf(BACKEND_GPU, BACKEND_CPU)
            }
            LiteRtAccelerator.CPU -> listOf(BACKEND_CPU)
        }
    }

    @JvmStatic
    fun mergeStreamText(current: String, incoming: String): String = when {
        incoming.isEmpty() -> current
        current.isEmpty() -> incoming
        incoming == current -> current
        incoming.startsWith(current) -> incoming
        else -> current + incoming
    }

    @JvmStatic
    fun isTransientError(error: String): Boolean {
        val lower = error.lowercase(Locale.US)
        return lower.contains("timeout") ||
            lower.contains("interrupted") ||
            lower.contains("busy") ||
            lower.contains("memory") ||
            lower.contains("oom")
    }

    @JvmStatic
    fun checkModelHealth(config: LiteRtModelConfig): LiteRtModelHealth {
        val normalized = config.modelPath.trim()
        if (normalized.isBlank()) {
            return LiteRtModelHealth(
                healthy = false,
                errorMessage =
                    "On-device IR model path is empty. Download and select a Gemma LiteRT model first.",
            )
        }
        val file = File(normalized)
        if (!file.exists()) {
            return LiteRtModelHealth(
                healthy = false,
                errorMessage = "On-device IR model path does not exist: $normalized",
            )
        }
        if (!file.isFile) {
            return LiteRtModelHealth(
                healthy = false,
                errorMessage = "On-device IR model path must be a .litertlm file: $normalized",
            )
        }
        if (!normalized.lowercase(Locale.US).endsWith(".litertlm")) {
            return LiteRtModelHealth(
                healthy = false,
                errorMessage = "On-device IR model path must use the .litertlm format: $normalized",
            )
        }
        if (file.length() <= 0L) {
            return LiteRtModelHealth(
                healthy = false,
                errorMessage = "On-device IR model file is empty: $normalized",
            )
        }
        if (file.length() < config.minimumFileSizeBytes) {
            val modelName = config.modelDisplayName?.takeIf { it.isNotBlank() } ?: file.name
            return LiteRtModelHealth(
                healthy = false,
                errorMessage = "On-device model is incomplete: ${file.length()} bytes; expected at least " +
                    "${config.minimumFileSizeBytes} bytes for $modelName.",
            )
        }
        return LiteRtModelHealth(healthy = true)
    }

    internal fun estimateTokens(text: String): Int {
        if (text.isBlank()) return 0
        return max(1, (text.length / 4.0).roundToInt())
    }

    internal fun maxContextTokens(
        prompt: String,
        requestedMaxOutputTokens: Int,
        modelMaxContextTokens: Int,
        modelMaxOutputTokens: Int,
    ): Int {
        val contextLimit = modelMaxContextTokens.coerceAtLeast(MIN_MODEL_CONTEXT_TOKENS)
        val outputLimit = min(requestedMaxOutputTokens, modelMaxOutputTokens)
        val estimatedTotal = estimateTokens(prompt) + outputLimit + CHAT_TEMPLATE_RESERVE_TOKENS
        val minimum = min(DEFAULT_MIN_CONTEXT_TOKENS, contextLimit)
        return estimatedTotal.coerceIn(minimum, contextLimit)
    }

    internal fun isInvalidGpuOutput(text: String): Boolean {
        val normalized = text.trim()
        if (normalized.isBlank()) return true
        if (normalized.contains("<pad>", ignoreCase = true)) return true
        return Regex("^(?:<(?:unused\\d+|bos|eos)>\\s*)+$", RegexOption.IGNORE_CASE)
            .matches(normalized)
    }
}

private data class EngineHolder(
    val engine: Engine,
    val backendName: String,
    val maxContextTokens: Int,
    val speculativeDecodingEnabled: Boolean,
)

private data class BenchmarkSnapshot(
    val inputTokens: Int,
    val outputTokens: Int,
    val outputTokensPerSecond: Double,
)

private object LiteRtSharedEngineCache {
    private val engineLock = Any()
    private val generationLock = Any()
    private var cachedPath: String? = null
    private var cachedMaxContextTokens: Int? = null
    private var cachedBackendName: String? = null
    private var cachedSpeculativeDecoding: Boolean? = null
    private var cachedAccelerator: LiteRtAccelerator? = null
    private var cachedEngine: EngineHolder? = null
    private var gpuSamplerLoadAttempted = false

    fun <T> withGenerationLock(block: () -> T): T = synchronized(generationLock, block)

    @OptIn(ExperimentalApi::class)
    fun getOrCreateEngine(
        modelFile: File,
        maxContextTokens: Int,
        forceCpu: Boolean,
        requireGpu: Boolean,
        enableSpeculativeDecoding: Boolean,
        accelerator: LiteRtAccelerator,
        npuNativeLibraryDir: String,
    ): EngineHolder {
        val canonicalPath = modelFile.canonicalPath
        synchronized(engineLock) {
            cachedEngine?.let { existing ->
                val cacheCanServeRequest = cachedPath == canonicalPath &&
                    (cachedMaxContextTokens ?: 0) >= maxContextTokens &&
                    (!forceCpu || cachedBackendName == BACKEND_CPU) &&
                    (!requireGpu || cachedBackendName == BACKEND_GPU) &&
                    cachedSpeculativeDecoding == enableSpeculativeDecoding &&
                    cachedAccelerator == accelerator
                if (cacheCanServeRequest) return existing
                releaseLocked()
            }
            val cacheDir = cacheDirFor(canonicalPath, modelFile)
            val candidates = LiteRtModelPolicy.backendOrder(
                forceCpu = forceCpu,
                requireGpu = requireGpu,
                accelerator = accelerator,
            ).map { backendName ->
                when (backendName) {
                    BACKEND_GPU -> backendName to Backend.GPU()
                    BACKEND_NPU -> backendName to Backend.NPU(nativeLibraryDir = npuNativeLibraryDir)
                    else -> backendName to Backend.CPU(
                        Runtime.getRuntime().availableProcessors().coerceIn(2, 6),
                    )
                }
            }
            var lastError: Throwable? = null
            for ((backendName, backend) in candidates) {
                var initializingEngine: Engine? = null
                try {
                    if (backendName == BACKEND_GPU) ensureGpuSamplerDependenciesLoaded()
                    val useMtp = LiteRtModelPolicy.mtpEnabledForBackend(
                        backendName,
                        enableSpeculativeDecoding,
                    )
                    if (useMtp) {
                        check(Capabilities(canonicalPath).use { it.hasSpeculativeDecodingSupport() }) {
                            "MTP was requested but this model has no drafter. Turn MTP off in Settings."
                        }
                    }
                    val holder = LiteRtRuntimeFlags.withEngineFlags(
                        speculativeDecoding = useMtp,
                    ) {
                        Log.i(
                            LOG_TAG,
                            "Initializing LiteRT engine backend=$backendName context=$maxContextTokens " +
                                "mtp=$useMtp mtpRequested=$enableSpeculativeDecoding model=$canonicalPath",
                        )
                        val engine = Engine(
                            EngineConfig(
                                modelPath = canonicalPath,
                                backend = backend,
                                maxNumTokens = maxContextTokens,
                                cacheDir = cacheDir,
                            ),
                        )
                        initializingEngine = engine
                        engine.initialize()
                        EngineHolder(
                            engine = engine,
                            backendName = backendName,
                            maxContextTokens = maxContextTokens,
                            speculativeDecodingEnabled = useMtp,
                        )
                    }
                    cachedEngine = holder
                    cachedPath = canonicalPath
                    cachedMaxContextTokens = maxContextTokens
                    cachedBackendName = backendName
                    cachedSpeculativeDecoding = enableSpeculativeDecoding
                    cachedAccelerator = accelerator
                    return holder
                } catch (failure: Throwable) {
                    runCatching { initializingEngine?.close() }
                    if (enableSpeculativeDecoding && backendName == BACKEND_GPU) throw failure
                    lastError = failure
                    Log.w(LOG_TAG, "LiteRT engine init failed backend=$backendName: ${failure.message}")
                }
            }
            throw lastError ?: IllegalStateException("LiteRT engine initialization failed.")
        }
    }

    fun release() {
        synchronized(engineLock) {
            releaseLocked()
        }
    }

    private fun releaseLocked() {
        cachedEngine?.engine?.close()
        cachedEngine = null
        cachedPath = null
        cachedMaxContextTokens = null
        cachedBackendName = null
        cachedSpeculativeDecoding = null
        cachedAccelerator = null
    }

    /**
     * LiteRT-LM discovers its OpenCL sampler through dlopen. Load the runtime first so the
     * runtime-builtin table is visible in this linker namespace before engine creation.
     */
    private fun ensureGpuSamplerDependenciesLoaded() {
        if (gpuSamplerLoadAttempted) return
        gpuSamplerLoadAttempted = true
        val loaded = mutableListOf<String>()
        for (library in listOf("c++_shared", "LiteRt", "LiteRtTopKOpenClSampler")) {
            try {
                System.loadLibrary(library)
                loaded += library
            } catch (error: UnsatisfiedLinkError) {
                Log.w(
                    LOG_TAG,
                    "Optional LiteRT GPU sampler library failed to load name=$library " +
                        "error=${error.message}",
                )
            }
        }
        Log.i(LOG_TAG, "Optional LiteRT GPU sampler load attempted loaded=${loaded.joinToString(",")}")
    }

    private fun cacheDirFor(canonicalPath: String, modelFile: File): String? {
        if (!canonicalPath.startsWith("/data/local/tmp")) return null
        return File(modelFile.parentFile, ".litert_cache").apply { mkdirs() }.absolutePath
    }
}

/** LiteRT exposes these Java getters but still hides them from Kotlin metadata. */
private fun readBenchmarkSnapshot(conversation: Any): BenchmarkSnapshot? = runCatching {
    val benchmark = conversation.javaClass.getMethod("getBenchmarkInfo").invoke(conversation)
    val benchmarkClass = benchmark.javaClass
    BenchmarkSnapshot(
        inputTokens =
            (benchmarkClass.getMethod("getLastPrefillTokenCount").invoke(benchmark) as Number).toInt(),
        outputTokens =
            (benchmarkClass.getMethod("getLastDecodeTokenCount").invoke(benchmark) as Number).toInt(),
        outputTokensPerSecond =
            (benchmarkClass.getMethod("getLastDecodeTokensPerSecond").invoke(benchmark) as Number)
                .toDouble(),
    )
}.getOrNull()

private fun Message.textContent(): String = contents.contents.joinToString(separator = "") { content ->
    when (content) {
        is Content.Text -> content.text
        else -> ""
    }
}

private const val LOG_TAG = "LiteRtModelRunner"
private const val BACKEND_GPU = "GPU"
private const val BACKEND_CPU = "CPU"
private const val BACKEND_NPU = "NPU"
private const val MIN_MODEL_CONTEXT_TOKENS = 1_024
private const val DEFAULT_MIN_CONTEXT_TOKENS = 8_192
private const val CHAT_TEMPLATE_RESERVE_TOKENS = 256
