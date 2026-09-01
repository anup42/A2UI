package com.samsung.genuicraft.inference

import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig
import com.samsung.genuicraft.InferenceBackendSettings
import java.io.File
import java.util.Locale
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Stage-3 on-device backend for exported LiteRT-LM/Gemma IR models.
 */
class OnDeviceLitertBackend(
    private val modelPath: String,
    private val acceleratorPreference: InferenceBackendSettings.Accelerator =
        InferenceBackendSettings.DEFAULT_ON_DEVICE_ACCELERATOR,
    private val npuNativeLibraryDir: String = "",
) : InferenceBackend {

    override fun generate(request: InferenceBackend.GenerateRequest): InferenceBackend.GenerateResponse {
        val startMs = System.currentTimeMillis()
        val health = checkHealth()
        if (!health.healthy) {
            return InferenceBackend.GenerateResponse(
                text = "",
                rawResponse = null,
                error = health.errorMessage ?: "On-device model is not ready.",
                streamDurationMs = null
            )
        }

        return try {
            val runtimeProfile = OnDeviceModelCatalog.entryForModelPath(modelPath)
            val promptParts = buildPromptParts(
                request,
                useRawTrainingWrapper = runtimeProfile?.useRawTrainingWrapper == true,
            )
            val modelFile = File(modelPath.trim())
            val maxContextTokens = maxContextTokensFor(
                prompt = promptParts.combinedForEstimates,
                requestedMaxOutputTokens = request.maxOutputTokens,
                modelMaxContextTokens = runtimeProfile?.maxContextTokens ?: ON_DEVICE_MAX_CONTEXT_TOKENS,
                modelMaxOutputTokens = runtimeProfile?.maxOutputTokens ?: ON_DEVICE_MAX_OUTPUT_TOKENS,
            )
            val requireGpu = runtimeProfile?.requireGpu == true
            val allowGpuQualityFallback = runtimeProfile?.allowGpuQualityFallback == true
            val requireGpuForInitialization = requireGpu && !(
                allowGpuQualityFallback &&
                    acceleratorPreference == InferenceBackendSettings.Accelerator.AUTO
                )
            val enableSpeculativeDecoding = runtimeProfile?.enableSpeculativeDecoding == true
            val holder = getOrCreateEngine(
                modelFile = modelFile,
                maxContextTokens = maxContextTokens,
                forceCpu = false,
                requireGpu = requireGpuForInitialization,
                enableSpeculativeDecoding = enableSpeculativeDecoding,
                acceleratorPreference = acceleratorPreference,
                npuNativeLibraryDir = npuNativeLibraryDir,
            )
            val generation = try {
                generateWithEngine(
                    holder = holder,
                    promptParts = promptParts,
                    temperature = request.temperature,
                    maxOutputTokens = min(
                        request.maxOutputTokens,
                        runtimeProfile?.maxOutputTokens ?: ON_DEVICE_MAX_OUTPUT_TOKENS,
                    ),
                    onStreamUpdate = request.onStreamUpdate,
                )
            } catch (gpuFailure: Throwable) {
                if (
                    holder.backendName != BACKEND_GPU ||
                    acceleratorPreference != InferenceBackendSettings.Accelerator.AUTO ||
                    (!allowGpuQualityFallback && requireGpu)
                ) {
                    throw gpuFailure
                }
                Log.w(LOG_TAG, "LiteRT GPU generation failed; retrying on CPU: ${gpuFailure.message}")
                closeCachedEngine()
                val cpuHolder = getOrCreateEngine(
                    modelFile = modelFile,
                    maxContextTokens = maxContextTokens,
                    forceCpu = true,
                    requireGpu = false,
                    enableSpeculativeDecoding = enableSpeculativeDecoding,
                    acceleratorPreference = InferenceBackendSettings.Accelerator.CPU,
                    npuNativeLibraryDir = npuNativeLibraryDir,
                )
                generateWithEngine(
                    holder = cpuHolder,
                    promptParts = promptParts,
                    temperature = request.temperature,
                    maxOutputTokens = min(
                        request.maxOutputTokens,
                        runtimeProfile?.maxOutputTokens ?: ON_DEVICE_MAX_OUTPUT_TOKENS,
                    ),
                    onStreamUpdate = request.onStreamUpdate,
                )
            }
            val qualityCheckedGeneration = if (
                generation.backendName == BACKEND_GPU &&
                acceleratorPreference == InferenceBackendSettings.Accelerator.AUTO &&
                allowGpuQualityFallback &&
                isInvalidGpuOutput(generation.text)
            ) {
                Log.w(
                    LOG_TAG,
                    "LiteRT GPU returned invalid special-token output; retrying on CPU. " +
                        "gpuText=${generation.text.take(160)}"
                )
                closeCachedEngine()
                val cpuHolder = getOrCreateEngine(
                    modelFile = modelFile,
                    maxContextTokens = maxContextTokens,
                    forceCpu = true,
                    requireGpu = false,
                    enableSpeculativeDecoding = enableSpeculativeDecoding,
                    acceleratorPreference = InferenceBackendSettings.Accelerator.CPU,
                    npuNativeLibraryDir = npuNativeLibraryDir,
                )
                generateWithEngine(
                    holder = cpuHolder,
                    promptParts = promptParts,
                    temperature = request.temperature,
                    maxOutputTokens = min(
                        request.maxOutputTokens,
                        runtimeProfile?.maxOutputTokens ?: ON_DEVICE_MAX_OUTPUT_TOKENS,
                    ),
                    onStreamUpdate = request.onStreamUpdate,
                )
            } else {
                generation
            }
            InferenceBackend.GenerateResponse(
                text = qualityCheckedGeneration.text,
                rawResponse = qualityCheckedGeneration.text,
                error = null,
                streamDurationMs = System.currentTimeMillis() - startMs,
                inputTokens = qualityCheckedGeneration.inputTokens ?: estimateTokens(promptParts.combinedForEstimates),
                outputTokens = qualityCheckedGeneration.outputTokens,
                outputTokensPerSecond = qualityCheckedGeneration.outputTokensPerSecond,
                runtimeBackend = qualityCheckedGeneration.backendName
            )
        } catch (t: Throwable) {
            InferenceBackend.GenerateResponse(
                text = "",
                rawResponse = null,
                error = "On-device LiteRT generation failed: ${t.message ?: t.javaClass.simpleName}",
                streamDurationMs = System.currentTimeMillis() - startMs
            )
        }
    }

    override fun checkHealth(): InferenceBackend.HealthCheckResult {
        val normalized = modelPath.trim()
        if (normalized.isBlank()) {
            return InferenceBackend.HealthCheckResult(
                healthy = false,
                errorMessage = "On-device IR model path is empty. Download and select a Gemma LiteRT model first."
            )
        }
        val file = File(normalized)
        if (!file.exists()) {
            return InferenceBackend.HealthCheckResult(
                healthy = false,
                errorMessage = "On-device IR model path does not exist: $normalized"
            )
        }
        if (!file.isFile) {
            return InferenceBackend.HealthCheckResult(
                healthy = false,
                errorMessage = "On-device IR model path must be a .litertlm file: $normalized"
            )
        }
        if (!normalized.lowercase(Locale.US).endsWith(".litertlm")) {
            return InferenceBackend.HealthCheckResult(
                healthy = false,
                errorMessage = "On-device IR model path must use the .litertlm format: $normalized"
            )
        }
        if (file.length() <= 0L) {
            return InferenceBackend.HealthCheckResult(
                healthy = false,
                errorMessage = "On-device IR model file is empty: $normalized"
            )
        }
        OnDeviceModelCatalog.entryForModelPath(normalized)?.let { entry ->
            if (file.length() < entry.minimumFileSizeBytes) {
                return InferenceBackend.HealthCheckResult(
                    healthy = false,
                    errorMessage = "On-device model is incomplete: ${file.length()} bytes; expected at least " +
                        "${entry.minimumFileSizeBytes} bytes for ${entry.displayName}."
                )
            }
        }
        return InferenceBackend.HealthCheckResult(healthy = true)
    }

    override fun classifyError(error: String): InferenceBackend.ErrorClass {
        val lower = error.lowercase(Locale.US)
        return if (
            lower.contains("timeout") ||
            lower.contains("interrupted") ||
            lower.contains("busy") ||
            lower.contains("memory") ||
            lower.contains("oom")
        ) {
            InferenceBackend.ErrorClass.TRANSIENT
        } else {
            InferenceBackend.ErrorClass.UNKNOWN
        }
    }

    private data class PromptParts(
        val system: String?,
        val user: String,
        val initialMessages: List<InferenceBackend.ConversationMessage>,
    ) {
        val combinedForEstimates: String
            get() = buildList {
                if (!system.isNullOrBlank()) add(system)
                initialMessages.forEach { message ->
                    if (message.content.isNotBlank()) add(message.content)
                }
                add(user)
            }.joinToString("\n\n")
    }

    private fun buildPromptParts(
        request: InferenceBackend.GenerateRequest,
        useRawTrainingWrapper: Boolean = false,
    ): PromptParts {
        val system = request.systemPrompt?.trim().orEmpty()
        val user = request.prompt.trim()
        if (useRawTrainingWrapper) {
            val combined = if (request.localSendSystemPrompt && system.isNotBlank()) {
                "$system\n\n$user"
            } else {
                user
            }
            return PromptParts(
                system = null,
                user = "<|im_start|>user\n$combined\n<|im_start|>assistant\n",
                initialMessages = emptyList(),
            )
        }
        return if (request.localSendSystemPrompt && system.isNotBlank()) {
            PromptParts(
                system = system,
                user = user,
                initialMessages = request.initialMessages,
            )
        } else {
            PromptParts(
                system = null,
                user = user,
                initialMessages = request.initialMessages,
            )
        }
    }

    private fun generateWithEngine(
        holder: EngineHolder,
        promptParts: PromptParts,
        temperature: Double,
        maxOutputTokens: Int,
        onStreamUpdate: ((InferenceBackend.StreamUpdate) -> Unit)?,
    ): GenerationOutput {
        val startedAt = System.currentTimeMillis()
        Log.i(
            LOG_TAG,
            "LiteRT IR generation start backend=${holder.backendName} context=${holder.maxContextTokens} " +
                "mtp=${holder.speculativeDecodingEnabled} " +
                "inputTokensApprox=${estimateTokens(promptParts.combinedForEstimates)}"
        )
        val runtimeBackendLabel = liteRtRuntimeBackendLabel(
            backendName = holder.backendName,
            speculativeDecodingEnabled = holder.speculativeDecodingEnabled,
        )
        val deterministic = temperature <= 0.0
        val conversationConfig = ConversationConfig(
            systemInstruction = promptParts.system?.let { Contents.of(it) },
            initialMessages = promptParts.initialMessages.map { message ->
                when (message.role) {
                    InferenceBackend.ConversationRole.USER -> Message.user(message.content)
                    InferenceBackend.ConversationRole.MODEL -> Message.model(message.content)
                }
            },
            samplerConfig = SamplerConfig(
                temperature = temperature,
                topK = if (deterministic) 1 else 32,
                topP = if (deterministic) 1.0 else 0.9,
            ),
            maxOutputToken = maxOutputTokens.coerceAtLeast(1),
        )
        onStreamUpdate?.invoke(
            InferenceBackend.StreamUpdate(
                text = "",
                outputTokens = 0,
                outputTokensPerSecond = null,
                runtimeBackend = runtimeBackendLabel,
                metricsAreEstimated = true,
                complete = false,
            )
        )
        val generation = holder.engine.createConversation(conversationConfig).use { conversation ->
            // LiteRT-LM's Android callback path can complete without exposing
            // the final text for GPU executions.  Use the synchronous final
            // message API, which is also the API used by the official Android
            // GPU validation probe, and extract the returned Content.Text.
            val responseMessage = conversation.sendMessage(promptParts.user)
            val text = messageText(responseMessage).trim()
            Log.i(
                LOG_TAG,
                "LiteRT sync response role=${responseMessage.role} " +
                    "contentTypes=${responseMessage.contents.contents.map { it.javaClass.name }} " +
                    "textChars=${text.length}",
            )
            val benchmark = readBenchmarkSnapshot(conversation)
            val benchmarkOutputTokens = benchmark?.outputTokens?.takeIf { it > 0 }
            val outputTokens = benchmarkOutputTokens ?: estimateTokens(text)
            val benchmarkRate = benchmark?.outputTokensPerSecond
                ?.takeIf { it.isFinite() && it > 0.0 }
            val elapsedMs = (System.currentTimeMillis() - startedAt).coerceAtLeast(1L)
            val outputTokensPerSecond = benchmarkRate ?: outputTokens * 1000.0 / elapsedMs
            onStreamUpdate?.invoke(
                InferenceBackend.StreamUpdate(
                    text = text,
                    outputTokens = outputTokens,
                    outputTokensPerSecond = outputTokensPerSecond,
                    runtimeBackend = runtimeBackendLabel,
                    metricsAreEstimated = benchmarkOutputTokens == null || benchmarkRate == null,
                    complete = true,
                )
            )
            GenerationOutput(
                text = text,
                inputTokens = benchmark?.inputTokens?.takeIf { it > 0 },
                outputTokens = outputTokens,
                outputTokensPerSecond = outputTokensPerSecond,
                backendName = runtimeBackendLabel,
            )
        }
        Log.i(
            LOG_TAG,
            "LiteRT IR generation complete backend=$runtimeBackendLabel elapsedMs=${System.currentTimeMillis() - startedAt} " +
                "outputTokens=${generation.outputTokens} decodeTokensPerSecond=${generation.outputTokensPerSecond}"
        )
        return generation
    }

    private fun messageText(message: Message): String {
        return message.contents.contents.joinToString(separator = "") { content ->
            when (content) {
                is Content.Text -> content.text
                else -> ""
            }
        }
    }

    private data class GenerationOutput(
        val text: String,
        val inputTokens: Int?,
        val outputTokens: Int,
        val outputTokensPerSecond: Double,
        val backendName: String,
    )

    private data class BenchmarkSnapshot(
        val inputTokens: Int,
        val outputTokens: Int,
        val outputTokensPerSecond: Double,
    )

    /**
     * LiteRT 0.14 exposes these getters in bytecode but hides them from Kotlin source metadata.
     * Reflection keeps this backend compatible until the public Kotlin API exposes the values.
     */
    private fun readBenchmarkSnapshot(conversation: Any): BenchmarkSnapshot? {
        return runCatching {
            val benchmark = conversation.javaClass
                .getMethod("getBenchmarkInfo")
                .invoke(conversation)
            val benchmarkClass = benchmark.javaClass
            BenchmarkSnapshot(
                inputTokens = (benchmarkClass.getMethod("getLastPrefillTokenCount").invoke(benchmark) as Number).toInt(),
                outputTokens = (benchmarkClass.getMethod("getLastDecodeTokenCount").invoke(benchmark) as Number).toInt(),
                outputTokensPerSecond =
                    (benchmarkClass.getMethod("getLastDecodeTokensPerSecond").invoke(benchmark) as Number).toDouble(),
            )
        }.getOrNull()
    }

    private fun estimateTokens(text: String): Int {
        if (text.isBlank()) {
            return 0
        }
        return max(1, (text.length / 4.0).roundToInt())
    }

    private fun isInvalidGpuOutput(text: String): Boolean {
        val normalized = text.trim()
        if (normalized.isBlank()) {
            return true
        }
        if (normalized.contains("<pad>", ignoreCase = true)) {
            return true
        }
        return Regex("^(?:<(?:unused\\d+|bos|eos)>\\s*)+$", RegexOption.IGNORE_CASE)
            .matches(normalized)
    }

    private fun maxContextTokensFor(
        prompt: String,
        requestedMaxOutputTokens: Int,
        modelMaxContextTokens: Int,
        modelMaxOutputTokens: Int,
    ): Int {
        val contextLimit = modelMaxContextTokens.coerceAtLeast(1_024)
        val outputLimit = min(requestedMaxOutputTokens, modelMaxOutputTokens)
        val estimatedTotal = estimateTokens(prompt) + outputLimit + 256
        val minimum = min(ON_DEVICE_MIN_CONTEXT_TOKENS, contextLimit)
        return estimatedTotal.coerceIn(minimum, contextLimit)
    }

    private companion object {
        private const val LOG_TAG = "OnDeviceLitertBackend"
        private const val BACKEND_GPU = "GPU"
        private const val BACKEND_CPU = "CPU"
        private const val BACKEND_NPU = "NPU"
        private const val ON_DEVICE_MIN_CONTEXT_TOKENS = 8192
        private const val ON_DEVICE_MAX_CONTEXT_TOKENS = 12288
        private const val ON_DEVICE_MAX_OUTPUT_TOKENS = 4096
        private val engineLock = Any()
        private var cachedPath: String? = null
        private var cachedMaxContextTokens: Int? = null
        private var cachedBackendName: String? = null
        private var cachedSpeculativeDecoding: Boolean? = null
        private var cachedAcceleratorPreference: InferenceBackendSettings.Accelerator? = null
        private var cachedEngine: EngineHolder? = null
        private var gpuSamplerLoadAttempted = false

        private data class EngineHolder(
            val engine: Engine,
            val backendName: String,
            val maxContextTokens: Int,
            val speculativeDecodingEnabled: Boolean,
        )

        @OptIn(ExperimentalApi::class)
        private fun getOrCreateEngine(
            modelFile: File,
            maxContextTokens: Int,
            forceCpu: Boolean,
            requireGpu: Boolean,
            enableSpeculativeDecoding: Boolean,
            acceleratorPreference: InferenceBackendSettings.Accelerator,
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
                        cachedAcceleratorPreference == acceleratorPreference
                    if (cacheCanServeRequest) {
                        return existing
                    }
                    closeCachedEngineLocked()
                }
                val cacheDir = cacheDirFor(canonicalPath, modelFile)
                ExperimentalFlags.enableSpeculativeDecoding = enableSpeculativeDecoding

                val backendCandidates = liteRtBackendOrder(
                    forceCpu = forceCpu,
                    requireGpu = requireGpu,
                    accelerator = acceleratorPreference,
                ).map { backendName ->
                    when (backendName) {
                        BACKEND_GPU -> backendName to Backend.GPU()
                        BACKEND_NPU -> backendName to Backend.NPU(
                            nativeLibraryDir = npuNativeLibraryDir,
                        )
                        else -> backendName to cpuBackend()
                    }
                }
                var lastError: Throwable? = null
                for ((backendName, backend) in backendCandidates) {
                    try {
                        if (backendName == BACKEND_GPU) {
                            ensureGpuSamplerDependenciesLoaded()
                        }
                        Log.i(
                            LOG_TAG,
                            "Initializing LiteRT engine backend=$backendName context=$maxContextTokens " +
                                "mtp=$enableSpeculativeDecoding model=$canonicalPath"
                        )
                        val engine = Engine(
                            EngineConfig(
                                modelPath = canonicalPath,
                                backend = backend,
                                maxNumTokens = maxContextTokens,
                                cacheDir = cacheDir
                            )
                        )
                        engine.initialize()
                        val holder = EngineHolder(
                            engine = engine,
                            backendName = backendName,
                            maxContextTokens = maxContextTokens,
                            speculativeDecodingEnabled = enableSpeculativeDecoding,
                        )
                        cachedEngine = holder
                        cachedPath = canonicalPath
                        cachedMaxContextTokens = maxContextTokens
                        cachedBackendName = backendName
                        cachedSpeculativeDecoding = enableSpeculativeDecoding
                        cachedAcceleratorPreference = acceleratorPreference
                        return holder
                    } catch (t: Throwable) {
                        lastError = t
                        Log.w(LOG_TAG, "LiteRT engine init failed backend=$backendName: ${t.message}")
                    }
                }
                throw lastError ?: IllegalStateException("LiteRT engine initialization failed.")
            }
        }

        private fun closeCachedEngine() {
            synchronized(engineLock) {
                closeCachedEngineLocked()
            }
        }

        private fun closeCachedEngineLocked() {
            cachedEngine?.engine?.close()
            cachedEngine = null
            cachedPath = null
            cachedMaxContextTokens = null
            cachedBackendName = null
            cachedSpeculativeDecoding = null
            cachedAcceleratorPreference = null
        }

        private fun cpuBackend(): Backend.CPU {
            val threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 6)
            return Backend.CPU(threads)
        }

        /**
         * LiteRT-LM discovers the OpenCL sampler with a native dlopen(). Load
         * the LiteRT runtime before the sampler so its exported runtime-builtin
         * table is visible in the app linker namespace. Otherwise LiteRT-LM
         * silently falls back to the statically linked sampler; on this device
         * that fallback returns token 0 for GPU logits.
         */
        private fun ensureGpuSamplerDependenciesLoaded() {
            if (gpuSamplerLoadAttempted) {
                return
            }
            gpuSamplerLoadAttempted = true
            val libraries = listOf(
                "c++_shared",
                "LiteRt",
                "LiteRtTopKOpenClSampler",
            )
            val loaded = mutableListOf<String>()
            for (library in libraries) {
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
            Log.i(
                LOG_TAG,
                "Optional LiteRT GPU sampler load attempted loaded=${loaded.joinToString(",")}",
            )
        }

        private fun cacheDirFor(canonicalPath: String, modelFile: File): String? {
            if (!canonicalPath.startsWith("/data/local/tmp")) {
                return null
            }
            return File(modelFile.parentFile, ".litert_cache").apply {
                mkdirs()
            }.absolutePath
        }
    }
}

internal fun liteRtRuntimeBackendLabel(
    backendName: String,
    speculativeDecodingEnabled: Boolean,
): String = if (speculativeDecodingEnabled) "$backendName+MTP" else backendName

internal fun liteRtBackendOrder(
    forceCpu: Boolean,
    requireGpu: Boolean,
    accelerator: InferenceBackendSettings.Accelerator = InferenceBackendSettings.Accelerator.AUTO,
): List<String> {
    if (forceCpu || accelerator == InferenceBackendSettings.Accelerator.CPU) {
        return listOf("CPU")
    }
    when (accelerator) {
        InferenceBackendSettings.Accelerator.GPU -> return listOf("GPU")
        InferenceBackendSettings.Accelerator.NPU -> return listOf("NPU")
        InferenceBackendSettings.Accelerator.AUTO -> Unit
        InferenceBackendSettings.Accelerator.CPU -> return listOf("CPU")
    }
    return when {
        requireGpu -> listOf("GPU")
        else -> listOf("GPU", "CPU")
    }
}

internal fun mergeLiteRtStreamText(current: String, incoming: String): String {
    return when {
        incoming.isEmpty() -> current
        current.isEmpty() -> incoming
        incoming == current -> current
        incoming.startsWith(current) -> incoming
        else -> current + incoming
    }
}
