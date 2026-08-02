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
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import java.io.File
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Stage-3 on-device backend for exported LiteRT-LM/Gemma IR models.
 */
class OnDeviceLitertBackend(
    private val modelPath: String
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
            val enableSpeculativeDecoding = runtimeProfile?.enableSpeculativeDecoding == true
            val holder = getOrCreateEngine(
                modelFile = modelFile,
                maxContextTokens = maxContextTokens,
                forceCpu = false,
                requireGpu = requireGpu,
                enableSpeculativeDecoding = enableSpeculativeDecoding,
            )
            val generation = try {
                generateWithEngine(holder, promptParts, request.temperature, request.onStreamUpdate)
            } catch (gpuFailure: Throwable) {
                if (holder.backendName != BACKEND_GPU || requireGpu) {
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
                )
                generateWithEngine(cpuHolder, promptParts, request.temperature, request.onStreamUpdate)
            }
            InferenceBackend.GenerateResponse(
                text = generation.text,
                rawResponse = generation.text,
                error = null,
                streamDurationMs = System.currentTimeMillis() - startMs,
                inputTokens = generation.inputTokens ?: estimateTokens(promptParts.combinedForEstimates),
                outputTokens = generation.outputTokens,
                outputTokensPerSecond = generation.outputTokensPerSecond,
                runtimeBackend = generation.backendName
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
        if (file.length() <= 0L) {
            return InferenceBackend.HealthCheckResult(
                healthy = false,
                errorMessage = "On-device IR model file is empty: $normalized"
            )
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
    ) {
        val combinedForEstimates: String
            get() = if (system.isNullOrBlank()) user else "$system\n\n$user"
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
            )
        }
        return if (request.localSendSystemPrompt && system.isNotBlank()) {
            PromptParts(system = system, user = user)
        } else {
            PromptParts(system = null, user = user)
        }
    }

    private fun generateWithEngine(
        holder: EngineHolder,
        promptParts: PromptParts,
        temperature: Double,
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
            samplerConfig = SamplerConfig(
                temperature = temperature,
                topK = if (deterministic) 1 else 32,
                topP = if (deterministic) 1.0 else 0.9,
            )
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
            val done = CountDownLatch(1)
            val failure = AtomicReference<Throwable?>(null)
            val textLock = Any()
            var streamedText = ""
            var firstTokenAtMs: Long? = null
            var lastUiUpdateAtMs = 0L

            conversation.sendMessageAsync(
                promptParts.user,
                object : MessageCallback {
                    override fun onMessage(message: Message) {
                        val chunk = messageText(message)
                        if (chunk.isEmpty()) {
                            return
                        }
                        val nowMs = System.currentTimeMillis()
                        val snapshot = synchronized(textLock) {
                            streamedText = mergeLiteRtStreamText(streamedText, chunk)
                            streamedText
                        }
                        if (firstTokenAtMs == null) {
                            firstTokenAtMs = nowMs
                        }
                        if (lastUiUpdateAtMs == 0L || nowMs - lastUiUpdateAtMs >= STREAM_UI_INTERVAL_MS) {
                            lastUiUpdateAtMs = nowMs
                            val tokenCount = estimateTokens(snapshot)
                            onStreamUpdate?.invoke(
                                InferenceBackend.StreamUpdate(
                                    text = snapshot,
                                    outputTokens = tokenCount,
                                    outputTokensPerSecond = estimatedDecodeRate(
                                        outputTokens = tokenCount,
                                        firstTokenAtMs = firstTokenAtMs,
                                        nowMs = nowMs,
                                    ),
                                    runtimeBackend = runtimeBackendLabel,
                                    metricsAreEstimated = true,
                                    complete = false,
                                )
                            )
                        }
                    }

                    override fun onDone() {
                        done.countDown()
                    }

                    override fun onError(throwable: Throwable) {
                        failure.set(throwable)
                        done.countDown()
                    }
                }
            )
            done.await()
            failure.get()?.let { throw it }

            val text = synchronized(textLock) { streamedText }.trim()
            val benchmark = readBenchmarkSnapshot(conversation)
            val benchmarkOutputTokens = benchmark?.outputTokens?.takeIf { it > 0 }
            val outputTokens = benchmarkOutputTokens ?: estimateTokens(text)
            val benchmarkRate = benchmark?.outputTokensPerSecond
                ?.takeIf { it.isFinite() && it > 0.0 }
            val outputTokensPerSecond = benchmarkRate ?: run {
                val elapsedMs = (System.currentTimeMillis() - (firstTokenAtMs ?: startedAt)).coerceAtLeast(1L)
                outputTokens * 1000.0 / elapsedMs
            }
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

    private fun estimatedDecodeRate(
        outputTokens: Int,
        firstTokenAtMs: Long?,
        nowMs: Long,
    ): Double? {
        val firstMs = firstTokenAtMs ?: return null
        val elapsedMs = nowMs - firstMs
        if (outputTokens <= 0 || elapsedMs < MIN_RATE_SAMPLE_MS) {
            return null
        }
        return outputTokens * 1000.0 / elapsedMs
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
        private const val STREAM_UI_INTERVAL_MS = 80L
        private const val MIN_RATE_SAMPLE_MS = 100L
        private const val ON_DEVICE_MIN_CONTEXT_TOKENS = 8192
        private const val ON_DEVICE_MAX_CONTEXT_TOKENS = 12288
        private const val ON_DEVICE_MAX_OUTPUT_TOKENS = 4096
        private val engineLock = Any()
        private var cachedPath: String? = null
        private var cachedMaxContextTokens: Int? = null
        private var cachedBackendName: String? = null
        private var cachedSpeculativeDecoding: Boolean? = null
        private var cachedEngine: EngineHolder? = null

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
        ): EngineHolder {
            val canonicalPath = modelFile.canonicalPath
            synchronized(engineLock) {
                cachedEngine?.let { existing ->
                    val cacheCanServeRequest = cachedPath == canonicalPath &&
                        (cachedMaxContextTokens ?: 0) >= maxContextTokens &&
                        (!forceCpu || cachedBackendName == BACKEND_CPU) &&
                        (!requireGpu || cachedBackendName == BACKEND_GPU) &&
                        cachedSpeculativeDecoding == enableSpeculativeDecoding
                    if (cacheCanServeRequest) {
                        return existing
                    }
                    closeCachedEngineLocked()
                }
                val cacheDir = cacheDirFor(canonicalPath, modelFile)
                ExperimentalFlags.enableSpeculativeDecoding = enableSpeculativeDecoding

                val backendCandidates = liteRtBackendOrder(forceCpu, requireGpu).map { backendName ->
                    when (backendName) {
                        BACKEND_GPU -> backendName to Backend.GPU()
                        else -> backendName to cpuBackend()
                    }
                }
                var lastError: Throwable? = null
                for ((backendName, backend) in backendCandidates) {
                    try {
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
        }

        private fun cpuBackend(): Backend.CPU {
            val threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 6)
            return Backend.CPU(threads)
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

internal fun liteRtBackendOrder(forceCpu: Boolean, requireGpu: Boolean): List<String> {
    return when {
        forceCpu -> listOf("CPU")
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
