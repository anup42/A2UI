package com.samsung.genuicraft.inference

import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import java.io.File
import java.util.Locale
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
            val promptParts = buildPromptParts(request)
            val maxContextTokens = maxContextTokensFor(promptParts.combinedForEstimates, request.maxOutputTokens)
            val modelFile = File(modelPath.trim())
            val holder = getOrCreateEngine(modelFile, maxContextTokens, forceCpu = false)
            val text = try {
                generateWithEngine(holder, promptParts, request.temperature)
            } catch (gpuFailure: Throwable) {
                if (holder.backendName != BACKEND_GPU) {
                    throw gpuFailure
                }
                Log.w(LOG_TAG, "LiteRT GPU generation failed; retrying on CPU: ${gpuFailure.message}")
                closeCachedEngine()
                val cpuHolder = getOrCreateEngine(modelFile, maxContextTokens, forceCpu = true)
                generateWithEngine(cpuHolder, promptParts, request.temperature)
            }
            InferenceBackend.GenerateResponse(
                text = text,
                rawResponse = text,
                error = null,
                streamDurationMs = System.currentTimeMillis() - startMs,
                inputTokens = estimateTokens(promptParts.combinedForEstimates),
                outputTokens = estimateTokens(text)
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

    private fun buildPromptParts(request: InferenceBackend.GenerateRequest): PromptParts {
        val system = request.systemPrompt?.trim().orEmpty()
        val user = request.prompt.trim()
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
    ): String {
        val startedAt = System.currentTimeMillis()
        Log.i(
            LOG_TAG,
            "LiteRT IR generation start backend=${holder.backendName} context=${holder.maxContextTokens} " +
                "inputTokens≈${estimateTokens(promptParts.combinedForEstimates)}"
        )
        val conversationConfig = ConversationConfig(
            systemInstruction = promptParts.system?.let { Contents.of(it) },
            samplerConfig = SamplerConfig(
                temperature = temperature,
                topK = 32,
                topP = 0.9
            )
        )
        val response = holder.engine.createConversation(conversationConfig).use { conversation ->
            conversation.sendMessage(promptParts.user)
        }
        val text = response.contents.contents.joinToString(separator = "") { content ->
            when (content) {
                is Content.Text -> content.text
                else -> content.toString()
            }
        }.trim()
        Log.i(
            LOG_TAG,
            "LiteRT IR generation complete backend=${holder.backendName} elapsedMs=${System.currentTimeMillis() - startedAt} " +
                "outputTokens≈${estimateTokens(text)}"
        )
        return text
    }

    private fun estimateTokens(text: String): Int {
        if (text.isBlank()) {
            return 0
        }
        return max(1, (text.length / 4.0).roundToInt())
    }

    private fun maxContextTokensFor(prompt: String, maxOutputTokens: Int): Int {
        val estimatedTotal = estimateTokens(prompt) + min(maxOutputTokens, ON_DEVICE_MAX_OUTPUT_TOKENS) + 256
        return estimatedTotal.coerceIn(ON_DEVICE_MIN_CONTEXT_TOKENS, ON_DEVICE_MAX_CONTEXT_TOKENS)
    }

    private companion object {
        private const val LOG_TAG = "OnDeviceLitertBackend"
        private const val BACKEND_GPU = "GPU"
        private const val BACKEND_CPU = "CPU"
        private const val ON_DEVICE_MIN_CONTEXT_TOKENS = 8192
        private const val ON_DEVICE_MAX_CONTEXT_TOKENS = 12288
        private const val ON_DEVICE_MAX_OUTPUT_TOKENS = 4096
        private val engineLock = Any()
        private var cachedPath: String? = null
        private var cachedMaxContextTokens: Int? = null
        private var cachedBackendName: String? = null
        private var cachedEngine: EngineHolder? = null

        private data class EngineHolder(
            val engine: Engine,
            val backendName: String,
            val maxContextTokens: Int,
        )

        private fun getOrCreateEngine(
            modelFile: File,
            maxContextTokens: Int,
            forceCpu: Boolean,
        ): EngineHolder {
            val canonicalPath = modelFile.canonicalPath
            synchronized(engineLock) {
                cachedEngine?.let { existing ->
                    val cacheCanServeRequest = cachedPath == canonicalPath &&
                        (cachedMaxContextTokens ?: 0) >= maxContextTokens &&
                        (!forceCpu || cachedBackendName == BACKEND_CPU)
                    if (cacheCanServeRequest) {
                        return existing
                    }
                    closeCachedEngineLocked()
                }
                val cacheDir = File(modelFile.parentFile, ".litert_cache").apply {
                    mkdirs()
                }

                val backendCandidates = if (forceCpu) {
                    listOf(BACKEND_CPU to cpuBackend())
                } else {
                    listOf(BACKEND_GPU to Backend.GPU(), BACKEND_CPU to cpuBackend())
                }
                var lastError: Throwable? = null
                for ((backendName, backend) in backendCandidates) {
                    try {
                        Log.i(
                            LOG_TAG,
                            "Initializing LiteRT engine backend=$backendName context=$maxContextTokens model=$canonicalPath"
                        )
                        val engine = Engine(
                            EngineConfig(
                                modelPath = canonicalPath,
                                backend = backend,
                                maxNumTokens = maxContextTokens,
                                cacheDir = cacheDir.absolutePath
                            )
                        )
                        engine.initialize()
                        val holder = EngineHolder(
                            engine = engine,
                            backendName = backendName,
                            maxContextTokens = maxContextTokens,
                        )
                        cachedEngine = holder
                        cachedPath = canonicalPath
                        cachedMaxContextTokens = maxContextTokens
                        cachedBackendName = backendName
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
        }

        private fun cpuBackend(): Backend.CPU {
            val threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 6)
            return Backend.CPU(threads)
        }
    }
}
