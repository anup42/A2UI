package com.samsung.genuicraft.inference

import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import java.io.File
import java.util.Locale
import kotlin.math.max
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
            val prompt = buildPrompt(request)
            val maxContextTokens = maxContextTokensFor(prompt, request.maxOutputTokens)
            val engine = getOrCreateEngine(File(modelPath.trim()), maxContextTokens)
            val conversationConfig = ConversationConfig(
                systemInstruction = null,
                samplerConfig = SamplerConfig(
                    temperature = request.temperature,
                    topK = 40,
                    topP = 0.95
                )
            )
            val response = engine.createConversation(conversationConfig).use { conversation ->
                conversation.sendMessage(prompt)
            }
            val text = response.contents.contents.joinToString(separator = "") { content ->
                when (content) {
                    is Content.Text -> content.text
                    else -> content.toString()
                }
            }.trim()
            InferenceBackend.GenerateResponse(
                text = text,
                rawResponse = text,
                error = null,
                streamDurationMs = System.currentTimeMillis() - startMs,
                inputTokens = estimateTokens(prompt),
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

    private fun buildPrompt(request: InferenceBackend.GenerateRequest): String {
        val system = request.systemPrompt?.trim().orEmpty()
        val user = request.prompt.trim()
        return if (request.localSendSystemPrompt && system.isNotBlank()) {
            "$system\n\n$user"
        } else {
            user
        }
    }

    private fun estimateTokens(text: String): Int {
        if (text.isBlank()) {
            return 0
        }
        return max(1, (text.length / 4.0).roundToInt())
    }

    private fun maxContextTokensFor(prompt: String, maxOutputTokens: Int): Int {
        val estimatedTotal = estimateTokens(prompt) + maxOutputTokens + 256
        return estimatedTotal.coerceIn(8192, 16384)
    }

    private companion object {
        private val engineLock = Any()
        private var cachedPath: String? = null
        private var cachedMaxContextTokens: Int? = null
        private var cachedEngine: Engine? = null

        private fun getOrCreateEngine(modelFile: File, maxContextTokens: Int): Engine {
            val canonicalPath = modelFile.canonicalPath
            synchronized(engineLock) {
                cachedEngine?.let { existing ->
                    if (cachedPath == canonicalPath && cachedMaxContextTokens == maxContextTokens) {
                        return existing
                    }
                    existing.close()
                    cachedEngine = null
                    cachedPath = null
                    cachedMaxContextTokens = null
                }
                val cacheDir = File(modelFile.parentFile, ".litert_cache").apply {
                    mkdirs()
                }
                val engine = Engine(
                    EngineConfig(
                        modelPath = canonicalPath,
                        backend = Backend.CPU(),
                        maxNumTokens = maxContextTokens,
                        cacheDir = cacheDir.absolutePath
                    )
                )
                engine.initialize()
                cachedEngine = engine
                cachedPath = canonicalPath
                cachedMaxContextTokens = maxContextTokens
                return engine
            }
        }
    }
}
