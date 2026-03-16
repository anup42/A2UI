package com.samsung.genuicraft.inference

import com.google.gson.GsonBuilder
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.UnknownHostException
import java.nio.charset.StandardCharsets
import java.util.Locale
import kotlin.math.min

class LocalServerBackend(
    private val baseUrl: String,
    private val modelPath: String
) : InferenceBackend {

    private val normalizedBaseUrl: String = baseUrl.trim().trimEnd('/')

    override fun generate(request: InferenceBackend.GenerateRequest): InferenceBackend.GenerateResponse {
        if (normalizedBaseUrl.isBlank()) {
            return InferenceBackend.GenerateResponse(
                text = "",
                rawResponse = null,
                error = "Local server URL is empty.",
                streamDurationMs = null
            )
        }
        val endpoint = URL("$normalizedBaseUrl/v1/generate")
        val connection = (endpoint.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 7000
            readTimeout = 300000
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
        }

        val body = JsonObject().apply {
            addProperty("prompt", request.prompt)
            if (request.localSendSystemPrompt && !request.systemPrompt.isNullOrBlank()) {
                addProperty("system_prompt", request.systemPrompt)
            }
            if (!request.localSystemPromptCacheKey.isNullOrBlank()) {
                addProperty("system_prompt_cache_key", request.localSystemPromptCacheKey)
            }
            addProperty("temperature", request.temperature)
            addProperty("max_output_tokens", min(request.maxOutputTokens, 8192))
            addProperty("json_mode", request.jsonMode)
            if (modelPath.isNotBlank()) {
                addProperty("model_path", modelPath)
            }
        }

        return try {
            connection.outputStream.use { out ->
                out.write(gson.toJson(body).toByteArray(StandardCharsets.UTF_8))
            }

            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val streamRead = InferenceStreamUtils.readStreamWithTiming(stream)
            val raw = streamRead.text

            if (code !in 200..299) {
                val short = raw.trim().ifBlank { "HTTP $code" }
                return InferenceBackend.GenerateResponse(
                    text = "",
                    rawResponse = raw,
                    error = "HTTP $code: ${short.take(320)}",
                    streamDurationMs = streamRead.streamDurationMs
                )
            }

            val text = extractLocalServerText(raw)
            if (text.isNullOrBlank()) {
                return InferenceBackend.GenerateResponse(
                    text = "",
                    rawResponse = raw,
                    error = "Local server response did not include text output.",
                    streamDurationMs = streamRead.streamDurationMs
                )
            }

            InferenceBackend.GenerateResponse(
                text = text,
                rawResponse = raw,
                error = null,
                streamDurationMs = streamRead.streamDurationMs
            )
        } catch (unknownHost: UnknownHostException) {
            InferenceBackend.GenerateResponse(
                text = "",
                rawResponse = null,
                error = "Local server host is unreachable ($normalizedBaseUrl): ${unknownHost.message ?: "unknown host"}",
                streamDurationMs = null
            )
        } catch (io: IOException) {
            InferenceBackend.GenerateResponse(
                text = "",
                rawResponse = null,
                error = "Local server request failed at $normalizedBaseUrl: ${io.message ?: io.javaClass.simpleName}",
                streamDurationMs = null
            )
        } finally {
            connection.disconnect()
        }
    }

    override fun checkHealth(): InferenceBackend.HealthCheckResult {
        if (normalizedBaseUrl.isBlank()) {
            return InferenceBackend.HealthCheckResult(
                healthy = false,
                errorMessage = "Local server URL is empty."
            )
        }
        val endpoint = URL("$normalizedBaseUrl/health")
        val connection = (endpoint.openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 5000
            readTimeout = 7000
        }
        return try {
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val raw = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
            if (code !in 200..299) {
                val short = raw.trim().ifBlank { "HTTP $code" }
                InferenceBackend.HealthCheckResult(
                    healthy = false,
                    errorMessage = "Local server health check failed: HTTP $code: ${short.take(200)}"
                )
            } else {
                InferenceBackend.HealthCheckResult(healthy = true)
            }
        } catch (unknownHost: UnknownHostException) {
            InferenceBackend.HealthCheckResult(
                healthy = false,
                errorMessage = "Local server host is unreachable: ${unknownHost.message ?: "unknown host"}"
            )
        } catch (io: IOException) {
            InferenceBackend.HealthCheckResult(
                healthy = false,
                errorMessage = "Could not connect to local server at $normalizedBaseUrl (${io.message ?: io.javaClass.simpleName})"
            )
        } finally {
            connection.disconnect()
        }
    }

    override fun classifyError(error: String): InferenceBackend.ErrorClass {
        val lower = error.lowercase(Locale.US)
        if (isLocalSystemPromptCacheMiss(lower)) return InferenceBackend.ErrorClass.LOCAL_CACHE_MISS
        if (isTransientError(lower)) return InferenceBackend.ErrorClass.TRANSIENT
        return InferenceBackend.ErrorClass.UNKNOWN
    }

    // ── response parsing ───────────────────────────────────────────────

    private fun extractLocalServerText(raw: String): String? {
        val root = runCatching { JsonParser.parseString(raw).asJsonObject }.getOrNull() ?: return null
        val directText = root.get("text")
            ?.takeIf { it.isJsonPrimitive }
            ?.asString
            ?.trim()
        if (!directText.isNullOrBlank()) {
            return directText
        }

        val outputText = root.get("output_text")
            ?.takeIf { it.isJsonPrimitive }
            ?.asString
            ?.trim()
        if (!outputText.isNullOrBlank()) {
            return outputText
        }

        val choices = root.getAsJsonArray("choices")
        if (choices != null && choices.size() > 0) {
            val first = runCatching { choices[0].asJsonObject }.getOrNull()
            val message = first?.getAsJsonObject("message")
            val content = message
                ?.get("content")
                ?.takeIf { it.isJsonPrimitive }
                ?.asString
                ?.trim()
            if (!content.isNullOrBlank()) {
                return content
            }
        }
        return null
    }

    // ── error classification ───────────────────────────────────────────

    private fun isLocalSystemPromptCacheMiss(lower: String): Boolean {
        return lower.contains("system prompt cache miss for key")
    }

    private fun isTransientError(lower: String): Boolean {
        return lower.contains("timed out") ||
            lower.contains("timeout") ||
            lower.contains("http 429") ||
            lower.contains("http 503")
    }

    private companion object {
        val gson = GsonBuilder().disableHtmlEscaping().create()
    }
}
