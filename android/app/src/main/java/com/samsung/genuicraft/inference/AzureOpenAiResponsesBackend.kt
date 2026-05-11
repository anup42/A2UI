package com.samsung.genuicraft.inference

import com.google.gson.GsonBuilder
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets
import java.util.Locale
import kotlin.math.min

class AzureOpenAiResponsesBackend(
    private val apiKey: String,
    private val responsesEndpoint: String,
    private val deployment: String
) : InferenceBackend {

    override fun generate(request: InferenceBackend.GenerateRequest): InferenceBackend.GenerateResponse {
        val cleanKey = apiKey.trim()
        if (cleanKey.isBlank()) {
            return InferenceBackend.GenerateResponse(
                text = "",
                rawResponse = null,
                error = "Azure OpenAI API key is missing. Add AZURE_OPENAI_API_KEY (or AZURE_OPENAI_SUBSCRIPTION_KEY) in runtime keys.",
                streamDurationMs = null
            )
        }

        val endpoint = runCatching { URL(normalizeResponsesEndpoint(responsesEndpoint)) }
            .getOrElse {
                return InferenceBackend.GenerateResponse(
                    text = "",
                    rawResponse = null,
                    error = "Azure OpenAI endpoint is invalid: ${it.message ?: it.javaClass.simpleName}",
                    streamDurationMs = null
                )
            }
        val connection = (endpoint.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 20000
            readTimeout = 180000
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("api-key", cleanKey)
        }
        val body = buildRequestPayload(request)

        return try {
            connection.outputStream.use { out ->
                out.write(body.toByteArray(StandardCharsets.UTF_8))
            }

            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val streamRead = InferenceStreamUtils.readStreamWithTiming(stream)
            val raw = streamRead.text
            val usage = parseTokenUsage(raw)

            if (code !in 200..299) {
                val short = extractErrorMessage(raw).ifBlank { "HTTP $code" }
                return InferenceBackend.GenerateResponse(
                    text = "",
                    rawResponse = raw,
                    error = "HTTP $code: ${short.take(320)} [provider=azure_openai endpoint=${endpoint.host}${endpoint.path} deployment=${cleanDeployment()}]",
                    streamDurationMs = streamRead.streamDurationMs,
                    inputTokens = usage.inputTokens,
                    outputTokens = usage.outputTokens
                )
            }

            val text = extractResponseText(raw)
            if (text.isNullOrBlank()) {
                return InferenceBackend.GenerateResponse(
                    text = "",
                    rawResponse = raw,
                    error = "Azure OpenAI response did not include output text.",
                    streamDurationMs = streamRead.streamDurationMs,
                    inputTokens = usage.inputTokens,
                    outputTokens = usage.outputTokens
                )
            }

            InferenceBackend.GenerateResponse(
                text = text,
                rawResponse = raw,
                error = null,
                streamDurationMs = streamRead.streamDurationMs,
                inputTokens = usage.inputTokens,
                outputTokens = usage.outputTokens
            )
        } catch (io: IOException) {
            InferenceBackend.GenerateResponse(
                text = "",
                rawResponse = null,
                error = io.message ?: io.javaClass.simpleName,
                streamDurationMs = null
            )
        } finally {
            connection.disconnect()
        }
    }

    override fun checkHealth(): InferenceBackend.HealthCheckResult {
        return InferenceBackend.HealthCheckResult(healthy = true)
    }

    override fun classifyError(error: String): InferenceBackend.ErrorClass {
        val lower = error.lowercase(Locale.US)
        if (lower.contains("json_object") ||
            lower.contains("response_format") ||
            lower.contains("text.format") ||
            (lower.contains("invalid") && lower.contains("format"))
        ) {
            return InferenceBackend.ErrorClass.STRUCTURED_OUTPUT_CONFIG
        }
        if (lower.contains("web_search") ||
            lower.contains("web search") ||
            (lower.contains("tool") && lower.contains("unsupported"))
        ) {
            return InferenceBackend.ErrorClass.SEARCH_TOOL_CONFIG
        }
        if (lower.contains("timed out") ||
            lower.contains("timeout") ||
            lower.contains("http 408") ||
            lower.contains("http 429") ||
            lower.contains("http 500") ||
            lower.contains("http 502") ||
            lower.contains("http 503") ||
            lower.contains("http 504")
        ) {
            return InferenceBackend.ErrorClass.TRANSIENT
        }
        return InferenceBackend.ErrorClass.UNKNOWN
    }

    private fun buildRequestPayload(request: InferenceBackend.GenerateRequest): String {
        return gson.toJson(JsonObject().apply {
            addProperty("model", cleanDeployment())
            if (!request.systemPrompt.isNullOrBlank()) {
                addProperty("instructions", request.systemPrompt)
            }
            addProperty("input", request.prompt)
            addProperty("temperature", request.temperature)
            addProperty("max_output_tokens", min(request.maxOutputTokens, 8192))
            addProperty("store", false)
            if (request.enableGoogleSearch && !request.jsonMode) {
                add("tools", com.google.gson.JsonArray().apply {
                    add(JsonObject().apply {
                        addProperty("type", "web_search_preview")
                    })
                })
            }
            if (request.jsonMode) {
                add("text", JsonObject().apply {
                    add("format", JsonObject().apply {
                        addProperty("type", "json_object")
                    })
                })
            }
        })
    }

    private fun cleanDeployment(): String {
        return deployment.trim().ifBlank { "gpt-5.4" }
    }

    private fun normalizeResponsesEndpoint(rawEndpoint: String): String {
        var normalized = rawEndpoint.trim().ifBlank {
            "https://genui1.openai.azure.com/openai/responses?api-version=2025-04-01-preview"
        }
        if (!normalized.startsWith("http://", ignoreCase = true) &&
            !normalized.startsWith("https://", ignoreCase = true)
        ) {
            normalized = "https://$normalized"
        }
        if (!normalized.contains("api-version=", ignoreCase = true)) {
            val separator = if (normalized.contains("?")) "&" else "?"
            normalized = "$normalized${separator}api-version=2025-04-01-preview"
        }
        return normalized
    }

    private fun parseTokenUsage(raw: String): TokenUsage {
        val root = runCatching { JsonParser.parseString(raw).asJsonObject }.getOrNull()
            ?: return TokenUsage(inputTokens = null, outputTokens = null)
        val usage = root.getAsJsonObject("usage") ?: return TokenUsage(inputTokens = null, outputTokens = null)
        return TokenUsage(
            inputTokens = usage.intOrNull("input_tokens") ?: usage.intOrNull("prompt_tokens"),
            outputTokens = usage.intOrNull("output_tokens") ?: usage.intOrNull("completion_tokens")
        )
    }

    private fun extractResponseText(raw: String): String? {
        val root = runCatching { JsonParser.parseString(raw).asJsonObject }.getOrNull()
            ?: return null
        root.stringOrNull("output_text")?.takeIf { it.isNotBlank() }?.let { return it.trim() }

        val builder = StringBuilder()
        for (item in root.elementsOrEmpty("output")) {
            val itemObj = item.asObjectOrNull() ?: continue
            for (part in itemObj.elementsOrEmpty("content")) {
                val partObj = part.asObjectOrNull() ?: continue
                val type = partObj.stringOrNull("type").orEmpty()
                val text = partObj.stringOrNull("text")
                    ?: partObj.stringOrNull("output_text")
                    ?: partObj.stringOrNull("refusal")
                if (text.isNullOrBlank()) {
                    continue
                }
                if (type.isBlank() ||
                    type == "output_text" ||
                    type == "text" ||
                    type == "refusal"
                ) {
                    if (builder.isNotEmpty()) {
                        builder.append('\n')
                    }
                    builder.append(text)
                }
            }
        }
        return builder.toString().trim().ifBlank { null }
    }

    private fun extractErrorMessage(raw: String): String {
        val root = runCatching { JsonParser.parseString(raw).asJsonObject }.getOrNull()
            ?: return raw.trim()
        val error = root.getAsJsonObject("error")
        return error?.stringOrNull("message")
            ?: error?.stringOrNull("code")
            ?: root.stringOrNull("message")
            ?: raw.trim()
    }

    private data class TokenUsage(
        val inputTokens: Int?,
        val outputTokens: Int?
    )

    private companion object {
        val gson = GsonBuilder().disableHtmlEscaping().create()
    }
}

private fun JsonElement.asObjectOrNull(): JsonObject? {
    return takeIf { it.isJsonObject }?.asJsonObject
}

private fun JsonObject.elementsOrEmpty(key: String): List<JsonElement> {
    val value = get(key) ?: return emptyList()
    return when {
        value.isJsonArray -> value.asJsonArray.toList()
        value.isJsonObject -> listOf(value)
        else -> emptyList()
    }
}

private fun JsonObject.stringOrNull(key: String): String? {
    val value = get(key) ?: return null
    if (!value.isJsonPrimitive) return null
    val primitive = value.asJsonPrimitive
    if (!primitive.isString) return null
    return primitive.asString
}

private fun JsonObject.intOrNull(key: String): Int? {
    val value = get(key) ?: return null
    if (!value.isJsonPrimitive) return null
    val primitive = value.asJsonPrimitive
    if (!primitive.isNumber) return null
    return runCatching { primitive.asInt }.getOrNull()
}
