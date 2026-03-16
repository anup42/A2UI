package com.samsung.genuicraft.inference

import android.util.Base64
import com.google.gson.GsonBuilder
import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.util.Locale
import kotlin.math.min

class GeminiBackend(
    private val apiKey: String,
    private val model: String
) : InferenceBackend {

    override fun generate(request: InferenceBackend.GenerateRequest): InferenceBackend.GenerateResponse {
        val encodedKey = URLEncoder.encode(apiKey, StandardCharsets.UTF_8.name())
        val endpoint = URL("$GEMINI_BASE_URL/v1beta/models/$model:generateContent?key=$encodedKey")
        val connection = (endpoint.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 20000
            readTimeout = 180000
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
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

            if (code !in 200..299) {
                val short = raw.trim().ifBlank { "HTTP $code" }
                return InferenceBackend.GenerateResponse(
                    text = "",
                    rawResponse = raw,
                    error = "HTTP $code: ${short.take(320)}",
                    streamDurationMs = streamRead.streamDurationMs
                )
            }

            val extraction = extractGeminiText(raw)
            if (extraction.text.isNullOrBlank()) {
                val details = extraction.diagnostics?.let { " $it" }.orEmpty()
                return InferenceBackend.GenerateResponse(
                    text = "",
                    rawResponse = raw,
                    error = "Gemini response did not include candidate text.$details",
                    streamDurationMs = streamRead.streamDurationMs
                )
            }

            InferenceBackend.GenerateResponse(
                text = extraction.text,
                rawResponse = raw,
                error = null,
                streamDurationMs = streamRead.streamDurationMs
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
        // Gemini API does not expose a separate health endpoint.
        return InferenceBackend.HealthCheckResult(healthy = true)
    }

    override fun classifyError(error: String): InferenceBackend.ErrorClass {
        val lower = error.lowercase(Locale.US)
        if (isSearchToolConfigError(lower)) return InferenceBackend.ErrorClass.SEARCH_TOOL_CONFIG
        if (isStructuredOutputConfigError(lower)) return InferenceBackend.ErrorClass.STRUCTURED_OUTPUT_CONFIG
        if (isGeminiCachedContentMissing(lower)) return InferenceBackend.ErrorClass.CACHED_CONTENT_MISSING
        if (isTransientError(lower)) return InferenceBackend.ErrorClass.TRANSIENT
        return InferenceBackend.ErrorClass.UNKNOWN
    }

    // ── request payload ────────────────────────────────────────────────

    private fun buildRequestPayload(request: InferenceBackend.GenerateRequest): String {
        val body = JsonObject().apply {
            if (!request.cachedContentName.isNullOrBlank()) {
                addProperty("cachedContent", request.cachedContentName)
            }
            add("contents", JsonArray().apply {
                add(JsonObject().apply {
                    addProperty("role", "user")
                    add("parts", JsonArray().apply {
                        add(JsonObject().apply {
                            addProperty("text", request.prompt)
                        })
                    })
                })
            })

            add("generationConfig", JsonObject().apply {
                addProperty("temperature", request.temperature)
                addProperty("maxOutputTokens", min(request.maxOutputTokens, 8192))
                if (request.jsonMode) {
                    addProperty("responseMimeType", "application/json")
                    if (request.structuredOutput) {
                        add("responseSchema", buildStage3ResponseSchema())
                    }
                }
            })

            if (!request.systemPrompt.isNullOrBlank()) {
                add("systemInstruction", JsonObject().apply {
                    add("parts", JsonArray().apply {
                        add(JsonObject().apply {
                            addProperty("text", request.systemPrompt)
                        })
                    })
                })
            }

            if (request.enableGoogleSearch) {
                add("tools", JsonArray().apply {
                    add(JsonObject().apply {
                        add("google_search", JsonObject())
                    })
                })
            }
        }
        return gson.toJson(body)
    }

    internal fun buildStage3ResponseSchema(): JsonObject {
        return JsonObject().apply {
            addProperty("type", "ARRAY")
            add("items", JsonObject().apply {
                addProperty("type", "OBJECT")
                add("properties", JsonObject().apply {
                    add("version", JsonObject().apply { addProperty("type", "STRING") })
                    add("createSurface", JsonObject().apply { addProperty("type", "OBJECT") })
                    add("updateComponents", JsonObject().apply { addProperty("type", "OBJECT") })
                })
                add("required", JsonArray().apply { add("version") })
            })
        }
    }

    // ── response parsing ───────────────────────────────────────────────

    private data class GeminiTextExtraction(
        val text: String?,
        val diagnostics: String?
    )

    private fun extractGeminiText(raw: String): GeminiTextExtraction {
        val root = runCatching { JsonParser.parseString(raw).asJsonObject }.getOrNull()
            ?: return GeminiTextExtraction(
                text = null,
                diagnostics = "Response was not valid JSON."
            )

        val candidates = root.getAsJsonArray("candidates")
        if (candidates == null || candidates.size() == 0) {
            return GeminiTextExtraction(
                text = null,
                diagnostics = buildGeminiDiagnostics(root, null)
            )
        }

        val first = candidates.firstOrNull()?.asJsonObject
        if (first == null) {
            return GeminiTextExtraction(
                text = null,
                diagnostics = buildGeminiDiagnostics(root, null)
            )
        }

        val content = first.getAsJsonObject("content")
        val parts = content?.getAsJsonArray("parts")
        if (parts == null || parts.size() == 0) {
            return GeminiTextExtraction(
                text = null,
                diagnostics = buildGeminiDiagnostics(root, first)
            )
        }

        val builder = StringBuilder()
        for (part in parts) {
            val partObj = runCatching { part.asJsonObject }.getOrNull() ?: continue
            val textPart = partObj.get("text")?.takeIf { it.isJsonPrimitive }?.asString
            if (!textPart.isNullOrBlank()) {
                if (builder.isNotEmpty()) {
                    builder.append('\n')
                }
                builder.append(textPart)
                continue
            }

            val inlineText = decodeInlineDataText(partObj)
            if (!inlineText.isNullOrBlank()) {
                if (builder.isNotEmpty()) {
                    builder.append('\n')
                }
                builder.append(inlineText)
            }
        }

        val text = builder.toString().trim().ifBlank { null }
        return GeminiTextExtraction(
            text = text,
            diagnostics = if (text == null) buildGeminiDiagnostics(root, first) else null
        )
    }

    private fun decodeInlineDataText(part: JsonObject): String? {
        val inlineData = part.getAsJsonObject("inlineData") ?: return null
        val mimeType = inlineData.get("mimeType")?.takeIf { it.isJsonPrimitive }?.asString
            ?.lowercase(Locale.US)
            .orEmpty()
        if (mimeType.isNotBlank() && !mimeType.startsWith("text/") && !mimeType.contains("json")) {
            return null
        }
        val data = inlineData.get("data")?.takeIf { it.isJsonPrimitive }?.asString ?: return null
        return runCatching {
            val decoded = Base64.decode(data, Base64.DEFAULT)
            String(decoded, Charsets.UTF_8).trim()
        }.getOrNull()?.ifBlank { null }
    }

    private fun buildGeminiDiagnostics(root: JsonObject, candidate: JsonObject?): String {
        val parts = mutableListOf<String>()
        root.getAsJsonObject("promptFeedback")?.let { feedback ->
            feedback.get("blockReason")?.takeIf { it.isJsonPrimitive }?.asString?.takeIf { it.isNotBlank() }?.let {
                parts += "blockReason=$it"
            }
            feedback.get("finishReason")?.takeIf { it.isJsonPrimitive }?.asString?.takeIf { it.isNotBlank() }?.let {
                parts += "finishReason=$it"
            }
        }
        candidate?.get("finishReason")?.takeIf { it.isJsonPrimitive }?.asString?.takeIf { it.isNotBlank() }?.let {
            parts += "finishReason=$it"
        }
        candidate?.get("safetyRatings")?.let { ratings ->
            val ratingSummary = ratings.toString().take(160)
            if (ratingSummary.isNotBlank()) {
                parts += "safetyRatings=$ratingSummary"
            }
        }
        if (parts.isEmpty()) {
            parts += "Raw response shape did not contain text parts."
        }
        return parts.joinToString("; ").take(280)
    }

    // ── error classification ───────────────────────────────────────────

    private fun isSearchToolConfigError(lower: String): Boolean {
        return lower.contains("google_search") ||
            lower.contains("unknown name \"tools\"") ||
            lower.contains("unknown field \"tools\"") ||
            (lower.contains("invalid_argument") && lower.contains("tool")) ||
            (lower.contains("http 400") && lower.contains("tool"))
    }

    private fun isStructuredOutputConfigError(lower: String): Boolean {
        return lower.contains("responseschema") ||
            lower.contains("response schema") ||
            lower.contains("unknown name \"responseschema\"") ||
            lower.contains("unknown field \"responseschema\"") ||
            (lower.contains("invalid_argument") && lower.contains("schema")) ||
            (lower.contains("http 400") && lower.contains("schema"))
    }

    private fun isGeminiCachedContentMissing(lower: String): Boolean {
        val referencesCache = lower.contains("cachedcontent") || lower.contains("cached content")
        if (!referencesCache) {
            return false
        }
        return lower.contains("not found") ||
            lower.contains("does not exist") ||
            lower.contains("http 404") ||
            (lower.contains("invalid_argument") && lower.contains("cache"))
    }

    private fun isTransientError(lower: String): Boolean {
        return lower.contains("timed out") ||
            lower.contains("timeout") ||
            lower.contains("http 429") ||
            lower.contains("http 503") ||
            lower.contains("candidate text")
    }

    private companion object {
        const val GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
        val gson = GsonBuilder().disableHtmlEscaping().create()
    }
}
