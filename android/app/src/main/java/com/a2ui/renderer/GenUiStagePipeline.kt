package com.samsung.genuicraft

import android.content.Context
import android.util.Base64
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URLEncoder
import java.net.URL
import java.nio.charset.StandardCharsets
import java.util.Locale
import kotlin.math.min

class GenUiStagePipeline(private val appContext: Context) {
    enum class Stage {
        STAGE2,
        STAGE3,
        STAGE4
    }

    data class StageUpdate(
        val stage: Stage,
        val message: String
    )

    data class PipelineResult(
        val queryText: String,
        val stage2Prompt: String,
        val stage2Response: String,
        val stage3Prompt: String,
        val stage3SystemPrompt: String?,
        val stage3Json: String,
        val usedFallback: Boolean,
        val warnings: List<String>,
        val renderResult: GenUiNativeRenderer.RenderResult
    )

    sealed interface Outcome {
        data class Success(val result: PipelineResult) : Outcome
        data class Failure(
            val stage: Stage,
            val message: String,
            val stage2Response: String? = null,
            val stage3Json: String? = null
        ) : Outcome
    }

    suspend fun execute(
        queryText: String,
        onStageUpdate: (StageUpdate) -> Unit
    ): Outcome = withContext(Dispatchers.IO) {
        val normalizedQuery = queryText.trim()
        if (normalizedQuery.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE2,
                message = "Query is empty."
            )
        }

        val apiKey = BuildConfig.GEMINI_API_KEY.trim()
        if (apiKey.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE2,
                message = "Gemini API key is missing. Set GEMINI_API_KEY before building the app."
            )
        }

        val responseTemplate = runCatching { loadPromptAsset(STAGE2_PROMPT_ASSET) }
            .getOrElse {
                return@withContext Outcome.Failure(
                    stage = Stage.STAGE2,
                    message = "Could not load stage 2 prompt: ${it.message ?: it.javaClass.simpleName}"
                )
            }

        val stage2Prompt = renderPrompt(
            responseTemplate,
            "query_text" to normalizedQuery,
            "intent" to "",
            "tags" to ""
        )

        postUpdate(onStageUpdate, Stage.STAGE2, "Fetching response")
        val stage2Call = generateWithRetry(
            apiKey = apiKey,
            model = MODEL_GEMINI_2_5_PRO,
            prompt = stage2Prompt,
            systemPrompt = null,
            temperature = 0.3,
            maxOutputTokens = 4096,
            jsonMode = false,
            enableGoogleSearch = true
        )
        if (stage2Call.error != null) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE2,
                message = stage2Call.error
            )
        }
        val stage2Response = stage2Call.text.trim()
        if (stage2Response.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE2,
                message = "Stage 2 returned empty output."
            )
        }

        val genUiTemplate = runCatching { loadPromptAsset(STAGE3_PROMPT_ASSET) }
            .getOrElse {
                return@withContext Outcome.Failure(
                    stage = Stage.STAGE3,
                    message = "Could not load stage 3 prompt: ${it.message ?: it.javaClass.simpleName}",
                    stage2Response = stage2Response
                )
            }

        val promptContext = prepareStage3PromptContext(genUiTemplate)
        val stage3Prompt = buildStage3UserPrompt(
            userTemplate = promptContext.userTemplate,
            stage2Response = stage2Response,
            assets = emptyList()
        )

        postUpdate(onStageUpdate, Stage.STAGE3, "Converting response into GenUICraft IR JSON")
        val stage3Call = generateWithRetry(
            apiKey = apiKey,
            model = MODEL_GEMINI_2_5_PRO,
            prompt = stage3Prompt,
            systemPrompt = promptContext.systemPrompt,
            temperature = 0.2,
            maxOutputTokens = 8192,
            jsonMode = true,
            enableGoogleSearch = false
        )

        if (stage3Call.error != null) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = stage3Call.error,
                stage2Response = stage2Response
            )
        }

        val warnings = mutableListOf<String>()
        var stage3JsonElement = extractJsonElement(stage3Call.text)
        var usedFallback = false

        if (stage3JsonElement == null) {
            warnings += "Stage 3 JSON parse failed; running repair pass."
            val repairCall = generateWithRetry(
                apiKey = apiKey,
                model = MODEL_GEMINI_2_5_PRO,
                prompt = buildRepairPrompt(stage3Call.text),
                systemPrompt = promptContext.systemPrompt,
                temperature = 0.2,
                maxOutputTokens = 8192,
                jsonMode = true,
                enableGoogleSearch = false
            )
            if (repairCall.error == null) {
                stage3JsonElement = extractJsonElement(repairCall.text)
            }
        }

        if (stage3JsonElement == null) {
            warnings += "Stage 3 fallback JSON was used."
            stage3JsonElement = buildFallbackGenUi(stage2Response)
            usedFallback = true
        }

        val normalizedGenUi = normalizeGenUiPayload(stage3JsonElement)
        var stage3Json = gson.toJson(normalizedGenUi)
        if (responseContainsInlineMedia(stage2Response) && !genUiPreservesInlineMedia(stage3Json) && !usedFallback) {
            warnings += "Stage 3 dropped inline media; using text-preserving fallback UI."
            stage3Json = gson.toJson(buildFallbackGenUi(stage2Response))
            usedFallback = true
        }

        postUpdate(onStageUpdate, Stage.STAGE4, "Rendering output")
        var renderResult = GenUiNativeRenderer.render(stage3Json, sourceDir = null)

        if (renderResult.errorMessage != null && !usedFallback) {
            warnings += "Native rendering failed for stage 3 output; using fallback UI."
            val fallback = buildFallbackGenUi(stage2Response)
            stage3Json = gson.toJson(fallback)
            renderResult = GenUiNativeRenderer.render(stage3Json, sourceDir = null)
            usedFallback = true
        }

        if (renderResult.errorMessage != null) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE4,
                message = renderResult.errorMessage,
                stage2Response = stage2Response,
                stage3Json = stage3Json
            )
        }

        return@withContext Outcome.Success(
            result = PipelineResult(
                queryText = normalizedQuery,
                stage2Prompt = stage2Prompt,
                stage2Response = stage2Response,
                stage3Prompt = stage3Prompt,
                stage3SystemPrompt = promptContext.systemPrompt,
                stage3Json = stage3Json,
                usedFallback = usedFallback,
                warnings = warnings,
                renderResult = renderResult
            )
        )
    }

    private suspend fun postUpdate(
        callback: (StageUpdate) -> Unit,
        stage: Stage,
        message: String
    ) {
        withContext(Dispatchers.Main) {
            callback(StageUpdate(stage = stage, message = message))
        }
    }

    private fun buildRepairPrompt(rawText: String): String {
        return (
            "The previous output was not valid JSON or failed schema validation. " +
                "Fix the output to be valid JSON that satisfies the schema requirements. " +
                "Return ONLY the corrected JSON.\n\nOriginal:\n${rawText.trim()}"
            )
    }

    private fun responseContainsInlineMedia(text: String): Boolean {
        return Regex(
            """(?im)^\s*(?:Media:\s*(?:Image|Icon)=|Image:\s*(?:https?://|/assets/|assets/)|Icon:\s*(?:https?://|/assets/|assets/))"""
        ).containsMatchIn(text)
    }

    private fun genUiPreservesInlineMedia(jsonText: String): Boolean {
        return Regex(
            """"component"\s*:\s*"(?:Image|Icon)"""",
            RegexOption.IGNORE_CASE
        ).containsMatchIn(jsonText) ||
            Regex(
                """(?i)Media:\s*(?:Image|Icon)=|(?:^|\\n)(?:Image|Icon):\s*(?:https?://|/assets/|assets/)"""
            ).containsMatchIn(jsonText)
    }

    private fun normalizeGenUiPayload(json: JsonElement): JsonElement {
        if (json.isJsonArray) {
            return json
        }
        if (!json.isJsonObject) {
            return json
        }

        val obj = json.asJsonObject
        val directArray = obj.get("genui_json")
        if (directArray != null && directArray.isJsonArray) {
            return directArray
        }
        val messages = obj.get("messages")
        if (messages != null && messages.isJsonArray) {
            return messages
        }
        val payload = obj.get("payload")
        if (payload != null) {
            if (payload.isJsonArray) {
                return payload
            }
            if (payload.isJsonObject) {
                val payloadObj = payload.asJsonObject
                val payloadMessages = payloadObj.get("messages")
                if (payloadMessages != null && payloadMessages.isJsonArray) {
                    return payloadMessages
                }
            }
        }
        return json
    }

    private fun buildFallbackGenUi(stage2Response: String): JsonArray {
        val textValue = stage2Response.trim().ifBlank { "No content generated." }
        val surfaceId = "surface_live"
        return JsonArray().apply {
            add(
                JsonObject().apply {
                    addProperty("version", "v0.9")
                    add("createSurface", JsonObject().apply {
                        addProperty("surfaceId", surfaceId)
                        addProperty("catalogId", "https://genui.local/specification/v0_9/standard_catalog.json")
                    })
                }
            )
            add(
                JsonObject().apply {
                    addProperty("version", "v0.9")
                    add("updateComponents", JsonObject().apply {
                        addProperty("surfaceId", surfaceId)
                        add("components", JsonArray().apply {
                            add(JsonObject().apply {
                                addProperty("id", "root")
                                addProperty("component", "Column")
                                add("children", JsonArray().apply {
                                    add("text_1")
                                })
                            })
                            add(JsonObject().apply {
                                addProperty("id", "text_1")
                                addProperty("component", "Text")
                                addProperty("variant", "body")
                                addProperty("text", textValue)
                            })
                        })
                    })
                }
            )
        }
    }

    private fun prepareStage3PromptContext(template: String): Stage3PromptContext {
        val placeholder = "{response_text}"
        if (!template.contains(placeholder)) {
            return Stage3PromptContext(systemPrompt = null, userTemplate = template)
        }

        val split = template.split(placeholder, limit = 2)
        val systemPrompt = "${split[0]}[RESPONSE_TEXT_IS_PROVIDED_IN_THE_USER_MESSAGE]${split[1]}".trim()
        val userTemplate = (
            "Convert the response text into valid GenUICraft JSON.\n" +
                "Return ONLY the JSON message array.\n\n" +
                "Response:\n{response_text}"
            )
        return Stage3PromptContext(systemPrompt = systemPrompt, userTemplate = userTemplate)
    }

    private fun buildStage3UserPrompt(
        userTemplate: String,
        stage2Response: String,
        assets: List<AssetMapping>
    ): String {
        val assetPolicy = if (assets.isEmpty()) {
            "Asset URL policy for this request:\n" +
                "- No local asset mapping is provided.\n" +
                "- Preserve media URLs from the response exactly as written.\n" +
                "- Do not invent local placeholder paths such as /image.jpg or /asset/foo.png."
        } else {
            "Asset URL policy for this request:\n" +
                "- Use only local media paths from the provided Assets mapping.\n" +
                "- Do not emit remote media URLs for images/icons.\n" +
                "- Do not invent local placeholder paths not present in the mapping."
        }

        val assetContext = if (assets.isEmpty()) {
            ""
        } else {
            val rows = assets.joinToString("\n") { "- ${it.url} -> ${it.localPath}" }
            "Assets (local copies of any URLs in the response; use ONLY these local paths):\n$rows"
        }

        val responseWithPolicy = "${stage2Response.trim()}\n\n$assetPolicy"
        val responseText = if (assetContext.isBlank()) {
            responseWithPolicy
        } else {
            "$responseWithPolicy\n\n$assetContext"
        }

        return renderPrompt(userTemplate, "response_text" to responseText)
    }

    private fun renderPrompt(template: String, vararg args: Pair<String, String>): String {
        var rendered = template
        args.forEach { (key, value) ->
            rendered = rendered.replace("{$key}", value)
        }
        return rendered
    }

    private fun loadPromptAsset(path: String): String {
        return appContext.assets.open(path).bufferedReader(Charsets.UTF_8).use { it.readText() }
    }

    private fun extractJsonElement(text: String): JsonElement? {
        val cleaned = text.trim()
        if (cleaned.isEmpty()) {
            return null
        }

        extractFencedBlock(cleaned)?.let { fenced ->
            try {
                return JsonParser.parseString(fenced)
            } catch (_: Exception) {
            }
        }

        if (cleaned.startsWith("[") || cleaned.startsWith("{")) {
            try {
                return JsonParser.parseString(cleaned)
            } catch (_: Exception) {
            }
        }

        findFirstBalanced(cleaned, '[', ']')?.let { arrayCandidate ->
            try {
                return JsonParser.parseString(arrayCandidate)
            } catch (_: Exception) {
            }
        }

        findFirstBalanced(cleaned, '{', '}')?.let { objectCandidate ->
            try {
                return JsonParser.parseString(objectCandidate)
            } catch (_: Exception) {
            }
        }

        return null
    }

    private fun extractFencedBlock(text: String): String? {
        val start = text.indexOf("```")
        if (start < 0) {
            return null
        }
        val end = text.indexOf("```", startIndex = start + 3)
        if (end <= start) {
            return null
        }
        val block = text.substring(start + 3, end).trim()
        return if (block.startsWith("json", ignoreCase = true)) {
            block.removePrefix("json").trim()
        } else {
            block
        }
    }

    private fun findFirstBalanced(text: String, open: Char, close: Char): String? {
        var index = text.indexOf(open)
        while (index >= 0) {
            val candidate = balancedSubstring(text, index, open, close)
            if (candidate != null) {
                return candidate
            }
            index = text.indexOf(open, startIndex = index + 1)
        }
        return null
    }

    private fun balancedSubstring(text: String, start: Int, open: Char, close: Char): String? {
        var depth = 0
        var inString = false
        var escape = false

        for (i in start until text.length) {
            val ch = text[i]
            if (escape) {
                escape = false
                continue
            }
            if (ch == '\\' && inString) {
                escape = true
                continue
            }
            if (ch == '"') {
                inString = !inString
                continue
            }
            if (inString) {
                continue
            }
            if (ch == open) {
                depth += 1
            } else if (ch == close) {
                depth -= 1
                if (depth == 0) {
                    return text.substring(start, i + 1)
                }
            }
        }
        return null
    }

    private fun generateWithRetry(
        apiKey: String,
        model: String,
        prompt: String,
        systemPrompt: String?,
        temperature: Double,
        maxOutputTokens: Int,
        jsonMode: Boolean,
        enableGoogleSearch: Boolean = false
    ): GeminiResponse {
        var attempt = 0
        var last: GeminiResponse = GeminiResponse(text = "", rawResponse = null, error = "Unknown Gemini error")
        while (attempt < 3) {
            attempt += 1
            last = generateOnce(
                apiKey = apiKey,
                model = model,
                prompt = prompt,
                systemPrompt = systemPrompt,
                temperature = temperature,
                maxOutputTokens = maxOutputTokens,
                jsonMode = jsonMode,
                enableGoogleSearch = enableGoogleSearch
            )
            if (last.error == null) {
                return last
            }
            if (enableGoogleSearch && isSearchToolConfigError(last.error)) {
                return generateOnce(
                    apiKey = apiKey,
                    model = model,
                    prompt = prompt,
                    systemPrompt = systemPrompt,
                    temperature = temperature,
                    maxOutputTokens = maxOutputTokens,
                    jsonMode = jsonMode,
                    enableGoogleSearch = false
                )
            }
            val lower = last.error.lowercase(Locale.US)
            val retryable = lower.contains("timed out") ||
                lower.contains("timeout") ||
                lower.contains("http 429") ||
                lower.contains("http 503") ||
                lower.contains("candidate text")
            if (!retryable || attempt >= 3) {
                return last
            }
            Thread.sleep(1000L * attempt)
        }
        return last
    }

    private fun generateOnce(
        apiKey: String,
        model: String,
        prompt: String,
        systemPrompt: String?,
        temperature: Double,
        maxOutputTokens: Int,
        jsonMode: Boolean,
        enableGoogleSearch: Boolean = false
    ): GeminiResponse {
        val encodedKey = URLEncoder.encode(apiKey, StandardCharsets.UTF_8.name())
        val endpoint = URL("https://generativelanguage.googleapis.com/v1beta/models/$model:generateContent?key=$encodedKey")
        val connection = (endpoint.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 20000
            readTimeout = 180000
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
        }

        val body = buildRequestPayload(
            prompt = prompt,
            systemPrompt = systemPrompt,
            temperature = temperature,
            maxOutputTokens = maxOutputTokens,
            jsonMode = jsonMode,
            enableGoogleSearch = enableGoogleSearch
        )

        return try {
            connection.outputStream.use { out ->
                out.write(body.toByteArray(StandardCharsets.UTF_8))
            }

            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val raw = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()

            if (code !in 200..299) {
                val short = raw.trim().ifBlank { "HTTP $code" }
                return GeminiResponse(
                    text = "",
                    rawResponse = raw,
                    error = "HTTP $code: ${short.take(320)}"
                )
            }

            val extraction = extractGeminiText(raw)
            if (extraction.text.isNullOrBlank()) {
                val details = extraction.diagnostics?.let { " $it" }.orEmpty()
                return GeminiResponse(
                    text = "",
                    rawResponse = raw,
                    error = "Gemini response did not include candidate text.$details"
                )
            }

            GeminiResponse(
                text = extraction.text,
                rawResponse = raw,
                error = null
            )
        } catch (io: IOException) {
            GeminiResponse(
                text = "",
                rawResponse = null,
                error = io.message ?: io.javaClass.simpleName
            )
        } finally {
            connection.disconnect()
        }
    }

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

    private fun buildRequestPayload(
        prompt: String,
        systemPrompt: String?,
        temperature: Double,
        maxOutputTokens: Int,
        jsonMode: Boolean,
        enableGoogleSearch: Boolean
    ): String {
        val body = JsonObject().apply {
            add("contents", JsonArray().apply {
                add(JsonObject().apply {
                    addProperty("role", "user")
                    add("parts", JsonArray().apply {
                        add(JsonObject().apply {
                            addProperty("text", prompt)
                        })
                    })
                })
            })

            add("generationConfig", JsonObject().apply {
                addProperty("temperature", temperature)
                addProperty("maxOutputTokens", min(maxOutputTokens, 8192))
                if (jsonMode) {
                    addProperty("responseMimeType", "application/json")
                }
            })

            if (!systemPrompt.isNullOrBlank()) {
                add("systemInstruction", JsonObject().apply {
                    add("parts", JsonArray().apply {
                        add(JsonObject().apply {
                            addProperty("text", systemPrompt)
                        })
                    })
                })
            }

            if (enableGoogleSearch) {
                add("tools", JsonArray().apply {
                    add(JsonObject().apply {
                        add("google_search", JsonObject())
                    })
                })
            }
        }
        return gson.toJson(body)
    }

    private fun isSearchToolConfigError(error: String): Boolean {
        val normalized = error.lowercase(Locale.US)
        return normalized.contains("google_search") ||
            normalized.contains("unknown name \"tools\"") ||
            normalized.contains("unknown field \"tools\"") ||
            (normalized.contains("invalid_argument") && normalized.contains("tool")) ||
            (normalized.contains("http 400") && normalized.contains("tool"))
    }

    private data class GeminiResponse(
        val text: String,
        val rawResponse: String?,
        val error: String?
    )

    private data class GeminiTextExtraction(
        val text: String?,
        val diagnostics: String?
    )

    private data class Stage3PromptContext(
        val systemPrompt: String?,
        val userTemplate: String
    )

    private data class AssetMapping(
        val url: String,
        val localPath: String
    )

    private companion object {
        const val STAGE2_PROMPT_ASSET = "pipeline_prompts/response_gen.md"
        const val STAGE3_PROMPT_ASSET = "pipeline_prompts/genui_gen.md"
        const val MODEL_GEMINI_2_5_PRO = "gemini-2.5-pro"

        val gson = com.google.gson.GsonBuilder().disableHtmlEscaping().create()
    }
}
