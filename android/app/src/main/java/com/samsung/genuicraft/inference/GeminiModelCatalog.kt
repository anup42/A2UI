package com.samsung.genuicraft

import com.google.gson.JsonParser
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URLEncoder
import java.net.URL
import java.nio.charset.StandardCharsets

object GeminiModelCatalog {
    private val excludedModelKeywords = listOf(
        "image",
        "tts",
        "computer-use",
        "robotics",
        "customtools"
    )

    suspend fun fetchAvailableModels(apiKey: String): Result<List<String>> {
        return runCatching {
            val encodedKey = URLEncoder.encode(apiKey.trim(), StandardCharsets.UTF_8.name())
            val allModels = mutableListOf<String>()
            var pageToken: String? = null
            var pageCount = 0
            do {
                pageCount += 1
                val endpointUrl = buildString {
                    append("https://aiplatform.googleapis.com/v1/publishers/google/models?key=")
                    append(encodedKey)
                    if (!pageToken.isNullOrBlank()) {
                        append("&pageToken=")
                        append(URLEncoder.encode(pageToken, StandardCharsets.UTF_8.name()))
                    }
                }

                val endpoint = URL(endpointUrl)
                val connection = (endpoint.openConnection() as HttpURLConnection).apply {
                    requestMethod = "GET"
                    connectTimeout = 20000
                    readTimeout = 60000
                }

                try {
                    val code = connection.responseCode
                    val stream = if (code in 200..299) connection.inputStream else connection.errorStream
                    val raw = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
                    if (code !in 200..299) {
                        val short = raw.trim().ifBlank { "HTTP $code" }
                        throw IOException("HTTP $code: ${short.take(220)}")
                    }

                    val root = JsonParser.parseString(raw).asJsonObject
                    val models = root.getAsJsonArray("models")
                    models?.forEach { element ->
                        val obj = runCatching { element.asJsonObject }.getOrNull() ?: return@forEach
                        val name = obj.get("name")?.takeIf { it.isJsonPrimitive }?.asString?.trim().orEmpty()
                        val modelId = name
                            .removePrefix("publishers/google/models/")
                            .removePrefix("models/")
                            .trim()
                        if (!modelId.startsWith("gemini")) {
                            return@forEach
                        }
                        val supports = obj.getAsJsonArray("supportedGenerationMethods")
                        val canGenerate = supports == null || supports.any { item ->
                            item.isJsonPrimitive && item.asString.equals("generateContent", ignoreCase = true)
                        }
                        if (!canGenerate) {
                            return@forEach
                        }

                        val normalized = GeminiModelSettings.normalizeModelName(modelId)
                        val lower = normalized.lowercase()
                        if (excludedModelKeywords.any { keyword -> lower.contains(keyword) }) {
                            return@forEach
                        }
                        allModels += normalized
                    }

                    pageToken = root.get("nextPageToken")
                        ?.takeIf { it.isJsonPrimitive }
                        ?.asString
                        ?.trim()
                        ?.takeIf { it.isNotBlank() }
                } finally {
                    connection.disconnect()
                }
            } while (!pageToken.isNullOrBlank() && pageCount < 10)

            val sorted = allModels
                .distinct()
                .sortedWith(
                    compareBy<String> { priorityFor(it) }
                        .thenBy { it }
                )
                .toMutableList()

            if (sorted.none { it == GeminiModelSettings.DEFAULT_MODEL }) {
                sorted.add(0, GeminiModelSettings.DEFAULT_MODEL)
            }
            sorted
        }
    }

    private fun priorityFor(model: String): Int {
        return when (model) {
            "gemini-2.5-pro" -> 0
            "gemini-2.5-flash" -> 1
            "gemini-2.5-flash-lite" -> 2
            "gemini-2.0-flash" -> 3
            "gemini-2.0-flash-lite" -> 4
            "gemini-pro-latest" -> 5
            "gemini-flash-latest" -> 6
            "gemini-flash-lite-latest" -> 7
            else -> 10
        }
    }
}
