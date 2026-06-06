package com.samsung.genuicraft

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.File

object GenUiRecordParser {
    fun parseRecords(raw: String, sourceDir: File?, sourceLabel: String): List<GenUiRecord> {
        val trimmed = raw.trim()
        if (trimmed.isEmpty()) {
            return emptyList()
        }

        runCatching { JsonParser.parseString(trimmed) }.getOrNull()?.let { parsed ->
            return toRecords(parsed, sourceDir, sourceLabel)
        }

        val records = mutableListOf<GenUiRecord>()
        raw.lineSequence().forEachIndexed { index, line ->
            val lineTrimmed = line.trim()
            if (lineTrimmed.isEmpty()) {
                return@forEachIndexed
            }
            val parsed = runCatching { JsonParser.parseString(lineTrimmed) }.getOrNull() ?: return@forEachIndexed
            records += toRecord(parsed, index + 1, sourceDir, sourceLabel)
        }
        return records
    }

    private fun toRecords(parsed: JsonElement, sourceDir: File?, sourceLabel: String): List<GenUiRecord> {
        if (parsed.isJsonArray) {
            return parsed.asJsonArray.mapIndexed { index, element ->
                toRecord(element, index + 1, sourceDir, sourceLabel)
            }
        }
        return listOf(toRecord(parsed, 1, sourceDir, sourceLabel))
    }

    private fun toRecord(
        element: JsonElement,
        index: Int,
        sourceDir: File?,
        sourceLabel: String
    ): GenUiRecord {
        val obj = if (element.isJsonObject) element.asJsonObject else null
        val uiId = obj?.getString("ui_id") ?: obj?.getAsJsonArrayOrNull("genui_json")?.firstSurfaceId()
        val headerTitle = obj?.extractHeaderTitle()
        val summary = obj?.extractSummary()
        val queryId = obj?.getString("query_id")
        val responseId = obj?.getString("response_id")
        val htmlAssetPath = obj?.getString("html_asset") ?: obj?.getString("htmlAsset")

        return GenUiRecord(
            title = headerTitle ?: uiId ?: "Item $index",
            rawJson = element.toString(),
            sourceDir = sourceDir,
            sourceLabel = sourceLabel,
            uiId = uiId,
            summary = summary,
            queryId = queryId,
            responseId = responseId,
            htmlAssetPath = htmlAssetPath
        )
    }

    private fun JsonObject.extractHeaderTitle(): String? {
        val directTitle =
            getString("title")
                ?: getString("header")
                ?: getString("heading")
                ?: getString("name")
        if (!directTitle.isNullOrBlank()) {
            return normalizeHeaderTitle(directTitle)
        }

        val fromComponents = getAsJsonArrayOrNull("genui_json")?.extractHeaderFromComponents()
        if (!fromComponents.isNullOrBlank()) {
            return fromComponents
        }

        return null
    }

    private fun JsonArray.extractHeaderFromComponents(): String? {
        var fallback: String? = null
        for (entry in this) {
            val msg = entry.asJsonObjectOrNull() ?: continue
            val components =
                msg.getAsJsonObjectOrNull("updateComponents")
                    ?.getAsJsonArrayOrNull("components")
                    ?: continue

            for (component in components) {
                val componentObj = component.asJsonObjectOrNull() ?: continue
                if (!componentObj.getString("component").equals("Text", ignoreCase = true)) {
                    continue
                }

                val text = componentObj.getString("text").orEmpty()
                if (text.isBlank()) {
                    continue
                }
                if (fallback == null) {
                    fallback = normalizeHeaderTitle(text)
                }

                val id = componentObj.getString("id")?.lowercase().orEmpty()
                val variant = componentObj.getString("variant")?.lowercase().orEmpty()
                val headingVariant = variant in setOf("h1", "h2", "h3", "h4", "headline", "title")
                val titleLikeId = id.contains("header") || id.contains("title")
                if (headingVariant || titleLikeId) {
                    return normalizeHeaderTitle(text)
                }
            }
        }
        return fallback
    }

    private fun normalizeHeaderTitle(raw: String): String {
        val firstLine = raw
            .replace("\r", "\n")
            .lineSequence()
            .map { it.trim() }
            .firstOrNull { it.isNotBlank() }
            ?: raw.trim()

        var candidate = firstLine
            .replace("â€¢", " ")
            .replace("•", " ")
            .replace("**", "")
            .replace(Regex("\\s+"), " ")
            .trim()

        if (candidate.contains(" | ")) {
            candidate = candidate.substringBefore(" | ").trim()
        }
        if (candidate.length > 96 && candidate.contains(".")) {
            val firstSentence = candidate.substringBefore(".").trim()
            if (firstSentence.length >= 18) {
                candidate = firstSentence
            }
        }

        return candidate.take(110)
    }

    private fun JsonObject.extractSummary(): String? {
        val directSummary =
            getString("summary")
                ?: getString("description")
                ?: getString("overview")
                ?: getString("intro")
        if (!directSummary.isNullOrBlank()) {
            return normalizeSummary(directSummary)
        }

        val fromComponents = getAsJsonArrayOrNull("genui_json")?.extractSummaryFromComponents()
        if (!fromComponents.isNullOrBlank()) {
            return fromComponents
        }

        val intent = getString("intent")?.trim().takeUnless { it.isNullOrEmpty() }
        val tags = getAsJsonArrayOrNull("tags")
            ?.mapNotNull { tag ->
                if (tag.isJsonPrimitive && tag.asJsonPrimitive.isString) tag.asString.trim() else null
            }
            ?.filter { it.isNotBlank() }
            .orEmpty()

        return when {
            intent != null && tags.isNotEmpty() -> normalizeSummary("$intent • ${tags.joinToString(", ")}")
            intent != null -> normalizeSummary(intent)
            else -> null
        }
    }

    private fun JsonArray.extractSummaryFromComponents(): String? {
        var fallbackText: String? = null
        for (entry in this) {
            val msg = entry.asJsonObjectOrNull() ?: continue
            val components =
                msg.getAsJsonObjectOrNull("updateComponents")
                    ?.getAsJsonArrayOrNull("components")
                    ?: continue

            for (component in components) {
                val componentObj = component.asJsonObjectOrNull() ?: continue
                if (!componentObj.getString("component").equals("Text", ignoreCase = true)) {
                    continue
                }
                val text = componentObj.getString("text").orEmpty()
                if (text.isBlank()) {
                    continue
                }
                if (fallbackText == null) {
                    fallbackText = text
                }

                val variant = componentObj.getString("variant")?.lowercase().orEmpty()
                if (variant.contains("body") || variant.contains("paragraph")) {
                    return normalizeSummary(text)
                }
            }
        }
        return fallbackText?.let(::normalizeSummary)
    }

    private fun normalizeSummary(raw: String): String {
        val cleanedLine = raw
            .replace("â€¢", " ")
            .replace("•", " ")
            .replace("\r", "\n")
            .lineSequence()
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .firstOrNull { line ->
                line.length >= 24 &&
                    !line.startsWith("-") &&
                    !line.contains("|")
            }
            ?: raw

        return cleanedLine
            .replace(Regex("\\s+"), " ")
            .trim()
            .take(220)
    }

    private fun JsonArray.firstSurfaceId(): String? {
        for (entry in this) {
            val msg = entry.asJsonObjectOrNull() ?: continue
            val create = msg.getAsJsonObjectOrNull("createSurface")
            val update = msg.getAsJsonObjectOrNull("updateComponents")
            val begin = msg.getAsJsonObjectOrNull("beginRendering")
            val surfaceId =
                create?.getString("surfaceId")
                    ?: update?.getString("surfaceId")
                    ?: begin?.getString("surfaceId")
            if (!surfaceId.isNullOrBlank()) {
                return surfaceId
            }
        }
        return null
    }

    private fun JsonElement.asJsonObjectOrNull(): JsonObject? =
        if (isJsonObject) asJsonObject else null

    private fun JsonObject.getAsJsonArrayOrNull(key: String): JsonArray? =
        get(key)?.takeIf { it.isJsonArray }?.asJsonArray

    private fun JsonObject.getAsJsonObjectOrNull(key: String): JsonObject? =
        get(key)?.takeIf { it.isJsonObject }?.asJsonObject

    private fun JsonObject.getString(key: String): String? {
        val value = get(key) ?: return null
        return if (value.isJsonPrimitive && value.asJsonPrimitive.isString) value.asString else null
    }
}

