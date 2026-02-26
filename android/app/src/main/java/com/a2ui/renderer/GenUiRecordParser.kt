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
        val queryId = obj?.getString("query_id")
        val responseId = obj?.getString("response_id")

        return GenUiRecord(
            title = uiId ?: "Item $index",
            rawJson = element.toString(),
            sourceDir = sourceDir,
            sourceLabel = sourceLabel,
            uiId = uiId,
            queryId = queryId,
            responseId = responseId
        )
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

