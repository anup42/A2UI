package com.samsung.genuicraft

import android.content.res.AssetManager
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser

object IrDemoRecordLoader {
    private const val DEFAULT_QUERIES_ASSET = "ir_demo_subset10_queries.jsonl"
    private const val DEFAULT_RESPONSES_ASSET = "ir_demo_subset10_responses.jsonl"

    fun loadDefault(assetManager: AssetManager): List<IrDemoRecord> {
        val queriesById = parseJsonlAsset(assetManager, DEFAULT_QUERIES_ASSET)
            .mapNotNull { element ->
                val obj = element.asJsonObjectOrNull() ?: return@mapNotNull null
                val queryId = obj.getString("query_id")?.trim().orEmpty()
                val queryText = decodeIrDemoQueryText(obj.getString("query_text").orEmpty())
                if (queryId.isBlank() || queryText.isBlank()) {
                    null
                } else {
                    queryId to queryText
                }
            }
            .toMap(linkedMapOf())

        return parseJsonlAsset(assetManager, DEFAULT_RESPONSES_ASSET)
            .mapNotNull { element ->
                val obj = element.asJsonObjectOrNull() ?: return@mapNotNull null
                val queryId = obj.getString("query_id")?.trim().orEmpty()
                val responseText = obj.getString("response_text")?.trim().orEmpty()
                if (queryId.isBlank() || responseText.isBlank()) {
                    return@mapNotNull null
                }
                val queryText = queriesById[queryId] ?: return@mapNotNull null
                IrDemoRecord(
                    queryId = queryId,
                    responseId = obj.getString("response_id")?.trim()?.takeIf { it.isNotBlank() },
                    queryText = queryText,
                    responseText = responseText
                )
            }
    }

    private fun parseJsonlAsset(assetManager: AssetManager, name: String): List<JsonElement> {
        val raw = assetManager.open(name).bufferedReader(Charsets.UTF_8).use { it.readText() }
        return raw
            .lineSequence()
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .mapNotNull { line -> runCatching { JsonParser.parseString(line) }.getOrNull() }
            .toList()
    }

    private fun JsonElement.asJsonObjectOrNull(): JsonObject? = if (isJsonObject) asJsonObject else null

    private fun JsonObject.getString(key: String): String? {
        val value = get(key) ?: return null
        return if (value.isJsonPrimitive && value.asJsonPrimitive.isString) value.asString else null
    }
}
