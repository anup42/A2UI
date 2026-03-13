package com.samsung.genuicraft

import android.content.Context
import android.util.Log
import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser

object IrDemoRecordRepository {
    private const val PREFS_NAME = "ir_demo_records_store"
    private const val KEY_RECORDS_JSON = "records_json"
    private const val KEY_INITIALIZED = "initialized"
    private const val TAG = "IrDemoRecordRepo"

    fun ensureInitialized(context: Context, defaultRecords: List<IrDemoRecord>): List<IrDemoRecord> {
        val existing = load(context)
        if (existing != null) {
            return existing
        }
        save(context, defaultRecords)
        return defaultRecords
    }

    fun load(context: Context): List<IrDemoRecord>? {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val initialized = prefs.getBoolean(KEY_INITIALIZED, false)
        val raw = prefs.getString(KEY_RECORDS_JSON, null)?.trim().orEmpty()
        if (!initialized && raw.isBlank()) {
            return null
        }
        if (raw.isBlank()) {
            return emptyList()
        }
        return runCatching {
            parseRecords(raw)
        }.onFailure {
            Log.w(TAG, "Failed to parse stored IR records", it)
        }.getOrDefault(emptyList())
    }

    fun save(context: Context, records: List<IrDemoRecord>) {
        val payload = JsonArray().apply {
            records.forEach { record ->
                add(JsonObject().apply {
                    addProperty("query_id", record.queryId)
                    addProperty("response_id", record.responseId)
                    addProperty("query_text", record.queryText)
                    addProperty("response_text", record.responseText)
                })
            }
        }.toString()

        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_INITIALIZED, true)
            .putString(KEY_RECORDS_JSON, payload)
            .apply()
    }

    fun addSavedResponse(
        context: Context,
        queryText: String,
        responseText: String
    ): IrDemoRecord? {
        val normalizedQuery = queryText.trim()
        val normalizedResponse = responseText.trim()
        if (normalizedQuery.isBlank() || normalizedResponse.isBlank()) {
            return null
        }

        val timestamp = System.currentTimeMillis()
        val record = IrDemoRecord(
            queryId = "saved_q_$timestamp",
            responseId = "saved_r_$timestamp",
            queryText = normalizedQuery,
            responseText = normalizedResponse
        )
        val updated = mutableListOf(record)
        updated += load(context).orEmpty()
        save(context, updated)
        return record
    }

    private fun parseRecords(raw: String): List<IrDemoRecord> {
        val root = JsonParser.parseString(raw)
        if (!root.isJsonArray) {
            return emptyList()
        }
        return root.asJsonArray.mapNotNull { element ->
            val obj = element.takeIf { it.isJsonObject }?.asJsonObject ?: return@mapNotNull null
            val queryId = obj.getString("query_id")?.trim().orEmpty()
            val queryText = obj.getString("query_text")?.trim().orEmpty()
            val responseText = obj.getString("response_text")?.trim().orEmpty()
            if (queryId.isBlank() || queryText.isBlank() || responseText.isBlank()) {
                return@mapNotNull null
            }
            IrDemoRecord(
                queryId = queryId,
                responseId = obj.getString("response_id")?.trim()?.takeIf { it.isNotBlank() },
                queryText = queryText,
                responseText = responseText
            )
        }
    }

    private fun JsonObject.getString(key: String): String? {
        val value = get(key) ?: return null
        return if (value.isJsonPrimitive && value.asJsonPrimitive.isString) value.asString else null
    }
}

