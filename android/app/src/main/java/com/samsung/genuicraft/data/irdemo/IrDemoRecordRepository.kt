package com.samsung.genuicraft

import android.content.Context
import android.util.Log
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonPrimitive
import com.google.gson.JsonParser
import java.io.File
import java.util.Locale

object IrDemoRecordRepository {
    private const val PREFS_NAME = "ir_demo_records_store"
    private const val KEY_RECORDS_JSON = "records_json"
    private const val KEY_INITIALIZED = "initialized"
    private const val TAG = "IrDemoRecordRepo"
    private const val MAX_SAVED_RESPONSE_CHARS = 120_000
    private const val MAX_RECORDS = 200

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
                    record.genUiJson?.takeIf { it.isNotBlank() }?.let { rawJson ->
                        add("genui_json", parseJsonOrString(rawJson))
                    }
                    record.assetsJson?.takeIf { it.isNotBlank() }?.let { rawJson ->
                        add("assets", parseJsonOrString(rawJson))
                    }
                    record.sourceDirPath?.takeIf { it.isNotBlank() }?.let { path ->
                        addProperty("source_dir_path", path)
                    }
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
        val normalizedQuery = decodeIrDemoQueryText(queryText)
        val normalizedResponse = responseText.trim().take(MAX_SAVED_RESPONSE_CHARS)
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
        if (updated.size > MAX_RECORDS) {
            updated.subList(MAX_RECORDS, updated.size).clear()
        }
        save(context, updated)
        return record
    }

    fun addSavedDemoArtifact(
        context: Context,
        queryText: String,
        responseText: String,
        genUiJson: String
    ): IrDemoRecord? {
        val normalizedQuery = decodeIrDemoQueryText(queryText)
        val normalizedResponse = responseText.trim()
        val normalizedIr = normalizeJsonText(genUiJson)
        if (normalizedQuery.isBlank() || normalizedResponse.isBlank() || normalizedIr.isBlank()) {
            return null
        }

        val timestamp = System.currentTimeMillis()
        val responseId = "demo_$timestamp"
        val record = IrDemoRecord(
            queryId = "Demo",
            responseId = responseId,
            queryText = normalizedQuery,
            responseText = normalizedResponse,
            genUiJson = normalizedIr,
            assetsJson = collectAssetsJson(normalizedIr, normalizedResponse),
            sourceDirPath = savedAssetDir(context, responseId).absolutePath
        )
        val updated = mutableListOf(record)
        updated += load(context).orEmpty()
        if (updated.size > MAX_RECORDS) {
            updated.subList(MAX_RECORDS, updated.size).clear()
        }
        save(context, updated)
        return record
    }

    fun buildRenderablePayload(record: IrDemoRecord): String? {
        val rawIr = record.genUiJson?.trim().orEmpty()
        if (rawIr.isBlank()) {
            return null
        }
        val root = JsonObject().apply {
            addProperty("ui_id", record.responseId ?: record.queryId)
            addProperty("query_id", record.queryId)
            record.responseId?.let { addProperty("response_id", it) }
            addProperty("query_text", record.queryText)
            addProperty("response_text", record.responseText)
            add("genui_json", parseJsonOrString(rawIr))
            record.assetsJson?.takeIf { it.isNotBlank() }?.let { rawAssets ->
                add("assets", parseJsonOrString(rawAssets))
            }
        }
        return root.toString()
    }

    private fun parseRecords(raw: String): List<IrDemoRecord> {
        val root = JsonParser.parseString(raw)
        if (!root.isJsonArray) {
            return emptyList()
        }
        return root.asJsonArray.mapNotNull { element ->
            val obj = element.takeIf { it.isJsonObject }?.asJsonObject ?: return@mapNotNull null
            val queryId = obj.getString("query_id")?.trim().orEmpty()
            val queryText = decodeIrDemoQueryText(obj.getString("query_text").orEmpty())
            val responseText = obj.getString("response_text")?.trim().orEmpty()
            if (queryId.isBlank() || queryText.isBlank() || responseText.isBlank()) {
                return@mapNotNull null
            }
            IrDemoRecord(
                queryId = queryId,
                responseId = obj.getString("response_id")?.trim()?.takeIf { it.isNotBlank() },
                queryText = queryText,
                responseText = responseText,
                genUiJson = obj.getJsonText("genui_json")
                    ?: obj.getJsonText("stage3_json")
                    ?: obj.getJsonText("ir_json"),
                assetsJson = obj.getJsonText("assets"),
                sourceDirPath = obj.getString("source_dir_path")?.trim()?.takeIf { it.isNotBlank() }
            )
        }
    }

    private fun parseJsonOrString(value: String): JsonElement {
        return runCatching { JsonParser.parseString(value) }
            .getOrElse { JsonPrimitive(value) }
    }

    private fun normalizeJsonText(value: String): String {
        val trimmed = value.trim()
        if (trimmed.isBlank()) {
            return ""
        }
        return runCatching { JsonParser.parseString(trimmed).toString() }
            .getOrDefault(trimmed)
    }

    private fun collectAssetsJson(genUiJson: String, responseText: String): String? {
        val assets = linkedMapOf<String, JsonObject>()
        runCatching { JsonParser.parseString(genUiJson) }
            .getOrNull()
            ?.let { parsed ->
                collectExplicitAssets(parsed, assets)
                collectAssetReferences(parsed, assets)
            }
        collectResponseAssetReferences(responseText, assets)
        if (assets.isEmpty()) {
            return null
        }
        return JsonArray().apply {
            assets.values.forEach(::add)
        }.toString()
    }

    private fun collectExplicitAssets(element: JsonElement, assets: MutableMap<String, JsonObject>) {
        if (!element.isJsonObject) {
            return
        }
        val explicit = element.asJsonObject.get("assets")?.takeIf { it.isJsonArray }?.asJsonArray ?: return
        explicit.forEach { entry ->
            val asset = entry.takeIf { it.isJsonObject }?.asJsonObject ?: return@forEach
            val url = asset.getString("url")?.trim().orEmpty()
            val path = asset.getString("path")?.trim().orEmpty()
            if (url.isBlank() && path.isBlank()) {
                return@forEach
            }
            val key = (path.ifBlank { url }).replace("\\", "/")
            assets.putIfAbsent(
                key,
                JsonObject().apply {
                    if (url.isNotBlank()) addProperty("url", url)
                    if (path.isNotBlank()) addProperty("path", path)
                    asset.getString("sha256")?.takeIf { it.isNotBlank() }?.let { addProperty("sha256", it) }
                    asset.get("bytes")?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isNumber }?.let {
                        add("bytes", it.deepCopy())
                    }
                }
            )
        }
    }

    private fun collectAssetReferences(element: JsonElement, assets: MutableMap<String, JsonObject>) {
        when {
            element.isJsonObject -> {
                val obj = element.asJsonObject
                obj.entrySet().forEach { (key, value) ->
                    if (key.lowercase(Locale.US) in assetReferenceKeys) {
                        value.asStringOrNull()?.let { addAssetReference(it, assets) }
                    }
                    collectAssetReferences(value, assets)
                }
            }

            element.isJsonArray -> {
                element.asJsonArray.forEach { collectAssetReferences(it, assets) }
            }
        }
    }

    private fun collectResponseAssetReferences(text: String, assets: MutableMap<String, JsonObject>) {
        val patterns = listOf(
            Regex("""(?i)\b(?:Image|Icon)\s*=\s*(\S+)"""),
            Regex("""(?i)^\s*(?:Image|Icon)\s*:\s*(\S+)""", RegexOption.MULTILINE),
            Regex("""!\[[^\]]*]\(([^)\s]+)\)""")
        )
        patterns.forEach { pattern ->
            pattern.findAll(text).forEach { match ->
                addAssetReference(match.groupValues.getOrNull(1).orEmpty(), assets)
            }
        }
    }

    private fun addAssetReference(rawValue: String, assets: MutableMap<String, JsonObject>) {
        val value = rawValue.trim().trim('"', '\'', ',', ';', ')', ']', '}').replace("\\", "/")
        if (!looksLikeAssetReference(value)) {
            return
        }
        val path = if (value.startsWith("http://", ignoreCase = true) || value.startsWith("https://", ignoreCase = true)) {
            "remote/${value.substringAfterLast('/').substringBefore('?').ifBlank { value.hashCode().toString() }}"
        } else {
            value.removePrefix("/").removePrefix("./").removePrefix("../")
        }
        assets.putIfAbsent(
            path,
            JsonObject().apply {
                addProperty("url", value)
                addProperty("path", path)
            }
        )
    }

    private fun looksLikeAssetReference(value: String): Boolean {
        if (value.isBlank()) return false
        val lower = value.lowercase(Locale.US)
        return lower.startsWith("http://") ||
            lower.startsWith("https://") ||
            lower.startsWith("assets/") ||
            lower.startsWith("/assets/") ||
            lower.startsWith("../assets/") ||
            lower.startsWith("./assets/") ||
            lower.startsWith("file:") ||
            lower.startsWith("content:")
    }

    private val assetReferenceKeys = setOf(
        "url",
        "src",
        "source",
        "image",
        "icon",
        "logo",
        "thumbnail",
        "media"
    )

    private fun savedAssetDir(context: Context, responseId: String): File {
        return File(context.filesDir, "ir_demo_saved_assets/$responseId")
    }

    private fun JsonObject.getString(key: String): String? {
        val value = get(key) ?: return null
        return if (value.isJsonPrimitive && value.asJsonPrimitive.isString) value.asString else null
    }

    private fun JsonObject.getJsonText(key: String): String? {
        val value = get(key) ?: return null
        if (value.isJsonNull) return null
        return if (value.isJsonPrimitive && value.asJsonPrimitive.isString) {
            value.asString.trim().takeIf { it.isNotBlank() }
        } else {
            value.toString().trim().takeIf { it.isNotBlank() }
        }
    }

    private fun JsonElement.asStringOrNull(): String? {
        return if (isJsonPrimitive && asJsonPrimitive.isString) asString else null
    }
}
