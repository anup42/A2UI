package com.samsung.genuicraft

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract
import android.util.Base64
import android.util.Log
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonPrimitive
import com.google.gson.JsonParser
import java.io.ByteArrayOutputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale

object IrDemoRecordRepository {
    private const val PREFS_NAME = "ir_demo_records_store"
    private const val KEY_RECORDS_JSON = "records_json"
    private const val KEY_INITIALIZED = "initialized"
    private const val TAG = "IrDemoRecordRepo"
    private const val MAX_SAVED_RESPONSE_CHARS = 120_000
    private const val MAX_RECORDS = 200
    private const val BUNDLE_FORMAT = "genuicraft.scenario.bundle"
    private const val BUNDLE_VERSION = 1
    private const val MAX_EMBEDDED_ASSETS = 16
    private const val MAX_EMBEDDED_ASSET_BYTES = 6 * 1024 * 1024

    data class ExportBundleResult(
        val fileName: String,
        val embeddedAssetCount: Int,
        val failedAssetCount: Int
    )

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

    fun exportDemoArtifactBundle(
        context: Context,
        folderUri: Uri,
        queryText: String,
        responseText: String,
        genUiJson: String
    ): ExportBundleResult {
        val normalizedQuery = decodeIrDemoQueryText(queryText)
        val normalizedResponse = responseText.trim()
        val normalizedIr = normalizeJsonText(genUiJson)
        require(normalizedQuery.isNotBlank()) { "Query is empty." }
        require(normalizedResponse.isNotBlank()) { "Response is empty." }
        require(normalizedIr.isNotBlank()) { "GenUI JSON is empty." }

        val timestamp = System.currentTimeMillis()
        val responseId = "export_$timestamp"
        val collectedAssets = collectAssetObjects(normalizedIr, normalizedResponse)
        val embeddedAssets = mutableListOf<JsonObject>()
        val replacements = linkedMapOf<String, String>()
        var failedAssetCount = 0

        collectedAssets
            .take(MAX_EMBEDDED_ASSETS)
            .forEachIndexed { index, asset ->
                val originalPath = asset.getString("path")?.trim().orEmpty()
                val source = asset.getString("url")?.trim().orEmpty().ifBlank { originalPath }
                if (source.isBlank()) {
                    return@forEachIndexed
                }
                val blob = runCatching { readAssetBlob(context, source) }.getOrNull()
                if (blob == null) {
                    failedAssetCount += 1
                    return@forEachIndexed
                }
                val localPath = buildPortableAssetPath(source, index, blob.mimeType)
                addReplacementVariants(replacements, source, localPath)
                addReplacementVariants(replacements, originalPath, localPath)
                embeddedAssets += JsonObject().apply {
                    addProperty("path", localPath)
                    addProperty("source", source)
                    addProperty("mime_type", blob.mimeType)
                    addProperty("encoding", "base64")
                    addProperty("data", Base64.encodeToString(blob.bytes, Base64.NO_WRAP))
                    addProperty("bytes", blob.bytes.size)
                }
            }

        val rewrittenIr = rewriteJsonAssetReferences(parseJsonOrString(normalizedIr), replacements).toString()
        val rewrittenResponse = rewriteTextAssetReferences(normalizedResponse, replacements)
        val assetsJson = buildLocalAssetsArray(embeddedAssets).toString()
        val record = JsonObject().apply {
            addProperty("query_id", "exported_q_$timestamp")
            addProperty("response_id", responseId)
            addProperty("query_text", normalizedQuery)
            addProperty("response_text", rewrittenResponse)
            add("genui_json", parseJsonOrString(rewrittenIr))
            add("assets", parseJsonOrString(assetsJson))
        }
        val bundle = JsonObject().apply {
            addProperty("format", BUNDLE_FORMAT)
            addProperty("version", BUNDLE_VERSION)
            addProperty("created_at_epoch_ms", timestamp)
            addProperty("asset_policy", "embedded_base64")
            add("record", record)
            add("asset_blobs", JsonArray().apply { embeddedAssets.forEach(::add) })
        }

        val fileName = buildExportFileName(normalizedQuery, timestamp)
        val outputUri = DocumentsContract.createDocument(
            context.contentResolver,
            folderUri,
            "application/json",
            fileName
        ) ?: error("Could not create export file.")
        context.contentResolver.openOutputStream(outputUri, "wt")?.use { output ->
            output.bufferedWriter(Charsets.UTF_8).use { writer ->
                writer.write(bundle.toString())
            }
        } ?: error("Could not open export file.")

        return ExportBundleResult(
            fileName = fileName,
            embeddedAssetCount = embeddedAssets.size,
            failedAssetCount = failedAssetCount + (collectedAssets.size - MAX_EMBEDDED_ASSETS).coerceAtLeast(0)
        )
    }

    fun importDemoArtifactBundle(context: Context, uri: Uri): IrDemoRecord? {
        val raw = context.contentResolver.openInputStream(uri)
            ?.bufferedReader(Charsets.UTF_8)
            ?.use { it.readText() }
            ?.trim()
            .orEmpty()
        if (raw.isBlank()) {
            return null
        }
        val root = runCatching { JsonParser.parseString(raw) }.getOrNull()?.asJsonObjectOrNull() ?: return null
        if (root.getString("format") != BUNDLE_FORMAT) {
            return null
        }
        val recordObj = root.get("record")?.asJsonObjectOrNull() ?: return null
        val timestamp = System.currentTimeMillis()
        val responseId = recordObj.getString("response_id")?.trim()?.takeIf { it.isNotBlank() }
            ?: "import_$timestamp"
        val safeResponseId = sanitizeFileSegment(responseId).ifBlank { "import_$timestamp" }
        val assetDir = savedAssetDir(context, safeResponseId).apply { mkdirs() }
        val importedAssetBlobs = root.get("asset_blobs")?.takeIf { it.isJsonArray }?.asJsonArray ?: JsonArray()
        importedAssetBlobs.forEach { entry ->
            val asset = entry.asJsonObjectOrNull() ?: return@forEach
            val path = asset.getString("path")?.replace("\\", "/")?.trim().orEmpty()
            val data = asset.getString("data")?.trim().orEmpty()
            if (path.isBlank() || data.isBlank()) {
                return@forEach
            }
            val bytes = runCatching { Base64.decode(data, Base64.DEFAULT) }.getOrNull() ?: return@forEach
            val outFile = File(assetDir, path).toPath().normalize().toFile()
            if (!outFile.path.startsWith(assetDir.path)) {
                return@forEach
            }
            outFile.parentFile?.mkdirs()
            outFile.writeBytes(bytes)
        }

        val queryText = decodeIrDemoQueryText(recordObj.getString("query_text").orEmpty())
        val responseText = recordObj.getString("response_text")?.trim().orEmpty()
        val genUiJson = recordObj.getJsonText("genui_json")
            ?: recordObj.getJsonText("stage3_json")
            ?: recordObj.getJsonText("ir_json")
        if (queryText.isBlank() || responseText.isBlank() || genUiJson.isNullOrBlank()) {
            return null
        }
        val assetsJson = recordObj.getJsonText("assets") ?: buildLocalAssetsArray(
            importedAssetBlobs.mapNotNull { it.asJsonObjectOrNull() }
        ).toString()
        val record = IrDemoRecord(
            queryId = recordObj.getString("query_id")?.trim()?.takeIf { it.isNotBlank() } ?: "Imported",
            responseId = safeResponseId,
            queryText = queryText,
            responseText = responseText,
            genUiJson = normalizeJsonText(genUiJson),
            assetsJson = assetsJson,
            sourceDirPath = assetDir.absolutePath
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
        val values = collectAssetObjects(genUiJson, responseText)
        if (values.isEmpty()) {
            return null
        }
        return JsonArray().apply {
            values.forEach(::add)
        }.toString()
    }

    private fun collectAssetObjects(genUiJson: String, responseText: String): List<JsonObject> {
        val assets = linkedMapOf<String, JsonObject>()
        runCatching { JsonParser.parseString(genUiJson) }
            .getOrNull()
            ?.let { parsed ->
                collectExplicitAssets(parsed, assets)
                collectAssetReferences(parsed, assets)
            }
        collectResponseAssetReferences(responseText, assets)
        return assets.values.toList()
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
        "uri",
        "path",
        "image",
        "icon",
        "logo",
        "thumbnail",
        "media"
    )

    private data class AssetBlob(
        val bytes: ByteArray,
        val mimeType: String
    )

    private fun readAssetBlob(context: Context, source: String): AssetBlob? {
        val normalized = source.replace("\\", "/").trim()
        if (normalized.isBlank()) {
            return null
        }
        return when {
            normalized.startsWith("http://", ignoreCase = true) ||
                normalized.startsWith("https://", ignoreCase = true) -> readNetworkAssetBlob(normalized)

            normalized.startsWith("content:", ignoreCase = true) -> {
                val uri = runCatching { Uri.parse(normalized) }.getOrNull() ?: return null
                val mime = context.contentResolver.getType(uri).orEmpty()
                    .ifBlank { guessMimeType(normalized) }
                if (!isEmbeddableAsset(normalized, mime)) return null
                val bytes = context.contentResolver.openInputStream(uri)?.use(::readLimitedBytes) ?: return null
                AssetBlob(bytes, mime)
            }

            normalized.startsWith("file:", ignoreCase = true) -> {
                val uri = runCatching { Uri.parse(normalized) }.getOrNull() ?: return null
                val file = uri.path?.let(::File) ?: return null
                if (!file.exists()) return null
                val mime = guessMimeType(file.name)
                if (!isEmbeddableAsset(normalized, mime)) return null
                AssetBlob(file.readBytes().takeIf { it.size <= MAX_EMBEDDED_ASSET_BYTES } ?: return null, mime)
            }

            normalized.startsWith("/assets/") ||
                normalized.startsWith("assets/") ||
                normalized.startsWith("../assets/") ||
                normalized.startsWith("./assets/") -> {
                val assetPath = normalized
                    .removePrefix("/")
                    .removePrefix("./")
                    .removePrefix("../")
                    .removePrefix("assets/")
                val mime = guessMimeType(assetPath)
                if (!isEmbeddableAsset(normalized, mime)) return null
                val bytes = context.assets.open(assetPath).use(::readLimitedBytes)
                AssetBlob(bytes, mime)
            }

            else -> null
        }
    }

    private fun readNetworkAssetBlob(source: String): AssetBlob? {
        val connection = (URL(source).openConnection() as? HttpURLConnection) ?: return null
        return try {
            connection.connectTimeout = 8_000
            connection.readTimeout = 12_000
            connection.instanceFollowRedirects = true
            connection.setRequestProperty("User-Agent", "GenUICraft/1.0")
            val code = connection.responseCode
            if (code !in 200..299) {
                return null
            }
            val mime = connection.contentType?.substringBefore(";")?.trim().orEmpty()
                .ifBlank { guessMimeType(source) }
            if (!isEmbeddableAsset(source, mime)) {
                return null
            }
            val bytes = connection.inputStream.use(::readLimitedBytes)
            AssetBlob(bytes, mime)
        } finally {
            connection.disconnect()
        }
    }

    private fun readLimitedBytes(input: java.io.InputStream): ByteArray {
        val buffer = ByteArray(16 * 1024)
        val out = ByteArrayOutputStream()
        while (true) {
            val read = input.read(buffer)
            if (read <= 0) break
            out.write(buffer, 0, read)
            if (out.size() > MAX_EMBEDDED_ASSET_BYTES) {
                error("Asset is larger than ${MAX_EMBEDDED_ASSET_BYTES / (1024 * 1024)} MB.")
            }
        }
        return out.toByteArray()
    }

    private fun buildPortableAssetPath(source: String, index: Int, mimeType: String): String {
        val namePart = source
            .substringBefore('?')
            .substringAfterLast('/')
            .substringBeforeLast('.', missingDelimiterValue = "")
            .take(42)
        val base = sanitizeFileSegment(namePart).ifBlank { "asset_${index + 1}" }
        return "assets/${base}_${index + 1}.${assetExtension(source, mimeType)}"
    }

    private fun assetExtension(source: String, mimeType: String): String {
        val urlExt = source.substringBefore('?').substringAfterLast('.', "").lowercase(Locale.US)
        if (urlExt in setOf("jpg", "jpeg", "png", "webp", "gif", "svg")) {
            return urlExt
        }
        return when (mimeType.lowercase(Locale.US)) {
            "image/jpeg" -> "jpg"
            "image/png" -> "png"
            "image/webp" -> "webp"
            "image/gif" -> "gif"
            "image/svg+xml" -> "svg"
            else -> "bin"
        }
    }

    private fun guessMimeType(value: String): String {
        return when (value.substringBefore('?').substringAfterLast('.', "").lowercase(Locale.US)) {
            "jpg", "jpeg" -> "image/jpeg"
            "png" -> "image/png"
            "webp" -> "image/webp"
            "gif" -> "image/gif"
            "svg" -> "image/svg+xml"
            else -> "application/octet-stream"
        }
    }

    private fun isEmbeddableAsset(source: String, mimeType: String): Boolean {
        val ext = source.substringBefore('?').substringAfterLast('.', "").lowercase(Locale.US)
        return mimeType.lowercase(Locale.US).startsWith("image/") ||
            ext in setOf("jpg", "jpeg", "png", "webp", "gif", "svg")
    }

    private fun addReplacementVariants(replacements: MutableMap<String, String>, raw: String, replacement: String) {
        val normalized = raw.replace("\\", "/").trim()
        if (normalized.isBlank()) {
            return
        }
        val canonical = normalized.removePrefix("/").removePrefix("./").removePrefix("../")
        listOf(
            normalized,
            canonical,
            "/$canonical",
            "./$canonical",
            "../$canonical"
        ).filter { it.isNotBlank() }.forEach { replacements.putIfAbsent(it, replacement) }
    }

    private fun rewriteJsonAssetReferences(element: JsonElement, replacements: Map<String, String>): JsonElement {
        if (replacements.isEmpty()) {
            return element.deepCopy()
        }
        return when {
            element.isJsonPrimitive && element.asJsonPrimitive.isString -> {
                JsonPrimitive(replacements[element.asString] ?: element.asString)
            }
            element.isJsonArray -> JsonArray().apply {
                element.asJsonArray.forEach { add(rewriteJsonAssetReferences(it, replacements)) }
            }
            element.isJsonObject -> JsonObject().apply {
                element.asJsonObject.entrySet().forEach { (key, value) ->
                    add(key, rewriteJsonAssetReferences(value, replacements))
                }
            }
            else -> element.deepCopy()
        }
    }

    private fun rewriteTextAssetReferences(text: String, replacements: Map<String, String>): String {
        return replacements.entries.fold(text) { acc, (from, to) ->
            if (from.isBlank()) acc else acc.replace(from, to)
        }
    }

    private fun buildLocalAssetsArray(embeddedAssets: List<JsonObject>): JsonArray {
        return JsonArray().apply {
            embeddedAssets.forEach { asset ->
                val path = asset.getString("path")?.trim().orEmpty()
                if (path.isBlank()) return@forEach
                add(JsonObject().apply {
                    addProperty("path", path)
                    addProperty("url", path)
                    asset.getString("source")?.takeIf { it.isNotBlank() }?.let { addProperty("source_url", it) }
                    asset.getString("mime_type")?.takeIf { it.isNotBlank() }?.let { addProperty("mime_type", it) }
                    asset.get("bytes")?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isNumber }?.let {
                        add("bytes", it.deepCopy())
                    }
                })
            }
        }
    }

    private fun buildExportFileName(queryText: String, timestamp: Long): String {
        val slug = sanitizeFileSegment(queryText)
            .lowercase(Locale.US)
            .take(48)
            .trim('_')
            .ifBlank { "scenario" }
        return "genuicraft_${slug}_$timestamp.json"
    }

    private fun sanitizeFileSegment(value: String): String {
        return value
            .trim()
            .replace(Regex("""[^A-Za-z0-9._-]+"""), "_")
            .trim('_', '.', '-')
    }

    private fun JsonElement.asJsonObjectOrNull(): JsonObject? = if (isJsonObject) asJsonObject else null

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
