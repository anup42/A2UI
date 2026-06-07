package com.samsung.genuicraft

import android.content.Context
import android.util.Log
import java.io.File

object GeminiApiKeyProvider {
    private const val LOG_TAG = "GeminiApiKeyProvider"
    private const val INTERNAL_KEYS_FILE = "genuicraft_keys.env"
    private const val LEGACY_INTERNAL_KEYS_FILE = "gemini_keys.env"
    private const val EXTERNAL_KEYS_FILE = "genuicraft_keys.env"
    private const val LEGACY_EXTERNAL_KEYS_FILE = "gemini_keys.env"

    private val ioLock = Any()
    @Volatile
    private var cachedKeys: Map<String, String> = emptyMap()
    @Volatile
    private var cacheReady: Boolean = false

    fun stage2ApiKey(context: Context): String {
        // Stage 2 intentionally reuses Stage 3 key resolution so one active key serves both stages.
        return stage3ApiKey(context)
    }

    fun stage3ApiKey(context: Context): String {
        return firstNonBlank(
            resolveKey(
                context,
                "GEMINI_STAGE3_API_KEY",
                "GEMINI_IR_API_KEY",
                "GEMINI_API_KEY_2"
            ),
            BuildConfig.GEMINI_STAGE3_API_KEY_DEFAULT,
            BuildConfig.GEMINI_STAGE2_API_KEY_DEFAULT
        )
    }

    fun modelCatalogApiKey(context: Context): String {
        return firstNonBlank(stage2ApiKey(context), stage3ApiKey(context))
    }

    fun vertexExpressApiKey(context: Context): String {
        return firstNonBlank(
            resolveKey(
                context,
                "VERTEX_EXPRESS_API_KEY",
                "GEMINI_VERTEX_EXPRESS_API_KEY"
            ),
            BuildConfig.VERTEX_EXPRESS_API_KEY_DEFAULT
        )
    }

    fun googleMapsApiKey(context: Context): String {
        return firstNonBlank(
            resolveKey(
                context,
                "GOOGLE_MAPS_API_KEY",
                "GOOGLE_PLACES_API_KEY",
                "PLACES_API_KEY"
            ),
            BuildConfig.GOOGLE_MAPS_API_KEY_DEFAULT
        )
    }

    fun runtimeVertexOauthAccessToken(context: Context): String {
        return resolveKey(
            context,
            "VERTEX_OAUTH_ACCESS_TOKEN",
            "GOOGLE_OAUTH_ACCESS_TOKEN"
        )
    }

    fun vertexOauthAccessToken(context: Context): String {
        return firstNonBlank(
            runtimeVertexOauthAccessToken(context),
            BuildConfig.VERTEX_OAUTH_ACCESS_TOKEN_DEFAULT
        )
    }

    fun runtimeVertexProjectId(context: Context): String {
        return resolveKey(
            context,
            "VERTEX_PROJECT_ID",
            "GOOGLE_CLOUD_PROJECT",
            "GCP_PROJECT_ID"
        )
    }

    fun vertexProjectId(context: Context): String {
        return firstNonBlank(
            runtimeVertexProjectId(context),
            BuildConfig.VERTEX_PROJECT_ID_DEFAULT
        )
    }

    fun azureOpenAiApiKey(context: Context): String {
        return firstNonBlank(
            resolveKey(
                context,
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_SUBSCRIPTION_KEY"
            ),
            BuildConfig.AZURE_OPENAI_API_KEY_DEFAULT
        )
    }

    fun setupHintPath(context: Context): String {
        return "/sdcard/Android/data/${context.packageName}/files/$EXTERNAL_KEYS_FILE"
    }

    fun refresh(context: Context) {
        synchronized(ioLock) {
            importExternalKeysIfPresentLocked(context)
            cachedKeys = readCurrentKeysLocked(context)
            cacheReady = true
        }
    }

    private fun loadKeys(context: Context): Map<String, String> {
        if (cacheReady) {
            return cachedKeys
        }
        synchronized(ioLock) {
            if (cacheReady) {
                return cachedKeys
            }
            importExternalKeysIfPresentLocked(context)
            cachedKeys = readCurrentKeysLocked(context)
            cacheReady = true
            return cachedKeys
        }
    }

    private fun resolveKey(context: Context, vararg candidates: String): String {
        val first = firstNonBlank(
            *candidates.map { key -> loadKeys(context)[key] }.toTypedArray()
        )
        if (first.isNotBlank()) {
            return first
        }
        // If keys were pushed while app was running, refresh cache and retry once.
        refresh(context)
        return firstNonBlank(
            *candidates.map { key -> loadKeys(context)[key] }.toTypedArray()
        )
    }

    private fun importExternalKeysIfPresentLocked(context: Context) {
        val externalFile = externalCandidates(context).firstOrNull { it.exists() } ?: return
        val externalKeys = parseEnvFile(externalFile)
        if (externalKeys.isEmpty()) {
            runCatching { externalFile.delete() }
            return
        }

        val internalFile = internalPrimaryFile(context)
        val merged = readCurrentKeysLocked(context).toMutableMap().apply {
            putAll(externalKeys)
        }

        runCatching {
            val parent = internalFile.parentFile
            if (parent != null && !parent.exists()) {
                parent.mkdirs()
            }
            internalFile.writeText(
                merged.entries.joinToString(separator = "\n") { (key, value) -> "$key=$value" } + "\n"
            )
            externalFile.delete()
        }.onFailure {
            Log.w(LOG_TAG, "Failed to import external key file: ${it.message ?: it.javaClass.simpleName}")
        }
    }

    private fun readCurrentKeysLocked(context: Context): Map<String, String> {
        val internal = internalCandidates(context)
            .firstOrNull { it.exists() }
            ?.let(::parseEnvFile)
            .orEmpty()
        if (internal.isNotEmpty()) {
            return internal
        }
        return externalCandidates(context)
            .firstOrNull { it.exists() }
            ?.let(::parseEnvFile)
            .orEmpty()
    }

    private fun parseEnvFile(file: File): Map<String, String> {
        val values = linkedMapOf<String, String>()
        runCatching {
            file.forEachLine { raw ->
                val line = raw.trim()
                if (line.isEmpty() || line.startsWith("#")) {
                    return@forEachLine
                }
                val normalized = if (line.startsWith("export ")) {
                    line.removePrefix("export ").trim()
                } else {
                    line
                }
                val delimiter = normalized.indexOf('=')
                if (delimiter <= 0) {
                    return@forEachLine
                }
                val key = normalized.substring(0, delimiter).trim()
                if (key.isBlank()) {
                    return@forEachLine
                }
                var value = normalized.substring(delimiter + 1).trim()
                if (
                    (value.startsWith("\"") && value.endsWith("\"")) ||
                    (value.startsWith("'") && value.endsWith("'"))
                ) {
                    value = value.substring(1, value.length - 1)
                }
                if (value.isNotBlank()) {
                    values[key] = value
                }
            }
        }.onFailure {
            Log.w(LOG_TAG, "Failed reading key file ${file.name}: ${it.message ?: it.javaClass.simpleName}")
        }
        return values
    }

    private fun internalPrimaryFile(context: Context): File {
        return File(context.filesDir, INTERNAL_KEYS_FILE)
    }

    private fun internalCandidates(context: Context): List<File> {
        return listOf(
            File(context.filesDir, INTERNAL_KEYS_FILE),
            File(context.filesDir, LEGACY_INTERNAL_KEYS_FILE)
        )
    }

    private fun externalCandidates(context: Context): List<File> {
        val appExternal = context.getExternalFilesDir(null)
        return listOfNotNull(
            appExternal?.let { File(it, EXTERNAL_KEYS_FILE) },
            appExternal?.let { File(it, LEGACY_EXTERNAL_KEYS_FILE) }
        )
    }

    private fun firstNonBlank(vararg values: String?): String {
        return values.firstOrNull { !it.isNullOrBlank() }?.orEmpty() ?: ""
    }
}
