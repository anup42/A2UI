package com.samsung.genuicraft.pipeline

import android.content.Context
import android.util.Log
import com.google.gson.GsonBuilder
import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.net.UnknownHostException
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.time.Instant

internal class PipelineCacheManager(private val appContext: Context) {

    data class CacheSetupResult(
        val name: String?,
        val created: Boolean,
        val error: String?
    )

    data class CachedInstructionEntry(
        val hash: String,
        val name: String,
        val expiresAtMs: Long
    )

    data class CachedInstructionCreateResult(
        val name: String?,
        val expiresAtMs: Long?,
        val error: String?
    )

    data class LocalSystemPromptCachePrimeResult(
        val cacheHit: Boolean?,
        val error: String?
    )

    fun invalidateStage2InstructionCache(reason: String?) {
        synchronized(stage2CacheLock) {
            stage2InstructionCache = null
        }
        appContext.getSharedPreferences(CACHE_PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .remove(CACHE_KEY_STAGE2_HASH)
            .remove(CACHE_KEY_STAGE2_NAME)
            .remove(CACHE_KEY_STAGE2_EXPIRES_AT_MS)
            .apply()
        if (!reason.isNullOrBlank()) {
            Log.w(LOG_TAG, "Stage2 cache invalidated: $reason")
        }
    }

    fun invalidateStage3InstructionCache(reason: String?) {
        synchronized(stage3CacheLock) {
            stage3InstructionCache = null
        }
        appContext.getSharedPreferences(CACHE_PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .remove(CACHE_KEY_STAGE3_HASH)
            .remove(CACHE_KEY_STAGE3_NAME)
            .remove(CACHE_KEY_STAGE3_EXPIRES_AT_MS)
            .apply()
        if (!reason.isNullOrBlank()) {
            Log.w(LOG_TAG, "Stage3 cache invalidated: $reason")
        }
    }

    fun ensureStage2InstructionCache(
        apiKey: String,
        model: String,
        systemPrompt: String?
    ): CacheSetupResult {
        if (systemPrompt.isNullOrBlank()) {
            return CacheSetupResult(name = null, created = false, error = null)
        }

        val cacheScope = buildGeminiCacheScope(apiKey = apiKey, model = model)
        val promptHash = sha256Hex("$cacheScope\n$systemPrompt")
        val now = System.currentTimeMillis()

        synchronized(stage2CacheLock) {
            val inMemory = stage2InstructionCache
            if (inMemory != null && inMemory.hash == promptHash && inMemory.expiresAtMs > now + CACHE_EXPIRY_SAFETY_MS) {
                return CacheSetupResult(name = inMemory.name, created = false, error = null)
            }
        }

        readPersistedStage2Cache()?.let { persisted ->
            if (persisted.hash == promptHash && persisted.expiresAtMs > now + CACHE_EXPIRY_SAFETY_MS) {
                synchronized(stage2CacheLock) {
                    stage2InstructionCache = persisted
                }
                return CacheSetupResult(name = persisted.name, created = false, error = null)
            }
        }

        val created = createCachedInstruction(
            apiKey = apiKey,
            model = model,
            promptHash = promptHash,
            systemPrompt = systemPrompt,
            displayNamePrefix = "genuicraft_stage2"
        )
        if (created.name != null) {
            val expiresAtMs = created.expiresAtMs ?: now + STAGE_INSTRUCTION_CACHE_TTL_SECONDS * 1000L
            val entry = CachedInstructionEntry(
                hash = promptHash,
                name = created.name,
                expiresAtMs = expiresAtMs
            )
            synchronized(stage2CacheLock) {
                stage2InstructionCache = entry
            }
            persistStage2Cache(entry)
            return CacheSetupResult(name = entry.name, created = true, error = null)
        }
        return CacheSetupResult(name = null, created = false, error = created.error)
    }

    fun ensureStage3InstructionCache(
        apiKey: String,
        model: String,
        systemPrompt: String?
    ): CacheSetupResult {
        if (systemPrompt.isNullOrBlank()) {
            return CacheSetupResult(name = null, created = false, error = null)
        }

        val cacheScope = buildGeminiCacheScope(apiKey = apiKey, model = model)
        val promptHash = sha256Hex("$cacheScope\n$systemPrompt")
        val now = System.currentTimeMillis()

        synchronized(stage3CacheLock) {
            val inMemory = stage3InstructionCache
            if (inMemory != null && inMemory.hash == promptHash && inMemory.expiresAtMs > now + CACHE_EXPIRY_SAFETY_MS) {
                return CacheSetupResult(name = inMemory.name, created = false, error = null)
            }
        }

        readPersistedStage3Cache()?.let { persisted ->
            if (persisted.hash == promptHash && persisted.expiresAtMs > now + CACHE_EXPIRY_SAFETY_MS) {
                synchronized(stage3CacheLock) {
                    stage3InstructionCache = persisted
                }
                return CacheSetupResult(name = persisted.name, created = false, error = null)
            }
        }

        val created = createCachedInstruction(
            apiKey = apiKey,
            model = model,
            promptHash = promptHash,
            systemPrompt = systemPrompt,
            displayNamePrefix = "genuicraft_stage3"
        )
        if (created.name != null) {
            val expiresAtMs = created.expiresAtMs ?: now + STAGE_INSTRUCTION_CACHE_TTL_SECONDS * 1000L
            val entry = CachedInstructionEntry(
                hash = promptHash,
                name = created.name,
                expiresAtMs = expiresAtMs
            )
            synchronized(stage3CacheLock) {
                stage3InstructionCache = entry
            }
            persistStage3Cache(entry)
            return CacheSetupResult(name = entry.name, created = true, error = null)
        }
        return CacheSetupResult(name = null, created = false, error = created.error)
    }

    fun createCachedInstruction(
        apiKey: String,
        model: String,
        promptHash: String,
        systemPrompt: String,
        displayNamePrefix: String
    ): CachedInstructionCreateResult {
        val encodedKey = URLEncoder.encode(apiKey, StandardCharsets.UTF_8.name())
        val endpoint = URL("https://generativelanguage.googleapis.com/v1beta/cachedContents?key=$encodedKey")
        val connection = (endpoint.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 20000
            readTimeout = 60000
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
        }

        val body = JsonObject().apply {
            addProperty("model", "models/$model")
            addProperty("displayName", "${displayNamePrefix}_${promptHash.take(12)}")
            addProperty("ttl", "${STAGE_INSTRUCTION_CACHE_TTL_SECONDS}s")
            add("systemInstruction", JsonObject().apply {
                add("parts", JsonArray().apply {
                    add(JsonObject().apply {
                        addProperty("text", systemPrompt)
                    })
                })
            })
            // Keep one tiny content part so cache creation stays valid across API variants.
            add("contents", JsonArray().apply {
                add(JsonObject().apply {
                    addProperty("role", "user")
                    add("parts", JsonArray().apply {
                        add(JsonObject().apply {
                            addProperty("text", "Use the cached GenUICraft conversion instructions.")
                        })
                    })
                })
            })
        }

        return try {
            connection.outputStream.use { out ->
                out.write(gson.toJson(body).toByteArray(StandardCharsets.UTF_8))
            }
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val raw = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
            if (code !in 200..299) {
                val short = raw.trim().ifBlank { "HTTP $code" }
                return CachedInstructionCreateResult(
                    name = null,
                    expiresAtMs = null,
                    error = "HTTP $code: ${short.take(180)}"
                )
            }

            val root = runCatching { JsonParser.parseString(raw).asJsonObject }.getOrNull()
                ?: return CachedInstructionCreateResult(
                    name = null,
                    expiresAtMs = null,
                    error = "Cache create response was not valid JSON."
                )
            val name = root.get("name")?.takeIf { it.isJsonPrimitive }?.asString?.trim()
            if (name.isNullOrBlank()) {
                return CachedInstructionCreateResult(
                    name = null,
                    expiresAtMs = null,
                    error = "Cache create response did not include name."
                )
            }
            val expireTime = root.get("expireTime")?.takeIf { it.isJsonPrimitive }?.asString
            CachedInstructionCreateResult(
                name = name,
                expiresAtMs = parseExpireTimeMillis(expireTime),
                error = null
            )
        } catch (io: IOException) {
            CachedInstructionCreateResult(
                name = null,
                expiresAtMs = null,
                error = io.message ?: io.javaClass.simpleName
            )
        } finally {
            connection.disconnect()
        }
    }

    fun buildLocalSystemPromptCacheKey(systemPrompt: String?): String? {
        if (systemPrompt.isNullOrBlank()) {
            return null
        }
        return LOCAL_STAGE3_SYSTEM_PROMPT_CACHE_KEY
    }

    fun shouldSendLocalSystemPrompt(cacheKey: String?): Boolean {
        if (cacheKey.isNullOrBlank()) {
            return true
        }
        synchronized(localSystemPromptCacheLock) {
            return !localReadySystemPromptCacheKeys.contains(cacheKey)
        }
    }

    fun markLocalSystemPromptCacheKeyReady(cacheKey: String?) {
        if (cacheKey.isNullOrBlank()) {
            return
        }
        synchronized(localSystemPromptCacheLock) {
            localReadySystemPromptCacheKeys += cacheKey
        }
    }

    fun ensureLocalSystemPromptCache(
        localServerBaseUrl: String,
        localModelPath: String,
        cacheKey: String?,
        systemPrompt: String?
    ): LocalSystemPromptCachePrimeResult {
        val normalizedKey = cacheKey?.trim().orEmpty()
        val normalizedPrompt = systemPrompt?.trim().orEmpty()
        if (normalizedKey.isBlank() || normalizedPrompt.isBlank()) {
            return LocalSystemPromptCachePrimeResult(cacheHit = null, error = null)
        }

        val baseUrl = localServerBaseUrl.trim().trimEnd('/')
        if (baseUrl.isBlank()) {
            return LocalSystemPromptCachePrimeResult(
                cacheHit = null,
                error = "Local server URL is empty."
            )
        }

        val endpoint = URL("$baseUrl/v1/cache/system_prompt")
        val connection = (endpoint.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 5000
            readTimeout = 30000
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
        }

        val body = JsonObject().apply {
            addProperty("cache_key", normalizedKey)
            addProperty("system_prompt", normalizedPrompt)
            if (localModelPath.isNotBlank()) {
                addProperty("model_path", localModelPath)
            }
        }

        return try {
            connection.outputStream.use { out ->
                out.write(gson.toJson(body).toByteArray(StandardCharsets.UTF_8))
            }
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val raw = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
            if (code !in 200..299) {
                val short = raw.trim().ifBlank { "HTTP $code" }
                LocalSystemPromptCachePrimeResult(
                    cacheHit = null,
                    error = "HTTP $code: ${short.take(220)}"
                )
            } else {
                val cacheHit = extractLocalSystemPromptPrimeHit(raw)
                markLocalSystemPromptCacheKeyReady(normalizedKey)
                Log.i(LOG_TAG, "Local Stage3 KV prefix cache prime success: key=$normalizedKey cacheHit=$cacheHit")
                LocalSystemPromptCachePrimeResult(cacheHit = cacheHit, error = null)
            }
        } catch (unknownHost: UnknownHostException) {
            LocalSystemPromptCachePrimeResult(
                cacheHit = null,
                error = "Local server host is unreachable: ${unknownHost.message ?: "unknown host"}"
            )
        } catch (io: IOException) {
            LocalSystemPromptCachePrimeResult(
                cacheHit = null,
                error = io.message ?: io.javaClass.simpleName
            )
        } finally {
            connection.disconnect()
        }
    }

    fun extractLocalSystemPromptPrimeHit(raw: String): Boolean? {
        val root = runCatching { JsonParser.parseString(raw).asJsonObject }.getOrNull() ?: return null
        val value = root.get("cache_hit")
        if (value != null && value.isJsonPrimitive) {
            return runCatching { value.asBoolean }.getOrNull()
        }
        return null
    }

    private fun readPersistedStage2Cache(): CachedInstructionEntry? {
        val prefs = appContext.getSharedPreferences(CACHE_PREFS_NAME, Context.MODE_PRIVATE)
        val hash = prefs.getString(CACHE_KEY_STAGE2_HASH, null)?.trim().orEmpty()
        val name = prefs.getString(CACHE_KEY_STAGE2_NAME, null)?.trim().orEmpty()
        val expiresAt = prefs.getLong(CACHE_KEY_STAGE2_EXPIRES_AT_MS, 0L)
        if (hash.isBlank() || name.isBlank() || expiresAt <= 0L) {
            return null
        }
        return CachedInstructionEntry(hash = hash, name = name, expiresAtMs = expiresAt)
    }

    private fun readPersistedStage3Cache(): CachedInstructionEntry? {
        val prefs = appContext.getSharedPreferences(CACHE_PREFS_NAME, Context.MODE_PRIVATE)
        val hash = prefs.getString(CACHE_KEY_STAGE3_HASH, null)?.trim().orEmpty()
        val name = prefs.getString(CACHE_KEY_STAGE3_NAME, null)?.trim().orEmpty()
        val expiresAt = prefs.getLong(CACHE_KEY_STAGE3_EXPIRES_AT_MS, 0L)
        if (hash.isBlank() || name.isBlank() || expiresAt <= 0L) {
            return null
        }
        return CachedInstructionEntry(hash = hash, name = name, expiresAtMs = expiresAt)
    }

    private fun persistStage2Cache(entry: CachedInstructionEntry) {
        appContext.getSharedPreferences(CACHE_PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(CACHE_KEY_STAGE2_HASH, entry.hash)
            .putString(CACHE_KEY_STAGE2_NAME, entry.name)
            .putLong(CACHE_KEY_STAGE2_EXPIRES_AT_MS, entry.expiresAtMs)
            .apply()
    }

    private fun persistStage3Cache(entry: CachedInstructionEntry) {
        appContext.getSharedPreferences(CACHE_PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(CACHE_KEY_STAGE3_HASH, entry.hash)
            .putString(CACHE_KEY_STAGE3_NAME, entry.name)
            .putLong(CACHE_KEY_STAGE3_EXPIRES_AT_MS, entry.expiresAtMs)
            .apply()
    }

    private fun parseExpireTimeMillis(value: String?): Long? {
        if (value.isNullOrBlank()) {
            return null
        }
        return runCatching { Instant.parse(value).toEpochMilli() }.getOrNull()
    }

    private fun buildGeminiCacheScope(apiKey: String, model: String): String {
        // Keep cache identity tied to API key as well as model.
        // Cached content created under one key can return 403/permission errors under another key.
        val keyFingerprint = sha256Hex(apiKey.trim()).take(16)
        return "$model|key:$keyFingerprint"
    }

    private fun sha256Hex(value: String): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(value.toByteArray(StandardCharsets.UTF_8))
        val hex = StringBuilder(digest.size * 2)
        digest.forEach { byte ->
            val v = byte.toInt() and 0xff
            if (v < 16) hex.append('0')
            hex.append(v.toString(16))
        }
        return hex.toString()
    }

    companion object {
        const val CACHE_PREFS_NAME = "genui_stage_pipeline_cache"
        const val CACHE_KEY_STAGE2_HASH = "stage2_cache_hash"
        const val CACHE_KEY_STAGE2_NAME = "stage2_cache_name"
        const val CACHE_KEY_STAGE2_EXPIRES_AT_MS = "stage2_cache_expires_at_ms"
        const val CACHE_KEY_STAGE3_HASH = "stage3_cache_hash"
        const val CACHE_KEY_STAGE3_NAME = "stage3_cache_name"
        const val CACHE_KEY_STAGE3_EXPIRES_AT_MS = "stage3_cache_expires_at_ms"
        const val CACHE_EXPIRY_SAFETY_MS = 60_000L
        const val STAGE_INSTRUCTION_CACHE_TTL_SECONDS = 21600
        const val LOCAL_STAGE3_SYSTEM_PROMPT_CACHE_KEY = "stage3_ir_system_prompt_v5"
        const val LOG_TAG = "PipelineCacheManager"

        val stage2CacheLock = Any()
        val stage3CacheLock = Any()
        val localSystemPromptCacheLock = Any()
        @Volatile
        var stage2InstructionCache: CachedInstructionEntry? = null
        @Volatile
        var stage3InstructionCache: CachedInstructionEntry? = null
        val localReadySystemPromptCacheKeys = mutableSetOf<String>()
        val gson = GsonBuilder().disableHtmlEscaping().create()
    }
}
