package com.samsung.genuicraft

import android.content.Context

object InferenceBackendSettings {
    private const val PREFS_NAME = "inference_backend_settings"
    private const val KEY_PROVIDER = "provider"
    private const val KEY_RESPONSE_PROVIDER = "response_provider"
    private const val KEY_IR_PROVIDER = "ir_provider"
    private const val KEY_GEMINI_API_MODE = "gemini_api_mode"
    private const val KEY_VERTEX_PROJECT_ID = "vertex_project_id"
    private const val KEY_VERTEX_LOCATION = "vertex_location"
    private const val KEY_VERTEX_ACCESS_TOKEN = "vertex_access_token"
    private const val KEY_LOCAL_SERVER_BASE_URL = "local_server_base_url"
    private const val KEY_LOCAL_MODEL_PATH = "local_model_path"

    const val DEFAULT_LOCAL_SERVER_BASE_URL = "http://10.0.2.2:8000"
    const val DEFAULT_LOCAL_MODEL_PATH = "Qwen/Qwen2.5-Coder-7B-Instruct"
    private const val FALLBACK_VERTEX_PROJECT_ID = "gen-lang-client-0741138863"
    const val DEFAULT_VERTEX_LOCATION = "us-central1"

    enum class Provider(val rawValue: String) {
        GEMINI("gemini"),
        LOCAL_SERVER("local_server");

        companion object {
            fun fromRawValue(value: String?): Provider {
                val normalized = value?.trim().orEmpty()
                return entries.firstOrNull { it.rawValue.equals(normalized, ignoreCase = true) } ?: GEMINI
            }
        }
    }

    enum class GeminiApiMode(val rawValue: String) {
        AI_STUDIO_DIRECT("ai_studio_direct"),
        VERTEX_AI_OAUTH("vertex_ai_oauth"),
        VERTEX_AI_EXPRESS_API_KEY("vertex_ai_express_api_key");

        companion object {
            fun fromRawValue(value: String?): GeminiApiMode {
                val normalized = value?.trim().orEmpty()
                if (normalized.equals("vertex_ai_express", ignoreCase = true)) {
                    // Backward compatibility with the previous mode name.
                    return VERTEX_AI_EXPRESS_API_KEY
                }
                return entries.firstOrNull { it.rawValue.equals(normalized, ignoreCase = true) }
                    ?: VERTEX_AI_EXPRESS_API_KEY
            }
        }
    }

    fun getProvider(context: Context): Provider {
        return getResponseProvider(context)
    }

    fun setProvider(context: Context, provider: Provider) {
        setResponseProvider(context, provider)
        setIrProvider(context, provider)
    }

    fun getResponseProvider(context: Context): Provider {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val responseRaw = prefs.getString(KEY_RESPONSE_PROVIDER, null)
        if (!responseRaw.isNullOrBlank()) {
            return Provider.fromRawValue(responseRaw)
        }
        val legacy = prefs.getString(KEY_PROVIDER, Provider.GEMINI.rawValue)
        return Provider.fromRawValue(legacy)
    }

    fun setResponseProvider(context: Context, provider: Provider) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_RESPONSE_PROVIDER, provider.rawValue)
            .putString(KEY_PROVIDER, provider.rawValue)
            .apply()
    }

    fun getIrProvider(context: Context): Provider {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val irRaw = prefs.getString(KEY_IR_PROVIDER, null)
        if (!irRaw.isNullOrBlank()) {
            return Provider.fromRawValue(irRaw)
        }
        val legacy = prefs.getString(KEY_PROVIDER, Provider.GEMINI.rawValue)
        return Provider.fromRawValue(legacy)
    }

    fun setIrProvider(context: Context, provider: Provider) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_IR_PROVIDER, provider.rawValue)
            .apply()
    }

    fun getGeminiApiMode(context: Context): GeminiApiMode {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val raw = prefs.getString(
            KEY_GEMINI_API_MODE,
            GeminiApiMode.VERTEX_AI_EXPRESS_API_KEY.rawValue
        )
        return GeminiApiMode.fromRawValue(raw)
    }

    fun setGeminiApiMode(context: Context, mode: GeminiApiMode) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_GEMINI_API_MODE, mode.rawValue)
            .apply()
    }

    fun getVertexProjectId(context: Context): String {
        val defaultProjectId = defaultVertexProjectId()
        val runtimeProjectId = GeminiApiKeyProvider.runtimeVertexProjectId(context).trim()
        if (runtimeProjectId.isNotBlank()) {
            return runtimeProjectId
        }
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_VERTEX_PROJECT_ID, defaultProjectId)
            .orEmpty()
            .trim()
            .ifBlank { defaultProjectId }
    }

    fun setVertexProjectId(context: Context, value: String) {
        val normalized = value.trim().ifBlank { defaultVertexProjectId() }
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_VERTEX_PROJECT_ID, normalized)
            .apply()
    }

    fun getVertexLocation(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_VERTEX_LOCATION, DEFAULT_VERTEX_LOCATION)
            .orEmpty()
            .trim()
            .ifBlank { DEFAULT_VERTEX_LOCATION }
    }

    fun setVertexLocation(context: Context, value: String) {
        val normalized = value.trim().ifBlank { DEFAULT_VERTEX_LOCATION }
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_VERTEX_LOCATION, normalized)
            .apply()
    }

    fun getVertexAccessToken(context: Context): String {
        val runtimeToken = GeminiApiKeyProvider.runtimeVertexOauthAccessToken(context).trim()
        if (runtimeToken.isNotBlank()) {
            return runtimeToken
        }
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val stored = prefs.getString(KEY_VERTEX_ACCESS_TOKEN, null)
            .orEmpty()
            .trim()
        if (stored.isNotBlank()) {
            return stored
        }
        return BuildConfig.VERTEX_OAUTH_ACCESS_TOKEN_DEFAULT.trim()
    }

    fun setVertexAccessToken(context: Context, value: String) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_VERTEX_ACCESS_TOKEN, value.trim())
            .apply()
    }

    fun getLocalServerBaseUrl(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val stored = prefs.getString(KEY_LOCAL_SERVER_BASE_URL, DEFAULT_LOCAL_SERVER_BASE_URL).orEmpty()
        return normalizeBaseUrl(stored).ifBlank { DEFAULT_LOCAL_SERVER_BASE_URL }
    }

    fun setLocalServerBaseUrl(context: Context, value: String) {
        val normalized = normalizeBaseUrl(value).ifBlank { DEFAULT_LOCAL_SERVER_BASE_URL }
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_LOCAL_SERVER_BASE_URL, normalized)
            .apply()
    }

    fun getLocalModelPath(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_LOCAL_MODEL_PATH, DEFAULT_LOCAL_MODEL_PATH)
            .orEmpty()
            .trim()
            .ifBlank { DEFAULT_LOCAL_MODEL_PATH }
    }

    fun setLocalModelPath(context: Context, value: String) {
        val normalized = value.trim().ifBlank { DEFAULT_LOCAL_MODEL_PATH }
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_LOCAL_MODEL_PATH, normalized)
            .apply()
    }

    private fun normalizeBaseUrl(value: String): String {
        var normalized = value.trim()
        if (normalized.isBlank()) {
            return ""
        }
        if (!normalized.startsWith("http://", ignoreCase = true) &&
            !normalized.startsWith("https://", ignoreCase = true)
        ) {
            normalized = "http://$normalized"
        }
        return normalized.trimEnd('/')
    }

    private fun defaultVertexProjectId(): String {
        return BuildConfig.VERTEX_PROJECT_ID_DEFAULT.trim().ifBlank { FALLBACK_VERTEX_PROJECT_ID }
    }
}
