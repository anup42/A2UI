package com.samsung.genuicraft

import android.content.Context

object InferenceBackendSettings {
    private const val PREFS_NAME = "inference_backend_settings"
    private const val KEY_PROVIDER = "provider"
    private const val KEY_RESPONSE_PROVIDER = "response_provider"
    private const val KEY_IR_PROVIDER = "ir_provider"
    private const val KEY_LOCAL_SERVER_BASE_URL = "local_server_base_url"
    private const val KEY_LOCAL_MODEL_PATH = "local_model_path"

    const val DEFAULT_LOCAL_SERVER_BASE_URL = "http://10.0.2.2:8000"
    const val DEFAULT_LOCAL_MODEL_PATH = "Qwen/Qwen2.5-Coder-7B-Instruct"

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
}
