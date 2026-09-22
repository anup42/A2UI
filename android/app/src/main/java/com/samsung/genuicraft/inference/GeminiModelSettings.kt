package com.samsung.genuicraft

import android.content.Context
import java.util.Locale

object GeminiModelSettings {
    private const val PREFS_NAME = "gemini_model_settings"
    private const val KEY_SELECTED_MODEL = "selected_model"
    private const val KEY_RESPONSE_MODEL = "selected_response_model"
    private const val KEY_IR_MODEL = "selected_ir_model"
    const val DEFAULT_RESPONSE_MODEL = "gemini-3.5-flash-lite"
    const val DEFAULT_IR_MODEL = "gemini-3.5-flash"
    const val GEMMA_4_31B_IT_MODEL = "gemma-4-31b-it"
    const val DEFAULT_MODEL = DEFAULT_RESPONSE_MODEL

    /**
     * Text-output models suitable for both response generation and GenUI IR conversion.
     * Keep explicit stable IDs ahead of preview, legacy, and auto-updating aliases so the
     * settings screen favors reproducible model selections.
     */
    val BUILT_IN_TEXT_MODEL_OPTIONS = listOf(
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview",
    )

    fun getSelectedModel(context: Context): String {
        return getResponseModel(context)
    }

    fun getResponseModel(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val stored = prefs.getString(KEY_RESPONSE_MODEL, null)
            ?: prefs.getString(KEY_SELECTED_MODEL, DEFAULT_RESPONSE_MODEL)
        val storedModel = normalizeModelName(stored.orEmpty().trim())
        val resolvedModel = normalizeVertexExpressResponseModel(storedModel)
        if (resolvedModel != storedModel) {
            prefs.edit().putString(KEY_RESPONSE_MODEL, resolvedModel).apply()
        }
        return resolvedModel
    }

    fun getIrModel(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val stored = prefs.getString(KEY_IR_MODEL, null)
            ?: prefs.getString(KEY_SELECTED_MODEL, DEFAULT_IR_MODEL)
        val storedModel = normalizeModelName(stored.orEmpty().trim())
        val resolvedModel = normalizeVertexExpressIrModel(storedModel)
        if (resolvedModel != storedModel) {
            prefs.edit().putString(KEY_IR_MODEL, resolvedModel).apply()
        }
        return resolvedModel
    }

    fun setResponseModel(context: Context, model: String) {
        val normalized = normalizeVertexExpressResponseModel(model)
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_RESPONSE_MODEL, normalized)
            .apply()
    }

    fun setIrModel(context: Context, model: String) {
        val normalized = normalizeVertexExpressIrModel(model)
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_IR_MODEL, normalized)
            .apply()
    }

    fun setSelectedModel(context: Context, model: String) {
        val normalized = normalizeVertexExpressResponseModel(model)
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_SELECTED_MODEL, normalized)
            .putString(KEY_RESPONSE_MODEL, normalized)
            .putString(KEY_IR_MODEL, normalized)
            .apply()
    }

    fun getResponseAndIrModels(context: Context): Pair<String, String> {
        val response = getResponseModel(context)
        val ir = getIrModel(context)
        return response to ir
    }

    fun getLegacyOrDefaultModel(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val stored = prefs.getString(KEY_SELECTED_MODEL, DEFAULT_MODEL).orEmpty().trim()
        return normalizeModelName(stored).ifBlank { DEFAULT_MODEL }
    }

    fun normalizeModelName(value: String): String {
        val trimmed = value.trim()
        return when {
            trimmed.startsWith("publishers/google/models/") ->
                trimmed.removePrefix("publishers/google/models/").trim()
            trimmed.startsWith("models/") ->
                trimmed.removePrefix("models/").trim()
            else -> trimmed
        }
    }

    internal fun normalizeVertexExpressResponseModel(value: String): String {
        val normalized = normalizeModelName(value)
        if (isUnavailableExpressSelection(normalized)) {
            return DEFAULT_RESPONSE_MODEL
        }
        return normalized.takeIf(::isVertexExpressCompatibleModel) ?: DEFAULT_RESPONSE_MODEL
    }

    internal fun normalizeVertexExpressIrModel(value: String): String {
        val normalized = normalizeModelName(value)
        if (isUnavailableExpressSelection(normalized)) {
            return DEFAULT_IR_MODEL
        }
        return normalized.takeIf(::isVertexExpressCompatibleModel) ?: DEFAULT_IR_MODEL
    }

    fun isVertexExpressCompatibleModel(value: String): Boolean {
        return normalizeModelName(value)
            .lowercase(Locale.US)
            .startsWith("gemini-")
    }

    private fun isUnavailableExpressSelection(value: String): Boolean {
        return value.lowercase(Locale.US) in setOf(
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-pro-latest",
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
        )
    }
}
