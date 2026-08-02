package com.samsung.genuicraft

import android.content.Context
import java.util.Locale

object GeminiModelSettings {
    private const val PREFS_NAME = "gemini_model_settings"
    private const val KEY_SELECTED_MODEL = "selected_model"
    private const val KEY_RESPONSE_MODEL = "selected_response_model"
    private const val KEY_IR_MODEL = "selected_ir_model"
    const val DEFAULT_RESPONSE_MODEL = "gemini-2.5-flash-lite"
    const val DEFAULT_IR_MODEL = "gemini-2.5-flash"
    const val GEMMA_4_31B_IT_MODEL = "gemma-4-31b-it"
    const val DEFAULT_MODEL = DEFAULT_RESPONSE_MODEL

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
        val normalized = normalizeModelName(stored.orEmpty().trim())
        return normalized.ifBlank { DEFAULT_IR_MODEL }
    }

    fun setResponseModel(context: Context, model: String) {
        val normalized = normalizeVertexExpressResponseModel(model)
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_RESPONSE_MODEL, normalized)
            .apply()
    }

    fun setIrModel(context: Context, model: String) {
        val normalized = normalizeModelName(model).ifBlank { DEFAULT_IR_MODEL }
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
        return normalized.takeIf(::isVertexExpressCompatibleModel) ?: DEFAULT_RESPONSE_MODEL
    }

    fun isVertexExpressCompatibleModel(value: String): Boolean {
        return normalizeModelName(value)
            .lowercase(Locale.US)
            .startsWith("gemini-")
    }
}
