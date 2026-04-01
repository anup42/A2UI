package com.samsung.genuicraft

import android.content.Context

object GeminiModelSettings {
    private const val PREFS_NAME = "gemini_model_settings"
    private const val KEY_SELECTED_MODEL = "selected_model"
    private const val KEY_RESPONSE_MODEL = "selected_response_model"
    private const val KEY_IR_MODEL = "selected_ir_model"
    const val DEFAULT_MODEL = "gemini-2.5-pro"

    fun getSelectedModel(context: Context): String {
        return getResponseModel(context)
    }

    fun getResponseModel(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val stored = prefs.getString(KEY_RESPONSE_MODEL, null)
            ?: prefs.getString(KEY_SELECTED_MODEL, DEFAULT_MODEL)
        val normalized = normalizeModelName(stored.orEmpty().trim())
        return normalized.ifBlank { DEFAULT_MODEL }
    }

    fun getIrModel(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val stored = prefs.getString(KEY_IR_MODEL, null)
            ?: prefs.getString(KEY_SELECTED_MODEL, DEFAULT_MODEL)
        val normalized = normalizeModelName(stored.orEmpty().trim())
        return normalized.ifBlank { DEFAULT_MODEL }
    }

    fun setResponseModel(context: Context, model: String) {
        val normalized = normalizeModelName(model).ifBlank { DEFAULT_MODEL }
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_RESPONSE_MODEL, normalized)
            .apply()
    }

    fun setIrModel(context: Context, model: String) {
        val normalized = normalizeModelName(model).ifBlank { DEFAULT_MODEL }
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_IR_MODEL, normalized)
            .apply()
    }

    fun setSelectedModel(context: Context, model: String) {
        val normalized = normalizeModelName(model).ifBlank { DEFAULT_MODEL }
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
        return trimmed
            .removePrefix("publishers/google/models/")
            .removePrefix("models/")
            .trim()
    }
}
