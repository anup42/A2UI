package com.samsung.genuicraft

import android.content.Context

object GeminiModelSettings {
    private const val PREFS_NAME = "gemini_model_settings"
    private const val KEY_SELECTED_MODEL = "selected_model"
    const val DEFAULT_MODEL = "gemini-2.5-pro"

    fun getSelectedModel(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val stored = prefs.getString(KEY_SELECTED_MODEL, DEFAULT_MODEL).orEmpty().trim()
        return normalizeModelName(stored).ifBlank { DEFAULT_MODEL }
    }

    fun setSelectedModel(context: Context, model: String) {
        val normalized = normalizeModelName(model).ifBlank { DEFAULT_MODEL }
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_SELECTED_MODEL, normalized)
            .apply()
    }

    fun normalizeModelName(value: String): String {
        val trimmed = value.trim()
        return if (trimmed.startsWith("models/")) {
            trimmed.removePrefix("models/").trim()
        } else {
            trimmed
        }
    }
}

