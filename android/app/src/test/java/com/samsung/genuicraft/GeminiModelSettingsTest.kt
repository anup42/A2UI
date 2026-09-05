package com.samsung.genuicraft

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GeminiModelSettingsTest {
    @Test
    fun invalidGemmaResponseSelectionMigratesToVertexFlashLite() {
        assertEquals(
            GeminiModelSettings.DEFAULT_RESPONSE_MODEL,
            GeminiModelSettings.normalizeVertexExpressResponseModel("gemma-4-31b-it"),
        )
        assertEquals("gemini-2.5-flash-lite", GeminiModelSettings.DEFAULT_RESPONSE_MODEL)
    }

    @Test
    fun vertexCompatibleGeminiSelectionIsPreserved() {
        assertEquals(
            "gemini-2.5-pro",
            GeminiModelSettings.normalizeVertexExpressResponseModel("models/gemini-2.5-pro"),
        )
        assertTrue(GeminiModelSettings.isVertexExpressCompatibleModel("gemini-2.5-flash-lite"))
    }

    @Test
    fun builtInOptionsContainCurrentTextModelsAndExcludeRetiredModels() {
        val options = GeminiModelSettings.BUILT_IN_TEXT_MODEL_OPTIONS

        assertEquals("gemini-3.8-flash", options.first())
        assertTrue(options.contains("gemini-3.5-flash"))
        assertTrue(options.contains("gemini-3.5-flash-lite"))
        assertTrue(options.contains("gemini-3.1-flash-lite"))
        assertTrue(options.contains("gemini-3.1-pro-preview"))
        assertTrue(options.all(GeminiModelSettings::isVertexExpressCompatibleModel))
        assertFalse(options.contains("gemini-3-flash-preview"))
        assertFalse(options.contains("gemini-3-pro-preview"))
        assertFalse(options.contains("gemini-2.0-flash"))
        assertFalse(options.contains("gemini-2.0-flash-lite"))
    }
}
