package com.samsung.genuicraft

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GeminiModelSettingsTest {
    @Test
    fun invalidGemmaResponseSelectionMigratesToVerifiedVertexFlash() {
        assertEquals(
            GeminiModelSettings.DEFAULT_RESPONSE_MODEL,
            GeminiModelSettings.normalizeVertexExpressResponseModel("gemma-4-31b-it"),
        )
        assertEquals("gemini-3.5-flash-lite", GeminiModelSettings.DEFAULT_RESPONSE_MODEL)
    }

    @Test
    fun unavailableResponseSelectionsMigrateToCurrentVertexDefault() {
        assertEquals(
            GeminiModelSettings.DEFAULT_RESPONSE_MODEL,
            GeminiModelSettings.normalizeVertexExpressResponseModel("gemini-2.5-flash-lite"),
        )
        assertEquals(
            GeminiModelSettings.DEFAULT_RESPONSE_MODEL,
            GeminiModelSettings.normalizeVertexExpressResponseModel("gemini-2.5-flash"),
        )
        assertEquals(
            GeminiModelSettings.DEFAULT_RESPONSE_MODEL,
            GeminiModelSettings.normalizeVertexExpressResponseModel("gemini-flash-latest"),
        )
    }

    @Test
    fun unavailableIrSelectionsMigrateToCurrentVertexDefault() {
        assertEquals("gemini-3.5-flash", GeminiModelSettings.DEFAULT_IR_MODEL)
        assertEquals(
            GeminiModelSettings.DEFAULT_IR_MODEL,
            GeminiModelSettings.normalizeVertexExpressIrModel("gemini-2.5-pro"),
        )
        assertEquals(
            GeminiModelSettings.DEFAULT_IR_MODEL,
            GeminiModelSettings.normalizeVertexExpressIrModel("gemini-pro-latest"),
        )
    }

    @Test
    fun vertexCompatibleGeminiSelectionIsPreserved() {
        assertEquals(
            "gemini-3.5-flash",
            GeminiModelSettings.normalizeVertexExpressResponseModel("models/gemini-3.5-flash"),
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
        assertFalse(options.any { it.startsWith("gemini-2.5-") })
        assertFalse(options.any { it.endsWith("-latest") })
        assertFalse(options.contains("gemini-3-flash-preview"))
        assertFalse(options.contains("gemini-3-pro-preview"))
        assertFalse(options.contains("gemini-2.0-flash"))
        assertFalse(options.contains("gemini-2.0-flash-lite"))
    }
}
