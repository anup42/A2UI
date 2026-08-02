package com.samsung.genuicraft

import org.junit.Assert.assertEquals
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
}
