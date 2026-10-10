package com.samsung.genuicraft.sdk.internal.renderer.native.intents.airquality

import androidx.compose.ui.graphics.Color
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class NativeAirQualityUiRendererPolicyTest {
    private fun summary(reading: String, category: String?): NativeAirQualitySummary {
        val fields = listOf(NativeAirQualityField("Reported AQI scale", reading)) +
            (category?.let { listOf(NativeAirQualityField("Pollution category", it)) } ?: emptyList())
        return NativeAirQualitySummary(fields, readingIndex = 0, categoryIndex = category?.let { 1 })
    }

    @Test fun `accent follows only the supplied category, never the AQI number`() {
        assertEquals(Color(0xFF8A4300), authoredAirQualityAccent(summary("AQI-IN 197", "Poor")))
        assertEquals(Color(0xFF8A4300), authoredAirQualityAccent(summary("AQI-IN 12", "Poor")))
        assertEquals(Color(0xFF1B6845), authoredAirQualityAccent(summary("AQI-IN 197", "Good")))
        assertNull(authoredAirQualityAccent(summary("AQI-IN 197", null)))
        assertNull(authoredAirQualityAccent(summary("AQI-IN 197", " ")))
    }

    @Test fun `unknown authored category keeps a neutral accent and its full value`() {
        val category = "Source-qualified category [13]"
        val airQuality = summary("AQI-IN 197 [4]", category)

        assertEquals(Color(0xFF315987), authoredAirQualityAccent(airQuality))
        assertEquals(category, airQuality.fields[airQuality.categoryIndex!!].value)
        assertEquals("AQI-IN 197 [4]", airQuality.reading.value)
    }
}
