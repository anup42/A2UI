package com.samsung.genuicraft.mcp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class WeatherForecastDurationTest {
    @Test
    fun intentClassifier_extractsWeatherLocationAndRequestedDays() {
        val result = McpIntentClassifier.classify("show weather in Bengaluru for next 15 days")

        assertEquals(McpSettings.Domain.WEATHER, result?.domain)
        assertEquals("bengaluru", result?.extractedEntities?.get("location"))
        assertEquals("15", result?.extractedEntities?.get("days"))
        assertNull(result?.extractedEntities?.get("date"))
    }

    @Test
    fun mcpClient_extractsForecastDaysFromRawQuery() {
        val method = McpClient::class.java.getDeclaredMethod("extractWeatherForecastDays", String::class.java)
        method.isAccessible = true

        assertEquals(15, method.invoke(McpClient, "show weather in Bengaluru for next 15 days"))
        assertEquals(14, method.invoke(McpClient, "Bengaluru 2-week forecast"))
    }

    @Test
    fun mcpClient_locationFallbackDropsForecastDurationSuffix() {
        val method = McpClient::class.java.getDeclaredMethod("extractLocationFallback", String::class.java)
        method.isAccessible = true

        assertEquals("Bengaluru", method.invoke(McpClient, "show weather in Bengaluru for next 15 days"))
        assertEquals("Bengaluru", method.invoke(McpClient, "show weather for next 15 days in Bengaluru"))
    }
}
