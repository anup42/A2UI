package com.samsung.genuicraft.mcp

import org.junit.Assert.assertEquals
import org.junit.Test

class McpResponseFormatterTest {

    @Test
    fun normalizeForStage3_fixesCommonMojibakeTokens() {
        val input = "Temp: 33Ã‚Â°C Ã¢â‚¬â€ feels like 35Ã‚Â°C Â· steady"
        val output = McpResponseFormatter.normalizeForStage3(input)

        assertEquals("Temp: 33°C — feels like 35°C · steady", output)
    }
}

