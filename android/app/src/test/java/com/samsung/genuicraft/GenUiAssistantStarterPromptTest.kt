package com.samsung.genuicraft

import java.time.LocalDate
import org.junit.Assert.assertEquals
import org.junit.Test

class GenUiAssistantStarterPromptTest {
    @Test
    fun flightStarterPrompt_usesDateFifteenDaysFromCurrentDate() {
        val prompt = flightStarterPrompt(LocalDate.of(2026, 8, 2))

        assertEquals("Show flights from BLR to LKO on 17 August 2026", prompt)
    }

    @Test
    fun flightStarterPrompt_handlesYearRollover() {
        val prompt = flightStarterPrompt(LocalDate.of(2026, 12, 24))

        assertEquals("Show flights from BLR to LKO on 8 January 2027", prompt)
    }
}
