package com.samsung.genuicraft.renderer.native

import org.junit.Assert.assertEquals
import org.junit.Test

class NativeTextFormatterTest {
    @Test
    fun sanitizeDisplayText_formatsIsoDateAsReadableShortDate() {
        val actual = NativeTextFormatter.sanitizeDisplayText("Forecast date: 2026-06-06")

        assertEquals("Forecast date: Jun 6", actual)
    }

    @Test
    fun sanitizeDisplayText_formatsIsoDatetimeAsReadableDateTime() {
        val actual = NativeTextFormatter.sanitizeDisplayText("Departure: 2026-06-06T14:30:00Z")

        assertEquals("Departure: Jun 6, 2:30 PM", actual)
    }

    @Test
    fun sanitizeDisplayText_formatsIsoDateRanges() {
        val actual = NativeTextFormatter.sanitizeDisplayText("Travel window: 2026-06-06 to 2026-06-08")

        assertEquals("Travel window: Jun 6 to Jun 8", actual)
    }

    @Test
    fun sanitizeDisplayText_doesNotRewriteDatesInsideUrls() {
        val url = "https://weather.test/report/2026-06-06?city=Bengaluru"
        val actual = NativeTextFormatter.sanitizeDisplayText(url)

        assertEquals(url, actual)
    }
}
