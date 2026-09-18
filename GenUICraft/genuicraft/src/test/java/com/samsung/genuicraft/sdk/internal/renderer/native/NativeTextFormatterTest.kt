package com.samsung.genuicraft.sdk.internal.renderer.native

import org.junit.Assert.assertEquals
import org.junit.Test

class NativeTextFormatterTest {
    @Test
    fun sanitizeDisplayText_preservesIsoYearAndDate() {
        val actual = NativeTextFormatter.sanitizeDisplayText("Forecast date: 2026-06-06")

        assertEquals("Forecast date: 2026-06-06", actual)
    }

    @Test
    fun sanitizeDisplayText_preservesIsoTimeAndTimezone() {
        val actual = NativeTextFormatter.sanitizeDisplayText("Departure: 2026-06-06T14:30:00Z")

        assertEquals("Departure: 2026-06-06T14:30:00Z", actual)
    }

    @Test
    fun sanitizeDisplayText_preservesBothDateRangeYears() {
        val actual = NativeTextFormatter.sanitizeDisplayText("Travel window: 2026-06-06 to 2026-06-08")

        assertEquals("Travel window: 2026-06-06 to 2026-06-08", actual)
    }

    @Test
    fun sanitizeDisplayText_doesNotRewriteDatesInsideUrls() {
        val url = "https://weather.test/report/2026-06-06?city=Bengaluru"
        val actual = NativeTextFormatter.sanitizeDisplayText(url)

        assertEquals(url, actual)
    }

    @Test
    fun sourceCaveatsExamplesAndRetrievalStatementsRemainVisible() {
        val source = "Data as of 2026-09-18.\nFor example: this is an illustrative sample (demo).\nSearching live weather is unavailable."
        assertEquals(source, NativeTextFormatter.sanitizeDisplayText(source))
    }

    @Test
    fun punctuationAndInternalSpacingAreNotRewritten() {
        val source = "Value :  10\n\n\nNext value: -5%"
        assertEquals(source, NativeTextFormatter.sanitizeDisplayText(source))
    }
}
