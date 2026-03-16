package com.samsung.genuicraft.pipeline

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PipelineMediaSanitizerTest {
    @Test
    fun ensureFlightListContent_injectsComparisonTable_whenMissing() {
        val response = """
            Flights from BLR to LKO on March 15 include options from IndiGo and Air India.
            IndiGo 06:15 AM to 08:50 AM, 2h 35m, Non-stop, INR 6,212.
            Air India 10:15 AM to 12:50 PM, 2h 35m, Non-stop, INR 6,266.
        """.trimIndent()

        val transformed = PipelineMediaSanitizer.ensureFlightListContent(
            responseText = response,
            queryText = "best flights from BLR to LKO on march 15 with fares and stops"
        )

        assertTrue(transformed.contains("Airline | Departure | Arrival | Duration | Stops | Fare"))
        assertTrue(transformed.contains("IndiGo"))
        assertTrue(transformed.contains("Air India"))
        assertTrue(PipelineMediaSanitizer.responseContainsFlightList(transformed))
    }

    @Test
    fun ensureFlightListContent_preservesExistingFlightTable() {
        val response = """
            Flight Comparison
            Airline | Departure | Arrival | Duration | Stops | Fare
            IndiGo | 06:15 AM | 08:50 AM | 2h 35m | Non-stop | INR 6,212
            Air India | 10:15 AM | 12:50 PM | 2h 35m | Non-stop | INR 6,266
        """.trimIndent()

        val transformed = PipelineMediaSanitizer.ensureFlightListContent(
            responseText = response,
            queryText = "best flights from BLR to LKO on march 15 with fares and stops"
        )

        assertEquals(response, transformed)
    }
}
