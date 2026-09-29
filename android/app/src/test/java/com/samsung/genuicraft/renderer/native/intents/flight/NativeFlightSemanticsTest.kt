package com.samsung.genuicraft.renderer.native.intents.flight

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeFlightSemanticsTest {
    @Test
    fun looksLikeFareValue_handlesInrAndDoesNotTreatTimeAsFare() {
        assertTrue(NativeFlightSemantics.looksLikeFareValue("INR 6,200"))
        assertTrue(NativeFlightSemantics.looksLikeFareValue("Rs 6200"))
        assertFalse(NativeFlightSemantics.looksLikeFareValue("06:15 AM"))
        assertFalse(NativeFlightSemantics.looksLikeFareValue("IndiGo 6E 6353"))
    }

    @Test
    fun splitFareDisplay_normalizesInrPrefix() {
        val (amount, suffix) = NativeFlightSemantics.splitFareDisplay("INR 6,212 /adult")
        assertEquals("\u20B96,212", amount)
        assertEquals("/adult", suffix)
    }

    @Test
    fun buildFlightRows_preservesLogoAndActionFields() {
        val rows = NativeFlightSemantics.buildFlightRows(
            header = listOf(
                "Airline",
                "Departure",
                "Arrival",
                "Duration",
                "Stops",
                "Fare",
                "Status",
                "Airline Logo",
                "Booking URL",
                "Action Label"
            ),
            body = listOf(
                listOf(
                    "IndiGo - 6E 356",
                    "BLR 07:15",
                    "LKO 15:05",
                    "7h 50m",
                    "1 stop",
                    "\u20B96,667 /adult",
                    "Best - Layover MAA 4h 15m",
                    "https://www.gstatic.com/flights/airline_logos/70px/6E.png",
                    "https://www.google.com/travel/flights?q=Flights%20from%20BLR%20to%20LKO",
                    "View fare"
                )
            )
        )

        assertEquals(1, rows?.size)
        val row = rows!!.first()
        assertEquals("https://www.gstatic.com/flights/airline_logos/70px/6E.png", row.logoUrl)
        assertEquals("https://www.google.com/travel/flights?q=Flights%20from%20BLR%20to%20LKO", row.actionUrl)
        assertEquals("View fare", row.actionLabel)
        assertEquals("1 stop", row.stops)
        assertEquals("BLR 07:15", row.depart)
        assertEquals("LKO 15:05", row.arrive)
    }

    @Test
    fun combinedTimeRange_preservesBothEndpointsAndExplicitDayOffsets() {
        listOf("-", "–", "—", "→", "->", "to").forEach { separator ->
            assertEquals("04:40" to "07:10", NativeFlightSemantics.parseFlightTimeRange("04:40 $separator 07:10"))
        }
        val overnight = NativeFlightSemantics.parseFlightTimeRange("BLR 11:40 PM – LKO 02:10 AM (+1)")!!
        assertEquals("BLR 11:40 PM", overnight.first)
        assertEquals("LKO 02:10 AM (+1)", overnight.second)
        assertEquals("11:40 PM", NativeFlightSemantics.normalizeFlightTime(overnight.first))
        assertEquals("02:10 AM+1", NativeFlightSemantics.normalizeFlightTime(overnight.second))
        assertEquals("LKO", NativeFlightSemantics.parseFlightPoint(overnight.second, null).code)
        assertEquals("23:40" to "02:10", NativeFlightSemantics.parseFlightTimeRange("23:40–02:10"))
    }

    @Test
    fun combinedTimeRange_doesNotGuessFromInvalidOrAmbiguousValues() {
        listOf("04:40", "04:40 or 07:10", "04:40 / 07:10", "04:40, 07:10", "25:40–07:10",
            "04:60–07:10", "13:40 PM–07:10 PM", "04:40–07:10–10:00", "2026-09-30", "4–7 hours"
        ).forEach { assertNull(it, NativeFlightSemantics.parseFlightTimeRange(it)) }
        assertNull(NativeFlightSemantics.normalizeStopLabel(null, null, "04:40", "07:10"))
    }

    @Test
    fun capturedBixbyCombinedFlights_preserveTimesAirlinesSharedDetailsAndCitations() {
        val headers = listOf("Flight Number", "Departure", "Details", "Fare", "Booking", "Date")
        val body = listOf(
            listOf("IndiGo 6E 6353", "04:40–07:10", "nonstop, approximately 2h 30m", "₹6,889", "[5][12]", "tomorrow"),
            listOf("Air India Express IX 2019", "10:30–13:05", "nonstop, approximately 2h 35m", "₹6,887", "[8]", "tomorrow"),
            listOf("Akasa Air", "07:25–10:00", "nonstop, approximately 2h 35m", "₹6,672", "[8]", "tomorrow")
        )
        assertTrue(NativeFlightSemantics.hasCombinedFlightTimes(headers, body))
        val rows = NativeFlightSemantics.buildFlightRows(headers, body)!!
        assertEquals(3, rows.size)
        rows.zip(body).forEach { (row, source) ->
            assertEquals(source[0], row.airline)
            assertEquals(source[1].substringBefore('–'), row.depart)
            assertEquals(source[1].substringAfter('–'), row.arrive)
            assertEquals(source[2], row.duration)
            assertEquals("Non-stop", NativeFlightSemantics.canonicalizeStopLabel(row.stops))
            assertEquals(source[3], row.fare)
            assertNull(row.actionUrl)
            assertTrue(row.details.contains("Sources" to source[4]))
            assertTrue(row.details.contains("Date" to "tomorrow"))
        }
    }

    @Test
    fun combinedTimeRange_supportsTimeHeaderAndKeepsExplicitArrival() {
        val headers = listOf("Airline", "Time", "Fare")
        val body = listOf(listOf("IndiGo", "23:40–02:10+1", "INR 6,200"))
        assertTrue(NativeFlightSemantics.hasCombinedFlightTimes(headers, body))
        val row = NativeFlightSemantics.buildFlightRows(headers, body)!!.single()
        assertEquals("23:40", row.depart)
        assertEquals("02:10+1", row.arrive)
        assertNull(row.stops)

        val explicit = NativeFlightSemantics.buildFlightRows(
            listOf("Airline", "Departure", "Arrival", "Fare"),
            listOf(listOf("IndiGo", "04:40–07:10", "07:20", "INR 6,200"))
        )!!.single()
        assertEquals("04:40", explicit.depart)
        assertEquals("07:20", explicit.arrive)
    }

    @Test
    fun combinedTimeRange_doesNotUseLayoverWindowAsFlightTimes() {
        val headers = listOf("Airline", "Layover", "Fare")
        val body = listOf(listOf("IndiGo", "04:40–07:10", "INR 6,200"))
        assertFalse(NativeFlightSemantics.hasCombinedFlightTimes(headers, body))
    }
}
