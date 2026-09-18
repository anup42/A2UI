package com.samsung.genuicraft.sdk.internal.renderer.native.intents.flight

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeFlightSemanticsTest {
    @Test
    fun looksLikeFareValue_handlesInrAndDoesNotTreatTimeAsFare() {
        assertTrue(NativeFlightSemantics.looksLikeFareValue("INR 6,200"))
        assertTrue(NativeFlightSemantics.looksLikeFareValue("Rs 6200"))
        assertFalse(NativeFlightSemantics.looksLikeFareValue("06:15 AM"))
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
    }
}
