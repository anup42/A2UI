package com.samsung.genuicraft.sdk.internal.renderer.native.intents.flight

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FlightMissingFareSemanticsTest {
    private val headers = listOf("Airline", "Departure", "Arrival", "Stops", "Fare", "Status")

    @Test fun `declared fare and stops columns survive missing-value content scoring`() {
        val detected = requireNotNull(NativeFlightSemantics.detectFlightColumns(headers))
        val resolved = NativeFlightSemantics.resolveFlightColumns(detected, listOf(
            listOf("Air India Express IX 1470", "06:05", "08:35", "not shown", "N/A", "not shown"),
        ))
        assertEquals(3, resolved.stops)
        assertEquals(4, resolved.fare)
        assertEquals(5, resolved.status)
    }

    @Test fun `missing fare note does not become a stops or status badge`() {
        val flight = requireNotNull(NativeFlightSemantics.buildFlightRows(headers, listOf(
            listOf("Air India Express IX 1470", "06:05", "08:35", "", "N/A", "not shown"),
        ))).single()
        assertEquals("Air India Express IX 1470", flight.airline)
        assertEquals("06:05", flight.depart)
        assertEquals("08:35", flight.arrive)
        assertNull(flight.stops)
        assertNull(flight.status)
        assertEquals("Fare not shown", flight.fare)
        assertEquals("Fare not shown" to null, NativeFlightSemantics.splitFareDisplay(flight.fare))
        assertNull(NativeFlightSemantics.normalizeStopLabel(flight.stops, flight.status, flight.depart, flight.arrive))
        assertNull(NativeFlightSemantics.normalizeFlightStatus(flight.status, null))
    }

    @Test fun `explicit full fare note remains available without a route claim`() {
        val note = "fare not shown in the result snippet[11]"
        val flight = requireNotNull(NativeFlightSemantics.buildFlightRows(headers, listOf(
            listOf("Air India Express IX 1470", "06:05", "08:35", "not shown", "N/A", note),
        ))).single()
        assertEquals("Fare not shown", flight.fare)
        assertNull(flight.stops)
        assertNull(flight.status)
        assertEquals(listOf("Fare note" to note), flight.details)
        assertNull(NativeFlightSemantics.canonicalizeStopLabel("stops not shown"))
        assertNull(NativeFlightSemantics.normalizeFlightStatus(note, null))
    }

    @Test fun `explicit non stop and real delay survive an unavailable fare`() {
        val flight = requireNotNull(NativeFlightSemantics.buildFlightRows(headers, listOf(
            listOf("IndiGo 6E-6558", "04:30", "06:55", "non-stop", "N/A", "Delayed"),
        ))).single()
        assertEquals("non-stop", flight.stops)
        assertEquals("Delayed", flight.status)
        assertEquals("N/A", flight.fare)
        assertEquals("Non-stop", NativeFlightSemantics.normalizeStopLabel(
            flight.stops, flight.status, flight.depart, flight.arrive,
        ))
        assertEquals("Delayed", NativeFlightSemantics.normalizeFlightStatus(flight.status, "Non-stop"))
        assertEquals("Fare unavailable" to null, NativeFlightSemantics.splitFareDisplay(flight.fare))
    }

    @Test fun `an actual price is not replaced by a contradictory missing fare note`() {
        val note = "fare not shown in the result snippet[11]"
        val flight = requireNotNull(NativeFlightSemantics.buildFlightRows(headers, listOf(
            listOf("Air India Express IX 1470", "06:05", "08:35", "non-stop", "₹8,015", note),
        ))).single()
        assertEquals("₹8,015", flight.fare)
        assertEquals("non-stop", flight.stops)
        assertNull(flight.status)
        assertEquals(listOf("Fare note" to note), flight.details)
    }
}
