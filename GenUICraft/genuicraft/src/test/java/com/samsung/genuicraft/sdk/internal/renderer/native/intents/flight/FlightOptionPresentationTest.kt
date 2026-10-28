package com.samsung.genuicraft.sdk.internal.renderer.native.intents.flight

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiWireCodec
import com.samsung.genuicraft.sdk.internal.pipeline.EntityListPresentation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FlightOptionPresentationTest {
    @Test fun capturedMorningFlightListUsesFourNativeCardsAndKeepsAfternoonRowGeneric() {
        val rows = listOf(
            listOf("IndiGo 6E-6558", "04:30 departure, 06:55 arrival, non-stop, fare from ₹8,015."),
            listOf("Air India Express IX 1470", "06:05 departure, 08:35 arrival, fare not shown in the result snippet."),
            listOf("Akasa Air QP 1421", "06:45 departure, 09:30 arrival, non-stop, fare from ₹7,542.[1]"),
            listOf("SpiceJet SG-615", "10:20 departure, 12:50 arrival, non-stop, fare from ₹7,733."),
            listOf("IndiGo 6E-196", "12:40 departure; this is listed in the broader route results, but it is after the morning window."),
        )
        val native = requireNotNull(FlightOptionPresentation.buildFlightOptionRows(listOf("Option", "Details"), rows))
        assertEquals(5, native.size)
        (0..3).forEach { index ->
            val flight = requireNotNull(native[index])
            assertEquals(rows[index][0], flight.airline)
            assertEquals(listOf("Details" to rows[index][1]), flight.details)
            assertNull(flight.status)
        }
        assertEquals("04:30", native[0]?.depart)
        assertEquals("06:55", native[0]?.arrive)
        assertEquals("from ₹8,015", native[0]?.fare)
        assertEquals("non-stop", native[0]?.stops)
        assertEquals("06:05", native[1]?.depart)
        assertEquals("08:35", native[1]?.arrive)
        assertEquals("Fare not shown", native[1]?.fare)
        assertNull(native[1]?.stops)
        assertEquals("06:45", native[2]?.depart)
        assertEquals("09:30", native[2]?.arrive)
        assertEquals("from ₹7,542", native[2]?.fare)
        assertEquals("10:20", native[3]?.depart)
        assertEquals("12:50", native[3]?.arrive)
        assertEquals("from ₹7,733", native[3]?.fare)
        assertNull(native[4])
        assertTrue(rows[4][1].contains("after the morning window"))
    }

    @Test fun missingFareNeedsTwoNamedEndpointsAndConflictingPricesStayGeneric() {
        val rows = listOf(
            listOf("IndiGo 6E-6558", "04:30 departure, 06:55 arrival, fare from ₹8,015."),
            listOf("Air India Express IX 1470", "06:05 departure, 08:35 arrival, fare not shown."),
            listOf("Air India Express IX 1471", "06:05 departure; fare not shown."),
            listOf("Air India Express IX 1472", "06:05 departure, 08:35 arrival, fare not shown, but another listing says ₹8,100."),
            listOf("Akasa Air QP 1421", "06:45 departure, 09:30 arrival, 07:00 connection, fare ₹7,542."),
            listOf("Air India Express IX 1470 / QP 1421", "06:05 departure, 08:35 arrival, fare ₹8,100."),
        )
        val native = requireNotNull(FlightOptionPresentation.buildFlightOptionRows(listOf("Option", "Details"), rows))
        assertTrue(native[0] != null)
        assertTrue(native[1] != null)
        (2..5).forEach { assertNull(native[it]) }
    }

    @Test fun capturedFiveEntriesProduceTwoTimedFlightsTwoQualifiedFareCardsAndOneAggregatorFallback() {
        val (headers, rows) = capturedCards("LIVE-FLIGHT", "b")
        assertEquals(5, rows.size)
        val native = requireNotNull(FlightOptionPresentation.buildFlightOptionRows(headers, rows))
        assertEquals(rows.size, native.size)
        val akasa = requireNotNull(native[0])
        assertEquals("Akasa Air", akasa.airline)
        assertEquals("12:25", akasa.depart)
        assertEquals("15:05", akasa.arrive)
        assertEquals("₹7,195", akasa.fare)
        assertEquals("non-stop", akasa.stops)
        assertNull(akasa.originCode)
        assertNull(akasa.destinationCode)
        assertNull(akasa.duration)
        assertEquals(listOf("Details" to rows[0][1]), akasa.details)

        val airIndia = requireNotNull(native[1])
        assertEquals("Air India Express", airIndia.airline)
        assertEquals("17:05", airIndia.depart)
        assertEquals("19:50", airIndia.arrive)
        assertEquals("₹7,298", airIndia.fare)
        assertEquals(listOf("Details" to rows[1][1]), airIndia.details)

        val fareCalendar = requireNotNull(native[2])
        assertEquals("IndiGo", fareCalendar.airline)
        assertEquals("₹15,599", fareCalendar.fare)
        assertNull(fareCalendar.depart)
        assertNull(fareCalendar.arrive)
        assertTrue(fareCalendar.details.single().second.contains("for 15 Oct as listed in the fare calendar"))

        val fromListing = requireNotNull(native[3])
        assertEquals("Air India Express", fromListing.airline)
        assertEquals("from ₹22,268", fromListing.fare)
        assertNull(fromListing.depart)
        assertNull(fromListing.arrive)
        assertEquals(listOf("Details" to rows[3][1]), fromListing.details)

        assertNull(native[4])
        assertTrue(rows[4][1].contains("₹12,882"))
        assertTrue(rows[4][1].contains("₹10,898"))
        assertTrue(rows[4][1].contains("₹14,010"))
    }

    @Test fun capturedProductsCannotBeRecastAsFlightOffers() {
        val (headers, rows) = capturedCards("BXP-010", "d")
        assertEquals(3, rows.size)
        assertNull(FlightOptionPresentation.buildFlightOptionRows(headers, rows))
    }

    @Test fun qualifiersAndCitationsStayVerbatimInDetailsWithoutInferredDatesOrAirports() {
        val rows = listOf(
            listOf("Akasa Air", "about ₹7,195 one-way, departing 12:25 and arriving 15:05. [1][2]"),
            listOf("Air India Express", "fare listing shows from ₹22,268 for Thu, Oct 15 2026. [3]"),
        )
        val native = requireNotNull(FlightOptionPresentation.buildFlightOptionRows(listOf("Airline", "Details"), rows))
        assertEquals("about ₹7,195", native[0]?.fare)
        assertEquals("from ₹22,268", native[1]?.fare)
        native.forEachIndexed { index, flight ->
            assertNotNull(flight)
            assertEquals(listOf("Details" to rows[index][1]), flight?.details)
            assertNull(flight?.originCode)
            assertNull(flight?.destinationCode)
            assertNull(flight?.duration)
            assertNull(flight?.status)
        }
    }

    @Test fun ambiguousRowsStayGenericAtTheirOriginalIndex() {
        val rows = listOf(
            listOf("Akasa Air", "₹7,195 one-way, departing 12:25 and arriving 15:05."),
            listOf("Air India Express", "₹7,298 one-way, departing 17:05 and arriving 19:50."),
            listOf("ixigo", "lists ₹12,882 with a 05:55 → 08:35 flight at ₹10,898 and options from ₹14,010."),
            listOf("IndiGo", "₹15,599 fare calendar entry; departing 25:25 and arriving 19:50."),
            listOf("IndiGo", "₹15,599 fare calendar entry; 12:25 → 15:05."),
            listOf("IndiGo", "₹15,599 fare calendar entry; departing 12:25 but arrival unspecified."),
            listOf("IndiGo", "₹15,599 and ₹16,200 fares for the same date."),
            listOf("IndiGo", "₹6,207–7,174 fare range for the date."),
            listOf("IndiGo", "₹7,271+ fare lower bound for the date."),
            listOf("IndiGo", "₹15,599 fare; departing 12:25+? and arriving 15:05."),
        )
        val native = requireNotNull(FlightOptionPresentation.buildFlightOptionRows(listOf("Option", "Details"), rows))
        assertEquals(rows.size, native.size)
        assertNotNull(native[0])
        assertNotNull(native[1])
        (2 until rows.size).forEach { assertNull("row $it should use the generic card", native[it]) }
    }

    @Test fun explicitDayOffsetAndAdjacentFareCitationsArePreserved() {
        val rows = listOf(
            listOf("Akasa Air", "approx. ₹7,195[1][2] one-way, departing 23:40 and arriving 02:10 (+1)."),
            listOf("Air India Express", "around ₹7,298 fare, departing 17:05 and arriving 19:50."),
        )
        val native = requireNotNull(FlightOptionPresentation.buildFlightOptionRows(listOf("Option", "Details"), rows))
        assertEquals("approx. ₹7,195[1][2]", native[0]?.fare)
        assertEquals("23:40", native[0]?.depart)
        assertEquals("02:10+1", native[0]?.arrive)
        assertEquals("around ₹7,298", native[1]?.fare)
        assertEquals(listOf("Details" to rows[0][1]), native[0]?.details)
    }

    @Test fun requiresTwoClearAirlineOffersAndTheNarrowColumnShape() {
        val one = listOf(
            listOf("Akasa Air", "₹7,195 fare, departing 12:25 and arriving 15:05."),
            listOf("ixigo", "lists ₹12,882 fare for the date."),
        )
        assertNull(FlightOptionPresentation.buildFlightOptionRows(listOf("Option", "Details"), one))
        assertNull(FlightOptionPresentation.buildFlightOptionRows(listOf("Name", "Details"), one))
        assertNull(FlightOptionPresentation.buildFlightOptionRows(listOf("Option", "Details", "Price"),
            one.map { it + "₹7,195" }))
    }

    private fun capturedCards(caseId: String, componentId: String): Pair<List<String>, List<List<String>>> {
        val resource = requireNotNull(javaClass.getResource(
            "/entity_list_presentation/$caseId.repaired.a2ui.json"
        ))
        val graph: JsonObject = A2uiWireCodec.decode(JsonParser.parseString(resource.readText()))
        val normalized = EntityListPresentation.normalize(graph)
        assertEquals(1, normalized.convertedLists)
        val props = normalized.graph.getAsJsonObject("elements")
            .getAsJsonObject(componentId).getAsJsonObject("props")
        val headers = props.getAsJsonArray("columns").map { it.asString }
        val rows = props.getAsJsonArray("rows").map { raw -> raw.asJsonArray.map { it.asString } }
        return headers to rows
    }
}
