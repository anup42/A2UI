package com.samsung.genuicraft.sdk.internal.renderer.native.intents.train

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class NativeTrainSemanticsTest {
    @Test
    fun buildTrainRows_mapsRecoveredComparisonWithoutCorrectingGeneratedValues() {
        val rows = NativeTrainSemantics.buildTrainRows(
            headers = listOf(
                "service",
                "departure_station",
                "typical_departure_time",
                "journey_duration",
                "seating_class"
            ),
            rows = listOf(
                listOf(
                    "Shatabdi Express (120007)",
                    "KSR Bengaluru (BC)",
                    "10:35–1004",
                    "2 h15 m to 2h25 m",
                    "CC, EC"
                ),
                listOf(
                    "Rajya Rani Express (20606)",
                    "KSR Bengaluru (BC)",
                    "11:000",
                    "2h30 m",
                    "N/A"
                )
            ),
            title = "Train comparison"
        )

        assertNotNull(rows)
        val first = rows!!.first()
        assertEquals(NativeTrainField("Service", "Shatabdi Express (120007)"), first.service)
        assertEquals(NativeTrainField("Departure station", "KSR Bengaluru (BC)"), first.station)
        assertEquals(NativeTrainField("Typical departure time", "10:35–1004"), first.departure)
        assertEquals(NativeTrainField("Journey duration", "2 h15 m to 2h25 m"), first.duration)
        assertEquals(NativeTrainField("Seating class", "CC, EC"), first.seating)
        assertEquals("11:000", rows[1].departure?.value)
        assertEquals("N/A", rows[1].seating?.value)
    }

    @Test
    fun buildTrainRows_supportsLegacyAliasesAndPreservesDuplicateRows() {
        val sourceRow = listOf("Wodeyar Express", "KSR Bengaluru", "15:5", "2h30 m", "CC,2S GN")
        val rows = NativeTrainSemantics.buildTrainRows(
            headers = listOf("service", "departure", "departure_time", "duration", "class"),
            rows = listOf(sourceRow, sourceRow),
            title = "train_comparison"
        )

        assertEquals(2, rows?.size)
        assertEquals(rows?.get(0), rows?.get(1))
        assertEquals("Departure station", rows?.first()?.station?.label)
        assertEquals("Departure time", rows?.first()?.departure?.label)
        assertEquals("Journey duration", rows?.first()?.duration?.label)
        assertEquals("Seating class", rows?.first()?.seating?.label)
    }

    @Test
    fun buildTrainRows_mapsReorderedCamelCaseHeadersAndRetainsRaggedExtrasWithUnits() {
        val rows = NativeTrainSemantics.buildTrainRows(
            headers = listOf(
                "TypicalDepartureTime",
                "TrainService",
                "Platform",
                "Journey duration (minutes)",
                "Departure_Station",
                "seatingClass",
                "Fare (INR)"
            ),
            rows = listOf(
                listOf("08:7", "Morning Express", "4", "127", "City Central", "CC", "₹700"),
                listOf("09:00", "Morning Express", "N/A", "130", "City Central", "2S")
            )
        )

        assertNotNull(rows)
        val first = rows!!.first()
        assertEquals(NativeTrainField("Train service", "Morning Express"), first.service)
        assertEquals(NativeTrainField("Departure station", "City Central"), first.station)
        assertEquals(NativeTrainField("Typical departure time", "08:7"), first.departure)
        assertEquals(NativeTrainField("Journey duration (minutes)", "127"), first.duration)
        assertEquals(NativeTrainField("Seating class", "CC"), first.seating)
        assertEquals(
            listOf(
                NativeTrainField("Platform", "4"),
                NativeTrainField("Fare (INR)", "₹700")
            ),
            first.extraFields
        )
        assertEquals(listOf(NativeTrainField("Platform", "N/A")), rows[1].extraFields)
    }

    @Test
    fun buildTrainRows_acceptsExplicitRailHeadersWithoutTitle() {
        val rows = NativeTrainSemantics.buildTrainRows(
            headers = listOf("Train name", "Origin station", "Scheduled departure", "Travel time", "Coach class"),
            rows = listOf(listOf("Intercity", "Central", "06:30", "3 h", "CC"))
        )

        assertEquals("Intercity", rows?.single()?.service?.value)
        assertEquals("Central", rows?.single()?.station?.value)
        assertEquals("06:30", rows?.single()?.departure?.value)
    }

    @Test
    fun buildTrainRows_rejectsGenericAndConflictingDomainTables() {
        val genericHeaders = listOf("service", "departure", "departure_time", "duration", "class")
        val genericRows = listOf(listOf("Option A", "Central", "10:00", "2 h", "Standard"))
        assertNull(NativeTrainSemantics.buildTrainRows(genericHeaders, genericRows, title = "Comparison"))

        listOf("Airline", "Temperature", "Product name").forEach { conflictingHeader ->
            assertNull(
                NativeTrainSemantics.buildTrainRows(
                    headers = genericHeaders + conflictingHeader,
                    rows = listOf(genericRows.single() + "conflicting value"),
                    title = "Train comparison"
                )
            )
        }
    }

    @Test
    fun buildTrainRows_rejectsWholeTableWhenAnyRowCannotMapRequiredFields() {
        val headers = listOf("Train service", "Departure station", "Departure time", "Duration")
        val valid = listOf("Express A", "Central", "10:00", "2 h")

        assertNull(
            NativeTrainSemantics.buildTrainRows(
                headers = headers,
                rows = listOf(valid, listOf("", "Central", "11:00", "2 h")),
                title = "Train comparison"
            )
        )
        assertNull(
            NativeTrainSemantics.buildTrainRows(
                headers = headers,
                rows = listOf(valid, listOf("Express B")),
                title = "Train comparison"
            )
        )
    }
}
