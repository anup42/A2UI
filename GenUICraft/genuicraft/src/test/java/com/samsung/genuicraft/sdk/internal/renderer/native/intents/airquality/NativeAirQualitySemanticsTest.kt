package com.samsung.genuicraft.sdk.internal.renderer.native.intents.airquality

import com.samsung.genuicraft.sdk.internal.renderer.FlatTableRenderMode
import com.samsung.genuicraft.sdk.internal.renderer.FlatTableShape
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.extractDirectTableModel
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.detectTableShape
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class NativeAirQualitySemanticsTest {
    private val headers = listOf("Detail", "Value")
    private val rows = listOf(
        listOf("Observation time", "8:30 AM IST"),
        listOf("Reported AQI scale", "AQI-IN 197"),
        listOf("Pollution category", "Poor"),
        listOf("Main pollutant", "PM2.5"),
    )

    @Test fun suppliedReadingAndContradictoryCategoryRemainExactWithoutThresholdInference() {
        val summary = NativeAirQualitySemantics.buildSummary(headers, rows)!!
        assertEquals(NativeAirQualityField("Reported AQI scale", "AQI-IN 197"), summary.reading)
        assertEquals(NativeAirQualityField("Pollution category", "Poor"), summary.fields[summary.categoryIndex!!])
        assertEquals(rows.map { NativeAirQualityField(it[0], it[1]) }, summary.fields)
        assertEquals(listOf(rows[0], rows[2], rows[3]).map { NativeAirQualityField(it[0], it[1]) }, summary.detailFields)
    }

    @Test fun absentAqiGenericStatusAndTimetableDoNotEnterTheAqiRoute() {
        assertNull(NativeAirQualitySemantics.buildSummary(headers, emptyList()))
        assertNull(NativeAirQualitySemantics.buildSummary(headers,
            listOf(listOf("AQI", "197"))))
        assertNull(NativeAirQualitySemantics.buildSummary(headers,
            listOf(listOf("AQI", ""), listOf("Pollution category", "Poor"))))
        assertNull(NativeAirQualitySemantics.buildSummary(headers,
            listOf(listOf("Observation time", "8:30 AM IST"), listOf("Status", "Pending"))))
        assertNull(NativeAirQualitySemantics.buildSummary(listOf("Time", "Status"), rows))
        assertNull(NativeAirQualitySemantics.buildSummary(listOf("Date", "AQI"),
            listOf(listOf("Monday", "197"), listOf("Tuesday", "200"))))
        assertNull(NativeAirQualitySemantics.buildSummary(headers,
            rows + listOf(listOf("Extra", "Value", "Unmapped fact"))))
    }

    @Test fun unknownFactsCitationsBlankCellsAndDuplicatesAreNotDiscarded() {
        val extra = listOf(
            listOf("Station", "Not provided [7]"),
            listOf("Source note", "Source wording including units and uncertainty. ".repeat(20) + "[14]"),
            listOf("Main pollutant", "PM2.5"),
            listOf("Observation time", "Not available"),
            listOf("AQI", "AQI-US 188 [9]"),
            listOf("Unspecified field", ""),
            listOf("", "Unknown supplied value"),
        )
        val summary = NativeAirQualitySemantics.buildSummary(headers, rows + extra)!!
        assertEquals((rows + extra).map { NativeAirQualityField(it[0], it[1]) }, summary.fields)
        assertEquals(rows.size + extra.size - 1, summary.detailFields.size)
        assertEquals("Not provided [7]", summary.detailFields[3].value)
        assertEquals("AQI-IN 197", summary.reading.value)
    }

    @Test fun onlyDetailValuePairsOverrideStatusShapeAndExplicitGridStillWins() {
        listOf("Detail", "Details").forEach { label ->
            assertEquals(FlatTableShape.KEY_VALUE, detectTableShape(listOf(label, "Value"), rows, "status"))
        }
        assertEquals(FlatTableShape.SCHEDULE_TIMELINE,
            detectTableShape(listOf("Detail", "Time"), listOf(listOf("Arrival", "08:30")), "schedule"))
        assertEquals(FlatTableShape.SCHEDULE_TIMELINE,
            detectTableShape(listOf("Time", "Status"), listOf(listOf("08:30", "Pending")), "status"))
        assertNull(NativeAirQualitySemantics.buildSummary(headers, rows, " TABLE "))
        val model = extractDirectTableModel(mapOf(
            "columns" to headers, "rows" to rows, "domain" to "status", "preferredPresentation" to "table",
        ), emptyMap(), compactScreen = true)!!
        assertEquals(FlatTableShape.KEY_VALUE, model.shape)
        assertEquals(FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL, model.renderMode)
        assertEquals(rows, model.rows)
    }
}
