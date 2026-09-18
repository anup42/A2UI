package com.samsung.genuicraft.sdk.internal.renderer.native.intents.weather

import com.samsung.genuicraft.sdk.internal.renderer.native.ParsedMediaEntry
import com.samsung.genuicraft.sdk.internal.renderer.native.TextBlock
import com.samsung.genuicraft.sdk.internal.renderer.native.WeatherRow
import java.time.LocalDate
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class NativeWeatherSemanticsTest {
    @Test
    fun currentWeatherHeading_doesNotTreatBookNowAsWeather() {
        assertFalse(NativeWeatherSemantics.isCurrentWeatherHeading("Book Now"))
        assertFalse(NativeWeatherSemantics.isCurrentWeatherHeading("Book now and save more"))
    }

    @Test
    fun inferWeatherCondition_doesNotMisclassifyCleartrip() {
        val inferred =
            NativeWeatherSemantics.inferWeatherConditionFromTextPool(
                "Book with Cleartrip for the lowest fares from Bengaluru to Lucknow."
            )
        assertNull(inferred)
    }

    @Test
    fun buildWeatherRows_rejectsFlightTable() {
        val header = listOf("Airline", "Departure", "Arrival", "Duration", "Stops", "Fare")
        val body =
            listOf(
                listOf("IndiGo", "07:10", "09:50", "2h 40m", "Non-stop", "INR 5,818"),
                listOf("Air India Express", "10:15", "12:50", "2h 35m", "Non-stop", "INR 5,820")
            )

        val rows = NativeWeatherSemantics.buildWeatherRows(header, body)
        assertNull(rows)
    }

    @Test
    fun buildWeatherRows_preservesRainChanceAndClothingAdvice() {
        val header = listOf("Day", "Forecast", "Temp", "Rain Chance", "Clothing Advice")
        val body = listOf(
            listOf(
                "Saturday",
                "Cloudy with showers",
                "26\u00B0C / 21\u00B0C",
                "75%",
                "Light cotton, quick-dry shoes, and compact umbrella."
            )
        )

        val rows = NativeWeatherSemantics.buildWeatherRows(header, body)

        assertNotNull(rows)
        val row = rows!!.single()
        assertEquals("Cloudy with showers", row.condition)
        assertEquals("26\u00B0C / 21\u00B0C", normalizeDegreeArtifacts(row.temp))
        assertEquals("75%", row.metrics.first { it.first == "Rain Chance" }.second)
        assertEquals(
            "Light cotton, quick-dry shoes, and compact umbrella.",
            row.metrics.first { it.first == "What to wear" }.second
        )
    }

    @Test
    fun dateOnlyForecast_preservesSourceOrderAndDoesNotInferTodayFromWeekday() {
        val rows =
            NativeWeatherSemantics.buildWeatherRows(
                header = listOf("Date", "High", "Low", "Rain"),
                body =
                    listOf(
                        listOf("Wed, Sep 9", "31", "20", "57%"),
                        listOf("Thu, Sep 10", "30", "21", "55%"),
                        listOf("Fri, Sep 11", "30", "21", "55%"),
                    ),
            )

        assertNotNull(rows)
        val sourceRows = rows!!
        assertEquals(listOf("Wed, Sep 9", "Thu, Sep 10", "Fri, Sep 11"), sourceRows.map { it.period })
        assertEquals(sourceRows, NativeWeatherSemantics.orderWeatherRows(sourceRows))
        val friday = LocalDate.of(2026, 9, 18)
        assertFalse(sourceRows.any { NativeWeatherSemantics.isTodayWeatherRow(it, friday) })
    }

    @Test
    fun weatherTableCarriesCelsiusAndFahrenheitFromHeadersIntoBareCells() {
        val celsiusRows =
            NativeWeatherSemantics.buildWeatherRows(
                header = listOf("Date", "High (°C)", "Low (°C)", "Rain"),
                body = listOf(listOf("Wed, Sep 9", "31", "20", "57%")),
            )!!
        val fahrenheitRows =
            NativeWeatherSemantics.buildWeatherRows(
                header = listOf("Date", "High (°F)", "Low (°F)", "Rain"),
                body = listOf(listOf("Wed, Sep 9", "88", "68", "57%")),
            )!!

        assertEquals("31°C", celsiusRows.single().high)
        assertEquals("20°C", celsiusRows.single().low)
        assertEquals("31°C / 20°C", NativeWeatherSemantics.weatherTemperatureText(celsiusRows.single()))
        assertEquals("88°F", fahrenheitRows.single().high)
        assertEquals("68°F", fahrenheitRows.single().low)
        assertEquals("88°F / 68°F", NativeWeatherSemantics.weatherTemperatureText(fahrenheitRows.single()))
    }

    @Test
    fun cellTemperatureUnitsRemainAuthoritativeAndAreNotDuplicated() {
        val rows =
            NativeWeatherSemantics.buildWeatherRows(
                header = listOf("Date", "High (°C)", "Low (°C)", "Rain"),
                body = listOf(listOf("Wed, Sep 9", "88°F", "68°F", "57%")),
            )!!

        assertEquals("88°F", rows.single().high)
        assertEquals("68°F", rows.single().low)
        assertEquals("88°F / 68°F", NativeWeatherSemantics.weatherTemperatureText(rows.single()))
    }

    @Test
    fun todayDetection_acceptsOnlyAnExplicitLabelOrFullyQualifiedCurrentDate() {
        val today = LocalDate.of(2026, 9, 18)

        assertFalse(
            NativeWeatherSemantics.isTodayWeatherRow(
                WeatherRow("Fri, Sep 11", null, null, "30", "21", null, emptyList()),
                today,
            ),
        )
        assertEquals(
            true,
            NativeWeatherSemantics.isTodayWeatherRow(
                WeatherRow("Fri, Sep 18, 2026", null, null, "30", "21", null, emptyList()),
                today,
            ),
        )
        assertEquals(
            true,
            NativeWeatherSemantics.isTodayWeatherRow(
                WeatherRow("Today", null, null, "30", "21", null, emptyList()),
                today,
            ),
        )
    }

    @Test
    fun undatedKeyValueWeatherUsesANeutralLabel() {
        val rows =
            NativeWeatherSemantics.buildCurrentWeatherRowsFromKeyValueTable(
                header = listOf("Metric", "Value"),
                body =
                    listOf(
                        listOf("Condition", "Cloudy"),
                        listOf("Temperature", "28 C"),
                        listOf("Humidity", "65%"),
                    ),
            )

        assertEquals("Forecast", rows?.single()?.period)
        assertFalse(NativeWeatherSemantics.isTodayWeatherRow(rows!!.single(), LocalDate.of(2026, 9, 18)))
    }

    @Test
    fun buildCurrentWeatherDetails_handlesPlaceholderIconAndBulletMetrics() {
        val details = NativeWeatherSemantics.buildCurrentWeatherDetails(
            title = "Current Weather in Bengaluru",
            sectionBlocks = listOf(
                TextBlock.Paragraph("Bengaluru, Karnataka"),
                TextBlock.MediaCards(
                    listOf(
                        ParsedMediaEntry(
                            label = "Icon",
                            url = "Icon",
                            iconLike = true
                        )
                    )
                ),
                TextBlock.Bullets(
                    listOf(
                        "28\u00B0C | Partly Cloudy",
                        "Feels like: 30\u00B0C",
                        "Wind: 11 km/h | Humidity: 65% | UV Index: High"
                    )
                )
            ),
            fallbackCondition = "Partly Cloudy",
            additionalLines = emptyList()
        )

        assertNotNull(details)
        assertEquals("28\u00B0C", normalizeDegreeArtifacts(details?.temperature))
        assertEquals("30\u00B0C", normalizeDegreeArtifacts(details?.feelsLike))
        assertEquals("65%", details?.humidity)
        assertEquals("11 km/h", details?.wind)
    }

    @Test
    fun weatherTemperatureText_prefersSingleUnitAcrossHighAndLow() {
        val row = WeatherRow(
            period = "Today",
            date = null,
            condition = "Cloudy",
            high = "33\u00B0C / 92\u00B0F",
            low = "22\u00B0C / 72\u00B0F",
            temp = null,
            metrics = emptyList()
        )

        val display = NativeWeatherSemantics.weatherTemperatureText(row)

        assertEquals("33\u00B0C / 22\u00B0C", normalizeDegreeArtifacts(display))
    }

    @Test
    fun weatherTemperatureText_collapsesMalformedFourReadingCell() {
        val row = WeatherRow(
            period = "Today",
            date = null,
            condition = "Cloudy",
            high = null,
            low = null,
            temp = "33F/92F/22F/72F",
            metrics = emptyList()
        )

        val display = NativeWeatherSemantics.weatherTemperatureText(row)

        assertEquals("33\u00B0F / 22\u00B0F", normalizeDegreeArtifacts(display))
    }

    private fun normalizeDegreeArtifacts(value: String?): String? {
        return value?.replace("Â", "")
    }
}
