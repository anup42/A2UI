package com.samsung.genuicraft.renderer.native.intents.weather

import com.samsung.genuicraft.renderer.native.ParsedMediaEntry
import com.samsung.genuicraft.renderer.native.TextBlock
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertEquals
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
                listOf("IndiGo", "07:10", "09:50", "2h 40m", "Non-stop", "₹5,818"),
                listOf("Air India Express", "10:15", "12:50", "2h 35m", "Non-stop", "₹5,820")
            )

        val rows = NativeWeatherSemantics.buildWeatherRows(header, body)
        assertNull(rows)
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
                        "28°C | Partly Cloudy",
                        "Feels like: 30°C",
                        "Wind: 11 km/h | Humidity: 65% | UV Index: High"
                    )
                )
            ),
            fallbackCondition = "Partly Cloudy",
            additionalLines = emptyList()
        )

        assertNotNull(details)
        assertEquals("28°C", details?.temperature)
        assertEquals("30°C", details?.feelsLike)
        assertEquals("65%", details?.humidity)
        assertEquals("11 km/h", details?.wind)
    }
}
