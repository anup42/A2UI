package com.samsung.genuicraft.sdk.internal.renderer.native.intents.weather

import com.samsung.genuicraft.sdk.internal.renderer.native.WeatherRow
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeWeatherPresentationPolicyTest {
    @Test
    fun firstStreamingForecastRowAndLaterRowsUseTheSamePanel() {
        val first = WeatherRow("Wed, Sep 9", null, null, "31°C", "20°C", null, emptyList())
        val next = WeatherRow("Thu, Sep 10", null, null, "30°C", "21°C", null, emptyList())

        assertTrue(NativeWeatherPresentationPolicy.shouldUseForecastPanel(listOf(first)))
        assertTrue(NativeWeatherPresentationPolicy.shouldUseForecastPanel(listOf(first, next)))
    }

    @Test
    fun singleCurrentReadingRetainsHeroAndEmptyOrConditionOnlyDoesNotImplyForecast() {
        val current = WeatherRow("Today", null, "Cloudy", "31°C", "20°C", "26°C", emptyList())
        val conditionOnly = WeatherRow("Today", null, "Cloudy", null, null, null, emptyList())

        assertFalse(NativeWeatherPresentationPolicy.shouldUseForecastPanel(listOf(current)))
        assertFalse(NativeWeatherPresentationPolicy.shouldUseForecastPanel(listOf(conditionOnly)))
        assertFalse(NativeWeatherPresentationPolicy.shouldUseForecastPanel(emptyList()))
    }

    @Test
    fun heroRetainsEveryMetricIncludingRepeatedLabelsAndLongAdvice() {
        val advice = "Carry a rain jacket and plan a covered route. ".repeat(12) + "[7]"
        val metrics = listOf(
            "Rain probability" to "5% [3]",
            "Wind" to "12 km/h",
            "Humidity" to "80%",
            "UV" to "4",
            "Wind" to "Gusts up to 25 km/h [9]",
            "What to wear" to advice,
        )
        val row = WeatherRow("Today", null, "Showers", "26°C", "21°C", "24°C", metrics)

        assertEquals(
            listOf("High" to "26°C", "Low" to "21°C", "Temperature" to "24°C") + metrics,
            NativeWeatherPresentationPolicy.heroMetrics(row, "24°C"),
        )
        assertTrue(NativeWeatherPresentationPolicy.accessibilityDescription(row).contains(advice))
        assertTrue(NativeWeatherPresentationPolicy.accessibilityDescription(row).contains("Gusts up to 25 km/h [9]"))
    }

    @Test
    fun dailyForecastRetainsTemperatureUnitsProbabilityAndCitationsWithoutInventingCondition() {
        val row = WeatherRow("Wed, Sep 9", null, null, "88°F [2]", "68°F", null,
            listOf("Rain probability" to "57% [14]"))

        assertFalse(NativeWeatherPresentationPolicy.hasExplicitCondition(row.condition))
        assertFalse(NativeWeatherPresentationPolicy.hasExplicitCondition(" "))
        assertEquals(0, NativeWeatherPresentationPolicy.precipitationIndex(row.metrics))
        assertEquals(
            "Wed, Sep 9. High: 88°F [2]. Low: 68°F. Rain probability: 57% [14]",
            NativeWeatherPresentationPolicy.accessibilityDescription(row),
        )
    }

    @Test
    fun precipitationSelectionDoesNotHideLowProbabilitiesOrOtherPrecipitationFields() {
        val row = WeatherRow("Thursday", null, null, "30°C", "21°C", null,
            listOf("Rain probability" to "0%", "Precipitation amount" to "0.2 mm [5]"))

        assertEquals(0, NativeWeatherPresentationPolicy.precipitationIndex(row.metrics))
        assertTrue(NativeWeatherPresentationPolicy.accessibilityDescription(row).contains("Rain probability: 0%"))
        assertTrue(NativeWeatherPresentationPolicy.accessibilityDescription(row).contains("Precipitation amount: 0.2 mm [5]"))
        assertEquals(row.metrics, NativeWeatherPresentationPolicy.heroMetrics(row, "").drop(2))
    }
}
