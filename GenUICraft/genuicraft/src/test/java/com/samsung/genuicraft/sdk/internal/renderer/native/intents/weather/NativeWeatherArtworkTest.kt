package com.samsung.genuicraft.sdk.internal.renderer.native.intents.weather

import com.samsung.genuicraft.sdk.internal.renderer.native.WeatherRow
import org.junit.Assert.assertEquals
import org.junit.Test

class NativeWeatherArtworkTest {
    @Test
    fun mixedConditionsSelectSpecificArtworkBeforeGeneralArtwork() {
        assertEquals(WeatherArtworkKind.PARTLY_CLOUDY, NativeWeatherArtwork.kindFor("Sunny intervals with clouds"))
        assertEquals(WeatherArtworkKind.STORM, NativeWeatherArtwork.kindFor("Rain showers and thunderstorms"))
        assertEquals(WeatherArtworkKind.SLEET, NativeWeatherArtwork.kindFor("Rain and snow"))
        assertEquals(WeatherArtworkKind.RAIN, NativeWeatherArtwork.kindFor("Light drizzle"))
    }

    @Test
    fun snowShowersAndDustStormsDoNotBecomeRainOrThunder() {
        assertEquals(WeatherArtworkKind.SNOW, NativeWeatherArtwork.kindFor("Snow showers"))
        assertEquals(WeatherArtworkKind.SNOW, NativeWeatherArtwork.kindFor("Snowstorm"))
        assertEquals(WeatherArtworkKind.SNOW, NativeWeatherArtwork.kindFor("Blizzard"))
        assertEquals(WeatherArtworkKind.FOG, NativeWeatherArtwork.kindFor("Sandstorm"))
        assertEquals(WeatherArtworkKind.FOG, NativeWeatherArtwork.kindFor("Dust storm"))
    }

    @Test
    fun clearSkyUsesAuthoredTimeAndUnknownConditionsStayNeutral() {
        assertEquals(WeatherArtworkKind.NIGHT, NativeWeatherArtwork.kindFor("Clear", "Tonight"))
        assertEquals(WeatherArtworkKind.NIGHT, NativeWeatherArtwork.kindFor(
            NativeWeatherSemantics.inferWeatherConditionFromIconUrl("https://openweathermap.org/img/wn/01n.png")))
        assertEquals(WeatherArtworkKind.SUN, NativeWeatherArtwork.kindFor("Clear", "Wed, Sep 9"))
        assertEquals(WeatherArtworkKind.TEMPERATURE, NativeWeatherArtwork.kindFor("Unknown", "Wed, Sep 9"))
        assertEquals(WeatherArtworkKind.TEMPERATURE, NativeWeatherArtwork.kindFor(null, "Wed, Sep 9"))
    }

    @Test
    fun forecastArtworkUsesConditionsOrExplicitRainChanceWithoutInventingConditions() {
        assertEquals(WeatherArtworkKind.SUN, NativeWeatherPresentationPolicy.forecastArtwork(row("Sunny", emptyList())))
        assertEquals(WeatherArtworkKind.RAIN, NativeWeatherPresentationPolicy.forecastArtwork(
            row(null, listOf("Rain probability" to "57% [14]"))))
        assertEquals(WeatherArtworkKind.DROP, NativeWeatherPresentationPolicy.forecastArtwork(
            row(null, listOf("Rain probability" to "0%"))))
        assertEquals(WeatherArtworkKind.TEMPERATURE, NativeWeatherPresentationPolicy.forecastArtwork(
            row(null, listOf("Rainfall amount" to "0.2 mm"))))
        assertEquals(WeatherArtworkKind.TEMPERATURE, NativeWeatherPresentationPolicy.forecastArtwork(
            row(null, listOf("Humidity" to "80%"))))
    }

    @Test
    fun forecastArtworkFindsBoundedChanceAfterAmountAndRejectsUnboundedValues() {
        assertEquals(WeatherArtworkKind.RAIN, NativeWeatherPresentationPolicy.forecastArtwork(
            row(null, listOf("Precipitation amount" to "0.2 mm", "Rain chance" to "55%"))))
        assertEquals(WeatherArtworkKind.TEMPERATURE, NativeWeatherPresentationPolicy.forecastArtwork(
            row(null, listOf("Rain increase" to "20%"))))
        assertEquals(WeatherArtworkKind.RAIN, NativeWeatherPresentationPolicy.forecastArtwork(
            row(null, listOf("PoP" to "55%"))))
        listOf("55-65%", "about 55%", "101%", "0.2 mm").forEach { value ->
            assertEquals(WeatherArtworkKind.TEMPERATURE, NativeWeatherPresentationPolicy.forecastArtwork(
                row(null, listOf("Rain chance" to value))))
        }
    }

    private fun row(condition: String?, metrics: List<Pair<String, String>>) = WeatherRow(
        period = "Wed, Sep 9",
        date = null,
        condition = condition,
        high = "31°C",
        low = "20°C",
        temp = null,
        metrics = metrics,
    )
}
