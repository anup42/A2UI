package com.samsung.genuicraft.mcp

import android.net.Uri
import android.util.Log
import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.samsung.genuicraft.inference.InferenceBackend

/**
 * Takes raw MCP API data and uses the LLM to format it into a Stage 2-compatible
 * rich response text that can then flow through Stage 3 (GenUI IR) and Stage 4 (render).
 */
object McpResponseFormatter {

    private const val LOG_TAG = "McpResponseFormatter"

    data class FormatResult(
        val formattedResponse: String,
        val error: String?,
        val streamDurationMs: Long?
    )

    private val mojibakeFixups = linkedMapOf(
        "ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â" to "â€”",
        "Ã¢â‚¬â€" to "â€”",
        "Ãƒâ€šÃ‚Â°C" to "Â°C",
        "ÃƒÂ¢Ã‹Å“Ã¢â‚¬Â¦" to "â˜…",
        "ÃƒÂ¢Ã‹Å“Ã¢â‚¬Â " to "â˜†",
        "ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¹" to "â‚¹",
        "MonÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦6=Sun" to "Monâ€¦6=Sun",
        "Ãƒâ€šÃ‚Â·" to "Â·",
        "Ã‚Â·" to "Â·"
    )

    fun normalizeForStage3(text: String): String {
        if (text.isBlank()) {
            return text
        }
        var normalized = text
        mojibakeFixups.forEach { (from, to) ->
            normalized = normalized.replace(from, to)
        }
        return normalized
    }

    /**
     * Uses the configured LLM backend to format raw MCP data into a structured
     * response suitable for the existing Stage 3 pipeline.
     */
    fun formatWithLlm(
        mcpResult: McpClient.McpResult,
        queryText: String,
        backend: InferenceBackend,
        maxOutputTokens: Int
    ): FormatResult {
        if (!mcpResult.success || mcpResult.data == null) {
            return FormatResult(
                formattedResponse = "",
                error = mcpResult.error ?: "MCP data unavailable",
                streamDurationMs = null
            )
        }

        // Use deterministic Kotlin formatters for domains where preserving structured rows/media is critical.
        // This avoids the LLM dropping photos, source links, or weather metrics needed by native templates.
        if (
            mcpResult.domain == McpSettings.Domain.WEATHER ||
            mcpResult.domain == McpSettings.Domain.RESTAURANTS ||
            mcpResult.domain == McpSettings.Domain.PLACES
        ) {
            val fallback = buildFallbackResponse(mcpResult.domain, mcpResult.data, queryText)
            return FormatResult(
                formattedResponse = normalizeForStage3(fallback),
                error = null,
                streamDurationMs = null
            )
        }

        val prompt = buildFormattingPrompt(mcpResult.domain, mcpResult.data, queryText)

        val response = backend.generate(
            InferenceBackend.GenerateRequest(
                prompt = prompt,
                systemPrompt = FORMATTING_SYSTEM_PROMPT,
                temperature = 0.3,
                maxOutputTokens = maxOutputTokens,
                jsonMode = false,
                enableGoogleSearch = false
            )
        )

        if (response.error != null) {
            Log.w(LOG_TAG, "LLM formatting failed: ${response.error}")
            // Fallback: generate a simple structured response from the raw data
            val fallback = buildFallbackResponse(mcpResult.domain, mcpResult.data, queryText)
            return FormatResult(
                formattedResponse = normalizeForStage3(fallback),
                error = null,
                streamDurationMs = response.streamDurationMs
            )
        }

        val text = response.text.trim()
        if (text.isBlank()) {
            val fallback = buildFallbackResponse(mcpResult.domain, mcpResult.data, queryText)
            return FormatResult(
                formattedResponse = normalizeForStage3(fallback),
                error = null,
                streamDurationMs = response.streamDurationMs
            )
        }

        return FormatResult(
            formattedResponse = normalizeForStage3(text),
            error = null,
            streamDurationMs = response.streamDurationMs
        )
    }

    private fun buildFormattingPrompt(domain: McpSettings.Domain, data: JsonObject, queryText: String): String {
        val dataStr = data.toString().take(24_000) // Limit data size for prompt

        return """User query: $queryText

Real-time ${domain.displayName} data (JSON):
```json
$dataStr
```

Format this real-time data into a complete, polished response following the formatting rules in your system prompt.
The data above is LIVE and REAL â€” present it as authoritative current information, not as examples or samples.
Do not add disclaimers about data accuracy. Present the data directly as the answer."""
    }

    private const val FORMATTING_SYSTEM_PROMPT = """You are formatting real-time API data into a high-quality, structured response that will be rendered as UI cards.

Rules:
1) Present the data as a complete, authoritative answer. This is LIVE data, not samples.
2) Use topic-specific headings (never generic like "Summary" or "Results").
3) Include Media lines with Bootstrap Icons for each major section:
   Media: Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/<icon-name>.svg
   Common icons: sun, cloud, airplane, shop, star, building, geo-alt, newspaper, cup-hot, calendar, thermometer-half
4) For tabular data (flights, comparisons, forecasts), use pipe tables with headers.
5) For restaurants and places, you MUST format each entry as a CARD block:
   ### <restaurant/place name>
   Media: Image=<photoUri from data> Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/shop.svg
   *<cuisine/type tags>*
   **Rating:** <rating> â˜…â˜…â˜…â˜… (<reviewCount> reviews)
   **Address:** <address>
   **Review:** "<first review text snippet>"
   Action: [Button: View on Maps] <googleMapsUri>
   Action: [Button: Visit Website] <websiteUri>

   CRITICAL for restaurants/places:
   - If a "photoUri" field exists in the data, ALWAYS include it as Media: Image=<photoUri>
   - If "reviews" array exists, include the first review text (truncated to 120 chars)
   - If "googleMapsUri" exists, include Action: [Button: View on Maps] <url>
   - If "websiteUri" exists, include Action: [Button: Visit Website] <url>
   - Show rating as numeric value plus â˜… star characters
   - Do NOT use a table for restaurants/places â€” use individual card blocks
6) For hotels, use option cards with rating, price, and book button.
7) Include a Sources section with real URLs when applicable.
8) Keep sections concise â€” prefer cards, tables, and bullets over long paragraphs.
9) For weather: start with current conditions block, then forecast table.
10) For flights: include Airline | Departure | Arrival | Duration | Stops | Fare table.
11) For news: include title, source, and publication time for each article.
12) Never use tokens like "EXAMPLE", "SAMPLE", "DEMO" in output."""

    /**
     * Builds the MCP data section as formatted text (Kotlin-side, no LLM call).
     * Returns blank string when the API returned no useful data so the pipeline can fall back
     * to normal Stage 2 LLM instead of showing "No X found."
     */
    fun buildDataSection(mcpResult: McpClient.McpResult, queryText: String): String {
        if (!mcpResult.success || mcpResult.data == null) {
            Log.w(LOG_TAG, "MCP ${mcpResult.domain.key} unavailable: ${mcpResult.error}")
            return ""
        }
        if (isMcpDataEmpty(mcpResult.domain, mcpResult.data)) {
            Log.w(LOG_TAG, "MCP ${mcpResult.domain.key} returned empty results for query: $queryText")
            return ""
        }
        return normalizeForStage3(buildFallbackResponse(mcpResult.domain, mcpResult.data, queryText))
    }

    private fun isMcpDataEmpty(domain: McpSettings.Domain, data: JsonObject): Boolean = when (domain) {
        McpSettings.Domain.HOTELS -> (data.getAsJsonArray("properties")?.size() ?: 0) == 0
        McpSettings.Domain.RESTAURANTS, McpSettings.Domain.PLACES ->
            (data.getAsJsonArray("results")?.size() ?: 0) == 0
        McpSettings.Domain.FLIGHTS -> (data.getAsJsonArray("flights")?.size() ?: 0) == 0
        McpSettings.Domain.NEWS -> (data.getAsJsonArray("results")?.size() ?: 0) == 0
        McpSettings.Domain.WEATHER -> false // weather always has current data
    }

    /** Returns null if element is Kotlin-null or a JSON null value. */
    private fun JsonObject.safeString(key: String): String? =
        get(key)?.takeIf { !it.isJsonNull }?.asString

    /** Safe asString on a JsonElement that might be JsonNull. */
    private fun com.google.gson.JsonElement?.safeString(): String? =
        this?.takeIf { !it.isJsonNull }?.asString

    private fun JsonObject.safeDouble(key: String): Double? =
        get(key)?.takeIf { !it.isJsonNull }?.asDouble

    private fun JsonObject.safeInt(key: String): Int? =
        get(key)?.takeIf { !it.isJsonNull }?.asInt

    private fun com.google.gson.JsonElement?.safeInt(): Int? =
        this?.takeIf { !it.isJsonNull }?.asInt

    /** Generates a simple structured response when LLM formatting fails. */
    private fun buildFallbackResponse(domain: McpSettings.Domain, data: JsonObject, queryText: String): String {
        return when (domain) {
            McpSettings.Domain.WEATHER -> buildWeatherFallback(data)
            McpSettings.Domain.FLIGHTS -> buildFlightsFallback(data)
            McpSettings.Domain.RESTAURANTS -> buildRestaurantsFallback(data)
            McpSettings.Domain.HOTELS -> buildHotelsFallback(data)
            McpSettings.Domain.PLACES -> buildPlacesFallback(data)
            McpSettings.Domain.NEWS -> buildNewsFallback(data)
        }
    }

    private fun buildWeatherFallback(data: JsonObject): String {
        val location = data.safeString("location") ?: "Unknown"
        val country = data.safeString("country") ?: ""
        val current = data.getAsJsonObject("current")
        val currentUnits = data.getAsJsonObject("current_units")
        val daily = data.getAsJsonObject("daily")
        val dailyUnits = data.getAsJsonObject("daily_units")
        val hourly = data.getAsJsonObject("hourly")

        val dates = daily?.getAsJsonArray("time")
        val maxTemps = daily?.getAsJsonArray("temperature_2m_max")
        val minTemps = daily?.getAsJsonArray("temperature_2m_min")
        val apparentMax = daily?.getAsJsonArray("apparent_temperature_max")
        val precipProbability = daily?.getAsJsonArray("precipitation_probability_max")
        val precipSum = daily?.getAsJsonArray("precipitation_sum")
        val rainSum = daily?.getAsJsonArray("rain_sum")
        val windSpeedMax = daily?.getAsJsonArray("wind_speed_10m_max")
        val windGustMax = daily?.getAsJsonArray("wind_gusts_10m_max")
        val windDirection = daily?.getAsJsonArray("wind_direction_10m_dominant")
        val uvMax = daily?.getAsJsonArray("uv_index_max")
        val codes = daily?.getAsJsonArray("weather_code")

        val currentCondition = wmoCodeToCondition(current?.safeInt("weather_code") ?: codes?.get(0)?.safeInt() ?: -1)
        val currentTemp = formatWeatherValue(current?.safeString("temperature_2m"), currentUnits?.safeString("temperature_2m"))
        val todayHigh = formatWeatherValue(maxTemps?.get(0)?.safeString(), dailyUnits?.safeString("temperature_2m_max"))
        val todayLow = formatWeatherValue(minTemps?.get(0)?.safeString(), dailyUnits?.safeString("temperature_2m_min"))
        val todayRainChance = formatWeatherValue(precipProbability?.get(0)?.safeString(), dailyUnits?.safeString("precipitation_probability_max"))
        val currentWind = formatWindValue(
            speed = current?.safeString("wind_speed_10m"),
            speedUnit = currentUnits?.safeString("wind_speed_10m"),
            direction = current?.safeString("wind_direction_10m")
        )
        val currentHumidity = formatWeatherValue(current?.safeString("relative_humidity_2m"), currentUnits?.safeString("relative_humidity_2m"))
        val currentUv = formatWeatherValue(current?.safeString("uv_index"), currentUnits?.safeString("uv_index"))

        val sb = StringBuilder()
        sb.appendLine("## Weather in $location${if (country.isNotBlank()) ", $country" else ""}")
        sb.appendLine()
        sb.appendLine(
            listOf(
                "**Today:** $currentCondition",
                currentTemp?.let { "$it now" },
                todayHigh?.let { "high $it" },
                todayLow?.let { "low $it" },
                todayRainChance?.let { "rain $it" },
                currentWind?.let { "wind $it" },
                currentHumidity?.let { "humidity $it" },
                currentUv?.let { "UV $it" }
            ).filterNotNull().joinToString(", ") + "."
        )
        sb.appendLine()

        sb.appendLine("| Day | Date | Condition | Temp | High | Low | Feels Like | Rain Chance | Rain | Wind | Gusts | Humidity | UV | Best Window | Morning | Afternoon | Evening | Night | What to wear |")
        sb.appendLine("|-----|------|-----------|------|------|-----|------------|-------------|------|------|-------|----------|----|-------------|---------|-----------|---------|-------|--------------|")

        val rowCount = listOfNotNull(dates?.size(), maxTemps?.size(), minTemps?.size()).minOrNull()?.coerceAtMost(7) ?: 0
        for (i in 0 until rowCount) {
            val date = dates?.get(i)?.safeString().orEmpty()
            val label = if (i == 0) "Today" else weatherDayLabel(date)
            val weatherCode = if (i == 0) current?.safeInt("weather_code") ?: codes?.get(i)?.safeInt() ?: -1 else codes?.get(i)?.safeInt() ?: -1
            val condition = wmoCodeToCondition(weatherCode)
            val hourlySummary = buildHourlyWeatherSummary(date, hourly)
            val temp = if (i == 0) currentTemp ?: "-" else "-"
            val high = formatWeatherValue(maxTemps?.get(i)?.safeString(), dailyUnits?.safeString("temperature_2m_max")) ?: "-"
            val low = formatWeatherValue(minTemps?.get(i)?.safeString(), dailyUnits?.safeString("temperature_2m_min")) ?: "-"
            val feelsLike = if (i == 0) {
                formatWeatherValue(current?.safeString("apparent_temperature"), currentUnits?.safeString("apparent_temperature"))
            } else {
                formatWeatherValue(apparentMax?.get(i)?.safeString(), dailyUnits?.safeString("apparent_temperature_max"))
            } ?: "-"
            val rainChance = formatWeatherValue(precipProbability?.get(i)?.safeString(), dailyUnits?.safeString("precipitation_probability_max")) ?: "-"
            val rain = formatWeatherValue(rainSum?.get(i)?.safeString(), dailyUnits?.safeString("rain_sum"))
                ?: formatWeatherValue(precipSum?.get(i)?.safeString(), dailyUnits?.safeString("precipitation_sum"))
                ?: "-"
            val wind = if (i == 0) {
                currentWind
            } else {
                formatWindValue(
                    speed = windSpeedMax?.get(i)?.safeString(),
                    speedUnit = dailyUnits?.safeString("wind_speed_10m_max"),
                    direction = windDirection?.get(i)?.safeString()
                )
            } ?: "-"
            val gust = formatWeatherValue(windGustMax?.get(i)?.safeString(), dailyUnits?.safeString("wind_gusts_10m_max")) ?: "-"
            val humidity = if (i == 0) currentHumidity else hourlySummary.humidity
            val uv = if (i == 0) currentUv else formatWeatherValue(uvMax?.get(i)?.safeString(), dailyUnits?.safeString("uv_index_max"))
            sb.appendLine(
                weatherTableRow(
                    label,
                    date,
                    condition,
                    temp,
                    high,
                    low,
                    feelsLike,
                    rainChance,
                    rain,
                    wind,
                    gust,
                    humidity ?: "-",
                    uv ?: "-",
                    hourlySummary.bestWindow,
                    hourlySummary.morning,
                    hourlySummary.afternoon,
                    hourlySummary.evening,
                    hourlySummary.night,
                    weatherAdvice(condition, rainChance, high, wind)
                )
            )
        }

        sb.appendLine()
        sb.appendLine("## Sources")
        sb.appendLine("- Open-Meteo Forecast API: https://open-meteo.com/en/docs")
        sb.appendLine("- Open-Meteo Geocoding API: https://open-meteo.com/en/docs/geocoding-api")

        return sb.toString()
    }

    private data class HourlyWeatherSummary(
        val morning: String = "-",
        val afternoon: String = "-",
        val evening: String = "-",
        val night: String = "-",
        val bestWindow: String = "-",
        val humidity: String? = null
    )

    private fun buildHourlyWeatherSummary(date: String, hourly: JsonObject?): HourlyWeatherSummary {
        if (date.isBlank() || hourly == null) {
            return HourlyWeatherSummary()
        }
        val times = hourly.getAsJsonArray("time") ?: return HourlyWeatherSummary()
        val codes = hourly.getAsJsonArray("weather_code")
        val precipitationProbability = hourly.getAsJsonArray("precipitation_probability")
        val humidityValues = hourly.getAsJsonArray("relative_humidity_2m")

        fun indexesFor(range: IntRange): List<Int> {
            return (0 until times.size()).filter { index ->
                val time = times[index].safeString().orEmpty()
                if (!time.startsWith(date)) return@filter false
                val hour = time.substringAfter('T', "").take(2).toIntOrNull() ?: return@filter false
                hour in range
            }
        }

        fun conditionFor(range: IntRange): String {
            val indexes = indexesFor(range)
            if (indexes.isEmpty()) return "-"
            val selectedIndex = indexes.maxByOrNull { index ->
                precipitationProbability?.get(index)?.safeString()?.toDoubleOrNull() ?: -1.0
            } ?: indexes[indexes.size / 2]
            return wmoCodeToShortCondition(codes?.get(selectedIndex)?.safeInt() ?: -1)
        }

        val dayIndexes = indexesFor(6..22)
        val bestWindow = dayIndexes.minByOrNull { index ->
            precipitationProbability?.get(index)?.safeString()?.toDoubleOrNull() ?: 101.0
        }?.let { index ->
            val hour = times[index].safeString().orEmpty().substringAfter('T', "").take(2).toIntOrNull()
            when (hour) {
                in 6..11 -> "Before noon"
                in 12..16 -> "Afternoon"
                in 17..20 -> "Evening"
                in 21..23 -> "Late evening"
                else -> "Anytime"
            }
        } ?: "-"

        val humidity = dayIndexes
            .mapNotNull { index -> humidityValues?.get(index)?.safeString()?.toDoubleOrNull() }
            .takeIf { it.isNotEmpty() }
            ?.average()
            ?.let { "${"%.0f".format(it)}%" }

        return HourlyWeatherSummary(
            morning = conditionFor(6..11),
            afternoon = conditionFor(12..16),
            evening = conditionFor(17..20),
            night = conditionFor(21..23),
            bestWindow = bestWindow,
            humidity = humidity
        )
    }

    private fun weatherTableRow(vararg cells: String): String {
        return cells.joinToString(prefix = "| ", separator = " | ", postfix = " |") { cell -> cleanWeatherTableCell(cell) }
    }

    private fun cleanWeatherTableCell(value: String): String =
        value.ifBlank { "-" }.replace("|", "/").replace(Regex("\\s+"), " ").trim()

    private fun weatherDayLabel(date: String): String {
        return try {
            val parsed = java.time.LocalDate.parse(date)
            parsed.dayOfWeek.name.take(3).lowercase().replaceFirstChar { it.uppercase() }
        } catch (_: Exception) {
            date
        }
    }

    private fun formatWeatherValue(value: String?, unit: String?): String? {
        val raw = value?.trim()?.takeIf { it.isNotBlank() } ?: return null
        val normalized = raw.toDoubleOrNull()?.let { numeric ->
            if (kotlin.math.abs(numeric - numeric.toInt()) < 0.05) numeric.toInt().toString() else "%.1f".format(numeric)
        } ?: raw
        val cleanUnit = unit?.trim().orEmpty()
        if (cleanUnit.isBlank()) return normalized
        val separator = if (cleanUnit in setOf("°C", "°F", "%")) "" else " "
        return "$normalized$separator$cleanUnit"
    }

    private fun formatWindValue(speed: String?, speedUnit: String?, direction: String?): String? {
        val speedText = formatWeatherValue(speed, speedUnit) ?: return null
        val directionText = windDirectionLabel(direction)
        return listOfNotNull(directionText, speedText).joinToString(" ")
    }

    private fun windDirectionLabel(direction: String?): String? {
        val degrees = direction?.trim()?.toDoubleOrNull() ?: return null
        val labels = listOf("N", "NE", "E", "SE", "S", "SW", "W", "NW")
        val index = (((degrees + 22.5) % 360) / 45.0).toInt().coerceIn(0, labels.lastIndex)
        return labels[index]
    }

    private fun weatherAdvice(condition: String, rainChance: String, high: String, wind: String): String {
        val rainPercent = Regex("(\\d{1,3})").find(rainChance)?.groupValues?.getOrNull(1)?.toIntOrNull()
        val highTemp = Regex("-?\\d{1,3}").find(high)?.value?.toIntOrNull()
        val windSpeed = Regex("(\\d{1,3})").find(wind)?.groupValues?.getOrNull(1)?.toIntOrNull()
        val rainy = (rainPercent != null && rainPercent >= 55) ||
            condition.contains("rain", ignoreCase = true) ||
            condition.contains("drizzle", ignoreCase = true) ||
            condition.contains("shower", ignoreCase = true) ||
            condition.contains("storm", ignoreCase = true)
        val windy = windSpeed != null && windSpeed >= 25
        return when {
            rainy && windy -> "Carry an umbrella or rain jacket; choose quick-dry shoes and avoid loose layers."
            rainy -> "Carry an umbrella; wear quick-dry footwear and light layers."
            highTemp != null && highTemp >= 32 -> "Wear breathable cotton, sunglasses, and carry water."
            windy -> "Use a light wind-resistant layer and secure loose accessories."
            else -> "Light comfortable clothing should work; carry a small layer if heading out late."
        }
    }


    /** Maps WMO weather interpretation code to a human-readable condition string. */
    private fun wmoCodeToCondition(code: Int): String = when (code) {
        0 -> "Clear sky"
        1 -> "Mainly clear"
        2 -> "Partly cloudy"
        3 -> "Overcast"
        45, 48 -> "Foggy"
        51, 53, 55 -> "Drizzle"
        56, 57 -> "Freezing drizzle"
        61, 63, 65 -> "Rain"
        66, 67 -> "Freezing rain"
        71, 73, 75 -> "Snowfall"
        77 -> "Snow grains"
        80, 81, 82 -> "Rain showers"
        85, 86 -> "Snow showers"
        95 -> "Thunderstorm"
        96, 99 -> "Thunderstorm with hail"
        else -> "Unknown"
    }

    private fun wmoCodeToShortCondition(code: Int): String = when (code) {
        0, 1 -> "Clear"
        2 -> "Clouds"
        3 -> "Overcast"
        45, 48 -> "Fog"
        51, 53, 55, 56, 57 -> "Drizzle"
        61, 63, 65, 66, 67 -> "Rain"
        71, 73, 75, 77, 85, 86 -> "Snow"
        80, 81, 82 -> "Showers"
        95, 96, 99 -> "Storm"
        else -> "-"
    }
private fun buildFlightsFallback(data: JsonObject): String {
        val origin = data.safeString("origin") ?: "?"
        val destination = data.safeString("destination") ?: "?"
        val flights = data.getAsJsonArray("flights")

        val sb = StringBuilder()
        sb.appendLine("## Flights from $origin to $destination")

        if (flights == null || flights.size() == 0) {
            sb.appendLine("No flights found for this route.")
        } else {
            sb.appendLine("| Airline | Departure | Arrival | Duration | Stops | Fare |")
            sb.appendLine("|---------|-----------|---------|----------|-------|------|")
            for (i in 0 until minOf(flights.size(), 8)) {
                    val flightObj = flights[i].asJsonObject
                    val price = flightObj.safeString("price") ?: "N/A"
                    val currency = "USD"
                    val route = flightObj.getAsJsonArray("flights")
                    val duration = route?.get(0)?.asJsonObject?.safeString("duration") ?: "N/A"
                    val airlines = route?.get(0)?.asJsonObject?.safeString("airline") ?: "â€”"
                    val stops = if (route != null) route.size() - 1 else 0
                    val stopsStr = if (stops <= 0) "Non-stop" else "$stops stop${if (stops > 1) "s" else ""}"
                    val firstLeg = route?.get(0)?.asJsonObject
                    val lastLeg = route?.get(route.size() - 1)?.asJsonObject
                    val depObj = firstLeg?.getAsJsonObject("departure_airport")
                    val arrObj = lastLeg?.getAsJsonObject("arrival_airport")
                    val dep = depObj?.safeString("time")?.substringAfter(" ")?.take(5) ?: "â€”"
                    val arr = arrObj?.safeString("time")?.substringAfter(" ")?.take(5) ?: "â€”"
                    sb.appendLine("| $airlines | $dep | $arr | $duration mins | $stopsStr | $currency $price |")
            }
        }

        sb.appendLine()
        sb.appendLine("## Sources")
        sb.appendLine("- SerpApi Google Flights Search: https://serpapi.com/")

        return sb.toString()
    }

    private fun buildRestaurantsFallback(data: JsonObject): String {
        val location = data.safeString("location") ?: "your area"
        val provider = data.safeString("provider") ?: "google_places"
        val results = data.getAsJsonArray("results")

        val sb = StringBuilder()
        sb.appendLine("## Top Restaurants in $location")

        if (results == null || results.size() == 0) {
            sb.appendLine("No restaurants found in $location.")
        } else {
            for (i in 0 until minOf(results.size(), 6)) {
                val biz = results[i].asJsonObject

                val name = biz.getAsJsonObject("displayName")?.safeString("text") ?: "Restaurant"
                val ratingRaw = biz.safeDouble("rating") ?: 0.0
                val ratingStr = if (ratingRaw > 0) "%.1f".format(ratingRaw) else "N/A"
                val fullStars = ratingRaw.toInt().coerceIn(0, 5)
                val emptyStars = 5 - fullStars
                val starsDisplay = if (ratingRaw > 0) "â˜…".repeat(fullStars) + "â˜†".repeat(emptyStars) else ""
                val reviewCount = biz.safeInt("userRatingCount") ?: 0
                val mapsUri = biz.safeString("googleMapsUri") ?: ""
                val websiteUri = biz.safeString("websiteUri") ?: ""
                val address = biz.safeString("formattedAddress") ?: ""
                val photoUri = biz.safeString("photoUri") ?: ""
                val distanceMeters = biz.safeDouble("distance")
                val priceLevel = when (biz.safeString("priceLevel")) {
                    "PRICE_LEVEL_INEXPENSIVE" -> "â‚¹"
                    "PRICE_LEVEL_MODERATE" -> "â‚¹â‚¹"
                    "PRICE_LEVEL_EXPENSIVE" -> "â‚¹â‚¹â‚¹"
                    "PRICE_LEVEL_VERY_EXPENSIVE" -> "â‚¹â‚¹â‚¹â‚¹"
                    else -> ""
                }

                val cuisineList = biz.getAsJsonArray("cuisineTags")
                    ?.mapNotNull { it.safeString()?.trim()?.takeIf { tag -> tag.isNotBlank() } }
                    ?: emptyList()

                // Tags: filter noise types, format as chip-friendly plain labels.
                val typeList = cuisineList + (biz.getAsJsonArray("types")
                    ?.mapNotNull { it.takeIf { e -> !e.isJsonNull }?.asString }
                    ?.filter { t ->
                        t !in setOf("restaurant", "food", "point_of_interest",
                            "establishment", "place_of_worship", "store", "catering", "catering.restaurant")
                    }
                    ?.map { it.substringAfterLast('.') }
                    ?.map { it.replace("_", " ").replaceFirstChar { c -> c.uppercase() } }
                    ?: emptyList())
                val uniqueTypeList = typeList.distinct().take(4)

                val editorial = biz.getAsJsonObject("editorialSummary")
                    ?.safeString("text")?.trim()
                    ?.takeIf { it.isNotBlank() }

                val openingHours = biz.getAsJsonObject("regularOpeningHours")
                val openNow = openingHours?.get("openNow")?.takeIf { !it.isJsonNull }?.asBoolean
                val openStatus = when (openNow) {
                    true -> "Open now"
                    false -> "Closed now"
                    null -> null
                }
                // Today's hours (weekdayDescriptions[0] = Monday, adjust by day-of-week)
                val todayHours = openingHours?.getAsJsonArray("weekdayDescriptions")
                    ?.let { arr ->
                        val dayIndex = (java.util.Calendar.getInstance()
                            .get(java.util.Calendar.DAY_OF_WEEK) + 5) % 7  // 0=Monâ€¦6=Sun
                        if (dayIndex < arr.size()) arr[dayIndex].asString?.substringAfter(":")?.trim() else null
                    }

                // First user review snippet (from same API call)
                val reviewSnippet = biz.getAsJsonArray("reviews")
                    ?.firstOrNull()?.asJsonObject
                    ?.getAsJsonObject("text")?.get("text")?.asString
                    ?.trim()
                    ?.takeIf { it.isNotBlank() && it != "null" }
                    ?.take(140)

                // Card block
                sb.appendLine()
                sb.appendLine("## ${i + 1}. $name")
                if (photoUri.isNotBlank()) {
                    sb.appendLine("Media: Image=$photoUri Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/shop.svg")
                } else {
                    sb.appendLine("Media: Image=${restaurantVisualUri(name)} Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/shop.svg")
                }

                // Tags: pipe-separated format so Stage 3 renders as chip row
                if (uniqueTypeList.isNotEmpty()) {
                    sb.appendLine("Tags: ${uniqueTypeList.joinToString(" | ")}")
                }

                // Rating row with price level
                val ratingLine = buildString {
                    if (ratingRaw > 0) append("$starsDisplay $ratingStr ($reviewCount reviews)")
                    if (priceLevel.isNotBlank()) {
                        if (isNotEmpty()) append("  Â·  ")
                        append(priceLevel)
                    }
                }
                if (ratingLine.isNotBlank()) sb.appendLine(ratingLine)

                if (address.isNotBlank()) {
                    sb.appendLine("- Address: $address")
                }
                if (distanceMeters != null && distanceMeters > 0) {
                    val distanceLabel = if (distanceMeters >= 1000) {
                        "%.1f km away".format(distanceMeters / 1000.0)
                    } else {
                        "${distanceMeters.toInt()} m away"
                    }
                    sb.appendLine("- Distance: $distanceLabel")
                }

                // Open status + today's hours
                val hoursLine = buildString {
                    if (openStatus != null) append(openStatus)
                    if (todayHours != null) {
                        if (isNotEmpty()) append(" Â· ")
                        append("Today: $todayHours")
                    }
                }
                if (hoursLine.isNotBlank()) sb.appendLine(hoursLine)

                // Editorial summary â€” concise description from Google
                if (editorial != null) sb.appendLine("- $editorial")

                // Action buttons
                if (mapsUri.isNotBlank()) sb.appendLine("Action: [Button: View on Map] $mapsUri")
                if (websiteUri.isNotBlank()) sb.appendLine("Action: [Button: Visit Website] $websiteUri")
            }
        }

        sb.appendLine()
        sb.appendLine("## Sources")
        if (provider == "google_maps_grounding") {
            sb.appendLine("- Google Maps grounding via Vertex AI: https://maps.google.com/")
        } else {
            sb.appendLine("- Google Places: https://maps.google.com/")
        }

        return sb.toString()
    }

    private fun restaurantVisualUri(name: String): String {
        val encodedTitle = Uri.encode(name.ifBlank { "Restaurant" })
        return "genuicraft://visual/restaurant?title=$encodedTitle"
    }

    private fun buildHotelsFallback(data: JsonObject): String {
        val location = data.safeString("location") ?: "your area"
        val queryText = data.safeString("query") ?: ""
        val properties = data.getAsJsonArray("properties") ?: return ""

        val sb = StringBuilder()
        val heading = if (queryText.isNotBlank() &&
            queryText.lowercase(java.util.Locale.US) != "hotels in $location".lowercase(java.util.Locale.US)
        ) {
            queryText.replaceFirstChar { it.uppercase() }
        } else {
            "Hotels in $location"
        }
        sb.appendLine("## $heading")
        sb.appendLine()

        for (i in 0 until minOf(properties.size(), 8)) {
            val hotel = properties[i].asJsonObject
            val name = hotel.safeString("name") ?: continue
            val hotelClass = hotel.safeString("hotel_class") ?: ""
            val ratingRaw = hotel.safeDouble("overall_rating") ?: 0.0
            val ratingStr = if (ratingRaw > 0) "%.1f".format(ratingRaw) else ""
            val reviewCount = hotel.safeInt("reviews") ?: 0
            val description = hotel.safeString("description")?.trim()
            val thumbnail = hotel.safeString("thumbnail")
                ?: hotel.safeString("image")
                ?: hotel.safeString("image_url")
                ?: hotel.safeString("photo")
                ?: hotel.safeString("photo_url")
                ?: hotel.getAsJsonArray("images")
                    ?.firstOrNull()
                    ?.let { imageEntry ->
                        if (imageEntry.isJsonObject) {
                            val imageObj = imageEntry.asJsonObject
                            imageObj.safeString("thumbnail")
                                ?: imageObj.safeString("original_image")
                                ?: imageObj.safeString("url")
                                ?: imageObj.safeString("image")
                                ?: imageObj.safeString("photo")
                        } else {
                            imageEntry.safeString()
                        }
                    }
                ?: ""
            val link = hotel.safeString("link") ?: ""
            val checkIn = hotel.safeString("check_in_time") ?: ""
            val checkOut = hotel.safeString("check_out_time") ?: ""

            val rateObj = hotel.get("rate_per_night")?.takeIf { !it.isJsonNull }?.asJsonObject
            val price = rateObj?.safeString("lowest") ?: rateObj?.safeString("extracted_lowest") ?: ""

            val amenities = hotel.getAsJsonArray("amenities")
                ?.mapNotNull { it.safeString()?.trim()?.takeIf { a -> a.isNotBlank() } }
                ?.take(5) ?: emptyList()

            sb.appendLine("## ${i + 1}. $name")
            if (thumbnail.isNotBlank()) sb.appendLine("Media: Image=$thumbnail")
            if (hotelClass.isNotBlank()) sb.appendLine("**$hotelClass**")
            if (ratingStr.isNotBlank()) {
                val reviewPart = if (reviewCount > 0) " ($reviewCount reviews)" else ""
                sb.appendLine("$ratingStr stars$reviewPart")
            }
            if (price.isNotBlank()) sb.appendLine("**Price:** From $price / night")
            if (!description.isNullOrBlank()) sb.appendLine(description)
            val checkInfo = buildString {
                if (checkIn.isNotBlank()) append("Check-in: $checkIn")
                if (checkOut.isNotBlank()) {
                    if (isNotEmpty()) append(" Â· ")
                    append("Check-out: $checkOut")
                }
            }
            if (checkInfo.isNotBlank()) sb.appendLine(checkInfo)
            if (amenities.isNotEmpty()) sb.appendLine("Tags: ${amenities.joinToString(" | ")}")
            if (link.isNotBlank()) sb.appendLine("Action: [Button: Book Now] $link")
            sb.appendLine()
        }

        sb.appendLine("## Sources")
        sb.appendLine("- Google Hotels via SerpApi: https://serpapi.com/")

        return sb.toString()
    }

    private fun buildPlacesFallback(data: JsonObject): String {
        val location = data.safeString("location") ?: "your area"
        val results = data.getAsJsonArray("results")

        val sb = StringBuilder()
        sb.appendLine("## Top Attractions in $location")

        if (results == null || results.size() == 0) {
            sb.appendLine("No places found in $location.")
        } else {
            for (i in 0 until minOf(results.size(), 6)) {
                val place = results[i].asJsonObject

                val name = place.getAsJsonObject("displayName")?.safeString("text") ?: "Place"
                val ratingRaw = place.safeDouble("rating") ?: 0.0
                val ratingStr = if (ratingRaw > 0) "%.1f".format(ratingRaw) else "N/A"
                val fullStars = ratingRaw.toInt().coerceIn(0, 5)
                val emptyStars = 5 - fullStars
                val starsDisplay = if (ratingRaw > 0) "â˜…".repeat(fullStars) + "â˜†".repeat(emptyStars) else ""
                val reviewCount = place.safeInt("userRatingCount") ?: 0
                val mapsUri = place.safeString("googleMapsUri") ?: ""
                val websiteUri = place.safeString("websiteUri") ?: ""
                val address = place.safeString("formattedAddress") ?: ""
                val photoUri = place.safeString("photoUri") ?: ""

                val typeList = place.getAsJsonArray("types")
                    ?.mapNotNull { it.takeIf { e -> !e.isJsonNull }?.asString }
                    ?.filter { t ->
                        t !in setOf("point_of_interest", "establishment", "store",
                            "food", "restaurant")
                    }
                    ?.map { it.replace("_", " ").replaceFirstChar { c -> c.uppercase() } }
                    ?.take(4) ?: emptyList()

                val editorial = place.getAsJsonObject("editorialSummary")
                    ?.safeString("text")?.trim()
                    ?.takeIf { it.isNotBlank() }

                val openingHours = place.getAsJsonObject("regularOpeningHours")
                val openNow = openingHours?.get("openNow")?.takeIf { !it.isJsonNull }?.asBoolean
                val openStatus = when (openNow) {
                    true -> "Open now"
                    false -> "Closed now"
                    null -> null
                }
                val todayHours = openingHours?.getAsJsonArray("weekdayDescriptions")
                    ?.let { arr ->
                        val dayIndex = (java.util.Calendar.getInstance()
                            .get(java.util.Calendar.DAY_OF_WEEK) + 5) % 7
                        if (dayIndex < arr.size()) arr[dayIndex].asString?.substringAfter(":")?.trim() else null
                    }

                val reviewSnippet = place.getAsJsonArray("reviews")
                    ?.firstOrNull()?.asJsonObject
                    ?.getAsJsonObject("text")?.get("text")?.asString
                    ?.trim()
                    ?.takeIf { it.isNotBlank() && it != "null" }
                    ?.take(140)

                sb.appendLine()
                sb.appendLine("## ${i + 1}. $name")
                if (photoUri.isNotBlank()) {
                    sb.appendLine("Media: Image=$photoUri Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/geo-alt.svg")
                } else {
                    sb.appendLine("Media: Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/geo-alt.svg")
                }

                if (typeList.isNotEmpty()) {
                    sb.appendLine("Tags: ${typeList.joinToString(" | ")}")
                }

                if (ratingRaw > 0) sb.appendLine("$starsDisplay $ratingStr ($reviewCount reviews)")

                val hoursLine = buildString {
                    if (openStatus != null) append(openStatus)
                    if (todayHours != null) {
                        if (isNotEmpty()) append(" Â· ")
                        append("Today: $todayHours")
                    }
                }
                if (hoursLine.isNotBlank()) sb.appendLine(hoursLine)

                if (editorial != null) sb.appendLine("- $editorial")

                if (mapsUri.isNotBlank()) sb.appendLine("Action: [Button: View on Maps] $mapsUri")
                if (websiteUri.isNotBlank()) sb.appendLine("Action: [Button: Visit Website] $websiteUri")
            }
        }

        sb.appendLine()
        sb.appendLine("## Sources")
        sb.appendLine("- Google Places: https://maps.google.com/")

        return sb.toString()
    }

    private fun buildNewsFallback(data: JsonObject): String {
        val topic = data.safeString("topic") ?: "Top Headlines"
        val articles = data.getAsJsonArray("results")

        val sb = StringBuilder()
        sb.appendLine("## $topic")

        if (articles == null || articles.size() == 0) {
            sb.appendLine("No articles found.")
        } else {
            for (i in 0 until minOf(articles.size(), 8)) {
                val article = articles[i].asJsonObject
                val title = article.safeString("title") ?: "Article"
                val source = article.safeString("source_name")
                    ?: article.safeString("source_id")
                    ?: ""
                val sourceLabel = source.ifBlank { "Unknown source" }
                val publishedAt = formatNewsPublishedDate(article.safeString("pubDate"))
                val description = article.safeString("description")?.take(150) ?: ""
                val url = article.safeString("link") ?: ""
                val imageUrl = article.safeString("image_url")
                    ?.takeIf { it.isNotBlank() }
                val categories = article.getAsJsonArray("category")
                    ?.mapNotNull { it.takeIf { e -> !e.isJsonNull }?.asString?.takeIf { c -> c.isNotBlank() && c != "top" } }
                    ?.map { it.replaceFirstChar { c -> c.uppercase() } }
                    ?.take(3) ?: emptyList()

                sb.appendLine()
                sb.appendLine("## ${i + 1}. $title")
                if (imageUrl != null) {
                    sb.appendLine("Media: Image=$imageUrl Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/newspaper.svg")
                } else {
                    sb.appendLine("Media: Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/newspaper.svg")
                }
                if (categories.isNotEmpty()) {
                    sb.appendLine("Tags: ${categories.joinToString(" | ")}")
                }
                val sourceAndTime = buildString {
                    append("- **$sourceLabel**")
                    if (publishedAt.isNotBlank()) {
                        append(" Â· ")
                        append(publishedAt)
                    }
                }
                sb.appendLine(sourceAndTime)
                if (description.isNotBlank() && description != "null") {
                    sb.appendLine("- $description")
                }
                if (url.isNotBlank() && url != "null") {
                    sb.appendLine("Action: [Button: Read Article] $url")
                }
            }
        }

        sb.appendLine()
        sb.appendLine("## Sources")
        sb.appendLine("- NewsData.io: https://newsdata.io/")

        return sb.toString()
    }

    private fun formatNewsPublishedDate(raw: String?): String {
        val value = raw?.trim().orEmpty()
        if (value.isBlank()) {
            return ""
        }

        val formatter = java.time.format.DateTimeFormatter.ofPattern("dd MMM yyyy, hh:mm a")
        val zoned = runCatching { java.time.ZonedDateTime.parse(value) }.getOrNull()
            ?: runCatching { java.time.OffsetDateTime.parse(value).toZonedDateTime() }.getOrNull()
        if (zoned != null) {
            return zoned.format(formatter)
        }

        val local = runCatching { java.time.LocalDateTime.parse(value) }.getOrNull()
        if (local != null) {
            return local.format(formatter)
        }

        val customPatterns = listOf(
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-dd HH:mm",
            "yyyy/MM/dd HH:mm:ss",
            "yyyy/MM/dd HH:mm"
        )
        customPatterns.forEach { pattern ->
            val parsed = runCatching {
                java.time.LocalDateTime.parse(
                    value,
                    java.time.format.DateTimeFormatter.ofPattern(pattern)
                )
            }.getOrNull()
            if (parsed != null) {
                return parsed.format(formatter)
            }
        }

        return if (value.length > 24) value.take(24) else value
    }
}

