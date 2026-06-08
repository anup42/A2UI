package com.samsung.genuicraft.mcp

import android.net.Uri
import android.util.Log
import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.samsung.genuicraft.inference.InferenceBackend
import java.net.URLEncoder
import java.nio.charset.StandardCharsets

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
        normalized = normalized
            .replace("Ã‚Â°C", "°C")
            .replace("Â°C", "°C")
            .replace("â€”", "—")
            .replace("Ã‚Â·", "·")
            .replace("Â·", "·")
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
            mcpResult.domain == McpSettings.Domain.FLIGHTS ||
            mcpResult.domain == McpSettings.Domain.RESTAURANTS ||
            mcpResult.domain == McpSettings.Domain.HOTELS ||
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
   - If "reservable" is true, include Action: [Button: Reserve Table] using websiteUri first, otherwise googleMapsLinks.placeUri/googleMapsUri.
   - Do NOT invent a direct reservation URL when the API only provides a Maps place link.
   - If "googleMapsUri" exists, include Action: [Button: View on Maps] <url>
   - If "websiteUri" exists, include Action: [Button: Visit Website] <url>
   - Show rating as numeric value plus â˜… star characters
   - Do NOT use a table for restaurants/places â€” use individual card blocks
6) For hotels, use option cards with rating, price, and book button.
7) Include a Sources section with real URLs when applicable.
8) Keep sections concise â€” prefer cards, tables, and bullets over long paragraphs.
9) For weather: use one forecast table only. Put current/today values in the first row and do not add a separate current conditions block.
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

    private fun JsonObject.safeBoolean(key: String): Boolean? =
        get(key)?.takeIf { !it.isJsonNull }?.let { runCatching { it.asBoolean }.getOrNull() }

    /** Generates a simple structured response when LLM formatting fails. */
    private fun buildFallbackResponse(domain: McpSettings.Domain, data: JsonObject, queryText: String): String {
        return when (domain) {
            McpSettings.Domain.WEATHER -> buildWeatherFallback(data)
            McpSettings.Domain.FLIGHTS -> buildFlightsFallback(data)
            McpSettings.Domain.RESTAURANTS -> buildRestaurantsFallback(data)
            McpSettings.Domain.HOTELS -> buildHotelsFallback(data)
            McpSettings.Domain.PLACES -> buildPlacesFallback(data, queryText)
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
        sb.appendLine("Weather forecast table (domain: weather, preferredPresentation: cards).")
        sb.appendLine()

        sb.appendLine("| Day | Date | Condition | Temp | High | Low | Feels Like | Rain Chance | Rain | Wind | Gusts | Humidity | UV | Best Window | Morning | Afternoon | Evening | Night | What to wear |")
        sb.appendLine("|-----|------|-----------|------|------|-----|------------|-------------|------|------|-------|----------|----|-------------|---------|-----------|---------|-------|--------------|")

        val requestedForecastDays = data.safeInt("requested_forecast_days")?.coerceIn(1, 16) ?: 16
        val rowCount = listOfNotNull(dates?.size(), maxTemps?.size(), minTemps?.size())
            .minOrNull()
            ?.coerceAtMost(requestedForecastDays)
            ?: 0
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
        sb.appendLine()
        sb.appendLine("## Quick Actions")
        sb.appendLine("Action: [Button: Open Forecast Source] https://open-meteo.com/en/docs")

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
        val outboundDate = data.safeString("outbound_date") ?: ""
        val returnDate = data.safeString("return_date")?.takeIf { it.isNotBlank() }
        val currency = data.safeString("currency") ?: "USD"
        val bestFlights = data.getAsJsonArray("flights") ?: JsonArray()
        val otherFlights = data.getAsJsonArray("other_flights") ?: JsonArray()

        val sb = StringBuilder()
        sb.appendLine("## Flights from $origin to $destination")
        sb.appendLine()
        val context = buildList {
            outboundDate.takeIf { it.isNotBlank() }?.let { add("Depart $it") }
            returnDate?.let { add("Return $it") }
            data.safeString("travel_class")?.takeIf { it.isNotBlank() }?.let { add("Class $it") }
            data.safeString("adults")?.takeIf { it.isNotBlank() }?.let { add("$it adult${if (it == "1") "" else "s"}") }
        }
        if (context.isNotEmpty()) {
            sb.appendLine(context.joinToString(" - "))
            sb.appendLine()
        }

        val rows = flightRowsForFormatter(bestFlights, groupLabel = "Best") +
            flightRowsForFormatter(otherFlights, groupLabel = "Other")

        if (rows.isEmpty()) {
            sb.appendLine("No flights found for this route.")
        } else {
            sb.appendLine("Google Flights returned ${bestFlights.size()} best options and ${otherFlights.size()} other options. Each row keeps airline logos, prices, layovers, and action links for native flight cards.")
            sb.appendLine()
            sb.appendLine("| Airline | Departure | Arrival | Duration | Stops | Fare | Status | Legs | Layover | Carbon | Booking | Airline Logo | Booking URL | Action Label |")
            sb.appendLine("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            rows.take(8).forEach { row ->
                val bookingUrl = flightSearchUrl(origin, destination, outboundDate)
                sb.appendLine(
                    "| ${tableCell(row.airline)} | ${tableCell(row.departure)} | ${tableCell(row.arrival)} | " +
                        "${tableCell(row.duration)} | ${tableCell(row.stops)} | ${tableCell(formatFlightPrice(row.price, currency))} | " +
                        "${tableCell(row.status)} | ${tableCell(row.legs)} | ${tableCell(row.layover)} | " +
                        "${tableCell(row.carbon)} | ${tableCell(row.bookingStatus)} | ${tableCell(row.logoUrl)} | " +
                        "${tableCell(bookingUrl)} | View fare |"
                )
            }
        }

        sb.appendLine()
        sb.appendLine("## Sources")
        sb.appendLine("- Google Flights via SerpApi: https://serpapi.com/")
        sb.appendLine("- Google Flights: https://www.google.com/travel/flights")
        sb.appendLine()
        sb.appendLine("## Quick Actions")
        sb.appendLine("Action: [Button: Search Google Flights] ${flightSearchUrl(origin, destination, outboundDate)}")

        return sb.toString()
    }

    private data class FlightFormatterRow(
        val airline: String,
        val departure: String,
        val arrival: String,
        val duration: String,
        val stops: String,
        val price: String,
        val status: String,
        val legs: String,
        val layover: String,
        val carbon: String,
        val bookingStatus: String,
        val logoUrl: String
    )

    private fun flightRowsForFormatter(flights: JsonArray, groupLabel: String): List<FlightFormatterRow> {
        return (0 until flights.size()).mapNotNull { index ->
            val flightObj = flights[index].takeIf { it.isJsonObject }?.asJsonObject ?: return@mapNotNull null
            val legs = flightObj.getAsJsonArray("flights") ?: JsonArray()
            val firstLeg = legs.firstOrNull()?.takeIf { it.isJsonObject }?.asJsonObject ?: return@mapNotNull null
            val lastLeg = legs.lastOrNull()?.takeIf { it.isJsonObject }?.asJsonObject ?: firstLeg
            val airline = flightAirlineLabel(legs)
            val departure = flightAirportCell(firstLeg.getAsJsonObject("departure_airport"))
            val arrival = flightAirportCell(lastLeg.getAsJsonObject("arrival_airport"))
            val totalDuration = flightObj.safeString("total_duration") ?: firstLeg.safeString("duration") ?: ""
            val layovers = flightObj.getAsJsonArray("layovers")
            val stopCount = layovers?.size() ?: (legs.size() - 1).coerceAtLeast(0)
            val stops = when {
                stopCount <= 0 -> "Non-stop"
                stopCount == 1 -> "1 stop"
                else -> "$stopCount stops"
            }
            val layoverStatus = flightLayoverSummary(layovers)
            val status = listOfNotNull(
                "$groupLabel flight".takeIf { groupLabel.isNotBlank() },
                flightObj.safeString("type")?.takeIf { it.isNotBlank() }
            ).joinToString(" - ")
            FlightFormatterRow(
                airline = airline,
                departure = departure,
                arrival = arrival,
                duration = formatFlightMinutes(totalDuration),
                stops = stops,
                price = flightObj.safeString("price") ?: "",
                status = status,
                legs = flightLegsSummary(legs),
                layover = layoverStatus,
                carbon = flightCarbonSummary(flightObj.getAsJsonObject("carbon_emissions")),
                bookingStatus = if (!flightObj.safeString("booking_token").isNullOrBlank()) "Booking token available" else "",
                logoUrl = flightObj.safeString("airline_logo") ?: firstLeg.safeString("airline_logo") ?: ""
            )
        }
    }

    private fun flightAirlineLabel(legs: JsonArray): String {
        val airlines = linkedSetOf<String>()
        val flightNumbers = mutableListOf<String>()
        legs.forEach { legElement ->
            val leg = legElement.takeIf { it.isJsonObject }?.asJsonObject ?: return@forEach
            leg.safeString("airline")?.trim()?.takeIf { it.isNotBlank() }?.let(airlines::add)
            leg.safeString("flight_number")?.trim()?.takeIf { it.isNotBlank() }?.let(flightNumbers::add)
        }
        val airlineText = airlines.joinToString(" + ").ifBlank { "Flight" }
        val numbers = flightNumbers.distinct().take(2).joinToString(" / ")
        return if (numbers.isBlank()) airlineText else "$airlineText - $numbers"
    }

    private fun flightAirportCell(airport: JsonObject?): String {
        if (airport == null) return ""
        val id = airport.safeString("id") ?: ""
        val time = flightDisplayTime(airport.safeString("time"))
        return listOf(id, time).filter { it.isNotBlank() }.joinToString(" ")
    }

    private fun flightDisplayTime(raw: String?): String {
        val value = raw?.trim().orEmpty()
        if (value.isBlank()) return ""
        val timePart = value.substringAfter(' ', value).takeLast(5)
        return if (Regex("""\d{1,2}:\d{2}""").matches(timePart)) timePart else value
    }

    private fun flightLayoverSummary(layovers: JsonArray?): String {
        if (layovers == null || layovers.size() == 0) return ""
        return (0 until layovers.size()).mapNotNull { index ->
            val layover = layovers[index].takeIf { it.isJsonObject }?.asJsonObject ?: return@mapNotNull null
            val airport = layover.safeString("id") ?: layover.safeString("name") ?: ""
            val duration = formatFlightMinutes(layover.safeString("duration") ?: "")
            val overnight = if (layover.safeBoolean("overnight") == true) " overnight" else ""
            listOf(airport, duration).filter { it.isNotBlank() }.joinToString(" ").plus(overnight).trim()
        }.joinToString("; ")
    }

    private fun flightLegsSummary(legs: JsonArray): String {
        return (0 until legs.size()).mapNotNull { index ->
            val leg = legs[index].takeIf { it.isJsonObject }?.asJsonObject ?: return@mapNotNull null
            val departure = leg.getAsJsonObject("departure_airport")
            val arrival = leg.getAsJsonObject("arrival_airport")
            val departureText = flightAirportCell(departure)
            val arrivalText = flightAirportCell(arrival)
            if (departureText.isBlank() || arrivalText.isBlank()) return@mapNotNull null
            val details = listOfNotNull(
                leg.safeString("flight_number")?.takeIf { it.isNotBlank() },
                leg.safeString("airplane")?.takeIf { it.isNotBlank() }
            ).joinToString(", ")
            if (details.isBlank()) {
                "$departureText -> $arrivalText"
            } else {
                "$departureText -> $arrivalText ($details)"
            }
        }.joinToString("; ")
    }

    private fun flightCarbonSummary(carbon: JsonObject?): String {
        if (carbon == null) return ""
        val difference = carbon.safeString("difference_percent")?.trim()?.takeIf { it.isNotBlank() } ?: return ""
        val numeric = difference.toDoubleOrNull()
        val formatted = if (numeric != null) {
            val sign = if (numeric > 0) "+" else ""
            "$sign${if (kotlin.math.abs(numeric - numeric.toInt()) < 0.05) numeric.toInt().toString() else "%.1f".format(numeric)}%"
        } else {
            difference
        }
        return "$formatted CO2 vs typical"
    }

    private fun formatFlightMinutes(raw: String): String {
        val minutes = raw.trim().toIntOrNull() ?: return raw
        val hours = minutes / 60
        val mins = minutes % 60
        return when {
            hours > 0 && mins > 0 -> "${hours}h ${mins}m"
            hours > 0 -> "${hours}h"
            else -> "${mins}m"
        }
    }

    private fun formatFlightPrice(raw: String, currency: String): String {
        val trimmed = raw.trim()
        if (trimmed.isBlank()) return ""
        if (Regex("""(?i)(?:₹|\$|€|£|inr|usd|eur|gbp|rs\.)""").containsMatchIn(trimmed)) {
            return trimmed
        }
        val numeric = trimmed.toDoubleOrNull()
        val amount = numeric?.let { value ->
            if (kotlin.math.abs(value - value.toInt()) < 0.05) "%,d".format(value.toInt()) else "%,.2f".format(value)
        } ?: trimmed
        val symbol = when (currency.uppercase(java.util.Locale.US)) {
            "INR" -> "₹"
            "USD" -> "\u0024"
            "EUR" -> "€"
            "GBP" -> "£"
            else -> "${currency.uppercase(java.util.Locale.US)} "
        }
        return "$symbol$amount /adult"
    }

    private fun flightSearchUrl(origin: String, destination: String, outboundDate: String): String {
        val query = listOf("Flights", "from", origin, "to", destination, outboundDate.takeIf { it.isNotBlank() }?.let { "on $it" })
            .filterNotNull()
            .joinToString(" ")
        val encoded = URLEncoder.encode(query, StandardCharsets.UTF_8.name()).replace("+", "%20")
        return "https://www.google.com/travel/flights?q=$encoded"
    }

    private fun mapsSearchUrl(query: String): String {
        val encoded = Uri.encode(query.ifBlank { "Google Maps" })
        return "https://www.google.com/maps/search/?api=1&query=$encoded"
    }

    private fun buildRestaurantsFallback(data: JsonObject): String {
        val location = data.safeString("location") ?: "your area"
        val provider = data.safeString("provider") ?: "google_places"
        val results = data.getAsJsonArray("results")

        fun cell(raw: String?): String = raw.orEmpty()
            .replace('\n', ' ')
            .replace("|", "/")
            .trim()

        val sb = StringBuilder()
        sb.appendLine("## Top Restaurants in $location")

        if (results == null || results.size() == 0) {
            sb.appendLine("No restaurants found in $location.")
        } else {
            sb.appendLine("These restaurants are from Google Places data. A primary photo URL, ratings, addresses, hours, and action links are attached to each row so the UI can show Bixby-style result cards.")
            sb.appendLine()
            sb.appendLine("| Restaurant | Rating | Reviews | Price | Status / Hours | Address | Tags | Amenities | Description | Photo URL | Photos | Book URL | Action Label | Maps URL | Website URL | Phone |")
            sb.appendLine("|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|")

            for (i in 0 until minOf(results.size(), 6)) {
                val biz = results[i].asJsonObject
                val name = biz.getAsJsonObject("displayName")?.safeString("text") ?: "Restaurant"
                val ratingRaw = biz.safeDouble("rating") ?: 0.0
                val ratingStr = if (ratingRaw > 0) "%.1f".format(ratingRaw) else ""
                val reviewCount = biz.safeInt("userRatingCount") ?: 0
                val mapsUri = biz.getAsJsonObject("googleMapsLinks")?.safeString("placeUri")
                    ?: biz.safeString("googleMapsUri")
                    ?: ""
                val websiteUri = biz.safeString("websiteUri") ?: ""
                val phone = biz.safeString("nationalPhoneNumber")
                    ?: biz.safeString("internationalPhoneNumber")
                    ?: ""
                val address = biz.safeString("formattedAddress") ?: ""
                val photoUris = biz.getAsJsonArray("photoUris")
                    ?.mapNotNull { it.safeString()?.trim()?.takeIf(String::isNotBlank) }
                    ?.take(5)
                    .orEmpty()
                    .ifEmpty { listOfNotNull(biz.safeString("photoUri")?.trim()?.takeIf(String::isNotBlank)) }
                    .take(5)
                val distanceMeters = biz.safeDouble("distance")
                val priceLevel = when (biz.safeString("priceLevel")) {
                    "PRICE_LEVEL_INEXPENSIVE" -> "Inexpensive"
                    "PRICE_LEVEL_MODERATE" -> "Moderate"
                    "PRICE_LEVEL_EXPENSIVE" -> "Expensive"
                    "PRICE_LEVEL_VERY_EXPENSIVE" -> "Very expensive"
                    else -> ""
                }

                val cuisineList = biz.getAsJsonArray("cuisineTags")
                    ?.mapNotNull { it.safeString()?.trim()?.takeIf { tag -> tag.isNotBlank() } }
                    ?: emptyList()
                val primaryType = biz.getAsJsonObject("primaryTypeDisplayName")
                    ?.safeString("text")
                    ?.trim()
                    ?.takeIf(String::isNotBlank)
                val typeList = cuisineList + listOfNotNull(primaryType) + (biz.getAsJsonArray("types")
                    ?.mapNotNull { it.takeIf { e -> !e.isJsonNull }?.asString }
                    ?.filter { t ->
                        t !in setOf(
                            "restaurant", "food", "point_of_interest",
                            "establishment", "place_of_worship", "store", "catering", "catering.restaurant"
                        )
                    }
                    ?.map { it.substringAfterLast('.') }
                    ?.map { it.replace("_", " ").replaceFirstChar { c -> c.uppercase() } }
                    ?: emptyList())
                val uniqueTypeList = typeList.distinct().take(4)

                val editorial = biz.getAsJsonObject("editorialSummary")
                    ?.safeString("text")?.trim()
                    ?.takeIf { it.isNotBlank() }
                val generativeSummary = biz.getAsJsonObject("generativeSummary")
                    ?.let { summary ->
                        summary.getAsJsonObject("overview")?.safeString("text")
                            ?: summary.safeString("text")
                    }
                    ?.trim()
                    ?.takeIf { it.isNotBlank() }
                val reviewSummary = biz.getAsJsonObject("reviewSummary")
                    ?.let { summary ->
                        summary.getAsJsonObject("text")?.safeString("text")
                            ?: summary.safeString("text")
                    }
                    ?.trim()
                    ?.takeIf { it.isNotBlank() }
                val reviewSnippet = biz.getAsJsonArray("reviews")
                    ?.firstOrNull()?.asJsonObject
                    ?.getAsJsonObject("text")?.get("text")?.asString
                    ?.trim()
                    ?.takeIf { it.isNotBlank() && it != "null" }
                    ?.take(140)

                val openingHours = biz.getAsJsonObject("currentOpeningHours")
                    ?: biz.getAsJsonObject("regularOpeningHours")
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
                val distanceLabel = if (distanceMeters != null && distanceMeters > 0) {
                    if (distanceMeters >= 1000) {
                        "%.1f km away".format(distanceMeters / 1000.0)
                    } else {
                        "${distanceMeters.toInt()} m away"
                    }
                } else {
                    ""
                }
                val status = buildList {
                    openStatus?.takeIf(String::isNotBlank)?.let(::add)
                    todayHours?.takeIf(String::isNotBlank)?.let { add("Today: $it") }
                    distanceLabel.takeIf(String::isNotBlank)?.let(::add)
                }.joinToString(" / ")
                val amenities = restaurantAmenities(biz).joinToString("; ")
                val description = listOfNotNull(editorial, generativeSummary, reviewSummary, reviewSnippet)
                    .firstOrNull { it.isNotBlank() }
                    .orEmpty()
                val reservable = biz.safeBoolean("reservable") == true
                val bookUrl = websiteUri.ifBlank { mapsUri }
                val actionLabel = when {
                    reservable -> "Reserve Table"
                    websiteUri.isNotBlank() -> "Menu / Details"
                    mapsUri.isNotBlank() -> "Open in Maps"
                    else -> ""
                }
                val primaryPhoto = photoUris.firstOrNull() ?: restaurantVisualUri(name)

                sb.appendLine(
                    "| ${cell(name)} | ${cell(ratingStr)} | ${cell(if (reviewCount > 0) "$reviewCount reviews" else "")} | " +
                        "${cell(priceLevel)} | ${cell(status)} | ${cell(address)} | ${cell(uniqueTypeList.joinToString("; "))} | " +
                        "${cell(amenities)} | ${cell(description)} | ${cell(primaryPhoto)} | ${cell(photoUris.joinToString(", "))} | " +
                        "${cell(bookUrl)} | ${cell(actionLabel)} | ${cell(mapsUri)} | ${cell(websiteUri)} | ${cell(phone)} |"
                )
            }
        }

        sb.appendLine()
        sb.appendLine("## Sources")
        if (provider == "google_maps_grounding") {
            sb.appendLine("- Google Maps grounding via Vertex AI: https://maps.google.com/")
        } else {
            sb.appendLine("- Google Places: https://maps.google.com/")
        }
        sb.appendLine()
        sb.appendLine("## Quick Actions")
        sb.appendLine("Action: [Button: View Restaurants on Maps] ${mapsSearchUrl("restaurants in $location")}")

        return sb.toString()
    }

    private fun restaurantAmenities(place: JsonObject): List<String> = buildList {
        fun addIf(key: String, label: String) {
            if (place.safeBoolean(key) == true) add(label)
        }
        addIf("reservable", "Reservations")
        addIf("dineIn", "Dine-in")
        addIf("takeout", "Takeout")
        addIf("delivery", "Delivery")
        addIf("outdoorSeating", "Outdoor seating")
        addIf("goodForChildren", "Good for kids")
        addIf("goodForGroups", "Good for groups")
        addIf("servesBreakfast", "Breakfast")
        addIf("servesBrunch", "Brunch")
        addIf("servesLunch", "Lunch")
        addIf("servesDinner", "Dinner")
        addIf("servesVegetarianFood", "Vegetarian options")
        addIf("servesCoffee", "Coffee")
        addIf("servesDessert", "Dessert")
    }.distinct().take(6)

    private fun restaurantVisualUri(name: String): String {
        val encodedTitle = Uri.encode(name.ifBlank { "Restaurant" })
        return "genuicraft://visual/restaurant?title=$encodedTitle"
    }

    private fun tableCell(raw: String?): String = raw.orEmpty()
        .replace('\n', ' ')
        .replace("|", "/")
        .replace(Regex("\\s+"), " ")
        .trim()

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
        sb.appendLine("These hotels are from Google Hotels data. Photos, ratings, prices, amenities, and action links are attached to each row so the UI can render rich hotel cards.")
        sb.appendLine()

        val checkInDate = data.safeString("check_in_date") ?: ""
        val checkOutDate = data.safeString("check_out_date") ?: ""
        val adults = data.safeString("adults") ?: ""
        val contextParts = buildList {
            if (checkInDate.isNotBlank() && checkOutDate.isNotBlank()) add("$checkInDate to $checkOutDate")
            if (adults.isNotBlank()) add("$adults adults")
        }
        if (contextParts.isNotEmpty()) {
            sb.appendLine(contextParts.joinToString(" - "))
            sb.appendLine()
        }

        sb.appendLine("| Hotel | Class | Rating | Reviews | Price / Night | Description | Amenities | Check-in | Check-out | Photo URL | Photo URLs | Booking URL | Action Label | Maps URL | Website URL | Photos Data URL |")
        sb.appendLine("|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|")

        for (i in 0 until minOf(properties.size(), 8)) {
            val hotel = properties[i].asJsonObject
            val name = hotel.safeString("name") ?: continue
            val hotelClass = hotel.safeString("hotel_class") ?: ""
            val ratingRaw = hotel.safeDouble("overall_rating") ?: 0.0
            val ratingStr = if (ratingRaw > 0) "%.1f".format(ratingRaw) else ""
            val reviewCount = hotel.safeInt("reviews") ?: 0
            val description = hotel.safeString("description")?.trim() ?: ""
            val photoUrls = hotelPhotoUrls(hotel)
            val thumbnail = photoUrls.firstOrNull() ?: ""
            val link = hotel.safeString("link") ?: ""
            val checkIn = hotel.safeString("check_in_time") ?: ""
            val checkOut = hotel.safeString("check_out_time") ?: ""
            val mapsUrl = hotelMapsUrl(hotel)
            val photosDataUrl = hotel.safeString("serpapi_google_hotels_photos_link") ?: ""

            val rateObj = hotel.get("rate_per_night")?.takeIf { !it.isJsonNull }?.asJsonObject
            val price = rateObj?.safeString("lowest") ?: rateObj?.safeString("extracted_lowest") ?: ""

            val amenities = hotel.getAsJsonArray("amenities")
                ?.mapNotNull { it.safeString()?.trim()?.takeIf { a -> a.isNotBlank() } }
                ?.take(5) ?: emptyList()

            sb.appendLine(
                "| ${tableCell(name)} | ${tableCell(hotelClass)} | ${tableCell(ratingStr)} | " +
                    "${tableCell(if (reviewCount > 0) "$reviewCount reviews" else "")} | ${tableCell(price)} | " +
                    "${tableCell(description)} | ${tableCell(amenities.joinToString("; "))} | " +
                    "${tableCell(checkIn)} | ${tableCell(checkOut)} | ${tableCell(thumbnail)} | " +
                    "${tableCell(photoUrls.joinToString(", "))} | ${tableCell(link)} | " +
                    "${tableCell(if (link.isNotBlank()) "Book / Website" else "View Details")} | " +
                    "${tableCell(mapsUrl)} | ${tableCell(link)} | ${tableCell(photosDataUrl)} |"
            )
        }

        sb.appendLine()
        sb.appendLine("## Sources")
        sb.appendLine("- Google Hotels via SerpApi: https://serpapi.com/")
        sb.appendLine("- Google Hotels: https://www.google.com/travel/hotels")
        sb.appendLine()
        sb.appendLine("## Quick Actions")
        sb.appendLine("Action: [Button: Search Google Hotels] ${googleHotelsSearchUrl(queryText.ifBlank { "hotels in $location" })}")

        return sb.toString()
    }

    private fun googleHotelsSearchUrl(query: String): String {
        val encoded = URLEncoder.encode(query.ifBlank { "hotels" }, StandardCharsets.UTF_8.name()).replace("+", "%20")
        return "https://www.google.com/travel/hotels?q=$encoded"
    }

    private fun hotelPhotoUrls(hotel: JsonObject): List<String> {
        val urls = linkedSetOf<String>()
        listOf("thumbnail", "image", "image_url", "photo", "photo_url").forEach { key ->
            hotel.safeString(key)?.trim()?.takeIf { it.isNotBlank() }?.let(urls::add)
        }
        hotel.getAsJsonArray("images")?.forEach { imageEntry ->
            when {
                imageEntry.isJsonObject -> {
                    val imageObj = imageEntry.asJsonObject
                    listOf("thumbnail", "original_image", "url", "image", "photo").forEach { key ->
                        imageObj.safeString(key)?.trim()?.takeIf { it.isNotBlank() }?.let(urls::add)
                    }
                }
                !imageEntry.isJsonNull -> imageEntry.safeString()?.trim()?.takeIf { it.isNotBlank() }?.let(urls::add)
            }
        }
        return urls.take(5)
    }

    private fun hotelMapsUrl(hotel: JsonObject): String {
        val gps = hotel.getAsJsonObject("gps_coordinates") ?: return ""
        val lat = gps.safeString("latitude") ?: return ""
        val lng = gps.safeString("longitude") ?: return ""
        return "https://www.google.com/maps/search/?api=1&query=$lat,$lng"
    }

    private fun buildPlacesFallback(data: JsonObject, queryText: String): String {
        if (looksLikeVacationItineraryQuery(queryText)) {
            return buildPlacesItineraryFallback(data, queryText)
        }
        return buildPlacesAttractionsFallback(data)
    }

    private fun looksLikeVacationItineraryQuery(queryText: String): Boolean {
        val normalized = queryText.lowercase()
        return normalized.contains("itinerary") ||
            normalized.contains("itenary") ||
            normalized.contains("vacation") ||
            normalized.contains("holiday") ||
            normalized.contains("trip plan") ||
            normalized.contains("travel plan") ||
            Regex("""\b\d+\s*(day|days)\b""").containsMatchIn(normalized)
    }

    private fun requestedItineraryDays(queryText: String, maxAvailableRows: Int): Int {
        val requested = Regex("""\b(\d{1,2})\s*(?:day|days)\b""", RegexOption.IGNORE_CASE)
            .find(queryText)
            ?.groupValues
            ?.getOrNull(1)
            ?.toIntOrNull()
        return (requested ?: 4).coerceIn(1, minOf(10, maxOf(1, maxAvailableRows)))
    }

    private fun markdownTableCell(value: String?): String =
        value.orEmpty()
            .replace("\r", " ")
            .replace("\n", " ")
            .replace("|", "/")
            .replace(Regex("\\s+"), " ")
            .trim()

    private fun placeDisplayName(place: JsonObject): String =
        place.getAsJsonObject("displayName")?.safeString("text")?.trim()?.takeIf { it.isNotBlank() }
            ?: "Place"

    private fun placePrimaryType(place: JsonObject): String {
        return place.safeString("primaryTypeDisplayName")
            ?: place.getAsJsonObject("primaryTypeDisplayName")?.safeString("text")
            ?: place.getAsJsonArray("types")
                ?.mapNotNull { it.takeIf { e -> !e.isJsonNull }?.asString }
                ?.firstOrNull { type ->
                    type !in setOf("point_of_interest", "establishment", "store", "food", "restaurant")
                }
                ?.replace("_", " ")
                ?.replaceFirstChar { c -> c.uppercase() }
            ?: "Attraction"
    }

    private fun placeBestSummary(place: JsonObject): String {
        val editorial = place.getAsJsonObject("editorialSummary")
            ?.safeString("text")
            ?.trim()
            ?.takeIf { it.isNotBlank() && it != "null" }
        if (!editorial.isNullOrBlank()) return editorial.take(180)
        val reviewSnippet = place.getAsJsonArray("reviews")
            ?.firstOrNull()
            ?.asJsonObject
            ?.getAsJsonObject("text")
            ?.safeString("text")
            ?.trim()
            ?.takeIf { it.isNotBlank() && it != "null" }
        if (!reviewSnippet.isNullOrBlank()) return reviewSnippet.take(160)
        return place.safeString("formattedAddress")?.take(140).orEmpty()
    }

    private fun placeFirstPhotoUri(place: JsonObject): String =
        place.safeString("photoUri")
            ?: place.getAsJsonArray("photoUris")?.firstOrNull()?.takeIf { !it.isJsonNull }?.asString
            ?: ""

    private fun placeThemeForDay(dayIndex: Int): String {
        return when (dayIndex % 4) {
            0 -> "Gardens, heritage and central city"
            1 -> "Temples, architecture and viewpoints"
            2 -> "Museums, parks and cafe evening"
            else -> "Nature edge and relaxed finish"
        }
    }

    private fun buildPlacesItineraryFallback(data: JsonObject, queryText: String): String {
        val location = data.safeString("location") ?: "your destination"
        val results = data.getAsJsonArray("results")
        val sb = StringBuilder()
        sb.appendLine("## ${requestedItineraryDays(queryText, results?.size() ?: 4)}-day $location vacation itinerary")

        if (results == null || results.size() == 0) {
            sb.appendLine("No places found in $location.")
            return sb.toString()
        }

        val dayCount = requestedItineraryDays(queryText, results.size())
        val stopLimit = minOf(results.size(), maxOf(dayCount, minOf(dayCount * 2, 10)))
        sb.appendLine()
        sb.appendLine("Vacation itinerary table (domain: schedule, preferredPresentation: cards).")
        sb.appendLine("Keep this as the only itinerary data table. Do not add a separate metrics, facts, overview, gallery, or image section; Android derives the visual hero and day cards from these rows.")
        sb.appendLine()
        sb.appendLine("| Day | Area | Place | Category | Rating | Reviews | Summary | Image URL | Maps URL | Website URL |")
        sb.appendLine("|---|---|---|---|---|---|---|---|---|---|")
        for (i in 0 until stopLimit) {
            val place = results[i].asJsonObject
            val dayIndex = i % dayCount
            val day = "Day ${dayIndex + 1}"
            val area = placeThemeForDay(dayIndex)
            val ratingRaw = place.safeDouble("rating") ?: 0.0
            val rating = if (ratingRaw > 0) "%.1f".format(ratingRaw) else ""
            val reviews = place.safeInt("userRatingCount")?.takeIf { it > 0 }?.let { "$it reviews" }.orEmpty()
            val row = listOf(
                day,
                area,
                placeDisplayName(place),
                placePrimaryType(place),
                rating,
                reviews,
                placeBestSummary(place),
                placeFirstPhotoUri(place),
                place.safeString("googleMapsUri").orEmpty(),
                place.safeString("websiteUri").orEmpty()
            ).joinToString(prefix = "| ", separator = " | ", postfix = " |") { markdownTableCell(it) }
            sb.appendLine(row)
        }

        sb.appendLine()
        sb.appendLine("## Sources")
        sb.appendLine("- Google Places: https://maps.google.com/")
        sb.appendLine()
        sb.appendLine("## Quick Actions")
        sb.appendLine("Action: [Button: View trip places on Maps] ${mapsSearchUrl("top attractions in $location")}")
        return sb.toString()
    }

    private fun buildPlacesAttractionsFallback(data: JsonObject): String {
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
        sb.appendLine()
        sb.appendLine("## Quick Actions")
        sb.appendLine("Action: [Button: View Places on Maps] ${mapsSearchUrl("places in $location")}")

        return sb.toString()
    }

    private fun buildNewsFallback(data: JsonObject): String {
        val topic = data.safeString("topic") ?: "Top Headlines"
        val location = data.safeString("location").orEmpty()
        val country = data.safeString("country").orEmpty().uppercase()
        val language = data.safeString("language").orEmpty().uppercase()
        val articles = data.getAsJsonArray("results")

        val sb = StringBuilder()
        val heading = buildString {
            append(topic.replaceFirstChar { it.uppercase() })
            if (location.isNotBlank()) append(" in $location")
        }
        sb.appendLine("## $heading")
        val meta = listOfNotNull(
            country.takeIf { it.isNotBlank() }?.let { "Country: $it" },
            language.takeIf { it.isNotBlank() }?.let { "Language: $it" }
        )
        if (meta.isNotEmpty()) {
            sb.appendLine(meta.joinToString(" | "))
        }

        if (articles == null || articles.size() == 0) {
            sb.appendLine("No articles found.")
        } else {
            sb.appendLine()
            sb.appendLine("News results table (domain: news, preferredPresentation: cards).")
            sb.appendLine("| Article | Source | Published | Category | Summary | Image URL | Source Icon | Article URL | Source URL | Action Label |")
            sb.appendLine("|---|---|---|---|---|---|---|---|---|---|")
            for (i in 0 until minOf(articles.size(), 8)) {
                val article = articles[i].asJsonObject
                val title = article.safeString("title") ?: "Article"
                val source = article.safeString("source_name")
                    ?: article.safeString("source_id")
                    ?: ""
                val sourceLabel = source.ifBlank { "Unknown source" }
                val publishedAt = formatNewsPublishedDate(article.safeString("pubDate"))
                val description = (article.safeString("description") ?: article.safeString("content"))
                    ?.replace(Regex("""\s+"""), " ")
                    ?.take(220)
                    ?: ""
                val url = article.safeString("link") ?: ""
                val imageUrl = article.safeString("image_url")
                    ?.takeIf { it.isNotBlank() }
                    ?: ""
                val sourceIcon = article.safeString("source_icon")
                    ?.takeIf { it.isNotBlank() }
                    ?: ""
                val sourceUrl = article.safeString("source_url")
                    ?.takeIf { it.isNotBlank() }
                    ?: ""
                val categories = article.getAsJsonArray("category")
                    ?.mapNotNull { it.takeIf { e -> !e.isJsonNull }?.asString?.takeIf { c -> c.isNotBlank() && c != "top" } }
                    ?.map { it.replaceFirstChar { c -> c.uppercase() } }
                    ?.take(3) ?: emptyList()
                sb.appendLine(
                    "| ${tableCell(title)} | ${tableCell(sourceLabel)} | ${tableCell(publishedAt)} | " +
                        "${tableCell(categories.joinToString("; "))} | ${tableCell(description)} | " +
                        "${tableCell(imageUrl)} | ${tableCell(sourceIcon)} | ${tableCell(url)} | " +
                        "${tableCell(sourceUrl)} | Read Article |"
                )
            }
        }

        sb.appendLine()
        sb.appendLine("## Sources")
        sb.appendLine("- NewsData.io: https://newsdata.io/")
        sb.appendLine()
        sb.appendLine("## Quick Actions")
        sb.appendLine("Action: [Button: Open News Source] https://newsdata.io/")

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

