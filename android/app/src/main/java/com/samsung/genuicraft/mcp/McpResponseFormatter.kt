package com.samsung.genuicraft.mcp

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

        // For restaurants and places, use the deterministic Kotlin formatter directly.
        // This guarantees photos, ratings, reviews, and action buttons are always present
        // instead of relying on LLM output which may omit them.
        if (mcpResult.domain == McpSettings.Domain.RESTAURANTS || mcpResult.domain == McpSettings.Domain.PLACES) {
            val fallback = buildFallbackResponse(mcpResult.domain, mcpResult.data, queryText)
            return FormatResult(
                formattedResponse = fallback,
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
                formattedResponse = fallback,
                error = null,
                streamDurationMs = response.streamDurationMs
            )
        }

        val text = response.text.trim()
        if (text.isBlank()) {
            val fallback = buildFallbackResponse(mcpResult.domain, mcpResult.data, queryText)
            return FormatResult(
                formattedResponse = fallback,
                error = null,
                streamDurationMs = response.streamDurationMs
            )
        }

        return FormatResult(
            formattedResponse = text,
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
The data above is LIVE and REAL — present it as authoritative current information, not as examples or samples.
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
   **Rating:** <rating> ★★★★ (<reviewCount> reviews)
   **Address:** <address>
   **Review:** "<first review text snippet>"
   Action: [Button: View on Maps] <googleMapsUri>
   Action: [Button: Visit Website] <websiteUri>

   CRITICAL for restaurants/places:
   - If a "photoUri" field exists in the data, ALWAYS include it as Media: Image=<photoUri>
   - If "reviews" array exists, include the first review text (truncated to 120 chars)
   - If "googleMapsUri" exists, include Action: [Button: View on Maps] <url>
   - If "websiteUri" exists, include Action: [Button: Visit Website] <url>
   - Show rating as numeric value plus ★ star characters
   - Do NOT use a table for restaurants/places — use individual card blocks
6) For hotels, use option cards with rating, price, and book button.
7) Include a Sources section with real URLs when applicable.
8) Keep sections concise — prefer cards, tables, and bullets over long paragraphs.
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
        return buildFallbackResponse(mcpResult.domain, mcpResult.data, queryText)
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
        val daily = data.getAsJsonObject("daily")

        val sb = StringBuilder()
        sb.appendLine("## Weather in $location${if (country.isNotBlank()) ", $country" else ""}")
        sb.appendLine()

        val dates = daily?.getAsJsonArray("time")
        val maxTemps = daily?.getAsJsonArray("temperature_2m_max")
        val minTemps = daily?.getAsJsonArray("temperature_2m_min")
        val precip = daily?.getAsJsonArray("precipitation_probability_max")
        val codes = daily?.getAsJsonArray("weather_code")
        // Unified weather table — triggers NativeWeatherIntentModule for weather-app style rendering
        sb.appendLine("| Day | Condition | Temp | High | Low | Rain % | Wind | Humidity | UV |")
        sb.appendLine("|-----|-----------|------|------|-----|--------|------|----------|-----|")

        // Today row: populated from current weather + daily[0]
        run {
            val weatherCode = current?.safeInt("weather_code") ?: codes?.get(0)?.safeInt() ?: -1
            val condition = wmoCodeToCondition(weatherCode)
            val currentTemp = current?.safeString("temperature_2m")?.let { "${it}°C" } ?: "—"
            val todayHigh = maxTemps?.get(0)?.safeString()?.let { "${it}°C" } ?: "—"
            val todayLow = minTemps?.get(0)?.safeString()?.let { "${it}°C" } ?: "—"
            val todayRain = precip?.get(0)?.safeString()?.let { "${it}%" } ?: "—"
            val wind = current?.safeString("wind_speed_10m")?.let { "${it} km/h" } ?: "—"
            val humidity = current?.safeString("relative_humidity_2m")?.let { "${it}%" } ?: "—"
            val uv = current?.safeDouble("uv_index")?.let { "${"%.0f".format(it)}" } ?: "—"
            sb.appendLine("| Today | $condition | $currentTemp | $todayHigh | $todayLow | $todayRain | $wind | $humidity | $uv |")
        }

        // Forecast rows (skip today at i=0 since it's already included above)
        if (dates != null && maxTemps != null && minTemps != null) {
            val count = minOf(dates.size(), maxTemps.size(), minTemps.size(), 7)
            for (i in 1 until count) {
                val date = dates[i].safeString() ?: continue
                val high = maxTemps[i].safeString()?.let { "${it}°C" } ?: "—"
                val low = minTemps[i].safeString()?.let { "${it}°C" } ?: "—"
                val rain = precip?.get(i)?.safeString()?.let { "${it}%" } ?: "—"
                val dayCode = codes?.get(i)?.safeInt() ?: -1
                val dayCondition = wmoCodeToCondition(dayCode)
                val dayLabel = try {
                    val parsed = java.time.LocalDate.parse(date)
                    parsed.dayOfWeek.name.take(3).lowercase().replaceFirstChar { it.uppercase() }
                } catch (_: Exception) { date }
                sb.appendLine("| $dayLabel | $dayCondition | — | $high | $low | $rain | — | — | — |")
            }
        }

        sb.appendLine()
        sb.appendLine("## Sources")
        sb.appendLine("- Open-Meteo Weather API: https://open-meteo.com/")

        return sb.toString()
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

private fun buildFlightsFallback(data: JsonObject): String {
        val origin = data.safeString("origin") ?: "?"
        val destination = data.safeString("destination") ?: "?"
        val flights = data.getAsJsonArray("flights")

        val sb = StringBuilder()
        sb.appendLine("## Flights from $origin to $destination")
        sb.appendLine("Media: Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/airplane.svg")

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
                    val airlines = route?.get(0)?.asJsonObject?.safeString("airline") ?: "—"
                    val stops = if (route != null) route.size() - 1 else 0
                    val stopsStr = if (stops <= 0) "Non-stop" else "$stops stop${if (stops > 1) "s" else ""}"
                    val firstLeg = route?.get(0)?.asJsonObject
                    val lastLeg = route?.get(route.size() - 1)?.asJsonObject
                    val depObj = firstLeg?.getAsJsonObject("departure_airport")
                    val arrObj = lastLeg?.getAsJsonObject("arrival_airport")
                    val dep = depObj?.safeString("time")?.substringAfter(" ")?.take(5) ?: "—"
                    val arr = arrObj?.safeString("time")?.substringAfter(" ")?.take(5) ?: "—"
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
        val results = data.getAsJsonArray("results")

        val sb = StringBuilder()
        sb.appendLine("## Top Restaurants in $location")
        sb.appendLine("Media: Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/cup-hot.svg")

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
                val starsDisplay = if (ratingRaw > 0) "★".repeat(fullStars) + "☆".repeat(emptyStars) else ""
                val reviewCount = biz.safeInt("userRatingCount") ?: 0
                val mapsUri = biz.safeString("googleMapsUri") ?: ""
                val websiteUri = biz.safeString("websiteUri") ?: ""
                val address = biz.safeString("formattedAddress") ?: ""
                val photoUri = biz.safeString("photoUri") ?: ""
                val priceLevel = when (biz.safeString("priceLevel")) {
                    "PRICE_LEVEL_INEXPENSIVE" -> "₹"
                    "PRICE_LEVEL_MODERATE" -> "₹₹"
                    "PRICE_LEVEL_EXPENSIVE" -> "₹₹₹"
                    "PRICE_LEVEL_VERY_EXPENSIVE" -> "₹₹₹₹"
                    else -> ""
                }

                // Tags: filter noise types, format as backtick spans for chip-like rendering
                val typeList = biz.getAsJsonArray("types")
                    ?.mapNotNull { it.takeIf { e -> !e.isJsonNull }?.asString }
                    ?.filter { t ->
                        t !in setOf("restaurant", "food", "point_of_interest",
                            "establishment", "place_of_worship", "store")
                    }
                    ?.map { it.replace("_", " ").replaceFirstChar { c -> c.uppercase() } }
                    ?.take(4) ?: emptyList()

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
                            .get(java.util.Calendar.DAY_OF_WEEK) + 5) % 7  // 0=Mon…6=Sun
                        if (dayIndex < arr.size()) arr[dayIndex].asString?.substringAfter(":")?.trim() else null
                    }

                // First user review snippet (from same API call)
                val reviewSnippet = biz.getAsJsonArray("reviews")
                    ?.firstOrNull()?.asJsonObject
                    ?.getAsJsonObject("text")?.get("text")?.asString
                    ?.trim()
                    ?.takeIf { it.isNotBlank() && it != "null" }
                    ?.take(140)

                // ── Card block ──────────────────────────────────────────
                sb.appendLine()
                sb.appendLine("## ${i + 1}. $name")
                if (photoUri.isNotBlank()) {
                    sb.appendLine("Media: Image=$photoUri Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/shop.svg")
                } else {
                    sb.appendLine("Media: Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/shop.svg")
                }

                // Tags: pipe-separated format so Stage 3 renders as chip row
                if (typeList.isNotEmpty()) {
                    sb.appendLine("Tags: ${typeList.joinToString(" | ")}")
                }

                // Rating row with price level
                val ratingLine = buildString {
                    if (ratingRaw > 0) append("$starsDisplay $ratingStr ($reviewCount reviews)")
                    if (priceLevel.isNotBlank()) {
                        if (isNotEmpty()) append("  ·  ")
                        append(priceLevel)
                    }
                }
                if (ratingLine.isNotBlank()) sb.appendLine(ratingLine)

                // Open status + today's hours
                val hoursLine = buildString {
                    if (openStatus != null) append(openStatus)
                    if (todayHours != null) {
                        if (isNotEmpty()) append(" · ")
                        append("Today: $todayHours")
                    }
                }
                if (hoursLine.isNotBlank()) sb.appendLine(hoursLine)

                // Editorial summary — concise description from Google
                if (editorial != null) sb.appendLine("- $editorial")

                // Action buttons
                if (mapsUri.isNotBlank()) sb.appendLine("Action: [Button: View on Maps] $mapsUri")
                if (websiteUri.isNotBlank()) sb.appendLine("Action: [Button: Visit Website] $websiteUri")
            }
        }

        sb.appendLine()
        sb.appendLine("## Sources")
        sb.appendLine("- Google Places: https://maps.google.com/")

        return sb.toString()
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
            val hotelClass = hotel.safeString("hotel_class") ?: ""     // e.g. "5-star hotel"
            val ratingRaw = hotel.safeDouble("overall_rating") ?: 0.0
            val ratingStr = if (ratingRaw > 0) "%.1f".format(ratingRaw) else ""
            val reviewCount = hotel.safeInt("reviews") ?: 0
            val description = hotel.safeString("description")?.trim()
            // thumbnail is inside images[0].thumbnail, not at property level
            val thumbnail = hotel.getAsJsonArray("images")
                ?.firstOrNull()?.asJsonObject
                ?.safeString("thumbnail")
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
            // Image first so it renders as the card hero image
            if (thumbnail.isNotBlank()) sb.appendLine("Media: Image=$thumbnail")
            if (hotelClass.isNotBlank()) sb.appendLine("**$hotelClass**")
            if (ratingStr.isNotBlank()) {
                val reviewPart = if (reviewCount > 0) " ($reviewCount reviews)" else ""
                sb.appendLine("⭐ $ratingStr$reviewPart")
            }
            if (price.isNotBlank()) sb.appendLine("**Price:** From $price / night")
            if (!description.isNullOrBlank()) sb.appendLine(description)
            val checkInfo = buildString {
                if (checkIn.isNotBlank()) append("Check-in: $checkIn")
                if (checkOut.isNotBlank()) {
                    if (isNotEmpty()) append(" · ")
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
        sb.appendLine("Media: Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/map.svg")

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
                val starsDisplay = if (ratingRaw > 0) "★".repeat(fullStars) + "☆".repeat(emptyStars) else ""
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
                        if (isNotEmpty()) append(" · ")
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
        sb.appendLine("Media: Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/newspaper.svg")

        if (articles == null || articles.size() == 0) {
            sb.appendLine("No articles found.")
        } else {
            for (i in 0 until minOf(articles.size(), 8)) {
                val article = articles[i].asJsonObject
                val title = article.safeString("title") ?: "Article"
                val source = article.safeString("source_name")
                    ?: article.safeString("source_id") ?: ""
                val publishedAt = article.safeString("pubDate")?.take(16) ?: ""
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
                sb.appendLine("- $source · $publishedAt")
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
}
