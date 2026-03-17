package com.samsung.genuicraft.mcp

import android.util.Log
import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.nio.charset.StandardCharsets

/**
 * MCP client that fetches real-time data from domain-specific APIs.
 * Each domain maps to a popular, well-documented public API:
 *
 * - Weather: Open-Meteo (free, no key required)
 * - Flights: Tequila Kiwi (API key required)
 * - Restaurants: Google Places (API key required)
 * - Hotels: Serpapi (API key required)
 * - Places: Google Places (API key required)
 * - News: NewsData.io (API key required)
 */
object McpClient {

    private const val LOG_TAG = "McpClient"
    private const val CONNECT_TIMEOUT = 10_000
    private const val READ_TIMEOUT = 30_000

    data class McpResult(
        val domain: McpSettings.Domain,
        val success: Boolean,
        val data: JsonObject?,
        val rawJson: String?,
        val error: String?
    )

    /**
     * Fetches real-time data for the given domain and query entities.
     * Runs on the calling thread (expected to be called from IO dispatcher).
     */
    fun fetch(
        domain: McpSettings.Domain,
        entities: Map<String, String>,
        apiKey: String,
        queryText: String
    ): McpResult {
        return try {
            when (domain) {
                McpSettings.Domain.WEATHER -> fetchWeather(entities, queryText)
                McpSettings.Domain.FLIGHTS -> fetchFlights(entities, apiKey, queryText)
                McpSettings.Domain.RESTAURANTS -> fetchRestaurants(entities, apiKey, queryText)
                McpSettings.Domain.HOTELS -> fetchHotels(entities, apiKey, queryText)
                McpSettings.Domain.PLACES -> fetchPlaces(entities, apiKey, queryText)
                McpSettings.Domain.NEWS -> fetchNews(entities, apiKey, queryText)
            }
        } catch (e: Exception) {
            Log.e(LOG_TAG, "MCP fetch failed for ${domain.key}", e)
            McpResult(
                domain = domain,
                success = false,
                data = null,
                rawJson = null,
                error = "${e.javaClass.simpleName}: ${e.message}"
            )
        }
    }

    // ── Weather: Open-Meteo (free, no API key) ──────────────────────────

    private fun fetchWeather(entities: Map<String, String>, queryText: String): McpResult {
        val location = entities["location"] ?: extractLocationFallback(queryText) ?: "New York"

        // Step 1: Geocode the location
        val geocodeUrl = "https://geocoding-api.open-meteo.com/v1/search?name=${enc(location)}&count=1&language=en&format=json"
        val geocodeResponse = httpGet(geocodeUrl)
        val geocodeJson = JsonParser.parseString(geocodeResponse).asJsonObject
        val results = geocodeJson.getAsJsonArray("results")
        if (results == null || results.size() == 0) {
            return McpResult(McpSettings.Domain.WEATHER, false, null, null, "Location not found: $location")
        }
        val place = results[0].asJsonObject
        val lat = place.get("latitude").asDouble
        val lon = place.get("longitude").asDouble
        val resolvedName = place.get("name")?.asString ?: location
        val country = place.get("country")?.asString ?: ""

        // Step 2: Fetch weather
        val weatherUrl = "https://api.open-meteo.com/v1/forecast?" +
            "latitude=$lat&longitude=$lon" +
            "&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,uv_index" +
            "&daily=temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max,sunrise,sunset" +
            "&timezone=auto&forecast_days=7"
        val weatherResponse = httpGet(weatherUrl)
        val weatherJson = JsonParser.parseString(weatherResponse).asJsonObject

        // Wrap with metadata
        val result = JsonObject().apply {
            addProperty("location", resolvedName)
            addProperty("country", country)
            addProperty("latitude", lat)
            addProperty("longitude", lon)
            add("current", weatherJson.getAsJsonObject("current"))
            add("current_units", weatherJson.getAsJsonObject("current_units"))
            add("daily", weatherJson.getAsJsonObject("daily"))
            add("daily_units", weatherJson.getAsJsonObject("daily_units"))
        }

        return McpResult(McpSettings.Domain.WEATHER, true, result, weatherResponse, null)
    }

    // ── Flights: Google Flights (via Serpapi) ─────────────────────────────────────

    private fun fetchFlights(entities: Map<String, String>, apiKey: String, queryText: String): McpResult {
        if (apiKey.isBlank()) {
            return McpResult(McpSettings.Domain.FLIGHTS, false, null, null, "SerpApi key not configured. Set it in Settings > MCP API Keys.")
        }

        val origin = entities["origin"]?.take(3)?.uppercase() ?: "JFK"
        val destination = entities["destination"]?.take(3)?.uppercase() ?: "LAX"
        val dateOutStr = getFormattedDateOffset(7, "yyyy-MM-dd")
        val dateRetStr = getFormattedDateOffset(9, "yyyy-MM-dd")

        // SerpApi Google Flights search
        val searchUrl = "https://serpapi.com/search.json?engine=google_flights" +
            "&departure_id=$origin&arrival_id=$destination" +
            "&outbound_date=$dateOutStr&return_date=$dateRetStr&currency=USD&api_key=${enc(apiKey)}"
            
        val flightResponse = httpGet(searchUrl)
        val flightJson = JsonParser.parseString(flightResponse).asJsonObject

        val result = JsonObject().apply {
            addProperty("origin", origin)
            addProperty("destination", destination)
            addProperty("query", queryText)
            add("flights", flightJson.get("best_flights") ?: JsonArray())
        }

        return McpResult(McpSettings.Domain.FLIGHTS, true, result, flightResponse, null)
    }

    // ── Restaurants: Google Places API (New) ────────────────────────────

    private fun fetchRestaurants(entities: Map<String, String>, apiKey: String, queryText: String): McpResult {
        if (apiKey.isBlank()) {
            return McpResult(McpSettings.Domain.RESTAURANTS, false, null, null, "Google Places API key not configured. Set it in Settings > MCP API Keys.")
        }

        val location = entities["location"] ?: extractLocationFallback(queryText) ?: "New York"
        val url = "https://places.googleapis.com/v1/places:searchText"
        val body = JsonObject().apply {
            addProperty("textQuery", "restaurants in $location")
            addProperty("maxResultCount", 8)
        }.toString()
        val headers = mapOf(
            "X-Goog-Api-Key" to apiKey,
            "X-Goog-FieldMask" to "places.displayName,places.formattedAddress,places.priceLevel," +
                "places.rating,places.userRatingCount,places.types,places.photos,places.reviews," +
                "places.editorialSummary,places.regularOpeningHours," +
                "places.websiteUri,places.googleMapsUri"
        )
        val response = httpPostWithHeaders(url, body, "application/json", headers)
        val json = JsonParser.parseString(response).asJsonObject

        // Enrich each place with a ready-to-use photoUri constructed from the first photo slot
        val places = json.getAsJsonArray("places") ?: JsonArray()
        places.forEach { el ->
            val place = el.asJsonObject
            val photoName = place.getAsJsonArray("photos")
                ?.firstOrNull()?.asJsonObject?.get("name")?.asString
            if (photoName != null) {
                place.addProperty("photoUri",
                    "https://places.googleapis.com/v1/$photoName/media?maxWidthPx=800&key=$apiKey")
            }
        }

        val result = JsonObject().apply {
            addProperty("location", location)
            addProperty("query", queryText)
            add("results", places)
        }

        return McpResult(McpSettings.Domain.RESTAURANTS, true, result, response, null)
    }

    // ── Hotels: SerpApi (Google Hotels) ─────────────────────────────────

    private fun fetchHotels(entities: Map<String, String>, apiKey: String, queryText: String): McpResult {
        if (apiKey.isBlank()) {
            return McpResult(McpSettings.Domain.HOTELS, false, null, null, "SerpApi key not configured. Set it in Settings > MCP API Keys.")
        }

        val location = entities["location"] ?: extractLocationFallback(queryText) ?: "New York"
        // Use the original query so qualifiers like "5 star", "budget", "near beach" are preserved.
        // Fall back to "hotels in <location>" only when the query doesn't already contain them.
        val searchQuery = if (queryText.lowercase(java.util.Locale.US).contains("hotel"))
            queryText else "hotels in $location"
        val url = "https://serpapi.com/search.json?" +
            "engine=google_hotels&q=${enc(searchQuery)}" +
            "&check_in_date=${getDateOffset(7)}&check_out_date=${getDateOffset(9)}" +
            "&adults=2&currency=INR&api_key=${enc(apiKey)}"
        val response = httpGet(url)
        val json = JsonParser.parseString(response).asJsonObject

        // Surface any SerpApi-level error
        val apiError = json.get("error")?.takeIf { !it.isJsonNull }?.asString
        if (apiError != null) {
            Log.w(LOG_TAG, "SerpApi hotels error for '$searchQuery': $apiError")
            return McpResult(McpSettings.Domain.HOTELS, false, null, response, apiError)
        }

        val properties = json.getAsJsonArray("properties") ?: JsonArray()
        Log.d(LOG_TAG, "Hotels: query='$searchQuery' location='$location' properties=${properties.size()}")

        val result = JsonObject().apply {
            addProperty("location", location)
            addProperty("query", queryText)
            add("properties", properties)
        }

        return McpResult(McpSettings.Domain.HOTELS, true, result, response, null)
    }

    // ── Places: Google Places API (New) ─────────────────────────────────

    private fun fetchPlaces(entities: Map<String, String>, apiKey: String, queryText: String): McpResult {
        if (apiKey.isBlank()) {
            return McpResult(McpSettings.Domain.PLACES, false, null, null, "Google Places API key not configured. Set it in Settings > MCP API Keys.")
        }

        val location = entities["location"] ?: extractLocationFallback(queryText) ?: "New York"
        val url = "https://places.googleapis.com/v1/places:searchText"
        val body = JsonObject().apply {
            addProperty("textQuery", "top attractions in $location")
            addProperty("maxResultCount", 8)
        }.toString()
        val headers = mapOf(
            "X-Goog-Api-Key" to apiKey,
            "X-Goog-FieldMask" to "places.displayName,places.formattedAddress,places.priceLevel," +
                "places.rating,places.userRatingCount,places.types,places.photos,places.reviews," +
                "places.editorialSummary,places.regularOpeningHours," +
                "places.websiteUri,places.googleMapsUri"
        )
        val response = httpPostWithHeaders(url, body, "application/json", headers)
        val json = JsonParser.parseString(response).asJsonObject

        // Enrich each place with a ready-to-use photoUri from the first photo slot
        val places = json.getAsJsonArray("places") ?: JsonArray()
        places.forEach { el ->
            val place = el.asJsonObject
            val photoName = place.getAsJsonArray("photos")
                ?.firstOrNull()?.asJsonObject?.get("name")?.asString
            if (photoName != null) {
                place.addProperty("photoUri",
                    "https://places.googleapis.com/v1/$photoName/media?maxWidthPx=800&key=$apiKey")
            }
        }

        val result = JsonObject().apply {
            addProperty("location", location)
            addProperty("query", queryText)
            add("results", places)
        }

        return McpResult(McpSettings.Domain.PLACES, true, result, response, null)
    }

    // ── News: NewsData.io ───────────────────────────────────────────────

    private fun fetchNews(entities: Map<String, String>, apiKey: String, queryText: String): McpResult {
        if (apiKey.isBlank()) {
            return McpResult(McpSettings.Domain.NEWS, false, null, null, "NewsData.io key not configured. Set it in Settings > MCP API Keys.")
        }

        val topic = entities["topic"]
        val url = if (!topic.isNullOrBlank()) {
            "https://newsdata.io/api/1/news?" +
                "apikey=${enc(apiKey)}&q=${enc(topic)}&country=in&language=en"
        } else {
            "https://newsdata.io/api/1/news?" +
                "apikey=${enc(apiKey)}&country=in&language=en"
        }
        val response = httpGet(url)
        val json = JsonParser.parseString(response).asJsonObject

        val result = JsonObject().apply {
            addProperty("topic", topic ?: "top headlines")
            addProperty("query", queryText)
            add("results", json.get("results") ?: JsonArray())
            addProperty("totalResults", json.get("totalResults")?.asInt ?: 0)
        }

        return McpResult(McpSettings.Domain.NEWS, true, result, response, null)
    }

    // ── HTTP helpers ────────────────────────────────────────────────────

    private fun httpGet(url: String): String {
        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = CONNECT_TIMEOUT
            readTimeout = READ_TIMEOUT
            setRequestProperty("Accept", "application/json")
        }
        return readResponse(connection)
    }

    private fun httpGetWithAuth(url: String, authKey: String, authValue: String? = null): String {
        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = CONNECT_TIMEOUT
            readTimeout = READ_TIMEOUT
            setRequestProperty("Accept", "application/json")
            if (authValue == null) {
                setRequestProperty("Authorization", authKey)
            } else {
                setRequestProperty(authKey, authValue)
            }
        }
        return readResponse(connection)
    }

    private fun httpPost(url: String, body: String, contentType: String): String {
        return httpPostWithHeaders(url, body, contentType, emptyMap())
    }

    private fun httpPostWithHeaders(url: String, body: String, contentType: String, headers: Map<String, String>): String {
        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = CONNECT_TIMEOUT
            readTimeout = READ_TIMEOUT
            doOutput = true
            setRequestProperty("Content-Type", contentType)
            setRequestProperty("Accept", "application/json")
            headers.forEach { (k, v) -> setRequestProperty(k, v) }
        }
        connection.outputStream.use { out ->
            out.write(body.toByteArray(StandardCharsets.UTF_8))
        }
        return readResponse(connection)
    }

    private fun readResponse(connection: HttpURLConnection): String {
        return try {
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            stream?.bufferedReader(StandardCharsets.UTF_8)?.use { it.readText() }
                ?: throw IOException("HTTP $code: no response body")
        } finally {
            connection.disconnect()
        }
    }

    private fun enc(value: String): String =
        URLEncoder.encode(value, StandardCharsets.UTF_8.name())

    private fun getDateOffset(daysFromNow: Int): String {
        return getFormattedDateOffset(daysFromNow, "yyyy-MM-dd")
    }
    
    private fun getFormattedDateOffset(daysFromNow: Int, format: String): String {
        val cal = java.util.Calendar.getInstance()
        cal.add(java.util.Calendar.DAY_OF_YEAR, daysFromNow)
        val sdf = java.text.SimpleDateFormat(format, java.util.Locale.US)
        return sdf.format(cal.time)
    }

    private fun extractLocationFallback(query: String): String? {
        val match = Regex("(?:in|for|at|near|around)\\s+([A-Z][a-zA-Z\\s,]+)", RegexOption.IGNORE_CASE)
            .find(query)
        return match?.groupValues?.get(1)?.trim()?.takeIf { it.length in 2..60 }
    }
}
