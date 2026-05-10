package com.samsung.genuicraft.mcp

import android.util.Log
import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.net.URLDecoder
import java.nio.charset.StandardCharsets
import java.time.DayOfWeek
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.time.format.DateTimeFormatterBuilder
import java.time.temporal.ChronoField
import java.time.temporal.TemporalAdjusters
import java.util.Locale

/**
 * MCP client that fetches real-time data from domain-specific APIs.
 * Each domain maps to a popular, well-documented public API:
 *
 * - Weather: Open-Meteo (free, no key required)
 * - Flights: Tequila Kiwi (API key required)
 * - Restaurants: Geoapify Places (API key required)
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
        val error: String?,
        val requestDebug: String? = null
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

    // â”€â”€ Weather: Open-Meteo (free, no API key) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    private fun fetchWeather(entities: Map<String, String>, queryText: String): McpResult {
        val location = entities["location"] ?: extractLocationFallback(queryText) ?: "New York"
        val language = normalizeLanguageCode(entities["language"], fallback = "en")
        val singleDate = normalizeIsoDate(entities["date"])
        val explicitStart = normalizeIsoDate(entities["start_date"])
        val explicitEnd = normalizeIsoDate(entities["end_date"])
        val (startDate, endDate) = normalizeDateRange(explicitStart, explicitEnd, singleDate)
        val forecastDays = parseIntInRange(entities["days"], min = 1, max = 16)
        val temperatureUnit = normalizeTemperatureUnit(entities["temperature_unit"])
        val windSpeedUnit = normalizeWindSpeedUnit(entities["wind_speed_unit"])
        val precipitationUnit = normalizePrecipitationUnit(entities["precipitation_unit"])
        val requestDebug = buildRequestDebug(
            domain = "weather",
            endpoint = "open-meteo",
            pairs = listOf(
                "location" to location,
                "language" to language,
                "start_date" to startDate,
                "end_date" to endDate,
                "days" to (forecastDays ?: 7).toString(),
                "temperature_unit" to temperatureUnit,
                "wind_speed_unit" to windSpeedUnit,
                "precipitation_unit" to precipitationUnit
            )
        )

        // Step 1: Geocode the location
        val geocodeUrl =
            "https://geocoding-api.open-meteo.com/v1/search?name=${enc(location)}&count=1&language=${enc(language)}&format=json"
        val geocodeResponse = httpGet(geocodeUrl)
        val geocodeJson = JsonParser.parseString(geocodeResponse).asJsonObject
        val results = geocodeJson.getAsJsonArray("results")
        if (results == null || results.size() == 0) {
            return McpResult(
                domain = McpSettings.Domain.WEATHER,
                success = false,
                data = null,
                rawJson = null,
                error = "Location not found: $location",
                requestDebug = requestDebug
            )
        }
        val place = results[0].asJsonObject
        val lat = place.get("latitude").asDouble
        val lon = place.get("longitude").asDouble
        val resolvedName = place.get("name")?.asString ?: location
        val country = place.get("country")?.asString ?: ""

        // Step 2: Fetch weather with optional date range/units supported by Open-Meteo
        val params = mutableListOf(
            "latitude=$lat",
            "longitude=$lon",
            "timezone=auto",
            "current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,uv_index",
            "daily=temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max,sunrise,sunset"
        )
        if (!startDate.isNullOrBlank() && !endDate.isNullOrBlank()) {
            params += "start_date=$startDate"
            params += "end_date=$endDate"
        } else {
            params += "forecast_days=${forecastDays ?: 7}"
        }
        if (!temperatureUnit.isNullOrBlank()) {
            params += "temperature_unit=$temperatureUnit"
        }
        if (!windSpeedUnit.isNullOrBlank()) {
            params += "wind_speed_unit=$windSpeedUnit"
        }
        if (!precipitationUnit.isNullOrBlank()) {
            params += "precipitation_unit=$precipitationUnit"
        }
        val weatherUrl = "https://api.open-meteo.com/v1/forecast?${params.joinToString("&")}"
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
            addProperty("requested_start_date", startDate ?: "")
            addProperty("requested_end_date", endDate ?: "")
            addProperty("requested_forecast_days", forecastDays ?: 7)
            addProperty("requested_temperature_unit", temperatureUnit ?: "")
            addProperty("requested_wind_speed_unit", windSpeedUnit ?: "")
            addProperty("requested_precipitation_unit", precipitationUnit ?: "")
        }

        return McpResult(
            domain = McpSettings.Domain.WEATHER,
            success = true,
            data = result,
            rawJson = weatherResponse,
            error = null,
            requestDebug = requestDebug
        )
    }

    // â”€â”€ Flights: Google Flights (via Serpapi) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    private fun fetchFlights(entities: Map<String, String>, apiKey: String, queryText: String): McpResult {
        if (apiKey.isBlank()) {
            return McpResult(
                domain = McpSettings.Domain.FLIGHTS,
                success = false,
                data = null,
                rawJson = null,
                error = "SerpApi key not configured. Set it in Settings > MCP API Keys.",
                requestDebug = null
            )
        }

        val decodedQuery = decodeQueryToken(queryText)
        val fallbackRoute = extractFlightRouteFallback(decodedQuery)
        val origin = normalizeAirportOrCity(entities["origin"])
            ?: fallbackRoute?.first
        val destination = normalizeAirportOrCity(entities["destination"])
            ?: fallbackRoute?.second
        if (origin.isNullOrBlank() || destination.isNullOrBlank()) {
            return McpResult(
                domain = McpSettings.Domain.FLIGHTS,
                success = false,
                data = null,
                rawJson = null,
                error = "Missing route. Please provide both origin and destination (for example: flights from BLR to LKO).",
                requestDebug = null
            )
        }
        if (origin.equals(destination, ignoreCase = true)) {
            return McpResult(
                domain = McpSettings.Domain.FLIGHTS,
                success = false,
                data = null,
                rawJson = null,
                error = "Origin and destination are the same. Please provide a valid route.",
                requestDebug = null
            )
        }
        val outboundDate = normalizeIsoDate(entities["departure_date"] ?: entities["date"])
            ?: getDateOffset(7)
        val returnDateFromEntity = normalizeIsoDate(entities["return_date"])
        val tripType = normalizeFlightType(entities["type"], decodedQuery, hasReturnDate = !returnDateFromEntity.isNullOrBlank())
        val adults = parseIntInRange(entities["adults"], min = 1, max = 9) ?: 1
        val children = parseIntInRange(entities["children"], min = 0, max = 6)
        val travelClass = normalizeFlightTravelClass(entities["travel_class"])
        val currency = normalizeCurrencyCode(entities["currency"], fallback = "USD")
        val country = normalizeCountryCode(entities["country"]) ?: "us"
        val language = normalizeLanguageCode(entities["language"], fallback = "en")
        val shouldSendReturnDate = tripType != "2"
        val effectiveReturnDate = if (shouldSendReturnDate) {
            returnDateFromEntity ?: normalizeIsoDate(
                parseRelativeOrFormattedDate(outboundDate)?.plusDays(2)?.toString()
            ) ?: getDateOffset(9)
        } else {
            null
        }
        val requestDebug = buildRequestDebug(
            domain = "flights",
            endpoint = "serpapi/google_flights",
            pairs = listOf(
                "origin" to origin,
                "destination" to destination,
                "outbound_date" to outboundDate,
                "return_date" to effectiveReturnDate,
                "trip_type" to tripType,
                "travel_class" to travelClass,
                "adults" to adults.toString(),
                "children" to (children ?: 0).toString(),
                "currency" to currency,
                "country" to country,
                "language" to language
            )
        )

        // SerpApi Google Flights search
        val params = mutableListOf(
            "engine=google_flights",
            "departure_id=${enc(origin)}",
            "arrival_id=${enc(destination)}",
            "outbound_date=$outboundDate",
            "currency=${enc(currency)}",
            "adults=$adults",
            "hl=${enc(language)}",
            "gl=${enc(country)}",
            "api_key=${enc(apiKey)}"
        )
        if (!effectiveReturnDate.isNullOrBlank()) {
            params += "return_date=$effectiveReturnDate"
        }
        if (!tripType.isNullOrBlank()) {
            params += "type=$tripType"
        }
        if (children != null && children > 0) {
            params += "children=$children"
        }
        if (!travelClass.isNullOrBlank()) {
            params += "travel_class=$travelClass"
        }
        val searchUrl = "https://serpapi.com/search.json?${params.joinToString("&")}"

        val flightResponse = httpGet(searchUrl)
        val flightJson = JsonParser.parseString(flightResponse).asJsonObject
        val apiError = flightJson.get("error")?.takeIf { !it.isJsonNull }?.asString
        if (apiError != null) {
            return McpResult(
                domain = McpSettings.Domain.FLIGHTS,
                success = false,
                data = null,
                rawJson = flightResponse,
                error = apiError,
                requestDebug = requestDebug
            )
        }

        val result = JsonObject().apply {
            addProperty("origin", origin)
            addProperty("destination", destination)
            addProperty("outbound_date", outboundDate)
            addProperty("return_date", effectiveReturnDate ?: "")
            addProperty("trip_type", tripType ?: "")
            addProperty("travel_class", travelClass ?: "")
            addProperty("currency", currency)
            addProperty("adults", adults)
            addProperty("children", children ?: 0)
            addProperty("query", queryText)
            add("flights", flightJson.get("best_flights") ?: JsonArray())
            add("other_flights", flightJson.get("other_flights") ?: JsonArray())
        }

        return McpResult(
            domain = McpSettings.Domain.FLIGHTS,
            success = true,
            data = result,
            rawJson = flightResponse,
            error = null,
            requestDebug = requestDebug
        )
    }

    // Restaurants: Geoapify Places API

    private fun fetchRestaurants(entities: Map<String, String>, apiKey: String, queryText: String): McpResult {
        if (apiKey.isBlank()) {
            return McpResult(
                domain = McpSettings.Domain.RESTAURANTS,
                success = false,
                data = null,
                rawJson = null,
                error = "Geoapify API key not configured. Set it in Settings > MCP API Keys.",
                requestDebug = null
            )
        }

        val location = entities["location"] ?: extractLocationFallback(queryText) ?: "New York"
        val language = normalizeLanguageCode(entities["language"], fallback = "en")
        val country = normalizeCountryCode(entities["country"])
        val requestDebug = buildRequestDebug(
            domain = "restaurants",
            endpoint = "geoapify/places",
            pairs = listOf(
                "location" to location,
                "language" to language,
                "country" to country,
                "categories" to "catering.restaurant",
                "max_results" to "8"
            )
        )

        val geocodeParams = mutableListOf(
            "text=${enc(location)}",
            "format=json",
            "limit=1",
            "lang=${enc(language)}",
            "apiKey=${enc(apiKey)}"
        )
        if (!country.isNullOrBlank()) {
            geocodeParams += "filter=countrycode:${enc(country)}"
        }
        val geocodeUrl = "https://api.geoapify.com/v1/geocode/search?${geocodeParams.joinToString("&")}"
        val geocodeResponse = httpGet(geocodeUrl)
        val geocodeJson = JsonParser.parseString(geocodeResponse).asJsonObject
        val geocodeResults = geocodeJson.getAsJsonArray("results")
        if (geocodeResults == null || geocodeResults.size() == 0) {
            return McpResult(
                domain = McpSettings.Domain.RESTAURANTS,
                success = false,
                data = null,
                rawJson = geocodeResponse,
                error = "Location not found: $location",
                requestDebug = requestDebug
            )
        }

        val geoPlace = geocodeResults[0].asJsonObject
        val lat = geoPlace.safeDouble("lat")
        val lon = geoPlace.safeDouble("lon")
        val placeId = geoPlace.safeString("place_id")
        val resolvedName = geoPlace.safeString("city")
            ?: geoPlace.safeString("name")
            ?: geoPlace.safeString("formatted")
            ?: location
        val resolvedCountry = geoPlace.safeString("country") ?: country.orEmpty()
        val resolvedCountryCode = geoPlace.safeString("country_code") ?: country.orEmpty()

        val spatialFilter = when {
            !placeId.isNullOrBlank() -> "place:$placeId"
            lat != null && lon != null -> "circle:$lon,$lat,10000"
            else -> "countrycode:${resolvedCountryCode.ifBlank { "auto" }}"
        }
        val placesParams = mutableListOf(
            "categories=catering.restaurant",
            "filter=${enc(spatialFilter)}",
            "limit=8",
            "lang=${enc(language)}",
            "apiKey=${enc(apiKey)}"
        )
        if (lat != null && lon != null) {
            placesParams += "bias=${enc("proximity:$lon,$lat")}"
        }
        val placesUrl = "https://api.geoapify.com/v2/places?${placesParams.joinToString("&")}"
        val placesResponse = httpGet(placesUrl)
        val placesJson = JsonParser.parseString(placesResponse).asJsonObject

        val restaurants = JsonArray()
        val features = placesJson.getAsJsonArray("features") ?: JsonArray()
        features.forEach { featureElement ->
            val feature = featureElement.asJsonObject
            val properties = feature.getAsJsonObject("properties") ?: JsonObject()
            val coordinates = feature.getAsJsonObject("geometry")?.getAsJsonArray("coordinates")
            val placeLon = properties.safeDouble("lon")
                ?: coordinates?.let { if (it.size() > 0 && !it[0].isJsonNull) it[0].asDouble else null }
            val placeLat = properties.safeDouble("lat")
                ?: coordinates?.let { if (it.size() > 1 && !it[1].isJsonNull) it[1].asDouble else null }
            val name = properties.safeString("name")
                ?: properties.safeString("address_line1")
                ?: "Restaurant"
            val address = properties.safeString("formatted")
                ?: listOfNotNull(
                    properties.safeString("address_line1"),
                    properties.safeString("address_line2")
                ).joinToString(", ").takeIf { it.isNotBlank() }
                ?: ""
            val websiteUri = geoapifyWebsite(properties)
            val mapUri = if (placeLat != null && placeLon != null) {
                "https://www.openstreetmap.org/?mlat=$placeLat&mlon=$placeLon#map=18/$placeLat/$placeLon"
            } else {
                ""
            }
            val categories = properties.getAsJsonArray("categories") ?: JsonArray()
            val restaurant = JsonObject().apply {
                add("displayName", JsonObject().apply { addProperty("text", name) })
                addProperty("formattedAddress", address)
                addProperty("websiteUri", websiteUri)
                addProperty("googleMapsUri", mapUri)
                addProperty("provider", "geoapify")
                properties.safeDouble("distance")?.let { addProperty("distance", it) }
                properties.safeString("place_id")?.let { addProperty("placeId", it) }
                if (placeLat != null) addProperty("latitude", placeLat)
                if (placeLon != null) addProperty("longitude", placeLon)
                add("types", categories)
                val cuisineTags = geoapifyCuisineTags(categories)
                if (cuisineTags.size() > 0) add("cuisineTags", cuisineTags)
            }
            restaurants.add(restaurant)
        }

        val result = JsonObject().apply {
            addProperty("location", resolvedName)
            addProperty("country", resolvedCountry)
            addProperty("country_code", resolvedCountryCode)
            addProperty("language", language)
            addProperty("query", queryText)
            addProperty("provider", "geoapify")
            if (lat != null) addProperty("latitude", lat)
            if (lon != null) addProperty("longitude", lon)
            add("results", restaurants)
        }

        return McpResult(
            domain = McpSettings.Domain.RESTAURANTS,
            success = true,
            data = result,
            rawJson = placesResponse,
            error = null,
            requestDebug = requestDebug
        )
    }

    // â”€â”€ Hotels: SerpApi (Google Hotels) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    private fun fetchHotels(entities: Map<String, String>, apiKey: String, queryText: String): McpResult {
        if (apiKey.isBlank()) {
            return McpResult(
                domain = McpSettings.Domain.HOTELS,
                success = false,
                data = null,
                rawJson = null,
                error = "SerpApi key not configured. Set it in Settings > MCP API Keys.",
                requestDebug = null
            )
        }

        val location = entities["location"] ?: extractLocationFallback(queryText) ?: "New York"
        val checkInDate = normalizeIsoDate(entities["check_in"] ?: entities["date"]) ?: getDateOffset(7)
        val checkoutCandidate = normalizeIsoDate(entities["check_out"])
        val checkOutDate = normalizeCheckoutDate(checkInDate, checkoutCandidate)
        val adults = parseIntInRange(entities["adults"], min = 1, max = 8) ?: 2
        val children = parseIntInRange(entities["children"], min = 0, max = 6)
        val currency = normalizeCurrencyCode(entities["currency"], fallback = "INR")
        val country = normalizeCountryCode(entities["country"])
        val language = normalizeLanguageCode(entities["language"], fallback = "en")

        // Use the original query so qualifiers like "5 star", "budget", "near beach" are preserved.
        // Fall back to "hotels in <location>" only when the query doesn't already contain them.
        val searchQuery = if (queryText.lowercase(Locale.US).contains("hotel"))
            queryText else "hotels in $location"
        val requestDebug = buildRequestDebug(
            domain = "hotels",
            endpoint = "serpapi/google_hotels",
            pairs = listOf(
                "query" to searchQuery,
                "location" to location,
                "check_in_date" to checkInDate,
                "check_out_date" to checkOutDate,
                "adults" to adults.toString(),
                "children" to (children ?: 0).toString(),
                "currency" to currency,
                "country" to country,
                "language" to language
            )
        )
        val params = mutableListOf(
            "engine=google_hotels",
            "q=${enc(searchQuery)}",
            "check_in_date=$checkInDate",
            "check_out_date=$checkOutDate",
            "adults=$adults",
            "currency=${enc(currency)}",
            "hl=${enc(language)}",
            "api_key=${enc(apiKey)}"
        )
        if (children != null && children > 0) {
            params += "children=$children"
        }
        if (!country.isNullOrBlank()) {
            params += "gl=${enc(country)}"
        }
        val url = "https://serpapi.com/search.json?${params.joinToString("&")}"
        val response = httpGet(url)
        val json = JsonParser.parseString(response).asJsonObject

        // Surface any SerpApi-level error
        val apiError = json.get("error")?.takeIf { !it.isJsonNull }?.asString
        if (apiError != null) {
            Log.w(LOG_TAG, "SerpApi hotels error for '$searchQuery': $apiError")
            return McpResult(
                domain = McpSettings.Domain.HOTELS,
                success = false,
                data = null,
                rawJson = response,
                error = apiError,
                requestDebug = requestDebug
            )
        }

        val properties = json.getAsJsonArray("properties") ?: JsonArray()
        Log.d(LOG_TAG, "Hotels: query='$searchQuery' location='$location' properties=${properties.size()}")

        val result = JsonObject().apply {
            addProperty("location", location)
            addProperty("query", queryText)
            addProperty("check_in_date", checkInDate)
            addProperty("check_out_date", checkOutDate)
            addProperty("adults", adults)
            addProperty("children", children ?: 0)
            addProperty("currency", currency)
            addProperty("country", country ?: "")
            addProperty("language", language)
            add("properties", properties)
        }

        return McpResult(
            domain = McpSettings.Domain.HOTELS,
            success = true,
            data = result,
            rawJson = response,
            error = null,
            requestDebug = requestDebug
        )
    }

    // â”€â”€ Places: Google Places API (New) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    private fun fetchPlaces(entities: Map<String, String>, apiKey: String, queryText: String): McpResult {
        if (apiKey.isBlank()) {
            return McpResult(
                domain = McpSettings.Domain.PLACES,
                success = false,
                data = null,
                rawJson = null,
                error = "Google Places API key not configured. Set it in Settings > MCP API Keys.",
                requestDebug = null
            )
        }

        val location = entities["location"] ?: extractLocationFallback(queryText) ?: "New York"
        val language = normalizeLanguageCode(entities["language"], fallback = "en")
        val country = normalizeCountryCode(entities["country"])
        val requestDebug = buildRequestDebug(
            domain = "places",
            endpoint = "google_places/searchText",
            pairs = listOf(
                "location" to location,
                "language" to language,
                "country" to country,
                "max_results" to "8"
            )
        )
        val url = "https://places.googleapis.com/v1/places:searchText"
        val body = JsonObject().apply {
            addProperty("textQuery", "top attractions in $location")
            addProperty("maxResultCount", 8)
            addProperty("languageCode", language)
            if (!country.isNullOrBlank()) {
                addProperty("regionCode", country.uppercase(Locale.US))
            }
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
            addProperty("country", country ?: "")
            addProperty("language", language)
            addProperty("query", queryText)
            add("results", places)
        }

        return McpResult(
            domain = McpSettings.Domain.PLACES,
            success = true,
            data = result,
            rawJson = response,
            error = null,
            requestDebug = requestDebug
        )
    }

    // â”€â”€ News: NewsData.io â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    private fun fetchNews(entities: Map<String, String>, apiKey: String, queryText: String): McpResult {
        if (apiKey.isBlank()) {
            return McpResult(
                domain = McpSettings.Domain.NEWS,
                success = false,
                data = null,
                rawJson = null,
                error = "NewsData.io key not configured. Set it in Settings > MCP API Keys.",
                requestDebug = null
            )
        }

        val topic = entities["topic"]?.trim().orEmpty()
        val location = entities["location"]?.trim().orEmpty()
        val country = normalizeCountryCode(entities["country"])
            ?: normalizeCountryFromLocation(location)
            ?: "in"
        val language = normalizeLanguageCode(entities["language"], fallback = "en")
        val startDateEntity = normalizeIsoDate(entities["start_date"] ?: entities["date"])
        val endDateEntity = normalizeIsoDate(entities["end_date"])
        val (startDate, endDate) = normalizeDateRange(startDateEntity, endDateEntity)
        val requestDebug = buildRequestDebug(
            domain = "news",
            endpoint = "newsdata.io/news",
            pairs = listOf(
                "topic" to topic,
                "location" to location,
                "country" to country,
                "language" to language,
                "from_date" to startDate,
                "to_date" to endDate
            )
        )

        val effectiveQuery = listOf(topic, location)
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .joinToString(" ")

        val params = mutableListOf(
            "apikey=${enc(apiKey)}",
            "country=${enc(country)}",
            "language=${enc(language)}"
        )
        if (effectiveQuery.isNotBlank()) {
            params += "q=${enc(effectiveQuery)}"
        }
        if (!startDate.isNullOrBlank()) {
            params += "from_date=$startDate"
        }
        if (!endDate.isNullOrBlank()) {
            params += "to_date=$endDate"
        }

        val url = "https://newsdata.io/api/1/news?${params.joinToString("&")}"
        val response = httpGet(url)
        val json = JsonParser.parseString(response).asJsonObject

        val result = JsonObject().apply {
            addProperty("topic", if (topic.isNotBlank()) topic else "top headlines")
            addProperty("location", location)
            addProperty("country", country)
            addProperty("language", language)
            addProperty("from_date", startDate ?: "")
            addProperty("to_date", endDate ?: "")
            addProperty("query", queryText)
            add("results", json.get("results") ?: JsonArray())
            addProperty("totalResults", json.get("totalResults")?.asInt ?: 0)
        }

        return McpResult(
            domain = McpSettings.Domain.NEWS,
            success = true,
            data = result,
            rawJson = response,
            error = null,
            requestDebug = requestDebug
        )
    }

    // â”€â”€ HTTP helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

    private fun JsonObject.safeString(key: String): String? {
        return get(key)
            ?.takeIf { !it.isJsonNull }
            ?.asString
            ?.trim()
            ?.takeIf { it.isNotBlank() && !it.equals("null", ignoreCase = true) }
    }

    private fun JsonObject.safeDouble(key: String): Double? {
        return get(key)
            ?.takeIf { !it.isJsonNull }
            ?.let { runCatching { it.asDouble }.getOrNull() }
    }

    private fun geoapifyWebsite(properties: JsonObject): String {
        properties.safeString("website")?.let { return it }
        properties.safeString("contact:website")?.let { return it }
        val raw = properties.get("datasource")
            ?.takeIf { it.isJsonObject }
            ?.asJsonObject
            ?.get("raw")
            ?.takeIf { it.isJsonObject }
            ?.asJsonObject
        return raw?.safeString("website")
            ?: raw?.safeString("contact:website")
            ?: raw?.safeString("url")
            ?: ""
    }

    private fun geoapifyCuisineTags(categories: JsonArray): JsonArray {
        val tags = JsonArray()
        categories
            .mapNotNull { element ->
                element
                    .takeIf { !it.isJsonNull }
                    ?.asString
                    ?.takeIf { it.startsWith("catering.restaurant.") }
                    ?.removePrefix("catering.restaurant.")
                    ?.replace('_', ' ')
                    ?.replace('-', ' ')
                    ?.split(' ')
                    ?.joinToString(" ") { word -> word.replaceFirstChar { c -> c.uppercase() } }
            }
            .distinct()
            .take(4)
            .forEach { tags.add(it) }
        return tags
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
        val formatter = DateTimeFormatter.ofPattern(format, Locale.US)
        return LocalDate.now().plusDays(daysFromNow.toLong()).format(formatter)
    }

    private fun parseIntInRange(raw: String?, min: Int, max: Int): Int? {
        val digits = raw
            ?.trim()
            ?.let { Regex("""\d{1,3}""").find(it)?.value }
            ?: return null
        val value = digits.toIntOrNull() ?: return null
        return value.coerceIn(min, max)
    }

    private fun normalizeCurrencyCode(raw: String?, fallback: String): String {
        val token = raw
            ?.trim()
            ?.uppercase(Locale.US)
            ?.replace(Regex("""[^A-Z]"""), "")
            .orEmpty()
        return if (token.length == 3) token else fallback
    }

    private fun normalizeLanguageCode(raw: String?, fallback: String = "en"): String {
        val token = raw
            ?.trim()
            ?.lowercase(Locale.US)
            ?.replace('_', '-')
            .orEmpty()
        val primary = token.substringBefore('-').replace(Regex("""[^a-z]"""), "")
        return if (primary.length == 2) primary else fallback
    }

    private fun normalizeCountryCode(raw: String?): String? {
        val token = raw?.trim().orEmpty()
        if (token.isBlank()) {
            return null
        }
        val plain = token.lowercase(Locale.US).replace(Regex("""[^a-z]"""), "")
        if (plain.length == 2) {
            return plain
        }
        return COUNTRY_NAME_TO_CODE[plain]
    }

    private fun normalizeCountryFromLocation(location: String?): String? {
        val plain = location
            ?.trim()
            ?.lowercase(Locale.US)
            ?.replace(Regex("""[^a-z]"""), "")
            .orEmpty()
        if (plain.isBlank()) {
            return null
        }
        if (plain.length == 2) {
            return plain
        }
        return COUNTRY_NAME_TO_CODE[plain]
    }

    private fun normalizeTemperatureUnit(raw: String?): String? {
        return when (raw?.trim()?.lowercase(Locale.US)) {
            "c", "celsius", "centigrade", "metric" -> "celsius"
            "f", "fahrenheit", "imperial" -> "fahrenheit"
            else -> null
        }
    }

    private fun normalizeWindSpeedUnit(raw: String?): String? {
        return when (raw?.trim()?.lowercase(Locale.US)) {
            "kmh", "km/h", "kph" -> "kmh"
            "mph" -> "mph"
            "ms", "m/s" -> "ms"
            "kn", "knot", "knots" -> "kn"
            else -> null
        }
    }

    private fun normalizePrecipitationUnit(raw: String?): String? {
        return when (raw?.trim()?.lowercase(Locale.US)) {
            "mm", "millimeter", "millimeters" -> "mm"
            "inch", "inches", "in" -> "inch"
            else -> null
        }
    }

    private fun normalizeAirportOrCity(raw: String?): String? {
        val value = decodeQueryToken(raw).trim()
        if (value.isBlank()) {
            return null
        }
        val cleaned = value
            .replace(Regex("""\b(airport|intl|international)\b""", RegexOption.IGNORE_CASE), "")
            .replace(Regex("""\s+"""), " ")
            .trim(' ', ',', '.', ';', ':', '-', '_')

        val normalizedLower = cleaned.lowercase(Locale.US)
        CITY_NAME_TO_IATA[normalizedLower]?.let { return it }

        val iata = Regex("""\b([A-Za-z]{3})\b""").find(cleaned)?.groupValues?.getOrNull(1)
        if (!iata.isNullOrBlank() && cleaned.length <= 8) {
            return iata.uppercase(Locale.US)
        }
        if (cleaned.length == 3 && cleaned.all { it.isLetter() }) {
            return cleaned.uppercase(Locale.US)
        }
        return cleaned
    }

    private fun normalizeFlightType(raw: String?, queryText: String, hasReturnDate: Boolean): String? {
        val haystack = listOfNotNull(raw, queryText).joinToString(" ").lowercase(Locale.US)
        return when {
            hasReturnDate || haystack.contains("round trip") || haystack.contains("roundtrip") -> "1"
            haystack.contains("one way") || haystack.contains("one-way") -> "2"
            haystack.contains("multi city") || haystack.contains("multi-city") -> "3"
            else -> null
        }
    }

    private fun normalizeFlightTravelClass(raw: String?): String? {
        val value = raw?.trim()?.lowercase(Locale.US).orEmpty()
        if (value.isBlank()) {
            return null
        }
        return when {
            value == "1" || value.contains("economy") -> "1"
            value == "2" || value.contains("premium economy") || value.contains("premium") -> "2"
            value == "3" || value.contains("business") -> "3"
            value == "4" || value.contains("first") -> "4"
            else -> null
        }
    }

    private fun normalizeCheckoutDate(checkInDate: String, checkOutCandidate: String?): String {
        val checkIn = parseRelativeOrFormattedDate(checkInDate) ?: LocalDate.now().plusDays(7)
        val candidate = parseRelativeOrFormattedDate(checkOutCandidate)
        if (candidate != null && candidate.isAfter(checkIn)) {
            return candidate.toString()
        }
        return checkIn.plusDays(1).toString()
    }

    private fun normalizeDateRange(
        start: String?,
        end: String?,
        singleDate: String? = null
    ): Pair<String?, String?> {
        if (!start.isNullOrBlank() || !end.isNullOrBlank()) {
            val parsedStart = parseRelativeOrFormattedDate(start)
            val parsedEnd = parseRelativeOrFormattedDate(end)
            if (parsedStart != null && parsedEnd != null) {
                return if (parsedStart <= parsedEnd) {
                    parsedStart.toString() to parsedEnd.toString()
                } else {
                    parsedEnd.toString() to parsedStart.toString()
                }
            }
            if (parsedStart != null) {
                return parsedStart.toString() to parsedStart.toString()
            }
            if (parsedEnd != null) {
                return parsedEnd.toString() to parsedEnd.toString()
            }
        }
        if (!singleDate.isNullOrBlank()) {
            val parsed = parseRelativeOrFormattedDate(singleDate) ?: return null to null
            return parsed.toString() to parsed.toString()
        }
        return null to null
    }

    private fun normalizeIsoDate(raw: String?): String? {
        return parseRelativeOrFormattedDate(raw)?.toString()
    }

    private fun parseRelativeOrFormattedDate(raw: String?): LocalDate? {
        val value = raw?.trim().orEmpty()
        if (value.isBlank()) {
            return null
        }

        val now = LocalDate.now()
        val lower = value.lowercase(Locale.US)

        if (lower == "today") return now
        if (lower == "tomorrow") return now.plusDays(1)
        if (lower.contains("day after tomorrow")) return now.plusDays(2)
        Regex("""\bin\s+(\d{1,2})\s+days?\b""").find(lower)?.groupValues?.getOrNull(1)?.toIntOrNull()?.let {
            return now.plusDays(it.toLong())
        }

        WEEKDAY_MAP.entries.firstOrNull { lower.contains(it.key) }?.let { (_, dow) ->
            val forceNext = lower.contains("next ")
            val adjuster = if (forceNext) TemporalAdjusters.next(dow) else TemporalAdjusters.nextOrSame(dow)
            return now.with(adjuster)
        }

        val cleaned = lower
            .replace(Regex("""\b(\d{1,2})(st|nd|rd|th)\b"""), "$1")
            .replace(",", " ")
            .replace(Regex("""\s+"""), " ")
            .trim()

        val exactIso = runCatching { LocalDate.parse(cleaned) }.getOrNull()
        if (exactIso != null) {
            return exactIso
        }

        DATE_PATTERNS.forEach { (pattern, hasYear) ->
            val formatter = DateTimeFormatterBuilder()
                .parseCaseInsensitive()
                .appendPattern(pattern)
                .parseDefaulting(ChronoField.YEAR, now.year.toLong())
                .toFormatter(Locale.US)
            val parsed = runCatching { LocalDate.parse(cleaned, formatter) }.getOrNull()
            if (parsed != null) {
                if (!hasYear && parsed.isBefore(now.minusDays(1))) {
                    return parsed.plusYears(1)
                }
                return parsed
            }
        }
        return null
    }

    private fun extractLocationFallback(query: String): String? {
        val match = Regex("(?:in|for|at|near|around)\\s+([A-Z][a-zA-Z\\s,]+)", RegexOption.IGNORE_CASE)
            .find(query)
        return match?.groupValues?.get(1)?.trim()?.takeIf { it.length in 2..60 }
    }

    private fun extractFlightRouteFallback(query: String): Pair<String, String>? {
        val normalizedQuery = decodeQueryToken(query)
        val match = Regex("""\bfrom\s+(.+?)\s+to\s+(.+?)(?:\s+(?:on|for|in|at)\b|$)""", RegexOption.IGNORE_CASE)
            .find(normalizedQuery)
            ?: return null
        val origin = normalizeAirportOrCity(match.groupValues.getOrNull(1)) ?: return null
        val destination = normalizeAirportOrCity(match.groupValues.getOrNull(2)) ?: return null
        return origin to destination
    }

    private fun decodeQueryToken(raw: String?): String {
        val value = raw?.trim().orEmpty()
        if (value.isBlank()) {
            return ""
        }
        val plusDecoded = value.replace('+', ' ')
        return runCatching {
            URLDecoder.decode(plusDecoded, StandardCharsets.UTF_8.name())
        }.getOrDefault(plusDecoded)
    }

    private fun buildRequestDebug(
        domain: String,
        endpoint: String,
        pairs: List<Pair<String, String?>>
    ): String {
        val args = pairs
            .mapNotNull { (key, value) ->
                val normalized = value?.trim().orEmpty()
                if (normalized.isBlank()) null else "$key=$normalized"
            }
            .joinToString(", ")
        return if (args.isBlank()) {
            "domain=$domain, endpoint=$endpoint"
        } else {
            "domain=$domain, endpoint=$endpoint, $args"
        }
    }

    private val DATE_PATTERNS = listOf(
        "d MMM uuuu" to true,
        "d MMMM uuuu" to true,
        "MMM d uuuu" to true,
        "MMMM d uuuu" to true,
        "uuuu/MM/dd" to true,
        "dd/MM/uuuu" to true,
        "MM/dd/uuuu" to true,
        "dd-MM-uuuu" to true,
        "MM-dd-uuuu" to true,
        "d/M/uuuu" to true,
        "d-M-uuuu" to true,
        "d MMM" to false,
        "d MMMM" to false,
        "MMM d" to false,
        "MMMM d" to false,
        "d/M" to false,
        "d-M" to false
    )

    private val WEEKDAY_MAP = linkedMapOf(
        "monday" to DayOfWeek.MONDAY,
        "tuesday" to DayOfWeek.TUESDAY,
        "wednesday" to DayOfWeek.WEDNESDAY,
        "thursday" to DayOfWeek.THURSDAY,
        "friday" to DayOfWeek.FRIDAY,
        "saturday" to DayOfWeek.SATURDAY,
        "sunday" to DayOfWeek.SUNDAY
    )

    private val COUNTRY_NAME_TO_CODE = mapOf(
        "india" to "in",
        "indian" to "in",
        "usa" to "us",
        "unitedstates" to "us",
        "unitedstatesofamerica" to "us",
        "america" to "us",
        "uk" to "gb",
        "unitedkingdom" to "gb",
        "england" to "gb",
        "canada" to "ca",
        "australia" to "au",
        "singapore" to "sg",
        "uae" to "ae",
        "unitedarabemirates" to "ae",
        "germany" to "de",
        "france" to "fr",
        "spain" to "es",
        "italy" to "it",
        "japan" to "jp"
    )

    private val CITY_NAME_TO_IATA = mapOf(
        "bengaluru" to "BLR",
        "bangalore" to "BLR",
        "lucknow" to "LKO",
        "delhi" to "DEL",
        "new delhi" to "DEL",
        "mumbai" to "BOM",
        "hyderabad" to "HYD",
        "chennai" to "MAA",
        "kolkata" to "CCU",
        "pune" to "PNQ",
        "ahmedabad" to "AMD",
        "kochi" to "COK",
        "goa" to "GOI",
        "dubai" to "DXB",
        "singapore" to "SIN",
        "tokyo" to "HND",
        "london" to "LHR",
        "new york" to "JFK",
        "san francisco" to "SFO",
        "los angeles" to "LAX",
        "seattle" to "SEA"
    )
}
