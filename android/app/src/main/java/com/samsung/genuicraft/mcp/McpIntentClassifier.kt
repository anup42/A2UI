package com.samsung.genuicraft.mcp

import java.util.Locale

/**
 * Classifies a user query into one of the supported MCP domains.
 * Uses keyword-based heuristics for fast, offline classification.
 * Returns null if the query doesn't match any MCP-supported domain.
 */
object McpIntentClassifier {

    data class ClassifiedIntent(
        val domain: McpSettings.Domain,
        val confidence: Float,
        val extractedEntities: Map<String, String>
    )

    private val WEATHER_KEYWORDS = setOf(
        "weather", "temperature", "forecast", "rain", "snow", "humidity",
        "sunny", "cloudy", "wind", "climate", "celsius", "fahrenheit",
        "hot", "cold", "storm", "precipitation", "uv index", "dew point"
    )

    private val FLIGHT_KEYWORDS = setOf(
        "flight", "flights", "airline", "airfare", "fly", "flying",
        "plane", "airport", "departure", "arrival", "layover", "stopover",
        "nonstop", "non-stop", "booking", "ticket", "round trip", "one way",
        "economy", "business class", "first class"
    )

    private val RESTAURANT_KEYWORDS = setOf(
        "restaurant", "restaurants", "food", "dining", "eat", "eating",
        "cuisine", "cafe", "bistro", "diner", "takeout", "delivery",
        "reservation", "menu", "brunch", "lunch", "dinner", "breakfast",
        "pizza", "sushi", "burger", "thai", "indian", "chinese", "mexican",
        "italian", "vegan", "vegetarian"
    )

    private val HOTEL_KEYWORDS = setOf(
        "hotel", "hotels", "accommodation", "stay", "lodge", "lodging",
        "motel", "resort", "inn", "hostel", "airbnb", "booking",
        "check-in", "check-out", "room", "suite", "bed and breakfast"
    )

    private val PLACES_KEYWORDS = setOf(
        "place", "places", "attraction", "attractions", "landmark",
        "sightseeing", "tourist", "tourism", "visit", "explore",
        "itinerary", "travel", "trip", "vacation", "holiday",
        "things to do", "must see", "top 10", "best places",
        "museum", "park", "temple", "beach", "mountain"
    )

    private val NEWS_KEYWORDS = setOf(
        "news", "headline", "headlines", "latest", "breaking",
        "current events", "today's news", "top stories",
        "article", "articles", "report", "reporting",
        "politics", "sports news", "tech news", "business news",
        "world news", "local news"
    )

    private val DOMAIN_KEYWORDS = mapOf(
        McpSettings.Domain.WEATHER to WEATHER_KEYWORDS,
        McpSettings.Domain.FLIGHTS to FLIGHT_KEYWORDS,
        McpSettings.Domain.RESTAURANTS to RESTAURANT_KEYWORDS,
        McpSettings.Domain.HOTELS to HOTEL_KEYWORDS,
        McpSettings.Domain.PLACES to PLACES_KEYWORDS,
        McpSettings.Domain.NEWS to NEWS_KEYWORDS
    )

    /**
     * Classifies the query into an MCP domain.
     * Returns the best-matching domain with confidence, or null if no match.
     */
    fun classify(query: String): ClassifiedIntent? {
        val lower = query.lowercase(Locale.US)
        val words = lower.split("\\s+".toRegex())

        val scores = mutableMapOf<McpSettings.Domain, Float>()

        for ((domain, keywords) in DOMAIN_KEYWORDS) {
            var score = 0f
            for (keyword in keywords) {
                if (keyword.contains(' ')) {
                    // Multi-word keyword: check substring
                    if (lower.contains(keyword)) {
                        score += 2f
                    }
                } else {
                    if (words.contains(keyword)) {
                        score += 1f
                    }
                }
            }
            if (score > 0f) {
                scores[domain] = score
            }
        }

        if (scores.isEmpty()) return null

        val bestEntry = scores.maxByOrNull { it.value } ?: return null
        val maxPossible = DOMAIN_KEYWORDS[bestEntry.key]?.size?.toFloat() ?: 1f
        val confidence = (bestEntry.value / maxPossible).coerceIn(0f, 1f)

        val entities = extractEntities(lower, bestEntry.key)

        return ClassifiedIntent(
            domain = bestEntry.key,
            confidence = confidence,
            extractedEntities = entities
        )
    }

    private fun extractEntities(query: String, domain: McpSettings.Domain): Map<String, String> {
        val entities = mutableMapOf<String, String>()

        when (domain) {
            McpSettings.Domain.WEATHER -> {
                // Try to extract city name: "weather in <city>"
                val cityMatch = Regex("(?:weather|forecast|temperature|climate)\\s+(?:in|for|at|of)\\s+(.+?)(?:\\s+(?:today|tomorrow|this week|next week|now))?$", RegexOption.IGNORE_CASE)
                    .find(query)
                if (cityMatch != null) {
                    entities["location"] = cityMatch.groupValues[1].trim()
                }
                val dateMatch = Regex("""\b(?:on|for)\s+([a-z0-9 ,/-]+)$""", RegexOption.IGNORE_CASE).find(query)
                if (dateMatch != null) {
                    entities["date"] = dateMatch.groupValues[1].trim()
                }
            }
            McpSettings.Domain.FLIGHTS -> {
                // "flights from <origin> to <destination>"
                val routeMatch = Regex("(?:flight|flights|fly|flying)\\s+(?:from\\s+)?(.+?)\\s+to\\s+(.+?)(?:\\s+(?:on|in|for|around)\\s+.*)?$", RegexOption.IGNORE_CASE)
                    .find(query)
                if (routeMatch != null) {
                    entities["origin"] = routeMatch.groupValues[1].trim()
                    entities["destination"] = routeMatch.groupValues[2].trim()
                }
                Regex("""\bon\s+([a-z0-9 ,/-]+)""", RegexOption.IGNORE_CASE)
                    .find(query)
                    ?.groupValues
                    ?.getOrNull(1)
                    ?.trim()
                    ?.takeIf { it.isNotBlank() }
                    ?.let { entities["departure_date"] = it }
                Regex("""\breturn(?:ing)?\s+(?:on\s+)?([a-z0-9 ,/-]+)""", RegexOption.IGNORE_CASE)
                    .find(query)
                    ?.groupValues
                    ?.getOrNull(1)
                    ?.trim()
                    ?.takeIf { it.isNotBlank() }
                    ?.let { entities["return_date"] = it }
            }
            McpSettings.Domain.RESTAURANTS -> {
                // "restaurants in <location>"
                val locMatch = Regex("(?:restaurant|restaurants|food|dining|eat|cuisine)\\s+(?:in|near|around|at)\\s+(.+?)(?:\\s+(?:for|with|that)\\s+.*)?$", RegexOption.IGNORE_CASE)
                    .find(query)
                if (locMatch != null) {
                    entities["location"] = locMatch.groupValues[1].trim()
                }
            }
            McpSettings.Domain.HOTELS -> {
                val locMatch = Regex("(?:hotel|hotels|stay|accommodation)\\s+(?:in|near|around|at)\\s+(.+?)(?:\\s+(?:for|from|on|check)\\s+.*)?$", RegexOption.IGNORE_CASE)
                    .find(query)
                if (locMatch != null) {
                    entities["location"] = locMatch.groupValues[1].trim()
                }
                Regex("""\bcheck[- ]?in\s+(?:on\s+)?([a-z0-9 ,/-]+)""", RegexOption.IGNORE_CASE)
                    .find(query)
                    ?.groupValues
                    ?.getOrNull(1)
                    ?.trim()
                    ?.takeIf { it.isNotBlank() }
                    ?.let { entities["check_in"] = it }
                Regex("""\bcheck[- ]?out\s+(?:on\s+)?([a-z0-9 ,/-]+)""", RegexOption.IGNORE_CASE)
                    .find(query)
                    ?.groupValues
                    ?.getOrNull(1)
                    ?.trim()
                    ?.takeIf { it.isNotBlank() }
                    ?.let { entities["check_out"] = it }
            }
            McpSettings.Domain.PLACES -> {
                val locMatch = Regex("(?:places|attractions|things to do|visit|explore|itinerary|travel|trip)\\s+(?:in|to|for|around|near)\\s+(.+?)(?:\\s+(?:for|with|this|next)\\s+.*)?$", RegexOption.IGNORE_CASE)
                    .find(query)
                if (locMatch != null) {
                    entities["location"] = locMatch.groupValues[1].trim()
                }
            }
            McpSettings.Domain.NEWS -> {
                val topicMatch = Regex("(?:news|headlines|latest|breaking)\\s+(?:about|on|for|regarding)\\s+(.+)$", RegexOption.IGNORE_CASE)
                    .find(query)
                if (topicMatch != null) {
                    entities["topic"] = topicMatch.groupValues[1].trim()
                }
                val locationMatch = Regex("""(?:news|headlines|latest|breaking).*\b(?:in|from)\s+([a-z][a-z\s]+)$""", RegexOption.IGNORE_CASE)
                    .find(query)
                if (locationMatch != null) {
                    entities["location"] = locationMatch.groupValues[1].trim()
                }
            }
        }

        return entities
    }
}
