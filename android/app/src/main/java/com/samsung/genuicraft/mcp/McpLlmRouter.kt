package com.samsung.genuicraft.mcp

import android.content.res.AssetManager
import android.util.Log
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.inference.InferenceBackend
import java.net.URLDecoder
import java.nio.charset.StandardCharsets
import java.util.Locale
import kotlin.math.min

/**
 * Uses the Stage 2 LLM in two steps:
 * 1) Route query to MCP domain (or none)
 * 2) If domain selected, extract domain-required entities with a dedicated prompt
 */
object McpLlmRouter {

    private const val LOG_TAG = "McpLlmRouter"
    private const val MCP_ROUTER_PROMPT_ASSET = "pipeline_prompts/response_gen_mcp.md"
    private const val MCP_ENTITY_PROMPT_ASSET = "pipeline_prompts/response_gen_mcp_entities.md"

    data class RouterResult(
        /** Non-null when the LLM identified a live-data domain to call. */
        val domain: McpSettings.Domain?,
        /** Extracted entities (location, origin, destination, topic). */
        val entities: Map<String, String>,
        /**
         * 2-3 sentence intro from the LLM to show before MCP data.
         * Non-null only when domain != null.
         */
        val introText: String?,
        /**
         * Complete formatted response produced by LLM when domain == null (no MCP needed).
         * Null when a live domain was selected.
         */
        val fullResponse: String?,
        val error: String?,
        val streamDurationMs: Long?
    )

    private data class ParsedRouterEnvelope(
        val domain: McpSettings.Domain?,
        val entities: Map<String, String>,
        val introText: String?,
        val fullResponse: String?
    )

    private data class EntityExtractionResult(
        val entities: Map<String, String>,
        val missingRequired: List<String>,
        val error: String?,
        val streamDurationMs: Long?
    )

    private val DOMAIN_REQUIRED_ENTITIES: Map<McpSettings.Domain, List<String>> = mapOf(
        McpSettings.Domain.WEATHER to listOf("location"),
        McpSettings.Domain.FLIGHTS to listOf("origin", "destination"),
        McpSettings.Domain.RESTAURANTS to listOf("location"),
        McpSettings.Domain.HOTELS to listOf("location"),
        McpSettings.Domain.PLACES to listOf("location"),
        McpSettings.Domain.NEWS to emptyList()
    )

    private val DOMAIN_ALLOWED_ENTITIES: Map<McpSettings.Domain, List<String>> = mapOf(
        McpSettings.Domain.WEATHER to listOf(
            "location",
            "date",
            "start_date",
            "end_date",
            "days",
            "temperature_unit",
            "wind_speed_unit",
            "precipitation_unit",
            "language"
        ),
        McpSettings.Domain.FLIGHTS to listOf(
            "origin",
            "destination",
            "date",
            "departure_date",
            "return_date",
            "type",
            "travel_class",
            "adults",
            "children",
            "currency",
            "country",
            "language"
        ),
        McpSettings.Domain.RESTAURANTS to listOf(
            "location",
            "country",
            "language"
        ),
        McpSettings.Domain.HOTELS to listOf(
            "location",
            "date",
            "check_in",
            "check_out",
            "adults",
            "children",
            "currency",
            "country",
            "language"
        ),
        McpSettings.Domain.PLACES to listOf(
            "location",
            "country",
            "language"
        ),
        McpSettings.Domain.NEWS to listOf(
            "topic",
            "location",
            "country",
            "language",
            "date",
            "start_date",
            "end_date"
        )
    )

    private val ALL_ENTITY_KEYS = setOf(
        "location",
        "origin",
        "destination",
        "topic",
        "date",
        "start_date",
        "end_date",
        "days",
        "departure_date",
        "return_date",
        "check_in",
        "check_out",
        "type",
        "travel_class",
        "adults",
        "children",
        "currency",
        "country",
        "language",
        "temperature_unit",
        "wind_speed_unit",
        "precipitation_unit"
    )

    private val COUNTRY_NAME_TO_CODE = mapOf(
        "india" to "in",
        "indian" to "in",
        "united states" to "us",
        "united states of america" to "us",
        "usa" to "us",
        "us" to "us",
        "united kingdom" to "gb",
        "uk" to "gb",
        "england" to "gb",
        "canada" to "ca",
        "australia" to "au",
        "singapore" to "sg",
        "uae" to "ae",
        "united arab emirates" to "ae",
        "germany" to "de",
        "france" to "fr",
        "italy" to "it",
        "spain" to "es",
        "japan" to "jp"
    )

    /**
     * Calls the Stage 2 backend with the MCP routing prompt and parses the result.
     * Runs on the calling thread — must be called from an IO context.
     */
    fun route(
        query: String,
        backend: InferenceBackend,
        assets: AssetManager,
        maxOutputTokens: Int
    ): RouterResult {
        val promptTemplate = loadPromptAsset(assets, MCP_ROUTER_PROMPT_ASSET) ?: return RouterResult(
            domain = null,
            entities = emptyMap(),
            introText = null,
            fullResponse = null,
            error = "Could not load MCP routing prompt.",
            streamDurationMs = null
        )

        val prompt = promptTemplate.replace("{query_text}", query)

        val routeResponse = backend.generate(
            InferenceBackend.GenerateRequest(
                prompt = prompt,
                systemPrompt = null,
                temperature = 0.1, // Low temperature for deterministic routing
                maxOutputTokens = maxOutputTokens,
                jsonMode = true,
                enableGoogleSearch = false,
                structuredOutput = false
            )
        )

        if (routeResponse.error != null) {
            Log.w(LOG_TAG, "MCP routing LLM call failed: ${routeResponse.error}")
            return RouterResult(null, emptyMap(), null, null, routeResponse.error, routeResponse.streamDurationMs)
        }

        val parsedRoute = parseRouterEnvelope(routeResponse.text.trim())
            ?: return RouterResult(
                domain = null,
                entities = emptyMap(),
                introText = null,
                fullResponse = null,
                error = "Failed to parse routing response.",
                streamDurationMs = routeResponse.streamDurationMs
            )

        var totalStreamDurationMs = routeResponse.streamDurationMs
        var mergedEntities = parsedRoute.entities
        val heuristicIntent = McpIntentClassifier.classify(query)
        val domain = parsedRoute.domain
            ?: heuristicIntent
                ?.takeIf { it.confidence >= 0.02f }
                ?.domain

        if (domain != null) {
            val required = requiredEntityKeys(domain)
            val extracted = extractEntitiesForDomain(
                query = query,
                domain = domain,
                requiredEntities = required,
                backend = backend,
                assets = assets,
                maxOutputTokens = maxOutputTokens
            )
            totalStreamDurationMs = mergeDurations(totalStreamDurationMs, extracted.streamDurationMs)

            if (!extracted.error.isNullOrBlank()) {
                Log.w(LOG_TAG, "MCP entity extraction call failed; using fallback entities. ${extracted.error}")
            }

            val heuristicEntities = heuristicIntent
                ?.takeIf { it.domain == domain }
                ?.extractedEntities
                .orEmpty()

            val queryEntities = extractQueryEntitiesByDomain(query, domain)
            mergedEntities = mergeEntityMaps(
                extracted.entities,
                parsedRoute.entities,
                heuristicEntities,
                queryEntities
            )
            mergedEntities = restrictToAllowedEntities(mergedEntities, domain)
            mergedEntities = enforceQueryEntityOverrides(
                merged = mergedEntities,
                queryDerived = queryEntities,
                domain = domain
            )
            mergedEntities = normalizeEntitiesForDomain(
                entities = mergedEntities,
                domain = domain
            )

            val missingRequired = required.filter { mergedEntities[it].isNullOrBlank() }.distinct()
            if (missingRequired.isNotEmpty()) {
                Log.w(
                    LOG_TAG,
                    "MCP domain=${domain.key} missing required entities=$missingRequired finalEntities=$mergedEntities query='$query'"
                )
            } else {
                Log.i(LOG_TAG, "MCP domain=${domain.key} extracted entities=$mergedEntities")
            }
        }

        return RouterResult(
            domain = domain,
            entities = mergedEntities,
            introText = parsedRoute.introText,
            fullResponse = if (domain == null) parsedRoute.fullResponse else null,
            error = null,
            streamDurationMs = totalStreamDurationMs
        )
    }

    private fun parseRouterEnvelope(rawJson: String): ParsedRouterEnvelope? {
        return try {
            val cleanJson = stripMarkdownFences(rawJson)

            val obj = JsonParser.parseString(cleanJson).asJsonObject

            val domainStr = jsonString(obj.get("mcp_domain"))?.trim()?.lowercase(Locale.US) ?: "none"
            val domain = McpSettings.Domain.fromKey(domainStr)

            val entities = parseEntityMap(obj.getAsJsonObject("entities"), ALL_ENTITY_KEYS)
            val intro = readOptionalText(obj.get("intro"))
            val fullResponse = readOptionalText(obj.get("full_response"))

            Log.i(LOG_TAG, "MCP router: domain=$domainStr entities=$entities hasIntro=${intro != null} hasFull=${fullResponse != null}")

            ParsedRouterEnvelope(
                domain = domain,
                entities = entities,
                introText = intro,
                fullResponse = fullResponse
            )
        } catch (e: Exception) {
            Log.w(LOG_TAG, "Failed to parse MCP router JSON: ${e.message}\nRaw: ${rawJson.take(500)}")
            null
        }
    }

    private fun extractEntitiesForDomain(
        query: String,
        domain: McpSettings.Domain,
        requiredEntities: List<String>,
        backend: InferenceBackend,
        assets: AssetManager,
        maxOutputTokens: Int
    ): EntityExtractionResult {
        val promptTemplate = loadPromptAsset(assets, MCP_ENTITY_PROMPT_ASSET)
            ?: return EntityExtractionResult(
                entities = emptyMap(),
                missingRequired = requiredEntities,
                error = "Could not load MCP entity extraction prompt.",
                streamDurationMs = null
            )

        val allowedEntities = allowedEntityKeys(domain)
        val requiredCsv = requiredEntities.joinToString(", ").ifBlank { "none" }
        val allowedCsv = allowedEntities.joinToString(", ").ifBlank { "none" }
        val prompt = promptTemplate
            .replace("{query_text}", query)
            .replace("{mcp_domain}", domain.key)
            .replace("{required_entities}", requiredCsv)
            .replace("{allowed_entities}", allowedCsv)

        val response = backend.generate(
            InferenceBackend.GenerateRequest(
                prompt = prompt,
                systemPrompt = null,
                temperature = 0.0,
                maxOutputTokens = min(maxOutputTokens, 768),
                jsonMode = true,
                enableGoogleSearch = false,
                structuredOutput = false
            )
        )

        if (response.error != null) {
            return EntityExtractionResult(
                entities = emptyMap(),
                missingRequired = requiredEntities,
                error = response.error,
                streamDurationMs = response.streamDurationMs
            )
        }

        return try {
            val cleanJson = stripMarkdownFences(response.text.trim())
            val root = JsonParser.parseString(cleanJson).asJsonObject

            val fromEntitiesBlock = parseEntityMap(root.getAsJsonObject("entities"), allowedEntities.toSet())
            val fromRootBlock = if (fromEntitiesBlock.isEmpty()) {
                parseEntityMap(root, allowedEntities.toSet())
            } else {
                emptyMap()
            }
            val parsedEntities = if (fromEntitiesBlock.isNotEmpty()) fromEntitiesBlock else fromRootBlock

            val missingFromModel = parseMissingRequired(root)
            val computedMissing = requiredEntities.filter { parsedEntities[it].isNullOrBlank() }
            val missing = (missingFromModel + computedMissing).distinct()

            EntityExtractionResult(
                entities = parsedEntities,
                missingRequired = missing,
                error = null,
                streamDurationMs = response.streamDurationMs
            )
        } catch (e: Exception) {
            EntityExtractionResult(
                entities = emptyMap(),
                missingRequired = requiredEntities,
                error = "Failed to parse entity extraction JSON: ${e.message}",
                streamDurationMs = response.streamDurationMs
            )
        }
    }

    private fun parseMissingRequired(root: JsonObject): List<String> {
        val result = mutableListOf<String>()
        val arr = root.getAsJsonArray("missing_required") ?: root.getAsJsonArray("missingRequired")
        arr?.forEach { element ->
            val key = jsonString(element)?.trim()?.lowercase(Locale.US).orEmpty()
            if (key.isNotBlank()) {
                result += key
            }
        }
        return result.distinct()
    }

    private fun loadPromptAsset(assets: AssetManager, path: String): String? {
        return try {
            assets.open(path).bufferedReader().use { it.readText() }
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Failed to load MCP prompt asset: $path", e)
            null
        }
    }

    private fun mergeDurations(a: Long?, b: Long?): Long? {
        return when {
            a == null && b == null -> null
            a == null -> b
            b == null -> a
            else -> a + b
        }
    }

    private fun requiredEntityKeys(domain: McpSettings.Domain): List<String> {
        return DOMAIN_REQUIRED_ENTITIES[domain].orEmpty()
    }

    private fun allowedEntityKeys(domain: McpSettings.Domain): List<String> {
        return DOMAIN_ALLOWED_ENTITIES[domain].orEmpty()
    }

    private fun restrictToAllowedEntities(
        entities: Map<String, String>,
        domain: McpSettings.Domain
    ): Map<String, String> {
        val allowed = allowedEntityKeys(domain).toSet()
        if (allowed.isEmpty()) {
            return entities
        }
        return entities
            .filterKeys { it in allowed }
            .mapValues { (_, value) -> value.trim() }
            .filterValues { it.isNotBlank() }
    }

    private fun mergeEntityMaps(vararg maps: Map<String, String>): Map<String, String> {
        val merged = linkedMapOf<String, String>()
        maps.forEach { map ->
            map.forEach { (key, value) ->
                if (merged[key].isNullOrBlank() && value.isNotBlank()) {
                    merged[key] = value
                }
            }
        }
        return merged
    }

    private fun parseEntityMap(
        source: JsonObject?,
        allowedKeys: Set<String>
    ): Map<String, String> {
        if (source == null) {
            return emptyMap()
        }
        val entities = linkedMapOf<String, String>()
        source.entrySet().forEach { (key, value) ->
            if (allowedKeys.isNotEmpty() && key !in allowedKeys) {
                return@forEach
            }
            val sanitized = sanitizeEntityValue(jsonString(value))
            if (!sanitized.isNullOrBlank()) {
                entities[key] = sanitized
            }
        }
        return entities
    }

    private fun readOptionalText(element: JsonElement?): String? {
        val value = jsonString(element)?.trim().orEmpty()
        if (value.isBlank()) {
            return null
        }
        val lower = value.lowercase(Locale.US)
        if (lower == "null" || lower == "none" || lower == "n/a") {
            return null
        }
        return value
    }

    private fun jsonString(element: JsonElement?): String? {
        return element
            ?.takeIf { !it.isJsonNull && it.isJsonPrimitive }
            ?.asString
    }

    private fun sanitizeEntityValue(raw: String?): String? {
        val decoded = decodeQueryToken(raw)
        val trimmed = decoded.trim().trim('"', '\'')
        if (trimmed.isBlank()) {
            return null
        }
        val lower = trimmed.lowercase(Locale.US)
        if (lower in setOf("null", "none", "n/a", "na", "unknown", "unspecified", "not specified", "-", "--")) {
            return null
        }
        return trimmed
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

    private fun normalizeEntitiesForDomain(
        entities: Map<String, String>,
        domain: McpSettings.Domain
    ): Map<String, String> {
        val normalized = linkedMapOf<String, String>()
        entities.forEach { (key, valueRaw) ->
            val value = sanitizeEntityValue(valueRaw) ?: return@forEach
            val converted = when (key) {
                "origin", "destination" -> normalizeAirportLikeValue(value)
                "currency" -> normalizeCurrencyCode(value) ?: value
                "country" -> normalizeCountryCode(value) ?: value
                "language" -> normalizeLanguageCode(value) ?: value
                "adults", "children", "days" -> extractDigits(value) ?: value
                "type" -> normalizeTripType(value) ?: value
                "travel_class" -> normalizeTravelClass(value) ?: value
                else -> value
            }
            val sanitizedConverted = sanitizeEntityValue(converted) ?: return@forEach
            normalized[key] = sanitizedConverted
        }

        if (domain == McpSettings.Domain.FLIGHTS) {
            val origin = normalized["origin"]
            val destination = normalized["destination"]
            if (!origin.isNullOrBlank() && !destination.isNullOrBlank() &&
                origin.equals(destination, ignoreCase = true)
            ) {
                normalized.remove("destination")
            }
        }
        return normalized
    }

    private fun enforceQueryEntityOverrides(
        merged: Map<String, String>,
        queryDerived: Map<String, String>,
        domain: McpSettings.Domain
    ): Map<String, String> {
        val allowed = allowedEntityKeys(domain).toSet()
        val result = merged.toMutableMap()
        queryDerived.forEach { (key, rawValue) ->
            if (key !in allowed) return@forEach
            val value = sanitizeEntityValue(rawValue) ?: return@forEach
            result[key] = value
        }
        return result
    }

    private fun extractQueryEntitiesByDomain(
        query: String,
        domain: McpSettings.Domain
    ): Map<String, String> {
        val lower = decodeQueryToken(query).lowercase(Locale.US)
        val entities = linkedMapOf<String, String>()
        when (domain) {
            McpSettings.Domain.FLIGHTS -> {
                val route = Regex("""\bfrom\s+(.+?)\s+to\s+(.+?)(?:\s+\b(on|for|in|at)\b|$)""", RegexOption.IGNORE_CASE)
                    .find(lower)
                if (route != null) {
                    val origin = route.groupValues.getOrNull(1).orEmpty().trim()
                    val destination = route.groupValues.getOrNull(2).orEmpty().trim()
                    if (origin.isNotBlank()) entities["origin"] = origin
                    if (destination.isNotBlank()) entities["destination"] = destination
                }
                Regex("""\bon\s+([a-z0-9 ,/\-]+)""", RegexOption.IGNORE_CASE)
                    .find(lower)
                    ?.groupValues
                    ?.getOrNull(1)
                    ?.trim()
                    ?.takeIf { it.isNotBlank() }
                    ?.let { entities["departure_date"] = it }
                Regex("""\breturn(?:ing)?\s+(?:on\s+)?([a-z0-9 ,/\-]+)""", RegexOption.IGNORE_CASE)
                    .find(lower)
                    ?.groupValues
                    ?.getOrNull(1)
                    ?.trim()
                    ?.takeIf { it.isNotBlank() }
                    ?.let { entities["return_date"] = it }
                if (lower.contains("one way") || lower.contains("one-way")) {
                    entities["type"] = "one_way"
                } else if (lower.contains("round trip") || lower.contains("round-trip") || lower.contains("return")) {
                    entities["type"] = "round_trip"
                }
                if (lower.contains("premium economy")) {
                    entities["travel_class"] = "premium_economy"
                } else if (lower.contains("business")) {
                    entities["travel_class"] = "business"
                } else if (lower.contains("first class") || lower.contains("first")) {
                    entities["travel_class"] = "first"
                } else if (lower.contains("economy")) {
                    entities["travel_class"] = "economy"
                }
                Regex("""\b(\d{1,2})\s+adults?\b""", RegexOption.IGNORE_CASE)
                    .find(lower)
                    ?.groupValues
                    ?.getOrNull(1)
                    ?.let { entities["adults"] = it }
                Regex("""\b(\d{1,2})\s+children\b""", RegexOption.IGNORE_CASE)
                    .find(lower)
                    ?.groupValues
                    ?.getOrNull(1)
                    ?.let { entities["children"] = it }
            }

            McpSettings.Domain.HOTELS -> {
                Regex("""\b(?:hotels?|stay|accommodation)\s+(?:in|at|near)\s+(.+?)(?:\s+\b(on|for|check)\b|$)""", RegexOption.IGNORE_CASE)
                    .find(lower)
                    ?.groupValues
                    ?.getOrNull(1)
                    ?.trim()
                    ?.takeIf { it.isNotBlank() }
                    ?.let { entities["location"] = it }
                Regex("""\bcheck[- ]?in\s+(?:on\s+)?([a-z0-9 ,/\-]+)""", RegexOption.IGNORE_CASE)
                    .find(lower)
                    ?.groupValues
                    ?.getOrNull(1)
                    ?.trim()
                    ?.takeIf { it.isNotBlank() }
                    ?.let { entities["check_in"] = it }
                Regex("""\bcheck[- ]?out\s+(?:on\s+)?([a-z0-9 ,/\-]+)""", RegexOption.IGNORE_CASE)
                    .find(lower)
                    ?.groupValues
                    ?.getOrNull(1)
                    ?.trim()
                    ?.takeIf { it.isNotBlank() }
                    ?.let { entities["check_out"] = it }
                Regex("""\b(\d{1,2})\s+adults?\b""", RegexOption.IGNORE_CASE)
                    .find(lower)
                    ?.groupValues
                    ?.getOrNull(1)
                    ?.let { entities["adults"] = it }
                Regex("""\b(\d{1,2})\s+children\b""", RegexOption.IGNORE_CASE)
                    .find(lower)
                    ?.groupValues
                    ?.getOrNull(1)
                    ?.let { entities["children"] = it }
            }

            McpSettings.Domain.NEWS -> {
                Regex("""\bnews\s+(?:about|on|for)\s+(.+)""", RegexOption.IGNORE_CASE)
                    .find(lower)
                    ?.groupValues
                    ?.getOrNull(1)
                    ?.trim()
                    ?.takeIf { it.isNotBlank() }
                    ?.let { entities["topic"] = it }
                Regex("""\b(?:in|from)\s+([a-z][a-z\s]+)$""", RegexOption.IGNORE_CASE)
                    .find(lower)
                    ?.groupValues
                    ?.getOrNull(1)
                    ?.trim()
                    ?.takeIf { it.isNotBlank() }
                    ?.let { entities["location"] = it }
            }

            else -> Unit
        }

        Regex("""\b(?:inr|usd|eur|gbp|aed|sgd|jpy)\b""", RegexOption.IGNORE_CASE)
            .find(lower)
            ?.value
            ?.uppercase(Locale.US)
            ?.let { entities["currency"] = it }
        Regex("""\b(?:english|hindi|french|german|japanese)\b""", RegexOption.IGNORE_CASE)
            .find(lower)
            ?.value
            ?.let { entities["language"] = it }
        Regex("""\b(?:india|usa|united states|uk|united kingdom|canada|australia|japan|singapore|uae)\b""", RegexOption.IGNORE_CASE)
            .find(lower)
            ?.value
            ?.let { entities["country"] = it }

        return entities
    }

    private fun normalizeAirportLikeValue(raw: String): String {
        var cleaned = decodeQueryToken(raw)
            .trim()
            .replace(Regex("""\b(airport|intl|international)\b""", RegexOption.IGNORE_CASE), "")
            .replace(Regex("""\s+"""), " ")
            .trim(',', '.', ';', ':', '-', ' ')
        if (cleaned.length == 3 && cleaned.all { it.isLetter() }) {
            return cleaned.uppercase(Locale.US)
        }
        Regex("""\b([a-z]{3})\b""", RegexOption.IGNORE_CASE)
            .find(cleaned)
            ?.groupValues
            ?.getOrNull(1)
            ?.takeIf { cleaned.length <= 8 }
            ?.let { return it.uppercase(Locale.US) }
        return cleaned
    }

    private fun normalizeCurrencyCode(raw: String?): String? {
        val token = raw
            ?.trim()
            ?.uppercase(Locale.US)
            ?.replace(Regex("""[^A-Z]"""), "")
            .orEmpty()
        return token.takeIf { it.length == 3 }
    }

    private fun normalizeCountryCode(raw: String?): String? {
        val plain = raw
            ?.trim()
            ?.lowercase(Locale.US)
            ?.replace(Regex("""[^a-z\s]"""), " ")
            ?.replace(Regex("""\s+"""), " ")
            ?.trim()
            .orEmpty()
        if (plain.length == 2 && plain.all { it.isLetter() }) {
            return plain
        }
        return COUNTRY_NAME_TO_CODE[plain]
    }

    private fun normalizeLanguageCode(raw: String?): String? {
        val value = raw?.trim()?.lowercase(Locale.US).orEmpty()
        if (value.isBlank()) return null
        val token = value.substringBefore('-').substringBefore('_')
        return when {
            token.length == 2 -> token
            token == "english" -> "en"
            token == "hindi" -> "hi"
            token == "french" -> "fr"
            token == "german" -> "de"
            token == "japanese" -> "ja"
            else -> null
        }
    }

    private fun extractDigits(raw: String?): String? {
        return Regex("""\d{1,3}""").find(raw.orEmpty())?.value
    }

    private fun normalizeTripType(raw: String?): String? {
        val value = raw?.trim()?.lowercase(Locale.US).orEmpty()
        if (value.isBlank()) return null
        return when {
            value in setOf("1", "round_trip", "roundtrip", "round trip", "return") -> "round_trip"
            value in setOf("2", "one_way", "one-way", "one way") -> "one_way"
            value in setOf("3", "multi_city", "multi-city", "multi city") -> "multi_city"
            else -> null
        }
    }

    private fun normalizeTravelClass(raw: String?): String? {
        val value = raw?.trim()?.lowercase(Locale.US).orEmpty()
        if (value.isBlank()) return null
        return when {
            value in setOf("1", "economy") -> "economy"
            value in setOf("2", "premium_economy", "premium economy") -> "premium_economy"
            value in setOf("3", "business", "business_class", "business class") -> "business"
            value in setOf("4", "first", "first_class", "first class") -> "first"
            else -> null
        }
    }

    private fun stripMarkdownFences(raw: String): String {
        return raw
            .removePrefix("```json")
            .removePrefix("```")
            .removeSuffix("```")
            .trim()
    }
}
