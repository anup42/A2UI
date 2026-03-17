package com.samsung.genuicraft.mcp

import android.content.res.AssetManager
import android.util.Log
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.inference.InferenceBackend
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
        val domain = parsedRoute.domain

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

            val heuristicEntities = McpIntentClassifier.classify(query)
                ?.takeIf { it.domain == domain }
                ?.extractedEntities
                .orEmpty()

            mergedEntities = mergeEntityMaps(
                extracted.entities,
                parsedRoute.entities,
                heuristicEntities
            )
            mergedEntities = restrictToAllowedEntities(mergedEntities, domain)

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
            fullResponse = parsedRoute.fullResponse,
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
        val trimmed = raw?.trim()?.trim('"', '\'').orEmpty()
        if (trimmed.isBlank()) {
            return null
        }
        val lower = trimmed.lowercase(Locale.US)
        if (lower in setOf("null", "none", "n/a", "na", "unknown", "unspecified", "not specified", "-", "--")) {
            return null
        }
        return trimmed
    }

    private fun stripMarkdownFences(raw: String): String {
        return raw
            .removePrefix("```json")
            .removePrefix("```")
            .removeSuffix("```")
            .trim()
    }
}
