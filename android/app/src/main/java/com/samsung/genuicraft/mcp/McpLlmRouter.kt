package com.samsung.genuicraft.mcp

import android.content.res.AssetManager
import android.util.Log
import com.google.gson.JsonParser
import com.samsung.genuicraft.inference.InferenceBackend

/**
 * Uses the Stage 2 LLM to decide which MCP domain (if any) to call,
 * extract entities from the query, and produce an intro or full response.
 *
 * This replaces keyword-based intent classification with LLM understanding.
 */
object McpLlmRouter {

    private const val LOG_TAG = "McpLlmRouter"
    private const val MCP_PROMPT_ASSET = "pipeline_prompts/response_gen_mcp.md"

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
        val promptTemplate = try {
            assets.open(MCP_PROMPT_ASSET).bufferedReader().readText()
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Failed to load MCP routing prompt", e)
            return RouterResult(null, emptyMap(), null, null, "Could not load MCP routing prompt: ${e.message}", null)
        }

        val prompt = promptTemplate.replace("{query_text}", query)

        val response = backend.generate(
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

        if (response.error != null) {
            Log.w(LOG_TAG, "MCP routing LLM call failed: ${response.error}")
            return RouterResult(null, emptyMap(), null, null, response.error, response.streamDurationMs)
        }

        return parseRouterResponse(response.text.trim(), response.streamDurationMs)
    }

    private fun parseRouterResponse(rawJson: String, streamDurationMs: Long?): RouterResult {
        return try {
            // Strip markdown code fences if present
            val cleanJson = rawJson
                .removePrefix("```json").removePrefix("```")
                .removeSuffix("```").trim()

            val obj = JsonParser.parseString(cleanJson).asJsonObject

            val domainStr = obj.get("mcp_domain")?.asString?.trim()?.lowercase() ?: "none"
            val domain = McpSettings.Domain.fromKey(domainStr)

            val entitiesObj = obj.getAsJsonObject("entities")
            val entities = mutableMapOf<String, String>()
            entitiesObj?.entrySet()?.forEach { (key, value) ->
                val v = if (value.isJsonNull) null else value.asString?.trim()
                if (!v.isNullOrBlank() && v != "null") {
                    entities[key] = v
                }
            }

            val intro = obj.get("intro")?.let {
                if (it.isJsonNull) null else it.asString?.trim()?.takeIf { s -> s.isNotBlank() && s != "null" }
            }

            val fullResponse = obj.get("full_response")?.let {
                if (it.isJsonNull) null else it.asString?.trim()?.takeIf { s -> s.isNotBlank() && s != "null" }
            }

            Log.i(LOG_TAG, "MCP router: domain=$domainStr entities=$entities hasIntro=${intro != null} hasFull=${fullResponse != null}")

            RouterResult(
                domain = domain,
                entities = entities,
                introText = intro,
                fullResponse = fullResponse,
                error = null,
                streamDurationMs = streamDurationMs
            )
        } catch (e: Exception) {
            Log.w(LOG_TAG, "Failed to parse MCP router JSON: ${e.message}\nRaw: ${rawJson.take(500)}")
            RouterResult(null, emptyMap(), null, null, "Failed to parse routing response: ${e.message}", streamDurationMs)
        }
    }
}
