package com.samsung.genuicraft.mcp

import android.content.Context

/**
 * Manages MCP (Model Context Protocol) feature toggle and per-domain API key configuration.
 * When MCP is enabled, the pipeline uses real-time data from MCP servers instead of
 * relying solely on the LLM's training data for supported domains.
 */
object McpSettings {

    private const val PREFS_NAME = "mcp_settings"
    private const val KEY_ENABLED = "mcp_enabled"

    /** Supported MCP domains. Each maps to a specific MCP server / API. */
    enum class Domain(
        val key: String,
        val displayName: String,
        val description: String,
        val apiKeyPrefKey: String
    ) {
        WEATHER(
            key = "weather",
            displayName = "Weather",
            description = "Real-time weather and forecasts via Open-Meteo",
            apiKeyPrefKey = "mcp_api_key_weather"
        ),
        FLIGHTS(
            key = "flights",
            displayName = "Flights",
            description = "Flight search and pricing via SerpApi (Google Flights)",
            apiKeyPrefKey = "mcp_api_key_flights"
        ),
        RESTAURANTS(
            key = "restaurants",
            displayName = "Restaurants",
            description = "Restaurant search and photos via Google Places",
            apiKeyPrefKey = "mcp_api_key_restaurants"
        ),
        HOTELS(
            key = "hotels",
            displayName = "Hotels",
            description = "Hotel search and booking via Booking.com / Serpapi",
            apiKeyPrefKey = "mcp_api_key_hotels"
        ),
        PLACES(
            key = "places",
            displayName = "Places & Travel",
            description = "Places, attractions, and travel info via Google Places",
            apiKeyPrefKey = "mcp_api_key_places"
        ),
        NEWS(
            key = "news",
            displayName = "News",
            description = "Latest news articles via NewsData.io",
            apiKeyPrefKey = "mcp_api_key_news"
        );

        companion object {
            fun fromKey(key: String): Domain? =
                entries.firstOrNull { it.key.equals(key, ignoreCase = true) }
        }
    }

    fun isEnabled(context: Context): Boolean {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getBoolean(KEY_ENABLED, false)
    }

    fun setEnabled(context: Context, enabled: Boolean) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_ENABLED, enabled)
            .apply()
    }

    fun getApiKey(context: Context, domain: Domain): String {
        return when (domain) {
            Domain.RESTAURANTS, Domain.PLACES -> {
                val prefKey = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                    .getString(domain.apiKeyPrefKey, "")
                    .orEmpty()
                    .trim()
                val defaultKey = com.samsung.genuicraft.GeminiApiKeyProvider.googleMapsApiKey(context)
                defaultKey.ifBlank {
                    prefKey.takeIf { it.startsWith("AIza") }.orEmpty()
                }
            }
            Domain.NEWS -> {
                val prefKey = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                    .getString(domain.apiKeyPrefKey, "")
                    .orEmpty()
                    .trim()
                prefKey.ifBlank { com.samsung.genuicraft.BuildConfig.NEWS_API_KEY_DEFAULT }
            }
            Domain.HOTELS, Domain.FLIGHTS -> {
                val prefKey = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                    .getString(domain.apiKeyPrefKey, "")
                    .orEmpty()
                    .trim()
                prefKey.ifBlank { com.samsung.genuicraft.BuildConfig.SERPAPI_KEY_DEFAULT }
            }
            else -> context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .getString(domain.apiKeyPrefKey, "")
                .orEmpty()
                .trim()
        }
    }

    fun setApiKey(context: Context, domain: Domain, apiKey: String) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(domain.apiKeyPrefKey, apiKey.trim())
            .apply()
    }

    /** Returns true if the domain requires an API key and one is configured. */
    fun isDomainReady(context: Context, domain: Domain): Boolean {
        return when (domain) {
            Domain.WEATHER -> true // Open-Meteo is free, no key needed
            else -> getApiKey(context, domain).isNotBlank()
        }
    }
}
