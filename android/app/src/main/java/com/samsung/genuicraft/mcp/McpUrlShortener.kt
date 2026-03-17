package com.samsung.genuicraft.mcp

/**
 * Replaces long URLs in MCP-formatted text with short placeholders like {{u1}}, {{u2}}
 * before sending to the Stage 3 LLM, and restores them in the GenUI JSON after.
 * This dramatically reduces token usage for long Google Places photo / Maps URLs.
 */
object McpUrlShortener {

    private val URL_REGEX = Regex("""https?://\S+""")

    data class ShortenResult(
        val shortenedText: String,
        val urlMap: Map<String, String>   // placeholder → real URL
    )

    /**
     * Scans [text] for URLs ≥60 chars and replaces each unique URL with a short
     * placeholder token `{{u1}}`, `{{u2}}`, etc.
     * Returns the shortened text and the mapping for later restoration.
     */
    fun shorten(text: String): ShortenResult {
        val seen = mutableMapOf<String, String>()   // realUrl → placeholder
        var counter = 1
        val shortened = URL_REGEX.replace(text) { match ->
            val url = match.value
            seen.getOrPut(url) { "{{u${counter++}}}" }
        }
        // Invert: placeholder → realUrl
        val urlMap = seen.entries.associate { (url, placeholder) -> placeholder to url }
        return ShortenResult(shortened, urlMap)
    }

    /**
     * Replaces all placeholder tokens in [text] with their original URLs using [urlMap].
     */
    fun restore(text: String, urlMap: Map<String, String>): String {
        if (urlMap.isEmpty()) return text
        var result = text
        for ((placeholder, realUrl) in urlMap) {
            result = result.replace(placeholder, realUrl)
        }
        return result
    }
}
