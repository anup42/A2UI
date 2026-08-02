package com.samsung.genuicraft.mcp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class McpUrlShortenerTest {
    @Test
    fun keepsShortIconUrlsReadable() {
        val iconUrl = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/star.svg"

        val result = McpUrlShortener.shorten("Media: Icon=$iconUrl")

        assertEquals("Media: Icon=$iconUrl", result.shortenedText)
        assertTrue(result.urlMap.isEmpty())
    }

    @Test
    fun shortensAndRestoresLongUrls() {
        val longUrl = "https://maps.example.com/place/photo?" + "token=" + "x".repeat(120)

        val result = McpUrlShortener.shorten("Photo=$longUrl")

        assertEquals("Photo={{u1}}", result.shortenedText)
        assertEquals("Photo=$longUrl", McpUrlShortener.restore(result.shortenedText, result.urlMap))
    }
}
