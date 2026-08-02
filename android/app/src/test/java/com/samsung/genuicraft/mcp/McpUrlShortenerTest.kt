package com.samsung.genuicraft.mcp

import com.google.gson.Gson
import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class McpUrlShortenerTest {
    @Test
    fun masksAndRestoresShortUrls() {
        val iconUrl = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/star.svg"

        val result = McpUrlShortener.shorten("Media: Icon=$iconUrl")

        assertEquals("Media: Icon={{u1}}", result.shortenedText)
        assertEquals("Media: Icon=$iconUrl", McpUrlShortener.restore(result.shortenedText, result.urlMap))
    }

    @Test
    fun shortensAndRestoresLongUrls() {
        val longUrl = "https://maps.example.com/place/photo?" + "token=" + "x".repeat(120)

        val result = McpUrlShortener.shorten("Photo=$longUrl")

        assertEquals("Photo={{u1}}", result.shortenedText)
        assertEquals("Photo=$longUrl", McpUrlShortener.restore(result.shortenedText, result.urlMap))
    }

    @Test
    fun masksAndRestoresLocalAssetPaths() {
        val text = """
            Image=../assets/flight-card.png
            Icon=assets\\icons\\plane.svg
            Photo="C:\\GenUI Assets\\arrival photo.webp"
            Cached=/sdcard/Android/data/com.example/files/media/hero.jpg
        """.trimIndent()

        val result = McpUrlShortener.shorten(text)

        assertEquals(4, result.urlMap.size)
        assertTrue(result.shortenedText.contains("Image={{u1}}"))
        assertTrue(result.shortenedText.contains("Icon={{u2}}"))
        assertTrue(result.shortenedText.contains("Photo=\"{{u3}}\""))
        assertTrue(result.shortenedText.contains("Cached={{u4}}"))
        assertEquals(text, McpUrlShortener.restore(result.shortenedText, result.urlMap))
    }

    @Test
    fun masksAllUriSchemesAndReusesPlaceholderForDuplicates() {
        val url = "content://media/external/images/media/42"
        val text = "Image=$url\nThumbnail=$url\nGenerated=genuicraft:weather-sunny\nInline=data:image/png;base64,AAAA"

        val result = McpUrlShortener.shorten(text)

        assertEquals(
            "Image={{u1}}\nThumbnail={{u1}}\nGenerated={{u2}}\nInline={{u3}}",
            result.shortenedText,
        )
        assertEquals(text, McpUrlShortener.restore(result.shortenedText, result.urlMap))
    }

    @Test
    fun masksExtensionlessAssetsAndAndroidResourceReferences() {
        val text = "Model=assets/models/gemma\nIcon=@drawable/ic_plane"

        val result = McpUrlShortener.shorten(text)

        assertEquals("Model={{u1}}\nIcon={{u2}}", result.shortenedText)
        assertEquals(text, McpUrlShortener.restore(result.shortenedText, result.urlMap))
    }

    @Test
    fun generatedPlaceholdersDoNotCollideWithExistingTokens() {
        val text = "Existing={{u1}}\nLink=https://example.org/x"

        val result = McpUrlShortener.shorten(text)

        assertEquals("Existing={{u1}}\nLink={{u2}}", result.shortenedText)
        assertEquals(text, McpUrlShortener.restore(result.shortenedText, result.urlMap))
    }

    @Test
    fun jsonRestorationEscapesPathsAndRemovesUnmappedReferences() {
        val originalPath = "C:\\GenUI Assets\\arrival photo.webp"
        val candidate = JsonParser.parseString(
            """{"image":"{{u1}}","invented":"../assets/fake.png"}"""
        )

        val restored = McpUrlShortener.restoreJsonReferences(
            candidate,
            mapOf("{{u1}}" to originalPath),
        )
        val serialized = Gson().toJson(restored.jsonElement)
        val reparsed = JsonParser.parseString(serialized).asJsonObject

        assertEquals(originalPath, reparsed.get("image").asString)
        assertEquals("", reparsed.get("invented").asString)
        assertEquals(1, restored.removedUnmappedReferenceCount)
    }
}
