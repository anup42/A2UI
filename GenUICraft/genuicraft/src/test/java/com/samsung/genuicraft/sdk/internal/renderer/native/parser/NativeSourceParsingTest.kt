package com.samsung.genuicraft.sdk.internal.renderer.native.parser

import com.samsung.genuicraft.sdk.internal.renderer.native.ParsedButton
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeSourceParsingTest {

    @Test
    fun isSourcePrefixedLinkLine_detectsSourceColonUrl() {
        assertTrue(NativeSourceParsing.isSourcePrefixedLinkLine("Source: https://example.com/report"))
        assertTrue(NativeSourceParsing.isSourcePrefixedLinkLine("References: [Spec](https://example.org/spec)"))
        assertFalse(NativeSourceParsing.isSourcePrefixedLinkLine("Source:"))
        assertFalse(NativeSourceParsing.isSourcePrefixedLinkLine("Random heading: text"))
    }

    @Test
    fun collectSourceLinks_acceptsSourcePrefixedLineOutsideSourcesSection() {
        val lines = listOf(
            "Source: https://example.com/reference",
            "Next Heading"
        )

        val result = NativeSourceParsing.collectSourceLinks(
            lines = lines,
            startIndex = 0,
            inSourcesSection = false,
            looksLikeSectionHeading = { it == "Next Heading" },
            parseSourceLinksFromLine = { line -> parseSourceLinks(line) }
        )

        assertNotNull(result)
        val (links, nextIndex) = result!!
        assertEquals(1, links.size)
        assertEquals("https://example.com/reference", links.first().url)
        assertEquals(1, nextIndex)
    }

    private fun parseSourceLinks(line: String): List<ParsedButton> {
        return NativeSourceParsing.parseSourceLinksFromLine(
            line = line,
            stripLeadingBulletMarker = NativeStructureParsing::stripLeadingBulletMarker,
            sanitizeDisplayText = { it.trim() },
            sanitizeUrlToken = NativeStructureParsing::sanitizeUrlToken,
            toExternalUrl = { raw ->
                val token = raw.trim()
                when {
                    token.startsWith("https://", ignoreCase = true) -> token
                    token.startsWith("http://", ignoreCase = true) -> token
                    token.startsWith("www.", ignoreCase = true) -> "https://$token"
                    else -> null
                }
            },
            containsUrlLikeToken = { value ->
                Regex("""(?i)(?:https?://|www\.)""").containsMatchIn(value)
            }
        )
    }
}
