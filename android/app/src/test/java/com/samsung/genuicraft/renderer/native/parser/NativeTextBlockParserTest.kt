package com.samsung.genuicraft.renderer.native.parser

import com.samsung.genuicraft.renderer.native.TextBlock
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeTextBlockParserTest {

    @Test
    fun parseTextBlocks_parsesBulletPrefixedQuickActionsAsActions() {
        val rawText = """
            Quick Actions
            • Action: [Button: Search on MakeMyTrip] https://www.makemytrip.com/flights/
            • Action: [Button: Search on IndiGo] https://www.goindigo.in/
        """.trimIndent()

        val blocks = parseBlocks(rawText)

        val actions = blocks.filterIsInstance<TextBlock.Actions>()
        assertEquals(1, actions.size)
        assertEquals(2, actions.first().actions.size)
        assertEquals("Search on MakeMyTrip", actions.first().actions[0].label)
        assertEquals("https://www.makemytrip.com/flights/", actions.first().actions[0].url)
        assertEquals("Search on IndiGo", actions.first().actions[1].label)
        assertEquals("https://www.goindigo.in/", actions.first().actions[1].url)
    }

    @Test
    fun parseTextBlocks_keepsNormalBulletsAsBulletBlock() {
        val rawText = """
            Highlights
            - Direct flights available
            - Lowest fare is around INR 6,500
        """.trimIndent()

        val blocks = parseBlocks(rawText)

        val bullets = blocks.filterIsInstance<TextBlock.Bullets>()
        assertEquals(1, bullets.size)
        assertTrue(bullets.first().items.any { it.contains("Direct flights", ignoreCase = true) })
        assertTrue(bullets.first().items.any { it.contains("Lowest fare", ignoreCase = true) })
    }

    private fun parseBlocks(rawText: String): List<TextBlock> {
        return NativeTextBlockParser.parseTextBlocks(
            rawText = rawText,
            collectSourceLinks = { _, _, _ -> null },
            collectBookingOptions = { _, _ -> null },
            collectTableRows = { _, _ -> null },
            collectMediaEntries = { _, _ -> null },
            collectNumberedSteps = { _, _ -> null },
            isBulletListLine = NativeStructureParsing::isBulletListLine,
            extractBulletListItem = NativeStructureParsing::extractBulletListItem,
            isPlaceholderListEntry = { false },
            parseButtonLine = { line ->
                NativeStructureParsing.parseButtonLine(
                    line,
                    NativeStructureParsing::sanitizeUrlToken
                )
            },
            looksLikeStandaloneLinkLine = { false },
            parseSourceLinksFromLine = { emptyList() },
            isSourcePrefixedLinkLine = { false },
            looksLikeSectionHeading = { line ->
                line.equals("Quick Actions", ignoreCase = true) ||
                    line.equals("Highlights", ignoreCase = true)
            },
            isStructuredBoundary = { false }
        )
    }
}
