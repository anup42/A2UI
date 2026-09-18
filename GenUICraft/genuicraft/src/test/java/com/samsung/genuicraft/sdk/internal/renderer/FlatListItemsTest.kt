package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.parse.parseFlatListItems
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.hasAuthoredListMarker
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FlatListItemsTest {
    @Test fun authoredNumberingInsideEmphasisDoesNotReceiveAnotherBullet() {
        listOf("1. Song", "**1. Song**", "**2.** Song", "_3. Song_", "(4) Song", "- Item", "* Item", "• Item")
            .forEach { assertTrue(it, hasAuthoredListMarker(it)) }
        listOf("**Train:** service", "31°C", "**2026 forecast**", "Plain item")
            .forEach { assertFalse(it, hasAuthoredListMarker(it)) }
    }

    @Test fun plainInstructionsArePreservedInOrder() {
        val items = parseFlatListItems(listOf("First instruction", "Second instruction"))
        assertEquals(listOf("First instruction", "Second instruction"), items.flatMap { it.text })
    }

    @Test fun structuredItemsKeepDescriptionAndLink() {
        val items = parseFlatListItems(listOf(mapOf("label" to "Manual", "description" to "Check thickness", "url" to "[SOURCE_URL_1]")))
        assertEquals(listOf("Manual", "Check thickness"), items.single().text)
        assertEquals(listOf("[SOURCE_URL_1]"), items.single().links)
    }

    @Test fun componentCallsAreNotMistakenForVisibleText() {
        assertTrue(parseFlatListItems(listOf(mapOf("call" to "Text", "args" to listOf("Hidden")))).isEmpty())
        assertTrue(parseFlatListItems(emptyList<Any>()).isEmpty())
    }
}
