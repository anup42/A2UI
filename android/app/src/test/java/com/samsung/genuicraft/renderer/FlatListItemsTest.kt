package com.samsung.genuicraft.renderer

import com.samsung.genuicraft.renderer.flat.parse.parseFlatListItems
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FlatListItemsTest {
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
