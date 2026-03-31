package com.samsung.genuicraft.renderer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FlatSpecRendererSupportTest {

    @Test
    fun setAtPath_updatesNestedMapPath() {
        val state = mutableMapOf<String, Any?>(
            "filters" to mapOf(
                "price" to mapOf(
                    "min" to 100
                )
            )
        )

        FlatSpecParser.setAtPath(state, "/filters/price/max", 500)

        assertEquals(100, FlatSpecParser.getAtPath(state, "/filters/price/min"))
        assertEquals(500, FlatSpecParser.getAtPath(state, "/filters/price/max"))
    }

    @Test
    fun setAtPath_updatesNestedListPath() {
        val state = mutableMapOf<String, Any?>()

        FlatSpecParser.setAtPath(state, "/items/0/name", "A")
        FlatSpecParser.setAtPath(state, "/items/1/name", "B")

        assertEquals("A", FlatSpecParser.getAtPath(state, "/items/0/name"))
        assertEquals("B", FlatSpecParser.getAtPath(state, "/items/1/name"))
    }

    @Test
    fun evaluateVisible_andStateExpressions_workForFlatSpec() {
        val state = mapOf<String, Any?>("tab" to "flights")
        val visibleExpr = mapOf(
            "\$state" to "/tab",
            "eq" to "flights"
        )
        val hiddenExpr = mapOf(
            "\$state" to "/tab",
            "eq" to "hotels"
        )

        assertTrue(FlatExprResolver.evaluateVisible(visibleExpr, state, null))
        assertTrue(!FlatExprResolver.evaluateVisible(hiddenExpr, state, null))
    }

    @Test
    fun resolve_itemAndTemplateExpressions() {
        val state = mapOf<String, Any?>(
            "user" to mapOf("name" to "Anup")
        )
        val item = mapOf<String, Any?>("url" to "https://example.com")

        val itemExpr = mapOf("\$item" to "url")
        val templateExpr = mapOf("\$template" to "Hello ${'$'}{/user/name}")

        assertEquals("https://example.com", FlatExprResolver.resolve(itemExpr, state, item))
        assertEquals("Hello Anup", FlatExprResolver.resolve(templateExpr, state, item))
    }
}

