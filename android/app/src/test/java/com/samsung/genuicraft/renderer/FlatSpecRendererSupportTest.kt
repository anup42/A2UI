package com.samsung.genuicraft.renderer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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

    @Test
    fun resolve_supportsNestedItemIndexAndBindItem() {
        val state = mapOf<String, Any?>(
            "items" to listOf(
                mapOf("meta" to mapOf("title" to "Alpha"), "name" to "First")
            )
        )
        val scope = RepeatScope(
            item = mapOf("meta" to mapOf("title" to "Alpha"), "name" to "First"),
            index = 0,
            basePath = "/items/0"
        )

        val nestedItemExpr = mapOf("\$item" to "meta/title")
        val indexExpr = mapOf("\$index" to true)
        val bindItemExpr = mapOf("\$bindItem" to "name")

        assertEquals("Alpha", FlatExprResolver.resolve(nestedItemExpr, state, scope, emptyMap()))
        assertEquals(0, FlatExprResolver.resolve(indexExpr, state, scope, emptyMap()))
        assertEquals("First", FlatExprResolver.resolve(bindItemExpr, state, scope, emptyMap()))
    }

    @Test
    fun resolve_recursesInsideMapsAndLists() {
        val state = mapOf<String, Any?>("user" to mapOf("name" to "Ravi"))
        val scope = RepeatScope(index = 2)
        val value = mapOf(
            "title" to mapOf("\$state" to "/user/name"),
            "meta" to listOf(mapOf("\$index" to true), "ok")
        )

        val resolved = FlatExprResolver.resolve(value, state, scope, emptyMap()) as Map<*, *>
        assertEquals("Ravi", resolved["title"])
        val meta = resolved["meta"] as List<*>
        assertEquals(2, meta[0])
        assertEquals("ok", meta[1])
    }

    @Test
    fun resolve_supportsComputedFunctions() {
        val state = mapOf<String, Any?>("user" to mapOf("name" to "Asha"))
        val computed = mapOf<String, FlatComputedFunction>(
            "greet" to { args -> "Hi ${args["name"]}" }
        )
        val value = mapOf(
            "\$computed" to "greet",
            "args" to mapOf("name" to mapOf("\$state" to "/user/name"))
        )

        val resolved = FlatExprResolver.resolve(value, state, null, computed)
        assertEquals("Hi Asha", resolved)
    }

    @Test
    fun actionRuntime_executesActionArraysInOrder() {
        val state = mutableMapOf<String, Any?>()
        FlatSpecParser.setAtPath(state, "/items", emptyList<Any>())

        val actions = listOf(
            mapOf(
                "action" to "setState",
                "params" to mapOf("statePath" to "/step", "value" to 1)
            ),
            mapOf(
                "action" to "pushState",
                "params" to mapOf("statePath" to "/items", "value" to mapOf("name" to "A"))
            ),
            mapOf(
                "action" to "removeState",
                "params" to mapOf("statePath" to "/items", "index" to 0)
            )
        )

        FlatActionRuntime.execute(
            actionCandidate = actions,
            stateStore = state,
            repeatScope = null,
            computedFunctions = emptyMap(),
            onOpenUrl = {}
        )

        assertEquals(1, FlatSpecParser.getAtPath(state, "/step"))
        val items = FlatSpecParser.getAtPath(state, "/items") as List<*>
        assertTrue(items.isEmpty())
    }

    @Test
    fun actionRuntime_resolvesBindItemStatePathForSetState() {
        val state = mutableMapOf<String, Any?>(
            "items" to listOf(mapOf("name" to "Old"))
        )
        val scope = RepeatScope(
            item = mapOf("name" to "Old"),
            index = 0,
            basePath = "/items/0"
        )
        val action = mapOf(
            "action" to "setState",
            "params" to mapOf(
                "statePath" to mapOf("\$item" to "name"),
                "value" to "New"
            )
        )

        FlatActionRuntime.execute(
            actionCandidate = action,
            stateStore = state,
            repeatScope = scope,
            computedFunctions = emptyMap(),
            onOpenUrl = {}
        )

        assertEquals("New", FlatSpecParser.getAtPath(state, "/items/0/name"))
    }

    @Test
    fun actionRuntime_pushStateResolvesStateValues() {
        val state = mutableMapOf<String, Any?>(
            "template" to mapOf("name" to "FromState"),
            "items" to emptyList<Any>()
        )
        val action = mapOf(
            "action" to "pushState",
            "params" to mapOf(
                "statePath" to "/items",
                "value" to mapOf("\$state" to "/template")
            )
        )

        FlatActionRuntime.execute(
            actionCandidate = action,
            stateStore = state,
            repeatScope = null,
            computedFunctions = emptyMap(),
            onOpenUrl = {}
        )

        val items = FlatSpecParser.getAtPath(state, "/items") as List<*>
        assertEquals(1, items.size)
        assertEquals("FromState", (items[0] as Map<*, *>)["name"])
    }

    @Test
    fun actionRuntime_validateFormWritesResult() {
        val state = mutableMapOf<String, Any?>()
        val action = mapOf(
            "action" to "validateForm",
            "params" to mapOf("statePath" to "/formValidation")
        )

        FlatActionRuntime.execute(
            actionCandidate = action,
            stateStore = state,
            repeatScope = null,
            computedFunctions = emptyMap(),
            onOpenUrl = {}
        )

        val result = FlatSpecParser.getAtPath(state, "/formValidation") as Map<*, *>
        assertEquals(true, result["valid"])
        assertTrue((result["errors"] as Map<*, *>).isEmpty())
    }

    @Test
    fun watchRuntime_triggersOnlyOnStateChange() {
        val elements = mapOf(
            "root" to FlatElement(
                type = "Stack",
                props = mapOf("direction" to "vertical"),
                children = emptyList(),
                watch = mapOf(
                    "/flag" to mapOf(
                        "action" to "setState",
                        "params" to mapOf("statePath" to "/touched", "value" to true)
                    )
                )
            )
        )
        val runtime = FlatWatchRuntime(elements)

        val initial = runtime.collectTriggered(mapOf("flag" to false))
        val unchanged = runtime.collectTriggered(mapOf("flag" to false))
        val changed = runtime.collectTriggered(mapOf("flag" to true))
        val stableAgain = runtime.collectTriggered(mapOf("flag" to true))

        assertTrue(initial.isEmpty())
        assertTrue(unchanged.isEmpty())
        assertEquals(1, changed.size)
        assertFalse(stableAgain.isNotEmpty())
    }
}

