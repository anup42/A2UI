package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.runtime.FlatActionRuntime
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FlatSpecStreamingStateTest {
    @Test
    fun generatedUpdatesRefreshDataAndPreserveCompatibleNestedUserEdits() {
        val holder = FlatSpecStateHolder(
            mapOf("form" to mapOf("name" to "", "city" to "Bengaluru"), "rows" to listOf("one")),
        )
        holder.setAtPath("/form/name", "Anup")

        holder.mergeGeneratedState(
            mapOf(
                "form" to mapOf("name" to "Suggested name", "city" to "Mumbai", "country" to "India"),
                "rows" to listOf("one", "two"),
            ),
        )

        assertEquals(mapOf("name" to "Anup", "city" to "Mumbai", "country" to "India"), holder.state["form"])
        assertEquals(listOf("one", "two"), holder.state["rows"])
    }

    @Test
    fun editedRowsKeepTheirEditWhileNewRowsAndUneditedFieldsArrive() {
        val holder = FlatSpecStateHolder(
            mapOf("rows" to listOf(mapOf("id" to "a", "name" to "old", "status" to "waiting"))),
        )
        holder.setAtPath("/rows/0/name", "my edit")

        holder.mergeGeneratedState(
            mapOf("rows" to listOf(
                mapOf("id" to "a", "name" to "generated", "status" to "ready"),
                mapOf("id" to "b", "name" to "new", "status" to "ready"),
            )),
        )

        assertEquals(listOf(
            mapOf("id" to "a", "name" to "my edit", "status" to "ready"),
            mapOf("id" to "b", "name" to "new", "status" to "ready"),
        ), holder.state["rows"])
    }

    @Test
    fun replacedEntitiesDoNotInheritAnotherRowsEdit() {
        val holder = FlatSpecStateHolder(mapOf("rows" to listOf(mapOf("id" to "a", "name" to "first"))))
        holder.setAtPath("/rows/0/name", "private edit")

        holder.mergeGeneratedState(mapOf("rows" to listOf(mapOf("id" to "b", "name" to "second"))))

        assertEquals(listOf(mapOf("id" to "b", "name" to "second")), holder.state["rows"])
    }

    @Test
    fun clearingASelectionSurvivesCompatibleGeneratedUpdates() {
        val holder = FlatSpecStateHolder(mapOf("selection" to "small", "status" to "waiting"))
        holder.setAtPath("/selection", null)

        holder.mergeGeneratedState(mapOf("selection" to "large", "status" to "ready"))

        assertTrue(holder.state.containsKey("selection"))
        assertEquals(null, holder.state["selection"])
        assertEquals("ready", holder.state["status"])
    }

    @Test
    fun changedTypesAndRemovedGeneratedFieldsFollowTheNewDocument() {
        val holder = FlatSpecStateHolder(mapOf("selection" to "old", "removed" to "original"))
        holder.setAtPath("/selection", "my edit")
        holder.setAtPath("/removed", "my removed edit")
        holder.setAtPath("/local", true)

        holder.mergeGeneratedState(mapOf("selection" to false))

        assertEquals(false, holder.state["selection"])
        assertFalse(holder.state.containsKey("removed"))
        assertEquals(true, holder.state["local"])
    }

    @Test
    fun generatedObjectsAreCopiedAndResetClearsTheOldEditBaseline() {
        val nested = mutableMapOf<String, Any?>("name" to "initial")
        val holder = FlatSpecStateHolder(mapOf("form" to nested))
        nested["name"] = "outside mutation"
        assertEquals(mapOf("name" to "initial"), holder.state["form"])
        holder.setAtPath("/form/name", "my edit")
        holder.reset(mapOf("form" to mapOf("name" to "reset")))

        holder.mergeGeneratedState(mapOf("form" to mapOf("name" to "next")))

        assertEquals(mapOf("name" to "next"), holder.state["form"])
    }

    @Test
    fun previewGatePreventsNativeStateAndNavigationSideEffects() {
        val state = mutableMapOf<String, Any?>("count" to 0.0, "rows" to listOf("one"))
        val urls = mutableListOf<String>()
        val actions = listOf(
            mapOf("action" to "setState", "params" to mapOf("statePath" to "/count", "value" to 1.0)),
            mapOf("action" to "pushState", "params" to mapOf("statePath" to "/rows", "value" to "two")),
            mapOf("action" to "openUrl", "params" to mapOf("url" to "https://www.samsung.com/")),
        )

        val result = executeFlatRenderAction(interactionEnabled = false) {
            FlatActionRuntime.executeDetailed(actions, state, null, emptyMap(), urls::add)
        }

        assertEquals(0, result.attempted)
        assertEquals(mapOf("count" to 0.0, "rows" to listOf("one")), state)
        assertTrue(urls.isEmpty())
        val accepted = executeFlatRenderAction(interactionEnabled = true) {
            FlatActionRuntime.executeDetailed(actions, state, null, emptyMap(), urls::add)
        }
        assertEquals(3, accepted.attempted)
        assertEquals(1.0, state["count"])
        assertEquals(listOf("one", "two"), state["rows"])
        assertEquals(listOf("https://www.samsung.com/"), urls)
    }
}
