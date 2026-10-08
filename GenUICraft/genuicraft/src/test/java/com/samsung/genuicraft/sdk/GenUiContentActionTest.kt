package com.samsung.genuicraft.sdk

import com.samsung.genuicraft.sdk.internal.renderer.FlatRenderEvent
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatSpec
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GenUiContentActionTest {
    @Test
    fun previewProjectionRemovesEventAndWatchBindingsWithoutMutatingTheFinalSpec() {
        val binding = mapOf("action" to "setState", "params" to mapOf("statePath" to "/count", "value" to 1.0))
        val element = FlatElement(
            type = "Button", props = mapOf("text" to "Continue"), children = emptyList(),
            on = mapOf("press" to binding), watch = mapOf("/count" to binding),
        )
        val spec = FlatSpec("button", mapOf("count" to 0.0), mapOf("button" to element))

        val preview = spec.withoutPreviewBindings()

        assertEquals(null, preview.elements.getValue("button").on)
        assertEquals(null, preview.elements.getValue("button").watch)
        assertEquals(spec.state, preview.state)
        assertEquals(element.props, preview.elements.getValue("button").props)
        assertEquals(mapOf("press" to binding), spec.elements.getValue("button").on)
        assertEquals(mapOf("/count" to binding), spec.elements.getValue("button").watch)
    }

    @Test
    fun openUrlIsDeliveredToTheHostWithoutLaunchingAnIntent() {
        val actions = mutableListOf<GenUiAction>()

        CallbackRendererHost(actions::add).openUrl("https://www.samsung.com/support/")

        assertEquals(
            listOf(
                GenUiAction(
                    name = "openUrl",
                    parameters = mapOf("url" to "https://www.samsung.com/support/"),
                )
            ),
            actions,
        )
    }

    @Test
    fun emitEventUsesItsEventNameAndPreservesStructuredParameters() {
        val actions = mutableListOf<GenUiAction>()

        dispatchExternalEvent(
            FlatRenderEvent(
                kind = FlatRenderEvent.Kind.ACTION,
                action = "emitEvent",
                params = linkedMapOf(
                    "name" to "continue",
                    "context" to linkedMapOf("selectedId" to "A1042"),
                    "wantResponse" to true,
                ),
                success = true,
            ),
            actions::add,
        )

        assertEquals("continue", actions.single().name)
        assertEquals("{\"selectedId\":\"A1042\"}", actions.single().parameters["context"])
        assertEquals("true", actions.single().parameters["wantResponse"])
    }

    @Test
    fun stateLocalActionsNeverEscapeThroughTheHostCallback() {
        val actions = mutableListOf<GenUiAction>()

        listOf("setState", "pushState", "removeState", "validateForm").forEach { name ->
            dispatchExternalEvent(
                FlatRenderEvent(
                    kind = FlatRenderEvent.Kind.ACTION,
                    action = name,
                    success = true,
                ),
                actions::add,
            )
        }

        assertTrue(actions.isEmpty())
    }
}
