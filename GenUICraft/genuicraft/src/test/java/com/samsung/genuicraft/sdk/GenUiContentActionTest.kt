package com.samsung.genuicraft.sdk

import com.samsung.genuicraft.sdk.internal.renderer.FlatRenderEvent
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GenUiContentActionTest {
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
