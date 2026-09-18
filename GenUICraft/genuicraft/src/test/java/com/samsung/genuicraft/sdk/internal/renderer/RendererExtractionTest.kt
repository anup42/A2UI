package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.capability.GeneratedRendererCapabilities
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.FLAT_ELEMENT_RENDERERS
import com.samsung.genuicraft.sdk.internal.renderer.flat.runtime.FlatActionRuntime
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RendererExtractionTest {
    @Test
    fun everyCatalogRuntimeTypeHasANativeRenderer() {
        assertEquals(GeneratedRendererCapabilities.runtimeTypes, FLAT_ELEMENT_RENDERERS.keys)
        assertTrue(FLAT_ELEMENT_RENDERERS.keys.containsAll(setOf("table", "chart", "image", "video", "audioplayer", "emailpreview")))
    }

    @Test
    fun externalNavigationUsesHostCallbackWhileStateMutationStaysInternal() {
        val opened = mutableListOf<String>()
        val state = mutableMapOf<String, Any?>("status" to "old")

        val navigation = FlatActionRuntime.executeDetailed(
            actionCandidate = mapOf(
                "action" to "openUrl",
                "params" to mapOf("url" to "https://www.samsung.com/support/"),
            ),
            stateStore = state,
            repeatScope = null,
            computedFunctions = emptyMap(),
            onOpenUrl = opened::add,
        )
        val mutation = FlatActionRuntime.executeDetailed(
            actionCandidate = mapOf(
                "action" to "setState",
                "params" to mapOf("path" to "/status", "value" to "new"),
            ),
            stateStore = state,
            repeatScope = null,
            computedFunctions = emptyMap(),
            onOpenUrl = opened::add,
        )

        assertEquals(listOf("https://www.samsung.com/support/"), opened)
        assertEquals("new", state["status"])
        assertTrue(navigation.actions.single().success)
        assertTrue(mutation.actions.single().success)
        assertEquals(1, mutation.actions.single().stateMutations.size)
    }
}
