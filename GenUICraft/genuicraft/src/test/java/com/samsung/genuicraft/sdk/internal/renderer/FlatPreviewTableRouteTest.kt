package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.FlatPreviewTableRoute
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.extractDirectTableModel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test

class FlatPreviewTableRouteTest {
    @Test
    fun growingBoundNumericRowsKeepTheInitialGridUntilFinalization() {
        val props = mapOf("columns" to listOf("Group", "Amount", "Score"), "statePath" to "/rows", "domain" to "generic")
        val first = requireNotNull(extractDirectTableModel(props, mapOf("rows" to listOf(listOf("A", "10", "20"))), true))
        val complete = requireNotNull(extractDirectTableModel(props, mapOf("rows" to listOf(
            listOf("A", "10", "20"), listOf("B", "30", "40"),
        )), true))
        assertNotEquals(first.renderMode, complete.renderMode)
        val route = FlatPreviewTableRoute()
        route.model(first, isFinal = false)

        val preview = route.model(complete, isFinal = false)

        assertEquals(first.renderMode, preview.renderMode)
        assertEquals(first.shape, preview.shape)
        assertEquals(complete.rows, preview.rows)
        assertEquals(complete, route.model(complete, isFinal = true))
    }

    @Test
    fun lateDomainAndAdaptiveInferenceDoNotChangeTheProvisionalRoute() {
        val props = mapOf("columns" to listOf("Group", "Amount", "Score"), "rows" to listOf(listOf("A", "10", "20")))
        val first = requireNotNull(extractDirectTableModel(props + ("domain" to "generic"), emptyMap(), true))
        val revised = requireNotNull(extractDirectTableModel(props + ("domain" to "comparison"), emptyMap(), true))
        val route = FlatPreviewTableRoute()
        route.model(first, isFinal = false)
        assertEquals(AdaptiveTablePresentation.TABLE, route.presentation(AdaptiveTablePresentation.TABLE, false))

        assertEquals(first.domain, route.model(revised, false).domain)
        assertEquals(AdaptiveTablePresentation.TABLE, route.presentation(AdaptiveTablePresentation.ENTITY_CARDS, false))
        assertEquals(revised, route.model(revised, true))
        assertEquals(AdaptiveTablePresentation.ENTITY_CARDS, route.presentation(AdaptiveTablePresentation.ENTITY_CARDS, true))
    }
}
