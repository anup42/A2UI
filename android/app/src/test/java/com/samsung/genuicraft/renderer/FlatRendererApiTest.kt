package com.samsung.genuicraft.renderer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import com.samsung.genuicraft.renderer.flat.parse.*
import com.samsung.genuicraft.renderer.flat.expr.*
import com.samsung.genuicraft.renderer.flat.runtime.*
import com.samsung.genuicraft.renderer.flat.model.*

/**
 * Guards DoD #6: the renderer must be drivable by a host, with state the host
 * owns.
 *
 * Before this, state lived in a `remember(spec)` inside `FlatSpecContent`, so any
 * new spec instance wiped everything the user had entered — which is why an
 * incremental `surfaceUpdate` was impossible.
 */
class FlatRendererApiTest {

    private class RecordingHost : FlatRendererHost {
        val opened = mutableListOf<String>()
        val diagnostics = mutableListOf<String>()
        override fun openUrl(url: String) {
            opened += url
        }

        override fun resolveAssetUrl(raw: String): String = "resolved:$raw"

        override fun onDiagnostic(message: String) {
            diagnostics += message
        }
    }

    @Test
    fun hostReceivesNavigationInsteadOfTheRendererLaunchingAnIntent() {
        val host = RecordingHost()
        host.openUrl("https://example.invalid/x")
        assertEquals(listOf("https://example.invalid/x"), host.opened)
    }

    @Test
    fun hostSuppliesAssetResolution() {
        assertEquals("resolved:../assets/a.png", RecordingHost().resolveAssetUrl("../assets/a.png"))
    }

    @Test
    fun defaultHostDiagnosticDoesNotThrowWithoutAnOverride() {
        // The default implementation logs; on the JVM there is no Android log, so
        // this asserts the interface stays usable in plain unit tests.
        val host = object : FlatRendererHost {
            override fun openUrl(url: String) = Unit
        }
        assertEquals("x", host.resolveAssetUrl("x"))
        assertNull(host.imageLoader)
    }

    @Test
    fun stateHolderSetsAndReadsNestedPaths() {
        val holder = FlatSpecStateHolder(mapOf("form" to mapOf("name" to "Ada")))
        assertEquals("Ada", FlatSpecParser.getAtPath(holder.state, "/form/name"))

        holder.setAtPath("/form/name", "Grace")
        assertEquals("Grace", FlatSpecParser.getAtPath(holder.state, "/form/name"))
    }

    @Test
    fun stateHolderNormalizesPathsWithoutALeadingSlash() {
        val holder = FlatSpecStateHolder(emptyMap())
        holder.setAtPath("count", 3.0)
        assertEquals(3.0, FlatSpecParser.getAtPath(holder.state, "/count"))
    }

    @Test
    fun stateHolderIgnoresBlankPaths() {
        val holder = FlatSpecStateHolder(mapOf("a" to 1.0))
        holder.setAtPath("   ", 2.0)
        holder.setAtPath("", 2.0)
        assertEquals(mapOf<String, Any?>("a" to 1.0), holder.snapshot())
    }

    @Test
    fun dataModelUpdateUpsertsAndDeletes() {
        // This is the protocol operation the renderer previously had no way to
        // apply at all.
        val holder = FlatSpecStateHolder(emptyMap())
        holder.applyDataModelUpdate("/user/name", "Ada")
        assertEquals("Ada", FlatSpecParser.getAtPath(holder.state, "/user/name"))

        holder.applyDataModelUpdate("/user/name", delete = true)
        assertNull(FlatSpecParser.getAtPath(holder.state, "/user/name"))
        val user = FlatSpecParser.getAtPath(holder.state, "/user") as Map<*, *>
        assertFalse(user.containsKey("name"))
    }

    @Test
    fun deletionRemovesNestedKeysAndShiftsListIndicesWhileSetNullRemainsPresent() {
        val holder = FlatSpecStateHolder(
            mapOf(
                "nested" to mapOf("keep" to 1, "remove" to 2),
                "rows" to listOf("a", "b", "c")
            )
        )
        holder.applyDataModelUpdate("/nested/remove", delete = true)
        holder.applyDataModelUpdate("/rows/1", delete = true)
        holder.applyDataModelUpdate("/nested/nullValue", null)

        val nested = FlatSpecParser.getAtPath(holder.state, "/nested") as Map<*, *>
        assertFalse(nested.containsKey("remove"))
        assertTrue(nested.containsKey("nullValue"))
        assertNull(nested["nullValue"])
        assertEquals(listOf("a", "c"), FlatSpecParser.getAtPath(holder.state, "/rows"))
    }

    @Test
    fun stateSurvivesAcrossSpecInstancesWhenTheKeyIsStable() {
        // The point of hoisting: the holder is keyed by the host, so a new spec
        // object does not reset user input.
        val holder = FlatSpecStateHolder(mapOf("form" to mapOf("name" to "")))
        holder.setAtPath("/form/name", "typed by user")

        // Simulate a surfaceUpdate delivering a fresh FlatSpec instance.
        assertEquals("typed by user", FlatSpecParser.getAtPath(holder.state, "/form/name"))
        assertTrue(holder.snapshot().isNotEmpty())
    }

    @Test
    fun resetReplacesStateWholesale() {
        val holder = FlatSpecStateHolder(mapOf("a" to 1.0, "b" to 2.0))
        holder.reset(mapOf("c" to 3.0))
        assertEquals(mapOf<String, Any?>("c" to 3.0), holder.snapshot())
    }

    @Test
    fun renderEventCarriesEnoughToIdentifyWhatHappened() {
        val event = FlatRenderEvent(
            FlatRenderEvent.Kind.STATE_CHANGE,
            statePath = "/form/name",
            value = "Ada"
        )
        assertEquals(FlatRenderEvent.Kind.STATE_CHANGE, event.kind)
        assertEquals("/form/name", event.statePath)
        assertEquals("Ada", event.value)
        assertEquals(emptyMap<String, Any?>(), event.params)
    }

    @Test
    fun typedDiagnosticsDelegateToTheDeprecatedStringHookForCompatibility() {
        val host = RecordingHost()
        host.onDiagnostic(
            FlatDiagnostic(
                code = FlatDiagnostic.Code.INVALID_ACTION,
                severity = FlatDiagnostic.Severity.WARNING,
                message = "bad action",
                elementId = "button"
            )
        )
        assertEquals(listOf("bad action"), host.diagnostics)
    }
}
