package com.samsung.genuicraft.sdk.internal.renderer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import com.samsung.genuicraft.sdk.internal.renderer.flat.parse.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.expr.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.runtime.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.*

/**
 * Guards DoD #5: unsupported input must produce a diagnostic rather than
 * failing silently.
 *
 * `FlatActionRuntime`'s `when` had no `else` branch, so an action the renderer
 * does not implement — `showMessage` and `showSurface` were both documented as
 * supported in the stage-3 scope notes but never wired up — was an invisible
 * no-op.
 */
class FlatRendererDiagnosticsTest {

    private fun executeAction(
        action: Map<String, Any?>,
        state: MutableMap<String, Any?> = mutableMapOf()
    ): Pair<Int, List<String>> {
        val diagnostics = mutableListOf<String>()
        val executed = FlatActionRuntime.execute(
            actionCandidate = action,
            stateStore = state,
            repeatScope = null,
            computedFunctions = emptyMap(),
            onOpenUrl = {},
            onDiagnostic = { message -> diagnostics += message }
        )
        return executed to diagnostics
    }

    @Test
    fun unsupportedActionIsReported() {
        val (executed, diagnostics) = executeAction(mapOf("action" to "showMessage"))

        assertEquals(1, diagnostics.size)
        assertTrue(
            "Diagnostic should name the action, was: ${diagnostics.first()}",
            diagnostics.first().contains("showmessage", ignoreCase = true)
        )
        assertTrue(
            "Diagnostic should list what is supported, was: ${diagnostics.first()}",
            diagnostics.first().contains("validateForm")
        )
        // Still counted, because the watch-action budget decrements on this and
        // returning 0 would change loop termination.
        assertEquals(1, executed)
    }

    @Test
    fun supportedActionProducesNoDiagnostic() {
        val state = mutableMapOf<String, Any?>()
        val (executed, diagnostics) = executeAction(
            mapOf(
                "action" to "setState",
                "params" to mapOf("statePath" to "/count", "value" to 7.0)
            ),
            state
        )

        assertEquals(emptyList<String>(), diagnostics)
        assertEquals(1, executed)
        assertEquals(7.0, state["count"])
    }

    @Test
    fun runtimeDiagnosticRetainsOriginatingElementId() {
        val diagnostics = mutableListOf<FlatDiagnostic>()
        FlatActionRuntime.executeDetailed(
            actionCandidate = mapOf(
                "action" to "openUrl",
                "params" to mapOf("url" to "javascript:alert(1)")
            ),
            stateStore = mutableMapOf(),
            repeatScope = null,
            computedFunctions = emptyMap(),
            onOpenUrl = {},
            onDiagnostic = diagnostics::add,
            elementId = "unsafe-link"
        )

        assertEquals("unsafe-link", diagnostics.single().elementId)
    }

    @Test
    fun everyDocumentedActionIsImplemented() {
        // The five actions genui_gen.md tells the model it may emit. If one of
        // these starts reporting a diagnostic, prompt and runtime have drifted.
        val documented = listOf(
            mapOf("action" to "openUrl", "params" to mapOf("url" to "https://www.samsung.com/in/")),
            mapOf("action" to "setState", "params" to mapOf("statePath" to "/a", "value" to 1.0)),
            mapOf("action" to "pushState", "params" to mapOf("statePath" to "/list", "value" to 1.0)),
            mapOf("action" to "removeState", "params" to mapOf("statePath" to "/list", "index" to 0.0)),
            mapOf("action" to "validateForm", "params" to mapOf("statePath" to "/v"))
        )

        val undiagnosed = documented.filter { action ->
            executeAction(action, mutableMapOf("list" to listOf(1.0))).second.isNotEmpty()
        }.map { action -> action["action"] }

        assertEquals(
            "These documented actions are not implemented by FlatActionRuntime",
            emptyList<Any?>(),
            undiagnosed
        )
    }

    // -- ChoicePicker selection mode (DoD #5) --------------------------------
    //
    // The picker used to always write a List back. A spec whose state held a
    // string was therefore type-mutated on first tap, permanently breaking any
    // `$state`/`eq` comparison against that string.

    @Test
    fun choicePickerInfersSingleSelectFromAStringValue() {
        assertTrue(!resolveChoicePickerMultiSelect(emptyMap(), "medium"))
    }

    @Test
    fun choicePickerInfersMultiSelectFromAListValue() {
        assertTrue(resolveChoicePickerMultiSelect(emptyMap(), listOf("a", "b")))
    }

    @Test
    fun choicePickerHonoursDeclaredModeOverValueType() {
        // Declared intent must win, otherwise an empty single-select picker
        // would be inferred as multi and widen its value on first tap.
        assertTrue(!resolveChoicePickerMultiSelect(mapOf("mode" to "single"), null))
        assertTrue(!resolveChoicePickerMultiSelect(mapOf("mode" to "mutuallyExclusive"), listOf("a")))
        assertTrue(resolveChoicePickerMultiSelect(mapOf("mode" to "multiple"), "a"))
        assertTrue(!resolveChoicePickerMultiSelect(mapOf("multiple" to false), listOf("a")))
        assertTrue(resolveChoicePickerMultiSelect(mapOf("multiple" to true), "a"))
    }

    @Test
    fun choicePickerDefaultsToMultiSelectWhenNothingIsDeclaredOrSet() {
        // Preserves the previous behaviour for specs that declare nothing and
        // start from an empty value.
        assertTrue(resolveChoicePickerMultiSelect(emptyMap(), null))
    }

    @Test
    fun blankActionNameIsIgnoredWithoutDiagnostic() {
        // A blank name is skipped before dispatch, so it must not be counted or
        // reported; otherwise malformed IR would spam the log.
        val (executed, diagnostics) = executeAction(mapOf("action" to "   "))
        assertEquals(emptyList<String>(), diagnostics)
        assertEquals(0, executed)
    }
}
