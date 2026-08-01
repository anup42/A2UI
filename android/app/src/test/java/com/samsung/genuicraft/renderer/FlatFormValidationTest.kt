package com.samsung.genuicraft.renderer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import com.samsung.genuicraft.renderer.flat.parse.*
import com.samsung.genuicraft.renderer.flat.expr.*
import com.samsung.genuicraft.renderer.flat.runtime.*
import com.samsung.genuicraft.renderer.flat.model.*

/**
 * Guards DoD #5 for `validateForm`, which used to write a hardcoded
 * `{"valid": true, "errors": {}}` regardless of input while the stage-3 prompt
 * told the model to emit it.
 */
class FlatFormValidationTest {

    private fun field(
        type: String = "TextField",
        statePath: String = "/form/name",
        props: Map<String, Any?> = emptyMap()
    ): Map<String, FlatElement> = mapOf(
        "f1" to FlatElement(
            type = type,
            props = props + mapOf("statePath" to statePath),
            children = emptyList()
        )
    )

    @SuppressWarnings("UNCHECKED_CAST")
    private fun errors(result: Map<String, Any?>): Map<String, String> =
        @Suppress("UNCHECKED_CAST")
        (result["errors"] as Map<String, String>)

    @Test
    fun noRulesMeansValid() {
        // This is why enabling real validation is safe for existing IR: a spec
        // whose controls declare no constraints behaves exactly as the stub did.
        val result = FlatFormValidation.validate(field(), mapOf("form" to mapOf("name" to "")))
        assertEquals(true, result["valid"])
        assertEquals(emptyMap<String, String>(), errors(result))
    }

    @Test
    fun requiredFieldFailsWhenEmptyAndPassesWhenFilled() {
        val elements = field(props = mapOf("required" to true, "label" to "Full name"))

        val empty = FlatFormValidation.validate(elements, mapOf("form" to mapOf("name" to "")))
        assertEquals(false, empty["valid"])
        assertTrue(errors(empty)["/form/name"]!!.contains("Full name"))

        val filled = FlatFormValidation.validate(elements, mapOf("form" to mapOf("name" to "Ada")))
        assertEquals(true, filled["valid"])
    }

    @Test
    fun requiredFieldFailsWhenStatePathIsAbsentEntirely() {
        val elements = field(props = mapOf("required" to true))
        val result = FlatFormValidation.validate(elements, emptyMap())
        assertEquals(false, result["valid"])
    }

    @Test
    fun emailInputTypeIsEnforced() {
        val elements = field(props = mapOf("inputType" to "email", "label" to "Email"))

        val bad = FlatFormValidation.validate(elements, mapOf("form" to mapOf("name" to "nope")))
        assertEquals(false, bad["valid"])

        val good = FlatFormValidation.validate(
            elements,
            mapOf("form" to mapOf("name" to "ada@example.com"))
        )
        assertEquals(true, good["valid"])
    }

    @Test
    fun lengthBoundsAreEnforced() {
        val elements = field(props = mapOf("minLength" to 3.0, "maxLength" to 5.0))
        assertEquals(
            false,
            FlatFormValidation.validate(elements, mapOf("form" to mapOf("name" to "ab")))["valid"]
        )
        assertEquals(
            false,
            FlatFormValidation.validate(elements, mapOf("form" to mapOf("name" to "abcdef")))["valid"]
        )
        assertEquals(
            true,
            FlatFormValidation.validate(elements, mapOf("form" to mapOf("name" to "abcd")))["valid"]
        )
    }

    @Test
    fun numericRangeIsEnforcedForSliders() {
        val elements = field(
            type = "Slider",
            statePath = "/form/qty",
            props = mapOf("min" to 1.0, "max" to 10.0)
        )
        assertEquals(
            false,
            FlatFormValidation.validate(elements, mapOf("form" to mapOf("qty" to 0.0)))["valid"]
        )
        assertEquals(
            false,
            FlatFormValidation.validate(elements, mapOf("form" to mapOf("qty" to 11.0)))["valid"]
        )
        assertEquals(
            true,
            FlatFormValidation.validate(elements, mapOf("form" to mapOf("qty" to 5.0)))["valid"]
        )
    }

    @Test
    fun unparseablePatternIsIgnoredRatherThanFailingTheField() {
        // A broken regex is a generation defect. Failing the user's input would
        // be misleading and unfixable from the UI.
        val elements = field(props = mapOf("pattern" to "([unclosed"))
        val result = FlatFormValidation.validate(elements, mapOf("form" to mapOf("name" to "x")))
        assertEquals(true, result["valid"])
    }

    @Test
    fun constraintsOtherThanRequiredDoNotFireOnEmptyValues() {
        // An optional field left blank must not report a format error.
        val elements = field(props = mapOf("inputType" to "email", "minLength" to 5.0))
        val result = FlatFormValidation.validate(elements, mapOf("form" to mapOf("name" to "")))
        assertEquals(true, result["valid"])
    }

    @Test
    fun controlsWithoutAStatePathAreSkipped() {
        val elements = mapOf(
            "f1" to FlatElement(
                type = "TextField",
                props = mapOf("required" to true, "label" to "Unbound"),
                children = emptyList()
            )
        )
        assertEquals(true, FlatFormValidation.validate(elements, emptyMap())["valid"])
    }

    @Test
    fun actionRuntimeWritesRealValidationResult() {
        val elements = field(props = mapOf("required" to true, "label" to "Full name"))
        val state = mutableMapOf<String, Any?>("form" to mapOf("name" to ""))

        FlatActionRuntime.execute(
            actionCandidate = mapOf(
                "action" to "validateForm",
                "params" to mapOf("statePath" to "/formValidation")
            ),
            stateStore = state,
            repeatScope = null,
            computedFunctions = emptyMap(),
            onOpenUrl = {},
            elements = elements
        )

        @Suppress("UNCHECKED_CAST")
        val written = state["formValidation"] as Map<String, Any?>
        assertEquals(false, written["valid"])
        assertTrue((written["errors"] as Map<*, *>).containsKey("/form/name"))
    }
}
