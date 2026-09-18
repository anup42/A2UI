package com.samsung.genuicraft.sdk.internal.pipeline

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import com.samsung.genuicraft.sdk.internal.renderer.flat.parse.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.expr.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.runtime.*

/**
 * Guards DoD #8: the flat-spec contract is implemented three times — Kotlin
 * ([FlatSpecContract]), Python (`dataset/src/pipeline/flat_spec_contract.py`)
 * and JSON Schema (`dataset/schema/genui_flatspec.schema.json`) — and they must
 * not drift.
 *
 * This reads the real schema rather than a mirrored copy, so there is no second
 * source of truth to fall out of date. It asserts behaviour through
 * [FlatSpecContract.coerceAndValidate] instead of reaching into the private
 * allow-list, so the test survives the renderer refactor.
 */
class FlatSpecContractSchemaParityTest {

    private companion object {
        val ACTIONS_EXPECTED = listOf(
            "openUrl",
            "setState",
            "pushState",
            "removeState",
            "validateForm",
            "emitEvent"
        )

        val INTENTIONALLY_UNSUPPORTED = emptySet<String>()
    }

    private fun schema(): JsonObject =
        requireNotNull(javaClass.classLoader?.getResourceAsStream("genui_flatspec.schema.json"))
            .reader(Charsets.UTF_8)
            .use { JsonParser.parseReader(it).asJsonObject }

    private fun schemaEnum(vararg path: String): List<String> {
        var node: JsonObject = schema()
        path.dropLast(1).forEach { key ->
            node = node.getAsJsonObject(key)
                ?: throw IllegalStateException("Missing schema node '$key' in ${path.toList()}")
        }
        return node.getAsJsonArray(path.last())
            .map { entry -> entry.asString }
    }

    private fun specWithElement(type: String): String = """
        {
          "root": "main",
          "state": {},
          "elements": {
            "main": {
              "type": "Stack",
              "props": { "direction": "vertical" },
              "children": ["subject"]
            },
            "subject": { "type": "$type", "props": { "text": "hello" }, "children": [] }
          }
        }
    """.trimIndent()

    @Test
    fun everySchemaElementTypeIsAcceptedByTheKotlinContract() {
        val schemaTypes = schemaEnum("\$defs", "element", "properties", "type", "enum")
        assertTrue("Schema declared no element types", schemaTypes.isNotEmpty())

        val rejected = schemaTypes.filterNot { type ->
            FlatSpecContract.coerceAndValidate(
                JsonParser.parseString(specWithElement(type))
            ).isValid
        }.toSet()

        assertEquals(
            "Divergence between genui_flatspec.schema.json (${schemaTypes.size} " +
                "types) and FlatSpecContract.allowedTypes changed. If a type was " +
                "added to the schema, add it to the Kotlin contract and the " +
                "Python contract. If Android now supports Row/Column, update " +
                "INTENTIONALLY_UNSUPPORTED.",
            INTENTIONALLY_UNSUPPORTED,
            rejected
        )
    }

    @Test
    fun schemaElementTypesCoverTheDocumentedCatalogSize() {
        // Pins the catalog size so adding a type to one side without the other
        // is a test failure rather than a silent divergence.
        val schemaTypes = schemaEnum("\$defs", "element", "properties", "type", "enum")
        assertEquals(
            "Flat-spec catalog changed. Update FlatSpecContract.allowedTypes, " +
                "dataset/src/pipeline/flat_spec_contract.py, android/README.md " +
                "and the stage-3 scope notes together.",
            25,
            schemaTypes.size
        )
    }

    @Test
    fun schemaActionsMatchTheImplementedActionRuntime() {
        val schemaActions = schemaEnum("\$defs", "action", "properties", "action", "enum")
        assertEquals(
            "Schema actions drifted from the actions FlatActionRuntime implements.",
            ACTIONS_EXPECTED.sorted(),
            schemaActions.sorted()
        )
    }

    @Test
    fun unknownElementTypeIsRejectedWithADiagnosticError() {
        val result = FlatSpecContract.coerceAndValidate(
            JsonParser.parseString(specWithElement("Hologram"))
        )
        assertTrue("Unknown type should be rejected", !result.isValid)
        assertTrue(
            "Error should name the offending type, was: ${result.error}",
            result.error.orEmpty().contains("Hologram", ignoreCase = true) ||
                result.error.orEmpty().contains("unsupported", ignoreCase = true)
        )
    }

    @Test
    fun compatibilityRowAndColumnCanonicalizeToStackDirections() {
        mapOf("Row" to "horizontal", "Column" to "vertical").forEach { (legacy, direction) ->
            val result = FlatSpecContract.coerceAndValidate(
                JsonParser.parseString(specWithElement(legacy))
            )
            assertTrue("$legacy should canonicalize", result.isValid)
            val subject = result.spec!!.getAsJsonObject("elements").getAsJsonObject("subject")
            assertEquals("Stack", subject.get("type").asString)
            assertEquals(direction, subject.getAsJsonObject("props").get("direction").asString)
        }
    }
}
