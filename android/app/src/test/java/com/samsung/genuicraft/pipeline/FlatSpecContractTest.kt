package com.samsung.genuicraft.pipeline

import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FlatSpecContractTest {

    @Test
    fun coerceAndValidate_acceptsValidFlatSpec() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "state": { "tab": "hotels" },
              "elements": {
                "main": { "type": "Column", "props": {}, "children": ["title"] },
                "title": { "type": "Text", "props": { "text": "Hello" }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertTrue(result.isValid)
        assertFalse(result.convertedFromLegacy)
        assertEquals("main", result.spec?.get("root")?.asString)
    }

    @Test
    fun coerceAndValidate_failsWhenChildReferenceMissing() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": { "type": "Column", "props": {}, "children": ["missing"] }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertFalse(result.isValid)
        assertTrue(result.error?.contains("missing child", ignoreCase = true) == true)
    }

    @Test
    fun coerceAndValidate_failsOnUnsupportedComponentType() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": { "type": "UnknownWidget", "props": {}, "children": [] }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertFalse(result.isValid)
        assertTrue(result.error?.contains("unsupported type", ignoreCase = true) == true)
    }

    @Test
    fun coerceAndValidate_convertsLegacyMessageArray() {
        val payload = JsonParser.parseString(
            """
            [
              {
                "version": "v0.9",
                "createSurface": {
                  "surfaceId": "surface_live",
                  "catalogId": "https://genui.local/specification/v0_9/standard_catalog.json"
                }
              },
              {
                "version": "v0.9",
                "updateComponents": {
                  "surfaceId": "surface_live",
                  "components": [
                    { "id": "root", "component": "Column", "children": ["title"] },
                    { "id": "title", "component": "Text", "text": "Legacy", "children": [] }
                  ]
                }
              }
            ]
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertTrue(result.isValid)
        assertTrue(result.convertedFromLegacy)
        assertEquals("root", result.spec?.get("root")?.asString)
        assertTrue(result.spec?.getAsJsonObject("elements")?.has("title") == true)
    }

    @Test
    fun buildFallbackFlatSpec_returnsRenderableObject() {
        val fallback = FlatSpecContract.buildFallbackFlatSpec("Fallback text")
        assertEquals("root", fallback.get("root").asString)
        val elements = fallback.getAsJsonObject("elements")
        assertNotNull(elements.getAsJsonObject("root"))
        assertTrue(elements.has("text_1"))
    }

    @Test
    fun coerceAndValidate_acceptsJsonRenderOnBindings() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": {
                  "type": "Column",
                  "props": {},
                  "children": ["cta"]
                },
                "cta": {
                  "type": "Button",
                  "props": { "label": "Open" },
                  "on": {
                    "press": { "action": "openUrl", "params": { "url": "https://example.com" } }
                  },
                  "children": []
                }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertTrue(result.isValid)
    }

    @Test
    fun coerceAndValidate_rejectsLegacyPropsAction() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": {
                  "type": "Button",
                  "props": {
                    "label": "Open",
                    "action": { "functionCall": { "call": "openUrl", "args": { "url": "https://example.com" } } }
                  },
                  "children": []
                }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertFalse(result.isValid)
        assertTrue(result.error?.contains("legacy props.action", ignoreCase = true) == true)
    }

    @Test
    fun coerceAndValidate_rejectsLegacyFunctionCallInOnBinding() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": {
                  "type": "Button",
                  "props": { "label": "Open" },
                  "on": {
                    "press": { "functionCall": { "call": "openUrl", "args": { "url": "https://example.com" } } }
                  },
                  "children": []
                }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertFalse(result.isValid)
        assertTrue(result.error?.contains("functionCall", ignoreCase = true) == true)
    }

    @Test
    fun coerceAndValidate_acceptsWatchBindings() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": {
                  "type": "Column",
                  "props": {},
                  "watch": {
                    "/form/submit": {
                      "action": "validateForm",
                      "params": { "statePath": "/formValidation" }
                    }
                  },
                  "children": ["text"]
                },
                "text": { "type": "Text", "props": { "text": "ok" }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertTrue(result.isValid)
    }
}
