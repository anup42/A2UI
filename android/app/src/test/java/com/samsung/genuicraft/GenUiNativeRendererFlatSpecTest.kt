package com.samsung.genuicraft

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class GenUiNativeRendererFlatSpecTest {

    private val flatSpecJson = """
        {
          "root": "main",
          "state": {
            "selected": "details",
            "items": [
              { "id": "card_1", "title": "Alpha", "image": "../assets/card_alpha.png" }
            ]
          },
          "elements": {
            "main": { "type": "Stack", "props": { "direction": "vertical", "gap": "md" }, "children": ["title", "cards", "details"] },
            "title": { "type": "Text", "props": { "text": "Example", "variant": "h2" }, "children": [] },
            "cards": {
              "type": "Stack",
              "props": { "direction": "vertical", "gap": "md" },
              "children": ["card"],
              "repeat": { "statePath": "/items", "key": "id" }
            },
            "card": { "type": "Card", "props": {}, "children": ["card_image", "card_title"] },
            "card_image": { "type": "Image", "props": { "url": { "${'$'}item": "image" }, "fit": "cover" }, "children": [] },
            "card_title": { "type": "Text", "props": { "text": { "${'$'}item": "title" } }, "children": [] },
            "details": {
              "type": "Text",
              "props": { "text": { "${'$'}template": "Selected ${'$'}{/selected}" } },
              "children": [],
              "visible": { "${'$'}state": "/selected", "eq": "details" }
            }
          }
        }
    """.trimIndent()

    @Test
    fun render_acceptsTopLevelFlatSpec() {
        val result = GenUiNativeRenderer.render(flatSpecJson, sourceDir = null)

        assertNull(result.errorMessage)
        assertEquals(1, result.surfaces.size)
        assertTrue(result.warnings.isEmpty())
        val surface = result.surfaces.single()
        val spec = surface.flatSpec
        assertNotNull(spec)
        assertEquals("main", surface.rootId)
        assertEquals("/items", spec!!.elements["cards"]?.repeat?.statePath)
        assertNotNull(spec.elements["details"]?.visible)
    }

    @Test
    fun render_acceptsFlatSpecWrappedInGenUiJsonObject() {
        val wrapper = JsonObject().apply {
            addProperty("ui_id", "u_test")
            add("genui_json", JsonParser.parseString(flatSpecJson))
        }

        val result = GenUiNativeRenderer.render(wrapper.toString(), sourceDir = null)

        assertNull(result.errorMessage)
        assertEquals(1, result.surfaces.size)
        assertNotNull(result.surfaces.single().flatSpec)
        assertTrue(result.warnings.none { it.contains("fallback", ignoreCase = true) })
    }

    @Test
    fun render_acceptsFlatSpecWrappedInPayloadObject() {
        val wrapper = JsonObject().apply {
            add(
                "payload",
                JsonObject().apply {
                    add("genui_json", JsonParser.parseString(flatSpecJson))
                }
            )
        }

        val result = GenUiNativeRenderer.render(wrapper.toString(), sourceDir = null)

        assertNull(result.errorMessage)
        assertEquals(1, result.surfaces.size)
        assertNotNull(result.surfaces.single().flatSpec)
    }

    @Test
    fun render_acceptsFlatSpecStoredAsJsonString() {
        val wrapper = JsonObject().apply {
            addProperty("stage3_json", flatSpecJson)
        }

        val result = GenUiNativeRenderer.render(wrapper.toString(), sourceDir = null)

        assertNull(result.errorMessage)
        assertEquals(1, result.surfaces.size)
        assertNotNull(result.surfaces.single().flatSpec)
    }

    @Test
    fun render_keepsLegacyMessageArraySupport() {
        val payload = JsonArray().apply {
            add(
                JsonObject().apply {
                    add(
                        "createSurface",
                        JsonObject().apply {
                            addProperty("surfaceId", "surface_live")
                        }
                    )
                }
            )
            add(
                JsonObject().apply {
                    add(
                        "updateComponents",
                        JsonObject().apply {
                            addProperty("surfaceId", "surface_live")
                            add(
                                "components",
                                JsonArray().apply {
                                    add(
                                        JsonObject().apply {
                                            addProperty("id", "root")
                                            addProperty("component", "Column")
                                            add(
                                                "children",
                                                JsonArray().apply { add("title") }
                                            )
                                        }
                                    )
                                    add(
                                        JsonObject().apply {
                                            addProperty("id", "title")
                                            addProperty("component", "Text")
                                            addProperty("text", "Legacy")
                                        }
                                    )
                                }
                            )
                        }
                    )
                }
            )
        }

        val result = GenUiNativeRenderer.render(payload.toString(), sourceDir = null)

        assertNull(result.errorMessage)
        assertEquals(1, result.surfaces.size)
        assertNull(result.surfaces.single().flatSpec)
        assertFalse(result.surfaces.single().components.isEmpty())
    }

    @Test
    fun render_bridgesStackTextHeavyFallbackPayloadForHistoricalData() {
        val payload = """
            {
              "root": "root",
              "state": {},
              "elements": {
                "root": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["content"] },
                "content": {
                  "type": "Text",
                  "props": { "variant": "body", "text": "## Weather\n| Day | Temp |\n| --- | --- |\n| Today | 32°C |" },
                  "children": []
                }
              }
            }
        """.trimIndent()

        val result = GenUiNativeRenderer.render(payload, sourceDir = null)

        assertNull(result.errorMessage)
        assertEquals(1, result.surfaces.size)
        assertNull(result.surfaces.single().flatSpec)
        assertFalse(result.surfaces.single().components.isEmpty())
        assertTrue(
            result.warnings.any { it.contains("text-heavy fallback content", ignoreCase = true) }
        )
    }
}
