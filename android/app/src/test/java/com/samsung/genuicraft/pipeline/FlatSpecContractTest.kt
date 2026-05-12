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
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["title"] },
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
    fun coerceAndValidate_acceptsEmailPreviewElement() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["email"] },
                "email": {
                  "type": "EmailPreview",
                  "props": {
                    "subject": "Interview follow-up",
                    "to": "Dr. Reed",
                    "body": ["Dear Dr. Reed,", "Thank you for your time."]
                  },
                  "children": []
                }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertTrue(result.error.orEmpty(), result.isValid)
    }

    @Test
    fun coerceAndValidate_acceptsChartElement() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "state": {
                "rows": [
                  { "age": "18-24", "tiktok": "75.0%" },
                  { "age": "25-34", "tiktok": "0.0%" }
                ]
              },
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["chart"] },
                "chart": {
                  "type": "Chart",
                  "props": {
                    "chartType": "bar",
                    "statePath": "/rows",
                    "columns": [{"key":"age","label":"Age"},{"key":"tiktok","label":"TikTok"}],
                    "xKey": "age",
                    "yKey": "tiktok"
                  },
                  "children": []
                }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertTrue(result.error.orEmpty(), result.isValid)
    }

    @Test
    fun coerceAndValidate_rejectsIconUrlInTableImageColumn() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "state": {
                "days": [
                  {
                    "dayDate": "Day 1",
                    "area": "Central Bengaluru",
                    "image": "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/signpost-split.svg"
                  }
                ]
              },
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["itinerary"] },
                "itinerary": {
                  "type": "Table",
                  "props": {
                    "columns": [
                      {"key":"dayDate","label":"Day"},
                      {"key":"area","label":"Area"},
                      {"key":"image","label":"Image"}
                    ],
                    "statePath": "/days",
                    "domain": "schedule",
                    "preferredPresentation": "cards"
                  },
                  "children": []
                }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertFalse(result.isValid)
        assertTrue(result.error.orEmpty().contains("icon/vector media"))
    }

    @Test
    fun coerceAndValidate_rejectsUnsafeOpenUrlAction() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "state": {},
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["button"] },
                "button": {
                  "type": "Button",
                  "props": { "label": "Open" },
                  "on": {
                    "press": { "action": "openUrl", "params": { "url": "javascript:alert(1)" } }
                  },
                  "children": []
                }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertFalse(result.isValid)
        assertTrue(result.error.orEmpty().contains("unsafe URL"))
    }

    @Test
    fun coerceAndValidate_rejectsUnsafeMediaElementSources() {
        val cases = listOf(
            "Image" to """"url": "https://loremflickr.com/1200/800/travel"""",
            "Video" to """"url": "file:///sdcard/clip.mp4"""",
            "AudioPlayer" to """"url": "http://example.org/audio.mp3""""
        )

        cases.forEach { (type, props) ->
            val payload = JsonParser.parseString(
                """
                {
                  "root": "main",
                  "state": {},
                  "elements": {
                    "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["media"] },
                    "media": { "type": "$type", "props": { $props }, "children": [] }
                  }
                }
                """.trimIndent()
            )

            val result = FlatSpecContract.coerceAndValidate(payload)
            assertFalse("$type should be rejected", result.isValid)
            assertTrue(result.error.orEmpty().contains("safe media policy"))
        }
    }

    @Test
    fun coerceAndValidate_acceptsBootstrapIconAsIconOnly() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "state": {},
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["icon"] },
                "icon": {
                  "type": "Icon",
                  "props": { "name": "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/geo-alt.svg" },
                  "children": []
                }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertTrue(result.error.orEmpty(), result.isValid)
    }

    @Test
    fun coerceAndValidate_acceptsFormulaElement() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["formula"] },
                "formula": {
                  "type": "Formula",
                  "props": {
                    "latex": "M = P \\\\frac{i(1+i)^n}{(1+i)^n - 1}",
                    "result": "${'$'}2,451.25"
                  },
                  "children": []
                }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertTrue(result.error.orEmpty(), result.isValid)
    }

    @Test
    fun coerceAndValidate_failsWhenElementsIsEmpty() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {}
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertFalse(result.isValid)
        assertEquals("Elements map is empty.", result.error)
    }

    @Test
    fun coerceAndValidate_rewritesMissingRootToExistingElement() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "missing_root",
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertTrue(result.isValid)
        assertEquals("main", result.spec?.get("root")?.asString)
    }

    @Test
    fun coerceAndValidate_failsWhenChildReferenceMissing() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["missing"] }
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
                    { "id": "root", "component": "Column", "children": ["row_title"] },
                    { "id": "row_title", "component": "Row", "children": ["title"], "justify": "spaceBetween" },
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
        val rootElement = result.spec?.getAsJsonObject("elements")?.getAsJsonObject("root")
        assertEquals("Stack", rootElement?.get("type")?.asString)
        assertFalse(rootElement?.getAsJsonObject("props")?.has("direction") == true)
        val rowElement = result.spec?.getAsJsonObject("elements")?.getAsJsonObject("row_title")
        assertEquals("Stack", rowElement?.get("type")?.asString)
        assertEquals("horizontal", rowElement?.getAsJsonObject("props")?.get("direction")?.asString)
        assertEquals("between", rowElement?.getAsJsonObject("props")?.get("justify")?.asString)
    }

    @Test
    fun buildFallbackFlatSpec_returnsRenderableObject() {
        val fallback = FlatSpecContract.buildFallbackFlatSpec("Fallback text")
        assertEquals("root", fallback.get("root").asString)
        val elements = fallback.getAsJsonObject("elements")
        assertNotNull(elements.getAsJsonObject("root"))
        assertTrue(elements.has("text_1"))
        assertEquals("Stack", elements.getAsJsonObject("root").get("type").asString)
        assertEquals(
            "vertical",
            elements.getAsJsonObject("root").getAsJsonObject("props").get("direction").asString
        )
    }

    @Test
    fun coerceAndValidate_acceptsJsonRenderOnBindings() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": {
                  "type": "Stack",
                  "props": { "direction": "vertical", "gap": "md" },
                  "children": ["cta"]
                },
                "cta": {
                  "type": "Button",
                  "props": { "label": "Open" },
                  "on": {
                    "press": { "action": "openUrl", "params": { "url": "https://www.google.com/search?q=genui" } }
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
    fun coerceAndValidate_rejectsLegacyRowTypeInFlatSpec() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": {
                  "type": "Row",
                  "props": {},
                  "children": ["text"]
                },
                "text": { "type": "Text", "props": { "text": "ok" }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertFalse(result.isValid)
        assertTrue(result.error?.contains("unsupported type", ignoreCase = true) == true)
    }

    @Test
    fun coerceAndValidate_acceptsStackLayoutProps() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": {
                  "type": "Stack",
                  "props": {
                    "direction": "horizontal",
                    "gap": "lg",
                    "align": "center",
                    "justify": "between",
                    "wrap": "wrap",
                    "padding": 8,
                    "paddingHorizontal": 16,
                    "paddingVertical": 4,
                    "margin": 2,
                    "marginHorizontal": 3,
                    "marginVertical": 1,
                    "width": 240,
                    "height": 80,
                    "flex": 1
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

    @Test
    fun coerceAndValidate_rejectsInvalidStackProps() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": {
                  "type": "Stack",
                  "props": {
                    "direction": "diagonal",
                    "gap": "xxl",
                    "padding": "large",
                    "className": "foo"
                  },
                  "children": ["text"]
                },
                "text": { "type": "Text", "props": { "text": "ok" }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertFalse(result.isValid)
    }

    @Test
    fun coerceAndValidate_acceptsWatchBindings() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": {
                  "type": "Stack",
                  "props": { "direction": "vertical" },
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

    @Test
    fun coerceAndValidate_rejectsClassNameProp() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": {
                  "type": "Stack",
                  "props": { "direction": "vertical", "className": "p-4" },
                  "children": ["text"]
                },
                "text": { "type": "Text", "props": { "text": "ok" }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertFalse(result.isValid)
        assertTrue(result.error?.contains("className", ignoreCase = true) == true)
    }

    @Test
    fun coerceAndValidate_rejectsLegacyColumnTypeInFlatSpec() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": {
                  "type": "Column",
                  "props": {},
                  "children": ["text"]
                },
                "text": { "type": "Text", "props": { "text": "ok" }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertFalse(result.isValid)
        assertTrue(result.error?.contains("unsupported type", ignoreCase = true) == true)
    }

    @Test
    fun coerceAndValidate_hoistsRepeatAndTemplateAliasesForCanonicalTable() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "state": { "rows": [ { "id": "a", "name": "Bengaluru", "temp": "29C" } ] },
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["table"] },
                "table": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["header", "body"] },
                "header": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["h1", "h2"] },
                "h1": { "type": "Text", "props": { "text": "City" }, "children": [] },
                "h2": { "type": "Text", "props": { "text": "Temp" }, "children": [] },
                "body": {
                  "type": "Stack",
                  "props": {
                    "repeat": { "path": "rows", "key": "id" },
                    "template": "row_template"
                  },
                  "children": []
                },
                "row_template": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["c1", "c2"] },
                "c1": { "type": "Text", "props": { "text": { "${'$'}item": "name" } }, "children": [] },
                "c2": { "type": "Text", "props": { "text": { "${'$'}item": "temp" } }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertTrue(result.isValid)
        val body = result.spec!!
            .getAsJsonObject("elements")
            .getAsJsonObject("body")
        assertTrue(body.has("repeat"))
        assertEquals("/rows", body.getAsJsonObject("repeat").get("statePath").asString)
        val props = body.getAsJsonObject("props")
        assertFalse(props.has("repeat"))
        assertFalse(props.has("template"))
        val children = body.getAsJsonArray("children").map { it.asString }
        assertTrue(children.contains("row_template"))
        assertTrue(result.warnings.any { it.contains("hoisted props.repeat", ignoreCase = true) })
    }

    @Test
    fun coerceAndValidate_normalizesTableAlignmentAliases() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": {
                  "type": "Stack",
                  "props": {
                    "direction": "horizontal",
                    "distribution": "between",
                    "verticalAlign": "center"
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
        val props = result.spec!!
            .getAsJsonObject("elements")
            .getAsJsonObject("main")
            .getAsJsonObject("props")
        assertEquals("between", props.get("mainAxisAlignment").asString)
        assertEquals("center", props.get("verticalAlignment").asString)
        assertFalse(props.has("distribution"))
        assertFalse(props.has("verticalAlign"))
    }

    @Test
    fun coerceAndValidate_promotesStyleAliasesToTopLevelProps() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": {
                  "type": "Stack",
                  "props": {
                    "direction": "horizontal",
                    "style": {
                      "distribution": "between",
                      "verticalAlign": "center",
                      "padding": 12
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
        val props = result.spec!!
            .getAsJsonObject("elements")
            .getAsJsonObject("main")
            .getAsJsonObject("props")
        assertEquals("between", props.get("mainAxisAlignment").asString)
        assertEquals("center", props.get("verticalAlignment").asString)
        assertEquals(12, props.get("padding").asInt)
        assertTrue(result.warnings.any { it.contains("promoted props.style", ignoreCase = true) })
    }

    @Test
    fun coerceAndValidate_reportsTableDiagnosticsForWideGenericTables() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "state": {
                "rows": [
                  { "id": "r1", "c1": "A", "c2": "B", "c3": "C", "c4": "D" },
                  { "id": "r2", "c1": "E", "c2": "F", "c3": "G", "c4": "H" }
                ]
              },
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["table"] },
                "table": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["header", "body"] },
                "header": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["h1", "h2", "h3", "h4"] },
                "h1": { "type": "Text", "props": { "text": "Col1" }, "children": [] },
                "h2": { "type": "Text", "props": { "text": "Col2" }, "children": [] },
                "h3": { "type": "Text", "props": { "text": "Col3" }, "children": [] },
                "h4": { "type": "Text", "props": { "text": "Col4" }, "children": [] },
                "body": {
                  "type": "Stack",
                  "props": { "direction": "vertical" },
                  "repeat": { "statePath": "/rows", "key": "id" },
                  "children": ["row"]
                },
                "row": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["r1", "r2", "r3", "r4"] },
                "r1": { "type": "Text", "props": { "text": { "${'$'}item": "c1" } }, "children": [] },
                "r2": { "type": "Text", "props": { "text": { "${'$'}item": "c2" } }, "children": [] },
                "r3": { "type": "Text", "props": { "text": { "${'$'}item": "c3" } }, "children": [] },
                "r4": { "type": "Text", "props": { "text": { "${'$'}item": "c4" } }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertTrue(result.isValid)
        assertTrue(result.tableDiagnostics.tableDetected)
        assertEquals(4, result.tableDiagnostics.columns)
        assertEquals(2, result.tableDiagnostics.rows)
        assertEquals("horizontal_scroll_table", result.tableDiagnostics.renderMode)
    }

    @Test
    fun coerceAndValidate_compactsStaticTableRowsIntoRepeatState() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["table"] },
                "table": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["header", "row1", "row2"] },
                "header": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["h1", "h2"] },
                "h1": { "type": "Text", "props": { "text": "City" }, "children": [] },
                "h2": { "type": "Text", "props": { "text": "Temp" }, "children": [] },
                "row1": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["r11", "r12"] },
                "r11": { "type": "Text", "props": { "text": "Bengaluru" }, "children": [] },
                "r12": { "type": "Text", "props": { "text": "29C" }, "children": [] },
                "row2": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["r21", "r22"] },
                "r21": { "type": "Text", "props": { "text": "Mysuru" }, "children": [] },
                "r22": { "type": "Text", "props": { "text": "27C" }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertTrue(result.isValid)
        val elements = result.spec!!.getAsJsonObject("elements")
        val table = elements.getAsJsonObject("table")
        val tableChildren = table.getAsJsonArray("children").map { it.asString }
        assertEquals(2, tableChildren.size)
        val body = elements.getAsJsonObject(tableChildren[1])
        assertTrue(body.has("repeat"))
        val repeatPath = body.getAsJsonObject("repeat").get("statePath").asString
        assertTrue(repeatPath.startsWith("/"))
        val stateKey = repeatPath.removePrefix("/")
        val rows = result.spec!!.getAsJsonObject("state").getAsJsonArray(stateKey)
        assertNotNull(rows)
        assertEquals(2, rows.size())
        assertTrue(result.tableDiagnostics.compactionApplied)
    }

    @Test
    fun coerceAndValidate_setsWeatherDomainDefaultsForTable() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "state": { "forecast": [ { "day": "Mon", "temp": "29C", "humidity": "65%" } ] },
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["table"] },
                "table": {
                  "type": "Stack",
                  "props": { "direction": "vertical", "domain": "generic", "preferredPresentation": "table" },
                  "children": ["header", "rows"]
                },
                "header": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["h1", "h2", "h3"] },
                "h1": { "type": "Text", "props": { "text": "Day" }, "children": [] },
                "h2": { "type": "Text", "props": { "text": "Temperature" }, "children": [] },
                "h3": { "type": "Text", "props": { "text": "Humidity" }, "children": [] },
                "rows": { "type": "Stack", "props": { "direction": "vertical" }, "repeat": { "statePath": "/forecast" }, "children": ["row"] },
                "row": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["c1", "c2", "c3"] },
                "c1": { "type": "Text", "props": { "text": { "${'$'}item": "day" } }, "children": [] },
                "c2": { "type": "Text", "props": { "text": { "${'$'}item": "temp" } }, "children": [] },
                "c3": { "type": "Text", "props": { "text": { "${'$'}item": "humidity" } }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertTrue(result.isValid)
        val tableProps = result.spec!!
            .getAsJsonObject("elements")
            .getAsJsonObject("table")
            .getAsJsonObject("props")
        assertEquals("weather", tableProps.get("domain").asString)
        assertEquals("cards", tableProps.get("preferredPresentation").asString)
        assertEquals("weather", result.tableDiagnostics.tableDomain)
        assertTrue(result.warnings.any { it.contains("rewrote props.domain from generic to weather", ignoreCase = true) })
    }

    @Test
    fun coerceAndValidate_acceptsCompactTableElement() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "state": {
                "forecast": [
                  { "day": "Sun, Apr 12", "condition": "Partly sunny / Clear", "high": "34°C / 94°F", "low": "23°C / 73°F" }
                ]
              },
              "elements": {
                "main": {
                  "type": "Stack",
                  "props": { "direction": "vertical" },
                  "children": ["title", "forecast_table"]
                },
                "title": { "type": "Text", "props": { "text": "Forecast", "variant": "h2" }, "children": [] },
                "forecast_table": {
                  "type": "Table",
                  "props": {
                    "domain": "generic",
                    "preferredPresentation": "table",
                    "statePath": "/forecast",
                    "columns": [
                      { "key": "day", "label": "Day" },
                      { "key": "condition", "label": "Conditions" },
                      { "key": "high", "label": "High (°C/°F)" },
                      { "key": "low", "label": "Low (°C/°F)" }
                    ]
                  },
                  "children": []
                }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertTrue(result.isValid)
        val tableProps = result.spec!!
            .getAsJsonObject("elements")
            .getAsJsonObject("forecast_table")
            .getAsJsonObject("props")
        assertEquals("weather", tableProps.get("domain").asString)
        assertEquals("cards", tableProps.get("preferredPresentation").asString)
        assertTrue(result.tableDiagnostics.tableDetected)
        assertEquals("weather", result.tableDiagnostics.tableDomain)
        assertEquals(4, result.tableDiagnostics.columns)
    }

    @Test
    fun coerceAndValidate_prunesUnreachableElementsAfterCanonicalization() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["title"] },
                "title": { "type": "Text", "props": { "text": "Hello" }, "children": [] },
                "orphan": { "type": "Text", "props": { "text": "remove me" }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val result = FlatSpecContract.coerceAndValidate(payload)
        assertTrue(result.isValid)
        val elements = result.spec!!.getAsJsonObject("elements")
        assertFalse(elements.has("orphan"))
        assertTrue(result.warnings.any { it.contains("unreachable element", ignoreCase = true) })
    }
}
