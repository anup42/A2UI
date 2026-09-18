package com.samsung.genuicraft.sdk.internal.renderer

import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.extractDirectTableModel
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.hasExplicitTablePresentation
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.extractFlatTableModel
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.looksLikeStudyPlanTable
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.looksLikeTravelItineraryTable
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.responsiveScheduleCardContent
import com.samsung.genuicraft.sdk.internal.renderer.flat.parse.FlatSpecParser
import org.junit.Assert.*
import org.junit.Test

class ExplicitTablePresentationTest {
    @Test fun `explicit two-column table retains meaning and units instead of key-value cards`() {
        val headers = listOf("Reading", "Temperature (°C)")
        val rows = listOf(listOf("Inside", "21"), listOf("Outside", "14"))
        val props = mapOf("columns" to headers, "rows" to rows, "preferredPresentation" to "table")
        val model = requireNotNull(extractDirectTableModel(props, emptyMap(), compactScreen = true))
        assertTrue(hasExplicitTablePresentation(props))
        assertEquals(FlatTableShape.KEY_VALUE, model.shape)
        assertEquals(FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL, model.renderMode)
        assertEquals(headers, model.columns.map { it.label })
        assertEquals(rows, model.rows)
    }

    @Test fun `explicit weather table wins while requested and default weather cards remain cards`() {
        val base = mapOf<String, Any?>(
            "columns" to listOf("Date", "High (°C)", "Low (°C)", "Rain (%)"),
            "rows" to listOf(listOf("Wed, Sep 9", "31", "20", "57")),
            "domain" to "weather"
        )
        val table = requireNotNull(extractDirectTableModel(base + ("preferredPresentation" to "table"), emptyMap(), true))
        assertEquals(FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL, table.renderMode)
        assertEquals(listOf("Date", "High (°C)", "Low (°C)", "Rain (%)"), table.columns.map { it.label })
        assertEquals(listOf("Wed, Sep 9", "31", "20", "57"), table.rows.single())
        listOf(base, base + ("preferredPresentation" to "cards")).forEach { props ->
            assertFalse(hasExplicitTablePresentation(props))
            assertEquals(FlatTableRenderMode.WEATHER_CARDS, requireNotNull(extractDirectTableModel(props, emptyMap(), true)).renderMode)
        }
    }

    @Test fun `process cards remain available when table is not explicitly requested`() {
        val base = mapOf<String, Any?>(
            "columns" to listOf("UI State", "Key Visuals", "On-Screen Text and Feedback"),
            "rows" to listOf(
                listOf("Ready to Scan", "Camera preview with QR scan frame.", "Scan your ticket QR code."),
                listOf("Check-in Success", "Green confirmation overlay.", "Check-in successful.")
            ),
            "domain" to "status"
        )
        listOf(base, base + ("preferredPresentation" to "cards")).forEach { props ->
            assertFalse(hasExplicitTablePresentation(props))
            assertEquals(FlatTableRenderMode.PROCESS_CARDS, requireNotNull(extractDirectTableModel(props, emptyMap(), true)).renderMode)
        }
    }

    @Test fun `explicit summary table keeps rows even when legacy itinerary suppression would match`() {
        val headers = listOf("Trip detail", "Value")
        val rows = listOf(listOf("Destination", "Mysuru"), listOf("Trip length", "3 days"))
        assertTrue(looksLikeTravelItinerarySummaryTable(headers, rows))
        val props = mapOf("columns" to headers, "rows" to rows, "preferredPresentation" to "table")
        assertTrue(hasExplicitTablePresentation(props))
        val model = requireNotNull(extractDirectTableModel(props, emptyMap(), true))
        assertEquals(FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL, model.renderMode)
        assertEquals(headers, model.columns.map { it.label })
        assertEquals(rows, model.rows)
    }

    @Test fun `wide displays use a grid and explicit value normalization is consistent`() {
        val props = mapOf<String, Any?>(
            "columns" to listOf("Substance", "Mass (mg)"),
            "rows" to listOf(listOf("Co", "25")),
            "preferredPresentation" to " TABLE "
        )
        assertTrue(hasExplicitTablePresentation(props))
        val model = requireNotNull(extractDirectTableModel(props, emptyMap(), compactScreen = false))
        assertEquals(FlatTableRenderMode.TABLE, model.renderMode)
        assertEquals("Mass (mg)", model.columns[1].label)
        assertEquals("Co", model.rows.single()[0])
    }

    @Test fun `implicit generic key-value presentation remains unchanged`() {
        val props = mapOf("columns" to listOf("Reading", "Value"), "rows" to listOf(listOf("Inside", "21")))
        assertFalse(hasExplicitTablePresentation(props))
        val model = requireNotNull(extractDirectTableModel(props, emptyMap(), compactScreen = true))
        assertEquals("table", model.preferredPresentation)
        assertEquals(FlatTableRenderMode.RESPONSIVE_CARD_ROWS, model.renderMode)
    }

    @Test fun `direct schedule card request bypasses grid and retains every labeled field`() {
        val headers = listOf(
            "Train service",
            "Departure station",
            "Typical departure time",
            "Journey duration",
            "Seating class"
        )
        val row = listOf(
            "**Shatabdi Express (12007)**",
            "KSR Bengaluru (SBC)",
            "**10:35–10:45**",
            "**About 2 h 15 m to 2 h 25 m**",
            "**CC, EC**"
        )
        val props = mapOf<String, Any?>(
            "columns" to headers,
            "rows" to listOf(row),
            "domain" to "schedule",
            "preferredPresentation" to "cards"
        )

        val model = requireNotNull(extractDirectTableModel(props, emptyMap(), compactScreen = true))
        assertEquals(FlatTableShape.SCHEDULE_TIMELINE, model.shape)
        assertEquals(FlatTableRenderMode.RESPONSIVE_CARD_ROWS, model.renderMode)
        assertFalse(shouldUseScrollableNativeTableGrid(model.renderMode, hasRows = true))
        assertFalse(looksLikeTravelItineraryTable(headers))
        assertFalse(looksLikeStudyPlanTable(headers))
        assertEquals(
            AdaptiveTablePresentation.TIMELINE_CARDS,
            selectAdaptiveTablePresentation(
                table = model,
                screenWidthDp = 412,
                isLandscape = false,
                autoHorizontalScroll = true,
                cardsRequested = false
            )
        )
        val content = requireNotNull(responsiveScheduleCardContent(headers, row))
        assertEquals(
            headers.zip(row),
            content.displayedCells.map { cell -> cell.label to cell.value }
        )
    }

    @Test fun `component schedule card mode bypasses grid while explicit direct table still uses it`() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "table",
              "elements": {
                "table": {
                  "type": "Stack",
                  "props": { "direction": "vertical", "domain": "schedule", "preferredPresentation": "cards" },
                  "children": ["header", "row"]
                },
                "header": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["h1", "h2", "h3", "h4", "h5"] },
                "h1": { "type": "Text", "props": { "text": "Train service" }, "children": [] },
                "h2": { "type": "Text", "props": { "text": "Departure station" }, "children": [] },
                "h3": { "type": "Text", "props": { "text": "Typical departure time" }, "children": [] },
                "h4": { "type": "Text", "props": { "text": "Journey duration" }, "children": [] },
                "h5": { "type": "Text", "props": { "text": "Seating class" }, "children": [] },
                "row": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["c1", "c2", "c3", "c4", "c5"] },
                "c1": { "type": "Text", "props": { "text": "Shatabdi Express (12007)" }, "children": [] },
                "c2": { "type": "Text", "props": { "text": "KSR Bengaluru (SBC)" }, "children": [] },
                "c3": { "type": "Text", "props": { "text": "10:35–10:45" }, "children": [] },
                "c4": { "type": "Text", "props": { "text": "About 2 h 15 m to 2 h 25 m" }, "children": [] },
                "c5": { "type": "Text", "props": { "text": "CC, EC" }, "children": [] }
              }
            }
            """.trimIndent()
        )
        val parsed = requireNotNull(FlatSpecParser.parse(payload))
        val table = requireNotNull(parsed.elements["table"])
        val componentModel = requireNotNull(
            extractFlatTableModel(
                containerChildren = table.children,
                containerProps = table.props,
                elements = parsed.elements,
                state = parsed.state,
                compactScreen = true
            )
        )
        assertEquals(FlatTableShape.SCHEDULE_TIMELINE, componentModel.shape)
        assertEquals(FlatTableRenderMode.RESPONSIVE_CARD_ROWS, componentModel.renderMode)
        assertFalse(shouldUseScrollableNativeTableGrid(componentModel.renderMode, hasRows = true))
        assertEquals(
            ResponsiveTableCardTemplate.SCHEDULE,
            responsiveTableCardTemplate(
                shape = componentModel.shape,
                headers = componentModel.headers,
                rows = listOf(listOf("A", "B", "C", "D", "E"))
            )
        )

        val explicitTable = requireNotNull(
            extractDirectTableModel(
                props = mapOf(
                    "columns" to componentModel.headers,
                    "rows" to listOf(listOf("A", "B", "C", "D", "E")),
                    "domain" to "schedule",
                    "preferredPresentation" to "table"
                ),
                state = emptyMap(),
                compactScreen = true
            )
        )
        assertEquals(FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL, explicitTable.renderMode)
        assertTrue(shouldUseScrollableNativeTableGrid(explicitTable.renderMode, hasRows = true))
    }
}
