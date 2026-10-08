package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.extractDirectTableModel
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.entityCardRowContent
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.selectEntityCompactBodyIndexes
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.selectEntityHighlightIndexes
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FlatDocumentChecklistPolicyTest {
    private val headers = listOf("Document", "Verification Details", "Source")
    private val rows = listOf(
        listOf("RERA registration certificate", "Confirms the project is registered under Karnataka RERA, which is required for qualifying apartment projects; verify it on the Karnataka RERA portal using the project name or registration number.", "[1][3][4][9]"),
        listOf("Sanctioned building plan", "Confirms the building layout, floors, and units were approved by the competent authority; verify it with the BBMP, BDA, or other local planning authority and cross-check the same plan details against RERA records.", "[13]"),
        listOf("Title documents / title chain", "Establishes who owns the land and whether the seller has a valid right to transfer it; verify the mother deed, prior sale deeds, partition deeds, and related title records at the Sub-Registrar’s office or through Karnataka’s registration record system, and have an independent lawyer review the chain.", "[2][5][10][14][15]"),
        listOf("Occupancy Certificate (OC)", "Confirms the building has been certified fit for occupation and generally matches approved norms; verify it with the BBMP or the issuing municipal/planning authority before taking possession.", "[7]"),
        listOf("Encumbrance Certificate (EC)", "Shows registered transactions and helps identify mortgages, charges, attachments, or other encumbrances on the property; verify it at the Sub-Registrar’s office or through Karnataka’s online registration services, and ensure the period covered is sufficiently long.", "")
    )
    private val columns = listOf(
        mapOf("key" to "item", "label" to headers[0]),
        mapOf("key" to "details", "label" to headers[1]),
        mapOf("key" to "source", "label" to headers[2]),
    )
    private val props = mapOf<String, Any?>("columns" to columns, "statePath" to "/checklist_items",
        "domain" to "generic", "preferredPresentation" to "cards")
    private val state = mapOf<String, Any?>("checklist_items" to rows.map { row ->
        mapOf("item" to row[0], "details" to row[1], "source" to row[2])
    })

    @Test fun `actual checklist uses cards on compact and wide screens preserving full prose citations and empty source`() {
        assertEquals(DocumentChecklistProfile(0, 1), documentChecklistProfile(headers, rows))
        listOf(360, 800).forEach { width ->
            val table = extractDirectTableModel(props, state, width < 600)!!
            assertEquals(FlatTableShape.ENTITY_ROW, table.shape)
            assertEquals(FlatTableRenderMode.RESPONSIVE_CARD_ROWS, table.renderMode)
            assertEquals("Document", table.primaryColumn)
            assertEquals(rows, table.rows)
            assertEquals(headers, table.columns.map { it.label })
            assertFalse(shouldUseScrollableNativeTableGrid(table.renderMode, table.rows.isNotEmpty()))
            assertEquals(AdaptiveTablePresentation.ENTITY_CARDS,
                selectAdaptiveTablePresentation(table, width, false, true, true))
            val primary = inferEntityPrimaryColumnIndex(headers, table.primaryColumn)
            val highlights = selectEntityHighlightIndexes(headers, table.rows, primary, table.highlightColumns)
            val compact = selectEntityCompactBodyIndexes(headers, table.rows, primary, highlights)
            assertFalse(1 in compact)
            rows.forEachIndexed { index, row ->
                val content = entityCardRowContent(headers, row, index, primary, highlights, compact)
                assertEquals(row[0], content.title)
                assertEquals(row[1], content.detailBodyCells.single { it.index == 1 }.value)
                assertEquals(headers.zip(row).filter { it.second.isNotBlank() },
                    content.representedSourceCells.map { it.label to it.value })
                assertTrue(content.actions.isEmpty())
                assertTrue(content.metadataCells.isEmpty())
            }
            assertEquals("", table.rows.last()[2])
        }
    }

    @Test fun `permuted document column and additional labeled fields retain source order and exact values`() {
        val labels = listOf("Source", "Verification Details", "Document name", "Caution")
        val values = rows.map { listOf(it[2], it[1], it[0], "Verify current records; no conclusion is inferred. [12]") }
        assertEquals(DocumentChecklistProfile(2, 1), documentChecklistProfile(labels, values))
        val table = extractDirectTableModel(mapOf("columns" to labels, "rows" to values,
            "domain" to "generic", "preferredPresentation" to "cards"), emptyMap(), true)!!
        assertEquals("Document name", table.primaryColumn)
        assertEquals(values, table.rows)
        table.rows.forEachIndexed { index, row ->
            val content = entityCardRowContent(labels, row, index, 2, emptyList())
            assertEquals(labels.zip(row).filter { it.second.isNotBlank() },
                content.representedSourceCells.map { it.label to it.value })
        }
    }

    @Test fun `weak duplicate ragged blank dynamic and action-bearing schemas decline checklist inference`() {
        assertNull(documentChecklistProfile(listOf("Item", "Verification Details", "Source"), rows))
        assertNull(documentChecklistProfile(listOf("Document", "Details", "Source"), rows))
        assertNull(documentChecklistProfile(headers + "Document", rows.map { it + "Other" }))
        assertNull(documentChecklistProfile(headers, emptyList()))
        assertNull(documentChecklistProfile(headers, listOf(rows.first().dropLast(1))))
        assertNull(documentChecklistProfile(headers, listOf(rows.first() + "Unheaded value")))
        listOf(0, 1).forEach { index ->
            assertNull(documentChecklistProfile(headers, listOf(rows.first().mapIndexed { i, value -> if (i == index) " " else value })))
        }
        listOf("Source link", "CTA", "Href", "Phone", "Image", "Feature", "Step").forEach { header ->
            assertNull(documentChecklistProfile(headers + header, rows.map { it + "Value" }))
        }
        listOf("https://example.com", "www.example.com", "example.com", "tel:+911234567890", "{{/live}}",
            "Open https://example.com to verify").forEach { value ->
            assertNull(documentChecklistProfile(headers, listOf(listOf(rows.first()[0], rows.first()[1], value))))
        }
    }

    @Test fun `explicit table conflicting identity and recognized feature matrix keep existing interpretation`() {
        val grid = extractDirectTableModel(props + ("preferredPresentation" to "table"), state, true)!!
        assertEquals(FlatTableShape.GENERIC_GRID, grid.shape)
        assertEquals(FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL, grid.renderMode)
        assertTrue(shouldUseScrollableNativeTableGrid(grid.renderMode, true))
        assertEquals(FlatTableShape.GENERIC_GRID,
            extractDirectTableModel(props + ("primaryColumn" to "unknown"), state, true)!!.shape)
        assertEquals(FlatTableShape.GENERIC_GRID,
            extractDirectTableModel(props - "preferredPresentation", state, true)!!.shape)
        val matrix = extractDirectTableModel(mapOf("columns" to listOf("Feature", "Document", "Verification Details"),
            "rows" to listOf(listOf("Format", "PDF", "Text"), listOf("Count", "2", "3")),
            "domain" to "comparison", "preferredPresentation" to "cards"), emptyMap(), true)!!
        assertEquals(FlatTableShape.FEATURE_MATRIX, matrix.shape)
    }
}
