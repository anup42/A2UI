package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.extractDirectTableModel
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.entityCardRowContent
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.selectEntityCompactBodyIndexes
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.selectEntityHighlightIndexes
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DeclaredEntityCardPresentationTest {
    private val columns = listOf(
        mapOf("key" to "title", "label" to "Title"),
        mapOf("key" to "platform", "label" to "Platform"),
        mapOf("key" to "releaseDate", "label" to "Streaming release date"),
        mapOf("key" to "description", "label" to "Spoiler-free description"),
    )
    private val films = listOf(
        mapOf("title" to "Bandar", "platform" to "ZEE5", "releaseDate" to "28 August 2026",
            "description" to "A tense drama centered on accusation, accountability, and justice, led by Bobby Deol. [33]",
            "platformReleaseDate" to "28 August 2026"),
        mapOf("title" to "Gandhari", "platform" to "Netflix", "releaseDate" to "3 September 2026",
            "description" to "An emotionally charged action-thriller about a mother’s relentless fight to protect her child. [21]",
            "platformReleaseDate" to "3 September 2026"),
        mapOf("title" to "Before Colours", "platform" to "ZEE5", "releaseDate" to "28 August 2026",
            "description" to "A newly added Hindi film on ZEE5’s latest releases list, available from 28 August 2026. [22]",
            "platformReleaseDate" to "28 August 2026"),
    )
    private val props = mapOf<String, Any?>("columns" to columns, "statePath" to "/films", "domain" to "generic",
        "preferredPresentation" to "cards", "primaryColumn" to "title")
    private val state = mapOf<String, Any?>("films" to films)

    @Test fun `accepted film rows use declared entity cards on compact and wide portrait screens without losing any declared cell`() {
        val expectedRows = films.map { row -> columns.map { row.getValue(it.getValue("key")) } }
        val headers = columns.map { it.getValue("label") }
        listOf(360, 800).forEach { width ->
            val table = extractDirectTableModel(props, state, compactScreen = width < 600)!!
            assertEquals(FlatTableShape.ENTITY_ROW, table.shape)
            assertEquals(FlatTableRenderMode.RESPONSIVE_CARD_ROWS, table.renderMode)
            assertEquals("Title", table.primaryColumn)
            assertEquals(expectedRows, table.rows)
            assertEquals(headers, table.columns.map { it.label })
            assertFalse(shouldUseScrollableNativeTableGrid(table.renderMode, table.rows.isNotEmpty()))
            assertEquals(AdaptiveTablePresentation.ENTITY_CARDS,
                selectAdaptiveTablePresentation(table, width, false, true, true))
            val primary = inferEntityPrimaryColumnIndex(headers, table.primaryColumn)
            val highlights = selectEntityHighlightIndexes(headers, table.rows, primary, table.highlightColumns)
            val compact = selectEntityCompactBodyIndexes(headers, table.rows, primary, highlights)
            table.rows.forEachIndexed { index, row ->
                val content = entityCardRowContent(headers, row, index, primary, highlights, compact)
                assertEquals(films[index].getValue("title"), content.title)
                assertEquals(headers.zip(row), content.representedSourceCells.map { it.label to it.value })
            }
        }
        assertEquals(films, state["films"])
        assertEquals(columns, props["columns"])
    }

    @Test fun `declared key and label aliases map to the same actual primary label and row value`() {
        listOf("releaseDate", "Streaming release date").forEach { alias ->
            val table = extractDirectTableModel(props + ("primaryColumn" to alias), state, true)!!
            assertEquals(FlatTableShape.ENTITY_ROW, table.shape)
            assertEquals("Streaming release date", table.primaryColumn)
            val headers = table.columns.map { it.label }
            val primary = inferEntityPrimaryColumnIndex(headers, table.primaryColumn)
            assertEquals(2, primary)
            table.rows.forEachIndexed { index, row ->
                val content = entityCardRowContent(headers, row, index, primary, emptyList())
                assertEquals(films[index].getValue("releaseDate"), content.title)
                assertEquals(headers.zip(row), content.representedSourceCells.map { it.label to it.value })
            }
        }
    }

    @Test fun `unknown undeclared or blank primary values and no row data remain generic`() {
        listOf("unknown", "", " ").forEach { primary ->
            assertEquals(FlatTableShape.GENERIC_GRID,
                extractDirectTableModel(props + ("primaryColumn" to primary), state, true)!!.shape)
        }
        assertEquals(FlatTableShape.GENERIC_GRID, extractDirectTableModel(props - "primaryColumn", state, true)!!.shape)
        listOf("", " ").forEach { blank ->
            val rows = films.mapIndexed { index, row -> if (index == 1) row + ("title" to blank) else row }
            assertEquals(FlatTableShape.GENERIC_GRID,
                extractDirectTableModel(props, mapOf("films" to rows), true)!!.shape)
        }
        assertEquals(FlatTableShape.GENERIC_GRID,
            extractDirectTableModel(props, mapOf("films" to emptyList<Map<String, String>>()), true)!!.shape)
    }

    @Test fun `explicit grids feature matrices and feature identities keep their existing shape and presentation`() {
        listOf(true, false).forEach { compact ->
            val grid = extractDirectTableModel(props + ("preferredPresentation" to "table"), state, compact)!!
            assertEquals(FlatTableShape.GENERIC_GRID, grid.shape)
            assertEquals("title", grid.primaryColumn)
            assertEquals(FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL, grid.renderMode)
            assertTrue(shouldUseScrollableNativeTableGrid(grid.renderMode, true))
        }
        val matrix = extractDirectTableModel(mapOf("columns" to listOf("Feature", "Phone A", "Phone B"),
            "rows" to listOf(listOf("Display", "AMOLED", "LCD"), listOf("Battery", "5,000mAh", "4,500mAh")),
            "domain" to "comparison", "preferredPresentation" to "cards", "primaryColumn" to "Feature"), emptyMap(), true)!!
        assertEquals(FlatTableShape.FEATURE_MATRIX, matrix.shape)
        assertEquals(AdaptiveTablePresentation.FEATURE_CARDS,
            selectAdaptiveTablePresentation(matrix, 360, false, true, true))
        val featureColumns = columns.mapIndexed { index, column -> if (index == 1) column + ("label" to "Feature") else column }
        val featureIdentity = extractDirectTableModel(props + mapOf("columns" to featureColumns, "primaryColumn" to "platform"), state, true)!!
        assertEquals(FlatTableShape.GENERIC_GRID, featureIdentity.shape)
        assertEquals("platform", featureIdentity.primaryColumn)
    }
}
