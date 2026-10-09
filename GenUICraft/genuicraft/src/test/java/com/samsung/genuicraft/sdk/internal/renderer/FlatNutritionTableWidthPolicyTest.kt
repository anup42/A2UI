package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.extractDirectTableModel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FlatNutritionTableWidthPolicyTest {
    private val headers = listOf("Food", "Typical serving", "Approx. protein")
    private val rows = listOf(
        listOf("Soy chunks, dry", "30 g", "15–16 g [3]"),
        listOf("Paneer", "100 g", "18 g [1][17]"),
        listOf("Tofu", "100 g", "8–10 g"),
        listOf("Cooked rajma", "1 cup", "12–15 g[6]"),
        listOf("Cooked chole", "1 cup", "14–15 g"),
        listOf("Cooked moong dal", "1 cup", "8–11 g[4]"),
        listOf("Roasted chana", "30 g", "6–7 g [2][16]"),
        listOf("Dahi/curd", "200 g", "7–8 g[18]"),
    )

    @Test fun `all eight food rows fit compact host while preserving table mode and every exact range unit and citation`() {
        val snapshot = rows.map { it.toList() }
        listOf(320, 360, 380).forEach { width ->
            val widths = nutritionTableColumnWidthsDp(headers, rows, width, 1f)!!
            assertEquals(3, widths.size)
            assertEquals(width, widths.sum())
            assertTrue(widths.all { it > 0 })
            assertTrue(widths[1] < 120)
            assertTrue(nativeTableStickyFirstColumn(headers, horizontalScrollEnabled = true))
        }
        val columns = listOf(
            mapOf("key" to "food", "label" to headers[0]),
            mapOf("key" to "serving", "label" to headers[1]),
            mapOf("key" to "protein", "label" to headers[2]),
        )
        val state = mapOf("protein_data" to rows.map { row ->
            mapOf("food" to row[0], "serving" to row[1], "protein" to row[2])
        })
        val table = extractDirectTableModel(mapOf("columns" to columns, "statePath" to "/protein_data",
            "domain" to "generic", "preferredPresentation" to "table", "numericColumns" to listOf("protein")), state, true)!!
        assertEquals(FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL, table.renderMode)
        assertTrue(shouldUseScrollableNativeTableGrid(table.renderMode, true))
        assertEquals(headers, table.columns.map { it.label })
        assertEquals(snapshot, table.rows)
        assertEquals(24, table.rows.sumOf { it.size })
        assertEquals(setOf("protein"), table.numericColumns)
        assertEquals(snapshot, rows)
        assertEquals(setOf(2), tableGridNumericColumns(table.rows, setOf(2)))
    }

    @Test fun `large font fits when possible and narrow overflow preserves a readable value viewport beside sticky Food`() {
        val comfortable = nutritionTableColumnWidthsDp(headers, rows, 380, 1.3f)!!
        assertEquals(380, comfortable.sum())
        val compact = nutritionTableColumnWidthsDp(headers, rows, 300, 1.3f)!!
        assertTrue(compact.sum() > 300)
        assertTrue(compact[0] <= 102)
        assertTrue(compact.drop(1).all { it <= 300 - compact[0] })
        val large = nutritionTableColumnWidthsDp(headers, rows, 240, 2f)!!
        assertTrue(large.sum() > 240)
        assertTrue(large.drop(1).all { it <= 240 - large[0] })
        assertNull(nutritionTableColumnWidthsDp(headers, rows, 800, 1f))
        assertNull(nutritionTableColumnWidthsDp(headers, rows, 0, 1f))
        assertNull(nutritionTableColumnWidthsDp(headers, rows, 360, Float.NaN))
    }

    @Test fun `other schemas qualifiers actions bindings missing cells and long values retain their existing widths`() {
        assertNull(nutritionTableColumnWidthsDp(listOf("Feature", "Serving", "Protein"), rows, 360, 1f))
        assertNull(nutritionTableColumnWidthsDp(listOf("Food", "Typical serving", "Calories"), rows, 360, 1f))
        assertNull(nutritionTableColumnWidthsDp(headers + "Source link", rows.map { it + "https://example.com" }, 360, 1f))
        assertNull(nutritionTableColumnWidthsDp(headers, emptyList(), 360, 1f))
        assertNull(nutritionTableColumnWidthsDp(headers, listOf(rows.first().dropLast(1)), 360, 1f))
        assertNull(nutritionTableColumnWidthsDp(headers, listOf(rows.first() + "Extra"), 360, 1f))
        listOf(
            listOf("", "30 g", "15–16 g [3]"),
            listOf("Soy chunks, dry", "1 cup cooked; portion varies", "15–16 g [3]"),
            listOf("Soy chunks, dry", "30 g", "About 15–16 g; varies by brand [3]"),
            listOf("Soy chunks, dry", "30 g", "Unavailable"),
            listOf("Soy chunks, dry", "30 g", "{{/protein}}"),
            listOf("www.example.com", "30 g", "15–16 g [3]"),
        ).forEach { row -> assertNull(nutritionTableColumnWidthsDp(headers, listOf(row), 360, 1f)) }
        assertTrue(!nativeTableStickyFirstColumn(headers, horizontalScrollEnabled = false))
    }
}
