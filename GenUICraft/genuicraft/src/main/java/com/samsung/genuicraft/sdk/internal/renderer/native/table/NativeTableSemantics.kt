package com.samsung.genuicraft.sdk.internal.renderer.native.table

import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.samsung.genuicraft.sdk.internal.renderer.native.TableCell
import com.samsung.genuicraft.sdk.internal.renderer.native.TableSpec
import java.util.Locale

internal object NativeTableSemantics {
    private val TABLE_PLACEHOLDER_CELL_REGEX = Regex("""^[:\-\u2013\u2014]+$""")

    fun tableBaseCellWidth(columnCount: Int): Dp {
        return when {
            columnCount <= 2 -> 220.dp
            columnCount == 3 -> 196.dp
            columnCount == 4 -> 172.dp
            else -> 156.dp
        }
    }

    fun tableColumnWidths(spec: TableSpec, columnCount: Int, weightedLayout: Boolean): List<Dp> {
        val baseWidth = tableBaseCellWidth(columnCount)
        if (!weightedLayout) {
            return List(columnCount) { baseWidth }
        }

        val columnWeights = MutableList(columnCount) { 1f }
        fun accumulate(cells: List<TableCell>) {
            for (column in 0 until columnCount) {
                val weight = cells.getOrNull(column)?.weight?.coerceIn(0.6f, 3.5f) ?: 1f
                columnWeights[column] = maxOf(columnWeights[column], weight)
            }
        }

        spec.header?.let(::accumulate)
        spec.rows.forEach(::accumulate)
        return columnWeights.map { baseWidth * it.coerceIn(0.6f, 3.5f) }
    }

    fun isPlaceholderTableStringRow(row: List<String>): Boolean {
        if (row.isEmpty()) return true
        return row.all(::isPlaceholderTableCellValue)
    }

    fun isPlaceholderTableCellRow(row: List<TableCell>): Boolean =
        isPlaceholderTableStringRow(row.map { it.text })

    fun sanitizeTableCellDisplayValue(value: String): String {
        val normalized = value.trim()
        return if (isPlaceholderTableCellValue(normalized)) "" else normalized
    }

    fun isPlaceholderTableCellValue(value: String): Boolean {
        val normalized = value.trim()
        if (normalized.isEmpty()) {
            return true
        }
        val compact = normalized.replace(Regex("""\s+"""), "")
        if (TABLE_PLACEHOLDER_CELL_REGEX.matches(compact)) {
            return true
        }
        return when (normalized.lowercase(Locale.US)) {
            "na", "n/a", "null", "none", "not available" -> true
            else -> false
        }
    }
}
