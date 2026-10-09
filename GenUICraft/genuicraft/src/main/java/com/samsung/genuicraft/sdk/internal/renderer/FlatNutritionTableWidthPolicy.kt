package com.samsung.genuicraft.sdk.internal.renderer

import java.util.Locale
import kotlin.math.ceil

/** Widths only: retain every authored food, serving, nutrient range, unit, and citation. */
internal fun nutritionTableColumnWidthsDp(
    headers: List<String>,
    rows: List<List<String>>,
    availableWidthDp: Int,
    fontScale: Float,
): List<Int>? {
    if (headers.size != 3 || rows.isEmpty() || availableWidthDp !in 1..599 ||
        !fontScale.isFinite() || fontScale !in 0.5f..3f) return null
    val labels = headers.map { it.lowercase(Locale.ROOT).replace(Regex("[^a-z0-9]+"), " ").trim() }
    if (labels[0] != "food" || labels[1] !in setOf("serving", "typical serving", "serving size") ||
        labels[2] !in setOf("protein", "approx protein", "approximate protein", "protein g", "approx protein g")) return null
    val amount = "\\d+(?:\\.\\d+)?(?:\\s*[–—-]\\s*\\d+(?:\\.\\d+)?)?"
    val serving = Regex("^$amount\\s*(?:g|kg|ml|l|cups?|pieces?|tbsp|tsp)$", RegexOption.IGNORE_CASE)
    val protein = Regex("^$amount\\s*g(?:\\s*\\[\\d+(?:\\s*[,–-]\\s*\\d+)*])*\\s*$", RegexOption.IGNORE_CASE)
    if (rows.any { row -> row.size != 3 || row[0].isBlank() ||
        row.any { it.contains("{{") || isLikelyHttpUrl(it) } ||
        row[1].trim().length > 16 || !serving.matches(row[1].trim()) ||
        row[2].trim().length > 24 || !protein.matches(row[2].trim()) }) return null

    // Serving headers may wrap; short scalar values need less space than a prose column.
    val minimums = listOf(104, 72, 112).map { ceil(it * fontScale.toDouble()).toInt() }
    if (minimums.sum() > availableWidthDp) {
        return tableViewportColumnWidthsDp(minimums, availableWidthDp, stickyFirstColumn = true)
    }
    val extra = availableWidthDp - minimums.sum()
    val foodExtra = (extra.toLong() * 3 / 5).toInt()
    return listOf(minimums[0] + foodExtra, minimums[1], minimums[2] + extra - foodExtra)
}
