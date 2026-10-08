package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.isActionLabelColumn
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.isUrlColumnLabel
import java.util.Locale

internal data class DocumentChecklistProfile(val document: Int, val verification: Int)

/** Classifies literal informational document rows; callers still honor the authored presentation. */
internal fun documentChecklistProfile(headers: List<String>, rows: List<List<String>>): DocumentChecklistProfile? {
    if (headers.size < 2 || rows.isEmpty()) return null
    val labels = headers.map { it.lowercase(Locale.ROOT).replace('_', ' ').trim().replace(Regex("\\s+"), " ") }
    if (labels.any(String::isBlank) || labels.distinct().size != labels.size) return null
    val document = labels.indexOfFirst { it in setOf("document", "document name") }
    val verification = labels.indexOf("verification details")
    if (document < 0 || verification < 0) return null
    if (labels.any { isUrlColumnLabel(it) || isActionLabelColumn(it) ||
            it in setOf("feature", "factor", "step", "stage", "criterion", "criteria") ||
            Regex("\\b(url|photo|image|media|action|website|phone|booking|reservation)\\b").containsMatchIn(it) }) return null
    if (rows.any { row -> row.size != headers.size || row[document].isBlank() || row[verification].isBlank() ||
            row.any { it.contains("{{") || isLikelyHttpUrl(it) || Regex("(?i)\\bhttps?://").containsMatchIn(it) } }) return null
    return DocumentChecklistProfile(document, verification)
}
