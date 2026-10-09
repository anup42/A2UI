package com.samsung.genuicraft.sdk.internal.renderer

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.tableRowAccessibilitySummary
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement
import com.samsung.genuicraft.sdk.internal.renderer.CitationText as Text

internal data class SparseProseRow(val partLabel: String, val body: String, val citation: String)
internal data class SparseProseTableProfile(
    val headers: List<String>, val originalRows: List<List<String>>,
    val paragraphs: List<SparseProseRow>, val title: String?,
)

private val sparsePart = Regex("part([1-9][0-9]*)")
private val sparseCitation = Regex("[\\s\\p{Z}]*\\[\\d+(?:\\s*[,\\u2013-]\\s*\\d+)*](?:[\\s\\p{Z}]*\\[\\d+(?:\\s*[,\\u2013-]\\s*\\d+)*])*[\\s\\p{Z}]*")

/** Only passive literal generic/cards partN/source tables with one long paragraph per row. */
internal fun sparseProseTableProfile(
    table: FlatElement?, ancestors: List<FlatElement?> = emptyList(), isRepeated: Boolean = false,
): SparseProseTableProfile? {
    val rows = passiveDirectLiteralTableRows(table, ancestors, isRepeated) ?: return null
    val props = table?.props ?: return null
    if (props["domain"] != "generic" || props["preferredPresentation"] != "cards" || rows.size < 2 ||
        (props["title"] != null && props["title"] !is String) || props["subtitle"] != null || props["presentation"] != null) return null
    val headers = (props["columns"] as List<*>).map { it as String }
    if (headers.count { it == "source" } != 1) return null
    val parts = headers.withIndex().filter { it.value != "source" }
    if (parts.size !in 2..8 || parts.map { sparsePart.matchEntire(it.value)?.groupValues?.get(1)?.toIntOrNull() } !=
        (1..parts.size).toList()) return null
    val sourceIndex = headers.indexOf("source")
    val paragraphs = rows.map { row ->
        val occupied = parts.filter { row[it.index].isNotEmpty() }
        if (occupied.size != 1) return null
        val part = occupied.single()
        val body = row[part.index]
        val citation = row[sourceIndex]
        if (body.isBlank() || body.length < 80 || (citation.isNotEmpty() && !sparseCitation.matches(citation))) return null
        SparseProseRow(part.value, body, citation)
    }
    if (parts.any { part -> paragraphs.none { it.partLabel == part.value } }) return null
    return SparseProseTableProfile(headers, rows, paragraphs, props["title"] as? String)
}

@Composable
internal fun RenderSparseProseTable(profile: SparseProseTableProfile, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier.fillMaxWidth().semantics {
            contentDescription = tableAccessibilitySummary(profile.headers, profile.originalRows)
        },
        shape = RoundedCornerShape(16.dp), colors = flatSpecCardColors(),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp), border = flatSpecCardBorder(),
    ) {
        Column(Modifier.fillMaxWidth().padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            profile.title?.takeIf { it.isNotBlank() }?.let { title ->
                Text(parseBoldMarkdown(title), style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                    modifier = Modifier.semantics { heading() })
            }
            // These are the authored column labels, including an empty source column's identity.
            Text(profile.headers.joinToString(" \u00b7 "), style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            profile.paragraphs.forEachIndexed { index, paragraph ->
                if (index > 0) HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.45f))
                Column(Modifier.fillMaxWidth().semantics(mergeDescendants = true) {
                    contentDescription = tableRowAccessibilitySummary(profile.headers, profile.originalRows[index], index)
                }, verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(paragraph.partLabel, style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Text(parseBoldMarkdown(paragraph.body), style = MaterialTheme.typography.bodyMedium)
                    if (paragraph.citation.isNotEmpty()) {
                        Text("source", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text(paragraph.citation, style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
    }
}
