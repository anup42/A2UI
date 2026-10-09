package com.samsung.genuicraft.sdk.internal.renderer

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.resolveDirectTableColumns
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.resolveDirectTableRows
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.tableRowAccessibilitySummary
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement
import com.samsung.genuicraft.sdk.internal.renderer.CitationText as Text

internal data class TimelineTableEntry(val period: String, val event: String, val citation: String)
internal data class TimelineTableProfile(
    val headers: List<String>, val originalRows: List<List<String>>,
    val entries: List<TimelineTableEntry>, val title: String?,
)

private val timelineBinding = Regex("\\$(?:/|[A-Za-z_])|@source")
private val timelineUrl = Regex("(?i)(?:[a-z][a-z0-9+.-]*://|www\\.|mailto:|tel:|geo:|intent:)")
private val timelineCitation = Regex("[\\s\\p{Z}]*\\[\\d+(?:\\s*[,\\u2013-]\\s*\\d+)*](?:[\\s\\p{Z}]*\\[\\d+(?:\\s*[,\\u2013-]\\s*\\d+)*])*[\\s\\p{Z}]*")
private val timelineFields = setOf("year", "event", "source")

/** A presentation of literal supplied rows, never a parsed or sorted history. */
internal fun timelineTableProfile(
    table: FlatElement?, state: Map<String, Any?>,
    ancestors: List<FlatElement?> = emptyList(), isRepeated: Boolean = false,
): TimelineTableProfile? {
    if (table == null || isRepeated || !table.type.equals("table", true) || table.children.isNotEmpty()) return null
    val props = table.props
    val allowed = setOf("columns", "rows", "statePath", "title", "domain", "preferredPresentation", "primaryColumn")
    if (props.keys.any { it !in allowed } || props["domain"] != "generic" || props["preferredPresentation"] != "cards" ||
        (props["title"] != null && props["title"] !is String) ||
        (props["primaryColumn"] != null && props["primaryColumn"] != "year" && props["primaryColumn"] != "Year")) return null
    if ((listOf(table) + ancestors).any { node ->
            node == null || node.repeat != null || node.visible != null ||
                !node.on.isNullOrEmpty() || !node.watch.isNullOrEmpty() ||
                dynamicTimelineValue(if (node === table) node.props - "statePath" else node.props)
        }) return null
    // A static pointer selects current literal state rows; cell bindings remain ineligible.
    val path = props["statePath"]
    if (path != null && (path !is String || !path.startsWith('/') || path.isBlank() ||
            dynamicTimelineValue(path))) return null
    if (!literalTimelineColumns(props["columns"])) return null
    val rawRows = resolveDirectTableRows(props, state)
    if (rawRows.size !in 2..64) return null
    val columns = resolveDirectTableColumns(props, rawRows)
    val headers = columns.map { it.label }
    val names = headers.map { it.lowercase() }
    if (names != listOf("year", "event") && names != listOf("year", "event", "source")) return null
    val keys = columns.map { it.key }
    val rows = rawRows.map { raw ->
        val values = when (raw) {
            is List<*> -> {
                if (raw.size != columns.size || raw.any { it !is String }) return null
                raw.map { it as String }
            }
            is Map<*, *> -> {
                if (raw.keys.any { it !is String || it !in keys } || raw.values.any { it !is String }) return null
                keys.mapIndexed { index, key ->
                    val value = raw[key]
                    if (value == null && index == 2 && names[index] == "source") "" else value as? String ?: return null
                }
            }
            else -> return null
        }
        if (values.any(::dynamicTimelineValue)) return null
        val period = values[0]
        val event = values[1]
        val citation = values.getOrNull(2).orEmpty()
        if (period.isBlank() || period.length > 80 || event.isBlank() ||
            (citation.isNotBlank() && !timelineCitation.matches(citation))) return null
        values
    }
    return TimelineTableProfile(headers, rows,
        rows.map { TimelineTableEntry(it[0], it[1], it.getOrNull(2).orEmpty()) }, props["title"] as? String)
}

/** Recheck final projection to prevent a stale or changed profile from altering another table. */
internal fun timelineMatchesResolved(profile: TimelineTableProfile, table: FlatDirectTableModel): Boolean =
    table.domain == "generic" && table.preferredPresentation == "cards" &&
        profile.headers == table.columns.map { it.label } &&
        profile.originalRows.map { row -> row.map(String::trim) } == table.rows

private fun literalTimelineColumns(value: Any?): Boolean = when (value) {
    is Map<*, *> -> value.isNotEmpty() && value.all { (key, label) ->
        key is String && label is String && key.lowercase() in timelineFields &&
            key.equals(label, true) && !dynamicTimelineValue(label)
    }
    is List<*> -> value.isNotEmpty() && value.all { column ->
        when (column) {
            is String -> column.lowercase() in timelineFields
            is Map<*, *> -> column.keys.all { it == "key" || it == "label" } &&
                (column["key"] as? String)?.lowercase() in timelineFields &&
                (column["label"] == null || (column["label"] is String &&
                    (column["key"] as String).equals(column["label"] as String, true)))
            else -> false
        }
    }
    else -> false
}

private fun dynamicTimelineValue(value: Any?): Boolean = when (value) {
    is String -> value.contains("{{") || value.contains("}}") || value.contains("\${") ||
        timelineBinding.containsMatchIn(value) || timelineUrl.containsMatchIn(value)
    is Map<*, *> -> value.any { (key, child) ->
        key !is String || key.startsWith("\$") ||
            key in setOf("statePath", "rowsPath", "dataPath", "on", "visible", "repeat", "watch", "condition", "action", "actions") ||
            dynamicTimelineValue(child)
    }
    is List<*> -> value.any(::dynamicTimelineValue)
    null, is Number, is Boolean -> false
    else -> true
}

@Composable
internal fun RenderTimelineTable(profile: TimelineTableProfile, modifier: Modifier = Modifier) {
    val marker = MaterialTheme.colorScheme.primary
    val connector = MaterialTheme.colorScheme.outlineVariant
    Card(
        modifier = modifier.fillMaxWidth().semantics {
            contentDescription = tableAccessibilitySummary(profile.headers, profile.originalRows)
        },
        shape = RoundedCornerShape(16.dp), colors = flatSpecCardColors(),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp), border = flatSpecCardBorder(),
    ) {
        Column(Modifier.fillMaxWidth().padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            profile.title?.takeIf { it.isNotBlank() }?.let { title ->
                Text(parseBoldMarkdown(title), style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                    modifier = Modifier.semantics { heading() })
            }
            Text(profile.headers.joinToString(" \u00b7 "), style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            Column(Modifier.fillMaxWidth()) {
                profile.entries.forEachIndexed { index, entry ->
                    Row(Modifier.fillMaxWidth().semantics(mergeDescendants = true) {
                        contentDescription = tableRowAccessibilitySummary(profile.headers, profile.originalRows[index], index)
                    }.drawBehind {
                        val x = 6.dp.toPx()
                        val y = 14.dp.toPx()
                        val stroke = 2.dp.toPx()
                        if (index > 0) drawLine(connector, Offset(x, 0f), Offset(x, y), stroke)
                        if (index < profile.entries.lastIndex) drawLine(connector, Offset(x, y), Offset(x, size.height), stroke)
                        drawCircle(marker, radius = 4.dp.toPx(), center = Offset(x, y))
                    }.padding(bottom = if (index == profile.entries.lastIndex) 0.dp else 12.dp)) {
                        Spacer(Modifier.width(22.dp))
                        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            // No fixed dimensions or line cap: literal periods wrap at narrow widths and large fonts.
                            Surface(shape = RoundedCornerShape(8.dp), color = MaterialTheme.colorScheme.primaryContainer,
                                contentColor = MaterialTheme.colorScheme.onPrimaryContainer) {
                                Text(entry.period, style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.SemiBold),
                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp))
                            }
                            Text(parseBoldMarkdown(entry.event), style = MaterialTheme.typography.bodyMedium)
                            if (entry.citation.isNotBlank()) {
                                Text(profile.headers[2], style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                                Text(entry.citation, style = MaterialTheme.typography.bodySmall)
                            }
                        }
                    }
                }
            }
        }
    }
}
