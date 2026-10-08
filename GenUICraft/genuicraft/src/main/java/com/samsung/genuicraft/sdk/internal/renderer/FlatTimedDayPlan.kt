package com.samsung.genuicraft.sdk.internal.renderer

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.isActionLabelColumn
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.isUrlColumnLabel
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.tableRowAccessibilitySummary
import com.samsung.genuicraft.sdk.internal.renderer.CitationText as Text
import java.util.Locale

internal data class TimedDayPlanProfile(val day: Int, val activities: List<Int>)

internal data class TimedDayPlanRowContent(
    val dayCell: ResponsiveTableCardCell,
    val bodyCells: List<ResponsiveTableCardCell>,
) {
    val representedSourceCells: List<ResponsiveTableCardCell>
        get() = (listOf(dayCell) + bodyCells).sortedBy { it.index }
}

private val timedActivityDuration = Regex("\\b\\d+(?:\\.\\d+)?\\s*(?:mins?|minutes?|hrs?|hours?|secs?|seconds?)\\b", RegexOption.IGNORE_CASE)

internal fun timedDayPlanProfile(headers: List<String>): TimedDayPlanProfile? {
    if (headers.any(String::isBlank)) return null
    val normalized = headers.map { it.lowercase(Locale.ROOT).trim() }
    if (normalized.distinct().size != normalized.size || headers.any { isUrlColumnLabel(it) || isActionLabelColumn(it) } ||
        normalized.any { Regex("\\b(photo|image|media|artwork|phone|booking|reservation)\\b").containsMatchIn(it) }) return null
    val day = normalized.indexOfFirst { it == "day" }
    if (day < 0) return null
    val activities = headers.indices.filter { index ->
        index != day && timedActivityDuration.containsMatchIn(headers[index]) &&
            headers[index].replace(timedActivityDuration, "").any(Char::isLetter)
    }
    if (activities.size < 2) return null
    return TimedDayPlanProfile(day, activities)
}

internal fun timedDayPlanRowContent(headers: List<String>, row: List<String>): TimedDayPlanRowContent? {
    val profile = timedDayPlanProfile(headers) ?: return null
    if (row.size != headers.size || row[profile.day].isBlank() || profile.activities.any { row[it].isBlank() } ||
        row.any { it.contains("{{") || isLikelyHttpUrl(it) || Regex("(?i)\\bhttps?://").containsMatchIn(it) }) return null
    val cells = headers.indices.map { index -> ResponsiveTableCardCell(index, headers[index], row[index]) }
    return TimedDayPlanRowContent(cells[profile.day], cells.filterNot { it.index == profile.day })
}

@Composable
internal fun renderTimedDayPlanRowIfPossible(headers: List<String>, row: List<String>): Boolean {
    val content = timedDayPlanRowContent(headers, row) ?: return false
    Card(
        modifier = Modifier.fillMaxWidth().semantics(mergeDescendants = true) {
            contentDescription = tableRowAccessibilitySummary(headers, row)
        },
        shape = RoundedCornerShape(16.dp),
        colors = flatSpecCardColors(),
    ) {
        Column(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = parseBoldMarkdown("${content.dayCell.label}: ${content.dayCell.value}"),
                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.semantics { heading() },
            )
            content.bodyCells.forEach { cell ->
                Column(modifier = Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                    Text(
                        text = parseBoldMarkdown(cell.label),
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Text(
                        text = parseBoldMarkdown(cell.value),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                }
            }
        }
    }
    return true
}
