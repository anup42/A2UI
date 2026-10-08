package com.samsung.genuicraft.sdk.internal.renderer

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import java.util.Locale
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.isActionLabelColumn
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.isUrlColumnLabel
import com.samsung.genuicraft.sdk.internal.renderer.CitationText as Text

internal data class ChartRankingProfile(val rank: Int, val title: Int, val date: Int) {
    fun detailIndexes(headers: List<String>): List<Int> = headers.indices.filter { it != rank && it != title }
}

internal fun chartRankingProfile(headers: List<String>, rows: List<List<String>>): ChartRankingProfile? {
    val labels = headers.map { it.lowercase(Locale.ROOT).replace('_', ' ').trim() }
    val rank = labels.indexOfFirst { it in setOf("rank", "position", "chart rank") }
    val title = labels.indexOfFirst { it in setOf("song title", "track title", "title", "song", "track") }
    val date = labels.indexOfFirst { it.contains("chart") && (it.contains("date") || it.contains("week")) }
    if (rank < 0 || title < 0 || date < 0 || setOf(rank, title, date).size != 3 || rows.isEmpty()) return null
    if (rows.any { it.size != headers.size || it[rank].isBlank() || it[title].isBlank() || it.any(::isLikelyHttpUrl) }) return null
    if (headers.any { isUrlColumnLabel(it) || isActionLabelColumn(it) } ||
        labels.any { Regex("\\b(image|photo|media|artwork|url)\\b").containsMatchIn(it) }) return null
    return ChartRankingProfile(rank, title, date)
}

@Composable
internal fun renderChartRankingIfPossible(headers: List<String>, rows: List<List<String>>, modifier: Modifier = Modifier, title: String? = null): Boolean {
    val profile = chartRankingProfile(headers, rows) ?: return false
    val commonDate = rows.map { it[profile.date] }.distinct().singleOrNull()?.takeIf(String::isNotBlank)
    Card(modifier.fillMaxWidth(), shape = RoundedCornerShape(16.dp), colors = flatSpecCardColors()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            title?.takeIf(String::isNotBlank)?.let { Text(parseBoldMarkdown(it), style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)) }
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(headers[profile.rank], style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(headers[profile.title], style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            rows.forEachIndexed { index, row ->
                if (index > 0) HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.Top) {
                    Surface(shape = RoundedCornerShape(10.dp), color = MaterialTheme.colorScheme.primaryContainer) {
                        Text(parseBoldMarkdown(row[profile.rank]), Modifier.widthIn(min = 28.dp).padding(horizontal = 8.dp, vertical = 6.dp), style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold), color = MaterialTheme.colorScheme.onPrimaryContainer)
                    }
                    Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                        Text(parseBoldMarkdown(row[profile.title]), style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold))
                        profile.detailIndexes(headers).filterNot { it == profile.date && commonDate != null }.forEach { cell ->
                            Text(parseBoldMarkdown("**${headers[cell]}:** ${row[cell]}"), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }
            if (commonDate != null) {
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                Text(parseBoldMarkdown("**${headers[profile.date]}:** $commonDate"), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
    return true
}
