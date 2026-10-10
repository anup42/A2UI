package com.samsung.genuicraft.sdk.internal.renderer

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import java.util.Locale
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.isActionLabelColumn
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.isUrlColumnLabel
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.tableRowAccessibilitySummary
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
    val roundedPanel = RoundedCornerShape(24.dp)
    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(roundedPanel)
            .background(Brush.linearGradient(listOf(Color(0xFF070712), Color(0xFF172044), Color(0xFF082F35))))
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(
                modifier = Modifier
                    .clip(RoundedCornerShape(16.dp))
                    .background(Brush.linearGradient(listOf(Color(0xFF342A83), Color(0xFF075960))))
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(1.dp)
            ) {
                Text("CHART", style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold), color = Color.White.copy(alpha = 0.82f))
                Text(rows.size.toString(), style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Bold), color = Color.White)
            }
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                Text(
                    parseBoldMarkdown(title?.takeIf(String::isNotBlank) ?: "Music chart"),
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                    color = Color.White
                )
                Text("${rows.size} ${if (rows.size == 1) "entry" else "entries"}", style = MaterialTheme.typography.bodySmall, color = Color.White.copy(alpha = 0.74f))
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.padding(horizontal = 10.dp)) {
            Text(headers[profile.rank], style = MaterialTheme.typography.labelSmall, color = Color.White.copy(alpha = 0.72f))
            Text(headers[profile.title], style = MaterialTheme.typography.labelSmall, color = Color.White.copy(alpha = 0.72f))
        }
        rows.forEachIndexed { index, row ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(14.dp))
                    .background(Color.White.copy(alpha = if (index == 0) 0.15f else 0.09f))
                    .semantics(mergeDescendants = true) {
                        contentDescription = tableRowAccessibilitySummary(headers, row, index)
                    }
                    .padding(horizontal = 11.dp, vertical = 9.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.Top
            ) {
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(10.dp))
                        .background(Color.White.copy(alpha = 0.15f))
                        .widthIn(min = 30.dp)
                        .padding(horizontal = 8.dp, vertical = 6.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(parseBoldMarkdown(row[profile.rank]), style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold), color = Color.White)
                }
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(parseBoldMarkdown(row[profile.title]), style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold), color = Color.White)
                    profile.detailIndexes(headers).filterNot { it == profile.date && commonDate != null }.forEach { cell ->
                        Text(parseBoldMarkdown("**${headers[cell]}:** ${row[cell]}"), style = MaterialTheme.typography.bodySmall, color = Color.White.copy(alpha = 0.78f))
                    }
                }
            }
        }
        if (commonDate != null) {
            Text(
                parseBoldMarkdown("**${headers[profile.date]}:** $commonDate"),
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(Color.White.copy(alpha = 0.09f))
                    .padding(horizontal = 11.dp, vertical = 9.dp),
                style = MaterialTheme.typography.bodySmall,
                color = Color.White.copy(alpha = 0.84f)
            )
        }
    }
    return true
}
