package com.samsung.genuicraft.sdk.internal.renderer

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.samsung.genuicraft.sdk.internal.renderer.CitationText as Text
import java.util.Locale

internal data class DepositRateSummaryProfile(val bank: Int, val rate: Int) {
    fun detailIndexes(headers: List<String>): List<Int> = headers.indices.filter { it != bank && it != rate }
}

internal fun depositRateSummaryProfile(headers: List<String>, rows: List<List<String>>): DepositRateSummaryProfile? {
    if (rows.isEmpty()) return null
    val labels = headers.map { it.lowercase(Locale.ROOT).replace('_', ' ').trim() }
    val bank = labels.indexOfFirst { it in setOf("bank", "bank name") }
    val rate = labels.indexOfFirst { Regex("\\b(fd|fixed deposit)\\b").containsMatchIn(it) && Regex("\\brates?\\b").containsMatchIn(it) && !Regex("\\b(date|effective|updated)\\b").containsMatchIn(it) }
    if (bank < 0 || rate < 0 || rows.any { it.size > headers.size || it.getOrNull(bank).isNullOrBlank() }) return null
    if (labels.any { it in setOf("feature", "factor", "step", "stage") ||
            Regex("\\b(url|photo|image|media|action|website|phone|booking|reservation)\\b").containsMatchIn(it) }) return null
    if (rows.any { row -> row.any { Regex("(?i)\\bhttps?://").containsMatchIn(it) } }) return null
    return DepositRateSummaryProfile(bank, rate)
}

@Composable
internal fun renderDepositRateSummaryIfPossible(
    headers: List<String>, rows: List<List<String>>, modifier: Modifier = Modifier, title: String? = null,
): Boolean {
    val profile = depositRateSummaryProfile(headers, rows) ?: return false
    Column(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        title?.takeIf(String::isNotBlank)?.let {
            Text(parseBoldMarkdown(it), style = MaterialTheme.typography.titleMedium, modifier = Modifier.semantics { heading() })
        }
        rows.forEach { row ->
            Card(Modifier.fillMaxWidth(), shape = RoundedCornerShape(16.dp), colors = flatSpecCardColors()) {
                Column(Modifier.fillMaxWidth().padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    BoxWithConstraints(Modifier.fillMaxWidth()) {
                        val stacked = maxWidth < 300.dp * LocalDensity.current.fontScale
                        val bankHeading: @Composable () -> Unit = {
                            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                                Text(headers[profile.bank], style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                Text(parseBoldMarkdown(row[profile.bank]), style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold), modifier = Modifier.semantics { heading() })
                            }
                        }
                        val rateMetric: @Composable () -> Unit = {
                            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                                Text(parseBoldMarkdown(row.getOrNull(profile.rate).orEmpty()), style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Bold), color = MaterialTheme.colorScheme.primary)
                                Text(headers[profile.rate], style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }
                        if (stacked) {
                            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) { bankHeading(); rateMetric() }
                        } else {
                            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
                                Column(Modifier.weight(1f)) { bankHeading() }
                                Column(Modifier.weight(1f)) { rateMetric() }
                            }
                        }
                    }
                    profile.detailIndexes(headers).forEach { index ->
                        Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                            Text(headers[index], style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Text(parseBoldMarkdown(row.getOrNull(index).orEmpty()), style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
        }
    }
    return true
}
