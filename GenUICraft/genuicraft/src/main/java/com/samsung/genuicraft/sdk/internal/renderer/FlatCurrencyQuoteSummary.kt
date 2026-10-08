package com.samsung.genuicraft.sdk.internal.renderer

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import java.util.Locale
import com.samsung.genuicraft.sdk.internal.renderer.CitationText as Text

internal data class CurrencyQuoteSummaryProfile(val currency: Int, val rate: Int, val quotedAt: Int?)

internal fun currencyQuoteSummaryProfile(headers: List<String>, rows: List<List<String>>): CurrencyQuoteSummaryProfile? {
    if (headers.size !in 2..3 || rows.isEmpty() || rows.any { it.size != headers.size || it.any(String::isBlank) }) return null
    val labels = headers.map { it.lowercase(Locale.ROOT).replace('_', ' ').trim() }
    val currency = labels.indexOfFirst { it in setOf("currency", "currency pair", "pair") }
    val rate = labels.indexOfFirst { it in setOf("rate", "exchange rate") }
    val quotedAt = labels.indexOfFirst { it in setOf("quoted as of", "as of", "quoted at") }.takeIf { it >= 0 }
    if (currency < 0 || rate < 0 || listOfNotNull(currency, rate, quotedAt).toSet().size != headers.size) return null
    val pair = Regex("[A-Z]{3}/[A-Z]{3}(?:\\s*\\[\\d+])*\\s*")
    if (rows.any { !pair.matches(it[currency]) || it.any { value ->
            isLikelyHttpUrl(value) || Regex("(?i)\\bhttps?://").containsMatchIn(value)
        } }) return null
    return CurrencyQuoteSummaryProfile(currency, rate, quotedAt)
}

@Composable
internal fun renderCurrencyQuoteSummaryIfPossible(
    headers: List<String>, rows: List<List<String>>, modifier: Modifier = Modifier, title: String? = null,
): Boolean {
    val profile = currencyQuoteSummaryProfile(headers, rows) ?: return false
    val commonDate = profile.quotedAt?.let { index -> rows.map { it[index] }.distinct().singleOrNull() }
    Column(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        title?.takeIf(String::isNotBlank)?.let {
            Text(parseBoldMarkdown(it), style = MaterialTheme.typography.titleMedium, modifier = Modifier.semantics { heading() })
        }
        Card(Modifier.fillMaxWidth(), shape = RoundedCornerShape(16.dp), colors = flatSpecCardColors()) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text(headers[profile.currency], Modifier.weight(1f), style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Text(headers[profile.rate], Modifier.weight(1f), style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.End)
                }
                rows.forEachIndexed { index, row ->
                    if (index > 0) HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                            Text(parseBoldMarkdown(row[profile.currency]), Modifier.weight(1f), style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold))
                            Text(parseBoldMarkdown(row[profile.rate]), Modifier.weight(1f), style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold), color = MaterialTheme.colorScheme.primary, textAlign = TextAlign.End)
                        }
                        if (commonDate == null) profile.quotedAt?.let {
                            Text("${headers[it]}: ${row[it]}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
                if (commonDate != null) {
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                    Text("${headers[profile.quotedAt!!]}: $commonDate", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
    }
    return true
}
