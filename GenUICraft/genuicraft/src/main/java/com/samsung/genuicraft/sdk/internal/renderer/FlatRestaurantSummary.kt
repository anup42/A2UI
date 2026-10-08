package com.samsung.genuicraft.sdk.internal.renderer

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.samsung.genuicraft.sdk.internal.renderer.CitationText as Text
import java.util.Locale

internal data class RestaurantSummaryProfile(val name: Int, val neighborhood: Int, val dish: Int) {
    fun detailIndexes(headers: List<String>): List<Int> = headers.indices.filter { it != name && it != neighborhood && it != dish }
}

internal fun restaurantSummaryProfile(headers: List<String>, rows: List<List<String>>): RestaurantSummaryProfile? {
    if (rows.isEmpty()) return null
    val labels = headers.map { it.lowercase(Locale.ROOT).replace('_', ' ').trim() }
    if (labels.any { it in setOf("feature", "factor", "step", "stage") }) return null
    val name = labels.indexOfFirst { it in setOf("restaurant", "restaurant name", "eatery", "cafe") }
    val area = labels.indexOfFirst { it in setOf("neighborhood", "neighbourhood", "area", "location") }
    val dish = labels.indexOfFirst { it in setOf("signature dish", "signature dishes", "speciality", "specialty") }
    if (name < 0 || area < 0 || dish < 0 || rows.any { it.size > headers.size || it.getOrNull(name).isNullOrBlank() }) return null
    // Keep the existing rich restaurant route for authored media and actions.
    if (labels.any { it.contains("url") || it.contains("photo") || it.contains("image") || it.contains("action") || it.contains("website") ||
            Regex("\\b(rating|reviews?|phone|telephone|maps?|booking|reservation)\\b").containsMatchIn(it) }) return null
    return RestaurantSummaryProfile(name, area, dish)
}

@Composable
internal fun renderRestaurantSummaryIfPossible(
    headers: List<String>, rows: List<List<String>>, modifier: Modifier = Modifier, title: String? = null,
): Boolean {
    val profile = restaurantSummaryProfile(headers, rows) ?: return false
    Column(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        title?.takeIf(String::isNotBlank)?.let {
            Text(parseBoldMarkdown(it), style = MaterialTheme.typography.titleMedium, modifier = Modifier.semantics { heading() })
        }
        rows.forEach { row ->
            Card(modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(16.dp), colors = flatSpecCardColors()) {
                Column(Modifier.fillMaxWidth().padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                        Text(headers[profile.name], style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.primary)
                        Text(parseBoldMarkdown(row[profile.name]), style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold), modifier = Modifier.semantics { heading() })
                        Text(parseBoldMarkdown("${headers[profile.neighborhood]}: ${row.getOrNull(profile.neighborhood).orEmpty()}"), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    Surface(shape = RoundedCornerShape(10.dp), color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.45f)) {
                        Column(Modifier.fillMaxWidth().padding(horizontal = 10.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                            Text(headers[profile.dish], style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onPrimaryContainer)
                            Text(parseBoldMarkdown(row.getOrNull(profile.dish).orEmpty()), style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Medium), color = MaterialTheme.colorScheme.onPrimaryContainer)
                        }
                    }
                    profile.detailIndexes(headers).forEach { index ->
                        Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                            Text(headers[index], style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Text(parseBoldMarkdown(row.getOrNull(index).orEmpty()), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurface)
                        }
                    }
                }
            }
        }
    }
    return true
}
