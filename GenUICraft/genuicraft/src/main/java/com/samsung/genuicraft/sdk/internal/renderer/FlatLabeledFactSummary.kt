package com.samsung.genuicraft.sdk.internal.renderer

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.key
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement
import com.samsung.genuicraft.sdk.internal.renderer.CitationText as Text

internal data class FlatLabeledFact(val id: String, val label: String, val value: String)

/** Explicit literal label/value runs only. Qualifications remain part of the original value. */
internal fun planFlatLabeledFacts(children: List<String>, elements: Map<String, FlatElement>): List<FlatLabeledFact>? {
    if (children.distinct().size != children.size) return null
    fun fact(id: String): FlatLabeledFact? {
        val node = elements[id] ?: return null
        if (!node.type.equals("text", true) || node.children.isNotEmpty() || node.repeat != null || node.visible != null ||
            !node.on.isNullOrEmpty() || !node.watch.isNullOrEmpty() ||
            node.props.keys.any { it !in setOf("text", "variant", "typography") }) return null
        val variant = (node.props["variant"] ?: node.props["typography"])?.toString()?.lowercase().orEmpty()
        if (variant !in setOf("label", "caption", "body", "")) return null
        val text = node.props["text"] as? String ?: return null
        if (text.contains("{{") || text.contains('\n') || isLikelyHttpUrl(text)) return null
        val separator = text.indexOf(':')
        if (separator !in 1..48) return null
        val label = text.substring(0, separator).trim()
        val value = text.substring(separator + 1).trim()
        if (label.isBlank() || label.split(Regex("\\s+")).size > 6 || label.endsWith('.') || value.isBlank() ||
            isLikelyHttpUrl(value) || Regex("(?i)\\bhttps?://").containsMatchIn(value)) return null
        return FlatLabeledFact(id, label, value)
    }
    val runs = mutableListOf<List<FlatLabeledFact>>()
    var run = mutableListOf<FlatLabeledFact>()
    children.forEach { id ->
        val next = fact(id)
        if (next != null) run += next else if (run.isNotEmpty()) { runs += run.toList(); run = mutableListOf() }
    }
    if (run.isNotEmpty()) runs += run
    return runs.filter { it.size >= 3 && it.map { field -> field.label.lowercase() }.distinct().size == it.size }.singleOrNull()
}

@Composable
internal fun RenderFlatLabeledFacts(fields: List<FlatLabeledFact>) {
    Card(Modifier.fillMaxWidth(), shape = RoundedCornerShape(16.dp), colors = flatSpecCardColors()) {
        BoxWithConstraints(Modifier.fillMaxWidth().padding(12.dp)) {
            val groups = labeledFactRows(fields, maxWidth >= 320.dp * LocalDensity.current.fontScale)
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                groups.forEachIndexed { index, group ->
                    key(group.first().id) {
                        if (index > 0) HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = .55f))
                        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                            group.forEach { field ->
                                val identity = field == fields.first() && Regex("(?i)\\b(name|title)$").containsMatchIn(field.label)
                                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                                    Text(parseBoldMarkdown(field.label), style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Text(parseBoldMarkdown(field.value),
                                        style = if (identity) MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold) else MaterialTheme.typography.bodyMedium,
                                        color = if (identity) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

internal fun labeledFactRows(fields: List<FlatLabeledFact>, canPair: Boolean): List<List<FlatLabeledFact>> {
    val groups = mutableListOf<List<FlatLabeledFact>>()
    fun compact(field: FlatLabeledFact) = field.label.length <= 28 && field.value.length <= 48
    var index = 0
    while (index < fields.size) {
        val firstIdentity = index == 0 && Regex("(?i)\\b(name|title)$").containsMatchIn(fields[index].label)
        val count = if (canPair && !firstIdentity && index + 1 < fields.size && compact(fields[index]) && compact(fields[index + 1])) 2 else 1
        groups += fields.subList(index, index + count)
        index += count
    }
    return groups
}
