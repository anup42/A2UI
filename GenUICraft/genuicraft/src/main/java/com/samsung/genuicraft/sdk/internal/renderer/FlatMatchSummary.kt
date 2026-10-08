package com.samsung.genuicraft.sdk.internal.renderer

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
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
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement
import com.samsung.genuicraft.sdk.internal.renderer.CitationText as Text

internal data class FlatMatchField(val rootId: String, val textId: String, val label: String, val value: String)
internal data class FlatMatchSummary(val fields: List<FlatMatchField>) {
    val rootIds: List<String> get() = fields.map { it.rootId }
    val scores: List<FlatMatchField> get() = fields.filter { it.label.matches(Regex("(?i).+ scores?")) }
    val result: FlatMatchField get() = fields.single { it.label.equals("result", true) }
    val details: List<FlatMatchField> get() = fields.filter { it !in scores && it != result }
}

/** Retains literal score strings; never guesses wickets, overs, innings, winners or match state. */
internal fun planFlatMatchSummary(children: List<String>, elements: Map<String, FlatElement>): FlatMatchSummary? {
    if (children.distinct().size != children.size) return null
    fun passive(node: FlatElement) = node.repeat == null && node.visible == null && node.on.isNullOrEmpty() && node.watch.isNullOrEmpty()
    fun field(id: String): FlatMatchField? {
        val root = elements[id] ?: return null
        if (!passive(root)) return null
        val node = if (root.type.equals("card", true) && root.props.isEmpty() && root.children.size == 1) {
            elements[root.children.single()] ?: return null
        } else root
        if (!node.type.equals("text", true) || !passive(node) || node.children.isNotEmpty() ||
            node.props.keys.any { it !in setOf("text", "variant", "typography") }) return null
        val variant = (node.props["variant"] ?: node.props["typography"])?.toString()?.lowercase().orEmpty()
        if (variant !in setOf("caption", "body", "")) return null
        val text = node.props["text"] as? String ?: return null
        if (text.contains("{{") || text.contains("\n") || isLikelyHttpUrl(text)) return null
        val separator = text.indexOf(':')
        if (separator !in 1..48) return null
        val label = text.substring(0, separator).trim()
        val value = text.substring(separator + 1).trim()
        if (label.isBlank() || value.isBlank()) return null
        return FlatMatchField(id, if (node === root) id else root.children.single(), label, value)
    }
    val runs = mutableListOf<List<FlatMatchField>>()
    var run = mutableListOf<FlatMatchField>()
    children.forEach { id ->
        val next = field(id)
        if (next != null) run += next else if (run.isNotEmpty()) { runs += run.toList(); run = mutableListOf() }
    }
    if (run.isNotEmpty()) runs += run
    return runs.map(::FlatMatchSummary).singleOrNull { summary ->
        summary.fields.size >= 4 && summary.scores.size == 2 &&
            summary.fields.count { it.label.equals("format", true) } == 1 &&
            summary.fields.count { it.label.equals("result", true) } == 1 &&
            summary.fields.map { it.label.lowercase() }.distinct().size == summary.fields.size &&
            summary.fields.map { it.textId }.distinct().size == summary.fields.size
    }
}

@Composable
internal fun RenderFlatMatchSummary(summary: FlatMatchSummary) {
    Card(Modifier.fillMaxWidth(), shape = RoundedCornerShape(16.dp), colors = flatSpecCardColors()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            summary.details.forEach { field ->
                Text(parseBoldMarkdown("**${field.label}:** ${field.value}"), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            summary.scores.forEach { field ->
                BoxWithConstraints(Modifier.fillMaxWidth()) {
                    if (maxWidth < 320.dp * LocalDensity.current.fontScale || field.value.length > 28) {
                        Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                            Text(field.label, style = MaterialTheme.typography.labelLarge)
                            Text(parseBoldMarkdown(field.value), style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold))
                        }
                    } else {
                        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                            Text(field.label, Modifier.weight(1f), style = MaterialTheme.typography.titleSmall)
                            Text(parseBoldMarkdown(field.value), Modifier.weight(1.3f), style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold))
                        }
                    }
                }
            }
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                Text(summary.result.label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(parseBoldMarkdown(summary.result.value), style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold), color = MaterialTheme.colorScheme.primary)
            }
        }
    }
}
