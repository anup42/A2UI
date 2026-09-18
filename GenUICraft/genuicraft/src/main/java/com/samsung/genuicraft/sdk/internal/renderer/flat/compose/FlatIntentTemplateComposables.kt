package com.samsung.genuicraft.sdk.internal.renderer.flat.compose

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Block
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.RadioButtonUnchecked
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.samsung.genuicraft.sdk.internal.renderer.RenderChildren

private data class AlertPalette(
    val container: Color,
    val content: Color,
    val icon: ImageVector
)

@Composable
private fun alertPalette(tone: String): AlertPalette = when (tone) {
    "success" -> AlertPalette(
        MaterialTheme.colorScheme.primaryContainer,
        MaterialTheme.colorScheme.onPrimaryContainer,
        Icons.Default.CheckCircle
    )
    "warning" -> AlertPalette(
        MaterialTheme.colorScheme.tertiaryContainer,
        MaterialTheme.colorScheme.onTertiaryContainer,
        Icons.Default.Warning
    )
    "error" -> AlertPalette(
        MaterialTheme.colorScheme.errorContainer,
        MaterialTheme.colorScheme.onErrorContainer,
        Icons.Default.Error
    )
    else -> AlertPalette(
        MaterialTheme.colorScheme.secondaryContainer,
        MaterialTheme.colorScheme.onSecondaryContainer,
        Icons.Default.Info
    )
}

/** Reusable notification/status card used by notification and QR-result recipes. */
@Composable
internal fun RenderAlert(context: FlatRenderContext) {
    val props = context.props
    val title = props["title"]?.toString()?.trim().orEmpty()
    val message = (props["message"] ?: props["text"])?.toString()?.trim().orEmpty()
    val tone = props["tone"]?.toString()?.trim()?.lowercase().orEmpty().ifBlank { "info" }
    val timestamp = props["timestamp"]?.toString()?.trim().orEmpty()
    val source = props["source"]?.toString()?.trim().orEmpty()
    val palette = alertPalette(tone)

    Surface(
        modifier = context.modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 6.dp)
            .semantics {
                contentDescription = listOf(title, message, timestamp, source)
                    .filter { it.isNotBlank() }
                    .joinToString(". ")
            },
        color = palette.container,
        contentColor = palette.content,
        shape = RoundedCornerShape(14.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(verticalAlignment = Alignment.Top) {
                val declaredIcon = props["icon"]?.toString()?.trim().orEmpty()
                if (declaredIcon.isNotBlank()) {
                    RenderIcon(
                        props = mapOf("name" to declaredIcon, "decorative" to true),
                        modifier = Modifier.size(22.dp)
                    )
                } else {
                    Icon(
                        imageVector = palette.icon,
                        contentDescription = null,
                        modifier = Modifier.size(22.dp),
                        tint = palette.content
                    )
                }
                Spacer(Modifier.width(10.dp))
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    if (title.isNotBlank()) {
                        Text(
                            text = title,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                            modifier = Modifier.semantics { heading() }
                        )
                    }
                    if (message.isNotBlank()) {
                        Text(text = message, style = MaterialTheme.typography.bodyMedium)
                    }
                    if (timestamp.isNotBlank() || source.isNotBlank()) {
                        Text(
                            text = listOf(timestamp, source).filter { it.isNotBlank() }.joinToString(" • "),
                            style = MaterialTheme.typography.labelSmall
                        )
                    }
                }
            }
            RenderChildren(
                children = context.children,
                elements = context.elements,
                state = context.state,
                repeatScope = context.repeatScope,
                repeatedChildScopes = context.repeatedChildScopes,
                onOpenUrl = context.onOpenUrl,
                onSetState = context.onSetState,
                onAction = context.onAction,
                activePath = context.activePath
            )
        }
    }
}

private data class ChecklistItem(
    val label: String,
    val detail: String,
    val state: String,
    val required: Boolean
)

private fun checklistItems(raw: Any?): List<ChecklistItem> = (raw as? List<*>)
    .orEmpty()
    .mapNotNull { item ->
        val map = item as? Map<*, *> ?: return@mapNotNull null
        val label = map["label"]?.toString()?.trim().orEmpty()
        if (label.isBlank()) return@mapNotNull null
        ChecklistItem(
            label = label,
            detail = map["detail"]?.toString()?.trim().orEmpty(),
            state = map["state"]?.toString()?.trim()?.lowercase().orEmpty().ifBlank { "pending" },
            required = map["required"] as? Boolean ?: false
        )
    }
    .take(20)

private fun checklistIcon(state: String): ImageVector = when (state) {
    "complete" -> Icons.Default.CheckCircle
    "warning" -> Icons.Default.Warning
    "blocked" -> Icons.Default.Block
    else -> Icons.Default.RadioButtonUnchecked
}

/** Source-aware checklist used by healthcare, legal, and career recipes. */
@Composable
internal fun RenderChecklist(context: FlatRenderContext) {
    val title = context.props["title"]?.toString()?.trim().orEmpty()
    val disclaimer = context.props["disclaimer"]?.toString()?.trim().orEmpty()
    val source = context.props["source"]?.toString()?.trim().orEmpty()
    val items = checklistItems(context.props["items"])

    Surface(
        modifier = context.modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 6.dp),
        shape = RoundedCornerShape(14.dp),
        tonalElevation = 1.dp
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            if (title.isNotBlank()) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.semantics { heading() }
                )
            }
            items.forEach { item ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .semantics {
                            contentDescription = buildString {
                                append(item.label)
                                append(". ")
                                append(item.state)
                                if (item.required) append(". Required")
                                if (item.detail.isNotBlank()) append(". ${item.detail}")
                            }
                        },
                    verticalAlignment = Alignment.Top
                ) {
                    Icon(
                        imageVector = checklistIcon(item.state),
                        contentDescription = null,
                        modifier = Modifier.size(20.dp),
                        tint = when (item.state) {
                            "complete" -> MaterialTheme.colorScheme.primary
                            "warning", "blocked" -> MaterialTheme.colorScheme.error
                            else -> MaterialTheme.colorScheme.onSurfaceVariant
                        }
                    )
                    Spacer(Modifier.width(10.dp))
                    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text(
                            text = item.label + if (item.required) " *" else "",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Medium
                        )
                        if (item.detail.isNotBlank()) {
                            Text(
                                text = item.detail,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }
            if (disclaimer.isNotBlank() || source.isNotBlank()) {
                Text(
                    text = listOf(disclaimer, source).filter { it.isNotBlank() }.joinToString("\n"),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            RenderChildren(
                children = context.children,
                elements = context.elements,
                state = context.state,
                repeatScope = context.repeatScope,
                repeatedChildScopes = context.repeatedChildScopes,
                onOpenUrl = context.onOpenUrl,
                onSetState = context.onSetState,
                onAction = context.onAction,
                activePath = context.activePath
            )
        }
    }
}
