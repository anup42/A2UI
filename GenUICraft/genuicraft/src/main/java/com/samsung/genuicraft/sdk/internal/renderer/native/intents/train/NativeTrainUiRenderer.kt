package com.samsung.genuicraft.sdk.internal.renderer.native.intents.train

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import com.samsung.genuicraft.sdk.internal.renderer.RendererFlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Train
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import com.samsung.genuicraft.sdk.internal.renderer.CitationText as Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp

/** Compact rail rows retain every supplied label/value without timetable inference or sorting. */
internal object NativeTrainUiRenderer {
    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    fun RenderTrainRows(
        rows: List<NativeTrainRow>,
        title: String? = null,
        modifier: Modifier = Modifier,
    ) {
        val colors = MaterialTheme.colorScheme
        Column(
            modifier = modifier.fillMaxWidth().semantics {
                contentDescription = "Train comparison, ${rows.size} services"
            },
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            // An enclosing card may already name the route. Do not add another default heading.
            title?.takeIf(String::isNotBlank)?.let { suppliedTitle ->
                RendererFlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    Text(
                        suppliedTitle.replace('_', ' ').replaceFirstChar { it.titlecase() },
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.semantics { heading() },
                    )
                    Text(
                        "${rows.size} ${if (rows.size == 1) "service" else "services"}",
                        style = MaterialTheme.typography.labelMedium,
                        color = colors.onSurfaceVariant,
                    )
                }
            }
            val fontScale = LocalDensity.current.fontScale.coerceAtLeast(1f)
            BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
                val columns = if (maxWidth >= 640.dp * fontScale + 16.dp) 2 else 1
                Surface(
                    shape = RoundedCornerShape(14.dp),
                    color = colors.surface,
                    border = BorderStroke(1.dp, colors.outlineVariant.copy(alpha = 0.45f)),
                ) {
                    Column {
                        rows.chunked(columns).forEachIndexed { groupIndex, group ->
                            if (groupIndex > 0) HorizontalDivider(
                                modifier = Modifier.padding(horizontal = 12.dp),
                                color = colors.outlineVariant.copy(alpha = 0.55f),
                            )
                            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                group.forEachIndexed { index, row ->
                                    TrainRow(row, groupIndex * columns + index, Modifier.weight(1f))
                                }
                                if (group.size < columns) Spacer(Modifier.weight(1f))
                            }
                        }
                    }
                }
            }
        }
    }

    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    private fun TrainRow(row: NativeTrainRow, index: Int, modifier: Modifier) {
        val colors = MaterialTheme.colorScheme
        Column(
            modifier = modifier.padding(horizontal = 12.dp, vertical = 9.dp).semantics {
                contentDescription = "Train service card ${index + 1}"
            },
            verticalArrangement = Arrangement.spacedBy(5.dp),
        ) {
            Row(
                horizontalArrangement = Arrangement.spacedBy(7.dp),
                verticalAlignment = Alignment.Top,
            ) {
                Icon(
                    Icons.Filled.Train,
                    contentDescription = null,
                    tint = colors.primary,
                    modifier = Modifier.padding(top = 2.dp).size(17.dp),
                )
                Text(
                    row.service.value,
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    color = colors.onSurface,
                    modifier = Modifier.weight(1f).semantics {
                        heading()
                        contentDescription = "${row.service.label}: ${row.service.value}"
                    },
                )
            }
            row.station?.let { station -> CompactField(station, muted = true) }

            // A departure range is one field, never a departure/arrival route.
            if (row.departure != null || row.duration != null) {
                val fontScale = LocalDensity.current.fontScale.coerceAtLeast(1f)
                BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
                    if (row.departure != null && row.duration != null && maxWidth >= 300.dp * fontScale) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(14.dp),
                            verticalAlignment = Alignment.Top,
                        ) {
                            TimingField(row.departure, prominent = true, modifier = Modifier.weight(1f))
                            TimingField(row.duration, prominent = false, modifier = Modifier.weight(1.25f))
                        }
                    } else {
                        Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
                            row.departure?.let { TimingField(it, prominent = true) }
                            row.duration?.let { TimingField(it, prominent = false) }
                        }
                    }
                }
            }

            // Class stays a single source string; additional columns remain labeled and complete.
            RendererFlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalArrangement = Arrangement.spacedBy(3.dp),
            ) {
                row.seating?.let { seating ->
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = colors.surfaceContainerLow,
                        border = BorderStroke(1.dp, colors.outlineVariant.copy(alpha = 0.35f)),
                    ) {
                        Column(modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)) {
                            CompactField(seating)
                        }
                    }
                }
                row.extraFields.forEach { field -> CompactField(field) }
            }
        }
    }

    @Composable
    private fun TimingField(
        field: NativeTrainField,
        prominent: Boolean,
        modifier: Modifier = Modifier,
    ) {
        val colors = MaterialTheme.colorScheme
        Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(
                field.label,
                style = MaterialTheme.typography.labelSmall,
                color = colors.onSurfaceVariant,
            )
            Text(
                field.value,
                style = if (prominent) MaterialTheme.typography.titleLarge.copy(fontFeatureSettings = "tnum")
                    else MaterialTheme.typography.bodyMedium,
                fontWeight = if (prominent) FontWeight.Bold else FontWeight.Medium,
                color = if (prominent) colors.primary else colors.onSurface,
            )
        }
    }

    @Composable
    private fun CompactField(field: NativeTrainField, muted: Boolean = false) {
        val colors = MaterialTheme.colorScheme
        Text(
            text = buildAnnotatedString {
                withStyle(SpanStyle(color = colors.onSurfaceVariant)) { append(field.label); append(": ") }
                withStyle(SpanStyle(color = if (muted) colors.onSurfaceVariant else colors.onSurface, fontWeight = FontWeight.Medium)) {
                    append(field.value)
                }
            },
            style = MaterialTheme.typography.bodySmall.copy(fontFeatureSettings = "tnum"),
        )
    }
}
