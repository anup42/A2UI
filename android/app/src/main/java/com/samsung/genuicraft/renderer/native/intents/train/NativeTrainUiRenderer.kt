package com.samsung.genuicraft.renderer.native.intents.train

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.EventSeat
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.Schedule
import androidx.compose.material.icons.filled.Train
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/** A presentation of supplied rail comparison cells, with no timetable inference or sorting. */
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
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Text(
                    title?.takeIf(String::isNotBlank)?.replace('_', ' ')
                        ?.replaceFirstChar { it.titlecase() } ?: "Train comparison",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(vertical = 3.dp).semantics { heading() },
                )
                Surface(shape = RoundedCornerShape(8.dp), color = colors.surfaceContainerHigh) {
                    Text(
                        "${rows.size} ${if (rows.size == 1) "service" else "services"}",
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                        style = MaterialTheme.typography.labelMedium,
                        color = colors.onSurfaceVariant,
                    )
                }
            }
            val fontScale = LocalDensity.current.fontScale.coerceAtLeast(1f)
            BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
                val columns = if (maxWidth >= 640.dp * fontScale + 16.dp) 2 else 1
                Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
                    rows.chunked(columns).forEachIndexed { groupIndex, group ->
                        Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                            group.forEachIndexed { index, row ->
                                TrainCard(row, groupIndex * columns + index, Modifier.weight(1f))
                            }
                            if (group.size < columns) Spacer(Modifier.weight(1f))
                        }
                    }
                }
            }
        }
    }

    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    private fun TrainCard(row: NativeTrainRow, index: Int, modifier: Modifier) {
        val colors = MaterialTheme.colorScheme
        Card(
            modifier = modifier.semantics { contentDescription = "Train service card ${index + 1}" },
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = colors.surface),
            border = BorderStroke(1.dp, colors.outlineVariant.copy(alpha = 0.5f)),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth().background(
                    Brush.horizontalGradient(listOf(colors.primary.copy(alpha = 0.12f), colors.primary.copy(alpha = 0.035f))),
                ).padding(horizontal = 16.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Surface(shape = RoundedCornerShape(11.dp), color = colors.primaryContainer) {
                    Icon(
                        Icons.Filled.Train,
                        contentDescription = null,
                        tint = colors.onPrimaryContainer,
                        modifier = Modifier.padding(9.dp).size(20.dp),
                    )
                }
                Text(
                    row.service.value,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = colors.onSurface,
                    modifier = Modifier.weight(1f).semantics { heading() },
                )
            }
            Column(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 14.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                row.departure?.let { departure ->
                    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Label(departure.label)
                        Text(
                            departure.value,
                            style = MaterialTheme.typography.headlineMedium.copy(fontFeatureSettings = "tnum"),
                            fontWeight = FontWeight.Bold,
                            color = colors.primary,
                        )
                    }
                }
                row.station?.let { station ->
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(
                            Icons.Filled.LocationOn,
                            contentDescription = null,
                            tint = colors.onSurfaceVariant,
                            modifier = Modifier.size(18.dp),
                        )
                        Field(station, Modifier.weight(1f))
                    }
                }
                row.extraFields.forEach { Field(it) }
            }
            if (row.duration != null || row.seating != null) {
                Canvas(Modifier.fillMaxWidth().padding(horizontal = 16.dp).height(1.dp)) {
                    drawLine(
                        color = colors.outlineVariant,
                        start = Offset(0f, size.height / 2),
                        end = Offset(size.width, size.height / 2),
                        strokeWidth = 1.dp.toPx(),
                        pathEffect = PathEffect.dashPathEffect(floatArrayOf(4.dp.toPx(), 4.dp.toPx())),
                    )
                }
                // Natural widths let long durations use the full row instead of a squeezed half-card.
                FlowRow(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    row.duration?.let { Metadata(it, Icons.Filled.Schedule) }
                    row.seating?.let { Metadata(it, Icons.Filled.EventSeat) }
                }
            }
        }
    }

    @Composable
    private fun Metadata(field: NativeTrainField, icon: ImageVector) {
        val colors = MaterialTheme.colorScheme
        Surface(
            shape = RoundedCornerShape(9.dp),
            color = colors.surfaceContainerHigh,
            modifier = Modifier.semantics(mergeDescendants = true) {},
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(icon, contentDescription = null, tint = colors.onSurfaceVariant, modifier = Modifier.size(16.dp))
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(field.label, style = MaterialTheme.typography.labelSmall, color = colors.onSurfaceVariant)
                    Text(
                        field.value,
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.Medium,
                        color = colors.onSurface,
                    )
                }
            }
        }
    }

    @Composable
    private fun Field(field: NativeTrainField, modifier: Modifier = Modifier) {
        Column(modifier, verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Label(field.label)
            Text(
                field.value,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.onSurface,
            )
        }
    }

    @Composable
    private fun Label(text: String) {
        Text(text, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}
