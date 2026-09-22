package com.samsung.genuicraft.sdk.internal.renderer.native.intents.train

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.EventSeat
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.Train
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/** A presentation of supplied rail comparison cells, with no timetable inference or sorting. */
internal object NativeTrainUiRenderer {
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
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Surface(shape = RoundedCornerShape(14.dp), color = colors.primaryContainer) {
                    Icon(
                        Icons.Filled.Train,
                        contentDescription = null,
                        tint = colors.onPrimaryContainer,
                        modifier = Modifier.padding(11.dp).size(26.dp),
                    )
                }
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        title?.takeIf(String::isNotBlank)?.replace('_', ' ')
                            ?.replaceFirstChar { it.titlecase() } ?: "Train comparison",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.semantics { heading() },
                    )
                    Text(
                        "${rows.size} ${if (rows.size == 1) "service" else "services"}",
                        style = MaterialTheme.typography.bodyMedium,
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

    @Composable
    private fun TrainCard(row: NativeTrainRow, index: Int, modifier: Modifier) {
        val colors = MaterialTheme.colorScheme
        Card(
            modifier = modifier.semantics { contentDescription = "Train service card ${index + 1}" },
            shape = RoundedCornerShape(22.dp),
            colors = CardDefaults.cardColors(containerColor = colors.surface),
            border = BorderStroke(1.dp, colors.outlineVariant.copy(alpha = 0.65f)),
        ) {
            Column(
                modifier = Modifier.fillMaxWidth().background(
                    Brush.verticalGradient(listOf(colors.primary.copy(alpha = 0.07f), colors.surface)),
                ).padding(18.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                Text(
                    row.service.value,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = colors.onSurface,
                    modifier = Modifier.semantics { heading() },
                )
                row.station?.let { station ->
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Icon(
                            Icons.Filled.LocationOn,
                            contentDescription = null,
                            tint = colors.primary,
                            modifier = Modifier.padding(top = 2.dp).size(18.dp),
                        )
                        Field(station, Modifier.weight(1f))
                    }
                }
                if (row.departure != null || row.duration != null) {
                    HorizontalDivider(color = colors.outlineVariant.copy(alpha = 0.65f))
                    Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                        row.departure?.let { departure ->
                            Column(Modifier.weight(1.15f), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                                Label(departure.label)
                                Text(
                                    departure.value,
                                    style = MaterialTheme.typography.headlineSmall,
                                    fontWeight = FontWeight.Bold,
                                    color = colors.primary,
                                )
                            }
                        }
                        row.duration?.let { duration ->
                            Surface(
                                modifier = Modifier.weight(1f),
                                shape = RoundedCornerShape(14.dp),
                                color = colors.secondaryContainer.copy(alpha = 0.55f),
                            ) {
                                Field(duration, Modifier.padding(12.dp))
                            }
                        }
                    }
                }
                row.seating?.let { seating ->
                    Row(horizontalArrangement = Arrangement.spacedBy(9.dp)) {
                        Icon(
                            Icons.Filled.EventSeat,
                            contentDescription = null,
                            tint = colors.onSurfaceVariant,
                            modifier = Modifier.padding(top = 2.dp).size(19.dp),
                        )
                        Field(seating, Modifier.weight(1f))
                    }
                }
                row.extraFields.forEach { Field(it) }
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
