package com.samsung.genuicraft.renderer

import com.samsung.genuicraft.renderer.flat.capability.GeneratedRendererCapabilities

import android.content.Intent
import android.content.res.Configuration
import android.net.Uri
import android.util.Log
import androidx.compose.foundation.clickable
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.relocation.BringIntoViewRequester
import androidx.compose.foundation.relocation.bringIntoViewRequester
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.OpenInNew
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Directions
import androidx.compose.material.icons.filled.EventAvailable
import androidx.compose.material.icons.filled.FlightTakeoff
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.CalendarToday
import androidx.compose.material.icons.filled.Language
import androidx.compose.material.icons.filled.PlayCircle
import androidx.compose.material.icons.filled.Link
import androidx.compose.material.icons.filled.RateReview
import androidx.compose.material.icons.filled.Restaurant
import androidx.compose.material.icons.filled.Star
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.BaselineShift
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import coil.ImageLoader
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.samsung.genuicraft.GenUiTokens
import com.samsung.genuicraft.GenUiCardTone
import com.samsung.genuicraft.genUiCardContainerColor
import com.google.gson.GsonBuilder
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.samsung.genuicraft.renderer.native.NativeTextFormatter
import com.samsung.genuicraft.renderer.native.NativePayloadParser
import com.samsung.genuicraft.renderer.native.ParsedButton
import com.samsung.genuicraft.renderer.native.intents.flight.NativeFlightSemantics
import com.samsung.genuicraft.renderer.native.intents.flight.NativeFlightUiRenderer
import com.samsung.genuicraft.renderer.native.intents.weather.NativeWeatherSemantics
import com.samsung.genuicraft.renderer.native.intents.weather.NativeWeatherUiRenderer
import com.samsung.genuicraft.renderer.native.media.NativeMediaVisualUtils
import com.samsung.genuicraft.renderer.native.parser.NativeSourceParsing
import com.samsung.genuicraft.renderer.native.parser.NativeStructureParsing
import com.samsung.genuicraft.security.SafeContentPolicy
import kotlinx.coroutines.launch
import java.util.Locale
import kotlin.math.roundToInt
import java.util.UUID
import java.net.URI
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.DatePicker
import androidx.compose.material3.DatePickerDialog
import androidx.compose.material3.rememberDatePickerState
import androidx.compose.material3.TimePicker
import androidx.compose.material3.rememberTimePickerState
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.TextButton
import androidx.compose.material3.IconButton
import androidx.compose.material.icons.filled.Schedule
import java.time.Instant
import java.time.ZoneOffset
import com.samsung.genuicraft.renderer.flat.domain.*

// moved from FlatSpecRenderer.kt (RenderChart)
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun RenderChart(
    props: Map<String, Any?>,
    state: Map<String, Any?>,
    modifier: Modifier = Modifier
) {
    val chartType = props["chartType"]?.toString()?.trim()?.lowercase().orEmpty().ifBlank { "bar" }
    val canonicalChartType = GeneratedRendererCapabilities.chartSubtypeAliases[chartType] ?: chartType
    val diagnosticSink = LocalFlatDiagnosticSink.current
    if (canonicalChartType !in GeneratedRendererCapabilities.chartSubtypes) {
        val diagnostic = FlatDiagnostic(
            code = FlatDiagnostic.Code.UNSUPPORTED_CHART_TYPE,
            severity = FlatDiagnostic.Severity.WARNING,
            message = "Unsupported chart type '$chartType'. Supported: ${GeneratedRendererCapabilities.chartSubtypes.sorted()}.",
            details = mapOf("chartType" to chartType)
        )
        LaunchedEffect(diagnostic.code, chartType) { diagnosticSink(diagnostic) }
        RenderUnsupportedElement("Unsupported chart: $chartType", modifier)
        return
    }
    val points = extractChartPoints(props, state)
    if (points.isEmpty()) {
        val diagnostic = FlatDiagnostic(
            code = FlatDiagnostic.Code.EMPTY_REQUIRED_DATA,
            severity = FlatDiagnostic.Severity.WARNING,
            message = "Chart requires at least one valid data point."
        )
        LaunchedEffect(diagnostic.code) { diagnosticSink(diagnostic) }
        RenderUnsupportedElement("Chart has no data", modifier)
        return
    }

    val title = props["title"]?.toString()?.trim().orEmpty()
    val subtitle = props["subtitle"]?.toString()?.trim().orEmpty()
    val yLabel = props["yLabel"]?.toString()?.trim().orEmpty()
    val maxValue = points.maxOf { it.value }.takeIf { it > 0.0 } ?: 1.0
    val currencyPrefix = inferChartCurrency(points)
    val total = points.sumOf { it.value }
    val peak = points.maxByOrNull { it.value }

    Card(
        modifier = modifier
            .fillMaxWidth()
            .semantics(mergeDescendants = true) {
                contentDescription = buildString {
                    if (title.isNotBlank()) append(title).append(". ")
                    append("Bar chart with ${points.size} values. ")
                    points.forEach { point ->
                        append(point.label).append(": ").append(point.displayValue).append(". ")
                    }
                }
            },
        shape = RoundedCornerShape(22.dp),
        colors = CardDefaults.cardColors(containerColor = genUiCardContainerColor(GenUiCardTone.Neutral)),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            if (title.isNotBlank() || subtitle.isNotBlank()) {
                Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                    if (title.isNotBlank()) {
                        Text(
                            text = parseBoldMarkdown(title),
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                    if (subtitle.isNotBlank()) {
                        Text(
                            text = parseBoldMarkdown(subtitle),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }

            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                points.forEachIndexed { index, point ->
                    val fraction = (point.value / maxValue).toFloat().coerceIn(0.04f, 1f)
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = parseBoldMarkdown(point.label),
                            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                            color = MaterialTheme.colorScheme.onSurface,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.width(76.dp)
                        )
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .height(30.dp)
                                .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                                .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.52f))
                        ) {
                            Box(
                                modifier = Modifier
                                    .fillMaxHeight()
                                    .fillMaxWidth(fraction)
                                    .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                                    .background(
                                        Brush.horizontalGradient(
                                            colors = listOf(
                                                MaterialTheme.colorScheme.primary.copy(alpha = 0.86f),
                                                MaterialTheme.colorScheme.tertiary.copy(alpha = 0.78f)
                                            )
                                        )
                                    )
                            )
                        }
                        Text(
                            text = parseBoldMarkdown(point.displayValue),
                            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                            color = MaterialTheme.colorScheme.onSurface,
                            textAlign = TextAlign.End,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.width(82.dp)
                        )
                    }
                    if (index < points.lastIndex) {
                        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.36f))
                    }
                }
            }

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(7.dp)
            ) {
                peak?.let { point ->
                    ChartSummaryChip("Peak: ${point.label} ${point.displayValue}")
                }
                ChartSummaryChip("Total: ${formatChartNumber(total, currencyPrefix)}")
                if (yLabel.isNotBlank()) {
                    ChartSummaryChip(yLabel)
                }
            }
        }
    }
}

// moved from FlatSpecRenderer.kt (RenderPercentageMatrixChart)
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun RenderPercentageMatrixChart(
    model: MultiSeriesChartModel,
    modifier: Modifier = Modifier,
    landscape: Boolean = false
) {
    val palette = listOf(
        MaterialTheme.colorScheme.primary,
        MaterialTheme.colorScheme.tertiary,
        MaterialTheme.colorScheme.secondary,
        MaterialTheme.colorScheme.error.copy(alpha = 0.82f),
        MaterialTheme.colorScheme.primary.copy(alpha = 0.58f),
        MaterialTheme.colorScheme.tertiary.copy(alpha = 0.58f)
    )
    val seriesLabels = model.rows
        .flatMap { row -> row.segments.map { segment -> segment.label } }
        .distinct()
    val colorBySeries = seriesLabels.mapIndexed { index, label ->
        label to palette[index % palette.size]
    }.toMap()
    val title = if (model.percentBased) {
        "Preference distribution"
    } else {
        "Data distribution"
    }
    val subtitle = "Grouped by ${model.categoryLabel}"

    Card(
        modifier = modifier
            .fillMaxWidth()
            .semantics(mergeDescendants = true) {
                contentDescription = buildString {
                    append(title).append(". ")
                    model.rows.forEach { row ->
                        append(row.label).append(": ")
                        append(row.segments.joinToString(", ") { segment -> "${segment.label} ${segment.displayValue}" })
                        append(". ")
                    }
                }
            },
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(containerColor = genUiCardContainerColor(GenUiCardTone.Neutral)),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.22f),
                            genUiCardContainerColor(GenUiCardTone.Neutral)
                        )
                    )
                )
                .padding(horizontal = 14.dp, vertical = 14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Surface(
                    shape = RoundedCornerShape(16.dp),
                    color = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f)
                ) {
                    Text(
                        text = "Graph",
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                    )
                }
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = title,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Text(
                        text = subtitle,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(7.dp)
            ) {
                seriesLabels.forEach { label ->
                    ChartLegendChip(
                        label = label,
                        color = colorBySeries[label] ?: MaterialTheme.colorScheme.primary
                    )
                }
            }

            Column(verticalArrangement = Arrangement.spacedBy(if (landscape) 8.dp else 11.dp)) {
                model.rows.forEachIndexed { index, row ->
                    PercentageMatrixChartRow(
                        row = row,
                        colorBySeries = colorBySeries,
                        compact = !landscape
                    )
                    if (index < model.rows.lastIndex) {
                        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.22f))
                    }
                }
            }
        }
    }
}
