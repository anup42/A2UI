package com.samsung.genuicraft.renderer.flat.domain

import com.samsung.genuicraft.renderer.*

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
import androidx.compose.material.icons.filled.Language
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
import com.samsung.genuicraft.renderer.flat.parse.*
import com.samsung.genuicraft.renderer.flat.expr.*
import com.samsung.genuicraft.renderer.flat.runtime.*
import com.samsung.genuicraft.renderer.flat.compose.table.*

// moved from FlatSpecRenderer.kt (ClimateComparisonRow)
internal data class ClimateComparisonRow(
    val place: String,
    val verdict: String?,
    val high: String?,
    val low: String?,
    val metrics: List<Pair<String, String>>,
    val sourceRow: List<String>
)


// moved from FlatSpecRenderer.kt (inferClimateCondition)
internal fun inferClimateCondition(row: ClimateComparisonRow): String {
    val textPool = buildString {
        append(row.verdict.orEmpty())
        row.metrics.forEach { (_, value) ->
            append(' ')
            append(value)
        }
    }.lowercase()
    val sunshine = row.metrics.firstOrNull { (label, _) ->
        normalizeTableHeaderForMatch(label).contains("sun")
    }?.second?.let(::firstNumberFromText)
    val rain = row.metrics.firstOrNull { (label, _) ->
        normalizeTableHeaderForMatch(label).contains("rain")
    }?.second?.let(::firstNumberFromText)
    return when {
        textPool.contains("storm") -> "Thunderstorm"
        textPool.contains("snow") -> "Snow"
        textPool.contains("rain") && textPool.contains("heavy") -> "Rain"
        rain != null && rain >= 12 -> "Rain"
        sunshine != null && sunshine >= 7 -> "Sunny"
        textPool.contains("sun") || textPool.contains("recommended") -> "Sunny"
        textPool.contains("dry") || textPool.contains("mild") -> "Partly Cloudy"
        else -> "Cloudy"
    }
}

// moved from FlatSpecRenderer.kt (RenderClimateComparisonCards)
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun RenderClimateComparisonCards(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    spacing: Dp = 8.dp,
    landscape: Boolean = false
) {
    val climateRows = buildClimateComparisonRows(headers, rows)
    if (climateRows.isEmpty()) return

    Column(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        verticalArrangement = Arrangement.spacedBy(spacing)
    ) {
        if (landscape && climateRows.size == 2) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(spacing)
            ) {
                climateRows.forEachIndexed { index, row ->
                    ClimateComparisonCard(
                        row = row,
                        rowIndex = index,
                        highlighted = isPreferredClimateRow(row),
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        } else {
            climateRows.forEachIndexed { index, row ->
                ClimateComparisonCard(
                    row = row,
                    rowIndex = index,
                    highlighted = isPreferredClimateRow(row),
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }
    }
}

// moved from FlatSpecRenderer.kt (isPreferredClimateRow)
internal fun isPreferredClimateRow(row: ClimateComparisonRow): Boolean {
    val text = "${row.place} ${row.verdict.orEmpty()} ${row.metrics.joinToString(" ") { it.second }}".lowercase()
    return text.contains("recommended") ||
        text.contains("pick") ||
        text.contains("best") ||
        text.contains("more sun") ||
        text.contains("warmer")
}

// moved from FlatSpecRenderer.kt (ClimateComparisonCard)
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun ClimateComparisonCard(
    row: ClimateComparisonRow,
    rowIndex: Int,
    highlighted: Boolean,
    modifier: Modifier = Modifier
) {
    val condition = inferClimateCondition(row)
    val background = if (highlighted) {
        Brush.linearGradient(
            colors = listOf(
                MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.92f),
                MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.78f)
            )
        )
    } else {
        Brush.linearGradient(
            colors = listOf(
                MaterialTheme.colorScheme.surfaceContainerHigh,
                MaterialTheme.colorScheme.surfaceContainerHighest.copy(alpha = 0.72f)
            )
        )
    }
    val chipColor = if (highlighted) {
        MaterialTheme.colorScheme.surface.copy(alpha = 0.58f)
    } else {
        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.62f)
    }
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(22.dp))
            .background(background)
            .semantics(mergeDescendants = true) {
                contentDescription = tableRowAccessibilitySummary(
                    headers = listOf("Place", "Verdict", "High", "Low") + row.metrics.map { it.first },
                    row = listOf(row.place, row.verdict.orEmpty(), row.high.orEmpty(), row.low.orEmpty()) +
                        row.metrics.map { it.second },
                    rowIndex = rowIndex
                )
            }
            .padding(horizontal = 14.dp, vertical = 13.dp)
    ) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                NativeWeatherUiRenderer.WeatherConditionIcon(
                    condition = condition,
                    size = 34.dp,
                    sanitizeDisplayText = NativeTextFormatter::sanitizeDisplayText
                )
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(3.dp)
                ) {
                    Text(
                        text = parseBoldMarkdown(row.place),
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    row.verdict?.let { verdict ->
                        Text(
                            text = parseBoldMarkdown(verdict),
                            style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold),
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.Bottom
            ) {
                row.high?.let { high ->
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(2.dp)
                    ) {
                        Text(
                            text = "High",
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Text(
                            text = parseBoldMarkdown(high),
                            style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Bold),
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
                row.low?.let { low ->
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(2.dp)
                    ) {
                        Text(
                            text = "Low",
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Text(
                            text = parseBoldMarkdown(low),
                            style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
            }

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(7.dp)
            ) {
                row.metrics.forEach { (label, value) ->
                    Surface(
                        shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                        color = chipColor
                    ) {
                        Text(
                            text = parseBoldMarkdown("$label: $value"),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.padding(horizontal = 9.dp, vertical = 5.dp)
                        )
                    }
                }
            }
        }
    }
}
