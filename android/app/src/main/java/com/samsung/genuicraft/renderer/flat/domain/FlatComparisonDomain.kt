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

// moved from FlatSpecRenderer.kt (RenderFeatureMatrixEntityCards)
@Composable
internal fun RenderFeatureMatrixEntityCards(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    spacing: Dp = 8.dp,
    columns: List<FlatDirectTableColumn> = emptyList(),
    entityMedia: Map<String, TableEntityMedia> = emptyMap()
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        verticalArrangement = Arrangement.spacedBy(spacing)
    ) {
        ResponsiveComparisonColumnCards(
            headers = headers,
            rows = rows,
            spacing = spacing,
            columns = columns,
            entityMedia = entityMedia
        )
    }
}

// moved from FlatSpecRenderer.kt (RenderEntityTableCards)
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun RenderEntityTableCards(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    spacing: Dp = 8.dp,
    primaryColumn: String?,
    highlightColumns: Set<String>,
    onOpenUrl: (String) -> Unit
) {
    if (rows.isEmpty()) return
    val primaryIndex = inferEntityPrimaryColumnIndex(headers, primaryColumn)
    val explicitHighlightIndexes = highlightColumns.mapNotNull { token ->
        headers.indexOfFirst { header -> normalizeColumnToken(header) == normalizeColumnToken(token) }
            .takeIf { it >= 0 }
    }.filterNot { it == primaryIndex || looksLikeLongDetailHeader(headers.getOrNull(it).orEmpty()) }
        .filter { index -> isCompactHighlightColumn(rows, index) }
    val inferredHighlightIndexes = explicitHighlightIndexes.ifEmpty {
        headers.indices
            .filterNot { it == primaryIndex }
            .filter { index ->
                val header = normalizeTableHeaderForMatch(headers[index])
                header.contains("price") ||
                    header.contains("cost") ||
                    header.contains("fare") ||
                    header.contains("rating") ||
                    header.contains("status") ||
                    header.contains("date") ||
                    header.contains("time")
            }
            .filter { index -> isCompactHighlightColumn(rows, index) }
            .take(2)
            .ifEmpty {
                headers.indices
                    .filterNot { it == primaryIndex || looksLikeLongDetailHeader(headers.getOrNull(it).orEmpty()) }
                    .filter { index -> isCompactHighlightColumn(rows, index) }
                    .take(2)
            }
    }.take(2)
    Column(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        verticalArrangement = Arrangement.spacedBy(spacing)
    ) {
        rows.forEachIndexed { rowIndex, row ->
            val title = row.getOrNull(primaryIndex).orEmpty().trim().ifBlank { "Item ${rowIndex + 1}" }
            val actionUrlIndex = headers.indices.firstOrNull { index ->
                isUrlColumnLabel(headers[index]) && SafeContentPolicy.isSafeActionUrl(row.getOrNull(index).orEmpty().trim())
            } ?: row.indexOfFirst { value -> SafeContentPolicy.isSafeActionUrl(value.trim()) }.takeIf { it >= 0 }
            val actionUrl = actionUrlIndex?.let { SafeContentPolicy.sanitizeActionUrl(row.getOrNull(it).orEmpty().trim()) }
            val actionLabel = entityActionLabel(headers, row, title)
            val showProviderBadge = shouldShowEntityProviderBadge(title, actionUrl)
            val bodyIndexes = headers.indices.filterNot { index ->
                val value = row.getOrNull(index).orEmpty().trim()
                index == primaryIndex ||
                    index == actionUrlIndex ||
                    index in inferredHighlightIndexes ||
                    value.isBlank() ||
                    isLikelyHttpUrl(value) ||
                    isUrlColumnLabel(headers[index]) ||
                    isActionLabelColumn(headers[index])
            }
            val (shortBodyIndexes, detailBodyIndexes) = bodyIndexes.partition { index ->
                val value = row.getOrNull(index).orEmpty()
                isCompactTableBadgeValue(value) && !looksLikeLongDetailHeader(headers.getOrNull(index).orEmpty())
            }
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics(mergeDescendants = true) {
                        contentDescription = tableRowAccessibilitySummary(headers, row, rowIndex)
                    },
                shape = RoundedCornerShape(16.dp),
                colors = flatSpecCardColors(),
                elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
            ) {
                Column(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 10.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        if (showProviderBadge) {
                            EntityProviderBadge(title)
                        }
                        Column(
                            modifier = Modifier.weight(1f),
                            verticalArrangement = Arrangement.spacedBy(2.dp)
                        ) {
                            Text(
                                text = parseBoldMarkdown(title),
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                                color = MaterialTheme.colorScheme.onSurface,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis
                            )
                            actionUrl?.let {
                                Text(
                                    text = "Tap to open related search",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                        }
                    }
                    if (inferredHighlightIndexes.isNotEmpty()) {
                        FlowRow(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            inferredHighlightIndexes.forEach { index ->
                                val value = row.getOrNull(index).orEmpty().trim()
                                if (value.isNotBlank()) {
                                    Surface(
                                        shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                                        color = MaterialTheme.colorScheme.primaryContainer
                                    ) {
                                        Text(
                                            text = parseBoldMarkdown("${tableHeaderLabel(headers, index)}: $value"),
                                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                                            color = MaterialTheme.colorScheme.onPrimaryContainer,
                                            maxLines = 2,
                                            overflow = TextOverflow.Ellipsis,
                                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                        )
                                    }
                                }
                            }
                        }
                    }
                    FlowRow(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        shortBodyIndexes.take(6).forEach { index ->
                            val value = row.getOrNull(index).orEmpty().trim()
                            if (value.isBlank() || isLikelyHttpUrl(value)) return@forEach
                            Surface(
                                shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                                color = MaterialTheme.colorScheme.surfaceContainerHighest
                            ) {
                                Text(
                                    text = parseBoldMarkdown("${tableHeaderLabel(headers, index)}: $value"),
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 5.dp)
                                )
                            }
                        }
                    }
                    detailBodyIndexes.take(3).forEach { index ->
                        ResponsiveFieldBlock(
                            label = tableHeaderLabel(headers, index),
                            value = row.getOrNull(index).orEmpty(),
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                    if (!actionUrl.isNullOrBlank()) {
                        Button(
                            onClick = { onOpenUrl(actionUrl) },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(actionLabel)
                        }
                    }
                }
            }
        }
    }
}
