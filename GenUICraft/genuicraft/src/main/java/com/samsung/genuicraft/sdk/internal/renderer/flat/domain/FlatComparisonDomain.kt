package com.samsung.genuicraft.sdk.internal.renderer.flat.domain

import com.samsung.genuicraft.sdk.internal.renderer.*

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
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import coil.ImageLoader
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.samsung.genuicraft.sdk.internal.theme.GenUiTokens
import com.samsung.genuicraft.sdk.internal.theme.GenUiCardTone
import com.samsung.genuicraft.sdk.internal.theme.genUiCardContainerColor
import com.google.gson.GsonBuilder
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.samsung.genuicraft.sdk.internal.renderer.native.NativeTextFormatter
import com.samsung.genuicraft.sdk.internal.renderer.native.NativePayloadParser
import com.samsung.genuicraft.sdk.internal.renderer.native.ParsedButton
import com.samsung.genuicraft.sdk.internal.renderer.native.intents.flight.NativeFlightSemantics
import com.samsung.genuicraft.sdk.internal.renderer.native.intents.flight.NativeFlightUiRenderer
import com.samsung.genuicraft.sdk.internal.renderer.native.intents.weather.NativeWeatherSemantics
import com.samsung.genuicraft.sdk.internal.renderer.native.intents.weather.NativeWeatherUiRenderer
import com.samsung.genuicraft.sdk.internal.renderer.native.media.NativeMediaVisualUtils
import com.samsung.genuicraft.sdk.internal.renderer.native.parser.NativeSourceParsing
import com.samsung.genuicraft.sdk.internal.renderer.native.parser.NativeStructureParsing
import com.samsung.genuicraft.sdk.internal.security.SafeContentPolicy
import kotlinx.coroutines.launch
import java.util.Locale
import kotlin.math.roundToInt
import java.util.UUID
import java.net.URI
import com.samsung.genuicraft.sdk.internal.renderer.flat.parse.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.expr.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.runtime.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.*

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

internal data class EntityCardActionLink(
    val urlCell: ResponsiveTableCardCell,
    val labelCell: ResponsiveTableCardCell?,
    val safeUrl: String
) {
    val buttonLabel: String
        get() = labelCell?.value
            ?.takeIf { value -> value.isNotBlank() && !isLikelyHttpUrl(value) }
            ?: urlCell.label
}

internal data class EntityCardRowContent(
    val primaryCell: ResponsiveTableCardCell?,
    val primaryLabel: String,
    val title: String,
    val highlightCells: List<ResponsiveTableCardCell>,
    val compactBodyCells: List<ResponsiveTableCardCell>,
    val detailBodyCells: List<ResponsiveTableCardCell>,
    val metadataCells: List<ResponsiveTableCardCell>,
    val actions: List<EntityCardActionLink>
) {
    val representedSourceCells: List<ResponsiveTableCardCell>
        get() = (
            listOfNotNull(primaryCell) +
                highlightCells +
                compactBodyCells +
                detailBodyCells +
                metadataCells
            ).sortedBy { cell -> cell.index }
}

internal fun entityCardRowContent(
    headers: List<String>,
    row: List<String>,
    rowIndex: Int,
    primaryIndex: Int,
    highlightIndexes: List<Int>
): EntityCardRowContent {
    val cells = responsiveTableCardCells(headers, row)
    val primaryCell = cells.firstOrNull { cell -> cell.index == primaryIndex }
    val primaryLabel = tableHeaderLabel(headers, primaryIndex)
    val title = primaryCell?.value ?: "Item ${rowIndex + 1}"

    val urlCells = cells.filter { cell ->
        (!isActionLabelColumn(cell.label) && isUrlColumnLabel(cell.label)) || isLikelyHttpUrl(cell.value)
    }
    val actionLabelCells = cells.filter { cell -> isActionLabelColumn(cell.label) }
    val metadataIndexes = (urlCells + actionLabelCells)
        .mapTo(mutableSetOf()) { cell -> cell.index }
        .apply { remove(primaryIndex) }
    val highlightCells = cells.filter { cell ->
        cell.index != primaryIndex && cell.index !in metadataIndexes && cell.index in highlightIndexes
    }
    val bodyCells = cells.filter { cell ->
        cell.index != primaryIndex &&
            cell.index !in metadataIndexes &&
            cell.index !in highlightIndexes
    }
    val (compactBodyCells, detailBodyCells) = bodyCells.partition { cell ->
        isCompactTableBadgeValue(cell.value) && !looksLikeLongDetailHeader(cell.label)
    }
    val metadataCells = cells.filter { cell -> cell.index in metadataIndexes }

    val usedActionLabels = mutableSetOf<Int>()
    val actions = urlCells.mapNotNull { urlCell ->
        val safeUrl = SafeContentPolicy.sanitizeActionUrl(urlCell.value) ?: return@mapNotNull null
        val urlStem = entityActionColumnStem(urlCell.label)
        val labelCell = actionLabelCells.firstOrNull { candidate ->
            candidate.index !in usedActionLabels &&
                urlStem.isNotBlank() &&
                entityActionColumnStem(candidate.label) == urlStem
        } ?: actionLabelCells.firstOrNull { candidate -> candidate.index !in usedActionLabels }
        labelCell?.let { usedActionLabels += it.index }
        EntityCardActionLink(urlCell = urlCell, labelCell = labelCell, safeUrl = safeUrl)
    }

    return EntityCardRowContent(
        primaryCell = primaryCell,
        primaryLabel = primaryLabel,
        title = title,
        highlightCells = highlightCells,
        compactBodyCells = compactBodyCells,
        detailBodyCells = detailBodyCells,
        metadataCells = metadataCells,
        actions = actions
    )
}

private fun entityActionColumnStem(label: String): String =
    normalizeTableHeaderForMatch(label)
        .split(' ')
        .filterNot { token -> token in setOf("action", "button", "cta", "label", "url", "link", "href") }
        .joinToString(" ")

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
            val content = entityCardRowContent(
                headers = headers,
                row = row,
                rowIndex = rowIndex,
                primaryIndex = primaryIndex,
                highlightIndexes = inferredHighlightIndexes
            )
            val showProviderBadge = shouldShowEntityProviderBadge(
                content.title,
                content.actions.firstOrNull()?.safeUrl
            )
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
                            EntityProviderBadge(content.title)
                        }
                        Column(
                            modifier = Modifier.weight(1f),
                            verticalArrangement = Arrangement.spacedBy(2.dp)
                        ) {
                            Text(
                                text = content.primaryLabel,
                                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                                color = MaterialTheme.colorScheme.primary
                            )
                            Text(
                                text = parseBoldMarkdown(content.title),
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                                color = MaterialTheme.colorScheme.onSurface
                            )
                        }
                    }
                    if (content.highlightCells.isNotEmpty()) {
                        FlowRow(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            content.highlightCells.forEach { cell ->
                                Surface(
                                    shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                                    color = MaterialTheme.colorScheme.primaryContainer
                                ) {
                                    Text(
                                        text = parseBoldMarkdown("${cell.label}: ${cell.value}"),
                                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                    )
                                }
                            }
                        }
                    }
                    FlowRow(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        content.compactBodyCells.forEach { cell ->
                            Surface(
                                shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                                color = MaterialTheme.colorScheme.surfaceContainerHighest
                            ) {
                                Text(
                                    text = parseBoldMarkdown("${cell.label}: ${cell.value}"),
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 5.dp)
                                )
                            }
                        }
                    }
                    content.detailBodyCells.forEach { cell ->
                        ResponsiveFieldBlock(
                            label = cell.label,
                            value = cell.value,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                    content.metadataCells.forEach { cell ->
                        ResponsiveFieldBlock(
                            label = cell.label,
                            value = cell.value,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                    content.actions.forEach { action ->
                        Button(
                            onClick = { onOpenUrl(action.safeUrl) },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(action.buttonLabel)
                        }
                    }
                }
            }
        }
    }
}
