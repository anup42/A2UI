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

// moved from FlatSpecRenderer.kt (looksLikeRankedFlightComparisonTable)
internal fun looksLikeRankedFlightComparisonTable(headers: List<String>): Boolean {
    val normalized = headers.map(::normalizeTableHeaderForMatch)
    val hasAirline = normalized.any { it.contains("airline") || it.contains("carrier") }
    val hasRank = normalized.any { it.contains("rank") || it.contains("score") || it.contains("order") }
    val hasCost = normalized.any { it.contains("cost") || it.contains("fare") || it.contains("price") }
    val hasDuration = normalized.any { it.contains("time") || it.contains("duration") || it.contains("travel") }
    val hasStops = normalized.any { it.contains("layover") || it.contains("stop") || it.contains("connection") }
    return hasAirline && (hasRank || hasCost) && (hasDuration || hasStops)
}

// moved from FlatSpecRenderer.kt (looksLikeMultiLegFlightTable)
internal fun looksLikeMultiLegFlightTable(headers: List<String>): Boolean {
    val normalized = headers.map(::normalizeTableHeaderForMatch)
    val legSignals = normalized.count { header ->
        header.contains("leg") ||
            Regex("""\b[a-z]{3}\s*[/-]\s*[a-z]{3}\b""").containsMatchIn(header)
    }
    val carrierSignals = normalized.count { header ->
        header.contains("carrier") || header.contains("airline")
    }
    return legSignals >= 2 && carrierSignals >= 1
}

// moved from FlatSpecRenderer.kt (RenderRankedFlightComparisonCards)
@Composable
internal fun RenderRankedFlightComparisonCards(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    spacing: Dp = 10.dp,
    onOpenUrl: (String) -> Unit = {}
) {
    if (rows.isEmpty()) return
    val rankIndex = rankedFlightColumnIndex(headers, listOf("rank", "order", "score"))
    val airlineIndex = rankedFlightColumnIndex(headers, listOf("airline", "carrier")) ?: 0
    val costIndex = rankedFlightColumnIndex(headers, listOf("cost", "fare", "price", "amount"))
    val durationIndex = rankedFlightColumnIndex(headers, listOf("travel time", "duration", "time"))
    val stopsIndex = rankedFlightColumnIndex(headers, listOf("stops", "stop"))
    val layoverIndex = rankedFlightColumnIndex(headers, listOf("layover", "connection"))
    val reasonIndex = rankedFlightColumnIndex(headers, listOf("justification", "reason", "why", "notes", "detail"))
    val statusIndex = rankedFlightColumnIndex(headers, listOf("status", "type"))
    val departureIndex = rankedFlightColumnIndex(headers, listOf("departure", "depart"))
    val arrivalIndex = rankedFlightColumnIndex(headers, listOf("arrival", "arrive"))
    val legsIndex = rankedFlightColumnIndex(headers, listOf("legs", "segments", "itinerary"))
    val carbonIndex = rankedFlightColumnIndex(headers, listOf("carbon", "emission", "co2"))
    val bookingStatusIndex = rankedFlightColumnIndex(headers, listOf("booking"))
    val logoIndex = rankedFlightColumnIndex(headers, listOf("airline logo", "logo", "icon", "image"))
    Column(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        verticalArrangement = Arrangement.spacedBy(spacing)
    ) {
        rows.forEachIndexed { rowIndex, row ->
            val rank = compactRankBadge(
                rawRank = row.getOrNull(rankIndex ?: -1).orEmpty().trim(),
                fallbackIndex = rowIndex
            )
            val airlineRaw = row.getOrNull(airlineIndex).orEmpty().trim().ifBlank { "Flight option ${rowIndex + 1}" }
            val airline = airlineRaw.substringBefore(" - ").trim().ifBlank { airlineRaw }
            val flightNumbers = airlineRaw.substringAfter(" - ", missingDelimiterValue = "").trim()
            val cost = row.getOrNull(costIndex ?: -1).orEmpty().trim()
            val duration = row.getOrNull(durationIndex ?: -1).orEmpty().trim()
            val stops = row.getOrNull(stopsIndex ?: -1).orEmpty().trim()
            val layover = row.getOrNull(layoverIndex ?: -1).orEmpty().trim()
            val reason = row.getOrNull(reasonIndex ?: -1).orEmpty().trim()
            val status = row.getOrNull(statusIndex ?: -1).orEmpty().trim()
            val departure = parseRankedFlightEndpoint(row.getOrNull(departureIndex ?: -1).orEmpty())
            val arrival = parseRankedFlightEndpoint(row.getOrNull(arrivalIndex ?: -1).orEmpty())
            val legs = parseRankedFlightLegs(row.getOrNull(legsIndex ?: -1).orEmpty())
            val carbon = row.getOrNull(carbonIndex ?: -1).orEmpty().trim()
            val bookingStatus = row.getOrNull(bookingStatusIndex ?: -1).orEmpty().trim()
            val logoUrl = row.getOrNull(logoIndex ?: -1).orEmpty().trim()
            val actionUrlIndex = headers.indices.firstOrNull { index ->
                isFlightActionUrlColumn(headers[index]) &&
                    SafeContentPolicy.isSafeActionUrl(row.getOrNull(index).orEmpty().trim())
            } ?: headers.indices.firstOrNull { index ->
                isUrlColumnLabel(headers[index]) &&
                    SafeContentPolicy.isSafeActionUrl(row.getOrNull(index).orEmpty().trim())
            }
            val actionUrl = actionUrlIndex?.let { index ->
                SafeContentPolicy.sanitizeActionUrl(row.getOrNull(index).orEmpty().trim())
            }
            val actionLabelIndex = headers.indices.firstOrNull { index ->
                isActionLabelColumn(headers[index]) && index != actionUrlIndex
            }
            val actionLabel = row.getOrNull(actionLabelIndex ?: -1)
                .orEmpty()
                .trim()
                .takeIf { it.isNotBlank() && !SafeContentPolicy.looksLikeUrl(it) }
                ?: "View fare"
            val best = rowIndex == 0 || rank == "#1"
            val accent = rankedFlightAccentColor(airline)
            val normalizedDuration = NativeFlightSemantics.normalizeDurationLabel(duration) ?: duration
            val normalizedStop = NativeFlightSemantics.canonicalizeStopLabel(stops) ?: stops
            val (fareValue, fareMeta) = NativeFlightSemantics.splitFareDisplay(cost)
            val subtitle = rankedFlightSubtitle(status, flightNumbers, normalizedStop, normalizedDuration, best)
            val chips = rankedFlightChips(
                stopLabel = normalizedStop,
                layover = layover,
                carbon = carbon,
                bookingStatus = bookingStatus
            )
            val cardShape = RoundedCornerShape(20.dp)
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics(mergeDescendants = true) {
                        contentDescription = tableRowAccessibilitySummary(headers, row, rowIndex)
                    },
                shape = cardShape,
                color = genUiCardContainerColor(GenUiCardTone.Neutral),
                tonalElevation = 0.dp,
                shadowElevation = 0.dp,
                border = BorderStroke(
                    width = GenUiTokens.BorderSm,
                    color = if (best) {
                        accent.copy(alpha = 0.55f)
                    } else {
                        MaterialTheme.colorScheme.outlineVariant
                    }
                )
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 11.dp),
                    verticalArrangement = Arrangement.spacedBy(9.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        RankedFlightAirlineBadge(
                            airline = airline,
                            rank = rank,
                            best = best,
                            logoUrl = logoUrl
                        )
                        Column(
                            modifier = Modifier.weight(1f),
                            verticalArrangement = Arrangement.spacedBy(3.dp)
                        ) {
                            Text(
                                text = airline,
                                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                                color = MaterialTheme.colorScheme.onSurface,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis
                            )
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(6.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                RankedFlightBestChip(if (best) "Best value" else rank, accent)
                            }
                            if (subtitle.isNotBlank()) {
                                Text(
                                    text = subtitle,
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                        }
                        if (cost.isNotBlank()) {
                            Column(
                                modifier = Modifier.widthIn(min = 78.dp, max = 116.dp),
                                horizontalAlignment = Alignment.End
                            ) {
                                Text(
                                    text = fareValue ?: cost,
                                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                                    color = MaterialTheme.colorScheme.onSurface,
                                    textAlign = TextAlign.End,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                                fareMeta?.let { suffix ->
                                    Text(
                                        text = suffix,
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        textAlign = TextAlign.End
                                    )
                                }
                            }
                        }
                    }

                    if (departure.code != null || departure.time != null || arrival.code != null || arrival.time != null) {
                        RankedFlightRouteLine(
                            departure = departure,
                            arrival = arrival,
                            accent = accent
                        )
                    } else {
                        RankedFlightRouteHint(
                            duration = normalizedDuration.takeIf { it.isNotBlank() },
                            stopLabel = normalizedStop.takeIf { it.isNotBlank() },
                            accent = accent
                        )
                    }

                    RankedFlightLegRows(legs = legs, accent = accent)
                    RankedFlightChipRow(chips = chips, accent = accent)

                    if (reason.isNotBlank()) {
                        Surface(
                            shape = RoundedCornerShape(14.dp),
                            color = MaterialTheme.colorScheme.surfaceContainerHighest
                        ) {
                            Text(
                                text = parseBoldMarkdown(reason),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                maxLines = 3,
                                overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp)
                            )
                        }
                    }
                    actionUrl?.let { safeUrl ->
                        Button(
                            onClick = { onOpenUrl(safeUrl) },
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(GenUiTokens.RadiusPill)
                        ) {
                            Icon(
                                imageVector = Icons.Filled.FlightTakeoff,
                                contentDescription = null,
                                modifier = Modifier.size(16.dp)
                            )
                            Text(
                                text = actionLabel,
                                modifier = Modifier.padding(start = 8.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}

// moved from FlatSpecRenderer.kt (RenderFlightItineraryTableCards)
@Composable
internal fun RenderFlightItineraryTableCards(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    spacing: Dp = 10.dp
) {
    if (rows.isEmpty()) return
    val carrierIndex = flightCarrierColumnIndex(headers)
    val legIndexes = flightLegColumnIndexes(headers)
    if (legIndexes.isEmpty()) return
    val detailIndexes = headers.indices.filterNot { index ->
        index == carrierIndex || index in legIndexes
    }
    Column(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        verticalArrangement = Arrangement.spacedBy(spacing)
    ) {
        rows.forEachIndexed { rowIndex, row ->
            val carrier = row.getOrNull(carrierIndex).orEmpty().trim().ifBlank { "Flight option ${rowIndex + 1}" }
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics(mergeDescendants = true) {
                        contentDescription = tableRowAccessibilitySummary(headers, row, rowIndex)
                    },
                shape = RoundedCornerShape(GenUiTokens.RadiusLg),
                colors = flatSpecCardColors(),
                elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 10.dp),
                    verticalArrangement = Arrangement.spacedBy(9.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Surface(
                            shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                            color = MaterialTheme.colorScheme.primaryContainer
                        ) {
                            Icon(
                                imageVector = Icons.Filled.FlightTakeoff,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.onPrimaryContainer,
                                modifier = Modifier.padding(8.dp).size(20.dp)
                            )
                        }
                        Column(
                            modifier = Modifier.weight(1f),
                            verticalArrangement = Arrangement.spacedBy(2.dp)
                        ) {
                            Text(
                                text = parseBoldMarkdown(carrier),
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            Text(
                                text = "Multi-city itinerary",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }

                    legIndexes.forEachIndexed { legOrder, columnIndex ->
                        val legCarrier = row.getOrNull(columnIndex).orEmpty().trim()
                        if (legCarrier.isBlank()) return@forEachIndexed
                        FlightLegTimelineRow(
                            badge = flightLegBadge(headers.getOrNull(columnIndex).orEmpty(), legOrder),
                            route = flightLegRoute(headers.getOrNull(columnIndex).orEmpty()),
                            carrier = legCarrier
                        )
                    }

                    detailIndexes.forEach { index ->
                        val value = row.getOrNull(index).orEmpty().trim()
                        if (value.isBlank()) return@forEach
                        FeatureMatrixField(
                            feature = tableHeaderLabel(headers, index),
                            value = value
                        )
                    }
                }
            }
        }
    }
}
