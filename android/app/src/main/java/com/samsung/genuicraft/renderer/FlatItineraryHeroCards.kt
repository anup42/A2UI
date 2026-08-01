package com.samsung.genuicraft.renderer

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
import com.samsung.genuicraft.renderer.flat.compose.*
import com.samsung.genuicraft.renderer.flat.compose.table.*

// moved from FlatSpecRenderer.kt (TravelItineraryHero)
@Composable
internal fun TravelItineraryHero(
    title: String,
    subtitle: String,
    days: Int,
    stops: Int,
    imageUrls: List<String>,
    area: String?
) {
    val shape = RoundedCornerShape(28.dp)
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(250.dp)
            .clip(shape)
            .background(
                Brush.linearGradient(
                    listOf(
                        Color(0xFF123B8E),
                        Color(0xFF1E5BC6),
                        Color(0xFF0F766E)
                    )
                )
            )
    ) {
        if (imageUrls.isNotEmpty()) {
            RenderImage(
                props = mapOf(
                    "url" to imageUrls.first(),
                    "fallbackUrls" to imageUrls.drop(1),
                    "fit" to "cover",
                    "height" to 250,
                    "alt" to title
                ),
                onOpenUrl = {},
                modifier = Modifier.fillMaxWidth()
            )
            Box(
                modifier = Modifier
                    .matchParentSize()
                    .background(
                        Brush.verticalGradient(
                            listOf(
                                Color(0xFF071833).copy(alpha = 0.68f),
                                Color(0xFF071833).copy(alpha = 0.46f),
                                Color(0xFF071833).copy(alpha = 0.88f)
                            )
                        )
                    )
            )
        } else {
            Box(
                modifier = Modifier
                    .matchParentSize()
                    .background(
                        Brush.radialGradient(
                            listOf(Color.White.copy(alpha = 0.26f), Color.Transparent)
                        )
                    )
            )
        }
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(18.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            Surface(
                shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                color = Color.White.copy(alpha = 0.17f),
                border = BorderStroke(1.dp, Color.White.copy(alpha = 0.16f))
            ) {
                Text(
                    text = area?.takeIf { it.isNotBlank() } ?: "MCP Places itinerary",
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                    color = Color.White,
                    modifier = Modifier.padding(horizontal = 11.dp, vertical = 6.dp),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
            Column(verticalArrangement = Arrangement.spacedBy(11.dp)) {
                Text(
                    text = parseBoldMarkdown(title),
                    style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Black),
                    color = Color.White,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
                Text(
                    text = parseBoldMarkdown(subtitle),
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color.White.copy(alpha = 0.88f),
                    maxLines = 3,
                    overflow = TextOverflow.Ellipsis
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(9.dp)
                ) {
                    TravelItineraryHeroStat("$days", "days", Modifier.weight(1f))
                    TravelItineraryHeroStat("$stops", "stops", Modifier.weight(1f))
                    TravelItineraryHeroStat("Maps", "actions", Modifier.weight(1f))
                }
            }
        }
    }
}

// moved from FlatSpecRenderer.kt (TravelItineraryNativeDayCard)
@Composable
internal fun TravelItineraryNativeDayCard(
    dayIndex: Int,
    day: String,
    headers: List<String>,
    rows: List<List<String>>,
    placeStopMode: Boolean,
    onOpenUrl: (String) -> Unit
) {
    val firstRow = rows.firstOrNull().orEmpty()
    val area = itineraryAreaTitle(headers, firstRow, day)
    val imageUrl = rows.firstNotNullOfOrNull { row ->
        firstPhotoLikeItineraryImage(headers, row).takeIf { it.isNotBlank() }
    }.orEmpty()
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics(mergeDescendants = true) {
                contentDescription = "Day ${dayIndex + 1}: $day, ${rows.size} itinerary entries"
            },
        shape = RoundedCornerShape(26.dp),
        colors = flatSpecCardColors(),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
        border = flatSpecCardBorder()
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(56.dp)
                        .clip(RoundedCornerShape(18.dp))
                        .background(
                            Brush.linearGradient(
                                listOf(MaterialTheme.colorScheme.primary, MaterialTheme.colorScheme.tertiary)
                            )
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "${dayIndex + 1}",
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Black),
                        color = MaterialTheme.colorScheme.onPrimary
                    )
                }
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(3.dp)
                ) {
                    Text(
                        text = parseBoldMarkdown(area.ifBlank { day }),
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Black),
                        color = MaterialTheme.colorScheme.onSurface,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis
                    )
                    Text(
                        text = NativeTextFormatter.sanitizeDisplayText("$day · ${rows.size} ${if (rows.size == 1) "stop" else "stops"}"),
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            if (!placeStopMode) {
                if (imageUrl.isNotBlank()) {
                    RenderImage(
                        props = mapOf(
                            "url" to imageUrl,
                            "fit" to "cover",
                            "aspectRatio" to 1.78f,
                            "alt" to itineraryImageAlt(headers, firstRow, area)
                        ),
                        onOpenUrl = onOpenUrl,
                        modifier = Modifier.fillMaxWidth()
                    )
                } else {
                    RenderGeneratedItineraryDayVisual(
                        day = day,
                        area = area,
                        modifier = Modifier.fillMaxWidth().height(136.dp)
                    )
                }
                TravelItineraryDaySections(headers = headers, row = firstRow, onOpenUrl = onOpenUrl)
            } else {
                TravelItineraryDayOverview(
                    headers = headers,
                    rows = rows,
                    day = day,
                    area = area
                )
                rows.forEachIndexed { stopIndex, row ->
                    TravelItineraryStopCard(
                        headers = headers,
                        row = row,
                        stopIndex = stopIndex,
                        fallbackDay = day,
                        onOpenUrl = onOpenUrl
                    )
                }
            }
        }
    }
}

// moved from FlatSpecRenderer.kt (TravelItineraryStopCard)
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun TravelItineraryStopCard(
    headers: List<String>,
    row: List<String>,
    stopIndex: Int,
    fallbackDay: String,
    onOpenUrl: (String) -> Unit
) {
    val title = itineraryCell(headers, row, ::isItineraryPlaceLabel)
        .ifBlank { itineraryAreaTitle(headers, row, fallbackDay) }
        .ifBlank { "Stop ${stopIndex + 1}" }
    val imageUrl = firstPhotoLikeItineraryImage(headers, row)
    val actions = collectItineraryActions(headers, row)
    val area = itineraryCell(headers, row, ::isItineraryAreaLabel)
    val fallbackDetail = buildList {
        area.takeIf { it.isNotBlank() && it != title && it != fallbackDay }?.let(::add)
        itineraryCell(headers, row, ::isItineraryCategoryLabel)
            .takeIf { it.isNotBlank() && it != title }
            ?.let(::add)
    }.distinct().joinToString(" - ")
    val metricIndexes = headers.indices.filter { index ->
        val label = tableHeaderLabel(headers, index)
        val value = row.getOrNull(index).orEmpty().trim()
        value.isNotBlank() && isItineraryMetricLabel(label) && !isLikelyHttpUrl(value)
    }.take(3)
    val bodyIndexes = headers.indices.filter { index ->
        val label = tableHeaderLabel(headers, index)
        val value = row.getOrNull(index).orEmpty().trim()
        val normalized = normalizeTableHeaderForMatch(label)
        value.isNotBlank() &&
            !isLikelyHttpUrl(value) &&
            !isImageColumnLabel(label) &&
            !isImageAltColumnLabel(label) &&
            !isIconColumnLabel(label) &&
            !isItineraryActionUrlLabel(label) &&
            !isUrlColumnLabel(label) &&
            !isItineraryMetricLabel(label) &&
            !isItineraryPlaceLabel(label) &&
            !isItineraryAreaLabel(label) &&
            !normalized.contains("day") &&
            !normalized.contains("date")
    }.take(2)
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(MaterialTheme.colorScheme.surfaceContainerHighest.copy(alpha = 0.46f))
            .padding(10.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.Top
    ) {
        if (imageUrl.isNotBlank()) {
            RenderImage(
                props = mapOf(
                    "url" to imageUrl,
                    "fit" to "cover",
                    "width" to 112,
                    "height" to 118,
                    "alt" to itineraryImageAlt(headers, row, title)
                ),
                onOpenUrl = onOpenUrl,
                modifier = Modifier
            )
        } else {
            RenderGeneratedItineraryDayVisual(
                day = "${stopIndex + 1}",
                area = title,
                modifier = Modifier.size(112.dp, 118.dp)
            )
        }
        Column(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(7.dp)
        ) {
            Text(
                text = parseBoldMarkdown(title),
                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Black),
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
            if (metricIndexes.isNotEmpty()) {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(5.dp)
                ) {
                    metricIndexes.forEach { index ->
                        TravelItineraryMetricChip(
                            label = tableHeaderLabel(headers, index),
                            value = row.getOrNull(index).orEmpty()
                        )
                    }
                }
            }
            bodyIndexes.forEach { index ->
                Text(
                    text = parseBoldMarkdown(row.getOrNull(index).orEmpty()),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 3,
                    overflow = TextOverflow.Ellipsis
                )
            }
            if (bodyIndexes.isEmpty() && fallbackDetail.isNotBlank()) {
                Text(
                    text = parseBoldMarkdown(NativeTextFormatter.sanitizeDisplayText(fallbackDetail)),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 3,
                    overflow = TextOverflow.Ellipsis
                )
            }
            if (actions.isNotEmpty()) {
                TravelItineraryActionRow(actions = actions, onOpenUrl = onOpenUrl)
            }
        }
    }
}
