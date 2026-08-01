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
import com.samsung.genuicraft.renderer.flat.compose.*
import com.samsung.genuicraft.renderer.flat.compose.table.*

// moved from FlatSpecRenderer.kt (looksLikeTravelItineraryTable)
internal fun looksLikeTravelItineraryTable(headers: List<String>): Boolean {
    if (headers.size < 3) return false
    val normalized = headers.map(::normalizeTableHeaderForMatch)
    val first = normalized.firstOrNull().orEmpty()
    val second = normalized.getOrNull(1).orEmpty()
    val hasDayColumn = first.contains("day") || first.contains("date")
    val hasTimeColumn = first.contains("time") || first.contains("slot")
    val hasDateColumn = normalized.any { it.contains("date") }
    val hasAreaColumn = normalized.any { header ->
        header.contains("area") ||
            header.contains("focus") ||
            header == "title" ||
            header.contains("route") ||
            header.contains("district") ||
            header.contains("neighborhood") ||
            header.contains("location")
    }
    val hasRoadTripColumn = normalized.any { header ->
        header.contains("driving") ||
            header.contains("drive") ||
            header.contains("miles") ||
            header.contains("scenic") ||
            header.contains("hike") ||
            header.contains("overnight") ||
            header.contains("stay")
    }
    val hasTimeActivityDetailsShape = hasTimeColumn &&
        (second.contains("activity") || second.contains("stop") || second.contains("place")) &&
        normalized.drop(2).any { header ->
            header.contains("detail") ||
                header.contains("location") ||
                header.contains("note") ||
                header.contains("plan")
        }
    val activityColumns = normalized.drop(1).count { header ->
        header.contains("activity") ||
            header.contains("morning") ||
            header.contains("afternoon") ||
            header.contains("evening") ||
            header.contains("stop") ||
            header.contains("plan")
    }
    val hasDiningColumn = normalized.any { header ->
        header.contains("dining") ||
            header.contains("food") ||
            header.contains("meal") ||
            header.contains("restaurant")
    }
    val hasSummaryItineraryShape = hasDayColumn &&
        hasDateColumn &&
        hasAreaColumn &&
        normalized.any { header ->
            header.contains("activity") ||
                header.contains("plan") ||
                header.contains("highlight")
        }
    return hasTimeActivityDetailsShape ||
        hasSummaryItineraryShape ||
        (hasDayColumn && hasRoadTripColumn) ||
        (hasDayColumn && (activityColumns >= 2 || hasDiningColumn))
}

// moved from FlatSpecRenderer.kt (RenderTravelItineraryTable)
@Composable
internal fun RenderTravelItineraryTable(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    title: String? = null,
    onOpenUrl: (String) -> Unit
) {
    val shownRows = rows.filter { row -> row.any { it.trim().isNotBlank() } }
    if (shownRows.isEmpty()) return
    val placeStopMode = looksLikePlaceStopItineraryTable(headers, shownRows)
    val groups = shownRows
        .mapIndexed { index, row -> itineraryDayToken(headers, row, index) to row }
        .groupBy({ it.first }, { it.second })
    val itineraryImages = allPhotoLikeItineraryImages(headers, shownRows)
    val heroImages = selectItineraryHeroImages(headers, shownRows, groups)
    val heroArea = shownRows.firstNotNullOfOrNull { row ->
        itineraryAreaTitle(headers, row, "").takeIf { it.isNotBlank() }
    }
    PrefetchFlatSpecImages(itineraryImages.take(12))
    Column(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = "Vacation itinerary with ${groups.size} days and ${shownRows.size} entries"
            },
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        TravelItineraryHero(
            title = itineraryHeroTitle(title, shownRows),
            subtitle = itineraryHeroSubtitle(groups.size.coerceAtLeast(1), shownRows.size),
            days = groups.size,
            stops = shownRows.size,
            imageUrls = heroImages,
            area = heroArea
        )
        TravelItineraryDayChipRow(groups.keys.toList())
        groups.entries.forEachIndexed { index, entry ->
            TravelItineraryNativeDayCard(
                dayIndex = index,
                day = entry.key,
                headers = headers,
                rows = entry.value,
                placeStopMode = placeStopMode,
                onOpenUrl = onOpenUrl
            )
        }
    }
}

// moved from FlatSpecRenderer.kt (RenderGeneratedItineraryDayVisual)
@Composable
internal fun RenderGeneratedItineraryDayVisual(
    day: String,
    area: String?,
    modifier: Modifier = Modifier
) {
    val title = area?.trim()?.takeIf { it.isNotBlank() } ?: day.ifBlank { "Itinerary" }
    val palette = itineraryVisualPalette("$day|$title")
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(16.dp))
            .background(Brush.linearGradient(listOf(palette.first, palette.second)))
            .accessibilitySemantics(
                props = emptyMap(),
                fallbackLabel = "$title itinerary visual",
                mergeDescendants = true
            ),
        contentAlignment = Alignment.Center
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.radialGradient(
                        colors = listOf(Color.White.copy(alpha = 0.26f), Color.Transparent)
                    )
                )
        )
        Text(
            text = day.take(18).ifBlank { "Trip" },
            color = Color.White.copy(alpha = 0.18f),
            style = MaterialTheme.typography.displaySmall.copy(fontWeight = FontWeight.Black),
            modifier = Modifier.align(Alignment.BottomEnd).padding(end = 14.dp, bottom = 8.dp),
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(18.dp),
            horizontalAlignment = Alignment.Start,
            verticalArrangement = Arrangement.Center
        ) {
            Surface(
                shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                color = Color.White.copy(alpha = 0.20f)
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Icon(
                        imageVector = Icons.Filled.Image,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(15.dp)
                    )
                    Text(
                        text = "Trip visual",
                        color = Color.White,
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold)
                    )
                }
            }
            Spacer(Modifier.height(12.dp))
            Text(
                text = title,
                color = Color.White,
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                text = day,
                color = Color.White.copy(alpha = 0.82f),
                style = MaterialTheme.typography.labelMedium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

// moved from FlatSpecRenderer.kt (TravelItineraryDayCard)
@Composable
internal fun TravelItineraryDayCard(
    headers: List<String>,
    row: List<String>
) {
    val cellCount = maxOf(headers.size, row.size)
    val cells = (0 until cellCount).map { index ->
        Triple(index, tableHeaderLabel(headers, index), row.getOrNull(index).orEmpty().trim())
    }.filter { (_, _, value) -> value.isNotBlank() }
    if (cells.isEmpty()) return
    val day = row.getOrNull(0).orEmpty().trim().ifBlank { "Day" }
    val imageCell = cells.firstOrNull { (_, label, _) -> isImageColumnLabel(label) }
    val imageAltCell = cells.firstOrNull { (_, label, _) -> isImageAltColumnLabel(label) }
    val iconCell = cells.firstOrNull { (_, label, _) -> isIconColumnLabel(label) }
    val rawImageUrl = imageCell?.third.orEmpty()
    val imageUrl = rawImageUrl.takeIf(::isPhotoLikeMediaUrl).orEmpty()
    val iconUrl = iconCell?.third?.takeIf { it.isNotBlank() }
        ?: rawImageUrl.takeIf(::isIconLikeMediaUrl)
    val dateCell = cells.firstOrNull { (index, label, _) ->
        index != 0 && normalizeTableHeaderForMatch(label).contains("date")
    }
    val areaCell = cells.firstOrNull { (index, label, _) ->
        index != 0 &&
            normalizeTableHeaderForMatch(label).let { normalized ->
                normalized.contains("area") ||
                    normalized.contains("focus") ||
                    normalized == "title" ||
                    normalized.contains("district") ||
                    normalized.contains("neighborhood") ||
                    normalized.contains("location")
            }
    }
    val excludedIndexes = setOfNotNull(
        0,
        imageCell?.first,
        imageAltCell?.first,
        iconCell?.first,
        dateCell?.first,
        areaCell?.first
    )
    val sections = cells.mapNotNull { (index, label, value) ->
        if (index in excludedIndexes) null else label to value
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp)
            .semantics(mergeDescendants = true) {
                contentDescription = tableRowAccessibilitySummary(headers, row)
            },
        shape = RoundedCornerShape(20.dp),
        colors = flatSpecCardColors(),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 13.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            if (imageUrl.isNotBlank()) {
                RenderImage(
                    props = mapOf(
                        "url" to imageUrl,
                        "fit" to "cover",
                        "height" to 132,
                        "alt" to imageAltCell?.third.orEmpty().ifBlank { day }
                    ),
                    onOpenUrl = {},
                    modifier = Modifier.fillMaxWidth()
                )
            } else {
                RenderGeneratedItineraryDayVisual(
                    day = day,
                    area = areaCell?.third,
                    modifier = Modifier.fillMaxWidth().height(132.dp)
                )
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                iconUrl?.let { icon ->
                    RenderIcon(
                        props = mapOf("name" to icon, "size" to "sm", "decorative" to true),
                        modifier = Modifier
                    )
                }
                Surface(
                    shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                    color = MaterialTheme.colorScheme.primaryContainer
                ) {
                    Text(
                        text = parseBoldMarkdown(day),
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                        modifier = Modifier.padding(horizontal = 11.dp, vertical = 5.dp)
                    )
                }
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(2.dp)
                ) {
                    dateCell?.third?.takeIf { it.isNotBlank() }?.let { date ->
                        Text(
                            text = parseBoldMarkdown(date),
                            style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                    areaCell?.third?.takeIf { it.isNotBlank() }?.let { area ->
                        Text(
                            text = parseBoldMarkdown(area),
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
            }
            sections.forEach { (label, value) ->
                ItinerarySectionBlock(label = label, value = value)
            }
        }
    }
}

// moved from FlatSpecRenderer.kt (ResponsiveScheduleRowCard)
@Composable
internal fun ResponsiveScheduleRowCard(
    headers: List<String>,
    row: List<String>
) {
    if (looksLikeTravelItineraryTable(headers)) {
        TravelItineraryDayCard(headers = headers, row = row)
        return
    }
    if (looksLikeStudyPlanTable(headers)) {
        StudyPlanWeekCard(headers = headers, row = row)
        return
    }

    val cellCount = maxOf(headers.size, row.size)
    val cells = (0 until cellCount).map { index ->
        Triple(index, tableHeaderLabel(headers, index), row.getOrNull(index).orEmpty().trim())
    }.filter { (_, _, value) -> value.isNotBlank() }
    if (cells.isEmpty()) return
    val iconCell = cells.firstOrNull { (_, label, value) ->
        isIconColumnLabel(label) && value.isNotBlank()
    }
    val contentCells = cells.filterNot { (index, _, _) -> index == iconCell?.first }
    if (contentCells.isEmpty()) return

    val promoteSecond = contentCells.size > 1 &&
        shouldPromoteTimelineSecondTitle(contentCells[0].second, contentCells[1].second)
    val titleCellIndex = if (promoteSecond) 1 else 0
    val badgeCell = if (promoteSecond) contentCells.firstOrNull() else null
    val titleCell = contentCells.getOrNull(titleCellIndex) ?: contentCells.first()
    val titleValue = formatTimelineTitle(titleCell.second, titleCell.third)
    val bodyCells = contentCells.filterNot { (index, _, _) ->
        index == titleCell.first || index == badgeCell?.first
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp)
            .semantics(mergeDescendants = true) {
                contentDescription = tableRowAccessibilitySummary(headers, row)
            },
        shape = RoundedCornerShape(16.dp),
        colors = flatSpecCardColors(),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        border = flatSpecCardBorder()
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            val badgeValue = badgeCell?.third.orEmpty()
            val titleIsLong = titleValue.length > 64 || titleValue.contains('\n')
            if (badgeValue.isNotBlank() && titleIsLong) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    if (iconCell != null) {
                        RenderIcon(
                            props = mapOf("name" to iconCell.third, "size" to "sm", "decorative" to true),
                            modifier = Modifier
                        )
                    }
                    Surface(
                        shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                        color = MaterialTheme.colorScheme.primaryContainer
                    ) {
                        Text(
                            text = parseBoldMarkdown(badgeValue),
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                            color = MaterialTheme.colorScheme.onPrimaryContainer,
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                        )
                    }
                }
                Text(
                    text = parseBoldMarkdown(titleValue),
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurface
                )
            } else {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalAlignment = Alignment.Top
                ) {
                    if (iconCell != null) {
                        RenderIcon(
                            props = mapOf("name" to iconCell.third, "size" to "sm", "decorative" to true),
                            modifier = Modifier
                        )
                    }
                    if (badgeValue.isNotBlank()) {
                        Surface(
                            shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                            color = MaterialTheme.colorScheme.primaryContainer
                        ) {
                            Text(
                                text = parseBoldMarkdown(badgeValue),
                                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                                color = MaterialTheme.colorScheme.onPrimaryContainer,
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                            )
                        }
                    }
                    Text(
                        text = parseBoldMarkdown(titleValue),
                        style = MaterialTheme.typography.titleSmall,
                        color = MaterialTheme.colorScheme.onSurface,
                        modifier = Modifier.weight(1f)
                    )
                }
            }

            bodyCells.forEach { (_, label, value) ->
                ScheduleDetailBlock(label = label, value = value)
            }
        }
    }
}

// moved from FlatSpecRenderer.kt (looksLikeStudyPlanTable)
internal fun looksLikeStudyPlanTable(headers: List<String>): Boolean {
    val normalized = headers.map(::normalizeTableHeaderForMatch)
    val hasWeek = normalized.any { it == "week" || it.contains("week") }
    val hasStudyFocus = normalized.any { header ->
        header.contains("topic") ||
            header.contains("focus") ||
            header.contains("session") ||
            header.contains("study") ||
            header.contains("practice")
    }
    val hasGoalOrAssessment = normalized.any { header ->
        header.contains("goal") ||
            header.contains("quiz") ||
            header.contains("test") ||
            header.contains("review")
    }
    val hasScheduleSignal = normalized.any { header ->
        header.contains("date") ||
            header.contains("day") ||
            header.contains("saturday") ||
            header.contains("duration")
    }
    return hasWeek && hasScheduleSignal && (hasStudyFocus || hasGoalOrAssessment)
}

// moved from FlatSpecRenderer.kt (StudyPlanWeekCard)
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun StudyPlanWeekCard(
    headers: List<String>,
    row: List<String>
) {
    val weekIndex = findTableColumnIndex(headers, listOf("week")) ?: 0
    val dateIndex = findTableColumnIndex(headers, listOf("date", "dates", "range"), exclude = setOf(weekIndex))
    val focusIndex = findTableColumnIndex(
        headers = headers,
        keywords = listOf("focus", "topic", "module", "subject"),
        exclude = setOf(weekIndex, dateIndex ?: -1)
    )
    val title = focusIndex?.let { row.getOrNull(it).orEmpty().trim() }
        ?.takeIf { it.isNotBlank() }
        ?: row.firstOrNull { it.trim().isNotBlank() }.orEmpty().trim().ifBlank { "Study week" }
    val rawWeek = row.getOrNull(weekIndex).orEmpty().trim()
    val weekLabel = when {
        rawWeek.isBlank() -> "Week"
        normalizeTableHeaderForMatch(rawWeek).contains("week") -> rawWeek
        else -> "Week $rawWeek"
    }
    val dateValue = dateIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty()
    val excluded = setOfNotNull(weekIndex, dateIndex, focusIndex)
    val detailCells = headers.indices.mapNotNull { index ->
        val value = row.getOrNull(index).orEmpty().trim()
        if (index in excluded || value.isBlank()) null else tableHeaderLabel(headers, index) to value
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp)
            .semantics(mergeDescendants = true) {
                contentDescription = tableRowAccessibilitySummary(headers, row)
            },
        shape = RoundedCornerShape(18.dp),
        colors = flatSpecCardColors(),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        border = flatSpecCardBorder()
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(9.dp)
        ) {
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Surface(
                    shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                    color = MaterialTheme.colorScheme.primaryContainer
                ) {
                    Text(
                        text = parseBoldMarkdown(weekLabel),
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                        modifier = Modifier.padding(horizontal = 11.dp, vertical = 5.dp)
                    )
                }
                if (dateValue.isNotBlank()) {
                    Surface(
                        shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                        color = MaterialTheme.colorScheme.surfaceContainerHighest
                    ) {
                        Text(
                            text = parseBoldMarkdown(dateValue),
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp)
                        )
                    }
                }
            }
            Text(
                text = parseBoldMarkdown(title),
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                color = MaterialTheme.colorScheme.onSurface
            )
            detailCells.forEach { (label, value) ->
                ScheduleDetailBlock(label = label, value = value)
            }
        }
    }
}
