package com.samsung.genuicraft.sdk.internal.renderer.flat.compose

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
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.parse.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.expr.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.runtime.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.legacy.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.*

// moved from FlatSpecRenderer.kt (extractDirectTableModel)
internal fun hasExplicitTablePresentation(props: Map<String, Any?>): Boolean =
    props["preferredPresentation"]?.toString()?.trim()?.equals("table", ignoreCase = true) == true

internal fun extractDirectTableModel(
    props: Map<String, Any?>,
    state: Map<String, Any?>,
    compactScreen: Boolean
): FlatDirectTableModel? {
    val rows = resolveDirectTableRows(props, state)
    val columns = resolveDirectTableColumns(props, rows)
    if (columns.size < 2) return null

    val resolvedRows = rows.map { row -> resolveDirectTableRow(row, columns, state) }
    val headerLabels = columns.map { column -> column.label }
    val explicitDomain = normalizeExplicitTableDomain(props["domain"]?.toString())
        ?.takeIf { it in SUPPORTED_TABLE_DOMAINS }
    val inferredDomain = inferTableDomainFromHeaders(headerLabels)
    val domain = when {
        explicitDomain != null -> explicitDomain
        else -> inferredDomain
    }
    val explicitPreferredPresentation = props["preferredPresentation"]?.toString()?.trim()?.lowercase()
        ?.takeIf { it in setOf("cards", "table") }
    val preferredPresentation = when {
        explicitPreferredPresentation == null -> if (domain in CARD_FIRST_TABLE_DOMAINS) "cards" else "table"
        else -> explicitPreferredPresentation
    }
    val shape = detectTableShape(headerLabels, resolvedRows, domain)
    val primaryColumn = props["primaryColumn"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
    val highlightColumns = stringSetFromTableProp(props["highlightColumns"])
    val numericColumns = stringSetFromTableProp(props["numericColumns"])
    val entityMedia = extractTableEntityMedia(props)
    val renderMode = when {
        explicitPreferredPresentation == "table" -> if (compactScreen || columns.size >= 3) {
            FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL
        } else {
            FlatTableRenderMode.TABLE
        }
        looksLikeProcessStateTable(headerLabels, resolvedRows, domain) -> FlatTableRenderMode.PROCESS_CARDS
        domain == "weather" -> FlatTableRenderMode.WEATHER_CARDS
        domain == "flight" -> FlatTableRenderMode.FLIGHT_CARDS
        domain == "booking" -> FlatTableRenderMode.BOOKING_CARDS
        domain == "restaurants" -> FlatTableRenderMode.RESTAURANT_CARDS
        domain == "news" -> FlatTableRenderMode.NEWS_CARDS
        domain == "playlist" -> FlatTableRenderMode.PLAYLIST_CARDS
        domain == "product" -> FlatTableRenderMode.PRODUCT_CARDS
        compactScreen && shape in setOf(
            FlatTableShape.PLAYLIST,
            FlatTableShape.ENTITY_ROW,
            FlatTableShape.FEATURE_MATRIX,
            FlatTableShape.KEY_VALUE,
            FlatTableShape.SCHEDULE_TIMELINE,
            FlatTableShape.NUMERIC_METRICS
        ) -> FlatTableRenderMode.RESPONSIVE_CARD_ROWS
        compactScreen && columns.size >= 4 -> FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL
        else -> FlatTableRenderMode.TABLE
    }
    return FlatDirectTableModel(
        columns = columns,
        rows = resolvedRows,
        domain = domain,
        preferredPresentation = preferredPresentation,
        shape = shape,
        primaryColumn = primaryColumn,
        highlightColumns = highlightColumns,
        numericColumns = numericColumns,
        entityMedia = entityMedia,
        renderMode = renderMode
    )
}

// moved from FlatSpecRenderer.kt (RenderDirectTable)
@Composable
internal fun RenderDirectTable(
    props: Map<String, Any?>,
    state: Map<String, Any?>,
    onOpenUrl: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    val configuration = LocalConfiguration.current
    val screenWidthDp = configuration.screenWidthDp
    val isLandscape = configuration.orientation == Configuration.ORIENTATION_LANDSCAPE
    val compactPortrait = screenWidthDp < 600 && !isLandscape
    val table = extractDirectTableModel(props, state, compactPortrait) ?: return
    val headers = table.columns.map { column -> column.label }
    val tableModifier = applyStackModifier(modifier, props, "vertical")
    // An explicit table request preserves header meaning and units. Heuristic card routes may
    // omit column labels or suppress summary rows, so they must not override this request.
    if (hasExplicitTablePresentation(props)) {
        val horizontalScrollEnabled = nativeTableShouldScroll(
            compactScreen = compactPortrait,
            screenWidthDp = screenWidthDp,
            headers = headers,
            rows = table.rows
        )
        Column(modifier = tableModifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
            props["title"]?.toString()?.takeIf { it.isNotBlank() }?.let { title ->
                Text(
                    text = parseBoldMarkdown(title),
                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                    modifier = Modifier.semantics { heading() }
                )
            }
            RenderAdaptiveTableGrid(
                headers = headers,
                rows = table.rows,
                horizontalScrollEnabled = horizontalScrollEnabled,
                stickyFirstColumn = nativeTableStickyFirstColumn(headers, horizontalScrollEnabled),
                numericColumns = numericColumnIndexes(table.columns, table.rows, table.numericColumns)
            )
        }
        return
    }
    val sourceLinks = if (shouldBypassSourceLinkIntercept(table.renderMode)) {
        emptyList()
    } else {
        collectFlatSourceLinksFromTableProps(props, state)
    }
    if (sourceLinks.isNotEmpty()) {
        RenderFlatSourceSection(
            section = FlatSourceSection(title = "Sources", links = sourceLinks),
            onOpenUrl = onOpenUrl,
            modifier = tableModifier,
            wrapInCard = true,
            showTitle = true
        )
        return
    }
    val isComparisonTable = table.domain == "comparison"
    val weatherRows = if (isComparisonTable) {
        null
    } else {
        NativeWeatherSemantics.buildWeatherRows(headers, table.rows)
    }
    if (!weatherRows.isNullOrEmpty()) {
        NativeWeatherUiRenderer.RenderWeatherRows(
            rows = weatherRows,
            sanitizeDisplayText = NativeTextFormatter::sanitizeDisplayText,
            weatherTemperatureText = NativeWeatherSemantics::weatherTemperatureText,
            orderWeatherRows = NativeWeatherSemantics::orderWeatherRows,
            isTodayWeatherRow = NativeWeatherSemantics::isTodayWeatherRow,
            weatherConditionIcon = { condition, size ->
                NativeWeatherUiRenderer.WeatherConditionIcon(
                    condition = condition,
                    size = size,
                    sanitizeDisplayText = NativeTextFormatter::sanitizeDisplayText
                )
            }
        )
        return
    }
    val currentWeatherRows = if (isComparisonTable) {
        null
    } else {
        NativeWeatherSemantics.buildCurrentWeatherRowsFromKeyValueTable(headers, table.rows)
    }
    if (!currentWeatherRows.isNullOrEmpty()) {
        NativeWeatherUiRenderer.RenderWeatherRows(
            rows = currentWeatherRows,
            sanitizeDisplayText = NativeTextFormatter::sanitizeDisplayText,
            weatherTemperatureText = NativeWeatherSemantics::weatherTemperatureText,
            orderWeatherRows = NativeWeatherSemantics::orderWeatherRows,
            isTodayWeatherRow = NativeWeatherSemantics::isTodayWeatherRow,
            weatherConditionIcon = { condition, size ->
                NativeWeatherUiRenderer.WeatherConditionIcon(
                    condition = condition,
                    size = size,
                    sanitizeDisplayText = NativeTextFormatter::sanitizeDisplayText
                )
            }
        )
        return
    }
    if (table.shape == FlatTableShape.KEY_VALUE && looksLikeTravelItinerarySummaryTable(headers, table.rows)) {
        return
    }

    if (table.shape == FlatTableShape.KEY_VALUE && table.rows.isNotEmpty()) {
        RenderKeyValueTablePanel(
            headers = headers,
            rows = table.rows,
            title = props["title"]?.toString(),
            modifier = tableModifier
        )
        return
    }

    if (looksLikeTravelItineraryTable(headers) || looksLikePlaceStopItineraryTable(headers, table.rows)) {
        RenderTravelItineraryTable(
            headers = headers,
            rows = table.rows,
            modifier = tableModifier,
            title = props["title"]?.toString(),
            onOpenUrl = onOpenUrl
        )
        return
    }

    if (table.renderMode == FlatTableRenderMode.RESTAURANT_CARDS) {
        val rendered = renderRestaurantRowsIfPossible(
            headers = headers,
            rows = table.rows,
            onOpenUrl = onOpenUrl,
            modifier = tableModifier,
            title = props["title"]?.toString()
        )
        if (rendered) {
            return
        }
    }

    if (table.renderMode == FlatTableRenderMode.NEWS_CARDS) {
        val rendered = renderNewsRowsIfPossible(
            headers = headers,
            rows = table.rows,
            onOpenUrl = onOpenUrl,
            modifier = tableModifier,
            title = props["title"]?.toString()
        )
        if (rendered) {
            return
        }
    }

    if (table.renderMode == FlatTableRenderMode.PRODUCT_CARDS) {
        val rendered = renderProductRowsIfPossible(
            headers = headers,
            rows = table.rows,
            onOpenUrl = onOpenUrl,
            modifier = tableModifier,
            title = props["title"]?.toString()
        )
        if (rendered) {
            return
        }
    }

    if (table.renderMode == FlatTableRenderMode.FLIGHT_CARDS && looksLikeRankedFlightComparisonTable(headers)) {
        RenderRankedFlightComparisonCards(
            headers = headers,
            rows = table.rows,
            modifier = applyStackModifier(modifier, props, "vertical"),
            spacing = stackGap(props).takeIf { it > 0.dp } ?: 10.dp,
            onOpenUrl = onOpenUrl
        )
        return
    }
    if (table.renderMode == FlatTableRenderMode.FLIGHT_CARDS && looksLikeMultiLegFlightTable(headers)) {
        RenderFlightItineraryTableCards(
            headers = headers,
            rows = table.rows,
            modifier = applyStackModifier(modifier, props, "vertical"),
            spacing = stackGap(props).takeIf { it > 0.dp } ?: 10.dp
        )
        return
    }
    if (table.renderMode == FlatTableRenderMode.FLIGHT_CARDS && shouldUseNativeFlightCards(headers)) {
        val flightRows = NativeFlightSemantics.buildFlightRows(headers, table.rows)
        if (!flightRows.isNullOrEmpty()) {
            NativeFlightUiRenderer.RenderFlightRows(flightRows, onOpenUrl = onOpenUrl)
            return
        }
    }
    if (table.renderMode == FlatTableRenderMode.BOOKING_CARDS) {
        val rendered = renderBookingRowsIfPossible(
            headers = headers,
            rows = table.rows,
            onOpenUrl = onOpenUrl
        )
        if (rendered) {
            return
        }
    }

    if (shouldUseScrollableNativeTableGrid(table.renderMode, table.rows.isNotEmpty())) {
        val horizontalScrollEnabled = nativeTableShouldScroll(
            compactScreen = compactPortrait,
            screenWidthDp = screenWidthDp,
            headers = headers,
            rows = table.rows
        )
        RenderAdaptiveTableGrid(
            headers = headers,
            rows = table.rows,
            modifier = tableModifier,
            horizontalScrollEnabled = horizontalScrollEnabled,
            stickyFirstColumn = nativeTableStickyFirstColumn(headers, horizontalScrollEnabled),
            numericColumns = numericColumnIndexes(table.columns, table.rows, table.numericColumns)
        )
        return
    }
    val cardsRequested =
        table.renderMode == FlatTableRenderMode.WEATHER_CARDS ||
            table.renderMode == FlatTableRenderMode.FLIGHT_CARDS ||
            table.renderMode == FlatTableRenderMode.BOOKING_CARDS ||
            table.renderMode == FlatTableRenderMode.RESTAURANT_CARDS ||
            table.renderMode == FlatTableRenderMode.NEWS_CARDS ||
            table.renderMode == FlatTableRenderMode.PLAYLIST_CARDS ||
            table.renderMode == FlatTableRenderMode.PRODUCT_CARDS

    if (table.renderMode == FlatTableRenderMode.PROCESS_CARDS) {
        RenderProcessStateTable(
            headers = headers,
            rows = table.rows,
            modifier = applyStackModifier(modifier, props, "vertical"),
            landscape = isLandscape || screenWidthDp >= 600
        )
        return
    }
    if (looksLikeIncidentStatusTable(headers, table.rows, table.domain)) {
        RenderIncidentStatusDashboard(
            headers = headers,
            rows = table.rows,
            modifier = applyStackModifier(modifier, props, "vertical"),
            landscape = isLandscape || screenWidthDp >= 600
        )
        return
    }
    if (looksLikeMarketHoldingsTable(headers, table.rows, table.domain)) {
        RenderMarketHoldingsTable(
            headers = headers,
            rows = table.rows,
            modifier = applyStackModifier(modifier, props, "vertical")
        )
        return
    }

    if (isFormulaVariablesTable(table)) {
        RenderFormulaVariablesTable(
            headers = headers,
            rows = table.rows,
            modifier = tableModifier
        )
        return
    }
    if (isCalculationBreakdownTable(table)) {
        RenderCalculationBreakdownTable(
            headers = headers,
            rows = table.rows,
            modifier = tableModifier
        )
        return
    }
    extractPercentageMatrixChartModel(
        columns = table.columns,
        rows = table.rows,
        domain = table.domain
    )?.let { chartModel ->
        RenderPercentageMatrixChart(
            model = chartModel,
            modifier = tableModifier,
            landscape = isLandscape || screenWidthDp >= 600
        )
        return
    }

    val spacing = 8.dp
    val autoHorizontalScroll = shouldUseHorizontalTableScroll(
        compactScreen = compactPortrait,
        screenWidthDp = screenWidthDp,
        headers = headers,
        rows = table.rows
    )
    val presentation = selectAdaptiveTablePresentation(
        table = table,
        screenWidthDp = screenWidthDp,
        isLandscape = isLandscape,
        autoHorizontalScroll = autoHorizontalScroll,
        cardsRequested = cardsRequested
    )
    when (presentation) {
        AdaptiveTablePresentation.CLIMATE_CARDS -> RenderClimateComparisonCards(
            headers = headers,
            rows = table.rows,
            modifier = tableModifier,
            spacing = spacing,
            landscape = isLandscape || screenWidthDp >= 600
        )
        AdaptiveTablePresentation.KEY_VALUE_PANEL -> RenderKeyValueTablePanel(
            headers = headers,
            rows = table.rows,
            title = props["title"]?.toString(),
            modifier = tableModifier
        )
        AdaptiveTablePresentation.ITINERARY_CARDS -> RenderTravelItineraryTable(
            headers = headers,
            rows = table.rows,
            modifier = tableModifier,
            title = props["title"]?.toString(),
            onOpenUrl = onOpenUrl
        )
        AdaptiveTablePresentation.TIMELINE_CARDS -> RenderTimelineTableCards(
            headers = headers,
            rows = table.rows,
            modifier = tableModifier,
            spacing = spacing
        )
        AdaptiveTablePresentation.FEATURE_CARDS -> RenderFeatureMatrixEntityCards(
            headers = headers,
            rows = table.rows,
            modifier = tableModifier,
            spacing = spacing,
            columns = table.columns,
            entityMedia = table.entityMedia
        )
        AdaptiveTablePresentation.ENTITY_CARDS -> RenderEntityTableCards(
            headers = headers,
            rows = table.rows,
            modifier = tableModifier,
            spacing = spacing,
            primaryColumn = table.primaryColumn,
            highlightColumns = table.highlightColumns,
            onOpenUrl = onOpenUrl
        )
        AdaptiveTablePresentation.PLAYLIST_ROWS -> RenderPlaylistTableRows(
            headers = headers,
            rows = table.rows,
            props = props,
            modifier = tableModifier,
            landscape = isLandscape || screenWidthDp >= 600
        )
        AdaptiveTablePresentation.METRIC_CARDS -> RenderMetricTableCards(
            headers = headers,
            rows = table.rows,
            modifier = tableModifier,
            spacing = spacing,
            numericColumns = numericColumnIndexes(table.columns, table.rows, table.numericColumns)
        )
        AdaptiveTablePresentation.TABLE,
        AdaptiveTablePresentation.HORIZONTAL_TABLE,
        AdaptiveTablePresentation.STICKY_HORIZONTAL_TABLE -> RenderAdaptiveTableGrid(
            headers = headers,
            rows = table.rows,
            modifier = tableModifier,
            horizontalScrollEnabled = presentation != AdaptiveTablePresentation.TABLE,
            stickyFirstColumn = presentation == AdaptiveTablePresentation.STICKY_HORIZONTAL_TABLE,
            numericColumns = numericColumnIndexes(table.columns, table.rows, table.numericColumns)
        )
    }
}
