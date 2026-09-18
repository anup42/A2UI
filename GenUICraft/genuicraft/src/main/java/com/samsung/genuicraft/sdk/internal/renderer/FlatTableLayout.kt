package com.samsung.genuicraft.sdk.internal.renderer

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
import androidx.compose.foundation.layout.BoxWithConstraints
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
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
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
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
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
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.*

// moved from FlatSpecRenderer.kt (RenderTableLayout)
@Composable
internal fun RenderTableLayout(
    tableModel: FlatTableModel,
    elementId: String,
    props: Map<String, Any?>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    onAction: (Any?, RepeatScope?) -> Int,
    activePath: Set<String>,
    modifier: Modifier = Modifier
) {
    val configuration = LocalConfiguration.current
    val screenWidthDp = configuration.screenWidthDp
    val isLandscape = configuration.orientation == Configuration.ORIENTATION_LANDSCAPE
    val compactScreen = isFlatCompactScreenWidth(screenWidthDp)
    val gap = stackGap(props)
    val tableModifier = applyStackModifier(modifier, props, "vertical")
    val spacing = if (gap > 0.dp) gap else 8.dp
    val bodyElement = tableModel.bodyContainerId?.let(elements::get)
    val repeatedRowScopes = buildRepeatScopes(bodyElement?.repeat, state).orEmpty()
    val tableRows = collectResolvedTableRows(
        tableModel = tableModel,
        elements = elements,
        state = state,
        repeatedRowScopes = repeatedRowScopes,
        repeatScope = repeatScope
    )
    val sourceLinks = if (shouldBypassSourceLinkIntercept(tableModel.renderMode)) {
        emptyList()
    } else {
        collectFlatSourceLinksFromRows(tableModel.headers, tableRows)
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
    val weatherRows = NativeWeatherSemantics.buildWeatherRows(tableModel.headers, tableRows)
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
    val currentWeatherRows = NativeWeatherSemantics.buildCurrentWeatherRowsFromKeyValueTable(tableModel.headers, tableRows)
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
    val cardsRequested =
        tableModel.renderMode == FlatTableRenderMode.WEATHER_CARDS ||
            tableModel.renderMode == FlatTableRenderMode.FLIGHT_CARDS ||
            tableModel.renderMode == FlatTableRenderMode.BOOKING_CARDS ||
            tableModel.renderMode == FlatTableRenderMode.RESTAURANT_CARDS ||
            tableModel.renderMode == FlatTableRenderMode.NEWS_CARDS
    val autoHorizontalScroll = shouldUseHorizontalTableScroll(
        compactScreen = compactScreen,
        screenWidthDp = screenWidthDp,
        headers = tableModel.headers,
        rows = tableRows
    )

    if (tableModel.shape == FlatTableShape.KEY_VALUE &&
        looksLikeTravelItinerarySummaryTable(tableModel.headers, tableRows)
    ) {
        return
    }

    if (tableModel.shape == FlatTableShape.KEY_VALUE && tableRows.isNotEmpty()) {
        RenderKeyValueTablePanel(
            headers = tableModel.headers,
            rows = tableRows,
            title = tableModel.title,
            modifier = tableModifier
        )
        return
    }

    if (looksLikeTravelItineraryTable(tableModel.headers) ||
        looksLikePlaceStopItineraryTable(tableModel.headers, tableRows)
    ) {
        RenderTravelItineraryTable(
            headers = tableModel.headers,
            rows = tableRows,
            modifier = tableModifier,
            title = tableModel.title,
            onOpenUrl = onOpenUrl
        )
        return
    }

    if (tableModel.renderMode == FlatTableRenderMode.RESTAURANT_CARDS) {
        val rendered = renderRestaurantRowsIfPossible(
            headers = tableModel.headers,
            rows = tableRows,
            onOpenUrl = onOpenUrl,
            modifier = tableModifier,
            title = tableModel.title
        )
        if (rendered) {
            return
        }
    }

    if (tableModel.renderMode == FlatTableRenderMode.NEWS_CARDS) {
        val rendered = renderNewsRowsIfPossible(
            headers = tableModel.headers,
            rows = tableRows,
            onOpenUrl = onOpenUrl,
            modifier = tableModifier,
            title = tableModel.title
        )
        if (rendered) {
            return
        }
    }

    if (shouldUseScrollableNativeTableGrid(tableModel.renderMode, tableRows.isNotEmpty())) {
        val horizontalScrollEnabled = nativeTableShouldScroll(
            compactScreen = compactScreen,
            screenWidthDp = screenWidthDp,
            headers = tableModel.headers,
            rows = tableRows
        )
        val columns = directColumnsFromHeaders(tableModel.headers)
        RenderAdaptiveTableGrid(
            headers = tableModel.headers,
            rows = tableRows,
            modifier = tableModifier,
            horizontalScrollEnabled = horizontalScrollEnabled,
            stickyFirstColumn = nativeTableStickyFirstColumn(tableModel.headers, horizontalScrollEnabled),
            numericColumns = numericColumnIndexes(columns, tableRows, emptySet())
        )
        return
    }

    if (tableModel.renderMode == FlatTableRenderMode.PROCESS_CARDS) {
        RenderProcessStateTable(
            headers = tableModel.headers,
            rows = tableRows,
            modifier = tableModifier,
            landscape = isLandscape || screenWidthDp >= 600
        )
        return
    }
    if (looksLikeIncidentStatusTable(tableModel.headers, tableRows, tableModel.domain)) {
        RenderIncidentStatusDashboard(
            headers = tableModel.headers,
            rows = tableRows,
            modifier = tableModifier,
            landscape = isLandscape || screenWidthDp >= 600
        )
        return
    }
    if (looksLikeMarketHoldingsTable(tableModel.headers, tableRows, tableModel.domain)) {
        RenderMarketHoldingsTable(
            headers = tableModel.headers,
            rows = tableRows,
            modifier = tableModifier
        )
        return
    }

    if (looksLikeClimateComparisonTable(tableModel.headers, tableRows, tableModel.domain)) {
        RenderClimateComparisonCards(
            headers = tableModel.headers,
            rows = tableRows,
            modifier = tableModifier,
            spacing = spacing,
            landscape = isLandscape || screenWidthDp >= 600
        )
        return
    }
    if (tableModel.renderMode == FlatTableRenderMode.FLIGHT_CARDS) {
        if (looksLikeMultiLegFlightTable(tableModel.headers)) {
            RenderFlightItineraryTableCards(
                headers = tableModel.headers,
                rows = tableRows,
                modifier = tableModifier,
                spacing = spacing
            )
            return
        }
        val flightRows = NativeFlightSemantics.buildFlightRows(tableModel.headers, tableRows)
        if (!flightRows.isNullOrEmpty()) {
            NativeFlightUiRenderer.RenderFlightRows(flightRows, onOpenUrl = onOpenUrl)
            return
        }
    }
    if (tableModel.renderMode == FlatTableRenderMode.BOOKING_CARDS) {
        val rendered = renderBookingRowsIfPossible(
            headers = tableModel.headers,
            rows = tableRows,
            onOpenUrl = onOpenUrl
        )
        if (rendered) {
            return
        }
    }
    val effectiveRenderMode = when {
        tableModel.renderMode == FlatTableRenderMode.RESPONSIVE_CARD_ROWS -> FlatTableRenderMode.RESPONSIVE_CARD_ROWS
        cardsRequested && autoHorizontalScroll && tableModel.columns <= 3 -> FlatTableRenderMode.RESPONSIVE_CARD_ROWS
        cardsRequested && autoHorizontalScroll -> FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL
        cardsRequested -> FlatTableRenderMode.TABLE
        tableModel.renderMode == FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL -> FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL
        autoHorizontalScroll && tableModel.columns <= 3 -> FlatTableRenderMode.RESPONSIVE_CARD_ROWS
        autoHorizontalScroll -> FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL
        else -> FlatTableRenderMode.TABLE
    }

    Column(
        modifier = tableModifier,
        verticalArrangement = Arrangement.spacedBy(spacing)
    ) {
        if (effectiveRenderMode == FlatTableRenderMode.TABLE ||
            effectiveRenderMode == FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL
        ) {
            val horizontalScrollEnabled = effectiveRenderMode == FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL
            val contentModifier = if (horizontalScrollEnabled) {
                Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
            } else {
                Modifier.fillMaxWidth()
            }
            val minTableWidth = estimateTableMinWidthDp(
                headers = tableModel.headers,
                rows = tableRows,
                baseMinDp = if (cardsRequested) 128 else 120
            ).dp
            Box(
                modifier = contentModifier.semantics {
                    contentDescription = tableAccessibilitySummary(
                        headers = tableModel.headers,
                        rows = tableRows,
                        horizontalScroll = horizontalScrollEnabled
                    )
                }
            ) {
                Column(
                    modifier = if (horizontalScrollEnabled) {
                        Modifier.widthIn(min = minTableWidth)
                    } else {
                        Modifier.fillMaxWidth()
                    },
                    verticalArrangement = Arrangement.spacedBy(spacing)
                ) {
                    RenderElement(
                        elementId = tableModel.headerRowId,
                        elements = elements,
                        state = state,
                        repeatScope = repeatScope,
                        onOpenUrl = onOpenUrl,
                        onSetState = onSetState,
                        onAction = onAction,
                        activePath = activePath + elementId
                    )
                    if (tableModel.rowTemplateId != null && repeatedRowScopes.isNotEmpty()) {
                        repeatedRowScopes.forEachIndexed { index, scopedRepeat ->
                            RenderElement(
                                elementId = tableModel.rowTemplateId,
                                elements = elements,
                                state = state,
                                repeatScope = scopedRepeat,
                                onOpenUrl = onOpenUrl,
                                onSetState = onSetState,
                                onAction = onAction,
                                activePath = activePath + elementId
                            )
                            if (index < repeatedRowScopes.lastIndex) {
                                HorizontalDivider(
                                    color = MaterialTheme.colorScheme.outlineVariant,
                                    modifier = Modifier.padding(horizontal = 8.dp)
                                )
                            }
                        }
                    } else {
                        tableModel.staticRowIds.forEachIndexed { index, rowId ->
                            RenderElement(
                                elementId = rowId,
                                elements = elements,
                                state = state,
                                repeatScope = repeatScope,
                                onOpenUrl = onOpenUrl,
                                onSetState = onSetState,
                                onAction = onAction,
                                activePath = activePath + elementId
                            )
                            if (index < tableModel.staticRowIds.lastIndex) {
                                HorizontalDivider(
                                    color = MaterialTheme.colorScheme.outlineVariant,
                                    modifier = Modifier.padding(horizontal = 8.dp)
                                )
                            }
                        }
                    }
                }
            }
            return@Column
        }

        if (tableRows.isNotEmpty()) {
            RenderResponsiveTableRows(
                headers = tableModel.headers,
                rows = tableRows,
                modifier = Modifier.fillMaxWidth(),
                spacing = spacing,
                shape = tableModel.shape
            )
            return@Column
        }

        if (tableModel.rowTemplateId != null && repeatedRowScopes.isNotEmpty()) {
            repeatedRowScopes.forEachIndexed { index, scopedRepeat ->
                RenderResponsiveTableRowCard(
                    rowId = tableModel.rowTemplateId,
                    rowScope = scopedRepeat,
                    rowIndex = index,
                    headers = tableModel.headers,
                    elements = elements,
                    state = state,
                    onOpenUrl = onOpenUrl,
                    onSetState = onSetState,
                    onAction = onAction,
                    activePath = activePath + elementId
                )
            }
        } else {
            tableModel.staticRowIds.forEachIndexed { index, rowId ->
                RenderResponsiveTableRowCard(
                    rowId = rowId,
                    rowScope = repeatScope,
                    rowIndex = index,
                    headers = tableModel.headers,
                    elements = elements,
                    state = state,
                    onOpenUrl = onOpenUrl,
                    onSetState = onSetState,
                    onAction = onAction,
                    activePath = activePath + elementId
                )
            }
        }
    }
}

// moved from FlatSpecRenderer.kt (RenderAdaptiveTableGrid)
@Composable
internal fun RenderAdaptiveTableGrid(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    horizontalScrollEnabled: Boolean,
    stickyFirstColumn: Boolean,
    numericColumns: Set<Int>
) {
    if (headers.isEmpty() || rows.isEmpty()) return
    val estimatedWidths = estimateTableColumnMinWidthsDp(
        headers = headers,
        rows = rows,
        baseMinDp = if (horizontalScrollEnabled) 120 else 96
    )
    val gridNumericColumns = tableGridNumericColumns(rows, numericColumns)
    val tableColor = MaterialTheme.colorScheme.surfaceContainerLow
    val scrollState = rememberScrollState()
    BoxWithConstraints(modifier = modifier.fillMaxWidth()) {
        val availableWidthDp = maxWidth.value.toInt().coerceAtLeast(1)
        val columnMinWidthsDp = tableViewportColumnWidthsDp(estimatedWidths, availableWidthDp, stickyFirstColumn)
        val minTableWidth = columnMinWidthsDp.sum().dp
        val scrollViewportDp = availableWidthDp - if (stickyFirstColumn) columnMinWidthsDp.first() else 0
        if (horizontalScrollEnabled && stickyFirstColumn && headers.size > 1) {
            val density = LocalDensity.current
            val trailingWidthsPx = remember(columnMinWidthsDp, density) {
                with(density) { columnMinWidthsDp.drop(1).map { it.dp.roundToPx() } }
            }
            LaunchedEffect(scrollState, trailingWidthsPx) {
                snapshotFlow { Triple(scrollState.isScrollInProgress, scrollState.value, scrollState.maxValue) }
                    .collect { (scrolling, offset, maximum) ->
                        if (scrolling || maximum <= 0 || maximum == Int.MAX_VALUE) return@collect
                        val boundaries = tableColumnSnapOffsetsPx(trailingWidthsPx, maximum)
                        val target = nearestTableColumnSnapOffsetPx(offset, boundaries)
                        if (target == offset) return@collect
                        // Let drag-to-fling and accessibility/programmatic scroll handoffs finish.
                        withFrameNanos { }
                        if (scrollState.isScrollInProgress || scrollState.value != offset ||
                            scrollState.maxValue != maximum) return@collect
                        try {
                            // One shared state moves header and every body row together. The target
                            // is itself a boundary, so the resulting idle emission cannot re-snap.
                            scrollState.scrollTo(target)
                        } catch (cancelled: CancellationException) {
                            // A new gesture may supersede settling; keep observing its completion.
                            if (!currentCoroutineContext().isActive) throw cancelled
                        }
                    }
            }
        }
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics {
                        contentDescription = tableAccessibilitySummary(
                            headers = headers,
                            rows = rows,
                            horizontalScroll = horizontalScrollEnabled
                        )
                    },
                shape = RoundedCornerShape(16.dp),
                color = tableColor
            ) {
                Box(
                    modifier = Modifier.fillMaxWidth().drawWithContent {
                        drawContent()
                        if (horizontalScrollEnabled && scrollState.canScrollForward) {
                            // Decoration must not participate in table measurement or touch handling.
                            val fadeWidth = 8.dp.toPx().coerceAtMost(size.width)
                            val fadeStart = size.width - fadeWidth
                            drawRect(
                                brush = Brush.horizontalGradient(
                                    colors = listOf(Color.Transparent, tableColor.copy(alpha = 0.65f)),
                                    startX = fadeStart,
                                    endX = size.width,
                                ),
                                topLeft = Offset(fadeStart, 0f),
                                size = Size(fadeWidth, size.height),
                            )
                        }
                    },
                ) {
                    if (stickyFirstColumn && headers.size > 1) {
                        RenderStickyFirstColumnTable(
                            headers = headers,
                            rows = rows,
                            columnMinWidthsDp = columnMinWidthsDp,
                            scrollState = scrollState,
                            numericColumns = gridNumericColumns
                        )
                    } else {
                        val contentModifier = if (horizontalScrollEnabled) {
                            Modifier
                                .horizontalScroll(scrollState)
                                .widthIn(min = minTableWidth)
                        } else {
                            Modifier.fillMaxWidth()
                        }
                        Column(
                            modifier = contentModifier.padding(vertical = 4.dp),
                            verticalArrangement = Arrangement.spacedBy(0.dp)
                        ) {
                            RenderTableGridRow(
                                headers = headers,
                                row = headers,
                                rowIndex = -1,
                                columnMinWidthsDp = columnMinWidthsDp,
                                numericColumns = gridNumericColumns,
                                weighted = !horizontalScrollEnabled,
                                isHeader = true
                            )
                            rows.forEachIndexed { index, row ->
                                RenderTableGridRow(
                                    headers = headers,
                                    row = row,
                                    rowIndex = index,
                                    columnMinWidthsDp = columnMinWidthsDp,
                                    numericColumns = gridNumericColumns,
                                    weighted = !horizontalScrollEnabled,
                                    isHeader = false
                                )
                            }
                        }
                    }
                }
            }
            if (horizontalScrollEnabled && scrollState.maxValue > 0) {
                val position = scrollState.value.toFloat() / scrollState.maxValue
                val trackColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.08f)
                val thumbColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.55f)
                Column(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 4.dp)
                        .semantics(mergeDescendants = true) {
                            contentDescription = "Table with ${headers.size} columns. Scroll horizontally."
                            stateDescription = "${(position * 100).roundToInt()} percent across"
                        },
                    verticalArrangement = Arrangement.spacedBy(3.dp)
                ) {
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Scroll horizontally", style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text("${headers.size} columns", style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    Box(modifier = Modifier.fillMaxWidth().height(3.dp).clip(RoundedCornerShape(2.dp))
                        .drawWithContent {
                            drawRect(trackColor)
                            val viewportPx = scrollViewportDp.dp.toPx()
                            val fraction = viewportPx / (viewportPx + scrollState.maxValue)
                            val thumbWidth = (size.width * fraction).coerceAtLeast(20.dp.toPx()).coerceAtMost(size.width)
                            drawRect(thumbColor, topLeft = Offset((size.width - thumbWidth) * position, 0f),
                                size = Size(thumbWidth, size.height))
                        })
                }
            }
        }
    }
}
