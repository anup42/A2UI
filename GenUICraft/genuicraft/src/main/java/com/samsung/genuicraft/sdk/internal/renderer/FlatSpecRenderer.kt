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
import androidx.compose.runtime.key
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
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.capability.GeneratedRendererCapabilities
import com.samsung.genuicraft.sdk.internal.renderer.flat.legacy.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.*

internal data class FlatSourceSection(
    val title: String?,
    val links: List<ParsedButton>
)

internal data class WatchEntry(
    val elementId: String,
    val key: String,
    val statePath: String,
    val actionBinding: Any?
)

/** Carries the originating element through the legacy two-argument action callback. */
internal data class SourcedActionCandidate(
    val candidate: Any?,
    val elementId: String
)

internal enum class FlatTableRenderMode {
    TABLE,
    TABLE_HORIZONTAL_SCROLL,
    PROCESS_CARDS,
    WEATHER_CARDS,
    FLIGHT_CARDS,
    BOOKING_CARDS,
    RESTAURANT_CARDS,
    NEWS_CARDS,
    PLAYLIST_CARDS,
    PRODUCT_CARDS,
    RESPONSIVE_CARD_ROWS
}
internal fun shouldBypassSourceLinkIntercept(renderMode: FlatTableRenderMode): Boolean =
    renderMode == FlatTableRenderMode.NEWS_CARDS

internal enum class FlatTableShape {
    PLAYLIST,
    ENTITY_ROW,
    FEATURE_MATRIX,
    KEY_VALUE,
    SCHEDULE_TIMELINE,
    NUMERIC_METRICS,
    GENERIC_GRID
}

internal enum class AdaptiveTablePresentation {
    TABLE,
    HORIZONTAL_TABLE,
    STICKY_HORIZONTAL_TABLE,
    CLIMATE_CARDS,
    ENTITY_CARDS,
    PLAYLIST_ROWS,
    FEATURE_CARDS,
    KEY_VALUE_PANEL,
    ITINERARY_CARDS,
    TIMELINE_CARDS,
    METRIC_CARDS
}

internal data class ChartPoint(
    val label: String,
    val value: Double,
    val displayValue: String
)

internal data class MultiSeriesChartSegment(
    val key: String,
    val label: String,
    val value: Double,
    val displayValue: String
)

internal data class MultiSeriesChartRow(
    val label: String,
    val segments: List<MultiSeriesChartSegment>
)

internal data class MultiSeriesChartModel(
    val categoryLabel: String,
    val rows: List<MultiSeriesChartRow>,
    val percentBased: Boolean
)

internal data class FlatTableModel(
    val headerRowId: String,
    val bodyContainerId: String?,
    val rowTemplateId: String?,
    val staticRowIds: List<String>,
    val headers: List<String>,
    val columns: Int,
    val rows: Int,
    val isWeather: Boolean,
    val isFlight: Boolean,
    val domain: String,
    val preferredPresentation: String,
    val title: String?,
    val shape: FlatTableShape,
    val cardMappingStatus: String,
    val renderMode: FlatTableRenderMode
)

internal data class FlatDirectTableColumn(
    val key: String,
    val label: String
)

internal data class FlatDirectTableModel(
    val columns: List<FlatDirectTableColumn>,
    val rows: List<List<String>>,
    val domain: String,
    val preferredPresentation: String,
    val shape: FlatTableShape,
    val primaryColumn: String?,
    val highlightColumns: Set<String>,
    val numericColumns: Set<String>,
    val entityMedia: Map<String, TableEntityMedia>,
    val renderMode: FlatTableRenderMode
)

internal data class TableEntityMedia(
    val image: String,
    val alt: String
)

internal fun useScrollableNativeTableRendering(): Boolean = true

internal fun shouldUseScrollableNativeTableGrid(
    renderMode: FlatTableRenderMode,
    hasRows: Boolean
): Boolean = hasRows && useScrollableNativeTableRendering() &&
    (renderMode == FlatTableRenderMode.TABLE || renderMode == FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL)

internal fun nativeTableShouldScroll(
    compactScreen: Boolean,
    screenWidthDp: Int,
    headers: List<String>,
    rows: List<List<String>>
): Boolean {
    return compactScreen ||
        headers.size >= 3 ||
        shouldUseHorizontalTableScroll(
            compactScreen = compactScreen,
            screenWidthDp = screenWidthDp,
            headers = headers,
            rows = rows
        )
}

internal fun nativeTableStickyFirstColumn(headers: List<String>, horizontalScrollEnabled: Boolean): Boolean =
    horizontalScrollEnabled && (headers.size >= 4 || (headers.size == 3 &&
        normalizeTableHeaderForMatch(headers.first()) in setOf(
            "term", "level", "feature", "factor", "item", "name", "type", "category",
            "option", "model", "route", "restaurant", "date", "day", "chemistry"
        )))

/** Keep a complete value column readable inside the actual host width, without truncating cells. */
internal fun tableViewportColumnWidthsDp(
    estimatedWidths: List<Int>,
    availableWidthDp: Int,
    stickyFirstColumn: Boolean
): List<Int> {
    if (estimatedWidths.isEmpty()) return emptyList()
    val available = availableWidthDp.coerceAtLeast(1)
    if (!stickyFirstColumn || estimatedWidths.size == 1) {
        return estimatedWidths.map { it.coerceIn(1, available) }
    }
    val first = estimatedWidths.first().coerceIn(88, 144)
        .coerceAtMost((available * 0.34f).toInt().coerceAtLeast(1))
    val remaining = (available - first).coerceAtLeast(1)
    return listOf(first) + estimatedWidths.drop(1).map { it.coerceIn(1, remaining) }
}

/** Data-column starts plus the exact scroll end; dimensions are already rounded to pixels. */
internal fun tableColumnSnapOffsetsPx(columnWidthsPx: List<Int>, maximumOffsetPx: Int): List<Int> {
    val maximum = maximumOffsetPx.coerceAtLeast(0)
    if (columnWidthsPx.isEmpty() || maximum == 0) return listOf(0)
    val offsets = mutableListOf(0)
    var start = 0L
    columnWidthsPx.dropLast(1).forEach { width ->
        start = (start + width.coerceAtLeast(0)).coerceAtMost(maximum.toLong())
        offsets += start.toInt()
    }
    offsets += maximum
    return offsets.distinct().sorted()
}

/** Ties choose the lower boundary, so settling is deterministic and idempotent. */
internal fun nearestTableColumnSnapOffsetPx(offsetPx: Int, boundariesPx: List<Int>): Int =
    boundariesPx.minByOrNull { boundary -> kotlin.math.abs(boundary.toLong() - offsetPx.toLong()) } ?: 0

/** This classification affects grid alignment only; domain/card routing keeps its existing rules. */
internal fun tableGridNumericColumns(rows: List<List<String>>, candidates: Set<Int>): Set<Int> =
    candidates.filter { index ->
        val values = rows.mapNotNull { it.getOrNull(index)?.trim()?.takeIf(String::isNotEmpty) }
        values.isNotEmpty() && values.all(::isScalarTableGridValue)
    }.toSet()

internal fun isScalarTableGridValue(value: String): Boolean {
    val plain = value.replace(Regex("\\[\\d+(?:\\s*[,–-]\\s*\\d+)*]"), "")
        .replace(Regex("[*_`]"), "").replace('−', '-').trim()
    val number = "[+-]?\\d[\\d,]*(?:\\.\\d+)?"
    val currency = "(?:[₹$€£]|INR|USD|EUR|GBP)"
    val unit = "(?:%|x|[kKmM]|bn|°[CF]?|km|m|cm|mm|kg|g|mg|ml|[lL]|[kMGT]?B|[kMGT]?Hz|W|kW|Wh|kWh|Wh/kg|km/h|m/s|mAh|h|hrs?|hours?|mins?|minutes?|days?|years?|INR|USD|EUR|GBP)"
    return Regex("^(?:$currency\\s*)?$number(?:\\s*[–—-]\\s*(?:$currency\\s*)?$number)?\\s*$unit?$", RegexOption.IGNORE_CASE)
        .matches(plain)
}

internal fun directColumnsFromHeaders(headers: List<String>): List<FlatDirectTableColumn> =
    headers.mapIndexed { index, header ->
        FlatDirectTableColumn(
            key = normalizeTableColumnKey(header, "col_${index + 1}"),
            label = header
        )
    }

internal data class GeneratedRestaurantVisual(
    val title: String
)

typealias FlatComputedFunction = (Map<String, Any?>) -> Any?

internal val LocalFlatSpecAssetResolver = staticCompositionLocalOf<(String) -> String> { { raw -> raw } }
internal val LocalFlatSpecComputedFunctions = staticCompositionLocalOf<Map<String, FlatComputedFunction>> { emptyMap() }
internal val LocalFlatSpecTextHorizontalPadding = staticCompositionLocalOf { 16.dp }
internal val LocalFlatDiagnosticSink = staticCompositionLocalOf<(FlatDiagnostic) -> Unit> { {} }
internal const val WATCH_ACTION_BUDGET = 32
internal val IMAGE_PROP_KEYS = listOf("url", "src", "image", "source", "name")
internal val ICON_PROP_KEYS = listOf("name", "icon", "source", "url", "src")
internal val MEDIA_PROP_KEYS = listOf("url", "src", "source", "name")
internal val MEDIA_OBJECT_KEYS = listOf("uri", "url", "src", "path", "value", "source", "image", "icon", "name")
internal val DIRECT_TABLE_ROW_LIST_KEYS = listOf("cells", "values", "row", "data")
internal const val FLAT_SPEC_RENDERER_TAG = "FlatSpecRenderer"
internal val PLAYLIST_TABLE_DOMAIN_ALIASES = setOf("playlist", "music", "entertainment")
internal val FORMULA_TABLE_DOMAIN_ALIASES = setOf("formula", "calculation", "calculator", "math")
internal val RESTAURANT_TABLE_DOMAIN_ALIASES = setOf("restaurant", "restaurants", "dining")
internal val NEWS_TABLE_DOMAIN_ALIASES = setOf("news", "headline", "headlines", "article", "articles")
internal val TRAVEL_TABLE_DOMAIN_ALIASES = setOf("travel", "trip", "vacation", "holiday", "itinerary", "places", "attractions")
internal val PRODUCT_TABLE_DOMAIN_ALIASES = setOf("product", "products", "shopping", "catalog")
internal val CARD_FIRST_TABLE_DOMAINS = GeneratedRendererCapabilities.cardFirstTableDomains
internal val SUPPORTED_TABLE_DOMAINS = GeneratedRendererCapabilities.tableDomains


internal val DefaultComputedFunctions: Map<String, FlatComputedFunction> = mapOf(
    "concat" to { args -> args.values.joinToString(separator = "") { it?.toString().orEmpty() } },
    "uppercase" to { args -> args["value"]?.toString()?.uppercase().orEmpty() },
    "lowercase" to { args -> args["value"]?.toString()?.lowercase().orEmpty() },
    "coalesce" to { args ->
        (args["values"] as? List<*>)?.firstOrNull { candidate ->
            when (candidate) {
                null -> false
                is String -> candidate.isNotBlank()
                else -> true
            }
        }
    },
    "sum" to { args ->
        (args["values"] as? List<*>)?.sumOf { (it as? Number)?.toDouble() ?: 0.0 } ?: 0.0
    }
)

internal fun toStringKeyMap(value: Any?): Map<String, Any?>? {
    val map = value as? Map<*, *> ?: return null
    return map.entries.associate { (k, v) -> k.toString() to v }
}

internal fun accessibilityMap(props: Map<String, Any?>): Map<String, Any?> {
    return toStringKeyMap(props["accessibility"])
        ?: toStringKeyMap(props["a11y"])
        ?: emptyMap()
}

internal fun accessibilityString(
    props: Map<String, Any?>,
    vararg keys: String
): String? {
    val nested = accessibilityMap(props)
    keys.forEach { key ->
        nested[key]?.toString()?.trim()?.takeIf { it.isNotBlank() }?.let { return it }
        props[key]?.toString()?.trim()?.takeIf { it.isNotBlank() }?.let { return it }
    }
    return null
}

internal fun accessibilityLabel(
    props: Map<String, Any?>,
    fallback: String? = null
): String? {
    return accessibilityString(
        props,
        "label",
        "ariaLabel",
        "accessibilityLabel",
        "contentDescription",
        "description",
        "alt"
    ) ?: fallback?.trim()?.takeIf { it.isNotBlank() }
}

internal fun isAccessibilityDecorative(props: Map<String, Any?>): Boolean {
    val value = accessibilityMap(props)["decorative"] ?: props["decorative"] ?: props["ariaHidden"]
    return when (value) {
        is Boolean -> value
        is String -> value.equals("true", ignoreCase = true) || value.equals("decorative", ignoreCase = true)
        else -> false
    }
}

internal fun Modifier.accessibilitySemantics(
    props: Map<String, Any?>,
    fallbackLabel: String? = null,
    semanticRole: Role? = null,
    state: String? = null,
    mergeDescendants: Boolean = false,
    isHeading: Boolean = false
): Modifier {
    if (isAccessibilityDecorative(props)) {
        return this
    }
    val label = accessibilityLabel(props, fallbackLabel)
    val effectiveState = state ?: accessibilityString(props, "stateDescription", "state")
    val hasSemantics = !label.isNullOrBlank() ||
        !effectiveState.isNullOrBlank() ||
        semanticRole != null ||
        isHeading
    if (!hasSemantics) {
        return this
    }
    return semantics(mergeDescendants = mergeDescendants) {
        if (!label.isNullOrBlank()) {
            contentDescription = label
        }
        if (!effectiveState.isNullOrBlank()) {
            stateDescription = effectiveState
        }
        if (semanticRole != null) {
            role = semanticRole
        }
        if (isHeading) {
            heading()
        }
    }
}

internal fun normalizePointer(path: String): String {
    val trimmed = path.trim()
    if (trimmed.isBlank()) return ""
    return if (trimmed.startsWith('/')) trimmed else "/$trimmed"
}

internal fun encodePointerToken(token: String): String {
    return token.replace("~", "~0").replace("/", "~1")
}

internal fun combinePointerPath(basePath: String?, token: String): String? {
    val normalizedBase = basePath?.trim()?.takeIf { it.isNotBlank() } ?: return null
    val pointerBase = normalizePointer(normalizedBase)
    return "${pointerBase.trimEnd('/')}/${encodePointerToken(token)}"
}

internal fun resolveItemValue(item: Any?, path: String): Any? {
    if (item == null) return null
    val normalized = path.trim()
    if (normalized.isBlank()) return item
    val fromPointer = FlatSpecParser.getAtPath(item, normalized)
    if (fromPointer != null) return fromPointer
    return if (normalized == "value") item else null
}

internal fun resolveBindItemPath(rawPath: String?, scope: RepeatScope?): String? {
    val basePath = scope?.basePath?.takeIf { it.isNotBlank() } ?: return null
    val requested = rawPath?.trim().orEmpty()
    if (requested.isBlank()) return normalizePointer(basePath)
    return combinePointerPath(basePath, requested)
}

internal fun deepCopyValue(value: Any?): Any? {
    return when (value) {
        is Map<*, *> -> value.entries.associate { (k, v) -> k.toString() to deepCopyValue(v) }
        is List<*> -> value.map { deepCopyValue(it) }
        else -> value
    }
}

internal fun deepEquals(left: Any?, right: Any?): Boolean {
    if (left === right) return true
    if (left == null || right == null) return false
    if (left is Map<*, *> && right is Map<*, *>) {
        if (left.size != right.size) return false
        return left.keys.all { key ->
            right.containsKey(key) && deepEquals(left[key], right[key])
        }
    }
    if (left is List<*> && right is List<*>) {
        if (left.size != right.size) return false
        return left.indices.all { index -> deepEquals(left[index], right[index]) }
    }
    return left == right
}


/**
 * Test-accessible parity boundary shared with the metric's checked-in JSON
 * vectors. Production rendering uses the same resolver and built-ins.
 */
internal fun resolveFlatExpressionForParity(
    expression: Any?,
    state: Map<String, Any?>,
    item: Any?,
    index: Int?,
    basePath: String?
): Any? = FlatExprResolver.resolve(
    expression,
    state,
    RepeatScope(item = item, index = index, basePath = basePath),
    DefaultComputedFunctions
)

internal fun evaluateFlatVisibilityForParity(
    expression: Any?,
    state: Map<String, Any?>,
    item: Any?,
    index: Int?,
    basePath: String?
): Boolean = FlatExprResolver.evaluateVisible(
    expression,
    state,
    RepeatScope(item = item, index = index, basePath = basePath),
    DefaultComputedFunctions
)

/**
 * Test-accessible table-routing boundary. Mirrors what [RenderDirectTable]
 * resolves before it renders anything, without needing a composition.
 */
internal fun resolveFlatTableRoutingForParity(
    props: Map<String, Any?>,
    state: Map<String, Any?>,
    screenWidthDp: Int
): Map<String, Any?>? {
    val table = extractDirectTableModel(
        props,
        state,
        compactScreen = isFlatCompactScreenWidth(screenWidthDp)
    ) ?: return null
    return linkedMapOf(
        "domain" to table.domain,
        "preferredPresentation" to table.preferredPresentation,
        "shape" to table.shape.name,
        "renderMode" to table.renderMode.name,
        "columns" to table.columns.size,
        "rows" to table.rows.size,
        "primaryColumn" to table.primaryColumn,
        "highlightColumns" to table.highlightColumns.sorted(),
        "numericColumns" to table.numericColumns.sorted(),
        "entityMediaKeys" to table.entityMedia.keys.sorted()
    )
}

/**
 * Single definition of the compact-width breakpoint used by table routing.
 * Composable call sites read this from `LocalConfiguration.screenWidthDp`.
 */
internal fun isFlatCompactScreenWidth(screenWidthDp: Int): Boolean = screenWidthDp <= 480

/**
 * Deterministic, `@Composable`-free description of every reachable element and
 * the table-routing decision the renderer would make for it.
 *
 * This is the invariance net for renderer refactors: moving declarations
 * between files, or deleting redundant heuristics that the contract already
 * stamps upstream, must not change this output. Diff the golden after each
 * move. It intentionally records prop *keys* rather than values, so it stays
 * readable in review and does not duplicate the parity vectors.
 */
internal fun describeFlatSpecRouting(spec: FlatSpec, screenWidthDp: Int): String {
    val compactScreen = isFlatCompactScreenWidth(screenWidthDp)
    val described = mutableListOf<Map<String, Any?>>()
    val visited = mutableSetOf<String>()

    fun childReferences(element: FlatElement): List<String> {
        val refs = mutableListOf<String>()
        refs += element.children
        when (element.type.trim().lowercase(Locale.US)) {
            "tabs" -> (element.props["tabs"] as? List<*>)?.forEach { tab ->
                val map = tab as? Map<*, *> ?: return@forEach
                val ref = map["child"]?.toString()
                    ?: map["content"]?.toString()
                    ?: map["id"]?.toString()
                    ?: map["element"]?.toString()
                if (!ref.isNullOrBlank()) refs += ref
            }
            "modal" -> {
                element.props["trigger"]?.toString()?.takeIf { it.isNotBlank() }?.let { refs += it }
                element.props["content"]?.toString()?.takeIf { it.isNotBlank() }?.let { refs += it }
            }
        }
        return refs
    }

    fun visit(elementId: String, depth: Int) {
        if (!visited.add(elementId)) return
        val element = spec.elements[elementId] ?: run {
            described += linkedMapOf<String, Any?>(
                "id" to elementId,
                "depth" to depth,
                "missing" to true
            )
            return
        }
        val type = element.type.trim().lowercase(Locale.US)
        val entry = linkedMapOf<String, Any?>(
            "id" to elementId,
            "depth" to depth,
            "type" to type,
            "propKeys" to element.props.keys.sorted(),
            "childCount" to element.children.size,
            "hasRepeat" to (element.repeat != null),
            "hasVisible" to (element.visible != null),
            "eventKeys" to (element.on?.keys?.sorted() ?: emptyList<String>()),
            "watchKeys" to (element.watch?.keys?.sorted() ?: emptyList<String>())
        )
        if (type == "table") {
            entry["directTable"] = resolveFlatTableRoutingForParity(
                element.props,
                spec.state,
                screenWidthDp
            )
        }
        // Legacy table-like stacks are routed by a separate extractor.
        if (type == "stack" || type == "row" || type == "column" || type == "list") {
            val legacy = extractFlatTableModel(
                element.children,
                element.props,
                spec.elements,
                spec.state,
                compactScreen
            )
            if (legacy != null) {
                entry["legacyTable"] = linkedMapOf<String, Any?>(
                    "domain" to legacy.domain,
                    "preferredPresentation" to legacy.preferredPresentation,
                    "shape" to legacy.shape.name,
                    "renderMode" to legacy.renderMode.name,
                    "columns" to legacy.columns,
                    "rows" to legacy.rows,
                    "cardMappingStatus" to legacy.cardMappingStatus
                )
            }
        }
        described += entry
        childReferences(element).forEach { childId -> visit(childId, depth + 1) }
    }

    visit(spec.root, 0)

    val payload = linkedMapOf<String, Any?>(
        "version" to FLAT_ROUTING_GOLDEN_VERSION,
        "root" to spec.root,
        "screenWidthDp" to screenWidthDp,
        "compactScreen" to compactScreen,
        "reachableElements" to described.size,
        "declaredElements" to spec.elements.size,
        "elements" to described
    )
    return GsonBuilder().setPrettyPrinting().serializeNulls().create().toJson(payload)
}

internal const val FLAT_ROUTING_GOLDEN_VERSION = "2.0.0"



@Composable
fun FlatSpecContent(
    spec: FlatSpec,
    resolveAssetUrl: (String) -> String = { raw -> raw },
    computedFunctions: Map<String, FlatComputedFunction> = emptyMap(),
    collapseRootHorizontalPadding: Boolean = false,
    modifier: Modifier = Modifier
) {
    // Legacy entry point. Keeps the previous behaviour, including state that is
    // keyed on the spec instance, so existing callers are unaffected.
    FlatSpecContent(
        spec = spec,
        host = rememberDefaultFlatRendererHost(resolveAssetUrl),
        stateHolder = rememberFlatSpecStateHolder(spec, spec.state),
        computedFunctions = computedFunctions,
        collapseRootHorizontalPadding = collapseRootHorizontalPadding,
        modifier = modifier
    )
}

/**
 * Host-agnostic entry point.
 *
 * Differences from the legacy overload: navigation, asset resolution, image
 * loading and diagnostics arrive via [host] instead of being reached for through
 * `LocalContext`; render state is owned by [stateHolder] so a surface update can
 * preserve it; and [onEvent] reports actions and state changes so a host or
 * agent can observe them.
 */
@Composable
fun FlatSpecContent(
    spec: FlatSpec,
    host: FlatRendererHost,
    stateHolder: FlatSpecStateHolder = rememberFlatSpecStateHolder(spec, spec.state),
    computedFunctions: Map<String, FlatComputedFunction> = emptyMap(),
    onEvent: (FlatRenderEvent) -> Unit = {},
    collapseRootHorizontalPadding: Boolean = false,
    modifier: Modifier = Modifier
) {
    val renderSpec = remember(spec, collapseRootHorizontalPadding) {
        if (collapseRootHorizontalPadding) spec.withCollapsedRootHorizontalPadding() else spec
    }
    val stateStore = stateHolder.state
    val combinedComputedFunctions = remember(computedFunctions) {
        DefaultComputedFunctions + computedFunctions
    }
    val watchRuntime = remember(renderSpec) { FlatWatchRuntime(renderSpec.elements) }
    val watchCascade = remember(renderSpec) { FlatWatchCascadeTransaction(WATCH_ACTION_BUDGET) }
    val emitDiagnostic: (FlatDiagnostic) -> Unit = { diagnostic ->
        host.onDiagnostic(diagnostic)
        onEvent(
            FlatRenderEvent(
                kind = FlatRenderEvent.Kind.DIAGNOSTIC,
                elementId = diagnostic.elementId,
                value = diagnostic.message,
                params = mapOf(
                    "code" to diagnostic.code.name,
                    "severity" to diagnostic.severity.name
                ) + diagnostic.details
            )
        )
    }

    val onOpenUrl: (String) -> Unit = { url ->
        onEvent(FlatRenderEvent(FlatRenderEvent.Kind.NAVIGATION, action = "openUrl", value = url))
        host.openUrl(url)
    }
    val onSetState: (String, Any?) -> Unit = { path, value ->
        val normalizedPath = normalizePointer(path)
        if (normalizedPath.isNotBlank()) {
            val before = deepCopyValue(FlatSpecParser.getAtPath(stateStore, normalizedPath))
            FlatSpecParser.setAtPath(stateStore, normalizedPath, value)
            val mutation = FlatStateMutation(
                path = normalizedPath,
                operation = FlatStateMutation.Operation.SET,
                before = before,
                after = deepCopyValue(value)
            )
            onEvent(
                FlatRenderEvent(
                    FlatRenderEvent.Kind.STATE_CHANGE,
                    statePath = normalizedPath,
                    value = value,
                    success = true,
                    stateMutations = listOf(mutation)
                )
            )
        }
    }
    val onAction: (Any?, RepeatScope?) -> Int = { actionCandidate, repeatScope ->
        val sourced = actionCandidate as? SourcedActionCandidate
        val result = FlatActionRuntime.executeDetailed(
            actionCandidate = sourced?.candidate ?: actionCandidate,
            stateStore = stateStore,
            repeatScope = repeatScope,
            computedFunctions = combinedComputedFunctions,
            onOpenUrl = onOpenUrl,
            onDiagnostic = emitDiagnostic,
            elements = renderSpec.elements,
            elementId = sourced?.elementId
        )
        result.actions.forEach { execution ->
            onEvent(
                FlatRenderEvent(
                    kind = FlatRenderEvent.Kind.ACTION,
                    action = execution.action,
                    params = execution.params,
                    elementId = sourced?.elementId,
                    success = execution.success,
                    stateMutations = execution.stateMutations
                )
            )
            execution.stateMutations.forEach { mutation ->
                onEvent(
                    FlatRenderEvent(
                        kind = FlatRenderEvent.Kind.STATE_CHANGE,
                        action = execution.action,
                        elementId = sourced?.elementId,
                        statePath = mutation.path,
                        value = mutation.after,
                        success = execution.success,
                        stateMutations = listOf(mutation)
                    )
                )
            }
        }
        result.attempted
    }

    LaunchedEffect(renderSpec) {
        snapshotFlow { stateStore.toMap() }.collect { snapshot ->
            val triggered = watchRuntime.collectTriggeredEntries(snapshot)
            if (triggered.isEmpty()) {
                watchCascade.reset()
                return@collect
            }
            watchCascade.beginEmission()
            triggered.forEach { trigger ->
                if (!watchCascade.enter(trigger)) {
                    if (watchCascade.claimTerminationDiagnostic()) {
                        emitDiagnostic(
                            FlatDiagnostic(
                                code = FlatDiagnostic.Code.WATCH_CASCADE_TERMINATED,
                                severity = FlatDiagnostic.Severity.ERROR,
                                message = "Watch cascade terminated after a repeated fingerprint or the $WATCH_ACTION_BUDGET-action budget.",
                                details = mapOf(
                                    "watch" to trigger.watchKey,
                                    "statePath" to trigger.statePath,
                                    "remainingActions" to watchCascade.remainingActions
                                )
                            )
                        )
                    }
                    return@forEach
                }
                val candidate = limitActionCandidate(trigger.actionBinding, watchCascade.remainingActions)
                val executed = onAction(
                    SourcedActionCandidate(candidate, trigger.elementId),
                    null
                )
                watchCascade.consume(executed)
            }
        }
    }

    CompositionLocalProvider(
        // Asset resolution and image loading both come from the host now, so a
        // consumer outside this app module can supply its own.
        LocalFlatSpecAssetResolver provides { raw -> host.resolveAssetUrl(raw) },
        LocalFlatImageLoader provides host.imageLoader,
        LocalFlatSpecComputedFunctions provides combinedComputedFunctions,
        LocalFlatDiagnosticSink provides emitDiagnostic
    ) {
        RenderElement(
            elementId = renderSpec.root,
            elements = renderSpec.elements,
            state = stateStore,
            repeatScope = null,
            onOpenUrl = onOpenUrl,
            onSetState = onSetState,
            onAction = onAction,
            activePath = emptySet(),
            modifier = modifier
        )
    }
}

internal fun limitActionCandidate(candidate: Any?, actionBudget: Int): Any? {
    if (actionBudget <= 0) return emptyList<Any?>()
    return if (candidate is List<*>) candidate.take(actionBudget) else candidate
}

internal fun FlatSpec.withCollapsedRootHorizontalPadding(): FlatSpec {
    val rootElement = elements[root] ?: return this
    val rootType = rootElement.type.trim().lowercase()
    if (rootType !in setOf("stack", "column", "row", "list", "container", "box")) return this

    val props = rootElement.props
    val hasAllPadding = props.containsKey("padding")
    val hasHorizontalPadding = props.containsKey("paddingHorizontal")
    if (!hasAllPadding && !hasHorizontalPadding) return this

    val adjustedProps = props.toMutableMap()
    if (hasAllPadding && !adjustedProps.containsKey("paddingVertical")) {
        adjustedProps["paddingVertical"] = props["padding"]
    }
    adjustedProps.remove("padding")
    adjustedProps["paddingHorizontal"] = 0

    return copy(
        elements = elements + (root to rootElement.copy(props = adjustedProps))
    )
}

@Composable
internal fun RenderElement(
    elementId: String,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    onAction: (Any?, RepeatScope?) -> Int,
    activePath: Set<String>,
    modifier: Modifier = Modifier
) {
    val diagnosticSink = LocalFlatDiagnosticSink.current
    if (elementId in activePath) {
        val diagnostic = FlatDiagnostic(
            code = FlatDiagnostic.Code.REFERENCE_CYCLE,
            severity = FlatDiagnostic.Severity.ERROR,
            message = "Renderer reference cycle detected at '$elementId'.",
            elementId = elementId
        )
        LaunchedEffect(diagnostic.code, elementId) { diagnosticSink(diagnostic) }
        RenderUnsupportedElement("Reference cycle: $elementId", modifier)
        return
    }
    val element = elements[elementId]
    if (element == null) {
        val diagnostic = FlatDiagnostic(
            code = FlatDiagnostic.Code.MISSING_ELEMENT,
            severity = FlatDiagnostic.Severity.ERROR,
            message = "Missing renderer element '$elementId'.",
            elementId = elementId
        )
        LaunchedEffect(diagnostic.code, elementId) { diagnosticSink(diagnostic) }
        RenderUnsupportedElement("Missing element: $elementId", modifier)
        return
    }
    val computedFunctions = LocalFlatSpecComputedFunctions.current
    if (!FlatExprResolver.evaluateVisible(element.visible, state, repeatScope, computedFunctions)) return

    val repeatedChildScopes: List<RepeatScope>? = buildRepeatScopes(element.repeat, state)
    val repeatKeyIssues = repeatedChildScopes.orEmpty().filter { it.keyIssue != null }
    if (repeatKeyIssues.isNotEmpty()) {
        LaunchedEffect(elementId, repeatKeyIssues) {
            repeatKeyIssues.forEach { scope ->
                val issue = scope.keyIssue ?: return@forEach
                diagnosticSink(
                    FlatDiagnostic(
                        code = FlatDiagnostic.Code.DUPLICATE_REPEAT_KEY,
                        severity = FlatDiagnostic.Severity.WARNING,
                        message = when (issue) {
                            RepeatKeyIssue.MISSING ->
                                "repeat.key '${element.repeat?.key}' is missing at index ${scope.index}; using index identity."
                            RepeatKeyIssue.DUPLICATE ->
                                "repeat.key '${element.repeat?.key}' is duplicated at index ${scope.index}; using index identity."
                        },
                        elementId = elementId,
                        details = mapOf(
                            "repeatKey" to element.repeat?.key,
                            "index" to scope.index,
                            "issue" to issue.name.lowercase(Locale.US)
                        )
                    )
                )
            }
        }
    }

    val resolvedProps = element.props.mapValues { (_, value) ->
        FlatExprResolver.resolve(value, state, repeatScope, computedFunctions)
    }

    RenderByType(
        elementId = elementId,
        type = element.type,
        props = resolvedProps,
        children = element.children,
        onMap = element.on,
        elements = elements,
        state = state,
        repeatScope = repeatScope,
        repeatedChildScopes = repeatedChildScopes,
        onOpenUrl = onOpenUrl,
        onSetState = onSetState,
        onAction = onAction,
        activePath = activePath + elementId,
        modifier = modifier
    )
}

@Composable
internal fun RenderByType(
    elementId: String,
    type: String,
    props: Map<String, Any?>,
    children: List<String>,
    onMap: Map<String, Any?>?,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    repeatedChildScopes: List<RepeatScope>?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    onAction: (Any?, RepeatScope?) -> Int,
    activePath: Set<String>,
    modifier: Modifier = Modifier
) {
    val compatibilityDirection = GeneratedRendererCapabilities.compatibilityTypeDirections[
        type.trim().lowercase(Locale.US)
    ]
    val effectiveProps = if (compatibilityDirection != null && "direction" !in props) {
        props + ("direction" to compatibilityDirection)
    } else {
        props
    }
    val sourcedOnAction: (Any?, RepeatScope?) -> Int = { candidate, scope ->
        // Child renderers pass their already-sourced value through parent
        // containers, so the innermost interactive element remains the source.
        onAction(
            candidate as? SourcedActionCandidate
                ?: SourcedActionCandidate(candidate, elementId),
            scope
        )
    }
    val context = FlatRenderContext(
        elementId = elementId,
        type = type,
        props = effectiveProps,
        children = children,
        onMap = onMap,
        elements = elements,
        state = state,
        repeatScope = repeatScope,
        repeatedChildScopes = repeatedChildScopes,
        onOpenUrl = onOpenUrl,
        onSetState = onSetState,
        onAction = sourcedOnAction,
        activePath = activePath,
        modifier = modifier
    )
    val renderer = flatRendererFor(type)
    if (renderer != null) {
        renderer(context)
        return
    }
    val diagnosticSink = LocalFlatDiagnosticSink.current
    val diagnostic = FlatDiagnostic(
        code = FlatDiagnostic.Code.UNSUPPORTED_TYPE,
        severity = FlatDiagnostic.Severity.ERROR,
        message = "Unsupported element type '$type' on '$elementId'.",
        elementId = elementId,
        details = mapOf("type" to type)
    )
    LaunchedEffect(diagnostic.code, elementId, type) { diagnosticSink(diagnostic) }
    Log.w(
        FLAT_SPEC_RENDERER_TAG,
        "Unsupported element type '$type' on '$elementId'" +
            if (children.isEmpty()) " (rendered as a placeholder)"
            else " (rendered its ${children.size} child element(s))"
    )
    if (children.isNotEmpty()) {
        RenderStack(
            elementId = elementId,
            props = mapOf("direction" to "vertical"),
            children = children,
            elements = elements,
            state = state,
            repeatScope = repeatScope,
            repeatedChildScopes = repeatedChildScopes,
            onOpenUrl = onOpenUrl,
            onSetState = onSetState,
            onAction = onAction,
            activePath = activePath,
            modifier = modifier
        )
    } else {
        RenderUnsupportedElement(type, modifier)
    }
}

@Composable
internal fun RenderUnsupportedElement(
    type: String,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
            .clip(RoundedCornerShape(8.dp)),
        color = MaterialTheme.colorScheme.errorContainer
    ) {
        Text(
            text = "Unsupported element: $type",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onErrorContainer,
            modifier = Modifier
                .padding(12.dp)
                .semantics { contentDescription = "Unsupported element type $type" }
        )
    }
}

@Composable
internal fun RenderChildren(
    children: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    repeatedChildScopes: List<RepeatScope>?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    onAction: (Any?, RepeatScope?) -> Int,
    activePath: Set<String>
) {
    val visibleChildren = children.filterNot { childId ->
        isDetachedMediaDumpElement(childId, elements) ||
            isRedundantWeatherLeadInElement(childId, children, elements, state) ||
            isRedundantTopMetricSummaryElement(childId, children, elements, state)
    }
    if (visibleChildren.isEmpty()) {
        return
    }
    if (repeatedChildScopes == null) {
        visibleChildren.forEach { childId ->
            RenderElement(
                elementId = childId,
                elements = elements,
                state = state,
                repeatScope = repeatScope,
                onOpenUrl = onOpenUrl,
                onSetState = onSetState,
                onAction = onAction,
                activePath = activePath
            )
        }
        return
    }

    if (repeatedChildScopes.isEmpty()) {
        return
    }

    repeatedChildScopes.forEach { scopedRepeat ->
        visibleChildren.forEach { childId ->
            key("$childId:${scopedRepeat.stableKey ?: scopedRepeat.index}") {
                RenderElement(
                    elementId = childId,
                    elements = elements,
                    state = state,
                    repeatScope = scopedRepeat,
                    onOpenUrl = onOpenUrl,
                    onSetState = onSetState,
                    onAction = onAction,
                    activePath = activePath
                )
            }
        }
    }
}

@Composable
internal fun RenderFlatSourceSection(
    section: FlatSourceSection,
    onOpenUrl: (String) -> Unit,
    modifier: Modifier = Modifier,
    wrapInCard: Boolean,
    showTitle: Boolean
) {
    val dedupedLinks = section.links.distinctBy { flatCanonicalSourceUrl(it.url) }
    if (dedupedLinks.isEmpty()) {
        return
    }
    val title = section.title
        ?.let(NativeSourceParsing::normalizeSourceHeadingToken)
        ?.takeIf { it.isNotBlank() }
        ?: "Sources"

    @Composable
    fun SourceContent(contentModifier: Modifier = Modifier) {
        Column(
            modifier = contentModifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            if (showTitle) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.semantics { heading() }
                )
            }
            dedupedLinks.forEach { source ->
                RenderFlatSourceRow(source, onOpenUrl)
            }
        }
    }

    if (wrapInCard) {
        Card(
            modifier = modifier
                .fillMaxWidth()
                .padding(vertical = 4.dp)
                .semantics(mergeDescendants = false) {
                    contentDescription = "$title, ${dedupedLinks.size} links"
                },
            colors = flatSpecCardColors(),
            elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
            shape = RoundedCornerShape(16.dp),
            border = flatSpecCardBorder()
        ) {
            SourceContent(
                contentModifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp)
            )
        }
    } else {
        SourceContent(modifier)
    }
}

@Composable
internal fun RenderFlatSourceRow(
    source: ParsedButton,
    onOpenUrl: (String) -> Unit
) {
    val label = source.label
        .takeIf { it.isNotBlank() }
        ?: flatSourceLabelFromUrl(source.url)
    val urlHint = flatSourceUrlDisplay(source.url)
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .clickable(role = Role.Button) { onOpenUrl(source.url) }
            .semantics(mergeDescendants = true) {
                contentDescription = "Open source $label"
                role = Role.Button
            },
        shape = RoundedCornerShape(14.dp),
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.72f),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.38f))
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Surface(
                modifier = Modifier.size(34.dp),
                shape = RoundedCornerShape(12.dp),
                color = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f)
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(
                        imageVector = Icons.Filled.Link,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(18.dp)
                    )
                }
            }
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                Text(
                    text = parseBoldMarkdown(label),
                    style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.SemiBold),
                    color = MaterialTheme.colorScheme.onSurface,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
                Text(
                    text = urlHint,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
            Icon(
                imageVector = Icons.AutoMirrored.Filled.OpenInNew,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(18.dp)
            )
        }
    }
}

internal fun isDetachedMediaDumpElement(
    elementId: String,
    elements: Map<String, FlatElement>
): Boolean {
    val root = elements[elementId] ?: return false
    val rootType = root.type.lowercase()
    if (rootType !in setOf("card", "stack", "column", "row", "list")) return false

    var hasMedia = rootType in setOf("image", "icon")
    var hasInteractiveOrStructuredContent = rootType in setOf(
        "table",
        "button",
        "tabs",
        "emailpreview",
        "codeblock",
        "consolelog",
        "chart"
    )
    val headings = mutableListOf<String>()
    val visited = mutableSetOf<String>()

    fun walk(id: String, depth: Int) {
        if (depth > 8 || !visited.add(id)) return
        val element = elements[id] ?: return
        val type = element.type.lowercase()
        if (type in setOf("image", "icon")) {
            hasMedia = true
        }
        if (type in setOf("table", "button", "tabs", "emailpreview", "codeblock", "consolelog", "chart")) {
            hasInteractiveOrStructuredContent = true
        }
        if (type == "text") {
            val text = FlatExprResolver.resolveString(
                value = textLikeValue(element.props),
                state = emptyMap(),
                repeatScope = null,
                computedFunctions = emptyMap()
            ).trim()
            if (text.isNotBlank()) {
                headings += text
            }
        }
        element.children.forEach { childId -> walk(childId, depth + 1) }
    }

    walk(elementId, 0)
    if (!hasMedia || hasInteractiveOrStructuredContent) return false
    val detachedHeadingTokens = setOf(
        "images",
        "icons",
        "visual guide",
        "key feature icons",
        "trip imagery",
        "weather icons",
        "related icons",
        "referenced icons",
        "gallery"
    )
    return headings.any { heading ->
        val token = heading.trim().lowercase()
        token in detachedHeadingTokens
    }
}

internal fun looksLikeCurrentWeatherMetricsTable(
    elementId: String,
    props: Map<String, Any?>,
    state: Map<String, Any?>
): Boolean {
    val table = extractDirectTableModel(props, state, compactScreen = true) ?: return false
    if (table.rows.isEmpty()) return false
    val twoColumnMetricTable = table.columns.size == 2 &&
        table.columns.any { column -> normalizeTableHeaderForMatch(column.label) in setOf("metric", "label", "field", "attribute") } &&
        table.columns.any { column -> normalizeTableHeaderForMatch(column.label) in setOf("value", "detail", "details") }
    if (table.shape != FlatTableShape.KEY_VALUE && !twoColumnMetricTable) return false
    val titleToken = listOfNotNull(
        elementId,
        props["title"]?.toString(),
        props["heading"]?.toString(),
        props["label"]?.toString(),
        props["name"]?.toString()
    ).joinToString(" ").let(NativeWeatherSemantics::normalizeWeatherText)
    val rowLabelTokens = table.rows
        .mapNotNull { row -> row.firstOrNull() }
        .map(NativeWeatherSemantics::normalizeWeatherText)
    val currentTitle = titleToken.contains("current") &&
        (titleToken.contains("weather") || titleToken.contains("metric") || titleToken.contains("condition"))
    val weatherMetricSignals = rowLabelTokens.count { label ->
        label.contains("temperature") ||
            label.contains("temp") ||
            label.contains("high") ||
            label.contains("low") ||
            label.contains("feel") ||
            label.contains("rain") ||
            label.contains("precip") ||
            label.contains("wind") ||
            label.contains("humidity") ||
            label.contains("uv") ||
            label.contains("best window")
    }
    return currentTitle || weatherMetricSignals >= 3
}

internal fun containsForecastWeatherTableElement(
    elementId: String,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    visited: MutableSet<String> = mutableSetOf(),
    depth: Int = 0
): Boolean {
    if (depth > 6 || !visited.add(elementId)) return false
    val element = elements[elementId] ?: return false
    if (element.type.equals("table", ignoreCase = true) && looksLikeForecastWeatherTable(element.props, state)) {
        return true
    }
    return element.children.any { childId ->
        containsForecastWeatherTableElement(childId, elements, state, visited, depth + 1)
    }
}

internal fun isRedundantTopMetricSummaryElement(
    elementId: String,
    siblingIds: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>
): Boolean {
    val index = siblingIds.indexOf(elementId)
    if (index < 0) return false
    if (looksLikeTopMetricContainerBlock(elementId, elements, state)) {
        return true
    }
    val hasLaterResultTable = siblingIds.drop(index + 1).any { siblingId ->
        containsResultTableElement(siblingId, elements, state)
    }
    if (!hasLaterResultTable) return false
    val element = elements[elementId] ?: return false
    if (element.type.equals("text", ignoreCase = true)) {
        val text = FlatExprResolver.resolveString(
            value = textLikeValue(element.props),
            state = emptyMap(),
            repeatScope = null,
            computedFunctions = emptyMap()
        ).trim()
        return isTopMetricHeadingText(text)
    }
    if (element.type.equals("table", ignoreCase = true)) {
        return looksLikeTopMetricTable(element.props, state)
    }
    return looksLikeTopMetricTextBlock(elementId, elements)
}

internal fun looksLikeTopMetricContainerBlock(
    elementId: String,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>
): Boolean {
    val root = elements[elementId] ?: return false
    val rootType = root.type.trim().lowercase()
    if (rootType !in setOf("card", "stack", "column", "row", "list")) return false
    val textValues = mutableListOf<String>()
    var hasTopMetricTable = false
    var hasDisqualifyingStructuredContent = false
    val visited = mutableSetOf<String>()

    fun walk(id: String, depth: Int) {
        if (depth > 6 || !visited.add(id)) return
        val element = elements[id] ?: return
        val type = element.type.trim().lowercase()
        when (type) {
            "table" -> {
                if (looksLikeTopMetricTable(element.props, state)) {
                    hasTopMetricTable = true
                } else {
                    hasDisqualifyingStructuredContent = true
                }
                return
            }
            "button", "tabs", "image", "emailpreview", "codeblock", "consolelog", "chart", "form" -> {
                hasDisqualifyingStructuredContent = true
                return
            }
            "text" -> {
                val text = FlatExprResolver.resolveString(
                    value = textLikeValue(element.props),
                    state = emptyMap(),
                    repeatScope = null,
                    computedFunctions = emptyMap()
                ).trim()
                if (text.isNotBlank()) {
                    textValues += text
                }
            }
        }
        element.children.forEach { childId -> walk(childId, depth + 1) }
    }

    walk(elementId, 0)
    if (!hasTopMetricTable || hasDisqualifyingStructuredContent) return false
    if (textValues.size > 18 || textValues.any { it.length > 120 }) return false
    return textValues.any(::isTopMetricHeadingText) || textValues.count(::isTopMetricLabelText) >= 2
}

internal fun containsResultTableElement(
    elementId: String,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    visited: MutableSet<String> = mutableSetOf(),
    depth: Int = 0
): Boolean {
    if (depth > 6 || !visited.add(elementId)) return false
    val element = elements[elementId] ?: return false
    if (element.type.equals("table", ignoreCase = true)) {
        val table = extractDirectTableModel(element.props, state, compactScreen = true)
        if (table != null && table.rows.isNotEmpty()) return true
    }
    return element.children.any { childId ->
        containsResultTableElement(childId, elements, state, visited, depth + 1)
    }
}

internal fun looksLikeTopMetricTable(
    props: Map<String, Any?>,
    state: Map<String, Any?>
): Boolean {
    val table = extractDirectTableModel(props, state, compactScreen = true) ?: return false
    if (table.shape != FlatTableShape.KEY_VALUE || table.rows.size !in 2..8) return false
    val labels = table.rows.mapNotNull { row -> row.firstOrNull()?.trim()?.takeIf(String::isNotBlank) }
    return labels.count(::isTopMetricLabelText) >= 2
}

internal fun looksLikeTopMetricTextBlock(
    elementId: String,
    elements: Map<String, FlatElement>
): Boolean {
    val root = elements[elementId] ?: return false
    val rootType = root.type.trim().lowercase()
    if (rootType !in setOf("card", "stack", "column", "row", "list")) return false
    val textValues = mutableListOf<String>()
    var hasDisqualifyingStructuredContent = false
    val visited = mutableSetOf<String>()

    fun walk(id: String, depth: Int) {
        if (depth > 6 || !visited.add(id)) return
        val element = elements[id] ?: return
        val type = element.type.trim().lowercase()
        if (type in setOf("table", "button", "tabs", "image", "emailpreview", "codeblock", "consolelog", "chart", "form")) {
            hasDisqualifyingStructuredContent = true
            return
        }
        if (type == "text") {
            val text = FlatExprResolver.resolveString(
                value = textLikeValue(element.props),
                state = emptyMap(),
                repeatScope = null,
                computedFunctions = emptyMap()
            ).trim()
            if (text.isNotBlank()) {
                textValues += text
            }
        }
        element.children.forEach { childId -> walk(childId, depth + 1) }
    }

    walk(elementId, 0)
    if (hasDisqualifyingStructuredContent || textValues.size !in 3..16) return false
    if (textValues.any { it.length > 48 }) return false
    if (textValues.size <= 4 && textValues.any(::isTopResultCountSummaryText)) {
        return true
    }
    val metricLabelCount = textValues.count(::isTopMetricLabelText)
    val valueLikeCount = textValues.count(::isShortMetricValueText)
    return metricLabelCount >= 2 && valueLikeCount >= 2
}

internal fun isTopMetricHeadingText(text: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(text)
    return normalized in setOf(
        "metrics",
        "trip metrics",
        "key metrics",
        "summary metrics",
        "quick metrics",
        "snapshot metrics",
        "overview metrics"
    )
}

internal fun isTopResultCountSummaryText(text: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(text)
    if (normalized.length > 90) return false
    return normalized.contains("returned") &&
        normalized.contains("options") &&
        (normalized.contains("best") || normalized.contains("other") || normalized.contains("total"))
}

internal fun isTopMetricLabelText(text: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(text)
    if (normalized.isBlank() || normalized.length > 36) return false
    val phrases = listOf(
        "fastest",
        "lowest",
        "highest",
        "average",
        "total",
        "options",
        "non stop",
        "nonstop",
        "one stop",
        "duration",
        "fare",
        "price",
        "count",
        "listed",
        "shown",
        "available",
        "score",
        "rating"
    )
    return phrases.any { normalized.contains(it) }
}

internal fun isShortMetricValueText(text: String): Boolean {
    val trimmed = text.trim()
    if (trimmed.isBlank() || trimmed.length > 32 || SafeContentPolicy.looksLikeUrl(trimmed)) return false
    return trimmed.any(Char::isDigit) ||
        trimmed.startsWith("$") ||
        trimmed.startsWith("₹") ||
        trimmed.startsWith("€") ||
        trimmed.startsWith("£")
}

internal fun looksLikeForecastWeatherTable(
    props: Map<String, Any?>,
    state: Map<String, Any?>
): Boolean {
    val table = extractDirectTableModel(props, state, compactScreen = true) ?: return false
    if (table.rows.size < 2) return false
    if (table.domain == "weather" || table.renderMode == FlatTableRenderMode.WEATHER_CARDS) return true
    val headers = table.columns.map { column -> column.label }
    return !NativeWeatherSemantics.buildWeatherRows(headers, table.rows).isNullOrEmpty()
}

internal fun isRedundantWeatherLeadInElement(
    elementId: String,
    siblingIds: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>
): Boolean {
    val hasForecastWeatherSibling = siblingIds.any { siblingId ->
        siblingId != elementId && containsForecastWeatherTableElement(siblingId, elements, state)
    }
    if (!hasForecastWeatherSibling) return false
    val element = elements[elementId] ?: return false
    if (element.type.equals("table", ignoreCase = true)) {
        return looksLikeCurrentWeatherMetricsTable(elementId, element.props, state)
    }
    return looksLikeCurrentWeatherLeadInElement(elementId, elements, state)
}

internal fun looksLikeCurrentWeatherLeadInElement(
    elementId: String,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>
): Boolean {
    val root = elements[elementId] ?: return false
    val rootType = root.type.trim().lowercase()
    if (rootType !in setOf("card", "stack", "column", "row", "list", "text")) return false
    val textValues = mutableListOf<String>()
    var hasStructuredTable = false
    var hasCurrentWeatherMetricTable = false
    var hasForecastWeatherTable = false
    val visited = mutableSetOf<String>()

    fun walk(id: String, depth: Int) {
        if (depth > 5 || !visited.add(id)) return
        val element = elements[id] ?: return
        val type = element.type.trim().lowercase()
        if (type == "table") {
            hasStructuredTable = true
            if (looksLikeCurrentWeatherMetricsTable(id, element.props, state)) {
                hasCurrentWeatherMetricTable = true
            }
            if (looksLikeForecastWeatherTable(element.props, state)) {
                hasForecastWeatherTable = true
            }
            return
        }
        if (type == "text") {
            val text = FlatExprResolver.resolveString(
                value = textLikeValue(element.props),
                state = emptyMap(),
                repeatScope = null,
                computedFunctions = emptyMap()
            ).trim()
            if (text.isNotBlank()) {
                textValues += text
            }
        }
        element.children.forEach { childId -> walk(childId, depth + 1) }
    }

    walk(elementId, 0)
    if (hasCurrentWeatherMetricTable && !hasForecastWeatherTable) {
        return true
    }
    if (hasStructuredTable || textValues.isEmpty()) return false
    val normalizedValues = textValues.map(NativeWeatherSemantics::normalizeWeatherText)
    if (normalizedValues.any { it in setOf("current metrics", "current weather", "current conditions") }) {
        return true
    }
    val joined = normalizedValues.joinToString(" ")
    if (joined.contains("forecast") || joined.contains("day by day") || joined.contains("7 day")) {
        return false
    }
    val hasCurrentCue = normalizedValues.any { value ->
        value == "today" ||
            value.contains("today") ||
            value.contains("current") ||
            value.contains("now")
    }
    val metricSignals = listOf(
        "temperature",
        "temp",
        "high",
        "low",
        "feel",
        "rain",
        "precip",
        "wind",
        "humidity",
        "uv",
        "best window",
        "clear sky",
        "mainly clear",
        "cloud",
        "overcast",
        "drizzle",
        "storm"
    ).count { signal -> joined.contains(signal) }
    return hasCurrentCue && metricSignals >= 2
}

internal fun asFloat(value: Any?): Float? = when (value) {
    is Number -> value.toFloat()
    is String -> value.toFloatOrNull()
    else -> null
}

internal fun asDp(value: Any?): androidx.compose.ui.unit.Dp? {
    val number = asFloat(value) ?: return null
    if (!number.isFinite()) return null
    return number.dp
}

internal fun asFlatSpacingDp(value: Any?): androidx.compose.ui.unit.Dp? {
    val numeric = asDp(value)
    if (numeric != null) return numeric
    val token = value?.toString()?.trim()?.lowercase()?.takeIf { it.isNotBlank() } ?: return null
    return when (token) {
        "none", "zero" -> 0.dp
        "xs", "extra-small", "extra_small" -> 4.dp
        "sm", "small" -> 8.dp
        "md", "medium" -> 16.dp
        "lg", "large" -> 20.dp
        "xl", "extra-large", "extra_large" -> 24.dp
        else -> null
    }
}

internal fun extractMediaUrlToken(
    value: Any?,
    depth: Int = 0
): String? {
    if (depth > 5 || value == null) return null
    return when (value) {
        is String -> value.trim().takeIf { it.isNotBlank() }
        is Map<*, *> -> {
            val map = value.entries.associate { (k, v) -> k.toString() to v }
            MEDIA_OBJECT_KEYS.firstNotNullOfOrNull { key ->
                extractMediaUrlToken(map[key], depth + 1)
            }
        }
        is List<*> -> value.firstNotNullOfOrNull { candidate ->
            extractMediaUrlToken(candidate, depth + 1)
        }
        else -> null
    }
}

internal fun extractMediaUrlTokens(
    value: Any?,
    depth: Int = 0
): List<String> {
    if (depth > 5 || value == null) return emptyList()
    return when (value) {
        is String -> value
            .split(Regex("""\s*[,;\n]\s*"""))
            .map { it.trim() }
            .filter { it.isNotBlank() }
        is Map<*, *> -> {
            val map = value.entries.associate { (k, v) -> k.toString() to v }
            MEDIA_OBJECT_KEYS.flatMap { key -> extractMediaUrlTokens(map[key], depth + 1) }
        }
        is List<*> -> value.flatMap { candidate -> extractMediaUrlTokens(candidate, depth + 1) }
        else -> emptyList()
    }
}

internal fun resolveMediaUrlCandidate(
    props: Map<String, Any?>,
    preferredKeys: List<String>
): String {
    return preferredKeys.firstNotNullOfOrNull { key ->
        extractMediaUrlToken(props[key])
    }.orEmpty()
}

internal fun isIconLikeMediaUrl(value: String): Boolean {
    val normalized = value.trim()
    if (normalized.isBlank()) return false
    val lower = normalized.lowercase()
    return NativeMediaVisualUtils.isVectorImagePath(lower) ||
        NativeMediaVisualUtils.looksLikeCompactIconUrl(lower) ||
        lower.contains("cdn.jsdelivr.net/npm/bootstrap-icons")
}

internal fun isPhotoLikeMediaUrl(value: String): Boolean {
    val normalized = value.trim()
    if (normalized.isBlank() || isIconLikeMediaUrl(normalized)) return false
    return NativeMediaVisualUtils.isPhotoLikeImageUrl(normalized)
}

internal fun parseUrlHost(rawUrl: String): String {
    val normalized = rawUrl.trim()
    if (normalized.isBlank()) return ""
    return runCatching {
        URI(normalized).host?.lowercase().orEmpty()
    }.getOrElse { "" }
}

internal fun flatSourceCueText(value: String?): Boolean {
    val normalized = NativeSourceParsing
        .normalizeSourceHeadingToken(value.orEmpty())
        .trim()
    if (normalized.isBlank()) return false
    val lower = normalized.lowercase()
    return NativeSourceParsing.isSourceHeadingLine(normalized) ||
        lower.contains("source") ||
        lower.contains("reference") ||
        lower.contains("citation")
}

internal fun flatSourceCueFromSelf(
    elementId: String,
    props: Map<String, Any?>
): Boolean {
    if (flatSourceCueText(elementId)) return true
    return listOf(
        "role",
        "semanticRole",
        "domain",
        "section",
        "title",
        "label",
        "heading",
        "accessibilityLabel",
        "ariaLabel"
    ).any { key -> flatSourceCueText(props[key]?.toString()) }
}

internal fun flatDirectSourceCue(
    childId: String,
    element: FlatElement,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    computedFunctions: Map<String, FlatComputedFunction>
): Boolean {
    if (flatSourceCueText(childId) || flatSourceCueFromSelf(childId, element.props)) {
        return true
    }
    if (element.type.equals("table", ignoreCase = true) && flatTablePropsLookLikeSourceLinks(element.props, state)) {
        return true
    }
    if (!element.type.equals("text", ignoreCase = true)) {
        return false
    }
    val rawText = listOf("text", "title", "label", "content", "value")
        .firstNotNullOfOrNull { key -> element.props[key] }
        ?: return false
    val resolved = FlatExprResolver.resolve(rawText, state, repeatScope, computedFunctions)
        ?.toString()
        .orEmpty()
    return resolved
        .lineSequence()
        .map { it.trim() }
        .filter { it.isNotBlank() }
        .any { NativeSourceParsing.isSourceHeadingLine(it) }
}

internal fun flatResolvedString(
    value: Any?,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    computedFunctions: Map<String, FlatComputedFunction>
): String {
    return FlatExprResolver.resolve(value, state, repeatScope, computedFunctions)
        ?.toString()
        ?.trim()
        .orEmpty()
}

internal fun toFlatExternalUrl(value: String?): String? {
    val normalized = NativePayloadParser.canonicalizeNetworkUrlToken(
        flatSanitizeUrlToken(value.orEmpty())
    )
    return SafeContentPolicy.sanitizeActionUrl(normalized)
}

internal fun flatSanitizeUrlToken(value: String): String =
    NativeStructureParsing.sanitizeUrlToken(value)

internal fun parseFlatSourceLinksFromLine(line: String): List<ParsedButton> {
    return NativeSourceParsing.parseSourceLinksFromLine(
        line = line,
        stripLeadingBulletMarker = NativeStructureParsing::stripLeadingBulletMarker,
        sanitizeDisplayText = NativeTextFormatter::sanitizeDisplayText,
        sanitizeUrlToken = ::flatSanitizeUrlToken,
        toExternalUrl = ::toFlatExternalUrl,
        containsUrlLikeToken = NativeTextFormatter::containsUrlLikeToken
    )
}

internal fun flatSourceTitleFromText(value: String): String? {
    return value
        .lineSequence()
        .map { it.trim() }
        .firstOrNull { NativeSourceParsing.isSourceHeadingLine(it) }
        ?.let(NativeSourceParsing::normalizeSourceHeadingToken)
        ?.takeIf { it.isNotBlank() }
}

internal fun flatSourceUrlCandidateFromAction(
    onMap: Map<String, Any?>?,
    props: Map<String, Any?>,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    computedFunctions: Map<String, FlatComputedFunction>
): String? {
    val eventCandidates = listOf("press", "click", "tap", "select", "open")
    val actionCandidates = eventCandidates
        .mapNotNull { event ->
            toStringKeyMap(onMap?.get(event))?.takeIf { it.isNotEmpty() }
        }
    val propCandidates = listOf(
        props["url"],
        props["href"],
        props["link"],
        props["source"]
    )
    val actionUrlCandidates = actionCandidates.flatMap { action ->
        val params = toStringKeyMap(action["params"]) ?: emptyMap()
        listOf(
            params["url"],
            params["href"],
            params["link"],
            action["url"],
            action["href"],
            action["link"]
        )
    }
    return (actionUrlCandidates + propCandidates)
        .firstNotNullOfOrNull { raw ->
            toFlatExternalUrl(
                flatResolvedString(
                    value = raw,
                    state = state,
                    repeatScope = repeatScope,
                    computedFunctions = computedFunctions
                )
            )
        }
}

internal fun flatSourceLabelFromButtonProps(
    props: Map<String, Any?>,
    fallbackUrl: String,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    computedFunctions: Map<String, FlatComputedFunction>
): String {
    return listOf("label", "text", "title", "content", "value")
        .firstNotNullOfOrNull { key ->
            flatResolvedString(props[key], state, repeatScope, computedFunctions)
                .takeIf { it.isNotBlank() && !NativeTextFormatter.containsUrlLikeToken(it) }
        }
        ?: flatSourceLabelFromUrl(fallbackUrl)
}

internal fun parseFlatSourceLinksFromListItems(value: Any?): List<ParsedButton> {
    val items = value as? List<*> ?: return emptyList()
    return items.flatMap { item ->
        when (item) {
            is String -> parseFlatSourceLinksFromLine(item)
            is Map<*, *> -> {
                val map = item.entries.associate { (key, entryValue) ->
                    key.toString() to entryValue
                }
                val explicitUrl = listOf("url", "href", "link", "source")
                    .firstNotNullOfOrNull { key ->
                        toFlatExternalUrl(map[key]?.toString())
                    }
                if (explicitUrl != null) {
                    val label = listOf("label", "title", "text", "name", "content", "value")
                        .firstNotNullOfOrNull { key ->
                            map[key]?.toString()
                                ?.trim()
                                ?.takeIf { it.isNotBlank() && !NativeTextFormatter.containsUrlLikeToken(it) }
                        }
                        ?: flatSourceLabelFromUrl(explicitUrl)
                    listOf(ParsedButton(label = label, url = explicitUrl))
                } else {
                    listOf("text", "label", "title", "content", "value", "name")
                        .flatMap { key ->
                            map[key]?.toString()
                                ?.takeIf { it.isNotBlank() }
                                ?.let(::parseFlatSourceLinksFromLine)
                                .orEmpty()
                        }
                }
            }
            else -> emptyList()
        }
    }.distinctBy { flatCanonicalSourceUrl(it.url) }
}

internal fun flatTablePropsLookLikeSourceLinks(
    props: Map<String, Any?>,
    state: Map<String, Any?>
): Boolean {
    val forceSourceContext = flatSourceCueFromSelf("table", props)
    val rows = resolveDirectTableRows(props, state)
    val columns = resolveDirectTableColumns(props, rows)
    val headers = columns.map { it.label }
    return (forceSourceContext || flatHeadersLookLikeSourceTable(headers)) &&
        collectFlatSourceLinksFromTableProps(props, state, forceSourceContext = forceSourceContext).isNotEmpty()
}

internal fun flatHeadersLookLikeSourceTable(headers: List<String>): Boolean {
    return headers.any { header ->
        val normalized = normalizeTableHeaderForMatch(header)
        normalized in setOf("source", "sources", "reference", "references", "citation", "citations") ||
            normalized.contains("source") ||
            normalized.contains("reference") ||
            normalized.contains("citation")
    }
}

internal fun collectFlatSourceLinksFromRows(
    headers: List<String>,
    rows: List<List<String>>,
    rawRows: List<Any?> = emptyList(),
    forceSourceContext: Boolean = false
): List<ParsedButton> {
    if (headers.isEmpty() || rows.isEmpty()) return emptyList()
    if (!forceSourceContext && !flatHeadersLookLikeSourceTable(headers)) return emptyList()
    val urlIndexes = headers.indices.filter { index -> isUrlColumnLabel(headers[index]) }
    val labelIndexes = headers.indices.filter { index ->
        val normalized = normalizeTableHeaderForMatch(headers[index])
        index !in urlIndexes &&
            !isActionLabelColumn(headers[index]) &&
            (
                normalized.contains("source") ||
                    normalized.contains("reference") ||
                    normalized.contains("citation") ||
                    normalized.contains("provider") ||
                    normalized.contains("name") ||
                    normalized.contains("title") ||
                    normalized.contains("label")
                )
    }
    return rows.mapIndexedNotNull { index, row ->
        val rawRow = rawRows.getOrNull(index)
        val url = urlIndexes
            .firstNotNullOfOrNull { urlIndex -> toFlatExternalUrl(row.getOrNull(urlIndex)) }
            ?: flatSourceUrlFromRawTableRow(rawRow)
            ?: row.firstNotNullOfOrNull { value -> toFlatExternalUrl(value) }
        if (url == null) {
            null
        } else {
            val label = labelIndexes
                .firstNotNullOfOrNull { labelIndex -> flatTableSourceLabelCandidate(row.getOrNull(labelIndex)) }
                ?: row.indices
                    .filterNot { it in urlIndexes || isActionLabelColumn(headers.getOrNull(it).orEmpty()) }
                    .firstNotNullOfOrNull { rowIndex -> flatTableSourceLabelCandidate(row.getOrNull(rowIndex)) }
                ?: flatSourceLabelFromUrl(url)
            ParsedButton(label = label, url = url)
        }
    }.distinctBy { flatCanonicalSourceUrl(it.url) }
}

internal fun collectFlatSourceLinksFromTableProps(
    props: Map<String, Any?>,
    state: Map<String, Any?>,
    forceSourceContext: Boolean = false
): List<ParsedButton> {
    val rawRows = resolveDirectTableRows(props, state)
    val columns = resolveDirectTableColumns(props, rawRows)
    if (columns.isEmpty() || rawRows.isEmpty()) return emptyList()
    val rows = rawRows.map { row -> resolveDirectTableRow(row, columns, state) }
    return collectFlatSourceLinksFromRows(
        headers = columns.map { it.label },
        rows = rows,
        rawRows = rawRows,
        forceSourceContext = forceSourceContext
    )
}

internal fun flatSourceUrlFromRawTableRow(rawRow: Any?): String? {
    val map = toStringKeyMap(rawRow)
    if (!map.isNullOrEmpty()) {
        val keyCandidates = listOf(
            "url",
            "href",
            "link",
            "source",
            "sourceUrl",
            "source_url",
            "actionUrl",
            "action_url",
            "targetUrl",
            "target_url"
        )
        keyCandidates.firstNotNullOfOrNull { key ->
            map[key]?.toString()?.let(::toFlatExternalUrl)
        }?.let { return it }
        map.values.firstNotNullOfOrNull { value ->
            when (value) {
                is String -> toFlatExternalUrl(value)
                is Map<*, *> -> flatSourceUrlFromRawTableRow(value)
                else -> null
            }
        }?.let { return it }
    }
    val list = rawRow as? List<*> ?: return null
    return list.firstNotNullOfOrNull { value ->
        when (value) {
            is String -> toFlatExternalUrl(value)
            is Map<*, *> -> flatSourceUrlFromRawTableRow(value)
            else -> null
        }
    }
}

internal fun flatTableSourceLabelCandidate(value: String?): String? {
    val trimmed = NativeTextFormatter.sanitizeDisplayText(value.orEmpty()).trim()
    if (trimmed.isBlank()) return null
    if (NativeTextFormatter.containsUrlLikeToken(trimmed)) return null
    val normalized = normalizeTableHeaderForMatch(trimmed)
    if (normalized in setOf("open", "view", "visit", "read", "go", "source", "sources", "link", "links")) {
        return null
    }
    return trimmed
}


internal fun extractFlatSourceSection(
    elementId: String,
    props: Map<String, Any?>,
    children: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    computedFunctions: Map<String, FlatComputedFunction>,
    allowChildSourceCue: Boolean
): FlatSourceSection? {
    val hasSelfCue = flatSourceCueFromSelf(elementId, props)
    val hasChildCue = allowChildSourceCue && children.any { childId ->
        elements[childId]?.let { child ->
            flatDirectSourceCue(
                childId = childId,
                element = child,
                state = state,
                repeatScope = repeatScope,
                computedFunctions = computedFunctions
            )
        } == true
    }
    if (!hasSelfCue && !hasChildCue) return null

    val section = collectFlatSourceLinks(
        elementIds = children,
        elements = elements,
        state = state,
        repeatScope = repeatScope,
        computedFunctions = computedFunctions
    )
    return section.takeIf { it.links.isNotEmpty() }
}

internal fun flatCanonicalSourceUrl(raw: String): String {
    val normalized = raw.trim()
    val uri = runCatching { Uri.parse(normalized) }.getOrNull()
        ?: return normalized.lowercase()
    val scheme = (uri.scheme ?: "https").lowercase()
    val host = uri.host?.lowercase()?.removePrefix("www.").orEmpty()
    if (host.isBlank()) {
        return normalized.lowercase()
    }
    val path = uri.path.orEmpty().trimEnd('/')
    val query = uri.query.orEmpty().trim()
    return buildString {
        append(scheme)
        append("://")
        append(host)
        if (path.isNotBlank()) append(path)
        if (query.isNotBlank()) {
            append('?')
            append(query)
        }
    }
}

internal fun flatSourceLabelFromUrl(url: String): String {
    val host = runCatching { Uri.parse(url).host?.removePrefix("www.").orEmpty() }
        .getOrElse { "" }
    return host
        .substringBefore('.')
        .replace('-', ' ')
        .replace('_', ' ')
        .split(Regex("""\s+"""))
        .filter { it.isNotBlank() }
        .joinToString(" ") { token ->
            token.replaceFirstChar { ch ->
                if (ch.isLowerCase()) ch.titlecase() else ch.toString()
            }
        }
        .ifBlank { "Source" }
}

internal fun flatSourceUrlDisplay(url: String): String {
    val uri = runCatching { Uri.parse(url) }.getOrNull() ?: return url
    val host = uri.host?.removePrefix("www.").orEmpty()
    if (host.isBlank()) return url
    val pathHint = uri.pathSegments
        ?.take(2)
        ?.filter { it.isNotBlank() }
        ?.joinToString("/")
        .orEmpty()
    return buildString {
        append(host)
        if (pathHint.isNotBlank()) {
            append('/')
            append(pathHint)
        }
    }.take(72)
}

internal fun deriveImageFallbackUrl(
    sourceUrl: String,
    props: Map<String, Any?>
): String? {
    val explicitFallback = extractMediaUrlToken(props["fallbackUrl"]).orEmpty()
    return explicitFallback.takeIf { it.isNotBlank() }
}

internal fun parseAspectRatio(value: Any?): Float? {
    return when (value) {
        is Number -> value.toFloat().takeIf { it.isFinite() && it > 0f }
        is String -> {
            val token = value.trim()
            if (token.isBlank()) return null
            if (token.contains(":")) {
                val parts = token.split(":", limit = 2)
                val first = parts.getOrNull(0)?.trim()?.toFloatOrNull()
                val second = parts.getOrNull(1)?.trim()?.toFloatOrNull()
                if (first != null && second != null && first > 0f && second > 0f) {
                    first / second
                } else {
                    null
                }
            } else {
                token.toFloatOrNull()?.takeIf { it.isFinite() && it > 0f }
            }
        }
        else -> null
    }
}

internal fun resolveImageScale(props: Map<String, Any?>): ContentScale {
    val token = (
        props["contentScale"]?.toString()
            ?: props["fit"]?.toString()
        )
        ?.trim()
        ?.lowercase()
        .orEmpty()
    return when (token) {
        "cover", "crop", "fill", "fillbounds", "fill-bounds" -> ContentScale.Crop
        "contain", "fit", "inside", "center", "none" -> ContentScale.Fit
        else -> ContentScale.Fit
    }
}

internal fun resolveIconSize(props: Map<String, Any?>): androidx.compose.ui.unit.Dp {
    asDp(props["size"])?.let { return it }
    asDp(props["iconSize"])?.let { return it }
    asDp(props["width"])?.let { return it }
    asDp(props["height"])?.let { return it }
    return when (props["size"]?.toString()?.trim()?.lowercase()) {
        "xs", "xsmall", "extra-small" -> 14.dp
        "sm", "small" -> 18.dp
        "md", "medium" -> 24.dp
        "lg", "large" -> 32.dp
        "xl", "xlarge", "extra-large" -> 40.dp
        else -> 24.dp
    }
}

internal fun stackDirection(props: Map<String, Any?>): String {
    val raw = (
        props["direction"]?.toString()
            ?: props["orientation"]?.toString()
            ?: props["axis"]?.toString()
        )
        ?.trim()
        ?.lowercase()
        .orEmpty()
    return when (raw) {
        "horizontal", "row", "x" -> "horizontal"
        "vertical", "column", "y" -> "vertical"
        else -> "vertical"
    }
}

internal fun stackGap(props: Map<String, Any?>): androidx.compose.ui.unit.Dp {
    val raw = props["gap"] ?: props["spacing"] ?: props["space"]
    if (raw is Number) {
        return raw.toFloat().dp
    }
    return when (raw?.toString()?.trim()?.lowercase()) {
        "none" -> 0.dp
        "xs", "xsmall", "extra-small" -> 2.dp
        "sm", "small" -> 4.dp
        "md", "medium" -> 8.dp
        "lg", "large" -> 12.dp
        "xl", "xlarge", "extra-large" -> 16.dp
        else -> 8.dp
    }
}

internal fun buildRepeatScopes(
    repeat: RepeatConfig?,
    state: Map<String, Any?>
): List<RepeatScope>? {
    val repeatConfig = repeat ?: return null
    val items = (FlatSpecParser.getAtPath(state, repeatConfig.statePath) as? List<*>).orEmpty()
    val seenKeys = mutableSetOf<String>()
    return items.mapIndexed { index, item ->
        val requestedKey = repeatConfig.key?.trim().orEmpty()
        val rawKey = if (requestedKey.isBlank()) null else {
            (item as? Map<*, *>)?.entries
                ?.firstOrNull { (key, _) -> key?.toString() == requestedKey }
                ?.value
        }
        val normalizedKey = rawKey?.toString()?.trim().orEmpty()
        val issue = when {
            requestedKey.isBlank() -> null
            normalizedKey.isBlank() -> RepeatKeyIssue.MISSING
            !seenKeys.add(normalizedKey) -> RepeatKeyIssue.DUPLICATE
            else -> null
        }
        RepeatScope(
            item = item,
            index = index,
            basePath = combinePointerPath(repeatConfig.statePath, index.toString()),
            stableKey = if (issue == null && normalizedKey.isNotBlank()) {
                "$requestedKey:$normalizedKey"
            } else {
                "index:$index"
            },
            keyIssue = issue
        )
    }
}

internal fun isHorizontalRowElement(element: FlatElement?): Boolean {
    if (element == null) return false
    return when (element.type.lowercase()) {
        "row" -> true
        "stack" -> stackDirection(element.props) == "horizontal"
        else -> false
    }
}

internal fun hasOnlyTextCellChildren(element: FlatElement, elements: Map<String, FlatElement>): Boolean {
    return element.children.isNotEmpty() && element.children.all { childId ->
        elements[childId]?.type?.equals("text", ignoreCase = true) == true
    }
}

internal fun textLikeValue(props: Map<String, Any?>): Any? {
    return props["text"] ?: props["title"] ?: props["label"] ?: props["content"] ?: props["value"]
}

internal fun resolveTableHeaderLabel(
    element: FlatElement?,
    state: Map<String, Any?>
): String {
    if (element == null) return ""
    val resolved = FlatExprResolver.resolveString(
        value = textLikeValue(element.props),
        state = state,
        repeatScope = null,
        computedFunctions = emptyMap()
    )
    return resolved.trim()
}














internal fun normalizeExplicitTableDomain(value: String?): String? {
    val token = value?.trim()?.lowercase(Locale.US)?.takeIf { it.isNotBlank() } ?: return null
    return GeneratedRendererCapabilities.tableDomainAliases[token] ?: token
}







internal fun looksLikeClimateComparisonTable(
    headers: List<String>,
    rows: List<List<String>>,
    domain: String = "generic"
): Boolean {
    if (headers.size < 3 || rows.isEmpty()) return false
    val firstHeader = normalizeTableHeaderForMatch(headers.firstOrNull().orEmpty())
    val firstColumnIsPlace = firstHeader in setOf(
        "city",
        "destination",
        "location",
        "place",
        "region",
        "area",
        "country"
    ) || isComparisonEntityHeader(headers.firstOrNull().orEmpty())
    if (!firstColumnIsPlace) return false

    val normalizedHeaders = headers.map(::normalizeTableHeaderForMatch)
    val climateSignals = normalizedHeaders.count { token ->
        token.contains("climate") ||
            token.contains("weather") ||
            token.contains("temp") ||
            token.contains("high") ||
            token.contains("low") ||
            token.contains("rain") ||
            token.contains("precip") ||
            token.contains("sunshine") ||
            token.contains("sunny") ||
            token.contains("humidity") ||
            token.contains("wind") ||
            token.contains("uv")
    }
    val hasTemperature = normalizedHeaders.any { token ->
        token.contains("temp") || token.contains("high") || token.contains("low")
    }
    val hasOutdoorMetric = normalizedHeaders.any { token ->
        token.contains("rain") ||
            token.contains("precip") ||
            token.contains("sunshine") ||
            token.contains("sunny") ||
            token.contains("humidity") ||
            token.contains("wind") ||
            token.contains("uv")
    }
    val comparisonLike = domain in setOf("comparison", "weather", "generic")
    return comparisonLike && climateSignals >= 2 && hasTemperature && hasOutdoorMetric
}

internal fun stringSetFromTableProp(value: Any?): Set<String> {
    return when (value) {
        is List<*> -> value.mapNotNull { item -> item?.toString()?.trim()?.takeIf { it.isNotBlank() } }.toSet()
        is String -> value
            .split(',', '|')
            .mapNotNull { item -> item.trim().takeIf { it.isNotBlank() } }
            .toSet()
        else -> emptySet()
    }
}

internal fun normalizeColumnToken(value: String): String = normalizeTableHeaderForMatch(value)

internal fun columnIndexForToken(columns: List<FlatDirectTableColumn>, token: String): Int? {
    val normalized = normalizeColumnToken(token)
    if (normalized.isBlank()) return null
    return columns.indexOfFirst { column ->
        normalizeColumnToken(column.key) == normalized ||
            normalizeColumnToken(column.label) == normalized
    }.takeIf { it >= 0 }
}

internal fun numericColumnIndexes(
    columns: List<FlatDirectTableColumn>,
    rows: List<List<String>>,
    explicitTokens: Set<String>
): Set<Int> {
    val explicitIndexes = explicitTokens.mapNotNull { token -> columnIndexForToken(columns, token) }.toSet()
    if (explicitIndexes.isNotEmpty()) return explicitIndexes
    return columns.indices.filter { index ->
        val header = normalizeTableHeaderForMatch(columns[index].label)
        val headerLooksNumeric = listOf(
            "price",
            "cost",
            "fare",
            "rate",
            "amount",
            "total",
            "score",
            "rating",
            "percent",
            "change",
            "value",
            "revenue",
            "sales",
            "count",
            "qty",
            "quantity"
        ).any { header.contains(it) }
        val values = rows.mapNotNull { row -> row.getOrNull(index)?.trim()?.takeIf { it.isNotBlank() } }
        val numericValues = values.count(::looksLikeNumericTableValue)
        headerLooksNumeric || (values.size >= 2 && numericValues >= values.size / 2)
    }.toSet()
}

internal fun looksLikeNumericTableValue(value: String): Boolean {
    val normalized = value.trim()
    if (normalized.isBlank()) return false
    return Regex("""^[₹$€£]?\s*[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|x|k|K|m|M|bn|hrs?|hours?|mins?|minutes?|days?|°[CF]?)?$""")
        .containsMatchIn(normalized)
}

internal fun looksLikeRankHeader(label: String): Boolean {
    val token = normalizeTableHeaderForMatch(label)
    return token in setOf("rank", "id", "number", "no") || label.trim() == "#"
}

internal fun looksLikeLongDetailHeader(label: String): Boolean {
    val token = normalizeTableHeaderForMatch(label)
    return listOf("note", "notes", "description", "detail", "details", "justification", "reason", "summary", "remarks")
        .any { keyword -> token.contains(keyword) }
}

internal fun isCompactTableBadgeValue(value: String): Boolean {
    val normalized = value.trim()
    if (normalized.isBlank() || isLikelyHttpUrl(normalized)) return false
    if (normalized.contains('\n')) return false
    if (Regex("""[.!?]\s+""").containsMatchIn(normalized)) return false
    return normalized.length <= 34
}

internal fun isCompactHighlightColumn(rows: List<List<String>>, index: Int): Boolean {
    val values = rows.mapNotNull { row -> row.getOrNull(index)?.trim()?.takeIf { it.isNotBlank() } }
    if (values.isEmpty()) return false
    val compactCount = values.count(::isCompactTableBadgeValue)
    return compactCount >= maxOf(1, values.size * 2 / 3)
}

internal fun inferEntityPrimaryColumnIndex(headers: List<String>, primaryColumn: String?): Int {
    primaryColumn?.let { token ->
        val explicit = headers.indexOfFirst { header -> normalizeColumnToken(header) == normalizeColumnToken(token) }
        if (explicit >= 0) return explicit
    }
    val first = headers.firstOrNull().orEmpty()
    if (looksLikeRankHeader(first) && headers.size > 1) {
        val nextEntity = headers.drop(1).indexOfFirst { header ->
            isComparisonEntityHeader(header) || normalizeTableHeaderForMatch(header) in setOf("airline", "carrier", "name", "title")
        }
        if (nextEntity >= 0) return nextEntity + 1
        return 1
    }
    return 0
}

internal fun shouldUseNativeFlightCards(headers: List<String>): Boolean {
    val normalized = headers.map(::normalizeTableHeaderForMatch)
    val routeSignals = normalized.count { header ->
        header.contains("depart") ||
            header.contains("departure") ||
            header.contains("arrive") ||
            header.contains("arrival") ||
            header.contains("origin") ||
            header.contains("destination") ||
            header.contains("from") ||
            header.contains("to")
    }
    val rankedComparisonSignals = normalized.count { header ->
        header.contains("rank") ||
            header.contains("cost") ||
            header.contains("layover") ||
            header.contains("justification") ||
            header.contains("reason")
    }
    return routeSignals >= 2 && rankedComparisonSignals < 2
}

internal fun rankedFlightColumnIndex(headers: List<String>, keywords: List<String>): Int? {
    return headers.indexOfFirst { header ->
        val token = normalizeTableHeaderForMatch(header)
        keywords.any { keyword -> token.contains(keyword) }
    }.takeIf { it >= 0 }
}


internal fun compactRankBadge(rawRank: String, fallbackIndex: Int): String {
    val number = Regex("""\d+""").find(rawRank)?.value ?: (fallbackIndex + 1).toString()
    return "#$number"
}




internal fun applyStackModifier(
    base: Modifier,
    props: Map<String, Any?>,
    direction: String
): Modifier {
    var out = base
    val marginAll = asFlatSpacingDp(props["margin"])
    val marginHorizontal = asFlatSpacingDp(props["marginHorizontal"]) ?: marginAll
    val marginVertical = asFlatSpacingDp(props["marginVertical"]) ?: marginAll
    if (marginHorizontal != null || marginVertical != null) {
        out = out.padding(
            horizontal = marginHorizontal ?: 0.dp,
            vertical = marginVertical ?: 0.dp
        )
    }

    val paddingAll = asFlatSpacingDp(props["padding"])
    val paddingHorizontal = asFlatSpacingDp(props["paddingHorizontal"]) ?: paddingAll
    val paddingVertical = asFlatSpacingDp(props["paddingVertical"]) ?: paddingAll
    if (paddingHorizontal != null || paddingVertical != null) {
        out = out.padding(
            horizontal = paddingHorizontal ?: 0.dp,
            vertical = paddingVertical ?: 0.dp
        )
    }

    val width = asDp(props["width"])
    val height = asDp(props["height"])
    if (width != null) {
        out = out.width(width)
    } else {
        out = out.fillMaxWidth()
    }
    if (height != null) {
        out = out.height(height)
    }

    val flex = asFloat(props["flex"]) ?: 0f
    if (flex > 0f) {
        out = if (direction == "horizontal") out.fillMaxHeight() else out.fillMaxWidth()
    }
    return out
}

internal fun hasHorizontalContainerPadding(props: Map<String, Any?>): Boolean =
    asFlatSpacingDp(props["padding"]) != null ||
        asFlatSpacingDp(props["paddingHorizontal"]) != null ||
        asFlatSpacingDp(props["contentPadding"]) != null ||
        asFlatSpacingDp(props["contentPaddingHorizontal"]) != null

internal fun rowChildFlex(element: FlatElement?): Float =
    asFloat(element?.props?.get("flex"))
        ?.takeIf { it > 0f }
        ?: 0f

internal fun isPaddedContainerElement(element: FlatElement?): Boolean {
    if (element == null || !hasHorizontalContainerPadding(element.props)) return false
    return when (element.type.trim().lowercase()) {
        "stack", "column", "row", "list", "container", "box" -> true
        else -> false
    }
}


@Composable
internal fun RowScope.RenderRowChildren(
    children: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    repeatedChildScopes: List<RepeatScope>?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    onAction: (Any?, RepeatScope?) -> Int,
    activePath: Set<String>
) {
    val scopes = repeatedChildScopes ?: listOf(repeatScope)
    scopes.forEach { scopedRepeat ->
        children.forEach { childId ->
            val child = elements[childId] ?: return@forEach
            val flex = rowChildFlex(child)
            RenderElement(
                elementId = childId,
                elements = elements,
                state = state,
                repeatScope = scopedRepeat,
                onOpenUrl = onOpenUrl,
                onSetState = onSetState,
                onAction = onAction,
                activePath = activePath,
                modifier = if (flex > 0f) Modifier.weight(flex) else Modifier
            )
        }
    }
}



internal fun extractTableEntityMedia(props: Map<String, Any?>): Map<String, TableEntityMedia> {
    val raw = toStringKeyMap(props["entityMedia"]) ?: return emptyMap()
    val out = linkedMapOf<String, TableEntityMedia>()
    raw.forEach { (rawKey, rawValue) ->
        val key = rawKey.trim().takeIf { it.isNotBlank() } ?: return@forEach
        val media = when (rawValue) {
            is String -> TableEntityMedia(
                image = rawValue.trim(),
                alt = key.replace('_', ' ')
            )
            else -> {
                val mediaProps = toStringKeyMap(rawValue) ?: return@forEach
                val image = resolveMediaUrlCandidate(mediaProps, IMAGE_PROP_KEYS)
                    .ifBlank { extractMediaUrlToken(mediaProps["path"]).orEmpty() }
                    .trim()
                if (image.isBlank()) return@forEach
                val alt = listOf("alt", "label", "title", "name")
                    .firstNotNullOfOrNull { candidate ->
                        mediaProps[candidate]?.toString()?.trim()?.takeIf { it.isNotBlank() }
                    }
                    ?: key.replace('_', ' ')
                TableEntityMedia(image = image, alt = alt)
            }
        }
        if (media.image.isBlank()) return@forEach
        tableEntityMediaLookupKeys(key).forEach { lookupKey ->
            out.putIfAbsent(lookupKey, media)
        }
    }
    return out
}

internal fun tableEntityMediaLookupKeys(value: String): Set<String> {
    val normalized = normalizeTableHeaderForMatch(value)
    val compact = normalized.replace(" ", "")
    val underscore = normalized.replace(" ", "_")
    return setOf(value.trim().lowercase(), normalized, compact, underscore)
        .filter { it.isNotBlank() }
        .toSet()
}

internal fun entityMediaForColumn(
    columns: List<FlatDirectTableColumn>,
    headers: List<String>,
    columnIndex: Int,
    entityMedia: Map<String, TableEntityMedia>
): TableEntityMedia? {
    if (entityMedia.isEmpty()) return null
    val candidates = linkedSetOf<String>()
    columns.getOrNull(columnIndex)?.let { column ->
        candidates += tableEntityMediaLookupKeys(column.key)
        candidates += tableEntityMediaLookupKeys(column.label)
    }
    headers.getOrNull(columnIndex)?.let { header ->
        candidates += tableEntityMediaLookupKeys(header)
    }
    return candidates.firstNotNullOfOrNull { key -> entityMedia[key] }
}


internal val RESTAURANT_ACTION_ROW_COLUMN_ALIASES = listOf(
    "photoUrl" to "Photo URL",
    "photoUrls" to "Photo URLs",
    "photos" to "Photos",
    "imageUrl" to "Photo URL",
    "images" to "Photos",
    "mapsUrl" to "Maps URL",
    "mapUrl" to "Maps URL",
    "googleMapsUri" to "Maps URL",
    "websiteUrl" to "Website URL",
    "websiteUri" to "Website URL",
    "bookUrl" to "Book URL",
    "reserveUrl" to "Reserve URL",
    "reservationUrl" to "Reserve URL",
    "actionLabel" to "Action Label",
    "buttonLabel" to "Action Label",
    "ctaLabel" to "Action Label",
    "phone" to "Phone",
    "telephone" to "Phone",
    "nationalPhoneNumber" to "Phone",
    "internationalPhoneNumber" to "Phone"
)

internal fun augmentRestaurantDirectTableColumns(
    props: Map<String, Any?>,
    rows: List<Any?>,
    columns: List<FlatDirectTableColumn>
): List<FlatDirectTableColumn> {
    if (columns.isEmpty() || rows.isEmpty()) return columns
    if (!shouldAugmentRestaurantDirectTableColumns(props, rows, columns)) return columns

    val rowMaps = rows.mapNotNull(::toStringKeyMap)
    if (rowMaps.isEmpty()) return columns

    val presentTokens = columns
        .flatMap { column -> listOf(column.key, column.label) }
        .map(::normalizeTableLookupToken)
        .filter { it.isNotBlank() }
        .toMutableSet()
    val augmented = columns.toMutableList()

    RESTAURANT_ACTION_ROW_COLUMN_ALIASES.forEach { (key, label) ->
        val keyToken = normalizeTableLookupToken(key)
        val labelToken = normalizeTableLookupToken(label)
        if (keyToken in presentTokens || labelToken in presentTokens) return@forEach
        val hasValue = rowMaps.any { row ->
            val lookup = buildDirectTableNormalizedRowLookup(row)
            val value = lookup[keyToken]
            tableCellDisplayText(value).isNotBlank()
        }
        if (hasValue) {
            augmented += FlatDirectTableColumn(key = key, label = label)
            presentTokens += keyToken
            presentTokens += labelToken
        }
    }
    return augmented
}

internal fun shouldAugmentRestaurantDirectTableColumns(
    props: Map<String, Any?>,
    rows: List<Any?>,
    columns: List<FlatDirectTableColumn>
): Boolean {
    val explicitDomain = normalizeExplicitTableDomain(props["domain"]?.toString())
    if (explicitDomain == "restaurants") return true

    val headers = columns.map { column -> column.label }
    if (inferTableDomainFromHeaders(headers) == "restaurants") return true

    val rowMaps = rows.mapNotNull(::toStringKeyMap)
    if (rowMaps.isEmpty()) return false
    val rowTokens = rowMaps
        .flatMap { row -> row.keys }
        .map(::normalizeTableLookupToken)
        .toSet()
    val hasRestaurantEntity = rowTokens.any {
        it in setOf("restaurant", "restaurantname", "name", "place", "placename")
    }
    val hasRestaurantData = rowTokens.any {
        it in setOf(
            "rating", "reviews", "reviewcount", "userratingcount", "address",
            "formattedaddress", "status", "statushours", "hours", "tags", "amenities"
        )
    }
    val hasRestaurantActionData = rowTokens.any {
        it in setOf(
            "phone", "telephone", "nationalphonenumber", "internationalphonenumber",
            "mapsurl", "mapurl", "googlemapsuri", "websiteurl", "websiteuri",
            "bookurl", "reserveurl", "reservationurl", "photourl", "photourls", "photos"
        )
    }
    return hasRestaurantEntity && (hasRestaurantData || hasRestaurantActionData)
}

internal fun parseDirectTableColumns(value: Any?): List<FlatDirectTableColumn> {
    val list = value as? List<*> ?: return emptyList()
    return list.mapIndexedNotNull { index, raw ->
        when (raw) {
            is String, is com.samsung.genuicraft.sdk.internal.renderer.flat.expr.FlatLiteralText -> {
                val label = raw.toString().trim()
                if (label.isBlank()) return@mapIndexedNotNull null
                FlatDirectTableColumn(
                    key = normalizeTableColumnKey(label, "col_${index + 1}"),
                    label = label
                )
            }
            is Map<*, *> -> {
                val map = toStringKeyMap(raw).orEmpty()
                val label = map["label"]?.toString()?.trim().orEmpty()
                val key = map["key"]?.toString()?.trim().orEmpty()
                val finalLabel = if (label.isNotBlank()) label else prettifyTableKey(key)
                val fallbackKey = if (key.isNotBlank()) key else finalLabel
                if (finalLabel.isBlank() && fallbackKey.isBlank()) {
                    null
                } else {
                    FlatDirectTableColumn(
                        key = normalizeTableColumnKey(fallbackKey, "col_${index + 1}"),
                        label = finalLabel.ifBlank { "Column ${index + 1}" }
                    )
                }
            }
            else -> null
        }
    }
}




internal fun extractDirectTableRowList(row: Map<String, Any?>): List<Any?>? {
    DIRECT_TABLE_ROW_LIST_KEYS.forEach { key ->
        val list = row[key] as? List<*>
        if (list != null) return list.toList()
    }
    val nestedRow = toStringKeyMap(row["row"])
    if (nestedRow != null) {
        DIRECT_TABLE_ROW_LIST_KEYS.forEach { key ->
            val list = nestedRow[key] as? List<*>
            if (list != null) return list.toList()
        }
    }
    return null
}

internal fun resolveCoilMediaModel(url: String): String {
    val normalized = url.replace("\\", "/").trim()
    if (normalized.isBlank()) {
        return ""
    }
    return when {
        normalized.startsWith("/assets/") ->
            "file:///android_asset/${normalized.removePrefix("/assets/")}"
        normalized.startsWith("assets/") ->
            "file:///android_asset/${normalized.removePrefix("assets/")}"
        normalized.startsWith("../assets/") ->
            "file:///android_asset/${normalized.removePrefix("../assets/")}"
        normalized.startsWith("./assets/") ->
            "file:///android_asset/${normalized.removePrefix("./assets/")}"
        else -> normalized
    }
}

internal fun resolveRowMapColumnValue(
    row: Map<String, Any?>,
    column: FlatDirectTableColumn,
    columnIndex: Int,
    normalizedLookup: Map<String, Any?>,
    orderedEntries: List<Map.Entry<String, Any?>>,
    usePositionalFallback: Boolean
): Any? {
    val pathCandidates = linkedSetOf<String>()
    if (column.key.isNotBlank()) {
        pathCandidates += column.key
    }
    if (column.label.isNotBlank()) {
        pathCandidates += column.label
        pathCandidates += normalizeTableColumnKey(column.label, column.label)
    }
    pathCandidates.forEach { candidate ->
        resolveRowMapPathValue(row, candidate)?.let { return it }
    }

    listOf(column.key, column.label).forEach { token ->
        val normalizedToken = normalizeTableLookupToken(token)
        if (normalizedToken.isNotBlank() && normalizedLookup.containsKey(normalizedToken)) {
            return normalizedLookup[normalizedToken]
        }
    }

    val ordinal = inferTableColumnOrdinal(column.key, column.label)
    if (ordinal != null && ordinal in orderedEntries.indices) {
        return orderedEntries[ordinal].value
    }
    if (usePositionalFallback && columnIndex in orderedEntries.indices) {
        return orderedEntries[columnIndex].value
    }
    return null
}

internal fun buildDirectTableNormalizedRowLookup(
    row: Map<String, Any?>
): Map<String, Any?> {
    val lookup = linkedMapOf<String, Any?>()
    row.forEach { (key, value) ->
        val normalized = normalizeTableLookupToken(key)
        if (normalized.isNotBlank() && !lookup.containsKey(normalized)) {
            lookup[normalized] = value
        }
        if (key.contains("/") || key.contains(".")) {
            val tail = key.substringAfterLast('/').substringAfterLast('.')
            val normalizedTail = normalizeTableLookupToken(tail)
            if (normalizedTail.isNotBlank() && !lookup.containsKey(normalizedTail)) {
                lookup[normalizedTail] = value
            }
        }
    }
    return lookup
}



internal fun resolveRowMapPathValue(
    row: Map<String, Any?>,
    key: String
): Any? {
    row[key]?.let { return it }
    val normalized = key.trim().trim('/')
    if (normalized.isBlank()) return null
    val segments = when {
        normalized.contains("/") -> normalized.split("/").filter { it.isNotBlank() }
        normalized.contains(".") -> normalized.split(".").filter { it.isNotBlank() }
        else -> emptyList()
    }
    if (segments.isEmpty()) return null
    var current: Any? = row
    for (segment in segments) {
        current = when (current) {
            is Map<*, *> -> current[segment]
            is List<*> -> segment.toIntOrNull()?.let(current::getOrNull)
            else -> return null
        }
    }
    return current
}

internal fun tableCellDisplayText(value: Any?): String {
    return when (value) {
        null -> ""
        is String -> value.trim()
        is Map<*, *> -> {
            val map = toStringKeyMap(value).orEmpty()
            (map["text"] ?: map["label"] ?: map["value"])?.toString()?.trim().orEmpty()
        }
        is List<*> -> value.joinToString(" / ") { entry -> tableCellDisplayText(entry) }.trim()
        else -> value.toString().trim()
    }
}


internal fun prettifyTableKey(key: String): String {
    val cleaned = key.trim()
        .replace('_', ' ')
        .replace('.', ' ')
        .replace('/', ' ')
        .replace(Regex("\\s+"), " ")
        .trim()
    if (cleaned.isBlank()) return key
    return cleaned.split(" ").joinToString(" ") { token ->
        token.lowercase().replaceFirstChar { ch -> ch.titlecase() }
    }
}

internal fun normalizedTableTextLength(value: String): Int {
    return value
        .trim()
        .replace(Regex("\\s+"), " ")
        .length
        .coerceAtMost(96)
}

internal fun estimateTableColumnMinWidthsDp(
    headers: List<String>,
    rows: List<List<String>>,
    baseMinDp: Int = 112
): List<Int> {
    val columnCount = maxOf(headers.size, rows.maxOfOrNull { row -> row.size } ?: 0)
    if (columnCount <= 0) return emptyList()
    val sampledRows = if (rows.size > 40) rows.take(40) else rows
    return (0 until columnCount).map { columnIndex ->
        val maxLength = buildList {
            add(headers.getOrNull(columnIndex).orEmpty())
            sampledRows.forEach { row ->
                add(row.getOrNull(columnIndex).orEmpty())
            }
        }.maxOf { text -> normalizedTableTextLength(text) }

        when {
            maxLength >= 56 -> 232
            maxLength >= 42 -> 208
            maxLength >= 30 -> 184
            maxLength >= 22 -> 164
            maxLength >= 14 -> 144
            else -> baseMinDp
        }
    }
}

internal fun estimateTableMinWidthDp(
    headers: List<String>,
    rows: List<List<String>>,
    baseMinDp: Int = 112
): Int {
    val columnWidths = estimateTableColumnMinWidthsDp(
        headers = headers,
        rows = rows,
        baseMinDp = baseMinDp
    )
    if (columnWidths.isEmpty()) return baseMinDp
    val columnSpacing = 8 * (columnWidths.size - 1).coerceAtLeast(0)
    val sidePadding = 16
    return columnWidths.sum() + columnSpacing + sidePadding
}


internal enum class ResponsiveTableCardTemplate {
    COMPARISON,
    SCHEDULE,
    GENERIC
}


internal fun tableHeaderLabel(headers: List<String>, index: Int): String {
    return headers.getOrNull(index)?.trim().orEmpty().ifBlank { "Column ${index + 1}" }
}

internal data class ResponsiveTableCardCell(
    val index: Int,
    val label: String,
    val value: String
)

internal fun responsiveTableCardCells(
    headers: List<String>,
    row: List<String>
): List<ResponsiveTableCardCell> {
    val cellCount = maxOf(headers.size, row.size)
    return (0 until cellCount).mapNotNull { index ->
        val value = row.getOrNull(index).orEmpty().trim()
        value.takeIf { it.isNotBlank() }?.let {
            ResponsiveTableCardCell(
                index = index,
                label = tableHeaderLabel(headers, index),
                value = value
            )
        }
    }
}

internal fun shouldPromoteTimelineSecondTitle(firstHeader: String, secondHeader: String): Boolean {
    val first = normalizeTableHeaderForMatch(firstHeader)
    val second = normalizeTableHeaderForMatch(secondHeader)
    if (first.isBlank() || second.isBlank()) return false
    if (second == "date" || second == "dates" || second.contains("date ")) return false
    return first.contains("time") ||
        first.contains("slot") ||
        first.contains("date") ||
        (first == "day" && !second.contains("date")) ||
        (first.contains("day") && first.contains("date"))
}

internal fun shouldPrefixTimelineTitle(label: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(label)
    return normalized == "total" ||
        normalized == "status" ||
        normalized == "current status" ||
        normalized == "core ml" ||
        normalized == "applications" ||
        normalized == "ethics"
}

internal fun formatTimelineTitle(label: String, value: String): String {
    val cleanValue = value.trim()
    if (cleanValue.isBlank()) return ""
    val cleanLabel = label.trim()
    val normalizedLabel = normalizeTableHeaderForMatch(cleanLabel)
    if ((normalizedLabel == "day" || normalizedLabel == "week") &&
        cleanValue.all { char -> char.isDigit() }
    ) {
        return "$cleanLabel $cleanValue"
    }
    return if (cleanLabel.isNotBlank() && shouldPrefixTimelineTitle(cleanLabel)) {
        "$cleanLabel: $cleanValue"
    } else {
        cleanValue
    }
}




internal fun pickResponsiveTableTemplate(
    headers: List<String>,
    rows: List<List<String>>
): ResponsiveTableCardTemplate {
    val columnCount = maxOf(headers.size, rows.maxOfOrNull { row -> row.size } ?: 0)
    val firstHeader = normalizeTableHeaderForMatch(headers.getOrNull(0).orEmpty())
    val secondHeader = normalizeTableHeaderForMatch(headers.getOrNull(1).orEmpty())
    val timeLike = listOf("time", "date", "day", "slot").any { token -> firstHeader.contains(token) }
    val eventLike = listOf("event", "activity", "task", "agenda", "schedule", "title", "segment").any { token ->
        secondHeader.contains(token)
    }
    if (columnCount >= 2 && timeLike && eventLike) {
        return ResponsiveTableCardTemplate.SCHEDULE
    }
    if (columnCount >= 3) {
        val firstValues = rows.mapNotNull { row ->
            row.getOrNull(0)?.trim()?.takeIf { it.isNotBlank() }
        }
        val averageFirstLength = if (firstValues.isEmpty()) {
            0
        } else {
            firstValues.sumOf { value -> value.length } / firstValues.size
        }
        val firstHeaderHint = listOf(
            "feature",
            "metric",
            "category",
            "item",
            "type",
            "frequency",
            "location",
            "model",
            "route",
            "plan",
            "option"
        ).any { token -> firstHeader.contains(token) }
        if (firstHeaderHint || averageFirstLength in 3..28) {
            return ResponsiveTableCardTemplate.COMPARISON
        }
    }
    return ResponsiveTableCardTemplate.GENERIC
}


internal fun looksLikePlaceStopItineraryTable(headers: List<String>, rows: List<List<String>>): Boolean {
    if (headers.size < 3 || rows.isEmpty()) return false
    val normalized = headers.map(::normalizeTableHeaderForMatch)
    val hasDay = normalized.any { it.contains("day") || it.contains("date") }
    val hasPlace = normalized.any { header ->
        header in setOf("place", "stop", "attraction", "site", "destination", "name") ||
            header.contains("place") ||
            header.contains("stop") ||
            header.contains("attraction") ||
            header.contains("destination")
    }
    val hasTravelMeta = normalized.any { header ->
        header.contains("rating") ||
            header.contains("review") ||
            header.contains("address") ||
            header.contains("maps") ||
            header.contains("website") ||
            header.contains("image") ||
            header.contains("photo")
    }
    return hasDay && hasPlace && hasTravelMeta
}

internal fun isItineraryActionUrlLabel(label: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(label)
    return normalized.contains("maps") ||
        normalized.contains("direction") ||
        normalized.contains("website") ||
        normalized.contains("booking") ||
        normalized.contains("reservation") ||
        normalized.contains("action url") ||
        normalized == "url" ||
        normalized == "link" ||
        normalized.endsWith(" link")
}

internal fun isItineraryMetricLabel(label: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(label)
    return normalized.contains("rating") ||
        normalized.contains("review") ||
        normalized.contains("price") ||
        normalized.contains("duration") ||
        normalized.contains("time") ||
        normalized.contains("distance") ||
        normalized.contains("difficulty")
}

internal fun isItineraryPlaceLabel(label: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(label)
    return normalized in setOf("place", "stop", "attraction", "site", "destination", "name", "activity") ||
        normalized.contains("place") ||
        normalized.contains("stop") ||
        normalized.contains("attraction") ||
        normalized.contains("destination")
}

internal fun isItineraryAreaLabel(label: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(label)
    return normalized.contains("area") ||
        normalized.contains("focus") ||
        normalized == "title" ||
        normalized.contains("route") ||
        normalized.contains("district") ||
        normalized.contains("neighborhood") ||
        normalized.contains("location")
}

internal fun isItineraryCategoryLabel(label: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(label)
    return normalized.contains("type") ||
        normalized.contains("category") ||
        normalized.contains("theme") ||
        normalized.contains("tag") ||
        normalized.contains("kind")
}

internal fun itineraryColumnIndex(headers: List<String>, predicate: (String) -> Boolean): Int? =
    headers.indices.firstOrNull { index -> predicate(headers[index]) }

internal fun itineraryCell(headers: List<String>, row: List<String>, predicate: (String) -> Boolean): String =
    itineraryColumnIndex(headers, predicate)?.let { index -> row.getOrNull(index).orEmpty().trim() }.orEmpty()

internal fun firstPhotoLikeItineraryImage(headers: List<String>, row: List<String>): String {
    return headers.indices.firstNotNullOfOrNull { index ->
        val header = headers.getOrNull(index).orEmpty()
        val value = row.getOrNull(index).orEmpty().trim()
        when {
            value.isBlank() -> null
            isImageColumnLabel(header) && isPhotoLikeMediaUrl(value) -> value
            normalizeTableHeaderForMatch(header).contains("photo") && isPhotoLikeMediaUrl(value) -> value
            else -> null
        }
    }.orEmpty()
}

internal fun allPhotoLikeItineraryImages(headers: List<String>, rows: List<List<String>>): List<String> =
    rows.flatMap { row ->
        headers.indices.mapNotNull { index ->
            val header = headers.getOrNull(index).orEmpty()
            val value = row.getOrNull(index).orEmpty().trim()
            when {
                value.isBlank() -> null
                isImageColumnLabel(header) && isPhotoLikeMediaUrl(value) -> value
                normalizeTableHeaderForMatch(header).contains("photo") && isPhotoLikeMediaUrl(value) -> value
                else -> null
            }
        }
    }.distinct()

internal fun selectItineraryHeroImages(
    headers: List<String>,
    rows: List<List<String>>,
    groups: Map<String, List<List<String>>>
): List<String> {
    val firstVisibleCardImage = groups.entries
        .firstOrNull()
        ?.value
        ?.firstNotNullOfOrNull { row -> firstPhotoLikeItineraryImage(headers, row).takeIf { it.isNotBlank() } }
        .orEmpty()
    val allImages = allPhotoLikeItineraryImages(headers, rows)
    return (
        allImages.filter { image -> image != firstVisibleCardImage } +
            allImages.filter { image -> image == firstVisibleCardImage }
        ).distinct()
}

internal fun itineraryImageAlt(headers: List<String>, row: List<String>, fallback: String): String {
    return itineraryColumnIndex(headers, ::isImageAltColumnLabel)
        ?.let { row.getOrNull(it).orEmpty().trim() }
        ?.takeIf { it.isNotBlank() }
        ?: fallback
}

internal fun itineraryActionLabel(label: String): String {
    val normalized = normalizeTableHeaderForMatch(label)
    return when {
        normalized.contains("direction") || normalized.contains("maps") || normalized.contains("map") -> "Directions"
        normalized.contains("website") -> "Website"
        normalized.contains("booking") || normalized.contains("reservation") -> "Book"
        else -> "Open"
    }
}

internal fun itineraryActionIcon(label: String): ImageVector {
    val normalized = normalizeTableHeaderForMatch(label)
    return when {
        normalized.contains("direction") || normalized.contains("maps") || normalized.contains("map") -> Icons.Filled.Directions
        normalized.contains("website") -> Icons.Filled.Language
        else -> Icons.AutoMirrored.Filled.OpenInNew
    }
}

internal fun collectItineraryActions(
    headers: List<String>,
    row: List<String>
): List<Pair<String, Pair<ImageVector, String>>> {
    return headers.indices.mapNotNull { index ->
        val header = tableHeaderLabel(headers, index)
        if (!isItineraryActionUrlLabel(header) && !isUrlColumnLabel(header)) return@mapNotNull null
        val url = SafeContentPolicy.sanitizeActionUrl(row.getOrNull(index).orEmpty().trim()) ?: return@mapNotNull null
        itineraryActionLabel(header) to (itineraryActionIcon(header) to url)
    }.distinctBy { it.second.second }.take(3)
}

internal fun itineraryDayToken(headers: List<String>, row: List<String>, fallbackIndex: Int): String {
    val explicit = itineraryColumnIndex(headers) { label ->
        val normalized = normalizeTableHeaderForMatch(label)
        normalized.contains("day") || normalized.contains("date")
    }?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty()
    return explicit.ifBlank { "Day ${fallbackIndex + 1}" }
}

internal fun itineraryAreaTitle(headers: List<String>, row: List<String>, day: String): String {
    val area = itineraryCell(headers, row, ::isItineraryAreaLabel)
    val place = itineraryCell(headers, row, ::isItineraryPlaceLabel)
    return area.ifBlank { place }.ifBlank { day }
}

internal fun itineraryHeroTitle(title: String?, rows: List<List<String>>): String {
    return title
        ?.trim()
        ?.takeIf { it.isNotBlank() }
        ?: rows.firstOrNull()
            ?.firstOrNull()
            ?.takeIf { it.isNotBlank() }
            ?.let { "Vacation itinerary" }
        ?: "Vacation itinerary"
}

internal fun itineraryHeroSubtitle(dayCount: Int, stopCount: Int): String {
    return "$dayCount day plan with $stopCount curated stops, place photos, ratings, and quick actions."
}



@Composable
internal fun TravelItineraryHeroStat(
    value: String,
    label: String,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(18.dp),
        color = Color.White.copy(alpha = 0.16f),
        border = BorderStroke(1.dp, Color.White.copy(alpha = 0.18f))
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(2.dp)
        ) {
            Text(
                text = value,
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Black),
                color = Color.White,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = Color.White.copy(alpha = 0.78f),
                maxLines = 1
            )
        }
    }
}

@Composable
internal fun TravelItineraryDayChipRow(days: List<String>) {
    if (days.isEmpty()) return
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        days.forEachIndexed { index, day ->
            Surface(
                shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                color = if (index == 0) {
                    MaterialTheme.colorScheme.primaryContainer
                } else {
                    MaterialTheme.colorScheme.surfaceContainerHighest
                },
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.45f))
            ) {
                Text(
                    text = NativeTextFormatter.sanitizeDisplayText(day),
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                    color = if (index == 0) {
                        MaterialTheme.colorScheme.onPrimaryContainer
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                    modifier = Modifier.padding(horizontal = 13.dp, vertical = 9.dp),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
internal fun TravelItineraryDayOverview(
    headers: List<String>,
    rows: List<List<String>>,
    day: String,
    area: String
) {
    val stopNames = rows.mapIndexedNotNull { index, row ->
        itineraryCell(headers, row, ::isItineraryPlaceLabel)
            .ifBlank { row.getOrNull(1).orEmpty().trim() }
            .takeIf { it.isNotBlank() }
            ?.let { "${index + 1}. $it" }
    }.take(4)
    if (stopNames.isEmpty()) return
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        color = MaterialTheme.colorScheme.surfaceContainerHighest.copy(alpha = 0.52f),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.36f))
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(5.dp)
        ) {
            Text(
                text = parseBoldMarkdown(NativeTextFormatter.sanitizeDisplayText("$day plan")),
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = parseBoldMarkdown(
                    NativeTextFormatter.sanitizeDisplayText(
                        "Focus: ${area.ifBlank { "curated local stops" }}. ${stopNames.joinToString("  ")}"
                    )
                ),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 4,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}


@Composable
internal fun TravelItineraryDaySections(
    headers: List<String>,
    row: List<String>,
    onOpenUrl: (String) -> Unit
) {
    val excludedIndexes = headers.indices.filter { index ->
        val label = tableHeaderLabel(headers, index)
        val value = row.getOrNull(index).orEmpty().trim()
        value.isBlank() ||
            isImageColumnLabel(label) ||
            isImageAltColumnLabel(label) ||
            isIconColumnLabel(label) ||
            isItineraryActionUrlLabel(label) ||
            isUrlColumnLabel(label) ||
            normalizeTableHeaderForMatch(label).let { it.contains("day") || it.contains("date") } ||
            isItineraryAreaLabel(label)
    }.toSet()
    headers.indices.filterNot { it in excludedIndexes }.forEach { index ->
        val value = row.getOrNull(index).orEmpty().trim()
        if (value.isNotBlank()) {
            ItinerarySectionBlock(label = tableHeaderLabel(headers, index), value = value)
        }
    }
    val actions = collectItineraryActions(headers, row)
    if (actions.isNotEmpty()) {
        TravelItineraryActionRow(actions = actions, onOpenUrl = onOpenUrl)
    }
}


@Composable
internal fun TravelItineraryMetricChip(label: String, value: String) {
    val normalizedLabel = normalizeTableHeaderForMatch(label)
    val text = when {
        normalizedLabel.contains("rating") -> NativeTextFormatter.sanitizeDisplayText(value).let { "★ $it" }
        normalizedLabel.contains("review") -> NativeTextFormatter.sanitizeDisplayText(value)
        else -> NativeTextFormatter.sanitizeDisplayText("${label.trim()}: ${value.trim()}")
    }
    Surface(
        shape = RoundedCornerShape(GenUiTokens.RadiusPill),
        color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.72f),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.10f))
    ) {
        Text(
            text = parseBoldMarkdown(text),
            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
            color = MaterialTheme.colorScheme.onPrimaryContainer,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 5.dp),
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun TravelItineraryActionRow(
    actions: List<Pair<String, Pair<ImageVector, String>>>,
    onOpenUrl: (String) -> Unit
) {
    FlowRow(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalArrangement = Arrangement.spacedBy(7.dp)
    ) {
        actions.forEachIndexed { index, (label, iconAndUrl) ->
            RestaurantActionPill(
                label = label,
                icon = iconAndUrl.first,
                primary = index == 0,
                onClick = { onOpenUrl(iconAndUrl.second) }
            )
        }
    }
}

internal fun compactItinerarySectionLabel(label: String): String {
    val normalized = normalizeTableHeaderForMatch(label)
    return when {
        normalized.contains("morning") -> "Morning"
        normalized.contains("afternoon") -> "Afternoon"
        normalized.contains("evening") -> "Evening"
        normalized.contains("dining") || normalized.contains("food") || normalized.contains("meal") -> "Food"
        normalized.contains("activity") -> "Activity"
        else -> label.trim().ifBlank { "Plan" }
    }
}

internal fun splitLeadingItineraryTitle(value: String): Pair<String?, String> {
    val trimmed = value.trim()
    val match = Regex("""^([^:;]{3,56})[:;]\s+(.+)$""").find(trimmed)
    if (match != null) {
        val title = match.groupValues[1].trim()
        val body = match.groupValues[2].trim()
        if (title.isNotBlank() && body.isNotBlank()) {
            return title to body
        }
    }
    return null to trimmed
}

@Composable
internal fun ItinerarySectionBlock(
    label: String,
    value: String,
    modifier: Modifier = Modifier
) {
    val (title, body) = splitLeadingItineraryTitle(value)
    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(MaterialTheme.colorScheme.surfaceContainerHighest.copy(alpha = 0.56f))
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalArrangement = Arrangement.spacedBy(5.dp)
    ) {
        Text(
            text = compactItinerarySectionLabel(label),
            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
            color = MaterialTheme.colorScheme.primary
        )
        if (!title.isNullOrBlank()) {
            Text(
                text = parseBoldMarkdown(title),
                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                color = MaterialTheme.colorScheme.onSurface
            )
        }
        if (body.isNotBlank()) {
            Text(
                text = parseBoldMarkdown(body),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}


internal fun itineraryVisualPalette(seed: String): Pair<Color, Color> {
    val palettes = listOf(
        Color(0xFF0F766E) to Color(0xFF38BDF8),
        Color(0xFF92400E) to Color(0xFFF59E0B),
        Color(0xFF1D4ED8) to Color(0xFF7C3AED),
        Color(0xFF9D174D) to Color(0xFFFF7A59),
        Color(0xFF166534) to Color(0xFF84CC16)
    )
    val safeHash = seed.hashCode().let { if (it == Int.MIN_VALUE) 0 else kotlin.math.abs(it) }
    return palettes[safeHash.rem(palettes.size)]
}


@Composable
internal fun ScheduleDetailBlock(
    label: String,
    value: String,
    modifier: Modifier = Modifier
) {
    val normalizedValue = value.trim()
    if (normalizedValue.isBlank()) return
    val showLabel = label.isNotBlank()
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(3.dp)
    ) {
        if (showLabel) {
            Text(
                text = label.trim(),
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        Text(
            text = parseBoldMarkdown(normalizedValue),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
internal fun ResponsiveFieldBlock(
    label: String,
    value: String,
    modifier: Modifier = Modifier
) {
    val normalizedValue = value.trim()
    if (normalizedValue.isBlank()) return
    val bulletItems = compactBulletItems(label, normalizedValue)
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(2.dp)
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        if (bulletItems.isNotEmpty()) {
            Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                bulletItems.forEach { item ->
                    Text(
                        text = parseBoldMarkdown("\u2022 $item"),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
            }
        } else {
            Text(
                text = parseBoldMarkdown(normalizedValue),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface
            )
        }
    }
}

internal fun compactBulletItems(label: String, value: String): List<String> {
    val normalizedLabel = normalizeTableHeaderForMatch(label)
    val taskLike = listOf("task", "deliverable", "requirement", "checklist", "step", "action")
        .any { normalizedLabel.contains(it) }
    if (!taskLike || !value.contains(';')) {
        return emptyList()
    }
    val items = value
        .split(';')
        .map { it.trim().trim('.', ';') }
        .filter { it.length >= 6 }
    return items.takeIf { it.size >= 2 }.orEmpty()
}

@Composable
internal fun flatSpecCardColors() = CardDefaults.cardColors(
    containerColor = genUiCardContainerColor(GenUiCardTone.Neutral),
    contentColor = MaterialTheme.colorScheme.onSurface
)

@Composable
internal fun flatSpecCardBorder() = BorderStroke(
    width = 1.dp,
    color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.55f)
)

internal fun tableAccessibilitySummary(
    headers: List<String>,
    rows: List<List<String>>,
    horizontalScroll: Boolean = false
): String {
    val rowLabel = if (rows.size == 1) "row" else "rows"
    val columnLabel = if (headers.size == 1) "column" else "columns"
    return buildString {
        append("Table with ${rows.size} $rowLabel and ${headers.size} $columnLabel")
        if (horizontalScroll) {
            append(". Additional columns are available")
        }
    }
}



internal fun marketTickerColumnIndex(headers: List<String>): Int =
    findTableColumnIndex(headers, listOf("ticker", "symbol", "stock", "asset", "holding", "security"))
        ?: 0

internal fun marketValueColumnIndex(headers: List<String>, tickerIndex: Int): Int =
    headers.indices.firstOrNull { index ->
        if (index == tickerIndex) {
            false
        } else {
            val token = normalizeTableHeaderForMatch(headers[index])
            (token.contains("current") && token.contains("value")) ||
                token.contains("market value") ||
                token == "value" ||
                (token.contains("price") && !token.contains("change"))
        }
    } ?: headers.indices.firstOrNull { it != tickerIndex } ?: 0

internal fun marketChangeAmountColumnIndex(headers: List<String>, exclude: Set<Int>): Int? =
    headers.indices.firstOrNull { index ->
        if (index in exclude) {
            false
        } else {
            val raw = headers[index]
            val token = normalizeTableHeaderForMatch(raw)
            (token.contains("change") && (raw.contains('$') || token.contains("usd") || token.contains("amount"))) ||
                token.contains("gain loss") ||
                token.contains("profit loss") ||
                token == "p l"
        }
    }

internal fun marketChangePercentColumnIndex(headers: List<String>, exclude: Set<Int>): Int? =
    headers.indices.firstOrNull { index ->
        if (index in exclude) {
            false
        } else {
            val raw = headers[index]
            val token = normalizeTableHeaderForMatch(raw)
            raw.contains('%') ||
                token.contains("pct") ||
                token.contains("percent") ||
                token.contains("percentage") ||
                (token.contains("change") && token.contains("rate")) ||
                token.contains("return")
        }
    }

internal fun marketTrendToken(vararg values: String?): Boolean? {
    values.forEach { raw ->
        val value = raw?.trim().orEmpty()
        val first = value.firstOrNull()
        when {
            value.startsWith("+") -> return true
            first == '-' || first?.code == 0x2212 -> return false
        }
    }
    return null
}

@Composable
internal fun marketTrendColor(positive: Boolean?): Color {
    return when (positive) {
        true -> Color(0xFF15803D)
        false -> Color(0xFFDC2626)
        null -> MaterialTheme.colorScheme.primary
    }
}



internal fun processStateColumnIndex(headers: List<String>): Int =
    findTableColumnIndex(headers, listOf("ui state", "screen state", "state", "status", "step", "stage"))
        ?: 0

internal fun processVisualColumnIndex(headers: List<String>, stateIndex: Int): Int? =
    findTableColumnIndex(
        headers,
        listOf("visual", "screen", "view", "interface"),
        exclude = setOf(stateIndex)
    )

internal fun processFeedbackColumnIndex(headers: List<String>, stateIndex: Int): Int? =
    findTableColumnIndex(
        headers,
        listOf("feedback", "message", "text", "result", "copy"),
        exclude = setOf(stateIndex)
    )

internal fun processStateAccent(title: String): Color {
    val token = title.lowercase()
    return when {
        token.contains("success") || token.contains("valid") || token.contains("complete") -> Color(0xFF5EEAD4)
        token.contains("invalid") || token.contains("error") || token.contains("fail") -> Color(0xFFF87171)
        token.contains("scan") || token.contains("ready") || token.contains("start") -> Color(0xFF60A5FA)
        else -> Color(0xFFA78BFA)
    }
}

internal fun incidentComponentColumnIndex(headers: List<String>): Int =
    findTableColumnIndex(headers, listOf("component", "service", "system", "module", "dependency"))
        ?: 0

internal fun incidentStatusColumnIndex(headers: List<String>, componentIndex: Int): Int =
    findTableColumnIndex(
        headers,
        listOf("current status", "status", "health", "state"),
        exclude = setOf(componentIndex)
    ) ?: headers.indices.firstOrNull { it != componentIndex } ?: 0

internal fun incidentNotesColumnIndex(headers: List<String>, excluded: Set<Int>): Int? =
    findTableColumnIndex(
        headers,
        listOf("notes", "impact", "details", "description", "message"),
        exclude = excluded
    )

internal fun incidentSeverityRank(status: String): Int {
    val token = status.lowercase()
    return when {
        token.contains("major") ||
            token.contains("outage") ||
            token.contains("down") ||
            token.contains("failed") -> 3
        token.contains("degraded") ||
            token.contains("partial") ||
            token.contains("delay") -> 2
        token.contains("maintenance") ||
            token.contains("warning") -> 1
        else -> 0
    }
}

internal fun incidentStatusAccent(status: String): Color = when (incidentSeverityRank(status)) {
    3 -> Color(0xFFF97373)
    2 -> Color(0xFFFBBF24)
    1 -> Color(0xFF60A5FA)
    else -> Color(0xFF34D399)
}

internal fun incidentStatusLabel(status: String): String {
    val cleaned = status.replace('_', ' ').replace('-', ' ').trim()
    if (cleaned.isBlank()) return "Unknown"
    return cleaned
        .lowercase()
        .split(Regex("\\s+"))
        .joinToString(" ") { word -> word.replaceFirstChar { it.uppercase() } }
}

internal fun incidentCompactStatusLabel(status: String): String = when (incidentSeverityRank(status)) {
    3 -> "Major"
    2 -> "Degraded"
    1 -> "Maintenance"
    else -> "OK"
}

internal fun incidentOverallLabel(rows: List<List<String>>, statusIndex: Int): String {
    val worst = rows
        .map { row -> row.getOrNull(statusIndex).orEmpty() }
        .maxByOrNull(::incidentSeverityRank)
        .orEmpty()
    return when (incidentSeverityRank(worst)) {
        3 -> "Major outage"
        2 -> "Partial outage"
        1 -> "Maintenance"
        else -> "Operational"
    }
}

@Composable
internal fun IncidentSeverityPill(
    status: String,
    modifier: Modifier = Modifier
) {
    val accent = incidentStatusAccent(status)
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(GenUiTokens.RadiusPill),
        color = accent.copy(alpha = 0.18f)
    ) {
        Text(
            text = parseBoldMarkdown(incidentStatusLabel(status)),
            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
            color = accent,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp)
        )
    }
}


@Composable
internal fun IncidentServiceRowCard(
    headers: List<String>,
    row: List<String>,
    rowIndex: Int,
    componentIndex: Int,
    statusIndex: Int,
    notesIndex: Int?
) {
    val component = row.getOrNull(componentIndex).orEmpty().ifBlank { "Service ${rowIndex + 1}" }
    val status = row.getOrNull(statusIndex).orEmpty()
    val notes = notesIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty()
    val accent = incidentStatusAccent(status)
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .semantics(mergeDescendants = true) {
                contentDescription = tableRowAccessibilitySummary(headers, row, rowIndex)
            },
        shape = RoundedCornerShape(18.dp),
        color = Color.White.copy(alpha = 0.095f),
        tonalElevation = 0.dp,
        shadowElevation = 0.dp
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 13.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(34.dp)
                        .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                        .background(accent.copy(alpha = 0.20f)),
                    contentAlignment = Alignment.Center
                ) {
                    Box(
                        modifier = Modifier
                            .size(12.dp)
                            .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                        .background(accent)
                    )
                }
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Text(
                        text = parseBoldMarkdown(component),
                        style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                        color = Color.White,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis
                    )
                    IncidentSeverityPill(status = status)
                }
            }
            if (notes.isNotBlank()) {
                Text(
                    text = parseBoldMarkdown(notes),
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.White.copy(alpha = 0.76f)
                )
            }
        }
    }
}



@Composable
internal fun ProcessStateStepCard(
    headers: List<String>,
    row: List<String>,
    rowIndex: Int,
    stateIndex: Int,
    visualIndex: Int?,
    feedbackIndex: Int?,
    modifier: Modifier = Modifier
) {
    val title = row.getOrNull(stateIndex).orEmpty().ifBlank { "State ${rowIndex + 1}" }
    val visual = visualIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty()
    val feedback = feedbackIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty()
    val accent = processStateAccent(title)
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .semantics(mergeDescendants = true) {
                contentDescription = tableRowAccessibilitySummary(headers, row, rowIndex)
            },
        shape = RoundedCornerShape(18.dp),
        color = Color.White.copy(alpha = 0.095f),
        tonalElevation = 0.dp,
        shadowElevation = 0.dp
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.Top
        ) {
            Box(
                modifier = Modifier
                    .size(34.dp)
                    .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                    .background(accent.copy(alpha = 0.24f)),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = (rowIndex + 1).toString(),
                    style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold),
                    color = accent
                )
            }
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(7.dp)
            ) {
                Text(
                    text = parseBoldMarkdown(title),
                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                    color = Color.White
                )
                if (visual.isNotBlank()) {
                    Text(
                        text = parseBoldMarkdown(visual),
                        style = MaterialTheme.typography.bodySmall,
                        color = Color.White.copy(alpha = 0.78f)
                    )
                }
                if (feedback.isNotBlank()) {
                    Surface(
                        shape = RoundedCornerShape(14.dp),
                        color = accent.copy(alpha = 0.16f)
                    ) {
                        Text(
                            text = parseBoldMarkdown(feedback),
                            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                            color = Color.White.copy(alpha = 0.90f),
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 7.dp)
                        )
                    }
                }
            }
        }
    }
}


internal fun findTableColumnIndex(
    headers: List<String>,
    keywords: List<String>,
    exclude: Set<Int> = emptySet()
): Int? {
    val normalizedKeywords = keywords.map { it.lowercase() }
    return headers.indices.firstOrNull { index ->
        if (index in exclude) {
            false
        } else {
            val token = normalizeTableHeaderForMatch(headers[index])
            normalizedKeywords.any { keyword -> token.contains(keyword) }
        }
    }
}

internal fun isLikelyHttpUrl(value: String): Boolean {
    return SafeContentPolicy.looksLikeUrl(value)
}

internal fun bookingActionLabelIndex(headers: List<String>): Int? =
    headers.indices.firstOrNull { index ->
        val token = normalizeTableHeaderForMatch(headers[index])
        token in setOf("actionlabel", "action label", "buttonlabel", "button label", "ctalabel", "cta label") ||
            (token.contains("action") && token.contains("label")) ||
            (token.contains("button") && token.contains("label")) ||
            (token.contains("cta") && token.contains("label"))
    }

internal fun bookingImageIndex(headers: List<String>): Int? =
    headers.indices.firstOrNull { index ->
        val token = normalizeTableHeaderForMatch(headers[index])
        if (token.contains("data") || token.contains("details") || token.contains("source")) {
            return@firstOrNull false
        }
        token in setOf(
            "image",
            "images",
            "imageurl",
            "image url",
            "imageurls",
            "image urls",
            "photo",
            "photos",
            "photourl",
            "photo url",
            "photourls",
            "photo urls",
            "thumbnail",
            "media"
        ) ||
            (token.contains("image") && token.contains("url")) ||
            (token.contains("photo") && token.contains("url"))
    }

internal fun bookingRatingIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("rating", "score", "stars", "review score"))

internal fun bookingReviewCountIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("review count", "reviews", "user rating count"))

internal fun bookingClassIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("hotel class", "class", "type", "category", "property type"))

internal fun bookingAmenitiesIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("amenities", "facilities", "features", "services", "highlights"))

internal fun bookingMapIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("map", "maps", "directions", "location url", "maps url"))

internal fun bookingWebsiteIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("website", "official", "site", "homepage"))

internal fun bookingPhotosDataIndex(headers: List<String>): Int? =
    headers.indices.firstOrNull { index ->
        val token = normalizeTableHeaderForMatch(headers[index])
        (token.contains("photo") || token.contains("image")) &&
            (token.contains("data") || token.contains("detail") || token.contains("source") || token.contains("gallery"))
    }

internal fun bookingRowActionLabel(headers: List<String>, row: List<String>): String? {
    val index = bookingActionLabelIndex(headers) ?: return null
    return row.getOrNull(index)
        ?.trim()
        ?.takeIf { it.isNotBlank() }
        ?.takeIf { !isLikelyHttpUrl(it) }
}

internal fun bookingRowImageUrl(headers: List<String>, row: List<String>): String? {
    val index = bookingImageIndex(headers) ?: return null
    return row.getOrNull(index)
        ?.trim()
        ?.takeIf { it.isNotBlank() }
}

internal fun bookingRowImageUrls(headers: List<String>, row: List<String>): List<String> {
    val index = bookingImageIndex(headers) ?: return emptyList()
    return splitRestaurantPhotoUrls(row.getOrNull(index).orEmpty())
        .ifEmpty { row.getOrNull(index)?.trim()?.takeIf { it.isNotBlank() }?.let(::listOf).orEmpty() }
        .distinct()
        .take(4)
}

internal fun restaurantTitleIndex(headers: List<String>): Int =
    findTableColumnIndex(headers, listOf("restaurant", "place", "name", "title")) ?: 0

internal fun restaurantRatingIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("rating", "score", "stars"))

internal fun restaurantReviewCountIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("review count", "reviews", "user rating count"))

internal fun restaurantPriceIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("price", "price level", "cost", "budget"))

internal fun restaurantStatusIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("status", "open", "hours", "open now", "opening"))

internal fun restaurantAddressIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("address", "location"))

internal fun restaurantDescriptionIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("description", "summary", "reason", "why", "notes", "details", "editorial"))
        ?: headers.indices.firstOrNull { index ->
            val token = normalizeTableHeaderForMatch(headers[index])
            token.contains("review") &&
                !token.contains("count") &&
                token !in setOf("review", "reviews", "userratingcount", "user rating count")
        }

internal fun restaurantTagsIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("tags", "type", "types", "cuisine", "category"))

internal fun restaurantPhotoIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("photo", "photos", "photo uri", "photouri", "photo url", "photourl", "photo urls", "photourls", "image", "images", "image url", "imageurl", "image urls", "imageurls", "media"))

internal fun restaurantPhotoIndexes(headers: List<String>): List<Int> =
    headers.indices.filter { index ->
        val token = normalizeTableHeaderForMatch(headers[index])
        listOf("photo", "photos", "photo uri", "photouri", "photo url", "photourl", "photo urls", "photourls", "image", "images", "image url", "imageurl", "image urls", "imageurls", "media").any(token::contains)
    }

internal fun restaurantMapsIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("maps", "map", "directions", "google maps", "googlemapsuri", "google maps uri"))

internal fun restaurantWebsiteIndex(headers: List<String>): Int? =
    headers.indices.firstOrNull { index ->
        val token = normalizeTableHeaderForMatch(headers[index])
        token.contains("website") || token == "site" || token.contains("websiteuri") || token.contains("website uri")
    }

internal fun restaurantBookUrlIndex(headers: List<String>): Int? =
    headers.indices.firstOrNull { index ->
        val token = normalizeTableHeaderForMatch(headers[index])
        (token.contains("book") || token.contains("reserve") || token.contains("reservation")) &&
            (token.contains("url") || token.contains("link") || token.contains("uri"))
    }

internal fun restaurantActionLabelIndex(headers: List<String>): Int? =
    headers.indices.firstOrNull { index ->
        val token = normalizeTableHeaderForMatch(headers[index])
        token in setOf("actionlabel", "action label", "buttonlabel", "button label", "ctalabel", "cta label") ||
            (token.contains("action") && token.contains("label")) ||
            (token.contains("button") && token.contains("label")) ||
            (token.contains("cta") && token.contains("label"))
    }

internal fun restaurantAmenitiesIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("amenities", "features", "services", "highlights"))

internal fun restaurantPhoneIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("phone", "telephone", "call", "national phone", "international phone"))

internal fun splitRestaurantTags(raw: String): List<String> =
    raw.split('|', ',', ';')
        .mapNotNull { it.trim().takeIf { token -> token.isNotBlank() && !isLikelyHttpUrl(token) } }
        .distinct()
        .take(4)

internal fun splitRestaurantPhotoUrls(raw: String): List<String> =
    raw.split('|', '\n', '\t')
        .flatMap { chunk -> chunk.split(Regex("""\s*,\s*(?=https://|genuicraft://|../|assets/)""")) }
        .mapNotNull { value ->
            value.trim()
                .removePrefix("[")
                .removeSuffix("]")
                .trim('"', '\'')
                .takeIf { it.isNotBlank() }
        }
        .filterNot(::isIconLikeMediaUrl)
        .distinct()
        .take(5)

internal fun compactRestaurantMeta(
    rating: String,
    reviewCount: String,
    price: String
): String = buildString {
    if (rating.isNotBlank()) append(rating)
    if (reviewCount.isNotBlank()) {
        if (isNotEmpty()) append("  ·  ")
        append(reviewCount)
    }
    if (price.isNotBlank()) {
        if (isNotEmpty()) append("  ·  ")
        append(price)
    }
}


@Composable
internal fun RestaurantTagChip(label: String) {
    val darkTheme = isSystemInDarkTheme()
    val contentColor = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = if (darkTheme) 0.86f else 0.88f)
    Surface(
        shape = RoundedCornerShape(GenUiTokens.RadiusPill),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = if (darkTheme) 0.26f else 0.38f),
        border = BorderStroke(
            1.dp,
            MaterialTheme.colorScheme.outlineVariant.copy(alpha = if (darkTheme) 0.28f else 0.36f)
        )
    ) {
        Text(
            text = parseBoldMarkdown(label),
            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Medium),
            color = contentColor,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
        )
    }
}

@Composable
internal fun RestaurantPhotoStrip(
    name: String,
    photos: List<String>,
    onOpenUrl: (String) -> Unit
) {
    if (photos.isEmpty()) return
    if (photos.size == 1) {
        RenderImage(
            props = mapOf(
                "url" to photos.first(),
                "fit" to "cover",
                "height" to 172,
                "alt" to "$name photo"
            ),
            onOpenUrl = onOpenUrl,
            modifier = Modifier.fillMaxWidth()
        )
        return
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState())
            .semantics {
                contentDescription = "Scrollable photos for $name"
            },
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        photos.take(5).forEachIndexed { index, photo ->
            RenderImage(
                props = mapOf(
                    "url" to photo,
                    "fit" to "cover",
                    "width" to if (index == 0) 248 else 214,
                    "height" to 156,
                    "alt" to "$name photo ${index + 1}"
                ),
                onOpenUrl = onOpenUrl,
                modifier = Modifier
            )
            if (index == photos.take(5).lastIndex) {
                Spacer(modifier = Modifier.width(2.dp))
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun RestaurantRatingChips(
    rating: String,
    reviews: String,
    price: String
) {
    if (rating.isBlank() && reviews.isBlank() && price.isBlank()) return
    FlowRow(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        if (rating.isNotBlank()) {
            RestaurantMetricChip(
                label = rating,
                icon = Icons.Filled.Star,
                emphasized = true
            )
        }
        if (reviews.isNotBlank()) {
            RestaurantMetricChip(
                label = reviews,
                icon = Icons.Filled.RateReview
            )
        }
        if (price.isNotBlank()) {
            RestaurantMetricChip(label = price)
        }
    }
}

@Composable
internal fun RestaurantMetricChip(
    label: String,
    icon: ImageVector? = null,
    emphasized: Boolean = false
) {
    val darkTheme = isSystemInDarkTheme()
    val containerColor = when {
        emphasized && darkTheme -> Color(0xFF3A2B0D)
        emphasized -> Color(0xFFFFF1CA)
        else -> MaterialTheme.colorScheme.surfaceVariant.copy(alpha = if (darkTheme) 0.24f else 0.34f)
    }
    val contentColor = when {
        emphasized && darkTheme -> Color(0xFFFFD98A)
        emphasized -> Color(0xFF5E430F)
        else -> MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = if (darkTheme) 0.88f else 0.90f)
    }
    Surface(
        shape = RoundedCornerShape(GenUiTokens.RadiusPill),
        color = containerColor,
        border = BorderStroke(
            1.dp,
            if (emphasized) {
                contentColor.copy(alpha = if (darkTheme) 0.26f else 0.18f)
            } else {
                MaterialTheme.colorScheme.outlineVariant.copy(alpha = if (darkTheme) 0.26f else 0.32f)
            }
        )
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
            horizontalArrangement = Arrangement.spacedBy(5.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (icon != null) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    modifier = Modifier.size(14.dp),
                    tint = contentColor
                )
            }
            Text(
                text = parseBoldMarkdown(label),
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                color = contentColor,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
internal fun RestaurantActionPill(
    label: String,
    icon: ImageVector,
    enabled: Boolean = true,
    primary: Boolean = false,
    onClick: () -> Unit
) {
    val containerColor = when {
        primary && enabled -> MaterialTheme.colorScheme.primary
        enabled -> MaterialTheme.colorScheme.surfaceVariant.copy(alpha = if (isSystemInDarkTheme()) 0.26f else 0.38f)
        else -> MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.30f)
    }
    val contentColor = when {
        primary && enabled -> MaterialTheme.colorScheme.onPrimary
        enabled -> MaterialTheme.colorScheme.onSurfaceVariant
        else -> MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.58f)
    }
    val border = when {
        primary && enabled -> null
        enabled -> BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = if (isSystemInDarkTheme()) 0.34f else 0.42f))
        else -> BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.38f))
    }
    val shape = RoundedCornerShape(GenUiTokens.RadiusPill)
    Surface(
        modifier = Modifier
            .then(if (primary) Modifier.widthIn(min = 150.dp) else Modifier)
            .heightIn(min = 40.dp)
            .then(
                if (enabled) {
                    Modifier.clickable(role = Role.Button, onClick = onClick)
                } else {
                    Modifier
                }
            ),
        shape = shape,
        color = containerColor,
        border = border
    ) {
        Row(
            modifier = Modifier
                .heightIn(min = 40.dp)
                .padding(horizontal = if (primary) 14.dp else 12.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                modifier = Modifier.size(16.dp),
                tint = contentColor
            )
            Spacer(modifier = Modifier.width(6.dp))
            Text(
                text = label,
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                color = contentColor,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

internal data class NewsArticleCardRow(
    val title: String,
    val source: String,
    val published: String,
    val category: String,
    val summary: String,
    val imageUrl: String,
    val sourceIcon: String,
    val articleUrl: String,
    val sourceUrl: String,
    val actionLabel: String,
    val sourceRow: List<String>
)

@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun renderNewsRowsIfPossible(
    headers: List<String>,
    rows: List<List<String>>,
    onOpenUrl: (String) -> Unit,
    modifier: Modifier = Modifier,
    title: String? = null
): Boolean {
    if (rows.isEmpty() || !isNewsTableHeaderSet(headers, "news")) return false
    val titleIndex = newsTitleIndex(headers) ?: 0
    val sourceIndex = newsSourceIndex(headers)
    val publishedIndex = newsPublishedIndex(headers)
    val categoryIndex = newsCategoryIndex(headers)
    val summaryIndex = newsSummaryIndex(headers)
    val imageIndex = newsImageIndex(headers)
    val sourceIconIndex = newsSourceIconIndex(headers)
    val articleUrlIndex = newsArticleUrlIndex(headers)
    val sourceUrlIndex = newsSourceUrlIndex(headers)
    val actionLabelIndex = newsActionLabelIndex(headers)

    val articles = rows.mapNotNull { row ->
        val headline = row.getOrNull(titleIndex).orEmpty().trim()
        if (headline.isBlank()) return@mapNotNull null
        NewsArticleCardRow(
            title = headline,
            source = sourceIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty(),
            published = publishedIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty(),
            category = categoryIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty(),
            summary = summaryIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty(),
            imageUrl = imageIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty(),
            sourceIcon = sourceIconIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty(),
            articleUrl = articleUrlIndex?.let { row.getOrNull(it).orEmpty().trim() }
                ?.let(SafeContentPolicy::sanitizeActionUrl)
                .orEmpty(),
            sourceUrl = sourceUrlIndex?.let { row.getOrNull(it).orEmpty().trim() }
                ?.let(SafeContentPolicy::sanitizeActionUrl)
                .orEmpty(),
            actionLabel = actionLabelIndex?.let { row.getOrNull(it).orEmpty().trim() }
                ?.takeIf { it.isNotBlank() && !isLikelyHttpUrl(it) }
                ?: "Read Article",
            sourceRow = row
        )
    }
    if (articles.isEmpty()) return false

    Column(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = "News results with ${articles.size} articles"
            },
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        title?.trim()?.takeIf { it.isNotBlank() }?.let { headingText ->
            Text(
                text = parseBoldMarkdown(headingText),
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                color = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 2.dp, vertical = 4.dp)
                    .semantics { heading() }
            )
        }
        NewsLeadStoryCard(article = articles.first(), onOpenUrl = onOpenUrl)
        articles.drop(1).forEach { article ->
            NewsArticleCard(article = article, onOpenUrl = onOpenUrl)
        }
    }
    return true
}

@Composable
internal fun NewsLeadStoryCard(
    article: NewsArticleCardRow,
    onOpenUrl: (String) -> Unit
) {
    val darkTheme = isSystemInDarkTheme()
    val accent = if (darkTheme) Color(0xFF7DD3FC) else Color(0xFF0F6D9E)
    val shape = RoundedCornerShape(24.dp)
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(shape)
            .background(genUiCardContainerColor(GenUiCardTone.Neutral))
            .border(flatSpecCardBorder(), shape)
            .semantics(mergeDescendants = true) {
                contentDescription = "Lead story, ${article.title}"
            }
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.linearGradient(
                        listOf(
                            accent.copy(alpha = if (darkTheme) 0.18f else 0.10f),
                            MaterialTheme.colorScheme.surface.copy(alpha = 0.0f)
                        )
                    )
                )
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            NewsArticleImage(
                title = article.title,
                imageUrl = article.imageUrl,
                aspectRatio = 1.68f,
                onOpenUrl = onOpenUrl
            )
            NewsSourceLine(article = article, prominent = true, onOpenUrl = onOpenUrl)
            Text(
                text = parseBoldMarkdown(article.title),
                style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
                color = MaterialTheme.colorScheme.onSurface,
                lineHeight = MaterialTheme.typography.titleLarge.lineHeight,
                maxLines = 4,
                overflow = TextOverflow.Ellipsis
            )
            if (article.summary.isNotBlank()) {
                Text(
                    text = parseBoldMarkdown(article.summary),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 4,
                    overflow = TextOverflow.Ellipsis
                )
            }
            NewsMetaChips(article)
            NewsActions(article, onOpenUrl, primaryFullWidth = true)
        }
    }
}

@Composable
internal fun NewsArticleCard(
    article: NewsArticleCardRow,
    onOpenUrl: (String) -> Unit
) {
    val shape = RoundedCornerShape(20.dp)
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(shape)
            .background(genUiCardContainerColor(GenUiCardTone.Neutral))
            .border(flatSpecCardBorder(), shape)
            .semantics(mergeDescendants = true) {
                contentDescription = tableRowAccessibilitySummary(
                    headers = listOf("Article", "Source", "Published", "Summary"),
                    row = listOf(article.title, article.source, article.published, article.summary)
                )
            }
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.Top
        ) {
            NewsArticleImage(
                title = article.title,
                imageUrl = article.imageUrl,
                aspectRatio = 1f,
                compact = true,
                onOpenUrl = onOpenUrl
            )
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(7.dp)
            ) {
                NewsSourceLine(article = article, prominent = false, onOpenUrl = onOpenUrl)
                Text(
                    text = parseBoldMarkdown(article.title),
                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                    color = MaterialTheme.colorScheme.onSurface,
                    maxLines = 3,
                    overflow = TextOverflow.Ellipsis
                )
                if (article.summary.isNotBlank()) {
                    Text(
                        text = parseBoldMarkdown(article.summary),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 3,
                        overflow = TextOverflow.Ellipsis
                    )
                }
                NewsMetaChips(article)
                NewsActions(article, onOpenUrl, primaryFullWidth = false)
            }
        }
    }
}

@Composable
internal fun NewsArticleImage(
    title: String,
    imageUrl: String,
    aspectRatio: Float,
    compact: Boolean = false,
    onOpenUrl: (String) -> Unit
) {
    val baseModifier = if (compact) Modifier.width(104.dp) else Modifier.fillMaxWidth()
    val imageModifier = baseModifier
        .aspectRatio(aspectRatio)
        .clip(RoundedCornerShape(14.dp))
    val safeImageUrl = SafeContentPolicy.sanitizeMediaUrl(imageUrl, SafeContentPolicy.MediaKind.IMAGE)
    if (!safeImageUrl.isNullOrBlank()) {
        val context = LocalContext.current
        val imageLoader = rememberFlatImageLoader()
        var failed by remember(safeImageUrl) { mutableStateOf(false) }
        if (failed) {
            NewsImagePlaceholder(title = title, modifier = imageModifier)
        } else {
            AsyncImage(
                model = ImageRequest.Builder(context)
                    .data(safeImageUrl)
                    .applyFlatSpecRemoteImageHeaders(safeImageUrl)
                    .crossfade(true)
                    .allowHardware(false)
                    .build(),
                imageLoader = imageLoader,
                contentDescription = "$title image",
                contentScale = ContentScale.Crop,
                modifier = imageModifier,
                onSuccess = { failed = false },
                onError = {
                    failed = true
                    Log.w(FLAT_SPEC_RENDERER_TAG, "News image failed for URL '$safeImageUrl'.")
                }
            )
        }
    } else {
        NewsImagePlaceholder(title = title, modifier = imageModifier)
    }
}

@Composable
internal fun NewsImagePlaceholder(title: String, modifier: Modifier = Modifier) {
    val seed = kotlin.math.abs(title.hashCode().takeIf { it != Int.MIN_VALUE } ?: 0)
    val palette = listOf(
        Color(0xFF0F766E) to Color(0xFF38BDF8),
        Color(0xFF7C3AED) to Color(0xFFF97316),
        Color(0xFF1D4ED8) to Color(0xFF22C55E),
        Color(0xFFB45309) to Color(0xFFEF4444)
    )[seed.rem(4)]
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(14.dp))
            .background(Brush.linearGradient(listOf(palette.first, palette.second))),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = "NEWS",
            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold),
            color = Color.White,
            modifier = Modifier.padding(12.dp)
        )
    }
}

@Composable
internal fun NewsSourceLine(
    article: NewsArticleCardRow,
    prominent: Boolean,
    onOpenUrl: (String) -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(7.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        NewsSourceIcon(
            source = article.source,
            sourceIcon = article.sourceIcon,
            sourceUrl = article.sourceUrl,
            onOpenUrl = onOpenUrl
        )
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = article.source.ifBlank { "News source" },
                style = if (prominent) {
                    MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                } else {
                    MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
                },
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            if (article.published.isNotBlank()) {
                Text(
                    text = article.published,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
internal fun NewsSourceIcon(
    source: String,
    sourceIcon: String,
    sourceUrl: String,
    onOpenUrl: (String) -> Unit
) {
    val action = sourceUrl.takeIf { it.isNotBlank() }
    val safeIconUrl = SafeContentPolicy.sanitizeMediaUrl(sourceIcon, SafeContentPolicy.MediaKind.IMAGE)
    Box(
        modifier = Modifier
            .size(30.dp)
            .clip(RoundedCornerShape(8.dp))
            .then(
                if (!action.isNullOrBlank()) {
                    Modifier.clickable(role = Role.Button) { onOpenUrl(action) }
                } else {
                    Modifier
                }
            ),
        contentAlignment = Alignment.Center
    ) {
        if (!safeIconUrl.isNullOrBlank()) {
            RenderImage(
                props = mapOf(
                    "url" to safeIconUrl,
                    "fit" to "contain",
                    "aspectRatio" to 1f,
                    "alt" to "${source.ifBlank { "News source" }} icon"
                ),
                onOpenUrl = onOpenUrl,
                modifier = Modifier.fillMaxSize()
            )
        } else {
            Icon(
                imageVector = Icons.Filled.Language,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.72f),
                modifier = Modifier.size(18.dp)
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun NewsMetaChips(article: NewsArticleCardRow) {
    val chips = splitRestaurantTags(article.category).ifEmpty {
        listOfNotNull(article.published.takeIf { it.isNotBlank() })
    }.take(4)
    if (chips.isEmpty()) return
    FlowRow(
        horizontalArrangement = Arrangement.spacedBy(7.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        chips.forEach { label ->
            RestaurantTagChip(label)
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun NewsActions(
    article: NewsArticleCardRow,
    onOpenUrl: (String) -> Unit,
    primaryFullWidth: Boolean
) {
    val articleUrl = article.articleUrl.takeIf { it.isNotBlank() }
    val sourceUrl = article.sourceUrl.takeIf { it.isNotBlank() && it != articleUrl }
    if (articleUrl == null && sourceUrl == null) return
    FlowRow(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        if (articleUrl != null) {
            RestaurantActionPill(
                label = article.actionLabel.ifBlank { "Read Article" },
                icon = Icons.AutoMirrored.Filled.OpenInNew,
                primary = true,
                onClick = { onOpenUrl(articleUrl) }
            )
        }
        if (sourceUrl != null) {
            NewsTextAction(
                label = "Source",
                icon = Icons.Filled.Language,
                onClick = { onOpenUrl(sourceUrl) }
            )
        }
    }
}

@Composable
internal fun NewsTextAction(
    label: String,
    icon: ImageVector,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .heightIn(min = 40.dp)
            .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
            .clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            modifier = Modifier.size(16.dp),
            tint = MaterialTheme.colorScheme.primary
        )
        Spacer(modifier = Modifier.width(6.dp))
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
            color = MaterialTheme.colorScheme.primary,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

internal fun newsTitleIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("article", "headline", "title", "story", "news"))

internal fun newsSourceIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("source", "publisher", "publication"), exclude = setOfNotNull(newsSourceUrlIndex(headers), newsSourceIconIndex(headers)))

internal fun newsPublishedIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("published", "pub date", "date", "time"))

internal fun newsCategoryIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("category", "section", "topic", "tag"))

internal fun newsSummaryIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("summary", "description", "snippet", "excerpt", "content"))

internal fun newsImageIndex(headers: List<String>): Int? =
    findTableColumnIndex(headers, listOf("image url", "image", "thumbnail", "photo", "media"))

internal fun newsSourceIconIndex(headers: List<String>): Int? =
    headers.indices.firstOrNull { index ->
        val token = normalizeTableHeaderForMatch(headers[index])
        token.contains("source icon") || token.contains("publisher icon") || token == "icon"
    }

internal fun newsArticleUrlIndex(headers: List<String>): Int? =
    headers.indices.firstOrNull { index ->
        val token = normalizeTableHeaderForMatch(headers[index])
        (token.contains("article") && token.contains("url")) ||
            token == "article link" ||
            token == "link" ||
            token == "url" ||
            token.contains("read url")
    }

internal fun newsSourceUrlIndex(headers: List<String>): Int? =
    headers.indices.firstOrNull { index ->
        val token = normalizeTableHeaderForMatch(headers[index])
        token.contains("source url") || token.contains("publisher url") || token == "source link"
    }

internal fun newsActionLabelIndex(headers: List<String>): Int? =
    headers.indices.firstOrNull { index ->
        val token = normalizeTableHeaderForMatch(headers[index])
        token.contains("action label") || token.contains("button label") || token.contains("cta label")
    }


@Composable
internal fun BookingPhotoStrip(
    name: String,
    photos: List<String>,
    onOpenUrl: (String) -> Unit
) {
    if (photos.isEmpty()) return
    if (photos.size == 1) {
        RenderImage(
            props = mapOf(
                "url" to photos.first(),
                "fit" to "cover",
                "aspectRatio" to 1.72f,
                "alt" to "$name photo"
            ),
            onOpenUrl = onOpenUrl,
            modifier = Modifier.fillMaxWidth()
        )
        return
    }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        photos.take(5).forEachIndexed { index, photo ->
            RenderImage(
                props = mapOf(
                    "url" to photo,
                    "fit" to "cover",
                    "width" to if (index == 0) 248 else 210,
                    "height" to 150,
                    "alt" to "$name photo ${index + 1}"
                ),
                onOpenUrl = onOpenUrl,
                modifier = Modifier
            )
        }
    }
}

@Composable
internal fun BookingMetricChip(
    label: String,
    icon: ImageVector? = null,
    emphasized: Boolean = false
) {
    val containerColor = if (emphasized) {
        MaterialTheme.colorScheme.primaryContainer
    } else {
        MaterialTheme.colorScheme.secondaryContainer
    }
    val contentColor = if (emphasized) {
        MaterialTheme.colorScheme.onPrimaryContainer
    } else {
        MaterialTheme.colorScheme.onSecondaryContainer
    }
    Surface(
        shape = RoundedCornerShape(GenUiTokens.RadiusPill),
        color = containerColor,
        border = BorderStroke(1.dp, contentColor.copy(alpha = if (isSystemInDarkTheme()) 0.30f else 0.18f))
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
            horizontalArrangement = Arrangement.spacedBy(5.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (icon != null) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    modifier = Modifier.size(14.dp),
                    tint = contentColor
                )
            }
            Text(
                text = parseBoldMarkdown(label),
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                color = contentColor,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
internal fun BookingDetailChip(label: String) {
    Surface(
        shape = RoundedCornerShape(GenUiTokens.RadiusPill),
        color = MaterialTheme.colorScheme.surfaceContainerHighest,
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.70f))
    ) {
        Text(
            text = parseBoldMarkdown(label),
            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.padding(horizontal = 9.dp, vertical = 6.dp)
        )
    }
}

@Composable
internal fun ResponsiveComparisonRowCard(
    headers: List<String>,
    row: List<String>
) {
    val title = row.getOrNull(0).orEmpty().trim()
    val titleLabel = tableHeaderLabel(headers, 0)
    val leftLabel = tableHeaderLabel(headers, 1)
    val rightLabel = tableHeaderLabel(headers, 2)
    val leftValue = row.getOrNull(1).orEmpty().trim()
    val rightValue = row.getOrNull(2).orEmpty().trim()
    val sideBySide = leftValue.length <= 44 && rightValue.length <= 44 &&
        (leftValue.length + rightValue.length) <= 80

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
                .padding(horizontal = 10.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            if (title.isNotBlank()) {
                Text(
                    text = titleLabel,
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                    color = MaterialTheme.colorScheme.primary
                )
                Text(
                    text = parseBoldMarkdown(title),
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
            if (sideBySide) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    ResponsiveFieldBlock(
                        label = leftLabel,
                        value = leftValue,
                        modifier = Modifier.weight(1f)
                    )
                    ResponsiveFieldBlock(
                        label = rightLabel,
                        value = rightValue,
                        modifier = Modifier.weight(1f)
                    )
                }
            } else {
                ResponsiveFieldBlock(label = leftLabel, value = leftValue, modifier = Modifier.fillMaxWidth())
                ResponsiveFieldBlock(label = rightLabel, value = rightValue, modifier = Modifier.fillMaxWidth())
            }

            val trailingCount = maxOf(headers.size, row.size)
            (3 until trailingCount).forEach { index ->
                ResponsiveFieldBlock(
                    label = tableHeaderLabel(headers, index),
                    value = row.getOrNull(index).orEmpty(),
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }
    }
}

@Composable
internal fun ResponsiveComparisonColumnCards(
    headers: List<String>,
    rows: List<List<String>>,
    spacing: Dp,
    columns: List<FlatDirectTableColumn> = emptyList(),
    entityMedia: Map<String, TableEntityMedia> = emptyMap()
) {
    if (headers.size < 3 || rows.isEmpty()) return
    val featureHeader = headers.firstOrNull().orEmpty().ifBlank { "Feature" }
    val maxColumns = maxOf(headers.size, rows.maxOfOrNull { row -> row.size } ?: 0)
    if (maxColumns < 3) return

    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(spacing)
    ) {
        (1 until maxColumns).forEach { columnIndex ->
            val columnTitle = headers.getOrNull(columnIndex).orEmpty().ifBlank { "Option ${columnIndex}" }
            val items = rows.mapNotNull { row ->
                val feature = row.getOrNull(0).orEmpty().trim()
                val value = row.getOrNull(columnIndex).orEmpty().trim()
                if (feature.isBlank() || value.isBlank()) null else feature to value
            }.take(8)
            if (items.isEmpty()) return@forEach
            val media = entityMediaForColumn(
                columns = columns,
                headers = headers,
                columnIndex = columnIndex,
                entityMedia = entityMedia
            )

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics(mergeDescendants = true) {
                        contentDescription = buildString {
                            append(columnTitle)
                            items.forEach { (feature, value) ->
                                append(". ")
                                append(feature)
                                append(": ")
                                append(value)
                            }
                        }
                    },
                shape = RoundedCornerShape(16.dp),
                colors = flatSpecCardColors(),
                elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
                border = flatSpecCardBorder()
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 14.dp, vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    if (media != null) {
                        ComparisonEntityMediaTile(media = media)
                    }
                    Text(
                        text = parseBoldMarkdown(columnTitle),
                        style = MaterialTheme.typography.titleSmall,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    items.forEach { (feature, value) ->
                        FeatureMatrixField(feature = feature, value = value)
                    }
                }
            }
        }
    }
}

@Composable
internal fun ComparisonEntityMediaTile(media: TableEntityMedia) {
    RenderImage(
        props = mapOf(
            "url" to media.image,
            "alt" to media.alt,
            "fit" to "cover",
            "height" to 104
        ),
        onOpenUrl = {},
        modifier = Modifier.fillMaxWidth()
    )
}

@Composable
internal fun FeatureMatrixField(
    feature: String,
    value: String
) {
    val longPair = feature.length > 18 || value.length > 48 || value.contains('\n')
    val bulletItems = compactBulletItems(feature, value)
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(MaterialTheme.colorScheme.surfaceContainerHighest.copy(alpha = 0.58f))
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        if (longPair) {
            Text(
                text = parseBoldMarkdown(feature),
                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            if (bulletItems.isNotEmpty()) {
                Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                    bulletItems.forEach { item ->
                        Text(
                            text = parseBoldMarkdown("\u2022 $item"),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
            } else {
                Text(
                    text = parseBoldMarkdown(value),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
        } else {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.Top
            ) {
                Text(
                    text = parseBoldMarkdown(feature),
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(0.42f)
                )
                Text(
                    text = parseBoldMarkdown(value),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.weight(0.58f)
                )
            }
        }
    }
}




@Composable
internal fun RenderTimelineTableCards(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    spacing: Dp = 8.dp
) {
    if (rows.isEmpty()) return
    Column(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        verticalArrangement = Arrangement.spacedBy(spacing)
    ) {
        rows.forEach { row ->
            if (row.all { value -> value.trim().isEmpty() }) {
                return@forEach
            }
            ResponsiveScheduleRowCard(headers = headers, row = row)
        }
    }
}

@Composable
internal fun ResponsiveGenericRowCard(
    headers: List<String>,
    row: List<String>
) {
    val cells = responsiveTableCardCells(headers, row)
    if (cells.isEmpty()) return

    val titleCell = cells.first().takeIf { cell ->
        val value = cell.value
        cells.size > 1 && value.length in 3..42
    }
    val bodyCells = if (titleCell != null) cells.drop(1) else cells

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp)
            .semantics(mergeDescendants = true) {
                contentDescription = tableRowAccessibilitySummary(headers, row)
            },
        shape = RoundedCornerShape(16.dp),
        colors = flatSpecCardColors(),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 10.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            if (titleCell != null) {
                Text(
                    text = titleCell.label,
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                    color = MaterialTheme.colorScheme.primary
                )
                Text(
                    text = parseBoldMarkdown(titleCell.value),
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
            bodyCells.forEach { cell ->
                ResponsiveFieldBlock(
                    label = cell.label,
                    value = cell.value,
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }
    }
}

@Composable
internal fun RenderResponsiveTableRows(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    spacing: Dp = 8.dp,
    shape: FlatTableShape? = null
) {
    if (rows.isEmpty()) return
    if (looksLikeMarketHoldingsTable(headers, rows)) {
        RenderMarketHoldingsTable(
            headers = headers,
            rows = rows,
            modifier = modifier
        )
        return
    }
    val template = responsiveTableCardTemplate(shape, headers, rows)
    val featureMatrixStyle = template == ResponsiveTableCardTemplate.COMPARISON &&
        headers.size >= 3 &&
        isComparisonFeatureHeader(headers.firstOrNull().orEmpty())
    if (featureMatrixStyle) {
        Column(
            modifier = modifier
                .fillMaxWidth()
                .semantics {
                    contentDescription = tableAccessibilitySummary(headers, rows)
                }
        ) {
            ResponsiveComparisonColumnCards(
                headers = headers,
                rows = rows,
                spacing = spacing
            )
        }
        return
    }
    Column(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        verticalArrangement = Arrangement.spacedBy(spacing)
    ) {
        rows.forEach { row ->
            if (row.all { value -> value.trim().isEmpty() }) {
                return@forEach
            }
            when (template) {
                ResponsiveTableCardTemplate.COMPARISON -> ResponsiveComparisonRowCard(headers = headers, row = row)
                ResponsiveTableCardTemplate.SCHEDULE -> ResponsiveScheduleRowCard(headers = headers, row = row)
                ResponsiveTableCardTemplate.GENERIC -> ResponsiveGenericRowCard(headers = headers, row = row)
            }
        }
    }
}

internal fun responsiveTableCardTemplate(
    shape: FlatTableShape?,
    headers: List<String>,
    rows: List<List<String>>
): ResponsiveTableCardTemplate = when (shape) {
    FlatTableShape.SCHEDULE_TIMELINE -> ResponsiveTableCardTemplate.SCHEDULE
    else -> pickResponsiveTableTemplate(headers, rows)
}

internal fun selectAdaptiveTablePresentation(
    table: FlatDirectTableModel,
    screenWidthDp: Int,
    isLandscape: Boolean,
    autoHorizontalScroll: Boolean,
    cardsRequested: Boolean
): AdaptiveTablePresentation {
    val regularOrLandscape = isLandscape || screenWidthDp >= 600
    val columnCount = table.columns.size
    val featureMatrix = table.shape == FlatTableShape.FEATURE_MATRIX
    if (looksLikeClimateComparisonTable(
            headers = table.columns.map { column -> column.label },
            rows = table.rows,
            domain = table.domain
        )
    ) {
        return AdaptiveTablePresentation.CLIMATE_CARDS
    }
    if (looksLikeTravelItineraryTable(table.columns.map { column -> column.label }) ||
        looksLikePlaceStopItineraryTable(table.columns.map { column -> column.label }, table.rows)
    ) {
        return AdaptiveTablePresentation.ITINERARY_CARDS
    }
    if (table.shape == FlatTableShape.PLAYLIST) {
        return AdaptiveTablePresentation.PLAYLIST_ROWS
    }
    if (table.shape == FlatTableShape.KEY_VALUE) {
        return AdaptiveTablePresentation.KEY_VALUE_PANEL
    }
    if (!isLandscape && cardsRequested) {
        when (table.shape) {
            FlatTableShape.FEATURE_MATRIX -> return AdaptiveTablePresentation.FEATURE_CARDS
            FlatTableShape.ENTITY_ROW -> return AdaptiveTablePresentation.ENTITY_CARDS
            FlatTableShape.NUMERIC_METRICS -> return AdaptiveTablePresentation.METRIC_CARDS
            FlatTableShape.SCHEDULE_TIMELINE -> return AdaptiveTablePresentation.TIMELINE_CARDS
            FlatTableShape.KEY_VALUE -> return AdaptiveTablePresentation.KEY_VALUE_PANEL
            FlatTableShape.PLAYLIST -> return AdaptiveTablePresentation.PLAYLIST_ROWS
            FlatTableShape.GENERIC_GRID -> Unit
        }
    }
    if (regularOrLandscape) {
        return when {
            autoHorizontalScroll && featureMatrix -> AdaptiveTablePresentation.STICKY_HORIZONTAL_TABLE
            autoHorizontalScroll || columnCount >= 5 -> AdaptiveTablePresentation.HORIZONTAL_TABLE
            else -> AdaptiveTablePresentation.TABLE
        }
    }
    return when (table.shape) {
        FlatTableShape.PLAYLIST -> AdaptiveTablePresentation.PLAYLIST_ROWS
        FlatTableShape.KEY_VALUE -> AdaptiveTablePresentation.KEY_VALUE_PANEL
        FlatTableShape.SCHEDULE_TIMELINE -> AdaptiveTablePresentation.TIMELINE_CARDS
        FlatTableShape.FEATURE_MATRIX -> AdaptiveTablePresentation.FEATURE_CARDS
        FlatTableShape.ENTITY_ROW -> AdaptiveTablePresentation.ENTITY_CARDS
        FlatTableShape.NUMERIC_METRICS -> AdaptiveTablePresentation.METRIC_CARDS
        FlatTableShape.GENERIC_GRID -> when {
            cardsRequested && columnCount <= 3 -> AdaptiveTablePresentation.ENTITY_CARDS
            autoHorizontalScroll && columnCount <= 3 -> AdaptiveTablePresentation.ENTITY_CARDS
            autoHorizontalScroll -> AdaptiveTablePresentation.HORIZONTAL_TABLE
            else -> AdaptiveTablePresentation.TABLE
        }
    }
}





internal fun looksLikeTravelItinerarySummaryTable(
    headers: List<String>,
    rows: List<List<String>>
): Boolean {
    if (headers.size > 2 || rows.size !in 2..8) return false
    val labelTokens = rows
        .mapNotNull { row -> row.getOrNull(0)?.trim()?.takeIf { it.isNotBlank() } }
        .map(::normalizeTableHeaderForMatch)
    if (labelTokens.isEmpty()) return false
    val summarySignals = labelTokens.count { label ->
        label in setOf(
            "trip length",
            "places listed",
            "places included",
            "top rated place",
            "main focus",
            "destination",
            "city",
            "days",
            "duration",
            "style",
            "itinerary focus",
            "itinerary style"
        ) ||
            label.contains("trip length") ||
            label.contains("places listed") ||
            label.contains("places included") ||
            label.contains("top rated") ||
            label.contains("main focus") ||
            label == "city" ||
            label == "style" ||
            label.contains("itinerary focus")
    }
    val travelSignals = labelTokens.any { label ->
        label.contains("trip") ||
            label.contains("itinerary") ||
            label.contains("places") ||
            label.contains("destination") ||
            label.contains("top rated")
    }
    return summarySignals >= 2 && travelSignals
}

@Composable
internal fun RenderKeyValueTablePanel(
    headers: List<String>,
    rows: List<List<String>>,
    title: String? = null,
    modifier: Modifier = Modifier
) {
    if (rows.isEmpty()) return
    val panelTitle = title?.trim()?.takeIf { it.isNotBlank() }
    Card(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        colors = CardDefaults.cardColors(containerColor = genUiCardContainerColor(GenUiCardTone.Neutral)),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        shape = RoundedCornerShape(20.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            if (panelTitle != null) {
                Text(
                    text = parseBoldMarkdown(panelTitle),
                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier
                        .padding(horizontal = 2.dp, vertical = 2.dp)
                        .semantics { heading() }
                )
            }
            rows.forEachIndexed { index, row ->
                val label = row.getOrNull(0).orEmpty().trim().ifBlank { tableHeaderLabel(headers, 0) }
                val value = row.getOrNull(1).orEmpty().trim()
                if (label.isBlank() && value.isBlank()) return@forEachIndexed
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .semantics(mergeDescendants = true) {
                            contentDescription = "$label: $value"
                        },
                    shape = RoundedCornerShape(16.dp),
                    color = MaterialTheme.colorScheme.surfaceContainerHighest.copy(
                        alpha = if (isSystemInDarkTheme()) 0.20f else 0.38f
                    ),
                    border = BorderStroke(
                        1.dp,
                        MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.26f)
                    )
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 14.dp, vertical = 13.dp),
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                        verticalAlignment = Alignment.Top
                    ) {
                        Text(
                            text = parseBoldMarkdown(label.uppercase(Locale.US)),
                            style = MaterialTheme.typography.labelSmall.copy(
                                fontWeight = FontWeight.Black,
                                letterSpacing = MaterialTheme.typography.labelSmall.letterSpacing
                            ),
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.weight(0.38f)
                        )
                        Text(
                            text = parseBoldMarkdown(value),
                            style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold),
                            color = MaterialTheme.colorScheme.onSurface,
                            overflow = TextOverflow.Clip,
                            modifier = Modifier.weight(0.62f)
                        )
                    }
                }
            }
        }
    }
}


internal fun playlistNumberColumnIndex(headers: List<String>): Int? {
    return headers.indexOfFirst { header ->
        val raw = header.trim().lowercase()
        val token = normalizeTableHeaderForMatch(header)
        raw == "#" || token in setOf("no", "number", "track number", "track no", "tracknumber")
    }.takeIf { it >= 0 }
}

internal fun playlistTitleColumnIndex(headers: List<String>): Int? {
    return headers.indexOfFirst { header ->
        val token = normalizeTableHeaderForMatch(header)
        token == "track" ||
            token == "song" ||
            token == "title" ||
            token.contains("track title") ||
            token.contains("song title")
    }.takeIf { it >= 0 }
}

internal fun playlistArtistColumnIndex(headers: List<String>): Int? {
    return headers.indexOfFirst { header ->
        val token = normalizeTableHeaderForMatch(header)
        token == "artist" ||
            token.contains("artist") ||
            token.contains("performer") ||
            token.contains("band")
    }.takeIf { it >= 0 }
}

internal fun playlistChipColumnIndexes(
    headers: List<String>,
    usedIndexes: Set<Int>
): List<Int> {
    return headers.indices.filterNot { it in usedIndexes }.filter { index ->
        val token = normalizeTableHeaderForMatch(headers[index])
        token.contains("mood") ||
            token.contains("genre") ||
            token.contains("tempo") ||
            token.contains("album") ||
            token.contains("duration") ||
            token.contains("era")
    }
}

internal fun splitPlaylistTrack(raw: String): Pair<String?, String> {
    val text = raw.trim()
    if (text.isBlank()) return null to ""
    val parts = text.split(Regex("""\s+[-\u2013\u2014]\s+"""), limit = 2)
    if (parts.size == 2 && parts[0].isNotBlank() && parts[1].isNotBlank()) {
        return parts[0].trim() to parts[1].trim()
    }
    val byParts = text.split(Regex("""\s+by\s+""", RegexOption.IGNORE_CASE), limit = 2)
    if (byParts.size == 2 && byParts[0].isNotBlank() && byParts[1].isNotBlank()) {
        return byParts[1].trim() to byParts[0].trim()
    }
    return null to text
}

internal fun playlistStringListProp(value: Any?): List<String> {
    return when (value) {
        is List<*> -> value.mapNotNull { it?.toString()?.trim()?.takeIf(String::isNotBlank) }
        is String -> value
            .split(',', '|')
            .mapNotNull { it.trim().takeIf(String::isNotBlank) }
        else -> emptyList()
    }
}

internal fun buildPlaylistTrackRows(
    headers: List<String>,
    rows: List<List<String>>
): List<PlaylistTrackRow> {
    val numberIndex = playlistNumberColumnIndex(headers)
    val titleIndex = playlistTitleColumnIndex(headers)
    val artistIndex = playlistArtistColumnIndex(headers)
    val usedIndexes = listOfNotNull(numberIndex, titleIndex, artistIndex).toSet()
    val chipIndexes = playlistChipColumnIndexes(headers, usedIndexes)
    return rows.mapIndexed { rowIndex, row ->
        val rawTrack = titleIndex?.let { row.getOrNull(it) }.orEmpty().trim()
            .ifBlank {
                row.indices
                    .firstOrNull { it != numberIndex && row.getOrNull(it).orEmpty().isNotBlank() }
                    ?.let { row.getOrNull(it).orEmpty().trim() }
                    .orEmpty()
            }
        val (parsedArtist, parsedTitle) = splitPlaylistTrack(rawTrack)
        PlaylistTrackRow(
            number = numberIndex?.let { row.getOrNull(it).orEmpty().trim() }
                ?.takeIf { it.isNotBlank() }
                ?: (rowIndex + 1).toString(),
            title = parsedTitle.ifBlank { "Track ${rowIndex + 1}" },
            artist = artistIndex?.let { row.getOrNull(it).orEmpty().trim() }
                ?.takeIf { it.isNotBlank() }
                ?: parsedArtist,
            chips = chipIndexes
                .mapNotNull { index -> row.getOrNull(index)?.trim()?.takeIf { it.isNotBlank() && !isLikelyHttpUrl(it) } }
                .take(3),
            sourceRow = row
        )
    }
}

internal fun climateColumnIndex(headers: List<String>, vararg keywords: String): Int? {
    return headers.indexOfFirst { header ->
        val token = normalizeTableHeaderForMatch(header)
        keywords.any { keyword -> token.contains(keyword) }
    }.takeIf { it >= 0 }
}

internal fun compactClimateMetricLabel(header: String): String {
    val token = normalizeTableHeaderForMatch(header)
    return when {
        token.contains("rain") || token.contains("precip") -> "Rain"
        token.contains("sunshine") || token.contains("sunny") -> "Sun"
        token.contains("humidity") -> "Humidity"
        token.contains("wind") -> "Wind"
        token.contains("uv") -> "UV"
        token.contains("best") -> "Best for"
        token.contains("condition") -> "Condition"
        else -> tableHeaderLabel(listOf(header), 0)
    }
}

internal fun firstNumberFromText(value: String?): Double? {
    return Regex("""-?\d+(?:\.\d+)?""")
        .find(value.orEmpty())
        ?.value
        ?.toDoubleOrNull()
}


internal fun buildClimateComparisonRows(
    headers: List<String>,
    rows: List<List<String>>
): List<ClimateComparisonRow> {
    if (headers.isEmpty()) return emptyList()
    val placeIndex = 0
    val verdictIndex = climateColumnIndex(headers, "verdict", "condition", "summary")
    val highIndex = climateColumnIndex(headers, "average high", "avg high", "high", "max")
    val lowIndex = climateColumnIndex(headers, "average low", "avg low", "low", "min")
    val reserved = setOfNotNull(placeIndex, verdictIndex, highIndex, lowIndex)
    return rows.mapIndexedNotNull { rowIndex, row ->
        val place = row.getOrNull(placeIndex).orEmpty().trim().ifBlank { "Place ${rowIndex + 1}" }
        if (place.isBlank()) return@mapIndexedNotNull null
        val metrics = headers.indices
            .filterNot { it in reserved }
            .mapNotNull { index ->
                val value = row.getOrNull(index).orEmpty().trim()
                if (value.isBlank() || isLikelyHttpUrl(value)) null else compactClimateMetricLabel(headers[index]) to value
            }
            .take(4)
        ClimateComparisonRow(
            place = place,
            verdict = verdictIndex?.let { row.getOrNull(it).orEmpty().trim() }?.takeIf { it.isNotBlank() },
            high = highIndex?.let { row.getOrNull(it).orEmpty().trim() }?.takeIf { it.isNotBlank() },
            low = lowIndex?.let { row.getOrNull(it).orEmpty().trim() }?.takeIf { it.isNotBlank() },
            metrics = metrics,
            sourceRow = row
        )
    }
}


internal fun parseChartNumber(value: String): Double? {
    val match = Regex("""-?\d[\d,]*(?:\.\d+)?""").find(value) ?: return null
    return match.value.replace(",", "").toDoubleOrNull()
}

internal fun chartColumnIndex(
    columns: List<FlatDirectTableColumn>,
    explicitKey: String?,
    fallbackIndex: Int
): Int {
    val normalizedKey = explicitKey?.let(::normalizeColumnToken)
    if (!normalizedKey.isNullOrBlank()) {
        val match = columns.indexOfFirst { column ->
            normalizeColumnToken(column.key) == normalizedKey ||
                normalizeColumnToken(column.label) == normalizedKey
        }
        if (match >= 0) return match
    }
    return fallbackIndex.coerceIn(0, (columns.size - 1).coerceAtLeast(0))
}

internal fun extractChartPoints(
    props: Map<String, Any?>,
    state: Map<String, Any?>
): List<ChartPoint> {
    val rawRows = (props["rows"] as? List<*>)
        ?: (props["data"] as? List<*>)
        ?: resolveDirectTableRows(props, state)
    if (rawRows.isEmpty()) return emptyList()
    val columns = resolveDirectTableColumns(props, rawRows)
    if (columns.size < 2) return emptyList()
    val xIndex = chartColumnIndex(columns, props["xKey"]?.toString(), 0)
    val yIndex = chartColumnIndex(columns, props["yKey"]?.toString(), 1)
    if (xIndex == yIndex) return emptyList()

    return rawRows.mapNotNull { row ->
        val resolved = resolveDirectTableRow(row, columns, state)
        val label = resolved.getOrNull(xIndex).orEmpty().trim()
        val displayValue = resolved.getOrNull(yIndex).orEmpty().trim()
        val value = parseChartNumber(displayValue)
        if (label.isBlank() || displayValue.isBlank() || value == null) {
            null
        } else {
            ChartPoint(label = label, value = value, displayValue = displayValue)
        }
    }
}

internal fun formatChartNumber(value: Double, currencyPrefix: String?): String {
    val rounded = value.roundToInt()
    val formatted = "%,d".format(rounded)
    return if (currencyPrefix.isNullOrBlank()) formatted else "$currencyPrefix$formatted"
}

internal fun inferChartCurrency(points: List<ChartPoint>): String? {
    val value = points.firstOrNull { it.displayValue.trim().startsWith("$") } ?: return null
    return value.displayValue.trim().takeWhile { !it.isDigit() && it != '-' }.takeIf { it.isNotBlank() }
}

internal fun extractPercentageMatrixChartModel(
    columns: List<FlatDirectTableColumn>,
    rows: List<List<String>>,
    domain: String
): MultiSeriesChartModel? {
    if (columns.size < 3 || rows.size !in 2..12) return null
    if (domain in setOf("weather", "flight", "booking", "restaurants", "playlist", "schedule", "status")) return null

    val categoryLabel = columns.firstOrNull()?.label?.trim().orEmpty()
    val categoryToken = normalizeTableHeaderForMatch(categoryLabel)
    if (categoryLabel.isBlank() || categoryToken in setOf("feature", "metric", "attribute", "criteria")) {
        return null
    }

    val candidateSeriesIndexes = columns.indices.drop(1).filter { columnIndex ->
        val values = rows.mapNotNull { row -> row.getOrNull(columnIndex)?.trim()?.takeIf { it.isNotBlank() } }
        values.isNotEmpty() && values.count { value -> parseChartNumber(value) != null } >= maxOf(1, values.size * 2 / 3)
    }
    if (candidateSeriesIndexes.size < 2 || candidateSeriesIndexes.size > 6) return null

    val candidateValues = candidateSeriesIndexes.flatMap { columnIndex ->
        rows.mapNotNull { row -> row.getOrNull(columnIndex)?.trim()?.takeIf { it.isNotBlank() } }
    }
    val percentLikeCount = candidateValues.count { value -> value.contains('%') }
    if (percentLikeCount < maxOf(2, candidateValues.size * 2 / 3)) return null

    val chartRows = rows.mapNotNull { row ->
        val label = row.getOrNull(0)?.trim().orEmpty()
        if (label.isBlank()) return@mapNotNull null
        val segments = candidateSeriesIndexes.mapNotNull { columnIndex ->
            val displayValue = row.getOrNull(columnIndex)?.trim().orEmpty()
            val value = parseChartNumber(displayValue)
            if (value == null) {
                null
            } else {
                val column = columns[columnIndex]
                MultiSeriesChartSegment(
                    key = column.key,
                    label = column.label.ifBlank { column.key },
                    value = value.coerceAtLeast(0.0),
                    displayValue = displayValue
                )
            }
        }
        if (segments.size < 2) null else MultiSeriesChartRow(label = label, segments = segments)
    }

    if (chartRows.size < 2) return null
    return MultiSeriesChartModel(
        categoryLabel = categoryLabel,
        rows = chartRows,
        percentBased = true
    )
}


@Composable
internal fun ChartSummaryChip(text: String) {
    Surface(
        shape = RoundedCornerShape(GenUiTokens.RadiusPill),
        color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.62f)
    ) {
        Text(
            text = parseBoldMarkdown(text),
            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
            color = MaterialTheme.colorScheme.onPrimaryContainer,
            modifier = Modifier.padding(horizontal = 9.dp, vertical = 5.dp)
        )
    }
}


@Composable
internal fun ChartLegendChip(
    label: String,
    color: Color
) {
    Surface(
        shape = RoundedCornerShape(GenUiTokens.RadiusPill),
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.74f),
        border = BorderStroke(1.dp, color.copy(alpha = 0.24f))
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 9.dp, vertical = 5.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                    .background(color)
            )
            Text(
                text = parseBoldMarkdown(label),
                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                color = MaterialTheme.colorScheme.onSurface
            )
        }
    }
}

@Composable
internal fun PercentageMatrixChartRow(
    row: MultiSeriesChartRow,
    colorBySeries: Map<String, Color>,
    compact: Boolean
) {
    val positiveSegments = row.segments.filter { segment -> segment.value > 0.0 }
    val total = positiveSegments.sumOf { segment -> segment.value }.takeIf { it > 0.0 } ?: 1.0
    val winner = row.segments.maxByOrNull { segment -> segment.value }

    Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = parseBoldMarkdown(row.label),
                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold),
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f)
            )
            winner?.let { segment ->
                Text(
                    text = "${segment.label} ${segment.displayValue}",
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.End,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(if (compact) 1.45f else 1f)
                )
            }
        }
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(if (compact) 30.dp else 26.dp)
                .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.52f))
        ) {
            if (positiveSegments.isNotEmpty()) {
                Row(modifier = Modifier.fillMaxSize()) {
                    positiveSegments.forEach { segment ->
                        Box(
                            modifier = Modifier
                                .fillMaxHeight()
                                .weight((segment.value / total).toFloat().coerceAtLeast(0.01f))
                                .background(colorBySeries[segment.label] ?: MaterialTheme.colorScheme.primary)
                        )
                    }
                }
            }
        }
    }
}



@Composable
internal fun PlaylistCoverArt(
    trackCount: Int,
    compact: Boolean = false,
    modifier: Modifier = Modifier
) {
    val padding = if (compact) 10.dp else 18.dp
    val numberStyle = if (compact) {
        MaterialTheme.typography.headlineLarge.copy(fontWeight = FontWeight.Bold)
    } else {
        MaterialTheme.typography.displaySmall.copy(fontWeight = FontWeight.Bold)
    }
    val labelStyle = if (compact) {
        MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
    } else {
        MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
    }
    val tracksStyle = if (compact) {
        MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold)
    } else {
        MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.SemiBold)
    }
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(26.dp))
            .background(
                Brush.linearGradient(
                    colors = listOf(
                        Color(0xFF050510),
                        Color(0xFF283179),
                        Color(0xFF08B6A2)
                    )
                )
            )
            .semantics {
                contentDescription = "Generated playlist cover art"
            }
            .padding(padding)
    ) {
        Box(
            modifier = Modifier
                .matchParentSize()
                .background(
                    Brush.radialGradient(
                        colors = listOf(
                            Color.White.copy(alpha = 0.18f),
                            Color.Transparent
                        )
                    )
                )
        )
        Column(
            modifier = Modifier.align(Alignment.BottomStart),
            verticalArrangement = Arrangement.spacedBy(2.dp)
        ) {
            Text(
                text = "PLAYLIST",
                style = labelStyle,
                color = Color.White.copy(alpha = 0.72f),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                text = "$trackCount",
                style = numberStyle,
                color = Color.White,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                text = if (trackCount == 1) "track" else "tracks",
                style = tracksStyle,
                color = Color.White.copy(alpha = 0.8f),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun PlaylistMetaTags(tags: List<String>) {
    if (tags.isEmpty()) return
    FlowRow(
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        tags.take(4).forEach { tag ->
            Surface(
                shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                color = Color.White.copy(alpha = 0.12f)
            ) {
                Text(
                    text = parseBoldMarkdown(tag),
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                    color = Color.White.copy(alpha = 0.88f),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.padding(horizontal = 9.dp, vertical = 5.dp)
                )
            }
        }
    }
}



@Composable
internal fun PlaylistTrackList(
    headers: List<String>,
    tracks: List<PlaylistTrackRow>,
    dense: Boolean,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(if (dense) 6.dp else 8.dp)
    ) {
        tracks.forEachIndexed { rowIndex, track ->
            PlaylistTrackRowView(
                headers = headers,
                track = track,
                rowIndex = rowIndex,
                dense = dense
            )
        }
    }
}


@Composable
internal fun EntityProviderBadge(title: String) {
    val accent = entityProviderAccentColor(title)
    val initials = entityProviderInitials(title)
    Box(
        modifier = Modifier
            .size(38.dp)
            .clip(RoundedCornerShape(13.dp))
            .background(
                Brush.linearGradient(
                    listOf(
                        accent,
                        accent.copy(alpha = 0.68f)
                    )
                )
            ),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = initials,
            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Black),
            color = Color.White,
            maxLines = 1,
            overflow = TextOverflow.Clip
        )
    }
}

@Composable
internal fun entityProviderAccentColor(title: String): Color {
    val normalized = normalizeTableHeaderForMatch(title)
    return when {
        normalized.contains("google") -> Color(0xFF4285F4)
        normalized.contains("sky") -> Color(0xFF00A7E1)
        normalized.contains("makemytrip") || normalized.contains("make my trip") -> Color(0xFFEF3E42)
        normalized.contains("cleartrip") -> Color(0xFFFF7A00)
        normalized.contains("ixigo") -> Color(0xFFFF6D00)
        normalized.contains("booking") -> Color(0xFF003B95)
        normalized.contains("expedia") -> Color(0xFFFFC72C)
        else -> MaterialTheme.colorScheme.primary
    }
}

internal fun entityProviderInitials(title: String): String {
    val normalized = normalizeTableHeaderForMatch(title)
    val explicit = when {
        normalized.contains("google") -> "G"
        normalized.contains("skyscanner") -> "SS"
        normalized.contains("makemytrip") || normalized.contains("make my trip") -> "MMT"
        normalized.contains("cleartrip") -> "CT"
        normalized.contains("ixigo") -> "IX"
        else -> ""
    }
    if (explicit.isNotBlank()) return explicit
    return title
        .split(Regex("""[^A-Za-z0-9]+"""))
        .filter { it.isNotBlank() }
        .take(2)
        .mapNotNull { it.firstOrNull()?.uppercaseChar()?.toString() }
        .joinToString("")
        .ifBlank { "UI" }
        .take(3)
}

internal fun shouldShowEntityProviderBadge(title: String, actionUrl: String?): Boolean {
    if (!actionUrl.isNullOrBlank()) return true
    val normalized = normalizeTableHeaderForMatch(title)
    return normalized.contains("google") ||
        normalized.contains("skyscanner") ||
        normalized.contains("makemytrip") ||
        normalized.contains("make my trip") ||
        normalized.contains("cleartrip") ||
        normalized.contains("ixigo") ||
        normalized.contains("booking") ||
        normalized.contains("expedia")
}



internal fun entityActionLabel(headers: List<String>, row: List<String>, title: String): String {
    val explicit = headers.indices
        .firstOrNull { index -> isActionLabelColumn(headers[index]) }
        ?.let { row.getOrNull(it).orEmpty().trim() }
        .orEmpty()
    if (explicit.isNotBlank() && !isLikelyHttpUrl(explicit)) {
        return explicit.take(28)
    }
    val compactTitle = title
        .replace(Regex("""\s+"""), " ")
        .trim()
        .take(22)
        .trim()
    return if (compactTitle.isBlank()) "Open" else "Open $compactTitle"
}


@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun RenderMetricTableCards(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    spacing: Dp = 8.dp,
    numericColumns: Set<Int>
) {
    if (rows.isEmpty()) return
    FlowRow(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        horizontalArrangement = Arrangement.spacedBy(spacing),
        verticalArrangement = Arrangement.spacedBy(spacing)
    ) {
        rows.forEachIndexed { rowIndex, row ->
            val title = row.getOrNull(0).orEmpty().trim().ifBlank { "Metric ${rowIndex + 1}" }
            val valueIndex = numericColumns.firstOrNull { it != 0 && row.getOrNull(it).orEmpty().isNotBlank() }
                ?: row.indices.firstOrNull { it != 0 && row.getOrNull(it).orEmpty().isNotBlank() }
            val value = valueIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty()
            Card(
                modifier = Modifier
                    .weight(1f)
                    .widthIn(min = 142.dp)
                    .semantics(mergeDescendants = true) {
                        contentDescription = tableRowAccessibilitySummary(headers, row, rowIndex)
                    },
                shape = RoundedCornerShape(16.dp),
                colors = flatSpecCardColors(),
                elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
            ) {
                Column(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Text(
                        text = parseBoldMarkdown(title),
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis
                    )
                    Text(
                        text = parseBoldMarkdown(value),
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
            }
        }
    }
}

internal fun flightCarrierColumnIndex(headers: List<String>): Int {
    val explicit = headers.indexOfFirst { header ->
        val normalized = normalizeTableHeaderForMatch(header)
        normalized.contains("carrier") || normalized.contains("airline")
    }
    return explicit.takeIf { it >= 0 } ?: 0
}

internal fun flightLegColumnIndexes(headers: List<String>): List<Int> {
    return headers.indices.filter { index ->
        val normalized = normalizeTableHeaderForMatch(headers[index])
        normalized.contains("leg") ||
            Regex("""\b[a-z]{3}\s*[/-]\s*[a-z]{3}\b""").containsMatchIn(normalized)
    }
}

internal fun flightLegBadge(header: String, index: Int): String {
    val prefix = header.substringBefore(":").trim()
    return prefix.takeIf { it.isNotBlank() && it.length <= 12 } ?: "Leg ${index + 1}"
}

internal fun flightLegRoute(header: String): String {
    val route = header.substringAfter(":", missingDelimiterValue = "").trim()
    return route.takeIf { it.isNotBlank() } ?: header.trim()
}

internal data class RankedFlightEndpoint(
    val code: String?,
    val time: String?
)

internal data class RankedFlightLeg(
    val fromCode: String,
    val fromTime: String,
    val toCode: String,
    val toTime: String,
    val detail: String?
)

@Composable
internal fun RankedFlightAirlineBadge(
    airline: String,
    rank: String,
    best: Boolean,
    logoUrl: String?
) {
    val accent = rankedFlightAccentColor(airline)
    val code = NativeFlightSemantics.airlineBadgeCode(airline).ifBlank { rank.removePrefix("#") }
    val safeLogo = remember(airline, logoUrl) {
        rankedFlightLogoUrl(airline, logoUrl)
    }
    var logoFailed by remember(safeLogo) { mutableStateOf(false) }
    val context = LocalContext.current
    val imageLoader = rememberFlatImageLoader()
    Box(
        modifier = Modifier
            .size(42.dp)
            .clip(RoundedCornerShape(GenUiTokens.RadiusPill)),
        contentAlignment = Alignment.Center
    ) {
        if (safeLogo != null && !logoFailed) {
            AsyncImage(
                model = ImageRequest.Builder(context)
                    .data(safeLogo)
                    .crossfade(true)
                    .allowHardware(false)
                    .build(),
                imageLoader = imageLoader,
                contentDescription = "$airline logo",
                contentScale = ContentScale.Fit,
                modifier = Modifier
                    .fillMaxSize()
                    .padding(6.dp),
                onError = { logoFailed = true }
            )
        } else {
            Text(
                text = code.take(3),
                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Black),
                color = accent,
                maxLines = 1,
                overflow = TextOverflow.Clip
            )
        }
    }
}

internal fun rankedFlightLogoUrl(airline: String, explicitLogoUrl: String?): String? {
    SafeContentPolicy.sanitizeMediaUrl(explicitLogoUrl, SafeContentPolicy.MediaKind.IMAGE)
        ?.let { return it }
    val code = NativeFlightSemantics.airlineBadgeCode(airline).trim()
    if (code.length !in 2..3 || !code.all { it.isLetterOrDigit() }) {
        return null
    }
    return "https://www.gstatic.com/flights/airline_logos/70px/${code.uppercase(Locale.US)}.png"
}


@Composable
internal fun RankedFlightEndpointCell(
    code: String?,
    time: String?,
    align: TextAlign,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        horizontalAlignment = if (align == TextAlign.End) Alignment.End else Alignment.Start,
        verticalArrangement = Arrangement.spacedBy(1.dp)
    ) {
        code?.takeIf { it.isNotBlank() }?.let {
            Text(
                text = it,
                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Black),
                color = MaterialTheme.colorScheme.onSurface,
                textAlign = align,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
        time?.takeIf { it.isNotBlank() }?.let {
            Text(
                text = it,
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = align,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}


@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun RankedFlightChipRow(chips: List<String>, accent: Color) {
    if (chips.isEmpty()) return
    FlowRow(
        horizontalArrangement = Arrangement.spacedBy(7.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        chips.take(5).forEach { chip ->
            Surface(
                shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                color = MaterialTheme.colorScheme.surfaceContainerHighest.copy(alpha = 0.82f),
                border = BorderStroke(GenUiTokens.BorderSm, accent.copy(alpha = 0.16f))
            ) {
                Text(
                    text = chip,
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 9.dp, vertical = 5.dp),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
internal fun RankedFlightBestChip(text: String, accent: Color) {
    Surface(
        shape = RoundedCornerShape(GenUiTokens.RadiusPill),
        color = accent.copy(alpha = 0.12f),
        border = BorderStroke(GenUiTokens.BorderSm, accent.copy(alpha = 0.28f))
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
            color = accent,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}


@Composable
internal fun rankedFlightAccentColor(airline: String): Color {
    val dark = isSystemInDarkTheme()
    val normalized = NativeFlightSemantics.normalizeMatchText(airline)
    return when {
        normalized.contains("indigo") -> if (dark) Color(0xFF83A3FF) else Color(0xFF2849B8)
        normalized.contains("akasa") -> if (dark) Color(0xFFD989F2) else Color(0xFF7A2E8E)
        normalized.contains("air india express") -> if (dark) Color(0xFFFF8A8A) else Color(0xFFE53935)
        normalized == "air india" || normalized.startsWith("air india ") -> if (dark) Color(0xFFFF8A8A) else Color(0xFFC62828)
        normalized.contains("vistara") -> if (dark) Color(0xFFD389F2) else Color(0xFF54206E)
        normalized.contains("spicejet") -> if (dark) Color(0xFFFFA074) else Color(0xFFD84315)
        normalized.contains("emirates") -> if (dark) Color(0xFFFF8A8A) else Color(0xFFB71C1C)
        normalized.contains("qatar") -> if (dark) Color(0xFFFF8FB7) else Color(0xFF7B1238)
        else -> MaterialTheme.colorScheme.primary
    }
}


internal fun isFlightActionUrlColumn(header: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(header)
    return normalized.contains("booking url") ||
        normalized.contains("book url") ||
        normalized.contains("action url") ||
        normalized.contains("cta url") ||
        normalized == "url" ||
        normalized == "link"
}

internal fun parseRankedFlightEndpoint(raw: String): RankedFlightEndpoint {
    val point = NativeFlightSemantics.parseFlightPoint(raw, fallbackCode = null)
    val code = point.code
        ?: Regex("""\b([A-Z]{3})\b""")
            .find(raw.uppercase(Locale.US))
            ?.groupValues
            ?.getOrNull(1)
    val time = point.time ?: NativeFlightSemantics.normalizeFlightTime(raw)
    return RankedFlightEndpoint(
        code = code?.takeIf { it.isNotBlank() },
        time = time?.takeIf { it.isNotBlank() }
    )
}

internal fun parseRankedFlightLegs(raw: String): List<RankedFlightLeg> {
    if (raw.isBlank()) return emptyList()
    val legPattern = Regex(
        """^\s*([A-Z]{3})\s+(\d{1,2}:\d{2}(?:\s?[AP]M)?(?:\+\d+)?)\s*->\s*([A-Z]{3})\s+(\d{1,2}:\d{2}(?:\s?[AP]M)?(?:\+\d+)?)\s*(?:\((.*)\))?\s*$""",
        RegexOption.IGNORE_CASE
    )
    return raw.split(';')
        .mapNotNull { segment ->
            val cleaned = NativeTextFormatter.sanitizeDisplayText(segment).trim()
            if (cleaned.isBlank()) return@mapNotNull null
            val match = legPattern.find(cleaned)
            if (match != null) {
                RankedFlightLeg(
                    fromCode = match.groupValues[1].uppercase(Locale.US),
                    fromTime = match.groupValues[2],
                    toCode = match.groupValues[3].uppercase(Locale.US),
                    toTime = match.groupValues[4],
                    detail = match.groupValues.getOrNull(5)?.takeIf { it.isNotBlank() }
                )
            } else {
                val endpointCodes = Regex("""\b([A-Z]{3})\b""")
                    .findAll(cleaned.uppercase(Locale.US))
                    .map { it.groupValues[1] }
                    .take(2)
                    .toList()
                val times = Regex("""\b\d{1,2}:\d{2}(?:\s?[AP]M)?(?:\+\d+)?\b""", RegexOption.IGNORE_CASE)
                    .findAll(cleaned)
                    .map { it.value }
                    .take(2)
                    .toList()
                if (endpointCodes.size >= 2 && times.size >= 2) {
                    RankedFlightLeg(
                        fromCode = endpointCodes[0],
                        fromTime = times[0],
                        toCode = endpointCodes[1],
                        toTime = times[1],
                        detail = cleaned.substringAfter(')', missingDelimiterValue = "")
                            .trim()
                            .takeIf { it.isNotBlank() }
                    )
                } else {
                    null
                }
            }
        }
}

internal fun rankedFlightSubtitle(
    status: String,
    flightNumbers: String,
    stopLabel: String,
    duration: String,
    best: Boolean
): String {
    val normalizedStatus = status
        .replace(" - ", " • ")
        .ifBlank { if (best) "Best flight" else "" }
    return listOf(normalizedStatus, flightNumbers, stopLabel, duration)
        .map { it.trim() }
        .filter { it.isNotBlank() && !SafeContentPolicy.looksLikeUrl(it) }
        .distinct()
        .take(3)
        .joinToString(" • ")
}

internal fun rankedFlightChips(
    stopLabel: String,
    layover: String,
    carbon: String,
    bookingStatus: String
): List<String> {
    return listOf(stopLabel, layover, carbon, bookingStatus)
        .map { NativeTextFormatter.sanitizeDisplayText(it).trim() }
        .filter { it.isNotBlank() && !SafeContentPolicy.looksLikeUrl(it) }
        .distinct()
}

@Composable
internal fun FlightLegTimelineRow(
    badge: String,
    route: String,
    carrier: String
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        color = MaterialTheme.colorScheme.surfaceContainerHighest.copy(alpha = 0.62f)
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.Top
        ) {
            Surface(
                shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                color = MaterialTheme.colorScheme.primaryContainer
            ) {
                Text(
                    text = badge,
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                )
            }
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                Text(
                    text = parseBoldMarkdown(route),
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    text = parseBoldMarkdown(carrier),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
        }
    }
}



@Composable
internal fun RenderStickyFirstColumnTable(
    headers: List<String>,
    rows: List<List<String>>,
    columnMinWidthsDp: List<Int>,
    scrollState: androidx.compose.foundation.ScrollState,
    numericColumns: Set<Int>
) {
    // Widths were bounded against the actual host viewport once for the complete table.
    val firstWidth = (columnMinWidthsDp.firstOrNull() ?: 128).dp
    val trailingWidths = columnMinWidthsDp.drop(1)
    Column(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        RenderStickyTableRow(
            headers = headers,
            row = headers,
            rowIndex = -1,
            firstWidth = firstWidth,
            trailingWidths = trailingWidths,
            scrollState = scrollState,
            numericColumns = numericColumns,
            isHeader = true
        )
        rows.forEachIndexed { index, row ->
            RenderStickyTableRow(
                headers = headers,
                row = row,
                rowIndex = index,
                firstWidth = firstWidth,
                trailingWidths = trailingWidths,
                scrollState = scrollState,
                numericColumns = numericColumns,
                isHeader = false
            )
        }
    }
}

@Composable
internal fun RenderStickyTableRow(
    headers: List<String>,
    row: List<String>,
    rowIndex: Int,
    firstWidth: Dp,
    trailingWidths: List<Int>,
    scrollState: androidx.compose.foundation.ScrollState,
    numericColumns: Set<Int>,
    isHeader: Boolean
) {
    val backgroundColor = tableGridRowColor(isHeader, rowIndex)
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(backgroundColor)
            .semantics(mergeDescendants = true) {
                contentDescription = if (isHeader) {
                    "Header. ${headers.joinToString(". ")}"
                } else {
                    tableRowAccessibilitySummary(headers, row, rowIndex)
                }
            },
        verticalAlignment = Alignment.Top
    ) {
        TableGridCell(
            text = row.getOrNull(0).orEmpty(),
            isHeader = isHeader,
            isNumeric = 0 in numericColumns,
            modifier = Modifier.width(firstWidth),
        )
        Row(
            modifier = Modifier
                .weight(1f)
                .horizontalScroll(scrollState)
        ) {
            headers.drop(1).forEachIndexed { offset, _ ->
                val columnIndex = offset + 1
                TableGridCell(
                    text = row.getOrNull(columnIndex).orEmpty(),
                    isHeader = isHeader,
                    isNumeric = columnIndex in numericColumns,
                    // Every row shares one ScrollState, so it must have exactly the same widths.
                    modifier = Modifier.width((trailingWidths.getOrNull(offset) ?: 120).dp),
                )
            }
        }
    }
}

@Composable
internal fun RenderTableGridRow(
    headers: List<String>,
    row: List<String>,
    rowIndex: Int,
    columnMinWidthsDp: List<Int>,
    numericColumns: Set<Int>,
    weighted: Boolean,
    isHeader: Boolean
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(tableGridRowColor(isHeader, rowIndex))
            .semantics(mergeDescendants = true) {
                contentDescription = if (isHeader) {
                    "Header. ${headers.joinToString(". ")}"
                } else {
                    tableRowAccessibilitySummary(headers, row, rowIndex)
                }
            },
        horizontalArrangement = Arrangement.spacedBy(0.dp),
        verticalAlignment = Alignment.Top
    ) {
        headers.indices.forEach { columnIndex ->
            val cellModifier = if (weighted) {
                Modifier.weight(1f)
            } else {
                Modifier.width((columnMinWidthsDp.getOrNull(columnIndex) ?: 120).dp)
            }
            TableGridCell(
                text = row.getOrNull(columnIndex).orEmpty(),
                isHeader = isHeader,
                isNumeric = columnIndex in numericColumns,
                modifier = cellModifier
            )
        }
    }
}

@Composable
internal fun TableGridCell(
    text: String,
    isHeader: Boolean,
    isNumeric: Boolean,
    modifier: Modifier = Modifier,
) {
    Text(
        text = parseBoldMarkdown(text),
        style = if (isHeader) {
            MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold)
        } else {
            MaterialTheme.typography.bodySmall
        },
        color = if (isHeader) {
            MaterialTheme.colorScheme.onSurfaceVariant
        } else {
            MaterialTheme.colorScheme.onSurface
        },
        textAlign = if (isNumeric) TextAlign.End else TextAlign.Start,
        // Fixed column widths may wrap; all supplied header and cell text must remain visible.
        modifier = modifier
            .padding(horizontal = 10.dp, vertical = if (isHeader) 9.dp else 8.dp)
            .then(if (isHeader) Modifier.semantics { heading() } else Modifier)
    )
}

@Composable
internal fun tableGridRowColor(isHeader: Boolean, rowIndex: Int): Color {
    return when {
        isHeader -> MaterialTheme.colorScheme.onSurface.copy(alpha = 0.055f)
        rowIndex >= 0 && rowIndex % 2 == 1 -> MaterialTheme.colorScheme.onSurface.copy(alpha = 0.025f)
        else -> Color.Transparent
    }
}


internal fun collectResolvedTableRows(
    tableModel: FlatTableModel,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    repeatedRowScopes: List<RepeatScope>,
    repeatScope: RepeatScope?
): List<List<String>> {
    if (tableModel.columns <= 0) return emptyList()
    val computedFunctions = DefaultComputedFunctions
    val rows = mutableListOf<List<String>>()
    fun resolveCellValue(cellId: String?, scope: RepeatScope?): String {
        val element = cellId?.let(elements::get) ?: return ""
        val resolved = FlatExprResolver.resolve(
            value = textLikeValue(element.props),
            state = state,
            repeatScope = scope,
            computedFunctions = computedFunctions
        )
        return resolved?.toString()?.trim().orEmpty()
    }
    if (tableModel.rowTemplateId != null && repeatedRowScopes.isNotEmpty()) {
        val template = elements[tableModel.rowTemplateId] ?: return emptyList()
        repeatedRowScopes.forEach { scope ->
            val row = (0 until tableModel.columns).map { columnIndex ->
                resolveCellValue(template.children.getOrNull(columnIndex), scope)
            }
            rows += row
        }
        return rows
    }
    tableModel.staticRowIds.forEach { rowId ->
        val rowElement = elements[rowId] ?: return@forEach
        val row = (0 until tableModel.columns).map { columnIndex ->
            resolveCellValue(rowElement.children.getOrNull(columnIndex), repeatScope)
        }
        rows += row
    }
    return rows
}

@Composable
internal fun RenderResponsiveTableRowCard(
    rowId: String,
    rowScope: RepeatScope?,
    rowIndex: Int,
    headers: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    onAction: (Any?, RepeatScope?) -> Int,
    activePath: Set<String>
) {
    val rowElement = elements[rowId] ?: return
    if (rowElement.children.isEmpty()) return
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp)
            .semantics {
                contentDescription = "Row ${rowIndex + 1}"
            },
        shape = RoundedCornerShape(16.dp),
        colors = flatSpecCardColors(),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 8.dp, vertical = 6.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            rowElement.children.forEachIndexed { index, cellId ->
                Text(
                    text = headers.getOrNull(index).orEmpty().ifBlank { "Column ${index + 1}" },
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 1.dp)
                )
                RenderElement(
                    elementId = cellId,
                    elements = elements,
                    state = state,
                    repeatScope = rowScope,
                    onOpenUrl = onOpenUrl,
                    onSetState = onSetState,
                    onAction = onAction,
                    activePath = activePath + rowId
                )
                if (index < rowElement.children.lastIndex) {
                    HorizontalDivider(
                        color = MaterialTheme.colorScheme.outlineVariant,
                        modifier = Modifier.padding(horizontal = 8.dp)
                    )
                }
            }
        }
    }
}



internal fun collectTextValuesForEmail(
    elementIds: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    activePath: Set<String> = emptySet()
): List<String> {
    val computedFunctions = emptyMap<String, FlatComputedFunction>()
    return elementIds.flatMap { elementId ->
        if (elementId in activePath) return@flatMap emptyList()
        val element = elements[elementId] ?: return@flatMap emptyList()
        val resolvedProps = element.props.mapValues { (_, value) ->
            FlatExprResolver.resolve(value, state, repeatScope, computedFunctions)
        }
        val ownText = if (element.type.equals("text", ignoreCase = true)) {
            emailFirstString(resolvedProps, "text", "title", "label", "content", "value")
                .takeIf { it.isNotBlank() }
                ?.let(::listOf)
                .orEmpty()
        } else {
            emptyList()
        }
        ownText + collectTextValuesForEmail(
            element.children,
            elements,
            state,
            repeatScope,
            activePath + elementId
        )
    }
}

internal fun extractEmailPreviewPropsFromCard(
    childIds: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?
): Map<String, Any?>? {
    val textValues = collectTextValuesForEmail(childIds, elements, state, repeatScope)
        .map { it.trim() }
        .filter { it.isNotBlank() }
    if (textValues.size < 5) return null

    fun labelValue(label: String): String {
        val prefix = "$label:"
        textValues.forEachIndexed { index, text ->
            if (text.equals(prefix, ignoreCase = true)) {
                return textValues.getOrNull(index + 1).orEmpty()
            }
            if (text.startsWith(prefix, ignoreCase = true)) {
                return text.substringAfter(":").trim()
            }
        }
        return ""
    }

    val subject = labelValue("Subject")
    val to = labelValue("To")
    val from = labelValue("From")
    val hasGreeting = textValues.any { it.startsWith("Dear ", ignoreCase = true) || it.startsWith("Hello ", ignoreCase = true) }
    val hasSignature = textValues.any { value ->
        val lower = value.lowercase()
        lower.startsWith("best regards") ||
            lower.startsWith("sincerely") ||
            lower.startsWith("regards") ||
            lower.startsWith("thank you")
    }
    if (subject.isBlank() || (!hasGreeting && !hasSignature)) return null

    val headerLabels = setOf("subject:", "to:", "from:", "date:")
    fun headerEndIndex(label: String): Int? {
        val prefix = "$label:"
        textValues.forEachIndexed { index, text ->
            if (text.equals(prefix, ignoreCase = true)) {
                return (index + 1).coerceAtMost(textValues.lastIndex)
            }
            if (text.startsWith(prefix, ignoreCase = true)) return index
        }
        return null
    }

    val greetingIndex = textValues.indexOfFirst { text ->
        text.startsWith("Dear ", ignoreCase = true) || text.startsWith("Hello ", ignoreCase = true)
    }
    val headerEnd = listOfNotNull(
        headerEndIndex("Subject"),
        headerEndIndex("To"),
        headerEndIndex("From"),
        headerEndIndex("Date")
    ).maxOrNull() ?: -1
    val bodyStart = greetingIndex.takeIf { it >= 0 } ?: (headerEnd + 1).coerceAtMost(textValues.size)
    val signatureStart = textValues.indexOfFirst { value ->
        val lower = value.lowercase()
        lower.startsWith("best regards") ||
            lower.startsWith("sincerely") ||
            lower.startsWith("regards")
    }.takeIf { it >= 0 } ?: textValues.size
    val body = textValues
        .drop(bodyStart)
        .take((signatureStart - bodyStart).coerceAtLeast(0))
        .filterNot { headerLabels.contains(it.lowercase()) }
        .filterNot { it.startsWith("Subject:", ignoreCase = true) || it.startsWith("To:", ignoreCase = true) || it.startsWith("From:", ignoreCase = true) }
    val signature = if (signatureStart < textValues.size) textValues.drop(signatureStart) else emptyList()
    val bodyWithGreeting = if (
        body.none { it.startsWith("Dear ", ignoreCase = true) || it.startsWith("Hello ", ignoreCase = true) } &&
        to.isNotBlank()
    ) {
        listOf("Dear $to,") + body
    } else {
        body
    }

    return mapOf(
        "title" to "Email draft",
        "subject" to subject,
        "to" to to,
        "from" to from,
        "body" to bodyWithGreeting,
        "signature" to signature
    )
}

internal fun emailFirstString(props: Map<String, Any?>, vararg keys: String): String {
    keys.forEach { key ->
        val value = props[key]
        if (value is String && value.isNotBlank()) return value.trim()
        if (value != null && value !is List<*> && value !is Map<*, *>) {
            val text = value.toString().trim()
            if (text.isNotBlank()) return text
        }
    }
    return ""
}

internal fun emailFirstNonBlank(map: Map<String, Any?>, vararg keys: String): String {
    keys.forEach { key ->
        val value = map[key]?.toString()?.trim().orEmpty()
        if (value.isNotBlank()) return value
    }
    return ""
}

internal fun emailStringList(value: Any?): List<String> {
    return when (value) {
        is List<*> -> value.mapNotNull { item ->
            when (item) {
                is String -> item.trim().takeIf { it.isNotBlank() }
                is Map<*, *> -> emailFirstNonBlank(
                    item.entries.associate { (key, entryValue) -> key.toString() to entryValue },
                    "text",
                    "body",
                    "paragraph",
                    "value",
                    "content"
                ).takeIf { it.isNotBlank() }
                null -> null
                else -> item.toString().trim().takeIf { it.isNotBlank() }
            }
        }
        is String -> value
            .split(Regex("""\n\s*\n"""))
            .map { it.trim() }
            .filter { it.isNotBlank() }
        null -> emptyList()
        else -> listOf(value.toString().trim()).filter { it.isNotBlank() }
    }
}

internal fun emailMetadataItems(props: Map<String, Any?>): List<Pair<String, String>> {
    val items = mutableListOf<Pair<String, String>>()
    fun add(label: String, value: String) {
        if (value.isNotBlank()) items += label to value
    }
    add("To", emailFirstString(props, "to", "recipient"))
    add("From", emailFirstString(props, "from", "sender"))
    add("Date", emailFirstString(props, "date", "sentAt", "interviewDate"))
    add("Role", emailFirstString(props, "role", "position"))
    add("Company", emailFirstString(props, "company", "organization"))

    val rawContext = props["context"] ?: props["metadata"] ?: props["details"]
    when (rawContext) {
        is Map<*, *> -> rawContext.forEach { (key, value) ->
            val label = key?.toString()?.trim().orEmpty()
            val text = value?.toString()?.trim().orEmpty()
            if (label.isNotBlank() && text.isNotBlank()) items += label to text
        }
        is List<*> -> rawContext.forEach { entry ->
            val map = toStringKeyMap(entry)
            if (map != null) {
                val label = emailFirstNonBlank(map, "label", "name", "key")
                val value = emailFirstNonBlank(map, "value", "text", "content")
                if (label.isNotBlank() && value.isNotBlank()) items += label to value
            } else {
                val value = entry?.toString()?.trim().orEmpty()
                if (value.isNotBlank()) items += "Info" to value
            }
        }
    }
    return items.distinct()
}



internal data class FencedCodeBlock(
    val code: String,
    val language: String? = null,
    val title: String? = null,
    val isConsole: Boolean = false
)

internal data class FormulaFractionParts(
    val prefix: String,
    val numerator: String,
    val denominator: String,
    val suffix: String
)

internal sealed class FormulaVisualSegment {
    data class TextSegment(val value: String) : FormulaVisualSegment()
    data class FractionSegment(val numerator: String, val denominator: String) : FormulaVisualSegment()
}


@Composable
internal fun RenderFormulaExpression(
    displayFormula: String,
    segments: List<FormulaVisualSegment>,
    scrollState: androidx.compose.foundation.ScrollState
) {
    val formulaTextStyle = MaterialTheme.typography.titleLarge.copy(
        fontFamily = FontFamily.Monospace,
        fontWeight = FontWeight.SemiBold
    )
    val hasFraction = segments.any { it is FormulaVisualSegment.FractionSegment }
    if (!hasFraction) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(scrollState)
                .padding(horizontal = 16.dp, vertical = 18.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = formulaAnnotatedString(displayFormula),
                style = formulaTextStyle,
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Visible
            )
        }
        return
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(scrollState)
            .padding(horizontal = 16.dp, vertical = 18.dp),
        contentAlignment = Alignment.Center
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            segments.forEachIndexed { index, segment ->
                when (segment) {
                    is FormulaVisualSegment.TextSegment -> {
                        val nextIsFraction = segments.getOrNull(index + 1) is FormulaVisualSegment.FractionSegment
                        val text = formulaDisplayTextSegment(segment.value, nextIsFraction)
                        if (text.isNotBlank()) {
                            Text(
                                text = formulaAnnotatedString(text),
                                style = formulaTextStyle,
                                color = MaterialTheme.colorScheme.onSurface,
                                maxLines = 1,
                                overflow = TextOverflow.Visible
                            )
                        }
                    }
                    is FormulaVisualSegment.FractionSegment -> FormulaFractionView(
                        numerator = segment.numerator,
                        denominator = segment.denominator
                    )
                }
            }
        }
    }
}

@Composable
internal fun FormulaFractionView(
    numerator: String,
    denominator: String,
    modifier: Modifier = Modifier
) {
    val numeratorText = formulaAnnotatedString(numerator)
    val denominatorText = formulaAnnotatedString(denominator)
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(4.dp),
        modifier = modifier.widthIn(min = 88.dp)
    ) {
        Text(
            text = numeratorText,
            style = MaterialTheme.typography.titleMedium.copy(
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.SemiBold
            ),
            color = MaterialTheme.colorScheme.onSurface,
            maxLines = 1,
            softWrap = false,
            overflow = TextOverflow.Visible,
            modifier = Modifier.padding(horizontal = 6.dp)
        )
        HorizontalDivider(
            modifier = Modifier.fillMaxWidth(),
            color = MaterialTheme.colorScheme.primary.copy(alpha = 0.70f),
            thickness = 1.5.dp
        )
        Text(
            text = denominatorText,
            style = MaterialTheme.typography.titleMedium.copy(
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.SemiBold
            ),
            color = MaterialTheme.colorScheme.onSurface,
            maxLines = 1,
            softWrap = false,
            overflow = TextOverflow.Visible,
            modifier = Modifier.padding(horizontal = 6.dp)
        )
    }
}



internal data class SupportedMarkdownText(
    val content: AnnotatedString,
    val headingLevel: Int?
)

internal fun parseFencedCodeBlock(raw: String): FencedCodeBlock? {
    val trimmed = raw.trim()
    val fence = when {
        trimmed.startsWith("```") -> "```"
        trimmed.startsWith("'''") -> "'''"
        else -> return null
    }
    if (!trimmed.endsWith(fence) || trimmed.length <= fence.length * 2) return null
    val inner = trimmed.substring(fence.length, trimmed.length - fence.length).trim('\n', '\r')
    if (inner.isBlank()) return FencedCodeBlock(code = "")
    val lines = inner.lines()
    val firstLine = lines.firstOrNull()?.trim().orEmpty()
    val languageToken = if (
        lines.size > 1 &&
        firstLine.matches(Regex("^[A-Za-z0-9_+\\-]{1,20}$"))
    ) {
        firstLine
    } else {
        null
    }
    val code = if (languageToken != null) {
        lines.drop(1).joinToString("\n")
    } else {
        inner
    }
    val isConsole = languageToken.equals("console", ignoreCase = true) ||
        languageToken.equals("terminal", ignoreCase = true) ||
        languageToken.equals("shell", ignoreCase = true) ||
        languageToken.equals("bash", ignoreCase = true)
    return FencedCodeBlock(code = code, language = languageToken, isConsole = isConsole)
}

internal fun inferCodeBlockFromPlainText(raw: String, props: Map<String, Any?>): FencedCodeBlock? {
    val text = raw.trim()
    if ('\n' !in text || text.length < 16) return null
    val variant = props["variant"]?.toString()?.lowercase().orEmpty()
    val hasOutput = Regex("""(?im)^\s*(Output|Console|Terminal|Result)\s*:""").containsMatchIn(text)
    val hasUsage = Regex("""(?im)^\s*(Usage|Example)\s*\d*\s*$""").containsMatchIn(text)
    val hasPythonCode = Regex("""(?m)^\s*(def |class |print\(|for |if |elif |else:|return |import |from )""")
        .containsMatchIn(text)
    val hasShellPrompt = Regex("""(?m)^\s*([$>#]|PS>|C:\\|adb |git |npm |python )""").containsMatchIn(text)
    val lineCount = text.lines().count { it.isNotBlank() }
    if (!hasOutput && !hasPythonCode && !hasShellPrompt) return null
    if (lineCount < 2) return null

    val isConsole = hasOutput || hasUsage || hasShellPrompt || variant == "console"
    val language = when {
        isConsole -> "console"
        hasPythonCode -> "python"
        else -> "text"
    }
    val title = when {
        isConsole && hasUsage -> text.lineSequence().firstOrNull()?.trim()?.takeIf { it.length <= 32 }
        isConsole -> "Console output"
        hasPythonCode -> "Python"
        else -> "Code"
    }
    return FencedCodeBlock(
        code = text,
        language = language,
        title = title,
        isConsole = isConsole
    )
}

internal fun looksLikeFormulaText(raw: String): Boolean {
    val text = raw.trim()
    if (text.length !in 5..220) return false
    if ('\n' in text) return false
    if (NativeTextFormatter.containsUrlLikeToken(text)) return false
    val hasEquation = '=' in text
    val hasMathSignal = listOf("\\frac", "^", "_", "√", "sqrt", "[", "]", "(", ")", "×", "÷", "/")
        .any { token -> text.contains(token) }
    val hasOperator = Regex("""[+\-−*/=]""").containsMatchIn(text)
    val hasVariable = Regex("""\b[A-Za-z]\b""").containsMatchIn(text)
    return hasEquation && hasMathSignal && hasOperator && hasVariable
}

internal fun normalizeFormulaText(raw: String): String {
    var value = raw.trim()
    if ((value.startsWith("$$") && value.endsWith("$$")) ||
        (value.startsWith("\\[") && value.endsWith("\\]"))
    ) {
        value = value.removePrefix("$$").removeSuffix("$$")
            .removePrefix("\\[").removeSuffix("\\]")
            .trim()
    } else if ((value.startsWith("$") && value.endsWith("$")) ||
        (value.startsWith("\\(") && value.endsWith("\\)"))
    ) {
        value = value.removePrefix("$").removeSuffix("$")
            .removePrefix("\\(").removeSuffix("\\)")
            .trim()
    }
    return value
        .replace("\\left", "")
        .replace("\\right", "")
        .replace("\\times", "×")
        .replace("\\cdot", "·")
        .replace("\\div", "÷")
        .replace("\\%", "%")
        .replace("–", "-")
        .replace("−", "-")
        .replace(Regex("""\\text\{([^}]*)\}""")) { match -> match.groupValues[1] }
        .replace(Regex("""\s+"""), " ")
        .trim()
}

internal fun plainBracketFractionToLatex(raw: String): String {
    val normalized = normalizeFormulaText(raw)
    if ("\\frac" in normalized) return normalized
    val match = Regex("""^(.*?)\[\s*(.+?)\s*]\s*/\s*\[\s*(.+?)\s*]$""").matchEntire(normalized)
        ?: return normalized
    val prefix = match.groupValues[1].trimEnd()
    val numerator = match.groupValues[2].trim()
    val denominator = match.groupValues[3].trim()
    return "$prefix \\frac{$numerator}{$denominator}".trim()
}

internal fun parseFormulaFraction(raw: String): FormulaFractionParts? {
    val text = normalizeFormulaText(raw)
    val index = text.indexOf("\\frac")
    if (index < 0) return null
    var cursor = index + "\\frac".length
    while (cursor < text.length && text[cursor].isWhitespace()) cursor++
    val numerator = readLatexGroup(text, cursor) ?: return null
    cursor = numerator.nextIndex
    while (cursor < text.length && text[cursor].isWhitespace()) cursor++
    val denominator = readLatexGroup(text, cursor) ?: return null
    return FormulaFractionParts(
        prefix = text.substring(0, index).trim(),
        numerator = numerator.value.trim(),
        denominator = denominator.value.trim(),
        suffix = text.substring(denominator.nextIndex).trim()
    )
}

internal fun parseFormulaSegments(raw: String): List<FormulaVisualSegment> {
    val text = plainBracketFractionToLatex(raw)
    if ("\\frac" !in text) return listOf(FormulaVisualSegment.TextSegment(text))
    val segments = mutableListOf<FormulaVisualSegment>()
    var cursor = 0
    while (cursor < text.length) {
        val fractionIndex = text.indexOf("\\frac", startIndex = cursor)
        if (fractionIndex < 0) {
            text.substring(cursor).takeIf { it.isNotBlank() }?.let {
                segments += FormulaVisualSegment.TextSegment(it)
            }
            break
        }
        text.substring(cursor, fractionIndex).takeIf { it.isNotBlank() }?.let {
            segments += FormulaVisualSegment.TextSegment(it)
        }
        var groupCursor = fractionIndex + "\\frac".length
        while (groupCursor < text.length && text[groupCursor].isWhitespace()) groupCursor++
        val numerator = readLatexGroup(text, groupCursor) ?: return listOf(FormulaVisualSegment.TextSegment(text))
        groupCursor = numerator.nextIndex
        while (groupCursor < text.length && text[groupCursor].isWhitespace()) groupCursor++
        val denominator = readLatexGroup(text, groupCursor) ?: return listOf(FormulaVisualSegment.TextSegment(text))
        segments += FormulaVisualSegment.FractionSegment(
            numerator = numerator.value.trim(),
            denominator = denominator.value.trim()
        )
        cursor = denominator.nextIndex
    }
    return segments.ifEmpty { listOf(FormulaVisualSegment.TextSegment(text)) }
}

internal fun formulaDisplayTextSegment(raw: String, nextIsFraction: Boolean): String {
    val text = raw.trim()
    if (!nextIsFraction || text.isBlank()) return text
    val last = text.last()
    val alreadyOperator = last in setOf('=', '+', '-', '*', '/', '×', '÷', '·', '(')
    return if (alreadyOperator) text else "$text ×"
}

internal data class LatexGroup(val value: String, val nextIndex: Int)

internal fun readLatexGroup(text: String, start: Int): LatexGroup? {
    if (start >= text.length || text[start] != '{') return null
    var depth = 0
    for (index in start until text.length) {
        when (text[index]) {
            '{' -> depth++
            '}' -> {
                depth--
                if (depth == 0) {
                    return LatexGroup(
                        value = text.substring(start + 1, index),
                        nextIndex = index + 1
                    )
                }
            }
        }
    }
    return null
}

internal fun formulaAnnotatedString(raw: String): AnnotatedString = buildAnnotatedString {
    val text = normalizeFormulaText(raw)
    var index = 0
    while (index < text.length) {
        val char = text[index]
        when {
            char == '^' || char == '_' -> {
                val token = readFormulaScriptToken(text, index + 1)
                if (token.value.isNotBlank()) {
                    pushStyle(
                        SpanStyle(
                            baselineShift = if (char == '^') BaselineShift.Superscript else BaselineShift.Subscript,
                            fontWeight = FontWeight.SemiBold
                        )
                    )
                    append(token.value)
                    pop()
                    index = token.nextIndex
                } else {
                    append(char)
                    index++
                }
            }
            char == '\\' -> {
                val command = readLatexCommand(text, index + 1)
                val replacement = latexCommandReplacement(command.value)
                if (replacement != null) {
                    append(replacement)
                    index = command.nextIndex
                } else {
                    append(char)
                    index++
                }
            }
            char == '{' || char == '}' -> index++
            else -> {
                append(char)
                index++
            }
        }
    }
}

internal data class FormulaToken(val value: String, val nextIndex: Int)

internal fun readFormulaScriptToken(text: String, start: Int): FormulaToken {
    if (start >= text.length) return FormulaToken("", start)
    if (text[start] == '{') {
        readLatexGroup(text, start)?.let { return FormulaToken(it.value, it.nextIndex) }
    }
    return FormulaToken(text[start].toString(), start + 1)
}

internal fun readLatexCommand(text: String, start: Int): FormulaToken {
    var index = start
    while (index < text.length && text[index].isLetter()) index++
    return FormulaToken(text.substring(start, index), index)
}

internal fun latexCommandReplacement(command: String): String? = when (command) {
    "alpha" -> "α"
    "beta" -> "β"
    "gamma" -> "γ"
    "delta" -> "δ"
    "Delta" -> "Δ"
    "theta" -> "θ"
    "lambda" -> "λ"
    "mu" -> "μ"
    "pi" -> "π"
    "sigma" -> "σ"
    "Sigma" -> "Σ"
    "sqrt" -> "√"
    "leq" -> "≤"
    "geq" -> "≥"
    "neq" -> "≠"
    "approx" -> "≈"
    "infty" -> "∞"
    else -> null
}

internal fun readableFormulaText(raw: String): String =
    normalizeFormulaText(raw)
        .replace("\\frac", " fraction ")
        .replace("{", " ")
        .replace("}", " ")
        .replace(Regex("""\s+"""), " ")
        .trim()

internal fun parseSupportedMarkdownText(raw: String): SupportedMarkdownText {
    val trimmedStart = raw.trimStart()
    val headingLevel = when {
        trimmedStart.startsWith("### ") -> 3
        trimmedStart.startsWith("## ") -> 2
        trimmedStart.startsWith("# ") -> 1
        else -> null
    }
    val withoutHeading = if (headingLevel != null) {
        trimmedStart.substring(headingLevel + 1).trimStart()
    } else {
        raw
    }
    return SupportedMarkdownText(
        content = parseBoldMarkdown(withoutHeading),
        headingLevel = headingLevel
    )
}

internal val LEADING_LABEL_REGEX = Regex("^(\\s*)([A-Za-z][A-Za-z0-9 ./()&+\\-]{0,40})([:;])(\\s*.*)$")

internal fun parseBoldMarkdown(raw: String): AnnotatedString {
    val cleanedRaw = NativeTextFormatter.sanitizeDisplayText(raw, preserveMarkdown = true)
    return buildAnnotatedString {
        cleanedRaw.split('\n').forEachIndexed { index, line ->
            if (index > 0) append('\n')
            val labelMatch = if (NativeTextFormatter.containsUrlLikeToken(line)) {
                null
            } else {
                LEADING_LABEL_REGEX.matchEntire(line)
            }
            if (labelMatch != null) {
                val leading = labelMatch.groupValues.getOrNull(1).orEmpty()
                val label = labelMatch.groupValues.getOrNull(2).orEmpty()
                val delimiter = labelMatch.groupValues.getOrNull(3).orEmpty()
                val rest = labelMatch.groupValues.getOrNull(4).orEmpty()
                append(leading)
                pushStyle(SpanStyle(fontWeight = FontWeight.SemiBold))
                append(label)
                append(delimiter)
                pop()
                appendMarkdownBoldSpans(rest)
            } else {
                appendMarkdownBoldSpans(line)
            }
        }
    }
}

internal fun AnnotatedString.Builder.appendMarkdownBoldSpans(segment: String) {
    var cursor = 0
    while (cursor < segment.length) {
        val start = segment.indexOf("**", cursor)
        if (start < 0) {
            append(segment.substring(cursor))
            break
        }
        val end = segment.indexOf("**", start + 2)
        if (end < 0) {
            append(segment.substring(cursor))
            break
        }
        if (start > cursor) append(segment.substring(cursor, start))
        val boldText = segment.substring(start + 2, end)
        if (boldText.isNotEmpty()) {
            pushStyle(SpanStyle(fontWeight = FontWeight.SemiBold))
            append(boldText)
            pop()
        }
        cursor = end + 2
    }
}

internal const val FLAT_SPEC_IMAGE_USER_AGENT = "A2UI GenUICraft Android/1.0 image-renderer"

internal fun ImageRequest.Builder.applyFlatSpecRemoteImageHeaders(url: String): ImageRequest.Builder = apply {
    val normalized = url.trim()
    if (normalized.startsWith("http://", ignoreCase = true) ||
        normalized.startsWith("https://", ignoreCase = true)
    ) {
        setHeader("User-Agent", FLAT_SPEC_IMAGE_USER_AGENT)
        setHeader("Accept", "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8")
    }
}

internal fun imageLogLabel(url: String): String {
    val host = parseUrlHost(url).ifBlank { "local/inline image" }
    return host.take(80)
}

@Composable
internal fun PrefetchFlatSpecImages(rawUrls: List<String>) {
    if (rawUrls.isEmpty()) return
    val resolveAssetUrl = LocalFlatSpecAssetResolver.current
    val context = LocalContext.current
    val imageLoader = rememberFlatImageLoader()
    val urls = remember(rawUrls, resolveAssetUrl) {
        rawUrls
            .mapNotNull { raw ->
                SafeContentPolicy.sanitizeMediaUrl(raw, SafeContentPolicy.MediaKind.IMAGE)
                    ?.let(resolveAssetUrl)
                    ?.let(::resolveCoilMediaModel)
                    ?.takeIf { it.isNotBlank() }
            }
            .distinct()
    }
    LaunchedEffect(urls) {
        urls.forEach { url ->
            imageLoader.enqueue(
                ImageRequest.Builder(context)
                    .data(url)
                    .applyFlatSpecRemoteImageHeaders(url)
                    .allowHardware(false)
                    .build()
            )
        }
    }
}


internal fun parseGeneratedRestaurantVisual(raw: String): GeneratedRestaurantVisual? {
    val uri = runCatching { Uri.parse(raw.trim()) }.getOrNull() ?: return null
    if (!uri.scheme.equals("genuicraft", ignoreCase = true)) return null
    if (!uri.host.equals("visual", ignoreCase = true)) return null
    val pathSegments = uri.pathSegments.orEmpty()
    if (pathSegments.firstOrNull()?.equals("restaurant", ignoreCase = true) != true) return null
    val title = uri.getQueryParameter("title")
        ?.trim()
        ?.takeIf { it.isNotBlank() }
        ?: "Restaurant"
    return GeneratedRestaurantVisual(title = title)
}

@Composable
internal fun RenderGeneratedRestaurantVisual(
    visual: GeneratedRestaurantVisual,
    modifier: Modifier,
    imageLabel: String?
) {
    val palette = restaurantVisualPalette(visual.title)
    Box(
        modifier = modifier
            .background(
                Brush.linearGradient(
                    colors = listOf(palette.first, palette.second)
                )
            )
            .accessibilitySemantics(
                props = emptyMap(),
                fallbackLabel = imageLabel ?: "${visual.title} restaurant visual",
                mergeDescendants = true
            ),
        contentAlignment = Alignment.Center
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.radialGradient(
                        colors = listOf(Color.White.copy(alpha = 0.22f), Color.Transparent)
                    )
                )
        )
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(18.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Icon(
                imageVector = Icons.Filled.Restaurant,
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(34.dp)
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = visual.title,
                color = Color.White,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                textAlign = TextAlign.Center,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

internal fun restaurantVisualPalette(seed: String): Pair<Color, Color> {
    val palettes = listOf(
        Color(0xFF8A3FFC) to Color(0xFFFF7A59),
        Color(0xFF0F766E) to Color(0xFFF59E0B),
        Color(0xFFB91C1C) to Color(0xFFF97316),
        Color(0xFF1D4ED8) to Color(0xFF06B6D4),
        Color(0xFF7C2D12) to Color(0xFFEAB308)
    )
    val index = kotlin.math.abs(seed.hashCode()).rem(palettes.size)
    return palettes[index]
}







internal fun bindPathFromValueExpression(value: Any?, repeatScope: RepeatScope?): String? {
    val valueMap = toStringKeyMap(value) ?: return null
    val bindState = valueMap["\$bindState"]?.toString()?.takeIf { it.isNotBlank() }
    if (bindState != null) {
        return bindState
    }
    val bindItem = valueMap["\$bindItem"]?.toString()
    val bindItemPath = resolveBindItemPath(bindItem, repeatScope)
    if (!bindItemPath.isNullOrBlank()) {
        return bindItemPath
    }
    return valueMap["\$state"]?.toString()?.takeIf { it.isNotBlank() }
}













/** First non-blank string among [keys], or null. */
internal fun firstNonBlankStringProp(props: Map<String, Any?>, vararg keys: String): String? {
    keys.forEach { key ->
        val value = props[key]?.toString()?.trim()
        if (!value.isNullOrBlank()) return value
    }
    return null
}

/**
 * Human-readable duration for a media tile.
 *
 * Accepts an already-formatted string, or seconds as a number, which is what the
 * generator tends to emit.
 */
internal fun flatMediaDurationLabel(props: Map<String, Any?>): String? {
    val raw = props["duration"] ?: props["length"] ?: props["runtime"] ?: return null
    (raw as? Number)?.let { number ->
        val total = number.toInt()
        if (total <= 0) return null
        val minutes = total / 60
        val seconds = total % 60
        return if (minutes > 0) "$minutes min ${seconds}s" else "${seconds}s"
    }
    return raw.toString().trim().takeIf { it.isNotBlank() }
}
