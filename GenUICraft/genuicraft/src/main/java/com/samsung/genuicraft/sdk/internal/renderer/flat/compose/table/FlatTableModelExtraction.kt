package com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table

import com.samsung.genuicraft.sdk.internal.renderer.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.legacy.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.expr.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.parse.*

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
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.legacy.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.*

// moved from FlatSpecRenderer.kt (shouldPreferComparisonCards)
internal fun shouldPreferComparisonCards(headers: List<String>, compactScreen: Boolean): Boolean {
    if (!compactScreen || headers.size < 3) return false
    val firstHeader = headers.firstOrNull().orEmpty()
    return isComparisonFeatureHeader(firstHeader) ||
        (headers.size >= 4 && isComparisonEntityHeader(firstHeader))
}

// moved from FlatSpecRenderer.kt (detectTableShape)
internal fun detectTableShape(
    headers: List<String>,
    rows: List<List<String>>,
    domain: String
): FlatTableShape {
    val columnCount = maxOf(headers.size, rows.maxOfOrNull { row -> row.size } ?: 0)
    val firstHeader = headers.firstOrNull().orEmpty()
    val firstHeaderToken = normalizeTableHeaderForMatch(firstHeader)
    val firstColumnValues = rows.mapNotNull { row -> row.getOrNull(0)?.trim()?.takeIf { it.isNotBlank() } }
    val compactFirstColumn = firstColumnValues.isNotEmpty() &&
        firstColumnValues.all { value -> value.length <= 44 }
    val numericLikeColumns = headers.indices.count { index ->
        rows.mapNotNull { row -> row.getOrNull(index)?.trim()?.takeIf { it.isNotBlank() } }
            .let { values -> values.size >= 2 && values.count(::looksLikeNumericTableValue) >= values.size / 2 }
    }
    return when {
        isPlaylistTableHeaderSet(headers, domain) -> FlatTableShape.PLAYLIST
        domain == "formula" && isFormulaVariableHeaderSet(headers) -> FlatTableShape.KEY_VALUE
        domain == "formula" && columnCount == 2 && firstHeaderToken == "input" -> FlatTableShape.KEY_VALUE
        domain == "formula" && isCalculationBreakdownHeaderSet(headers) -> FlatTableShape.NUMERIC_METRICS
        columnCount <= 2 &&
            firstHeaderToken in setOf("metric", "feature", "field", "label", "item", "name", "attribute", "key") ->
            FlatTableShape.KEY_VALUE
        columnCount <= 2 && domain == "generic" -> FlatTableShape.KEY_VALUE
        domain in setOf("schedule", "status") -> FlatTableShape.SCHEDULE_TIMELINE
        isComparisonFeatureHeader(firstHeader) && columnCount >= 3 -> FlatTableShape.FEATURE_MATRIX
        domain == "comparison" && isComparisonFeatureHeader(firstHeader) -> FlatTableShape.FEATURE_MATRIX
        domain in CARD_FIRST_TABLE_DOMAINS -> FlatTableShape.ENTITY_ROW
        domain == "comparison" && compactFirstColumn -> FlatTableShape.ENTITY_ROW
        columnCount <= 2 -> FlatTableShape.KEY_VALUE
        isComparisonEntityHeader(firstHeader) && columnCount >= 3 -> FlatTableShape.ENTITY_ROW
        numericLikeColumns >= 2 && columnCount <= 4 -> FlatTableShape.NUMERIC_METRICS
        else -> FlatTableShape.GENERIC_GRID
    }
}

// moved from FlatSpecRenderer.kt (extractFlatTableModel)
internal fun extractFlatTableModel(
    containerChildren: List<String>,
    containerProps: Map<String, Any?>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    compactScreen: Boolean
): FlatTableModel? {
    if (containerChildren.size < 2) return null
    // If the IR already contains an explicit Table component, let that component render directly.
    // The row-based heuristic below is only for legacy table-like stacks.
    if (containerChildren.any { childId ->
            elements[childId]?.type?.equals("table", ignoreCase = true) == true
        }
    ) {
        return null
    }

    val headerRowId = containerChildren.firstOrNull { childId ->
        val element = elements[childId] ?: return@firstOrNull false
        isHorizontalRowElement(element) &&
            element.repeat == null &&
            element.children.size >= 2 &&
            hasOnlyTextCellChildren(element, elements)
    } ?: return null
    val headerRow = elements[headerRowId] ?: return null

    val bodyContainerId = containerChildren.firstOrNull { childId ->
        if (childId == headerRowId) return@firstOrNull false
        val bodyElement = elements[childId] ?: return@firstOrNull false
        if (bodyElement.repeat == null) return@firstOrNull false
        val templateId = bodyElement.children.firstOrNull() ?: return@firstOrNull false
        val template = elements[templateId] ?: return@firstOrNull false
        isHorizontalRowElement(template) && template.children.size >= 2
    }

    val staticRowIds = if (bodyContainerId == null) {
        containerChildren.filter { childId ->
            childId != headerRowId && isHorizontalRowElement(elements[childId])
        }
    } else {
        emptyList()
    }
    if (bodyContainerId == null && staticRowIds.isEmpty()) {
        return null
    }

    val tableMemberIds = buildSet {
        add(headerRowId)
        bodyContainerId?.let { add(it) }
        addAll(staticRowIds)
    }
    val nonTableChildren = containerChildren.filterNot { it in tableMemberIds }
    if (
        nonTableChildren.size > 2 ||
        nonTableChildren.any { childId ->
            val type = elements[childId]?.type?.trim()?.lowercase().orEmpty()
            type !in setOf("text", "divider")
        }
    ) {
        return null
    }

    val bodyElement = bodyContainerId?.let(elements::get)
    val rowTemplateId = bodyElement?.children?.firstOrNull()
    val rowTemplate = rowTemplateId?.let(elements::get)
    val rows = if (bodyElement?.repeat != null) {
        buildRepeatScopes(bodyElement.repeat, state)?.size ?: 0
    } else {
        staticRowIds.size
    }

    val templateColumns = when {
        rowTemplate != null -> rowTemplate.children.size
        staticRowIds.isNotEmpty() -> staticRowIds.maxOf { id -> elements[id]?.children?.size ?: 0 }
        else -> 0
    }
    val columns = maxOf(headerRow.children.size, templateColumns)
    if (columns < 2) return null

    val headers = (0 until columns).map { index ->
        val cellId = headerRow.children.getOrNull(index)
        val label = resolveTableHeaderLabel(cellId?.let(elements::get), state)
        label.ifBlank { "Column ${index + 1}" }
    }
    val inferredDomain = inferTableDomainFromHeaders(headers)
    val isWeather = inferredDomain == "weather"
    val isFlight = inferredDomain == "flight"
    val explicitDomain = normalizeExplicitTableDomain(containerProps["domain"]?.toString())
        ?.takeIf { it in SUPPORTED_TABLE_DOMAINS }
    val domain = when {
        explicitDomain != null -> explicitDomain
        else -> inferredDomain
    }
    val explicitPreferredPresentation = containerProps["preferredPresentation"]?.toString()?.trim()?.lowercase()
        ?.takeIf { it in setOf("cards", "table") }
    val preferredPresentation = when {
        explicitPreferredPresentation == null -> if (domain in CARD_FIRST_TABLE_DOMAINS) "cards" else "table"
        explicitDomain == "generic" &&
            explicitPreferredPresentation == "table" &&
            domain in CARD_FIRST_TABLE_DOMAINS -> "cards"
        else -> explicitPreferredPresentation
    }
    val resolvedTableRows = collectResolvedTableRows(
        tableModel = FlatTableModel(
            headerRowId = headerRowId,
            bodyContainerId = bodyContainerId,
            rowTemplateId = rowTemplateId,
            staticRowIds = staticRowIds,
            headers = headers,
            columns = columns,
            rows = rows,
            isWeather = isWeather,
            isFlight = isFlight,
            domain = domain,
            preferredPresentation = preferredPresentation,
            title = containerProps["title"]?.toString()?.trim()?.takeIf { it.isNotBlank() },
            shape = FlatTableShape.GENERIC_GRID,
            cardMappingStatus = "not_applicable",
            renderMode = FlatTableRenderMode.TABLE
        ),
        elements = elements,
        state = state,
        repeatedRowScopes = bodyElement?.repeat?.let { buildRepeatScopes(it, state) }.orEmpty(),
        repeatScope = null
    )
    val shape = detectTableShape(
        headers = headers,
        rows = resolvedTableRows,
        domain = domain
    )
    val cardMappingStatus = when (domain) {
        "comparison" -> "pending_runtime_mapping"
        in CARD_FIRST_TABLE_DOMAINS -> "pending_runtime_mapping"
        else -> "not_applicable"
    }
    val comparisonCardsPreferred = domain == "comparison" && shouldPreferComparisonCards(headers, compactScreen)
    val renderMode = when {
        looksLikeProcessStateTable(headers, resolvedTableRows, domain) -> FlatTableRenderMode.PROCESS_CARDS
        domain == "weather" -> FlatTableRenderMode.WEATHER_CARDS
        domain == "flight" -> FlatTableRenderMode.FLIGHT_CARDS
        domain == "booking" -> FlatTableRenderMode.BOOKING_CARDS
        domain == "restaurants" -> FlatTableRenderMode.RESTAURANT_CARDS
        domain == "news" -> FlatTableRenderMode.NEWS_CARDS
        domain == "playlist" -> FlatTableRenderMode.PLAYLIST_CARDS
        comparisonCardsPreferred -> FlatTableRenderMode.RESPONSIVE_CARD_ROWS
        domain in setOf("schedule", "status") -> FlatTableRenderMode.RESPONSIVE_CARD_ROWS
        compactScreen && columns >= 4 -> FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL
        else -> FlatTableRenderMode.TABLE
    }

    return FlatTableModel(
        headerRowId = headerRowId,
        bodyContainerId = bodyContainerId,
        rowTemplateId = rowTemplateId,
        staticRowIds = staticRowIds,
        headers = headers,
        columns = columns,
        rows = rows,
        isWeather = isWeather,
        isFlight = isFlight,
        domain = domain,
        preferredPresentation = preferredPresentation,
        title = containerProps["title"]?.toString()?.trim()?.takeIf { it.isNotBlank() },
        shape = shape,
        cardMappingStatus = cardMappingStatus,
        renderMode = renderMode
    )
}

// moved from FlatSpecRenderer.kt (resolveDirectTableColumns)
internal fun resolveDirectTableColumns(
    props: Map<String, Any?>,
    rows: List<Any?>
): List<FlatDirectTableColumn> {
    val explicitColumns = parseDirectTableColumns(props["columns"])
    if (explicitColumns.isNotEmpty()) {
        return augmentRestaurantDirectTableColumns(props, rows, explicitColumns)
    }

    val tablePayload = toStringKeyMap(props["table"])
    val payloadColumns = parseDirectTableColumns(tablePayload?.get("columns"))
    if (payloadColumns.isNotEmpty()) {
        return augmentRestaurantDirectTableColumns(props, rows, payloadColumns)
    }

    val firstMapRow = rows.firstOrNull { row -> toStringKeyMap(row) != null }?.let(::toStringKeyMap)
    if (!firstMapRow.isNullOrEmpty()) {
        return firstMapRow.keys.map { key ->
            FlatDirectTableColumn(
                key = key,
                label = prettifyTableKey(key)
            )
        }
    }

    val firstListRow = rows.firstOrNull { row -> row is List<*> } as? List<*>
    if (!firstListRow.isNullOrEmpty() && firstListRow.size >= 2) {
        return firstListRow.indices.map { index ->
            FlatDirectTableColumn(
                key = index.toString(),
                label = "Column ${index + 1}"
            )
        }
    }
    return emptyList()
}

// moved from FlatSpecRenderer.kt (resolveDirectTableRows)
internal fun resolveDirectTableRows(
    props: Map<String, Any?>,
    state: Map<String, Any?>
): List<Any?> {
    val directRows = props["rows"] as? List<*>
    if (directRows != null) return directRows.toList()

    val statePath = props["statePath"]?.toString()?.trim().orEmpty()
    if (statePath.isNotBlank()) {
        val fromState = FlatSpecParser.getAtPath(state, normalizePointer(statePath))
        if (fromState is List<*>) {
            return fromState.toList()
        }
    }

    val tablePayload = toStringKeyMap(props["table"])
    val payloadRows = tablePayload?.get("rows") as? List<*>
    if (payloadRows != null) return payloadRows.toList()

    return emptyList()
}

// moved from FlatSpecRenderer.kt (resolveDirectTableRow)
internal fun resolveDirectTableRow(
    row: Any?,
    columns: List<FlatDirectTableColumn>,
    state: Map<String, Any?>
): List<String> {
    val mapRow = toStringKeyMap(row)
    if (mapRow != null) {
        extractDirectTableRowList(mapRow)?.let { listRow ->
            return columns.mapIndexed { index, _ ->
                tableCellDisplayText(
                    resolveDirectTableCellValue(listRow.getOrNull(index), state)
                )
            }
        }
        val orderedEntries = mapRow.entries.toList()
        val normalizedLookup = buildDirectTableNormalizedRowLookup(mapRow)
        val directMatchCount = columns.count { column ->
            resolveRowMapColumnValue(
                row = mapRow,
                column = column,
                columnIndex = -1,
                normalizedLookup = normalizedLookup,
                orderedEntries = orderedEntries,
                usePositionalFallback = false
            ) != null
        }
        val usePositionalFallback = directMatchCount == 0 && orderedEntries.size >= columns.size
        return columns.mapIndexed { index, column ->
            tableCellDisplayText(
                resolveDirectTableCellValue(
                    value = resolveRowMapColumnValue(
                        row = mapRow,
                        column = column,
                        columnIndex = index,
                        normalizedLookup = normalizedLookup,
                        orderedEntries = orderedEntries,
                        usePositionalFallback = usePositionalFallback
                    ),
                    state = state
                )
            )
        }
    }
    val listRow = row as? List<*>
    if (listRow != null) {
        return columns.mapIndexed { index, _ ->
            tableCellDisplayText(
                resolveDirectTableCellValue(listRow.getOrNull(index), state)
            )
        }
    }
    return columns.mapIndexed { index, _ ->
        if (index == 0) {
            tableCellDisplayText(resolveDirectTableCellValue(row, state))
        } else {
            ""
        }
    }
}

// moved from FlatSpecRenderer.kt (resolveDirectTableCellValue)
internal fun resolveDirectTableCellValue(
    value: Any?,
    state: Map<String, Any?>
): Any? {
    return FlatExprResolver.resolve(
        value = value,
        state = state,
        repeatScope = null,
        computedFunctions = DefaultComputedFunctions
    )
}

// moved from FlatSpecRenderer.kt (normalizeTableLookupToken)
internal fun normalizeTableLookupToken(raw: String): String {
    return raw.trim().lowercase().replace(Regex("[^a-z0-9]+"), "")
}

// moved from FlatSpecRenderer.kt (inferTableColumnOrdinal)
internal fun inferTableColumnOrdinal(
    key: String,
    label: String
): Int? {
    val ordinalRegex = Regex("""(?:^|[_\s-])(column|col)?[_\s-]*(\d+)$""")
    listOf(key, label).forEach { token ->
        val trimmed = token.trim().lowercase()
        if (trimmed.isBlank()) return@forEach
        val direct = trimmed.toIntOrNull()
        if (direct != null && direct > 0) {
            return direct - 1
        }
        val match = ordinalRegex.find(trimmed)
        val value = match?.groupValues?.getOrNull(2)?.toIntOrNull()
        if (value != null && value > 0) {
            return value - 1
        }
    }
    return null
}

// moved from FlatSpecRenderer.kt (normalizeTableColumnKey)
internal fun normalizeTableColumnKey(raw: String, fallback: String): String {
    val normalized = raw.trim()
        .replace(Regex("[^A-Za-z0-9_./-]+"), "_")
        .trim('_')
    return if (normalized.isBlank()) fallback else normalized
}

// moved from FlatSpecRenderer.kt (shouldUseHorizontalTableScroll)
internal fun shouldUseHorizontalTableScroll(
    compactScreen: Boolean,
    screenWidthDp: Int,
    headers: List<String>,
    rows: List<List<String>>
): Boolean {
    val narrowScreen = compactScreen || screenWidthDp <= 720
    if (!narrowScreen) return false
    val columnCount = maxOf(headers.size, rows.maxOfOrNull { row -> row.size } ?: 0)
    if (columnCount <= 1) return false
    if (columnCount >= 4) return true
    val availableWidthDp = (screenWidthDp - 24).coerceAtLeast(240)
    val requiredMinWidthDp = estimateTableMinWidthDp(
        headers = headers,
        rows = rows,
        baseMinDp = 120
    )
    return requiredMinWidthDp > availableWidthDp
}

// moved from FlatSpecRenderer.kt (normalizeTableHeaderForMatch)
internal fun normalizeTableHeaderForMatch(raw: String): String {
    return raw
        .trim()
        .lowercase()
        .replace(Regex("[^a-z0-9]+"), " ")
        .replace(Regex("\\s+"), " ")
        .trim()
}

// moved from FlatSpecRenderer.kt (isIconColumnLabel)
internal fun isIconColumnLabel(label: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(label)
    return normalized == "icon" || normalized == "media icon" || normalized == "visual"
}

// moved from FlatSpecRenderer.kt (isImageColumnLabel)
internal fun isImageColumnLabel(label: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(label)
    return normalized == "image" ||
        normalized == "photo" ||
        normalized == "picture" ||
        normalized == "thumbnail" ||
        normalized == "hero image" ||
        normalized == "image url" ||
        normalized == "photo url" ||
        normalized == "media image"
}

// moved from FlatSpecRenderer.kt (isImageAltColumnLabel)
internal fun isImageAltColumnLabel(label: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(label)
    return normalized == "alt" ||
        normalized == "image alt" ||
        normalized == "photo alt" ||
        normalized == "caption"
}

// moved from FlatSpecRenderer.kt (tableRowAccessibilitySummary)
internal fun tableRowAccessibilitySummary(
    headers: List<String>,
    row: List<String>,
    rowIndex: Int? = null
): String {
    val pairs = (0 until maxOf(headers.size, row.size)).mapNotNull { index ->
        val value = row.getOrNull(index).orEmpty().trim()
        if (value.isBlank()) {
            null
        } else {
            val label = tableHeaderLabel(headers, index)
            "$label: $value"
        }
    }
    val prefix = rowIndex?.let { "Row ${it + 1}. " }.orEmpty()
    return prefix + pairs.joinToString(". ")
}

// moved from FlatSpecRenderer.kt (isUrlColumnLabel)
internal fun isUrlColumnLabel(header: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(header)
    return normalized == "url" ||
        normalized == "link" ||
        normalized == "href" ||
        normalized.contains("website") ||
        normalized.contains("booking url") ||
        normalized.contains("action url") ||
        normalized.endsWith(" link")
}

// moved from FlatSpecRenderer.kt (isActionLabelColumn)
internal fun isActionLabelColumn(header: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(header)
    return normalized.contains("action label") ||
        normalized.contains("button label") ||
        normalized.contains("cta label") ||
        normalized == "action" ||
        normalized == "cta"
}
