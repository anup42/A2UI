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
import androidx.compose.material.icons.filled.FlightTakeoff
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Link
import androidx.compose.material.icons.filled.Restaurant
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
import coil.decode.SvgDecoder
import coil.request.ImageRequest
import com.samsung.genuicraft.GenUiTokens
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.samsung.genuicraft.renderer.native.NativeTextFormatter
import com.samsung.genuicraft.renderer.native.NativePayloadParser
import com.samsung.genuicraft.renderer.native.ParsedButton
import com.samsung.genuicraft.renderer.native.intents.flight.NativeFlightSemantics
import com.samsung.genuicraft.renderer.native.intents.flight.NativeFlightUiRenderer
import com.samsung.genuicraft.renderer.native.intents.weather.NativeWeatherSemantics
import com.samsung.genuicraft.renderer.native.intents.weather.NativeWeatherUiRenderer
import com.samsung.genuicraft.renderer.native.parser.NativeSourceParsing
import com.samsung.genuicraft.renderer.native.parser.NativeStructureParsing
import kotlinx.coroutines.launch
import kotlin.math.roundToInt
import java.util.UUID
import java.net.URI

data class FlatSpec(
    val root: String,
    val state: Map<String, Any?>,
    val elements: Map<String, FlatElement>
)

data class FlatElement(
    val type: String,
    val props: Map<String, Any?>,
    val children: List<String>,
    val repeat: RepeatConfig? = null,
    val visible: Any? = null,
    val on: Map<String, Any?>? = null,
    val watch: Map<String, Any?>? = null
)

data class RepeatConfig(
    val statePath: String,
    val key: String? = null
)

data class RepeatScope(
    val item: Any? = null,
    val index: Int? = null,
    val basePath: String? = null
)

private data class FlatSourceSection(
    val title: String?,
    val links: List<ParsedButton>
)

private data class WatchEntry(
    val key: String,
    val statePath: String,
    val actionBinding: Any?
)

internal enum class FlatTableRenderMode {
    TABLE,
    TABLE_HORIZONTAL_SCROLL,
    PROCESS_CARDS,
    WEATHER_CARDS,
    FLIGHT_CARDS,
    BOOKING_CARDS,
    PLAYLIST_CARDS,
    RESPONSIVE_CARD_ROWS
}

internal enum class FlatTableShape {
    PLAYLIST,
    ENTITY_ROW,
    FEATURE_MATRIX,
    KEY_VALUE,
    SCHEDULE_TIMELINE,
    NUMERIC_METRICS,
    GENERIC_GRID
}

private enum class AdaptiveTablePresentation {
    TABLE,
    HORIZONTAL_TABLE,
    STICKY_HORIZONTAL_TABLE,
    CLIMATE_CARDS,
    ENTITY_CARDS,
    PLAYLIST_ROWS,
    FEATURE_CARDS,
    KEY_VALUE_PANEL,
    TIMELINE_CARDS,
    METRIC_CARDS
}

private data class PlaylistTrackRow(
    val number: String,
    val title: String,
    val artist: String?,
    val chips: List<String>,
    val sourceRow: List<String>
)

private data class ClimateComparisonRow(
    val place: String,
    val verdict: String?,
    val high: String?,
    val low: String?,
    val metrics: List<Pair<String, String>>,
    val sourceRow: List<String>
)

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

private data class GeneratedRestaurantVisual(
    val title: String
)

typealias FlatComputedFunction = (Map<String, Any?>) -> Any?

private val LocalFlatSpecAssetResolver = staticCompositionLocalOf<(String) -> String> { { raw -> raw } }
private val LocalFlatSpecComputedFunctions = staticCompositionLocalOf<Map<String, FlatComputedFunction>> { emptyMap() }
private val LocalFlatSpecTextHorizontalPadding = staticCompositionLocalOf { 16.dp }
private const val WATCH_ACTION_BUDGET = 32
private val IMAGE_PROP_KEYS = listOf("url", "src", "image", "source", "name")
private val ICON_PROP_KEYS = listOf("name", "icon", "source", "url", "src")
private val MEDIA_OBJECT_KEYS = listOf("uri", "url", "src", "path", "value", "source", "image", "icon", "name")
private val DIRECT_TABLE_ROW_LIST_KEYS = listOf("cells", "values", "row", "data")
private const val FLAT_SPEC_RENDERER_TAG = "FlatSpecRenderer"
private val PLAYLIST_TABLE_DOMAIN_ALIASES = setOf("playlist", "music", "entertainment")
private val FORMULA_TABLE_DOMAIN_ALIASES = setOf("formula", "calculation", "calculator", "math")
private val CARD_FIRST_TABLE_DOMAINS = setOf("weather", "flight", "booking", "schedule", "status", "playlist")
private val SUPPORTED_TABLE_DOMAINS = CARD_FIRST_TABLE_DOMAINS + setOf("generic", "comparison", "formula")

object FlatSpecParser {

    fun isFlatSpec(json: JsonElement): Boolean {
        if (!json.isJsonObject) return false
        val obj = json.asJsonObject
        return obj.has("root") && obj.has("elements")
    }

    fun parse(json: JsonElement): FlatSpec? {
        if (!isFlatSpec(json)) return null
        val obj = json.asJsonObject
        val root = obj.get("root")?.takeIf { it.isJsonPrimitive }?.asString ?: return null

        val stateMap = mutableMapOf<String, Any?>()
        obj.getAsJsonObject("state")?.entrySet()?.forEach { (key, value) ->
            stateMap[key] = toKotlin(value)
        }

        val elements = mutableMapOf<String, FlatElement>()
        obj.getAsJsonObject("elements")?.entrySet()?.forEach { (id, value) ->
            if (value.isJsonObject) {
                parseElement(value.asJsonObject)?.let { elements[id] = it }
            }
        }

        if (!elements.containsKey(root)) return null
        return FlatSpec(root = root, state = stateMap, elements = elements)
    }

    private fun parseElement(obj: JsonObject): FlatElement? {
        val rawType = obj.get("type")?.takeIf { it.isJsonPrimitive }?.asString ?: return null
        val props = mutableMapOf<String, Any?>()
        obj.getAsJsonObject("props")?.entrySet()?.forEach { (key, value) ->
            props[key] = toKotlin(value)
        }
        obj.entrySet().forEach { (key, value) ->
            if (key in setOf("type", "props", "children", "child", "repeat", "visible", "on", "watch")) return@forEach
            if (!props.containsKey(key)) {
                props[key] = toKotlin(value)
            }
        }

        val children = mutableListOf<String>()
        obj.getAsJsonArray("children")
            ?.mapNotNull { child -> child.takeIf { it.isJsonPrimitive }?.asString }
            ?.forEach { childId ->
                if (childId.isNotBlank() && childId !in children) {
                    children += childId
                }
            }
        obj.get("child")
            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.trim()
            ?.takeIf { it.isNotBlank() }
            ?.let { childId ->
                if (childId !in children) {
                    children += childId
                }
            }
        listOf("template", "itemTemplate", "child").forEach { key ->
            props[key]
                ?.toString()
                ?.trim()
                ?.takeIf { it.isNotBlank() }
                ?.let { childId ->
                    if (childId !in children) {
                        children += childId
                    }
                }
        }

        val repeat = parseRepeat(obj.getAsJsonObject("repeat")) ?: parseRepeat(props["repeat"])

        val visible = obj.get("visible")?.let(::toKotlin) ?: props["visible"]
        val on = obj.getAsJsonObject("on")
            ?.entrySet()
            ?.associate { (key, value) -> key to toKotlin(value) } ?: toStringKeyMap(props["on"])
        val watch = obj.getAsJsonObject("watch")
            ?.entrySet()
            ?.associate { (key, value) -> key to toKotlin(value) } ?: toStringKeyMap(props["watch"])

        val type = normalizeElementType(rawType, props)

        return FlatElement(
            type = type,
            props = props,
            children = children,
            repeat = repeat,
            visible = visible,
            on = on,
            watch = watch
        )
    }

    private fun parseRepeat(value: JsonObject?): RepeatConfig? {
        val statePath = value?.get("statePath")
            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.trim()
            ?: value?.get("path")
                ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                ?.asString
                ?.trim()
            ?: return null
        if (statePath.isBlank()) return null
        val key = value?.get("key")
            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.trim()
            ?.takeIf { it.isNotBlank() }
        return RepeatConfig(
            statePath = normalizePointer(statePath),
            key = key
        )
    }

    private fun parseRepeat(value: Any?): RepeatConfig? {
        val map = toStringKeyMap(value) ?: return null
        val statePath = map["statePath"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
            ?: map["path"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
            ?: return null
        val key = map["key"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
        return RepeatConfig(
            statePath = normalizePointer(statePath),
            key = key
        )
    }

    private fun normalizeElementType(rawType: String, props: MutableMap<String, Any?>): String {
        return when (rawType.trim().lowercase()) {
            "row" -> {
                props.putIfAbsent("direction", "horizontal")
                "Stack"
            }
            "column" -> {
                props.putIfAbsent("direction", "vertical")
                "Stack"
            }
            else -> rawType
        }
    }

    fun toKotlin(element: JsonElement): Any? = when {
        element.isJsonNull -> null
        element.isJsonPrimitive -> {
            val primitive = element.asJsonPrimitive
            when {
                primitive.isBoolean -> primitive.asBoolean
                primitive.isNumber -> {
                    val number = primitive.asNumber
                    val asLong = number.toLong()
                    if (asLong.toDouble() == number.toDouble()) asLong else number.toDouble()
                }
                else -> primitive.asString
            }
        }
        element.isJsonArray -> element.asJsonArray.map(::toKotlin)
        element.isJsonObject -> element.asJsonObject.entrySet().associate { (key, value) ->
            key to toKotlin(value)
        }
        else -> null
    }

    fun getAtPath(root: Any?, path: String): Any? {
        val tokens = pointerTokens(path)
        if (tokens.isEmpty()) return root
        var current: Any? = root
        tokens.forEach { token ->
            current = when (current) {
                is Map<*, *> -> current[token]
                is List<*> -> {
                    val list = current as List<*>
                    token.toIntOrNull()?.let { index ->
                        if (index >= 0 && index < list.size) list[index] else null
                    }
                }
                else -> return null
            }
        }
        return current
    }

    fun setAtPath(stateStore: MutableMap<String, Any?>, path: String, value: Any?) {
        val tokens = pointerTokens(path)
        if (tokens.isEmpty()) return
        val topKey = tokens.first()
        if (tokens.size == 1) {
            stateStore[topKey] = value
            return
        }

        val existingRoot = stateStore[topKey]
        val updatedRoot = setInNode(existingRoot, tokens.drop(1), value)
        stateStore[topKey] = updatedRoot
    }

    private fun setInNode(node: Any?, tokens: List<String>, value: Any?): Any? {
        if (tokens.isEmpty()) return value
        val head = tokens.first()
        val tail = tokens.drop(1)
        val index = head.toIntOrNull()
        return if (index != null) {
            val list = when (node) {
                is List<*> -> node.toMutableList()
                else -> mutableListOf()
            }
            while (list.size <= index) {
                list.add(null)
            }
            list[index] = setInNode(list[index], tail, value)
            list
        } else {
            val map = when (node) {
                is Map<*, *> -> node.entries.associate { (k, v) -> k.toString() to v }.toMutableMap()
                else -> mutableMapOf()
            }
            map[head] = setInNode(map[head], tail, value)
            map
        }
    }

    private fun pointerTokens(path: String): List<String> {
        if (path.isBlank() || path == "/") return emptyList()
        return path
            .removePrefix("/")
            .split('/')
            .filter { it.isNotEmpty() }
            .map { decodePointerToken(it) }
    }

    private fun decodePointerToken(token: String): String {
        return token.replace("~1", "/").replace("~0", "~")
    }
}

private val DefaultComputedFunctions: Map<String, FlatComputedFunction> = mapOf(
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

private fun toStringKeyMap(value: Any?): Map<String, Any?>? {
    val map = value as? Map<*, *> ?: return null
    return map.entries.associate { (k, v) -> k.toString() to v }
}

private fun accessibilityMap(props: Map<String, Any?>): Map<String, Any?> {
    return toStringKeyMap(props["accessibility"])
        ?: toStringKeyMap(props["a11y"])
        ?: emptyMap()
}

private fun accessibilityString(
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

private fun accessibilityLabel(
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

private fun isAccessibilityDecorative(props: Map<String, Any?>): Boolean {
    val value = accessibilityMap(props)["decorative"] ?: props["decorative"] ?: props["ariaHidden"]
    return when (value) {
        is Boolean -> value
        is String -> value.equals("true", ignoreCase = true) || value.equals("decorative", ignoreCase = true)
        else -> false
    }
}

private fun Modifier.accessibilitySemantics(
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

private fun normalizePointer(path: String): String {
    val trimmed = path.trim()
    if (trimmed.isBlank()) return ""
    return if (trimmed.startsWith('/')) trimmed else "/$trimmed"
}

private fun encodePointerToken(token: String): String {
    return token.replace("~", "~0").replace("/", "~1")
}

private fun combinePointerPath(basePath: String?, token: String): String? {
    val normalizedBase = basePath?.trim()?.takeIf { it.isNotBlank() } ?: return null
    val pointerBase = normalizePointer(normalizedBase)
    return "${pointerBase.trimEnd('/')}/${encodePointerToken(token)}"
}

private fun resolveItemValue(item: Any?, path: String): Any? {
    if (item == null) return null
    val normalized = path.trim()
    if (normalized.isBlank()) return item
    val fromPointer = FlatSpecParser.getAtPath(item, normalized)
    if (fromPointer != null) return fromPointer
    return if (normalized == "value") item else null
}

private fun resolveBindItemPath(rawPath: String?, scope: RepeatScope?): String? {
    val basePath = scope?.basePath?.takeIf { it.isNotBlank() } ?: return null
    val requested = rawPath?.trim().orEmpty()
    if (requested.isBlank()) return normalizePointer(basePath)
    return combinePointerPath(basePath, requested)
}

private fun deepCopyValue(value: Any?): Any? {
    return when (value) {
        is Map<*, *> -> value.entries.associate { (k, v) -> k.toString() to deepCopyValue(v) }
        is List<*> -> value.map { deepCopyValue(it) }
        else -> value
    }
}

private fun deepEquals(left: Any?, right: Any?): Boolean {
    if (left === right) return true
    if (left == null || right == null) return false
    if (left is Map<*, *> && right is Map<*, *>) {
        if (left.size != right.size) return false
        return left.keys.all { key -> deepEquals(left[key], right[key]) }
    }
    if (left is List<*> && right is List<*>) {
        if (left.size != right.size) return false
        return left.indices.all { index -> deepEquals(left[index], right[index]) }
    }
    return left == right
}

object FlatExprResolver {

    private val expressionKeys = setOf(
        "\$item",
        "\$state",
        "\$bindState",
        "\$bindItem",
        "\$index",
        "\$cond",
        "\$template",
        "\$computed",
        "literalString",
        "literalNumber",
        "literalBoolean"
    )

    fun resolve(value: Any?, state: Map<String, Any?>, item: Map<String, Any?>?): Any? {
        return resolve(value, state, RepeatScope(item = item), emptyMap())
    }

    fun resolve(
        value: Any?,
        state: Map<String, Any?>,
        repeatScope: RepeatScope?,
        computedFunctions: Map<String, FlatComputedFunction>
    ): Any? {
        return when (value) {
            is Map<*, *> -> resolveMap(value, state, repeatScope, computedFunctions)
            is List<*> -> value.map { child ->
                resolve(child, state, repeatScope, computedFunctions)
            }
            is String -> resolveInlineTemplateString(value, state, repeatScope)
            else -> value
        }
    }

    fun resolveString(value: Any?, state: Map<String, Any?>, item: Map<String, Any?>?): String {
        return resolve(value, state, RepeatScope(item = item), emptyMap())?.toString().orEmpty()
    }

    fun resolveString(
        value: Any?,
        state: Map<String, Any?>,
        repeatScope: RepeatScope?,
        computedFunctions: Map<String, FlatComputedFunction>
    ): String {
        return resolve(value, state, repeatScope, computedFunctions)?.toString().orEmpty()
    }

    fun resolveBoolean(value: Any?, state: Map<String, Any?>, item: Map<String, Any?>?): Boolean {
        return resolveBoolean(value, state, RepeatScope(item = item), emptyMap())
    }

    fun resolveBoolean(
        value: Any?,
        state: Map<String, Any?>,
        repeatScope: RepeatScope?,
        computedFunctions: Map<String, FlatComputedFunction>
    ): Boolean {
        return when (val resolved = resolve(value, state, repeatScope, computedFunctions)) {
            is Boolean -> resolved
            is Number -> resolved.toInt() != 0
            is String -> resolved.equals("true", ignoreCase = true)
            else -> false
        }
    }

    fun evaluateVisible(visible: Any?, state: Map<String, Any?>, item: Map<String, Any?>?): Boolean {
        return evaluateVisible(visible, state, RepeatScope(item = item), emptyMap())
    }

    fun evaluateVisible(
        visible: Any?,
        state: Map<String, Any?>,
        repeatScope: RepeatScope?,
        computedFunctions: Map<String, FlatComputedFunction>
    ): Boolean {
        if (visible == null) return true
        return evaluateCondition(visible, state, repeatScope, computedFunctions)
    }

    private fun resolveMap(
        map: Map<*, *>,
        state: Map<String, Any?>,
        repeatScope: RepeatScope?,
        computedFunctions: Map<String, FlatComputedFunction>
    ): Any? {
        val stringMap = toStringKeyMap(map) ?: return map
        val hasExpressionKey = stringMap.keys.any { key -> key in expressionKeys }
        if (!hasExpressionKey) {
            return stringMap.mapValues { (_, value) ->
                resolve(value, state, repeatScope, computedFunctions)
            }
        }

        return when {
            stringMap.containsKey("\$item") -> {
                val itemPath = stringMap["\$item"]?.toString().orEmpty()
                resolveItemValue(repeatScope?.item, itemPath)
            }

            stringMap.containsKey("\$state") ->
                FlatSpecParser.getAtPath(state, stringMap["\$state"]?.toString().orEmpty())

            stringMap.containsKey("\$bindState") ->
                FlatSpecParser.getAtPath(state, stringMap["\$bindState"]?.toString().orEmpty())

            stringMap.containsKey("\$bindItem") -> {
                val itemPath = stringMap["\$bindItem"]?.toString()
                val resolvedPath = resolveBindItemPath(itemPath, repeatScope)
                if (resolvedPath.isNullOrBlank()) null else FlatSpecParser.getAtPath(state, resolvedPath)
            }

            stringMap.containsKey("\$index") -> repeatScope?.index

            stringMap.containsKey("\$cond") -> {
                val condition = evaluateCondition(stringMap["\$cond"], state, repeatScope, computedFunctions)
                if (condition) {
                    resolve(stringMap["\$then"], state, repeatScope, computedFunctions)
                } else {
                    resolve(stringMap["\$else"], state, repeatScope, computedFunctions)
                }
            }

            stringMap.containsKey("\$template") -> interpolate(
                template = stringMap["\$template"]?.toString().orEmpty(),
                state = state,
                repeatScope = repeatScope
            )

            stringMap.containsKey("\$computed") -> {
                val functionName = stringMap["\$computed"]?.toString()?.trim().orEmpty()
                val args = toStringKeyMap(stringMap["args"])
                    ?.mapValues { (_, argValue) ->
                        resolve(argValue, state, repeatScope, computedFunctions)
                    }
                    ?: emptyMap()
                computedFunctions[functionName]?.invoke(args)
            }

            stringMap.containsKey("literalString") -> stringMap["literalString"]
            stringMap.containsKey("literalNumber") -> stringMap["literalNumber"]
            stringMap.containsKey("literalBoolean") -> stringMap["literalBoolean"]
            else -> stringMap.mapValues { (_, value) ->
                resolve(value, state, repeatScope, computedFunctions)
            }
        }
    }

    private fun evaluateCondition(
        condition: Any?,
        state: Map<String, Any?>,
        repeatScope: RepeatScope?,
        computedFunctions: Map<String, FlatComputedFunction>
    ): Boolean {
        if (condition == null) return true
        if (condition is Boolean) return condition
        if (condition is List<*>) {
            return condition.all { child -> evaluateCondition(child, state, repeatScope, computedFunctions) }
        }
        if (condition !is Map<*, *>) return true
        val expr = toStringKeyMap(condition) ?: return true

        if (expr.containsKey("\$and")) {
            val list = expr["\$and"] as? List<*> ?: return true
            return list.all { child -> evaluateCondition(child, state, repeatScope, computedFunctions) }
        }
        if (expr.containsKey("\$or")) {
            val list = expr["\$or"] as? List<*> ?: return false
            return list.any { child -> evaluateCondition(child, state, repeatScope, computedFunctions) }
        }

        val rawValue: Any? = when {
            expr.containsKey("\$state") -> FlatSpecParser.getAtPath(state, expr["\$state"]?.toString().orEmpty())
            expr.containsKey("\$item") -> resolveItemValue(repeatScope?.item, expr["\$item"]?.toString().orEmpty())
            expr.containsKey("\$index") -> repeatScope?.index
            expr.containsKey("value") -> resolve(expr["value"], state, repeatScope, computedFunctions)
            else -> null
        }

        val eqValue = resolve(expr["eq"], state, repeatScope, computedFunctions)
        val neqValue = resolve(expr["neq"], state, repeatScope, computedFunctions)
        val gtValue = resolve(expr["gt"], state, repeatScope, computedFunctions)
        val gteValue = resolve(expr["gte"], state, repeatScope, computedFunctions)
        val ltValue = resolve(expr["lt"], state, repeatScope, computedFunctions)
        val lteValue = resolve(expr["lte"], state, repeatScope, computedFunctions)

        val result = when {
            expr.containsKey("eq") -> deepEquals(rawValue, eqValue)
            expr.containsKey("neq") -> !deepEquals(rawValue, neqValue)
            expr.containsKey("gt") -> toDouble(rawValue) > toDouble(gtValue)
            expr.containsKey("gte") -> toDouble(rawValue) >= toDouble(gteValue)
            expr.containsKey("lt") -> toDouble(rawValue) < toDouble(ltValue)
            expr.containsKey("lte") -> toDouble(rawValue) <= toDouble(lteValue)
            else -> truthy(rawValue)
        }
        val invert = expr["not"] == true
        return if (invert) !result else result
    }

    private fun toDouble(value: Any?): Double {
        return when (value) {
            is Number -> value.toDouble()
            is String -> value.toDoubleOrNull() ?: 0.0
            else -> 0.0
        }
    }

    private fun truthy(value: Any?): Boolean {
        return when (value) {
            null -> false
            is Boolean -> value
            is Number -> value.toDouble() != 0.0
            is String -> value.isNotBlank() && !value.equals("false", ignoreCase = true)
            else -> true
        }
    }

    private fun interpolate(template: String, state: Map<String, Any?>, repeatScope: RepeatScope?): String {
        return resolveInlineTemplateString(template, state, repeatScope)
    }

    private fun resolveInlineTemplateString(template: String, state: Map<String, Any?>, repeatScope: RepeatScope?): String {
        if (!template.contains("\$item") &&
            !template.contains("\$index") &&
            !template.contains("\${/") &&
            !template.contains("\${index") &&
            !Regex("""[$]\{[^}]+\}""").containsMatchIn(template)
        ) {
            return template
        }
        val item = repeatScope?.item
        val dollarItemResolved = Regex("""[$]\{\s*[$]item[./]([^}]+?)\s*\}""").replace(template) { match ->
            val itemPath = match.groupValues.getOrNull(1).orEmpty().trim()
            resolveItemValue(item, itemPath)?.toString().orEmpty()
        }
        val mustacheItemResolved = Regex("""\{\{\s*[$]item[./]([^}]+?)\s*\}\}""").replace(dollarItemResolved) { match ->
            val itemPath = match.groupValues.getOrNull(1).orEmpty().trim()
            resolveItemValue(item, itemPath)?.toString().orEmpty()
        }
        val itemResolved = Regex("""(?<![$])\{\s*[$]item[./]([^}]+?)\s*\}""").replace(mustacheItemResolved) { match ->
            val itemPath = match.groupValues.getOrNull(1).orEmpty().trim()
            resolveItemValue(item, itemPath)?.toString().orEmpty()
        }
        val legacyIndexResolved = Regex("""(?<![$])\{\s*[$]index\s*(?:\+\s*1)?\s*\}""").replace(itemResolved) { match ->
            val raw = match.value
            val index = repeatScope?.index ?: return@replace ""
            if (raw.contains("+")) (index + 1).toString() else index.toString()
        }
        return Regex("""[$]\{([^}]+)\}""").replace(legacyIndexResolved) { match ->
            val rawPath = match.groupValues.getOrNull(1).orEmpty()
            when (rawPath.trim()) {
                "index", "index_0" -> repeatScope?.index?.toString().orEmpty()
                "index_1" -> repeatScope?.index?.let { (it + 1).toString() }.orEmpty()
                else -> {
                    val path = normalizePointer(rawPath)
                    val stateValue = FlatSpecParser.getAtPath(state, path)
                    if (stateValue != null) {
                        stateValue.toString()
                    } else {
                        resolveItemValue(item, rawPath.trim())?.toString().orEmpty()
                    }
                }
            }
        }
    }
}

internal object FlatActionRuntime {

    fun execute(
        actionCandidate: Any?,
        stateStore: MutableMap<String, Any?>,
        repeatScope: RepeatScope?,
        computedFunctions: Map<String, FlatComputedFunction>,
        onOpenUrl: (String) -> Unit
    ): Int {
        val bindings = toActionBindings(actionCandidate)
        var executed = 0
        bindings.forEach { binding ->
            val actionName = binding["action"]?.toString()?.trim()?.lowercase().orEmpty()
            if (actionName.isBlank()) return@forEach

            val rawParams = toStringKeyMap(binding["params"]) ?: emptyMap()
            val resolvedParams = rawParams.mapValues { (_, value) ->
                FlatExprResolver.resolve(value, stateStore, repeatScope, computedFunctions)
            }

            when (actionName) {
                "openurl" -> {
                    val url = firstNonBlankString(resolvedParams, "url", "href", "link", "targetUrl")
                    if (url.isNotBlank()) {
                        onOpenUrl(url)
                    }
                }

                "setstate" -> {
                    val path = resolveStatePathParam(
                        rawParams,
                        stateStore,
                        repeatScope,
                        computedFunctions,
                        "statePath",
                        "path"
                    )
                    if (path.isNotBlank()) {
                        FlatSpecParser.setAtPath(stateStore, path, resolvedParams["value"])
                    }
                }

                "pushstate" -> {
                    val path = resolveStatePathParam(
                        rawParams,
                        stateStore,
                        repeatScope,
                        computedFunctions,
                        "statePath",
                        "path"
                    )
                    if (path.isNotBlank()) {
                        val current = (FlatSpecParser.getAtPath(stateStore, path) as? List<*>)?.toMutableList()
                            ?: mutableListOf()
                        val generatedId = UUID.randomUUID().toString()
                        val value = replaceGeneratedIdToken(resolvedParams["value"], generatedId)
                        current.add(value)
                        FlatSpecParser.setAtPath(stateStore, path, current.toList())
                        val clearStatePath = resolveStatePathParam(
                            rawParams,
                            stateStore,
                            repeatScope,
                            computedFunctions,
                            "clearStatePath"
                        )
                        if (clearStatePath.isNotBlank()) {
                            FlatSpecParser.setAtPath(stateStore, clearStatePath, null)
                        }
                    }
                }

                "removestate" -> {
                    val path = resolveStatePathParam(
                        rawParams,
                        stateStore,
                        repeatScope,
                        computedFunctions,
                        "statePath",
                        "path"
                    )
                    val index = toInt(resolvedParams["index"])
                    if (path.isNotBlank() && index != null) {
                        val current = (FlatSpecParser.getAtPath(stateStore, path) as? List<*>)?.toMutableList()
                        if (current != null && index in current.indices) {
                            current.removeAt(index)
                            FlatSpecParser.setAtPath(stateStore, path, current.toList())
                        }
                    }
                }

                "validateform" -> {
                    val targetPath = resolveStatePathParam(
                        rawParams,
                        stateStore,
                        repeatScope,
                        computedFunctions,
                        "resultStatePath",
                        "statePath",
                        "path"
                    )
                        .ifBlank { "/formValidation" }
                    val validationResult = mapOf(
                        "valid" to true,
                        "errors" to emptyMap<String, String>()
                    )
                    FlatSpecParser.setAtPath(stateStore, targetPath, validationResult)
                }
            }
            executed += 1
        }
        return executed
    }

    private fun toActionBindings(actionCandidate: Any?): List<Map<String, Any?>> {
        return when (actionCandidate) {
            is List<*> -> actionCandidate.mapNotNull { entry ->
                val binding = toStringKeyMap(entry) ?: return@mapNotNull null
                binding.takeIf { it["action"] is String }
            }

            else -> {
                val binding = toStringKeyMap(actionCandidate)
                if (binding != null && binding["action"] is String) listOf(binding) else emptyList()
            }
        }
    }

    private fun resolveStatePathParam(
        rawParams: Map<String, Any?>,
        state: Map<String, Any?>,
        repeatScope: RepeatScope?,
        computedFunctions: Map<String, FlatComputedFunction>,
        vararg keys: String
    ): String {
        keys.forEach { key ->
            val rawValue = rawParams[key] ?: return@forEach
            val rawMap = toStringKeyMap(rawValue)
            val resolved = if (rawMap != null && rawMap.containsKey("\$item")) {
                resolveBindItemPath(rawMap["\$item"]?.toString(), repeatScope)
            } else {
                FlatExprResolver.resolve(rawValue, state, repeatScope, computedFunctions)?.toString()
            }
            val normalized = resolved?.trim().orEmpty()
            if (normalized.isNotBlank()) {
                return normalized
            }
        }
        return ""
    }

    private fun firstNonBlankString(map: Map<String, Any?>, vararg keys: String): String {
        keys.forEach { key ->
            val value = map[key]?.toString()?.trim().orEmpty()
            if (value.isNotBlank()) {
                return value
            }
        }
        return ""
    }

    private fun toInt(value: Any?): Int? {
        return when (value) {
            is Number -> value.toInt()
            is String -> value.toIntOrNull()
            else -> null
        }
    }

    private fun replaceGeneratedIdToken(value: Any?, generatedId: String): Any? {
        return when (value) {
            "\$id" -> generatedId
            is Map<*, *> -> value.entries.associate { (k, v) ->
                k.toString() to replaceGeneratedIdToken(v, generatedId)
            }

            is List<*> -> value.map { replaceGeneratedIdToken(it, generatedId) }
            else -> value
        }
    }
}

internal class FlatWatchRuntime(elements: Map<String, FlatElement>) {

    private val entries: List<WatchEntry> = elements.flatMap { (elementId, element) ->
        element.watch.orEmpty().map { (statePath, binding) ->
            WatchEntry(
                key = "$elementId::$statePath",
                statePath = normalizePointer(statePath),
                actionBinding = binding
            )
        }
    }
    private val lastValues = mutableMapOf<String, Any?>()
    private var initialized = false

    fun collectTriggered(state: Map<String, Any?>): List<Any?> {
        if (entries.isEmpty()) return emptyList()
        val triggered = mutableListOf<Any?>()
        entries.forEach { entry ->
            val currentValue = deepCopyValue(FlatSpecParser.getAtPath(state, entry.statePath))
            val previousValue = lastValues[entry.key]
            if (initialized && !deepEquals(previousValue, currentValue)) {
                triggered += entry.actionBinding
            }
            lastValues[entry.key] = currentValue
        }
        if (!initialized) {
            initialized = true
        }
        return triggered
    }
}

@Composable
fun FlatSpecContent(
    spec: FlatSpec,
    resolveAssetUrl: (String) -> String = { raw -> raw },
    computedFunctions: Map<String, FlatComputedFunction> = emptyMap(),
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val stateStore = remember(spec) {
        mutableStateMapOf<String, Any?>().apply { putAll(spec.state) }
    }
    val combinedComputedFunctions = remember(computedFunctions) {
        DefaultComputedFunctions + computedFunctions
    }
    val watchRuntime = remember(spec) { FlatWatchRuntime(spec.elements) }

    val onOpenUrl: (String) -> Unit = { url ->
        runCatching { context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url))) }
    }
    val onSetState: (String, Any?) -> Unit = { path, value ->
        val normalizedPath = normalizePointer(path)
        if (normalizedPath.isNotBlank()) {
            FlatSpecParser.setAtPath(stateStore, normalizedPath, value)
        }
    }
    val onAction: (Any?, RepeatScope?) -> Int = { actionCandidate, repeatScope ->
        FlatActionRuntime.execute(
            actionCandidate = actionCandidate,
            stateStore = stateStore,
            repeatScope = repeatScope,
            computedFunctions = combinedComputedFunctions,
            onOpenUrl = onOpenUrl
        )
    }

    LaunchedEffect(spec) {
        snapshotFlow { stateStore.toMap() }.collect { snapshot ->
            val triggered = watchRuntime.collectTriggered(snapshot)
            var remainingBudget = WATCH_ACTION_BUDGET
            triggered.forEach { candidate ->
                if (remainingBudget <= 0) return@forEach
                val executed = onAction(candidate, null)
                remainingBudget -= executed.coerceAtLeast(1)
            }
        }
    }

    CompositionLocalProvider(
        LocalFlatSpecAssetResolver provides resolveAssetUrl,
        LocalFlatSpecComputedFunctions provides combinedComputedFunctions
    ) {
        RenderElement(
            elementId = spec.root,
            elements = spec.elements,
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

@Composable
private fun RenderElement(
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
    if (elementId in activePath) return
    val element = elements[elementId] ?: return
    val computedFunctions = LocalFlatSpecComputedFunctions.current
    if (!FlatExprResolver.evaluateVisible(element.visible, state, repeatScope, computedFunctions)) return

    val repeatedChildScopes: List<RepeatScope>? = buildRepeatScopes(element.repeat, state)

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
private fun RenderByType(
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
    when (type.lowercase()) {
        "stack" -> RenderStack(elementId, props, children, elements, state, repeatScope, repeatedChildScopes, onOpenUrl, onSetState, onAction, activePath, modifier)
        "row" -> RenderStack(elementId, props + mapOf("direction" to "horizontal"), children, elements, state, repeatScope, repeatedChildScopes, onOpenUrl, onSetState, onAction, activePath, modifier)
        "column" -> RenderStack(elementId, props + mapOf("direction" to "vertical"), children, elements, state, repeatScope, repeatedChildScopes, onOpenUrl, onSetState, onAction, activePath, modifier)
        "list" -> RenderList(children, elements, state, repeatScope, repeatedChildScopes, onOpenUrl, onSetState, onAction, activePath, modifier)
        "card" -> RenderCard(elementId, props, children, elements, state, repeatScope, repeatedChildScopes, onOpenUrl, onSetState, onAction, activePath, modifier)
        "table" -> RenderDirectTable(props, state, onOpenUrl, modifier)
        "formula" -> RenderFormula(props, modifier)
        "chart", "barchart", "bar_chart" -> RenderChart(props, state, modifier)
        "code", "codeblock", "code_block", "pre", "preformatted" -> RenderCodeBlock(
            codeBlock = codeBlockFromProps(props, defaultLanguage = "text", isConsole = false),
            modifier = modifier
        )
        "console", "consolelog", "console_log", "terminal", "logoutput", "log_output" -> RenderCodeBlock(
            codeBlock = codeBlockFromProps(props, defaultLanguage = "console", isConsole = true),
            modifier = modifier
        )
        "text" -> RenderText(props, modifier)
        "emailpreview", "email_preview" -> RenderEmailPreview(props, modifier)
        "image" -> RenderImage(props, onOpenUrl, modifier)
        "icon" -> RenderIcon(props, modifier)
        "button" -> RenderButton(props, onMap, repeatScope, onAction, modifier)
        "divider" -> RenderDivider(modifier)
        "tabs" -> RenderTabs(props, elements, state, repeatScope, onOpenUrl, onSetState, onAction, activePath, modifier)
        "modal" -> RenderModal(props, children, elements, state, repeatScope, onOpenUrl, onSetState, onAction, activePath, modifier)
        "textfield" -> RenderTextField(props, onSetState, state, repeatScope, modifier)
        "checkbox" -> RenderCheckBox(props, onSetState, state, repeatScope, modifier)
        "choicepicker" -> RenderChoicePicker(props, onSetState, state, repeatScope, modifier)
        "slider" -> RenderSlider(props, onSetState, state, repeatScope, modifier)
        "datetimeinput" -> RenderDateTimeInput(props, onSetState, state, repeatScope, modifier)
        "video" -> RenderVideo(props, onOpenUrl, modifier)
        "audioplayer" -> RenderAudioPlayer(props, onOpenUrl, modifier)
        else -> if (children.isNotEmpty()) {
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
        }
    }
}

@Composable
private fun RenderChildren(
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
        isDetachedMediaDumpElement(childId, elements)
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

@Composable
private fun RenderFlatSourceSection(
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
private fun RenderFlatSourceRow(
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

private fun asFloat(value: Any?): Float? = when (value) {
    is Number -> value.toFloat()
    is String -> value.toFloatOrNull()
    else -> null
}

private fun asDp(value: Any?): androidx.compose.ui.unit.Dp? {
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

internal fun resolveMediaUrlCandidate(
    props: Map<String, Any?>,
    preferredKeys: List<String>
): String {
    return preferredKeys.firstNotNullOfOrNull { key ->
        extractMediaUrlToken(props[key])
    }.orEmpty()
}

private fun mediaTextContext(props: Map<String, Any?>): String {
    val tokens = buildList<String> {
        listOf("alt", "title", "label", "caption", "text").forEach { key ->
            props[key]?.toString()?.trim()?.takeIf { it.isNotBlank() }?.let(::add)
        }
        (props["accessibility"] as? Map<*, *>)
            ?.get("label")
            ?.toString()
            ?.trim()
            ?.takeIf { it.isNotBlank() }
            ?.let(::add)
    }
    return tokens.joinToString(" ").lowercase()
}

private fun parseUrlHost(rawUrl: String): String {
    val normalized = rawUrl.trim()
    if (normalized.isBlank()) return ""
    return runCatching {
        URI(normalized).host?.lowercase().orEmpty()
    }.getOrElse { "" }
}

private fun flatSourceCueText(value: String?): Boolean {
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

private fun flatSourceCueFromSelf(
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

private fun flatDirectSourceCue(
    childId: String,
    element: FlatElement,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    computedFunctions: Map<String, FlatComputedFunction>
): Boolean {
    if (flatSourceCueText(childId) || flatSourceCueFromSelf(childId, element.props)) {
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

private fun flatResolvedString(
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

private fun toFlatExternalUrl(value: String?): String? {
    val normalized = NativePayloadParser.canonicalizeNetworkUrlToken(
        flatSanitizeUrlToken(value.orEmpty())
    )
    if (normalized.isBlank()) return null
    val scheme = runCatching { Uri.parse(normalized).scheme?.lowercase() }.getOrNull()
    return normalized.takeIf { scheme == "http" || scheme == "https" }
}

private fun flatSanitizeUrlToken(value: String): String =
    NativeStructureParsing.sanitizeUrlToken(value)

private fun parseFlatSourceLinksFromLine(line: String): List<ParsedButton> {
    return NativeSourceParsing.parseSourceLinksFromLine(
        line = line,
        stripLeadingBulletMarker = NativeStructureParsing::stripLeadingBulletMarker,
        sanitizeDisplayText = NativeTextFormatter::sanitizeDisplayText,
        sanitizeUrlToken = ::flatSanitizeUrlToken,
        toExternalUrl = ::toFlatExternalUrl,
        containsUrlLikeToken = NativeTextFormatter::containsUrlLikeToken
    )
}

private fun flatSourceTitleFromText(value: String): String? {
    return value
        .lineSequence()
        .map { it.trim() }
        .firstOrNull { NativeSourceParsing.isSourceHeadingLine(it) }
        ?.let(NativeSourceParsing::normalizeSourceHeadingToken)
        ?.takeIf { it.isNotBlank() }
}

private fun flatSourceUrlCandidateFromAction(
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

private fun flatSourceLabelFromButtonProps(
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

private fun parseFlatSourceLinksFromListItems(value: Any?): List<ParsedButton> {
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

private fun collectFlatSourceLinks(
    elementIds: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    computedFunctions: Map<String, FlatComputedFunction>,
    activePath: Set<String> = emptySet()
): FlatSourceSection {
    var title: String? = null
    val links = mutableListOf<ParsedButton>()

    elementIds.forEach { childId ->
        if (childId in activePath) return@forEach
        val element = elements[childId] ?: return@forEach
        val resolvedProps = element.props.mapValues { (_, value) ->
            FlatExprResolver.resolve(value, state, repeatScope, computedFunctions)
        }
        when (element.type.lowercase()) {
            "text" -> {
                val rawText = listOf("text", "title", "label", "content", "value")
                    .firstNotNullOfOrNull { key -> resolvedProps[key]?.toString() }
                    .orEmpty()
                title = title ?: flatSourceTitleFromText(rawText)
                rawText
                    .lineSequence()
                    .map { it.trim() }
                    .filter { it.isNotBlank() && !NativeSourceParsing.isSourceHeadingLine(it) }
                    .flatMap { parseFlatSourceLinksFromLine(it).asSequence() }
                    .forEach(links::add)
            }
            "button" -> {
                val url = flatSourceUrlCandidateFromAction(
                    onMap = element.on,
                    props = resolvedProps,
                    state = state,
                    repeatScope = repeatScope,
                    computedFunctions = computedFunctions
                )
                if (url != null) {
                    links += ParsedButton(
                        label = flatSourceLabelFromButtonProps(
                            props = resolvedProps,
                            fallbackUrl = url,
                            state = state,
                            repeatScope = repeatScope,
                            computedFunctions = computedFunctions
                        ),
                        url = url
                    )
                }
            }
            "list" -> {
                links += parseFlatSourceLinksFromListItems(resolvedProps["items"])
                val nestedChildren = element.children.ifEmpty {
                    resolvedProps["child"]?.toString()?.takeIf { it.isNotBlank() }?.let(::listOf)
                        ?: emptyList()
                }
                if (nestedChildren.isNotEmpty()) {
                    val nested = collectFlatSourceLinks(
                        elementIds = nestedChildren,
                        elements = elements,
                        state = state,
                        repeatScope = repeatScope,
                        computedFunctions = computedFunctions,
                        activePath = activePath + childId
                    )
                    title = title ?: nested.title
                    links += nested.links
                }
            }
            else -> {
                val nestedChildren = element.children.ifEmpty {
                    resolvedProps["child"]?.toString()?.takeIf { it.isNotBlank() }?.let(::listOf)
                        ?: emptyList()
                }
                if (nestedChildren.isNotEmpty()) {
                    val nested = collectFlatSourceLinks(
                        elementIds = nestedChildren,
                        elements = elements,
                        state = state,
                        repeatScope = repeatScope,
                        computedFunctions = computedFunctions,
                        activePath = activePath + childId
                    )
                    title = title ?: nested.title
                    links += nested.links
                }
            }
        }
    }

    return FlatSourceSection(
        title = title,
        links = links.distinctBy { flatCanonicalSourceUrl(it.url) }
    )
}

private fun extractFlatSourceSection(
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

private fun flatCanonicalSourceUrl(raw: String): String {
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

private fun flatSourceLabelFromUrl(url: String): String {
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

private fun flatSourceUrlDisplay(url: String): String {
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

private fun looksLikeWeatherContext(sourceUrl: String, props: Map<String, Any?>): Boolean {
    val combined = "${sourceUrl.lowercase()} ${mediaTextContext(props)}"
    return listOf(
        "weather",
        "forecast",
        "temperature",
        "humidity",
        "wind",
        "rain",
        "sun",
        "cloud",
        "bengaluru",
        "bangalore"
    ).any { token -> combined.contains(token) }
}

private fun buildImageFallbackSeed(sourceUrl: String, props: Map<String, Any?>): String {
    val host = parseUrlHost(sourceUrl)
    val weather = looksLikeWeatherContext(sourceUrl, props)
    return when {
        weather -> "genuicraft_weather_hero"
        host.contains("wikimedia.org") -> "genuicraft_city_hero"
        else -> "genuicraft_media_hero"
    }
}

internal fun deriveImageFallbackUrl(
    sourceUrl: String,
    props: Map<String, Any?>
): String? {
    val explicitFallback = extractMediaUrlToken(props["fallbackUrl"]).orEmpty()
    if (explicitFallback.isNotBlank()) {
        return explicitFallback
    }
    val host = parseUrlHost(sourceUrl)
    if (host.isBlank()) return null
    if (!host.contains("upload.wikimedia.org") && !host.contains("wikipedia.org")) {
        return null
    }
    val seed = buildImageFallbackSeed(sourceUrl, props)
    return "https://picsum.photos/seed/$seed/1280/720"
}

private fun parseAspectRatio(value: Any?): Float? {
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

private fun resolveImageScale(props: Map<String, Any?>): ContentScale {
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

private fun resolveIconSize(props: Map<String, Any?>): androidx.compose.ui.unit.Dp {
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

private fun stackDirection(props: Map<String, Any?>): String {
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

private fun stackGap(props: Map<String, Any?>): androidx.compose.ui.unit.Dp {
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

private fun buildRepeatScopes(
    repeat: RepeatConfig?,
    state: Map<String, Any?>
): List<RepeatScope>? {
    val repeatConfig = repeat ?: return null
    val items = (FlatSpecParser.getAtPath(state, repeatConfig.statePath) as? List<*>).orEmpty()
    return items.mapIndexed { index, item ->
        RepeatScope(
            item = item,
            index = index,
            basePath = combinePointerPath(repeatConfig.statePath, index.toString())
        )
    }
}

private fun isHorizontalRowElement(element: FlatElement?): Boolean {
    if (element == null) return false
    return when (element.type.lowercase()) {
        "row" -> true
        "stack" -> stackDirection(element.props) == "horizontal"
        else -> false
    }
}

private fun hasOnlyTextCellChildren(element: FlatElement, elements: Map<String, FlatElement>): Boolean {
    return element.children.isNotEmpty() && element.children.all { childId ->
        elements[childId]?.type?.equals("text", ignoreCase = true) == true
    }
}

private fun textLikeValue(props: Map<String, Any?>): Any? {
    return props["text"] ?: props["title"] ?: props["label"] ?: props["content"] ?: props["value"]
}

private fun resolveTableHeaderLabel(
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

private fun isWeatherHeaderLabel(label: String): Boolean {
    if (label.isBlank()) return false
    val token = label.lowercase()
    val keywords = listOf(
        "weather",
        "temp",
        "temperature",
        "condition",
        "high",
        "low",
        "max",
        "min",
        "humidity",
        "wind",
        "rain",
        "rainy",
        "precip",
        "forecast",
        "climate",
        "sunshine",
        "sunny",
        "uv",
        "feels"
    )
    return keywords.any { keyword -> token.contains(keyword) }
}

private fun isFlightHeaderLabel(label: String): Boolean {
    if (label.isBlank()) return false
    val token = label.lowercase()
    val keywords = listOf(
        "airline",
        "carrier",
        "flight",
        "depart",
        "departure",
        "arrival",
        "arrive",
        "duration",
        "fare",
        "price",
        "cost",
        "stops",
        "layover"
    )
    return keywords.any { keyword -> token.contains(keyword) }
}

private fun isStrongFlightHeaderLabel(label: String): Boolean {
    if (label.isBlank()) return false
    val token = label.lowercase()
    val keywords = listOf(
        "airline",
        "carrier",
        "flight",
        "depart",
        "departure",
        "arrival",
        "arrive",
        "takeoff",
        "landing",
        "origin",
        "destination"
    )
    return keywords.any { keyword -> token.contains(keyword) }
}

private fun isBookingEntityHeaderLabel(label: String): Boolean {
    if (label.isBlank()) return false
    val token = label.lowercase()
    val keywords = listOf(
        "hotel",
        "property",
        "provider",
        "option",
        "listing",
        "vendor",
        "airline",
        "plan",
        "package",
        "name"
    )
    return keywords.any { keyword -> token.contains(keyword) }
}

private fun isBookingValueHeaderLabel(label: String): Boolean {
    if (label.isBlank()) return false
    val token = label.lowercase()
    val keywords = listOf(
        "price",
        "cost",
        "fare",
        "rate",
        "night",
        "duration",
        "room",
        "amenity",
        "wifi",
        "rating",
        "review",
        "book",
        "reserve",
        "deal",
        "url",
        "link"
    )
    return keywords.any { keyword -> token.contains(keyword) }
}

private fun isScheduleHeaderLabel(label: String): Boolean {
    if (label.isBlank()) return false
    val token = label.lowercase()
    val timeKeywords = listOf("time", "date", "day", "slot", "start", "end")
    val eventKeywords = listOf("event", "activity", "agenda", "session", "task", "title", "stop", "location")
    return timeKeywords.any { token.contains(it) } || eventKeywords.any { token.contains(it) }
}

private fun isStatusHeaderLabel(label: String): Boolean {
    if (label.isBlank()) return false
    val token = label.lowercase()
    val keywords = listOf(
        "status",
        "state",
        "stage",
        "progress",
        "eta",
        "updated",
        "resolved",
        "tracking",
        "phase"
    )
    return keywords.any { keyword -> token.contains(keyword) }
}

internal fun looksLikeProcessStateTable(
    headers: List<String>,
    rows: List<List<String>>,
    domain: String = "generic"
): Boolean {
    if (rows.size !in 2..8) return false
    if (domain !in setOf("status", "schedule", "generic")) return false
    val tokens = headers.map(::normalizeTableHeaderForMatch)
    val hasState = tokens.any { token ->
        token == "state" ||
            token == "ui state" ||
            token == "screen state" ||
            token == "status" ||
            token == "step" ||
            token == "stage"
    }
    val hasVisuals = tokens.any { token ->
        token.contains("visual") ||
            token.contains("screen") ||
            token.contains("view") ||
            token.contains("interface")
    }
    val hasFeedback = tokens.any { token ->
        token.contains("feedback") ||
            token.contains("message") ||
            token.contains("text") ||
            token.contains("result")
    }
    val rowSignals = rows.flatten().joinToString(" ").lowercase().let { combined ->
        listOf("scan", "success", "invalid", "error", "ready", "state", "overlay", "check-in", "qr").count {
            combined.contains(it)
        }
    }
    return hasState && (hasVisuals || hasFeedback) && rowSignals >= 2
}

internal fun looksLikeIncidentStatusTable(
    headers: List<String>,
    rows: List<List<String>>,
    domain: String = "generic"
): Boolean {
    if (rows.isEmpty()) return false
    if (domain !in setOf("status", "generic")) return false
    val tokens = headers.map(::normalizeTableHeaderForMatch)
    val hasComponent = tokens.any { token ->
        token == "component" ||
            token == "service" ||
            token == "system" ||
            token == "module" ||
            token.contains("service")
    }
    val hasStatus = tokens.any { token ->
        token == "status" ||
            token == "current status" ||
            token == "health" ||
            token.contains("status")
    }
    val hasNotes = tokens.any { token ->
        token == "notes" ||
            token == "impact" ||
            token == "details" ||
            token == "description" ||
            token.contains("message")
    }
    val rowSignals = rows.flatten().joinToString(" ").lowercase().let { combined ->
        listOf("outage", "degraded", "operational", "partial", "incident", "failure", "delayed").count {
            combined.contains(it)
        }
    }
    return hasComponent && hasStatus && hasNotes && rowSignals >= 1
}

internal fun looksLikeMarketHoldingsTable(
    headers: List<String>,
    rows: List<List<String>>,
    domain: String = "generic"
): Boolean {
    if (headers.size < 3 || rows.isEmpty()) return false
    if (domain !in setOf("generic", "comparison", "finance", "portfolio", "market", "investment", "schedule")) {
        return false
    }
    val tokens = headers.map(::normalizeTableHeaderForMatch)
    val hasTicker = tokens.any { token ->
        token in setOf("ticker", "symbol", "stock", "asset", "holding", "security") ||
            token.contains("ticker") ||
            token.contains("symbol")
    }
    val hasCurrentValue = tokens.any { token ->
        (token.contains("current") && token.contains("value")) ||
            token.contains("market value") ||
            (token == "value") ||
            (token.contains("price") && !token.contains("change"))
    }
    val hasChange = tokens.any { token ->
        token.contains("change") ||
            token.contains("return") ||
            token.contains("gain") ||
            token.contains("loss") ||
            token.contains("p l")
    }
    val hasPercentColumn = tokens.any { token ->
        token.contains("pct") ||
            token.contains("percent") ||
            token.contains("percentage")
    } || headers.any { header -> header.contains('%') }
    val firstColumnIndex = marketTickerColumnIndex(headers)
    val tickerLikeRows = rows.count { row ->
        row.getOrNull(firstColumnIndex).orEmpty().trim().matches(Regex("""[A-Z][A-Z0-9.\-]{1,6}"""))
    }
    val currencyOrPercentValues = rows.flatten().count { value ->
        val token = value.trim()
        token.contains('$') ||
            token.contains('%') ||
            token.startsWith("+") ||
            token.startsWith("-")
    }
    return hasTicker &&
        hasCurrentValue &&
        (hasChange || hasPercentColumn) &&
        tickerLikeRows >= maxOf(1, rows.size / 2) &&
        currencyOrPercentValues >= rows.size
}

private fun isComparisonFeatureHeader(label: String): Boolean {
    if (label.isBlank()) return false
    val token = label.lowercase()
    val keywords = listOf(
        "feature",
        "metric",
        "criteria",
        "criterion",
        "attribute",
        "spec",
        "dimension",
        "parameter",
        "category",
        "aspect",
        "factor",
        "topic"
    )
    return keywords.any { keyword -> token.contains(keyword) }
}

private fun isComparisonEntityHeader(label: String): Boolean {
    if (label.isBlank()) return false
    val token = normalizeTableHeaderForMatch(label)
    val keywords = listOf(
        "model",
        "product",
        "item",
        "option",
        "name",
        "plan",
        "package",
        "provider",
        "service",
        "tool",
        "device",
        "site",
        "destination",
        "hotel",
        "route",
        "airline",
        "project",
        "task",
        "drink",
        "recipe"
    )
    return keywords.any { keyword -> token.contains(keyword) }
}

private fun normalizeExplicitTableDomain(value: String?): String? {
    val token = value?.trim()?.lowercase()?.takeIf { it.isNotBlank() } ?: return null
    return when {
        token in PLAYLIST_TABLE_DOMAIN_ALIASES -> "playlist"
        token in FORMULA_TABLE_DOMAIN_ALIASES -> "formula"
        else -> token
    }
}

private fun isPlaylistTableHeaderSet(headers: List<String>, domain: String = "generic"): Boolean {
    if (domain == "playlist") return true
    val tokens = headers.map(::normalizeTableHeaderForMatch)
    val hasTrackNumber = headers.any { it.trim() == "#" } ||
        tokens.any { token ->
            token in setOf("no", "number", "track number", "track no", "tracknumber", "track")
        }
    val hasTrackTitle = tokens.any { token ->
        token == "track" ||
            token == "song" ||
            token == "title" ||
            token == "songtitle" ||
            token == "tracktitle" ||
            token.contains("track title") ||
            token.contains("track name") ||
            token.contains("song title") ||
            token.contains("song name")
    }
    val hasArtist = tokens.any { token ->
        token == "artist" ||
            token.contains("artist") ||
            token.contains("performer") ||
            token.contains("band")
    }
    val hasMusicMetadata = tokens.any { token ->
        token.contains("album") ||
            token.contains("genre") ||
            token.contains("mood") ||
            token.contains("tempo") ||
            token.contains("phase") ||
            token.contains("set")
    }
    return hasTrackTitle && (hasTrackNumber || hasArtist || hasMusicMetadata)
}

internal fun isFormulaVariableHeaderSet(headers: List<String>): Boolean {
    val tokens = headers.map(::normalizeTableHeaderForMatch)
    val hasVariable = tokens.any { token ->
        token in setOf("variable", "symbol", "term", "parameter")
    }
    val hasDescription = tokens.any { token ->
        token.contains("description") || token.contains("meaning") || token.contains("definition")
    }
    val hasValue = tokens.any { token ->
        token in setOf("value", "amount", "input value", "given")
    }
    return headers.size in 2..4 && hasVariable && (hasDescription || hasValue)
}

internal fun isCalculationBreakdownHeaderSet(headers: List<String>): Boolean {
    val tokens = headers.map(::normalizeTableHeaderForMatch)
    val hasLabel = tokens.any { token ->
        token in setOf("component", "item", "metric", "field", "label", "cost component")
    }
    val hasAmount = tokens.any { token ->
        token in setOf("amount", "value", "cost", "total", "payment") ||
            token.contains("amount") ||
            token.contains("payment")
    }
    return headers.size == 2 && hasLabel && hasAmount
}

private fun inferTableDomainFromHeaders(headers: List<String>): String {
    val weatherSignals = headers.count(::isWeatherHeaderLabel)
    val flightSignals = headers.count(::isFlightHeaderLabel)
    val strongFlightSignals = headers.count(::isStrongFlightHeaderLabel)
    val bookingEntitySignals = headers.count(::isBookingEntityHeaderLabel)
    val bookingValueSignals = headers.count(::isBookingValueHeaderLabel)
    val scheduleSignals = headers.count(::isScheduleHeaderLabel)
    val statusSignals = headers.count(::isStatusHeaderLabel)
    val featureLike = isComparisonFeatureHeader(headers.firstOrNull().orEmpty())
    val entityLike = isComparisonEntityHeader(headers.firstOrNull().orEmpty())
    return when {
        isPlaylistTableHeaderSet(headers) -> "playlist"
        isFormulaVariableHeaderSet(headers) || isCalculationBreakdownHeaderSet(headers) -> "formula"
        weatherSignals >= 2 -> "weather"
        strongFlightSignals >= 1 && flightSignals >= 2 -> "flight"
        bookingEntitySignals >= 1 && bookingValueSignals >= 2 -> "booking"
        scheduleSignals >= 2 && statusSignals >= 1 -> "status"
        scheduleSignals >= 2 -> "schedule"
        featureLike && headers.size >= 3 -> "comparison"
        entityLike && headers.size >= 4 -> "comparison"
        else -> "generic"
    }
}

private fun shouldPreferComparisonCards(headers: List<String>, compactScreen: Boolean): Boolean {
    if (!compactScreen || headers.size < 3) return false
    val firstHeader = headers.firstOrNull().orEmpty()
    return isComparisonFeatureHeader(firstHeader) ||
        (headers.size >= 4 && isComparisonEntityHeader(firstHeader))
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

private fun stringSetFromTableProp(value: Any?): Set<String> {
    return when (value) {
        is List<*> -> value.mapNotNull { item -> item?.toString()?.trim()?.takeIf { it.isNotBlank() } }.toSet()
        is String -> value
            .split(',', '|')
            .mapNotNull { item -> item.trim().takeIf { it.isNotBlank() } }
            .toSet()
        else -> emptySet()
    }
}

private fun normalizeColumnToken(value: String): String = normalizeTableHeaderForMatch(value)

private fun columnIndexForToken(columns: List<FlatDirectTableColumn>, token: String): Int? {
    val normalized = normalizeColumnToken(token)
    if (normalized.isBlank()) return null
    return columns.indexOfFirst { column ->
        normalizeColumnToken(column.key) == normalized ||
            normalizeColumnToken(column.label) == normalized
    }.takeIf { it >= 0 }
}

private fun numericColumnIndexes(
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

private fun looksLikeNumericTableValue(value: String): Boolean {
    val normalized = value.trim()
    if (normalized.isBlank()) return false
    return Regex("""^[₹$€£]?\s*[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|x|k|K|m|M|bn|hrs?|hours?|mins?|minutes?|days?|°[CF]?)?$""")
        .containsMatchIn(normalized)
}

private fun looksLikeRankHeader(label: String): Boolean {
    val token = normalizeTableHeaderForMatch(label)
    return token in setOf("rank", "id", "number", "no") || label.trim() == "#"
}

private fun looksLikeLongDetailHeader(label: String): Boolean {
    val token = normalizeTableHeaderForMatch(label)
    return listOf("note", "notes", "description", "detail", "details", "justification", "reason", "summary", "remarks")
        .any { keyword -> token.contains(keyword) }
}

private fun isCompactTableBadgeValue(value: String): Boolean {
    val normalized = value.trim()
    if (normalized.isBlank() || isLikelyHttpUrl(normalized)) return false
    if (normalized.contains('\n')) return false
    if (Regex("""[.!?]\s+""").containsMatchIn(normalized)) return false
    return normalized.length <= 34
}

private fun isCompactHighlightColumn(rows: List<List<String>>, index: Int): Boolean {
    val values = rows.mapNotNull { row -> row.getOrNull(index)?.trim()?.takeIf { it.isNotBlank() } }
    if (values.isEmpty()) return false
    val compactCount = values.count(::isCompactTableBadgeValue)
    return compactCount >= maxOf(1, values.size * 2 / 3)
}

private fun inferEntityPrimaryColumnIndex(headers: List<String>, primaryColumn: String?): Int {
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

private fun shouldUseNativeFlightCards(headers: List<String>): Boolean {
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

private fun rankedFlightColumnIndex(headers: List<String>, keywords: List<String>): Int? {
    return headers.indexOfFirst { header ->
        val token = normalizeTableHeaderForMatch(header)
        keywords.any { keyword -> token.contains(keyword) }
    }.takeIf { it >= 0 }
}

private fun looksLikeRankedFlightComparisonTable(headers: List<String>): Boolean {
    val normalized = headers.map(::normalizeTableHeaderForMatch)
    val hasAirline = normalized.any { it.contains("airline") || it.contains("carrier") }
    val hasRank = normalized.any { it.contains("rank") || it.contains("score") || it.contains("order") }
    val hasCost = normalized.any { it.contains("cost") || it.contains("fare") || it.contains("price") }
    val hasDuration = normalized.any { it.contains("time") || it.contains("duration") || it.contains("travel") }
    val hasStops = normalized.any { it.contains("layover") || it.contains("stop") || it.contains("connection") }
    return hasAirline && (hasRank || hasCost) && (hasDuration || hasStops)
}

private fun compactRankBadge(rawRank: String, fallbackIndex: Int): String {
    val number = Regex("""\d+""").find(rawRank)?.value ?: (fallbackIndex + 1).toString()
    return "#$number"
}

private fun looksLikeMultiLegFlightTable(headers: List<String>): Boolean {
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

private fun detectTableShape(
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
        isComparisonEntityHeader(firstHeader) && columnCount >= 3 -> FlatTableShape.ENTITY_ROW
        numericLikeColumns >= 2 && columnCount <= 4 -> FlatTableShape.NUMERIC_METRICS
        else -> FlatTableShape.GENERIC_GRID
    }
}

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
        explicitDomain != null && explicitDomain in CARD_FIRST_TABLE_DOMAINS -> explicitDomain
        explicitDomain == "generic" && inferredDomain in CARD_FIRST_TABLE_DOMAINS -> inferredDomain
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
        shape = shape,
        cardMappingStatus = cardMappingStatus,
        renderMode = renderMode
    )
}

private fun applyStackModifier(
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

private fun hasHorizontalContainerPadding(props: Map<String, Any?>): Boolean =
    asFlatSpacingDp(props["padding"]) != null ||
        asFlatSpacingDp(props["paddingHorizontal"]) != null ||
        asFlatSpacingDp(props["contentPadding"]) != null ||
        asFlatSpacingDp(props["contentPaddingHorizontal"]) != null

internal fun rowChildFlex(element: FlatElement?): Float =
    asFloat(element?.props?.get("flex"))
        ?.takeIf { it > 0f }
        ?: 0f

private fun isPaddedContainerElement(element: FlatElement?): Boolean {
    if (element == null || !hasHorizontalContainerPadding(element.props)) return false
    return when (element.type.trim().lowercase()) {
        "stack", "column", "row", "list", "container", "box" -> true
        else -> false
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun RenderStack(
    elementId: String,
    props: Map<String, Any?>,
    children: List<String>,
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
    val compactScreen = LocalConfiguration.current.screenWidthDp <= 480
    val computedFunctions = LocalFlatSpecComputedFunctions.current
    extractFlatSourceSection(
        elementId = elementId,
        props = props,
        children = children,
        elements = elements,
        state = state,
        repeatScope = repeatScope,
        computedFunctions = computedFunctions,
        allowChildSourceCue = false
    )?.let { sourceSection ->
        RenderFlatSourceSection(
            section = sourceSection,
            onOpenUrl = onOpenUrl,
            modifier = modifier,
            wrapInCard = false,
            showTitle = sourceSection.title != null
        )
        return
    }
    val tableModel = if (repeatedChildScopes == null) {
        extractFlatTableModel(
            containerChildren = children,
            containerProps = props,
            elements = elements,
            state = state,
            compactScreen = compactScreen
        )
    } else {
        null
    }
    if (tableModel != null) {
        RenderTableLayout(
            tableModel = tableModel,
            elementId = elementId,
            props = props,
            elements = elements,
            state = state,
            repeatScope = repeatScope,
            onOpenUrl = onOpenUrl,
            onSetState = onSetState,
            onAction = onAction,
            activePath = activePath,
            modifier = modifier
        )
        return
    }

    val direction = stackDirection(props)
    val gap = stackGap(props)
    val align = (
        props["align"]?.toString()
            ?: props["crossAxisAlignment"]?.toString()
            ?: if (direction == "horizontal") props["verticalAlignment"]?.toString() else props["horizontalAlignment"]?.toString()
        )
        ?.trim()
        ?.lowercase()
        .orEmpty()
    val justify = (
        props["justify"]?.toString()
            ?: props["mainAxisAlignment"]?.toString()
            ?: if (direction == "horizontal") props["horizontalArrangement"]?.toString() else props["verticalArrangement"]?.toString()
        )
        ?.trim()
        ?.lowercase()
        .orEmpty()
    val wrap = props["wrap"]?.toString()?.trim()?.lowercase() == "wrap"
    val childElements = children.mapNotNull { childId -> elements[childId] }
    val allButtonChildren = childElements.isNotEmpty() &&
        childElements.all { child -> child.type.equals("button", ignoreCase = true) }
    val hasLongButtonLabel = childElements.any { child ->
        child.props["label"]?.toString()?.trim()?.length ?: 0 > 20
    }
    val autoWrapButtonRow = direction == "horizontal" &&
        compactScreen &&
        !wrap &&
        allButtonChildren &&
        children.size >= 2
    val hasImageChild = childElements.any { child ->
        child.type.equals("image", ignoreCase = true)
    }
    val hasLongTextChild = childElements.any { child ->
        if (!child.type.equals("text", ignoreCase = true)) return@any false
        val rawText = (
            child.props["text"]
                ?: child.props["title"]
                ?: child.props["label"]
                ?: child.props["content"]
                ?: child.props["value"]
            )
            ?.toString()
            .orEmpty()
        rawText.length >= 90
    }
    val forceVerticalMediaTextRow = direction == "horizontal" &&
        compactScreen &&
        !wrap &&
        children.size in 2..3 &&
        hasImageChild &&
        hasLongTextChild
    val compactIconTextRow = direction == "horizontal" &&
        compactScreen &&
        !wrap &&
        children.size == 2 &&
        childElements.getOrNull(0)?.type?.equals("icon", ignoreCase = true) == true &&
        childElements.getOrNull(1)?.type?.equals("text", ignoreCase = true) == true
    val forceVerticalCardRow = direction == "horizontal" &&
        compactScreen &&
        !wrap &&
        children.size >= 2 &&
        childElements.all { child -> child.type.equals("card", ignoreCase = true) }
    val compactTextButtonRow = direction == "horizontal" &&
        compactScreen &&
        !wrap &&
        children.size == 2 &&
        childElements.any { child -> child.type.equals("text", ignoreCase = true) } &&
        childElements.any { child -> child.type.equals("button", ignoreCase = true) }
    val forceVerticalButtonStack = autoWrapButtonRow && hasLongButtonLabel
    val stackModifier = applyStackModifier(modifier, props, direction)
    val childTextHorizontalPadding = if (hasHorizontalContainerPadding(props)) {
        0.dp
    } else {
        LocalFlatSpecTextHorizontalPadding.current
    }

    if (direction == "horizontal") {
        val horizontalArrangement: Arrangement.Horizontal = when (justify) {
            "center" -> if (gap > 0.dp) Arrangement.spacedBy(gap, Alignment.CenterHorizontally) else Arrangement.Center
            "end" -> if (gap > 0.dp) Arrangement.spacedBy(gap, Alignment.End) else Arrangement.End
            "between" -> Arrangement.SpaceBetween
            "around" -> Arrangement.SpaceAround
            else -> if (gap > 0.dp) Arrangement.spacedBy(gap) else Arrangement.Start
        }
        if (compactIconTextRow) {
            CompositionLocalProvider(LocalFlatSpecTextHorizontalPadding provides childTextHorizontalPadding) {
                Row(
                    modifier = stackModifier,
                    horizontalArrangement = Arrangement.spacedBy(gap),
                    verticalAlignment = Alignment.Top
                ) {
                    RenderElement(
                        elementId = children[0],
                        elements = elements,
                        state = state,
                        repeatScope = repeatScope,
                        onOpenUrl = onOpenUrl,
                        onSetState = onSetState,
                        onAction = onAction,
                        activePath = activePath
                    )
                    RenderElement(
                        elementId = children[1],
                        elements = elements,
                        state = state,
                        repeatScope = repeatScope,
                        onOpenUrl = onOpenUrl,
                        onSetState = onSetState,
                        onAction = onAction,
                        activePath = activePath,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
            return
        }
        if (forceVerticalButtonStack || forceVerticalCardRow) {
            CompositionLocalProvider(LocalFlatSpecTextHorizontalPadding provides childTextHorizontalPadding) {
                Column(
                    modifier = stackModifier,
                    verticalArrangement = Arrangement.spacedBy(gap)
                ) {
                    RenderChildren(
                        children = children,
                        elements = elements,
                        state = state,
                        repeatScope = repeatScope,
                        repeatedChildScopes = repeatedChildScopes,
                        onOpenUrl = onOpenUrl,
                        onSetState = onSetState,
                        onAction = onAction,
                        activePath = activePath
                    )
                }
            }
            return
        }
        if (compactTextButtonRow) {
            val textChildId = children.firstOrNull { childId ->
                elements[childId]?.type?.equals("text", ignoreCase = true) == true
            }
            val buttonChildId = children.firstOrNull { childId ->
                elements[childId]?.type?.equals("button", ignoreCase = true) == true
            }
            if (textChildId != null && buttonChildId != null) {
                CompositionLocalProvider(LocalFlatSpecTextHorizontalPadding provides childTextHorizontalPadding) {
                    Column(
                        modifier = stackModifier,
                        verticalArrangement = Arrangement.spacedBy(if (gap > 0.dp) gap else 6.dp)
                    ) {
                        RenderElement(
                            elementId = textChildId,
                            elements = elements,
                            state = state,
                            repeatScope = repeatScope,
                            onOpenUrl = onOpenUrl,
                            onSetState = onSetState,
                            onAction = onAction,
                            activePath = activePath
                        )
                        RenderElement(
                            elementId = buttonChildId,
                            elements = elements,
                            state = state,
                            repeatScope = repeatScope,
                            onOpenUrl = onOpenUrl,
                            onSetState = onSetState,
                            onAction = onAction,
                            activePath = activePath,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }
                return
            }
        }
        if (forceVerticalMediaTextRow) {
            CompositionLocalProvider(LocalFlatSpecTextHorizontalPadding provides childTextHorizontalPadding) {
                Column(
                    modifier = stackModifier,
                    verticalArrangement = Arrangement.spacedBy(gap)
                ) {
                    RenderChildren(
                        children = children,
                        elements = elements,
                        state = state,
                        repeatScope = repeatScope,
                        repeatedChildScopes = repeatedChildScopes,
                        onOpenUrl = onOpenUrl,
                        onSetState = onSetState,
                        onAction = onAction,
                        activePath = activePath
                    )
                }
            }
            return
        }
        if (wrap || autoWrapButtonRow) {
            CompositionLocalProvider(LocalFlatSpecTextHorizontalPadding provides childTextHorizontalPadding) {
                FlowRow(
                    modifier = stackModifier,
                    horizontalArrangement = horizontalArrangement,
                    verticalArrangement = Arrangement.spacedBy(gap)
                ) {
                    RenderChildren(
                        children = children,
                        elements = elements,
                        state = state,
                        repeatScope = repeatScope,
                        repeatedChildScopes = repeatedChildScopes,
                        onOpenUrl = onOpenUrl,
                        onSetState = onSetState,
                        onAction = onAction,
                        activePath = activePath
                    )
                }
            }
            return
        }
        val verticalAlignment = when (align) {
            "center" -> Alignment.CenterVertically
            "end" -> Alignment.Bottom
            else -> Alignment.Top
        }
        CompositionLocalProvider(LocalFlatSpecTextHorizontalPadding provides childTextHorizontalPadding) {
            Row(
                modifier = stackModifier,
                horizontalArrangement = horizontalArrangement,
                verticalAlignment = verticalAlignment
            ) {
                RenderRowChildren(
                    children = children,
                    elements = elements,
                    state = state,
                    repeatScope = repeatScope,
                    repeatedChildScopes = repeatedChildScopes,
                    onOpenUrl = onOpenUrl,
                    onSetState = onSetState,
                    onAction = onAction,
                    activePath = activePath
                )
            }
        }
        return
    }

    val verticalArrangement: Arrangement.Vertical = when (justify) {
        "center" -> if (gap > 0.dp) Arrangement.spacedBy(gap, Alignment.CenterVertically) else Arrangement.Center
        "end" -> if (gap > 0.dp) Arrangement.spacedBy(gap, Alignment.Bottom) else Arrangement.Bottom
        "between" -> Arrangement.SpaceBetween
        "around" -> Arrangement.SpaceAround
        else -> if (gap > 0.dp) Arrangement.spacedBy(gap) else Arrangement.Top
    }
    val horizontalAlignment = when (align) {
        "center" -> Alignment.CenterHorizontally
        "end" -> Alignment.End
        else -> Alignment.Start
    }
    CompositionLocalProvider(LocalFlatSpecTextHorizontalPadding provides childTextHorizontalPadding) {
        Column(
            modifier = stackModifier,
            verticalArrangement = verticalArrangement,
            horizontalAlignment = horizontalAlignment
        ) {
            RenderChildren(
                children = children,
                elements = elements,
                state = state,
                repeatScope = repeatScope,
                repeatedChildScopes = repeatedChildScopes,
                onOpenUrl = onOpenUrl,
                onSetState = onSetState,
                onAction = onAction,
                activePath = activePath
            )
        }
    }
}

@Composable
private fun RowScope.RenderRowChildren(
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

@Composable
private fun RenderTableLayout(
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
    val compactScreen = screenWidthDp <= 480
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
    val cardsRequested =
        tableModel.renderMode == FlatTableRenderMode.WEATHER_CARDS ||
            tableModel.renderMode == FlatTableRenderMode.FLIGHT_CARDS ||
            tableModel.renderMode == FlatTableRenderMode.BOOKING_CARDS
    val autoHorizontalScroll = shouldUseHorizontalTableScroll(
        compactScreen = compactScreen,
        screenWidthDp = screenWidthDp,
        headers = tableModel.headers,
        rows = tableRows
    )

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

    if (tableModel.renderMode == FlatTableRenderMode.WEATHER_CARDS) {
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
            NativeFlightUiRenderer.RenderFlightRows(flightRows)
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
                spacing = spacing
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
        explicitDomain != null && explicitDomain in CARD_FIRST_TABLE_DOMAINS -> explicitDomain
        explicitDomain == "generic" && inferredDomain in CARD_FIRST_TABLE_DOMAINS -> inferredDomain
        explicitDomain != null -> explicitDomain
        else -> inferredDomain
    }
    val explicitPreferredPresentation = props["preferredPresentation"]?.toString()?.trim()?.lowercase()
        ?.takeIf { it in setOf("cards", "table") }
    val preferredPresentation = when {
        explicitPreferredPresentation == null -> if (domain in CARD_FIRST_TABLE_DOMAINS) "cards" else "table"
        explicitDomain == "generic" &&
            explicitPreferredPresentation == "table" &&
            domain in CARD_FIRST_TABLE_DOMAINS -> "cards"
        else -> explicitPreferredPresentation
    }
    val shape = detectTableShape(headerLabels, resolvedRows, domain)
    val primaryColumn = props["primaryColumn"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
    val highlightColumns = stringSetFromTableProp(props["highlightColumns"])
    val numericColumns = stringSetFromTableProp(props["numericColumns"])
    val entityMedia = extractTableEntityMedia(props)
    val renderMode = when {
        looksLikeProcessStateTable(headerLabels, resolvedRows, domain) -> FlatTableRenderMode.PROCESS_CARDS
        domain == "weather" -> FlatTableRenderMode.WEATHER_CARDS
        domain == "flight" -> FlatTableRenderMode.FLIGHT_CARDS
        domain == "booking" -> FlatTableRenderMode.BOOKING_CARDS
        domain == "playlist" -> FlatTableRenderMode.PLAYLIST_CARDS
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

private fun tableEntityMediaLookupKeys(value: String): Set<String> {
    val normalized = normalizeTableHeaderForMatch(value)
    val compact = normalized.replace(" ", "")
    val underscore = normalized.replace(" ", "_")
    return setOf(value.trim().lowercase(), normalized, compact, underscore)
        .filter { it.isNotBlank() }
        .toSet()
}

private fun entityMediaForColumn(
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

private fun resolveDirectTableColumns(
    props: Map<String, Any?>,
    rows: List<Any?>
): List<FlatDirectTableColumn> {
    val explicitColumns = parseDirectTableColumns(props["columns"])
    if (explicitColumns.isNotEmpty()) return explicitColumns

    val tablePayload = toStringKeyMap(props["table"])
    val payloadColumns = parseDirectTableColumns(tablePayload?.get("columns"))
    if (payloadColumns.isNotEmpty()) return payloadColumns

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

private fun parseDirectTableColumns(value: Any?): List<FlatDirectTableColumn> {
    val list = value as? List<*> ?: return emptyList()
    return list.mapIndexedNotNull { index, raw ->
        when (raw) {
            is String -> {
                val label = raw.trim()
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

private fun resolveDirectTableRows(
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

private fun resolveDirectTableRow(
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

private fun resolveDirectTableCellValue(
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

private fun extractDirectTableRowList(row: Map<String, Any?>): List<Any?>? {
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

private fun resolveRowMapColumnValue(
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

private fun buildDirectTableNormalizedRowLookup(
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

private fun normalizeTableLookupToken(raw: String): String {
    return raw.trim().lowercase().replace(Regex("[^a-z0-9]+"), "")
}

private fun inferTableColumnOrdinal(
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

private fun resolveRowMapPathValue(
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

private fun tableCellDisplayText(value: Any?): String {
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

private fun normalizeTableColumnKey(raw: String, fallback: String): String {
    val normalized = raw.trim()
        .replace(Regex("[^A-Za-z0-9_./-]+"), "_")
        .trim('_')
    return if (normalized.isBlank()) fallback else normalized
}

private fun prettifyTableKey(key: String): String {
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

private fun normalizedTableTextLength(value: String): Int {
    return value
        .trim()
        .replace(Regex("\\s+"), " ")
        .length
        .coerceAtMost(96)
}

private fun estimateTableColumnMinWidthsDp(
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

private fun estimateTableMinWidthDp(
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

private fun shouldUseHorizontalTableScroll(
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

private enum class ResponsiveTableCardTemplate {
    COMPARISON,
    SCHEDULE,
    GENERIC
}

private fun normalizeTableHeaderForMatch(raw: String): String {
    return raw
        .trim()
        .lowercase()
        .replace(Regex("[^a-z0-9]+"), " ")
        .replace(Regex("\\s+"), " ")
        .trim()
}

private fun tableHeaderLabel(headers: List<String>, index: Int): String {
    return headers.getOrNull(index)?.trim().orEmpty().ifBlank { "Column ${index + 1}" }
}

private fun shouldPromoteTimelineSecondTitle(firstHeader: String, secondHeader: String): Boolean {
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

private fun shouldPrefixTimelineTitle(label: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(label)
    return normalized == "total" ||
        normalized == "status" ||
        normalized == "current status" ||
        normalized == "core ml" ||
        normalized == "applications" ||
        normalized == "ethics"
}

private fun formatTimelineTitle(label: String, value: String): String {
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

private fun isIconColumnLabel(label: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(label)
    return normalized == "icon" || normalized == "media icon" || normalized == "visual"
}

private fun isImageColumnLabel(label: String): Boolean {
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

private fun isImageAltColumnLabel(label: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(label)
    return normalized == "alt" ||
        normalized == "image alt" ||
        normalized == "photo alt" ||
        normalized == "caption"
}

private fun pickResponsiveTableTemplate(
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

private fun looksLikeTravelItineraryTable(headers: List<String>): Boolean {
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

private fun compactItinerarySectionLabel(label: String): String {
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

private fun splitLeadingItineraryTitle(value: String): Pair<String?, String> {
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
private fun ItinerarySectionBlock(
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

@Composable
private fun TravelItineraryDayCard(
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
            val imageUrl = imageCell?.third.orEmpty()
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
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                iconCell?.third?.takeIf { it.isNotBlank() }?.let { icon ->
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

@Composable
private fun ScheduleDetailBlock(
    label: String,
    value: String,
    modifier: Modifier = Modifier
) {
    val normalizedValue = value.trim()
    if (normalizedValue.isBlank()) return
    val normalizedLabel = normalizeTableHeaderForMatch(label)
    val showLabel = normalizedLabel.isNotBlank() &&
        !normalizedLabel.contains("detail") &&
        !normalizedLabel.contains("description") &&
        !normalizedLabel.contains("note")
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(3.dp)
    ) {
        if (showLabel) {
            Text(
                text = label.trim(),
                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                color = MaterialTheme.colorScheme.primary
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
private fun ResponsiveFieldBlock(
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
private fun flatSpecCardColors() = CardDefaults.cardColors(
    containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
    contentColor = MaterialTheme.colorScheme.onSurface
)

@Composable
private fun flatSpecCardBorder() = BorderStroke(
    width = 1.dp,
    color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.55f)
)

private fun tableAccessibilitySummary(
    headers: List<String>,
    rows: List<List<String>>,
    horizontalScroll: Boolean = false
): String {
    val rowLabel = if (rows.size == 1) "row" else "rows"
    val columnLabel = if (headers.size == 1) "column" else "columns"
    return buildString {
        append("Table with ${rows.size} $rowLabel and ${headers.size} $columnLabel")
        if (horizontalScroll) {
            append(". Scroll horizontally to view all columns")
        }
    }
}

private fun tableRowAccessibilitySummary(
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

private fun marketTickerColumnIndex(headers: List<String>): Int =
    findTableColumnIndex(headers, listOf("ticker", "symbol", "stock", "asset", "holding", "security"))
        ?: 0

private fun marketValueColumnIndex(headers: List<String>, tickerIndex: Int): Int =
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

private fun marketChangeAmountColumnIndex(headers: List<String>, exclude: Set<Int>): Int? =
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

private fun marketChangePercentColumnIndex(headers: List<String>, exclude: Set<Int>): Int? =
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

private fun marketTrendToken(vararg values: String?): Boolean? {
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
private fun marketTrendColor(positive: Boolean?): Color {
    return when (positive) {
        true -> Color(0xFF15803D)
        false -> Color(0xFFDC2626)
        null -> MaterialTheme.colorScheme.primary
    }
}

@Composable
private fun RenderMarketHoldingsTable(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier
) {
    if (rows.isEmpty()) return
    val tickerIndex = marketTickerColumnIndex(headers)
    val valueIndex = marketValueColumnIndex(headers, tickerIndex)
    val amountIndex = marketChangeAmountColumnIndex(headers, setOf(tickerIndex, valueIndex))
    val percentIndex = marketChangePercentColumnIndex(headers, setOf(tickerIndex, valueIndex) + listOfNotNull(amountIndex))
    Card(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        shape = RoundedCornerShape(18.dp),
        colors = flatSpecCardColors(),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 10.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(7.dp)
        ) {
            rows.forEachIndexed { rowIndex, row ->
                MarketHoldingRow(
                    headers = headers,
                    row = row,
                    rowIndex = rowIndex,
                    tickerIndex = tickerIndex,
                    valueIndex = valueIndex,
                    amountIndex = amountIndex,
                    percentIndex = percentIndex
                )
            }
        }
    }
}

@Composable
private fun MarketHoldingRow(
    headers: List<String>,
    row: List<String>,
    rowIndex: Int,
    tickerIndex: Int,
    valueIndex: Int,
    amountIndex: Int?,
    percentIndex: Int?
) {
    val ticker = row.getOrNull(tickerIndex).orEmpty().trim().ifBlank { "Asset ${rowIndex + 1}" }
    val value = row.getOrNull(valueIndex).orEmpty().trim()
    val amount = amountIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty()
    val percent = percentIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty()
    val trend = marketTrendToken(percent, amount)
    val accent = marketTrendColor(trend)
    val valueLabel = tableHeaderLabel(headers, valueIndex)
    val changeLabel = percentIndex?.let { tableHeaderLabel(headers, it) }
        ?: amountIndex?.let { tableHeaderLabel(headers, it) }
        ?: "Change"

    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .semantics(mergeDescendants = true) {
                contentDescription = tableRowAccessibilitySummary(headers, row, rowIndex)
            },
        shape = RoundedCornerShape(14.dp),
        color = MaterialTheme.colorScheme.surfaceContainerHighest.copy(alpha = 0.54f)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 10.dp, vertical = 9.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Box(
                modifier = Modifier
                    .widthIn(min = 54.dp, max = 66.dp)
                    .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                    .background(accent.copy(alpha = 0.12f))
                    .padding(horizontal = 10.dp, vertical = 8.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = ticker,
                    style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold),
                    color = accent,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(1.dp)
            ) {
                Text(
                    text = valueLabel,
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Text(
                    text = parseBoldMarkdown(value),
                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                    color = MaterialTheme.colorScheme.onSurface,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
            Column(
                horizontalAlignment = Alignment.End,
                verticalArrangement = Arrangement.spacedBy(3.dp),
                modifier = Modifier.widthIn(min = 88.dp, max = 118.dp)
            ) {
                val primaryChange = percent.ifBlank { amount }
                if (primaryChange.isNotBlank()) {
                    Surface(
                        shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                        color = accent.copy(alpha = 0.12f)
                    ) {
                        Text(
                            text = parseBoldMarkdown(primaryChange),
                            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                            color = accent,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                        )
                    }
                }
                if (amount.isNotBlank() && amount != primaryChange) {
                    Text(
                        text = parseBoldMarkdown(amount),
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                        color = accent,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                } else {
                    Text(
                        text = changeLabel,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
        }
    }
}

private fun processStateColumnIndex(headers: List<String>): Int =
    findTableColumnIndex(headers, listOf("ui state", "screen state", "state", "status", "step", "stage"))
        ?: 0

private fun processVisualColumnIndex(headers: List<String>, stateIndex: Int): Int? =
    findTableColumnIndex(
        headers,
        listOf("visual", "screen", "view", "interface"),
        exclude = setOf(stateIndex)
    )

private fun processFeedbackColumnIndex(headers: List<String>, stateIndex: Int): Int? =
    findTableColumnIndex(
        headers,
        listOf("feedback", "message", "text", "result", "copy"),
        exclude = setOf(stateIndex)
    )

private fun processStateAccent(title: String): Color {
    val token = title.lowercase()
    return when {
        token.contains("success") || token.contains("valid") || token.contains("complete") -> Color(0xFF5EEAD4)
        token.contains("invalid") || token.contains("error") || token.contains("fail") -> Color(0xFFF87171)
        token.contains("scan") || token.contains("ready") || token.contains("start") -> Color(0xFF60A5FA)
        else -> Color(0xFFA78BFA)
    }
}

private fun incidentComponentColumnIndex(headers: List<String>): Int =
    findTableColumnIndex(headers, listOf("component", "service", "system", "module", "dependency"))
        ?: 0

private fun incidentStatusColumnIndex(headers: List<String>, componentIndex: Int): Int =
    findTableColumnIndex(
        headers,
        listOf("current status", "status", "health", "state"),
        exclude = setOf(componentIndex)
    ) ?: headers.indices.firstOrNull { it != componentIndex } ?: 0

private fun incidentNotesColumnIndex(headers: List<String>, excluded: Set<Int>): Int? =
    findTableColumnIndex(
        headers,
        listOf("notes", "impact", "details", "description", "message"),
        exclude = excluded
    )

private fun incidentSeverityRank(status: String): Int {
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

private fun incidentStatusAccent(status: String): Color = when (incidentSeverityRank(status)) {
    3 -> Color(0xFFF97373)
    2 -> Color(0xFFFBBF24)
    1 -> Color(0xFF60A5FA)
    else -> Color(0xFF34D399)
}

private fun incidentStatusLabel(status: String): String {
    val cleaned = status.replace('_', ' ').replace('-', ' ').trim()
    if (cleaned.isBlank()) return "Unknown"
    return cleaned
        .lowercase()
        .split(Regex("\\s+"))
        .joinToString(" ") { word -> word.replaceFirstChar { it.uppercase() } }
}

private fun incidentCompactStatusLabel(status: String): String = when (incidentSeverityRank(status)) {
    3 -> "Major"
    2 -> "Degraded"
    1 -> "Maintenance"
    else -> "OK"
}

private fun incidentOverallLabel(rows: List<List<String>>, statusIndex: Int): String {
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
private fun IncidentSeverityPill(
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
private fun IncidentStatusHero(
    rows: List<List<String>>,
    componentIndex: Int,
    statusIndex: Int,
    modifier: Modifier = Modifier
) {
    val overall = incidentOverallLabel(rows, statusIndex)
    val affectedCount = rows.count { row ->
        incidentSeverityRank(row.getOrNull(statusIndex).orEmpty()) > 0
    }.coerceAtLeast(rows.size)
    val worstRawStatus = rows
        .map { row -> row.getOrNull(statusIndex).orEmpty() }
        .maxByOrNull(::incidentSeverityRank)
        .orEmpty()
    val accent = incidentStatusAccent(worstRawStatus)
    val topServices = rows
        .sortedByDescending { row -> incidentSeverityRank(row.getOrNull(statusIndex).orEmpty()) }
        .take(3)

    Surface(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(24.dp),
        color = Color.White.copy(alpha = 0.10f),
        tonalElevation = 0.dp,
        shadowElevation = 0.dp
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Text(
                        text = "System status",
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.SemiBold),
                        color = Color.White.copy(alpha = 0.72f)
                    )
                    Text(
                        text = parseBoldMarkdown(overall),
                        style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Black),
                        color = Color.White,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis
                    )
                }
                Box(
                    modifier = Modifier
                        .size(54.dp)
                        .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                        .background(accent.copy(alpha = 0.18f)),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "!",
                        style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Black),
                        color = accent
                    )
                }
            }
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IncidentSeverityPill(status = worstRawStatus.ifBlank { overall })
                Surface(
                    shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                    color = Color.White.copy(alpha = 0.12f)
                ) {
                    Text(
                        text = "$affectedCount affected",
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                        color = Color.White.copy(alpha = 0.82f),
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp)
                    )
                }
            }
            Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                topServices.forEach { row ->
                    val component = row.getOrNull(componentIndex).orEmpty().ifBlank { "Service" }
                    val status = row.getOrNull(statusIndex).orEmpty()
                    val rowAccent = incidentStatusAccent(status)
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(9.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(
                            modifier = Modifier
                                .size(10.dp)
                                .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                                .background(rowAccent)
                        )
                        Text(
                            text = parseBoldMarkdown(component),
                            style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold),
                            color = Color.White.copy(alpha = 0.86f),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.weight(1f)
                        )
                        Text(
                            text = incidentCompactStatusLabel(status),
                            style = MaterialTheme.typography.labelSmall,
                            color = Color.White.copy(alpha = 0.64f),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun IncidentServiceRowCard(
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
private fun RenderIncidentStatusDashboard(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    landscape: Boolean
) {
    if (rows.isEmpty()) return
    val componentIndex = incidentComponentColumnIndex(headers)
    val statusIndex = incidentStatusColumnIndex(headers, componentIndex)
    val notesIndex = incidentNotesColumnIndex(headers, setOf(componentIndex, statusIndex))
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        shape = RoundedCornerShape(28.dp),
        color = Color.Transparent,
        tonalElevation = 0.dp,
        shadowElevation = 0.dp
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.linearGradient(
                        listOf(
                            Color(0xFF130D1B),
                            Color(0xFF2B1724),
                            Color(0xFF0F2433)
                        )
                    )
                )
                .padding(14.dp)
        ) {
            if (landscape) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(14.dp),
                    verticalAlignment = Alignment.Top
                ) {
                    IncidentStatusHero(
                        rows = rows,
                        componentIndex = componentIndex,
                        statusIndex = statusIndex,
                        modifier = Modifier.widthIn(min = 220.dp, max = 280.dp)
                    )
                    Column(
                        modifier = Modifier
                            .weight(1f)
                            .heightIn(max = 430.dp)
                            .verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(9.dp)
                    ) {
                        rows.forEachIndexed { rowIndex, row ->
                            IncidentServiceRowCard(
                                headers = headers,
                                row = row,
                                rowIndex = rowIndex,
                                componentIndex = componentIndex,
                                statusIndex = statusIndex,
                                notesIndex = notesIndex
                            )
                        }
                    }
                }
            } else {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    IncidentStatusHero(
                        rows = rows,
                        componentIndex = componentIndex,
                        statusIndex = statusIndex
                    )
                    rows.forEachIndexed { rowIndex, row ->
                        IncidentServiceRowCard(
                            headers = headers,
                            row = row,
                            rowIndex = rowIndex,
                            componentIndex = componentIndex,
                            statusIndex = statusIndex,
                            notesIndex = notesIndex
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ProcessScannerHero(
    rows: List<List<String>>,
    stateIndex: Int,
    modifier: Modifier = Modifier
) {
    val states = rows.mapNotNull { row -> row.getOrNull(stateIndex)?.trim()?.takeIf(String::isNotBlank) }
    val primaryState = states.firstOrNull().orEmpty()
    Surface(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(24.dp),
        color = Color.White.copy(alpha = 0.10f),
        tonalElevation = 0.dp,
        shadowElevation = 0.dp
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Process flow",
                    style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold),
                    color = Color.White.copy(alpha = 0.92f)
                )
                Surface(
                    shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                    color = Color.White.copy(alpha = 0.12f)
                ) {
                    Text(
                        text = "${rows.size} states",
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                        color = Color.White.copy(alpha = 0.82f),
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp)
                    )
                }
            }
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(172.dp)
                    .clip(RoundedCornerShape(22.dp))
                    .background(
                        Brush.linearGradient(
                            listOf(
                                Color(0xFF020617),
                                Color(0xFF0F172A),
                                Color(0xFF12324B)
                            )
                        )
                    ),
                contentAlignment = Alignment.Center
            ) {
                Box(
                    modifier = Modifier
                        .size(112.dp)
                        .clip(RoundedCornerShape(18.dp))
                        .background(Color.White.copy(alpha = 0.08f)),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(7.dp)
                    ) {
                        Text(
                            text = "QR",
                            style = MaterialTheme.typography.headlineMedium.copy(fontWeight = FontWeight.Black),
                            color = Color.White
                        )
                        Box(
                            modifier = Modifier
                                .width(72.dp)
                                .height(3.dp)
                                .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                                .background(Color(0xFF5EEAD4))
                        )
                        Text(
                            text = primaryState.ifBlank { "Scan state" },
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                            color = Color.White.copy(alpha = 0.72f),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.padding(horizontal = 8.dp)
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ProcessStateStepCard(
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

@Composable
private fun RenderProcessStateTable(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    landscape: Boolean
) {
    if (rows.isEmpty()) return
    val stateIndex = processStateColumnIndex(headers)
    val visualIndex = processVisualColumnIndex(headers, stateIndex)
    val feedbackIndex = processFeedbackColumnIndex(headers, stateIndex)
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        shape = RoundedCornerShape(28.dp),
        color = Color.Transparent,
        tonalElevation = 0.dp,
        shadowElevation = 0.dp
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.linearGradient(
                        listOf(
                            Color(0xFF07111F),
                            Color(0xFF102A43),
                            Color(0xFF062E2E)
                        )
                    )
                )
                .padding(14.dp)
        ) {
            if (landscape) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(14.dp),
                    verticalAlignment = Alignment.Top
                ) {
                    ProcessScannerHero(
                        rows = rows,
                        stateIndex = stateIndex,
                        modifier = Modifier.widthIn(min = 220.dp, max = 280.dp)
                    )
                    Column(
                        modifier = Modifier
                            .weight(1f)
                            .heightIn(max = 430.dp)
                            .verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(9.dp)
                    ) {
                        rows.forEachIndexed { rowIndex, row ->
                            ProcessStateStepCard(
                                headers = headers,
                                row = row,
                                rowIndex = rowIndex,
                                stateIndex = stateIndex,
                                visualIndex = visualIndex,
                                feedbackIndex = feedbackIndex
                            )
                        }
                    }
                }
            } else {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    ProcessScannerHero(rows = rows, stateIndex = stateIndex)
                    rows.forEachIndexed { rowIndex, row ->
                        ProcessStateStepCard(
                            headers = headers,
                            row = row,
                            rowIndex = rowIndex,
                            stateIndex = stateIndex,
                            visualIndex = visualIndex,
                            feedbackIndex = feedbackIndex
                        )
                    }
                }
            }
        }
    }
}

private fun findTableColumnIndex(
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

private fun isLikelyHttpUrl(value: String): Boolean {
    val normalized = value.trim().lowercase()
    return normalized.startsWith("https://") || normalized.startsWith("http://")
}

private fun bookingActionLabelIndex(headers: List<String>): Int? =
    headers.indices.firstOrNull { index ->
        val token = normalizeTableHeaderForMatch(headers[index])
        token in setOf("actionlabel", "action label", "buttonlabel", "button label", "ctalabel", "cta label") ||
            (token.contains("action") && token.contains("label")) ||
            (token.contains("button") && token.contains("label")) ||
            (token.contains("cta") && token.contains("label"))
    }

private fun bookingImageIndex(headers: List<String>): Int? =
    headers.indices.firstOrNull { index ->
        val token = normalizeTableHeaderForMatch(headers[index])
        token in setOf("image", "imageurl", "image url", "photo", "photourl", "photo url", "thumbnail", "media") ||
            (token.contains("image") && token.contains("url")) ||
            (token.contains("photo") && token.contains("url"))
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

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun renderBookingRowsIfPossible(
    headers: List<String>,
    rows: List<List<String>>,
    onOpenUrl: (String) -> Unit
): Boolean {
    if (rows.isEmpty()) return false
    val titleIndex = findTableColumnIndex(
        headers = headers,
        keywords = listOf("hotel", "property", "provider", "option", "listing", "vendor", "airline", "name", "route", "plan")
    ) ?: 0
    val priceIndex = findTableColumnIndex(
        headers = headers,
        keywords = listOf("price", "cost", "fare", "rate", "night", "budget")
    )
    val linkIndex = findTableColumnIndex(
        headers = headers,
        keywords = listOf("url", "link", "book", "booking", "reserve", "website")
    )
    val actionLabelIndex = bookingActionLabelIndex(headers)
    val imageIndex = bookingImageIndex(headers)
    val secondaryIndex = findTableColumnIndex(
        headers = headers,
        keywords = listOf("duration", "time", "date", "location", "room", "type", "class", "stops", "status"),
        exclude = setOf(titleIndex, priceIndex ?: -1)
    )
    val hasBookingSignal = priceIndex != null || linkIndex != null || headers.any(::isBookingEntityHeaderLabel)
    if (!hasBookingSignal) return false

    val shownRows = rows.filter { row ->
        row.any { value -> value.trim().isNotBlank() }
    }
    if (shownRows.isEmpty()) return false

    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        shownRows.forEach { row ->
            val title = row.getOrNull(titleIndex).orEmpty().trim().ifBlank { "Option" }
            val price = priceIndex?.let { index -> row.getOrNull(index).orEmpty().trim() }.orEmpty()
            val secondary = secondaryIndex?.let { index -> row.getOrNull(index).orEmpty().trim() }.orEmpty()
            val actionUrl = linkIndex?.let { index -> row.getOrNull(index).orEmpty().trim() }
                ?.takeIf(::isLikelyHttpUrl)
            val actionLabel = bookingRowActionLabel(headers, row) ?: "Open Option"
            val imageUrl = bookingRowImageUrl(headers, row).orEmpty()
            val chips = buildList {
                headers.forEachIndexed { index, header ->
                    if (index in setOf(titleIndex, priceIndex, secondaryIndex, linkIndex, actionLabelIndex, imageIndex)) {
                        return@forEachIndexed
                    }
                    val value = row.getOrNull(index).orEmpty().trim()
                    if (value.isBlank()) return@forEachIndexed
                    add(header.ifBlank { "Detail" } to value)
                }
            }.take(4)

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics {
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
                    if (imageUrl.isNotBlank()) {
                        RenderImage(
                            props = mapOf(
                                "url" to imageUrl,
                                "fit" to "cover",
                                "aspectRatio" to 1.65f,
                                "alt" to title
                            ),
                            onOpenUrl = onOpenUrl
                        )
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        verticalAlignment = Alignment.Top
                    ) {
                        Column(
                            modifier = Modifier.weight(1f),
                            verticalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Text(
                                text = parseBoldMarkdown(title),
                                style = MaterialTheme.typography.titleSmall,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            if (secondary.isNotBlank()) {
                                Text(
                                    text = parseBoldMarkdown(secondary),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                        if (price.isNotBlank()) {
                            Surface(
                                shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                                color = MaterialTheme.colorScheme.primaryContainer
                            ) {
                                Text(
                                    text = parseBoldMarkdown(price),
                                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp)
                                )
                            }
                        }
                    }
                    if (chips.isNotEmpty()) {
                        FlowRow(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            chips.forEach { (label, value) ->
                                Surface(
                                    shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                                    color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.62f)
                                ) {
                                    Text(
                                        text = parseBoldMarkdown("${label.trim()}: ${value.trim()}"),
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 5.dp)
                                    )
                                }
                            }
                        }
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
    return true
}

@Composable
private fun ResponsiveComparisonRowCard(
    headers: List<String>,
    row: List<String>
) {
    val title = row.getOrNull(0).orEmpty().trim()
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
private fun ResponsiveComparisonColumnCards(
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
private fun ComparisonEntityMediaTile(media: TableEntityMedia) {
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
private fun FeatureMatrixField(
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
private fun ResponsiveScheduleRowCard(
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

private fun looksLikeStudyPlanTable(headers: List<String>): Boolean {
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

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun StudyPlanWeekCard(
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

@Composable
private fun RenderTimelineTableCards(
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
private fun ResponsiveGenericRowCard(
    headers: List<String>,
    row: List<String>
) {
    val cellCount = maxOf(headers.size, row.size)
    val cells = (0 until cellCount).map { index ->
        tableHeaderLabel(headers, index) to row.getOrNull(index).orEmpty().trim()
    }.filter { (_, value) -> value.isNotBlank() }
    if (cells.isEmpty()) return

    val title = cells.first().second.takeIf { value ->
        cells.size > 1 && value.length in 3..42
    }
    val bodyCells = if (title != null) cells.drop(1) else cells

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
            if (!title.isNullOrBlank()) {
                Text(
                    text = parseBoldMarkdown(title),
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
            bodyCells.forEach { (label, value) ->
                ResponsiveFieldBlock(
                    label = label,
                    value = value,
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }
    }
}

@Composable
private fun RenderResponsiveTableRows(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    spacing: Dp = 8.dp
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
    val template = pickResponsiveTableTemplate(headers, rows)
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

private fun selectAdaptiveTablePresentation(
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
    if (table.shape == FlatTableShape.PLAYLIST) {
        return AdaptiveTablePresentation.PLAYLIST_ROWS
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

internal fun isFormulaVariablesTable(table: FlatDirectTableModel): Boolean =
    table.domain == "formula" && isFormulaVariableHeaderSet(table.columns.map { it.label })

internal fun isCalculationBreakdownTable(table: FlatDirectTableModel): Boolean {
    if (table.domain != "formula") return false
    if (!isCalculationBreakdownHeaderSet(table.columns.map { it.label })) return false
    return table.rows.any { row ->
        row.any { value ->
            value.contains('$') ||
                value.contains('€') ||
                value.contains('£') ||
                value.contains('₹') ||
                looksLikeNumericTableValue(value)
        }
    }
}

@Composable
private fun RenderFormulaVariablesTable(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier
) {
    if (rows.isEmpty()) return
    val variableIndex = headers.indexOfFirst { normalizeTableHeaderForMatch(it) in setOf("variable", "symbol", "term", "parameter", "input") }
        .takeIf { it >= 0 } ?: 0
    val descriptionIndex = headers.indexOfFirst {
        normalizeTableHeaderForMatch(it).let { token ->
            token.contains("description") || token.contains("meaning") || token.contains("definition")
        }
    }
    val valueIndex = headers.indexOfFirst {
        normalizeTableHeaderForMatch(it).let { token ->
            token in setOf("value", "amount", "input value", "given")
        }
    }
    Card(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = "Formula variables table with ${rows.size} variables"
            },
        shape = RoundedCornerShape(18.dp),
        colors = flatSpecCardColors(),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.62f))
                    .padding(horizontal = 10.dp, vertical = 7.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text(
                    text = headers.getOrNull(variableIndex).orEmpty().ifBlank { "Variable" },
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.width(58.dp)
                )
                Text(
                    text = headers.getOrNull(descriptionIndex).orEmpty().ifBlank { "Meaning" },
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f)
                )
                if (valueIndex >= 0) {
                    Text(
                        text = headers.getOrNull(valueIndex).orEmpty().ifBlank { "Value" },
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.End,
                        modifier = Modifier.widthIn(min = 72.dp)
                    )
                }
            }
            rows.forEachIndexed { index, row ->
                val variable = row.getOrNull(variableIndex).orEmpty().trim().ifBlank { "-" }
                val description = row.getOrNull(descriptionIndex).orEmpty().trim()
                val value = row.getOrNull(valueIndex).orEmpty().trim()
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 10.dp, vertical = 8.dp)
                        .semantics(mergeDescendants = true) {
                            contentDescription = listOf(variable, description, value)
                                .filter(String::isNotBlank)
                                .joinToString(": ")
                        },
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .width(58.dp)
                            .heightIn(min = 36.dp)
                            .clip(RoundedCornerShape(11.dp))
                            .background(MaterialTheme.colorScheme.primaryContainer),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = variable,
                            style = MaterialTheme.typography.titleMedium.copy(
                                fontWeight = FontWeight.Bold,
                                fontFamily = FontFamily.Monospace
                            ),
                            color = MaterialTheme.colorScheme.onPrimaryContainer,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                    Text(
                        text = parseBoldMarkdown(description),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                        modifier = Modifier.weight(1f)
                    )
                    if (valueIndex >= 0) {
                        Text(
                            text = value,
                            style = MaterialTheme.typography.labelLarge.copy(
                                fontWeight = FontWeight.SemiBold,
                                fontFamily = FontFamily.Monospace
                            ),
                            color = MaterialTheme.colorScheme.primary,
                            textAlign = TextAlign.End,
                            modifier = Modifier.widthIn(min = 72.dp)
                        )
                    }
                }
                if (index < rows.lastIndex) {
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.55f))
                }
            }
        }
    }
}

@Composable
private fun RenderCalculationBreakdownTable(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier
) {
    if (rows.isEmpty()) return
    val labelIndex = headers.indexOfFirst {
        normalizeTableHeaderForMatch(it) in setOf("component", "item", "metric", "field", "label", "cost component")
    }.takeIf { it >= 0 } ?: 0
    val amountIndex = headers.indexOfFirst {
        normalizeTableHeaderForMatch(it).let { token ->
            token.contains("amount") || token.contains("payment") || token == "value" || token == "cost" || token == "total"
        }
    }.takeIf { it >= 0 } ?: rows.firstOrNull()?.indices?.firstOrNull { it != labelIndex } ?: 1
    val labelHeader = headers.getOrNull(labelIndex).orEmpty().ifBlank { "Item" }
    val amountHeader = headers.getOrNull(amountIndex).orEmpty().ifBlank { "Value" }
    val inputStyleTable = normalizeTableHeaderForMatch(labelHeader) in setOf("metric", "input", "field") &&
        normalizeTableHeaderForMatch(amountHeader) == "value"
    val labelWeight = if (inputStyleTable) 0.66f else 0.54f
    val amountWeight = 1f - labelWeight
    Card(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = "Calculation breakdown table with ${rows.size} rows"
            },
        shape = RoundedCornerShape(18.dp),
        colors = flatSpecCardColors(),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.62f))
                    .padding(horizontal = 10.dp, vertical = 7.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    text = labelHeader,
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(labelWeight)
                )
                Text(
                    text = amountHeader,
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.End,
                    modifier = Modifier.weight(amountWeight)
                )
            }
            rows.forEachIndexed { index, row ->
                val label = row.getOrNull(labelIndex).orEmpty().trim()
                val amount = row.getOrNull(amountIndex).orEmpty().trim()
                val highlight = label.contains("monthly", ignoreCase = true) ||
                    label.contains("lifetime", ignoreCase = true) ||
                    label.contains("total", ignoreCase = true)
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 10.dp, vertical = 8.dp)
                        .semantics(mergeDescendants = true) {
                            contentDescription = "$label: $amount"
                        },
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        text = parseBoldMarkdown(label),
                        style = MaterialTheme.typography.bodyMedium.copy(
                            fontWeight = if (highlight) FontWeight.SemiBold else FontWeight.Normal
                        ),
                        color = MaterialTheme.colorScheme.onSurface,
                        modifier = Modifier.weight(labelWeight)
                    )
                    Text(
                        text = amount,
                        style = MaterialTheme.typography.bodyMedium.copy(
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace
                        ),
                        color = if (highlight) {
                            MaterialTheme.colorScheme.primary
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        },
                        textAlign = TextAlign.End,
                        maxLines = 1,
                        softWrap = false,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(amountWeight)
                    )
                }
                if (index < rows.lastIndex) {
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.6f))
                }
            }
        }
    }
}

@Composable
private fun RenderKeyValueTablePanel(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier
) {
    if (rows.isEmpty()) return
    Card(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        colors = flatSpecCardColors(),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp)) {
            rows.forEachIndexed { index, row ->
                val label = row.getOrNull(0).orEmpty().trim().ifBlank { tableHeaderLabel(headers, 0) }
                val value = row.getOrNull(1).orEmpty().trim()
                if (label.isBlank() && value.isBlank()) return@forEachIndexed
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 7.dp)
                        .semantics(mergeDescendants = true) {
                            contentDescription = "$label: $value"
                        },
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment = Alignment.Top
                ) {
                    Text(
                        text = parseBoldMarkdown(label),
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.weight(0.42f)
                    )
                    Text(
                        text = parseBoldMarkdown(value),
                        style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold),
                        color = MaterialTheme.colorScheme.onSurface,
                        modifier = Modifier.weight(0.58f)
                    )
                }
                if (index < rows.lastIndex) {
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.6f))
                }
            }
        }
    }
}

@Composable
private fun RenderFeatureMatrixEntityCards(
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

private fun playlistNumberColumnIndex(headers: List<String>): Int? {
    return headers.indexOfFirst { header ->
        val raw = header.trim().lowercase()
        val token = normalizeTableHeaderForMatch(header)
        raw == "#" || token in setOf("no", "number", "track number", "track no", "tracknumber")
    }.takeIf { it >= 0 }
}

private fun playlistTitleColumnIndex(headers: List<String>): Int? {
    return headers.indexOfFirst { header ->
        val token = normalizeTableHeaderForMatch(header)
        token == "track" ||
            token == "song" ||
            token == "title" ||
            token.contains("track title") ||
            token.contains("song title")
    }.takeIf { it >= 0 }
}

private fun playlistArtistColumnIndex(headers: List<String>): Int? {
    return headers.indexOfFirst { header ->
        val token = normalizeTableHeaderForMatch(header)
        token == "artist" ||
            token.contains("artist") ||
            token.contains("performer") ||
            token.contains("band")
    }.takeIf { it >= 0 }
}

private fun playlistChipColumnIndexes(
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

private fun splitPlaylistTrack(raw: String): Pair<String?, String> {
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

private fun playlistStringListProp(value: Any?): List<String> {
    return when (value) {
        is List<*> -> value.mapNotNull { it?.toString()?.trim()?.takeIf(String::isNotBlank) }
        is String -> value
            .split(',', '|')
            .mapNotNull { it.trim().takeIf(String::isNotBlank) }
        else -> emptyList()
    }
}

private fun buildPlaylistTrackRows(
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

private fun climateColumnIndex(headers: List<String>, vararg keywords: String): Int? {
    return headers.indexOfFirst { header ->
        val token = normalizeTableHeaderForMatch(header)
        keywords.any { keyword -> token.contains(keyword) }
    }.takeIf { it >= 0 }
}

private fun compactClimateMetricLabel(header: String): String {
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

private fun firstNumberFromText(value: String?): Double? {
    return Regex("""-?\d+(?:\.\d+)?""")
        .find(value.orEmpty())
        ?.value
        ?.toDoubleOrNull()
}

private fun inferClimateCondition(row: ClimateComparisonRow): String {
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

private fun buildClimateComparisonRows(
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

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun RenderClimateComparisonCards(
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

private fun parseChartNumber(value: String): Double? {
    val match = Regex("""-?\d[\d,]*(?:\.\d+)?""").find(value) ?: return null
    return match.value.replace(",", "").toDoubleOrNull()
}

private fun chartColumnIndex(
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

private fun formatChartNumber(value: Double, currencyPrefix: String?): String {
    val rounded = value.roundToInt()
    val formatted = "%,d".format(rounded)
    return if (currencyPrefix.isNullOrBlank()) formatted else "$currencyPrefix$formatted"
}

private fun inferChartCurrency(points: List<ChartPoint>): String? {
    val value = points.firstOrNull { it.displayValue.trim().startsWith("$") } ?: return null
    return value.displayValue.trim().takeWhile { !it.isDigit() && it != '-' }.takeIf { it.isNotBlank() }
}

internal fun extractPercentageMatrixChartModel(
    columns: List<FlatDirectTableColumn>,
    rows: List<List<String>>,
    domain: String
): MultiSeriesChartModel? {
    if (columns.size < 3 || rows.size !in 2..12) return null
    if (domain in setOf("weather", "flight", "booking", "playlist", "schedule", "status")) return null

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

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun RenderChart(
    props: Map<String, Any?>,
    state: Map<String, Any?>,
    modifier: Modifier = Modifier
) {
    val chartType = props["chartType"]?.toString()?.trim()?.lowercase().orEmpty().ifBlank { "bar" }
    if (chartType !in setOf("bar", "bar_chart", "column")) return
    val points = extractChartPoints(props, state)
    if (points.isEmpty()) return

    val title = props["title"]?.toString()?.trim().orEmpty()
    val subtitle = props["subtitle"]?.toString()?.trim().orEmpty()
    val yLabel = props["yLabel"]?.toString()?.trim().orEmpty()
    val maxValue = points.maxOf { it.value }.takeIf { it > 0.0 } ?: 1.0
    val currencyPrefix = inferChartCurrency(points)
    val total = points.sumOf { it.value }
    val peak = points.maxByOrNull { it.value }

    Card(
        modifier = modifier
            .fillMaxWidth()
            .semantics(mergeDescendants = true) {
                contentDescription = buildString {
                    if (title.isNotBlank()) append(title).append(". ")
                    append("Bar chart with ${points.size} values. ")
                    points.forEach { point ->
                        append(point.label).append(": ").append(point.displayValue).append(". ")
                    }
                }
            },
        shape = RoundedCornerShape(22.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            if (title.isNotBlank() || subtitle.isNotBlank()) {
                Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                    if (title.isNotBlank()) {
                        Text(
                            text = parseBoldMarkdown(title),
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                    if (subtitle.isNotBlank()) {
                        Text(
                            text = parseBoldMarkdown(subtitle),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }

            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                points.forEachIndexed { index, point ->
                    val fraction = (point.value / maxValue).toFloat().coerceIn(0.04f, 1f)
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = parseBoldMarkdown(point.label),
                            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                            color = MaterialTheme.colorScheme.onSurface,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.width(76.dp)
                        )
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .height(30.dp)
                                .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                                .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.52f))
                        ) {
                            Box(
                                modifier = Modifier
                                    .fillMaxHeight()
                                    .fillMaxWidth(fraction)
                                    .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                                    .background(
                                        Brush.horizontalGradient(
                                            colors = listOf(
                                                MaterialTheme.colorScheme.primary.copy(alpha = 0.86f),
                                                MaterialTheme.colorScheme.tertiary.copy(alpha = 0.78f)
                                            )
                                        )
                                    )
                            )
                        }
                        Text(
                            text = parseBoldMarkdown(point.displayValue),
                            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                            color = MaterialTheme.colorScheme.onSurface,
                            textAlign = TextAlign.End,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.width(82.dp)
                        )
                    }
                    if (index < points.lastIndex) {
                        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.36f))
                    }
                }
            }

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(7.dp)
            ) {
                peak?.let { point ->
                    ChartSummaryChip("Peak: ${point.label} ${point.displayValue}")
                }
                ChartSummaryChip("Total: ${formatChartNumber(total, currencyPrefix)}")
                if (yLabel.isNotBlank()) {
                    ChartSummaryChip(yLabel)
                }
            }
        }
    }
}

@Composable
private fun ChartSummaryChip(text: String) {
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

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun RenderPercentageMatrixChart(
    model: MultiSeriesChartModel,
    modifier: Modifier = Modifier,
    landscape: Boolean = false
) {
    val palette = listOf(
        MaterialTheme.colorScheme.primary,
        MaterialTheme.colorScheme.tertiary,
        MaterialTheme.colorScheme.secondary,
        MaterialTheme.colorScheme.error.copy(alpha = 0.82f),
        MaterialTheme.colorScheme.primary.copy(alpha = 0.58f),
        MaterialTheme.colorScheme.tertiary.copy(alpha = 0.58f)
    )
    val seriesLabels = model.rows
        .flatMap { row -> row.segments.map { segment -> segment.label } }
        .distinct()
    val colorBySeries = seriesLabels.mapIndexed { index, label ->
        label to palette[index % palette.size]
    }.toMap()
    val title = if (model.percentBased) {
        "Preference distribution"
    } else {
        "Data distribution"
    }
    val subtitle = "Grouped by ${model.categoryLabel}"

    Card(
        modifier = modifier
            .fillMaxWidth()
            .semantics(mergeDescendants = true) {
                contentDescription = buildString {
                    append(title).append(". ")
                    model.rows.forEach { row ->
                        append(row.label).append(": ")
                        append(row.segments.joinToString(", ") { segment -> "${segment.label} ${segment.displayValue}" })
                        append(". ")
                    }
                }
            },
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.22f),
                            MaterialTheme.colorScheme.surfaceContainerLow
                        )
                    )
                )
                .padding(horizontal = 14.dp, vertical = 14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Surface(
                    shape = RoundedCornerShape(16.dp),
                    color = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f)
                ) {
                    Text(
                        text = "Graph",
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                    )
                }
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = title,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Text(
                        text = subtitle,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(7.dp)
            ) {
                seriesLabels.forEach { label ->
                    ChartLegendChip(
                        label = label,
                        color = colorBySeries[label] ?: MaterialTheme.colorScheme.primary
                    )
                }
            }

            Column(verticalArrangement = Arrangement.spacedBy(if (landscape) 8.dp else 11.dp)) {
                model.rows.forEachIndexed { index, row ->
                    PercentageMatrixChartRow(
                        row = row,
                        colorBySeries = colorBySeries,
                        compact = !landscape
                    )
                    if (index < model.rows.lastIndex) {
                        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.22f))
                    }
                }
            }
        }
    }
}

@Composable
private fun ChartLegendChip(
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
private fun PercentageMatrixChartRow(
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

private fun isPreferredClimateRow(row: ClimateComparisonRow): Boolean {
    val text = "${row.place} ${row.verdict.orEmpty()} ${row.metrics.joinToString(" ") { it.second }}".lowercase()
    return text.contains("recommended") ||
        text.contains("pick") ||
        text.contains("best") ||
        text.contains("more sun") ||
        text.contains("warmer")
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ClimateComparisonCard(
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

@Composable
private fun PlaylistCoverArt(
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
private fun PlaylistMetaTags(tags: List<String>) {
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
private fun PlaylistHeroBlock(
    title: String,
    subtitle: String,
    tags: List<String>,
    trackCount: Int,
    landscape: Boolean,
    modifier: Modifier = Modifier
) {
    if (landscape) {
        Column(
            modifier = modifier,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            PlaylistCoverArt(
                trackCount = trackCount,
                compact = false,
                modifier = Modifier.fillMaxWidth().aspectRatio(1f)
            )
            Text(
                text = parseBoldMarkdown(title),
                style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
                color = Color.White,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                text = parseBoldMarkdown(subtitle),
                style = MaterialTheme.typography.bodyMedium,
                color = Color.White.copy(alpha = 0.74f)
            )
            PlaylistMetaTags(tags)
        }
    } else {
        Row(
            modifier = modifier,
            horizontalArrangement = Arrangement.spacedBy(14.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            PlaylistCoverArt(
                trackCount = trackCount,
                compact = true,
                modifier = Modifier.size(92.dp)
            )
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Text(
                    text = parseBoldMarkdown(title),
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                    color = Color.White,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
                Text(
                    text = parseBoldMarkdown(subtitle),
                    style = MaterialTheme.typography.labelMedium,
                    color = Color.White.copy(alpha = 0.72f),
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
                PlaylistMetaTags(tags)
            }
        }
    }
}

@Composable
private fun PlaylistTrackRowView(
    headers: List<String>,
    track: PlaylistTrackRow,
    rowIndex: Int,
    dense: Boolean,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(Color.White.copy(alpha = if (rowIndex == 0) 0.13f else 0.07f))
            .padding(horizontal = if (dense) 10.dp else 12.dp, vertical = if (dense) 7.dp else 9.dp)
            .semantics(mergeDescendants = true) {
                contentDescription = tableRowAccessibilitySummary(headers, track.sourceRow, rowIndex)
            },
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(if (dense) 30.dp else 34.dp)
                .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                .background(Color.White.copy(alpha = 0.13f)),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = track.number,
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                color = Color.White.copy(alpha = 0.9f),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
        Column(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(2.dp)
        ) {
            Text(
                text = parseBoldMarkdown(track.title),
                style = (if (dense) MaterialTheme.typography.bodyMedium else MaterialTheme.typography.titleSmall)
                    .copy(fontWeight = FontWeight.SemiBold),
                color = Color.White,
                maxLines = if (dense) 1 else 2,
                overflow = TextOverflow.Ellipsis
            )
            if (!track.artist.isNullOrBlank()) {
                Text(
                    text = parseBoldMarkdown(track.artist),
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.White.copy(alpha = 0.68f),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
        if (track.chips.isNotEmpty()) {
            Text(
                text = track.chips.first(),
                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                color = Color.White.copy(alpha = 0.64f),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.widthIn(max = 92.dp)
            )
        }
    }
}

@Composable
private fun PlaylistTrackList(
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
private fun RenderPlaylistTableRows(
    headers: List<String>,
    rows: List<List<String>>,
    props: Map<String, Any?>,
    modifier: Modifier = Modifier,
    landscape: Boolean
) {
    if (rows.isEmpty()) return
    val tracks = buildPlaylistTrackRows(headers, rows)
    val trackCount = tracks.size
    val title = props["title"]?.toString()?.trim()?.takeIf { it.isNotBlank() } ?: "Playlist"
    val subtitle = props["subtitle"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
        ?: "$trackCount ${if (trackCount == 1) "track" else "tracks"}"
    val tags = playlistStringListProp(props["mood"]) + playlistStringListProp(props["genre"])
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        shape = RoundedCornerShape(26.dp),
        color = Color.Transparent,
        tonalElevation = 0.dp,
        shadowElevation = 0.dp
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.linearGradient(
                        colors = listOf(
                            Color(0xFF070712),
                            Color(0xFF172044),
                            Color(0xFF082F35)
                        )
                    )
                )
                .padding(14.dp)
        ) {
            if (landscape) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(18.dp),
                    verticalAlignment = Alignment.Top
                ) {
                    PlaylistHeroBlock(
                        title = title,
                        subtitle = subtitle,
                        tags = tags,
                        trackCount = trackCount,
                        landscape = true,
                        modifier = Modifier.widthIn(min = 210.dp, max = 260.dp)
                    )
                    PlaylistTrackList(
                        headers = headers,
                        tracks = tracks,
                        dense = true,
                        modifier = Modifier
                            .weight(1f)
                            .heightIn(max = 430.dp)
                            .verticalScroll(rememberScrollState())
                    )
                }
            } else {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    PlaylistHeroBlock(
                        title = title,
                        subtitle = subtitle,
                        tags = tags,
                        trackCount = trackCount,
                        landscape = false,
                        modifier = Modifier.fillMaxWidth()
                    )
                    PlaylistTrackList(
                        headers = headers,
                        tracks = tracks,
                        dense = false,
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }
        }
    }
}

@Composable
private fun EntityProviderBadge(title: String) {
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
private fun entityProviderAccentColor(title: String): Color {
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

private fun entityProviderInitials(title: String): String {
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

private fun shouldShowEntityProviderBadge(title: String, actionUrl: String?): Boolean {
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

private fun isUrlColumnLabel(header: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(header)
    return normalized == "url" ||
        normalized == "link" ||
        normalized == "href" ||
        normalized.contains("website") ||
        normalized.contains("booking url") ||
        normalized.contains("action url") ||
        normalized.endsWith(" link")
}

private fun isActionLabelColumn(header: String): Boolean {
    val normalized = normalizeTableHeaderForMatch(header)
    return normalized.contains("action label") ||
        normalized.contains("button label") ||
        normalized.contains("cta label") ||
        normalized == "action" ||
        normalized == "cta"
}

private fun entityActionLabel(headers: List<String>, row: List<String>, title: String): String {
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
private fun RenderEntityTableCards(
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
                isUrlColumnLabel(headers[index]) && isLikelyHttpUrl(row.getOrNull(index).orEmpty().trim())
            } ?: row.indexOfFirst { value -> isLikelyHttpUrl(value.trim()) }.takeIf { it >= 0 }
            val actionUrl = actionUrlIndex?.let { row.getOrNull(it).orEmpty().trim() }
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

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun RenderMetricTableCards(
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

private fun flightCarrierColumnIndex(headers: List<String>): Int {
    val explicit = headers.indexOfFirst { header ->
        val normalized = normalizeTableHeaderForMatch(header)
        normalized.contains("carrier") || normalized.contains("airline")
    }
    return explicit.takeIf { it >= 0 } ?: 0
}

private fun flightLegColumnIndexes(headers: List<String>): List<Int> {
    return headers.indices.filter { index ->
        val normalized = normalizeTableHeaderForMatch(headers[index])
        normalized.contains("leg") ||
            Regex("""\b[a-z]{3}\s*[/-]\s*[a-z]{3}\b""").containsMatchIn(normalized)
    }
}

private fun flightLegBadge(header: String, index: Int): String {
    val prefix = header.substringBefore(":").trim()
    return prefix.takeIf { it.isNotBlank() && it.length <= 12 } ?: "Leg ${index + 1}"
}

private fun flightLegRoute(header: String): String {
    val route = header.substringAfter(":", missingDelimiterValue = "").trim()
    return route.takeIf { it.isNotBlank() } ?: header.trim()
}

@Composable
private fun RankedFlightAirlineBadge(airline: String, rank: String, best: Boolean) {
    val accent = rankedFlightAccentColor(airline)
    val code = NativeFlightSemantics.airlineBadgeCode(airline).ifBlank { rank.removePrefix("#") }
    Box(
        modifier = Modifier
            .size(42.dp)
            .clip(RoundedCornerShape(14.dp))
            .background(
                Brush.linearGradient(
                    colors = listOf(
                        accent,
                        accent.copy(alpha = if (best) 0.72f else 0.58f)
                    )
                )
            )
            .border(
                width = GenUiTokens.BorderSm,
                color = Color.White.copy(alpha = 0.35f),
                shape = RoundedCornerShape(14.dp)
            ),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = code.take(3),
            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Black),
            color = Color.White,
            maxLines = 1,
            overflow = TextOverflow.Clip
        )
    }
}

@Composable
private fun RankedFlightBestChip(text: String, accent: Color) {
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
private fun RankedFlightRouteHint(
    duration: String?,
    stopLabel: String?,
    accent: Color
) {
    if (duration.isNullOrBlank() && stopLabel.isNullOrBlank()) return
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(MaterialTheme.colorScheme.surfaceContainerHighest)
            .padding(horizontal = 10.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(9.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(24.dp)
                .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                .background(accent.copy(alpha = 0.14f)),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Filled.FlightTakeoff,
                contentDescription = null,
                tint = accent,
                modifier = Modifier.size(14.dp)
            )
        }
        HorizontalDivider(
            modifier = Modifier.weight(1f),
            color = accent.copy(alpha = 0.34f)
        )
        duration?.takeIf { it.isNotBlank() }?.let { value ->
            Text(
                text = value,
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
        stopLabel?.takeIf { it.isNotBlank() }?.let { value ->
            Surface(
                shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.86f)
            ) {
                Text(
                    text = value,
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                    color = accent,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
private fun rankedFlightAccentColor(airline: String): Color {
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

@Composable
private fun RenderRankedFlightComparisonCards(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    spacing: Dp = 10.dp
) {
    if (rows.isEmpty()) return
    val rankIndex = rankedFlightColumnIndex(headers, listOf("rank", "order", "score"))
    val airlineIndex = rankedFlightColumnIndex(headers, listOf("airline", "carrier")) ?: 0
    val costIndex = rankedFlightColumnIndex(headers, listOf("cost", "fare", "price", "amount"))
    val durationIndex = rankedFlightColumnIndex(headers, listOf("travel time", "duration", "time"))
    val layoverIndex = rankedFlightColumnIndex(headers, listOf("layover", "stop", "connection"))
    val reasonIndex = rankedFlightColumnIndex(headers, listOf("justification", "reason", "why", "notes", "detail"))
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
            val airline = row.getOrNull(airlineIndex).orEmpty().trim().ifBlank { "Flight option ${rowIndex + 1}" }
            val cost = row.getOrNull(costIndex ?: -1).orEmpty().trim()
            val duration = row.getOrNull(durationIndex ?: -1).orEmpty().trim()
            val layover = row.getOrNull(layoverIndex ?: -1).orEmpty().trim()
            val reason = row.getOrNull(reasonIndex ?: -1).orEmpty().trim()
            val best = rowIndex == 0 || rank == "#1"
            val accent = rankedFlightAccentColor(airline)
            val normalizedDuration = NativeFlightSemantics.normalizeDurationLabel(duration) ?: duration
            val normalizedStop = NativeFlightSemantics.canonicalizeStopLabel(layover) ?: layover
            val (fareValue, fareMeta) = NativeFlightSemantics.splitFareDisplay(cost)
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics(mergeDescendants = true) {
                        contentDescription = tableRowAccessibilitySummary(headers, row, rowIndex)
                    },
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surface
                ),
                elevation = CardDefaults.cardElevation(defaultElevation = if (best) 2.dp else 0.dp),
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
                        RankedFlightAirlineBadge(airline = airline, rank = rank, best = best)
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

                    RankedFlightRouteHint(
                        duration = normalizedDuration.takeIf { it.isNotBlank() },
                        stopLabel = normalizedStop.takeIf { it.isNotBlank() },
                        accent = accent
                    )

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
                }
            }
        }
    }
}

@Composable
private fun FlightLegTimelineRow(
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
private fun RenderFlightItineraryTableCards(
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

@Composable
private fun RenderAdaptiveTableGrid(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    horizontalScrollEnabled: Boolean,
    stickyFirstColumn: Boolean,
    numericColumns: Set<Int>
) {
    if (headers.isEmpty() || rows.isEmpty()) return
    val columnMinWidthsDp = estimateTableColumnMinWidthsDp(
        headers = headers,
        rows = rows,
        baseMinDp = if (horizontalScrollEnabled) 120 else 96
    )
    val minTableWidth = estimateTableMinWidthDp(
        headers = headers,
        rows = rows,
        baseMinDp = if (horizontalScrollEnabled) 120 else 96
    ).dp
    val tableColor = MaterialTheme.colorScheme.surfaceContainerLow
    val scrollState = rememberScrollState()
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        if (horizontalScrollEnabled) {
            HorizontalTableScrollHint()
        }
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
            Box(modifier = Modifier.fillMaxWidth()) {
                if (stickyFirstColumn && headers.size > 1) {
                    RenderStickyFirstColumnTable(
                        headers = headers,
                        rows = rows,
                        columnMinWidthsDp = columnMinWidthsDp,
                        scrollState = scrollState,
                        numericColumns = numericColumns
                    )
                } else {
                    val contentModifier = if (horizontalScrollEnabled) {
                        Modifier
                            .widthIn(min = minTableWidth)
                            .horizontalScroll(scrollState)
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
                            numericColumns = numericColumns,
                            weighted = !horizontalScrollEnabled,
                            isHeader = true
                        )
                        rows.forEachIndexed { index, row ->
                            RenderTableGridRow(
                                headers = headers,
                                row = row,
                                rowIndex = index,
                                columnMinWidthsDp = columnMinWidthsDp,
                                numericColumns = numericColumns,
                                weighted = !horizontalScrollEnabled,
                                isHeader = false
                            )
                        }
                    }
                }
                if (horizontalScrollEnabled) {
                    Box(
                        modifier = Modifier
                            .matchParentSize()
                            .background(
                                Brush.horizontalGradient(
                                    colors = listOf(
                                        Color.Transparent,
                                        Color.Transparent,
                                        tableColor.copy(alpha = 0.92f)
                                    )
                                )
                            )
                            .semantics {}
                    )
                }
            }
        }
    }
}

@Composable
private fun HorizontalTableScrollHint() {
    Text(
        text = "Swipe horizontally to view all columns",
        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
        color = MaterialTheme.colorScheme.primary,
        modifier = Modifier.padding(horizontal = 8.dp)
    )
}

@Composable
private fun RenderStickyFirstColumnTable(
    headers: List<String>,
    rows: List<List<String>>,
    columnMinWidthsDp: List<Int>,
    scrollState: androidx.compose.foundation.ScrollState,
    numericColumns: Set<Int>
) {
    val firstWidth = (columnMinWidthsDp.firstOrNull() ?: 128).coerceIn(112, 184).dp
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
private fun RenderStickyTableRow(
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
            modifier = Modifier.width(firstWidth)
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
                    modifier = Modifier.widthIn(min = (trailingWidths.getOrNull(offset) ?: 120).dp)
                )
            }
        }
    }
}

@Composable
private fun RenderTableGridRow(
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
                Modifier.widthIn(min = (columnMinWidthsDp.getOrNull(columnIndex) ?: 120).dp)
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
private fun TableGridCell(
    text: String,
    isHeader: Boolean,
    isNumeric: Boolean,
    modifier: Modifier = Modifier
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
        maxLines = if (isHeader) 2 else 3,
        overflow = TextOverflow.Ellipsis,
        modifier = modifier
            .padding(horizontal = 10.dp, vertical = if (isHeader) 9.dp else 8.dp)
            .then(if (isHeader) Modifier.semantics { heading() } else Modifier)
    )
}

@Composable
private fun tableGridRowColor(isHeader: Boolean, rowIndex: Int): Color {
    return when {
        isHeader -> MaterialTheme.colorScheme.onSurface.copy(alpha = 0.055f)
        rowIndex >= 0 && rowIndex % 2 == 1 -> MaterialTheme.colorScheme.onSurface.copy(alpha = 0.025f)
        else -> Color.Transparent
    }
}

@Composable
private fun RenderDirectTable(
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
    val cardsRequested =
        table.renderMode == FlatTableRenderMode.WEATHER_CARDS ||
            table.renderMode == FlatTableRenderMode.FLIGHT_CARDS ||
            table.renderMode == FlatTableRenderMode.BOOKING_CARDS ||
            table.renderMode == FlatTableRenderMode.PLAYLIST_CARDS

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

    if (compactPortrait && table.renderMode == FlatTableRenderMode.WEATHER_CARDS) {
        val weatherRows = NativeWeatherSemantics.buildWeatherRows(headers, table.rows)
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
    }
    if (table.renderMode == FlatTableRenderMode.FLIGHT_CARDS && looksLikeRankedFlightComparisonTable(headers)) {
        RenderRankedFlightComparisonCards(
            headers = headers,
            rows = table.rows,
            modifier = applyStackModifier(modifier, props, "vertical"),
            spacing = stackGap(props).takeIf { it > 0.dp } ?: 10.dp
        )
        return
    }
    if (compactPortrait && table.renderMode == FlatTableRenderMode.FLIGHT_CARDS && shouldUseNativeFlightCards(headers)) {
        val flightRows = NativeFlightSemantics.buildFlightRows(headers, table.rows)
        if (!flightRows.isNullOrEmpty()) {
            NativeFlightUiRenderer.RenderFlightRows(flightRows)
            return
        }
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

    val tableModifier = applyStackModifier(modifier, props, "vertical")
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
            modifier = tableModifier
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

private fun collectResolvedTableRows(
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
private fun RenderResponsiveTableRowCard(
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

@Composable
private fun RenderList(
    children: List<String>,
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
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        RenderChildren(
            children = children,
            elements = elements,
            state = state,
            repeatScope = repeatScope,
            repeatedChildScopes = repeatedChildScopes,
            onOpenUrl = onOpenUrl,
            onSetState = onSetState,
            onAction = onAction,
            activePath = activePath
        )
    }
}

@Composable
private fun RenderCard(
    elementId: String,
    props: Map<String, Any?>,
    children: List<String>,
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
    val allChildren = children.ifEmpty {
        props["child"]?.toString()?.let { listOf(it) } ?: emptyList()
    }
    val computedFunctions = LocalFlatSpecComputedFunctions.current
    extractFlatSourceSection(
        elementId = elementId,
        props = props,
        children = allChildren,
        elements = elements,
        state = state,
        repeatScope = repeatScope,
        computedFunctions = computedFunctions,
        allowChildSourceCue = true
    )?.let { sourceSection ->
        RenderFlatSourceSection(
            section = sourceSection,
            onOpenUrl = onOpenUrl,
            modifier = modifier,
            wrapInCard = true,
            showTitle = true
        )
        return
    }
    extractEmailPreviewPropsFromCard(allChildren, elements, state, repeatScope)?.let { emailProps ->
        RenderEmailPreview(emailProps, modifier)
        return
    }
    val hasExplicitCardPadding = props.containsKey("contentPadding") ||
        props.containsKey("contentPaddingHorizontal") ||
        props.containsKey("contentPaddingVertical") ||
        props.containsKey("padding") ||
        props.containsKey("paddingHorizontal") ||
        props.containsKey("paddingVertical")
    val defaultContentPadding = if (!hasExplicitCardPadding &&
        allChildren.size == 1 &&
        isPaddedContainerElement(elements[allChildren.first()])
    ) {
        0.dp
    } else {
        10.dp
    }
    val contentPaddingAll = asFlatSpacingDp(props["contentPadding"]) ?: asFlatSpacingDp(props["padding"])
    val contentPaddingHorizontal = asFlatSpacingDp(props["contentPaddingHorizontal"])
        ?: asFlatSpacingDp(props["paddingHorizontal"])
        ?: contentPaddingAll
        ?: defaultContentPadding
    val contentPaddingVertical = asFlatSpacingDp(props["contentPaddingVertical"])
        ?: asFlatSpacingDp(props["paddingVertical"])
        ?: contentPaddingAll
        ?: defaultContentPadding
    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
            .accessibilitySemantics(
                props = props,
                mergeDescendants = false
            ),
        colors = flatSpecCardColors(),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        shape = RoundedCornerShape(16.dp),
        border = flatSpecCardBorder()
    ) {
        CompositionLocalProvider(LocalFlatSpecTextHorizontalPadding provides 0.dp) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(
                        horizontal = contentPaddingHorizontal,
                        vertical = contentPaddingVertical
                    )
            ) {
                RenderChildren(
                    children = allChildren,
                    elements = elements,
                    state = state,
                    repeatScope = repeatScope,
                    repeatedChildScopes = repeatedChildScopes,
                    onOpenUrl = onOpenUrl,
                    onSetState = onSetState,
                    onAction = onAction,
                    activePath = activePath
                )
            }
        }
    }
}

private fun collectTextValuesForEmail(
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

private fun extractEmailPreviewPropsFromCard(
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

private fun emailFirstString(props: Map<String, Any?>, vararg keys: String): String {
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

private fun emailFirstNonBlank(map: Map<String, Any?>, vararg keys: String): String {
    keys.forEach { key ->
        val value = map[key]?.toString()?.trim().orEmpty()
        if (value.isNotBlank()) return value
    }
    return ""
}

private fun emailStringList(value: Any?): List<String> {
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

private fun emailMetadataItems(props: Map<String, Any?>): List<Pair<String, String>> {
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

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun RenderEmailPreview(props: Map<String, Any?>, modifier: Modifier = Modifier) {
    val clipboard = LocalClipboardManager.current
    val title = emailFirstString(props, "title", "heading").ifBlank { "Email Draft" }
    val subtitle = emailFirstString(props, "subtitle", "preheader", "summary")
    val subject = emailFirstString(props, "subject", "emailSubject")
    val body = emailStringList(props["body"] ?: props["paragraphs"] ?: props["emailBody"])
    val signature = emailStringList(props["signature"] ?: props["signoff"])
    val metadata = emailMetadataItems(props)
    val copyBody = (body + signature).joinToString("\n\n")
    val label = accessibilityLabel(
        props,
        listOfNotNull("Email preview", subject, metadata.firstOrNull()?.second)
            .filter { it.isNotBlank() }
            .joinToString(". ")
    )

    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
            .accessibilitySemantics(
                props = props,
                fallbackLabel = label,
                mergeDescendants = false
            ),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        shape = RoundedCornerShape(24.dp),
        border = flatSpecCardBorder()
    ) {
        Column(modifier = Modifier.fillMaxWidth()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        Brush.horizontalGradient(
                            listOf(
                                MaterialTheme.colorScheme.primaryContainer,
                                MaterialTheme.colorScheme.tertiaryContainer
                            )
                        )
                    )
                    .padding(horizontal = 18.dp, vertical = 16.dp)
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = title,
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                            color = MaterialTheme.colorScheme.onPrimaryContainer,
                            modifier = Modifier.semantics { heading() }
                        )
                        Surface(
                            shape = RoundedCornerShape(999.dp),
                            color = MaterialTheme.colorScheme.surface.copy(alpha = 0.72f)
                        ) {
                            Text(
                                text = "Draft",
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.onSurface,
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                            )
                        }
                    }
                    if (subtitle.isNotBlank()) {
                        Text(
                            text = subtitle,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.82f)
                        )
                    }
                }
            }

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 18.dp, vertical = 16.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                if (metadata.isNotEmpty()) {
                    FlowRow(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        metadata.take(5).forEach { (metaLabel, value) ->
                            Surface(
                                shape = RoundedCornerShape(12.dp),
                                color = MaterialTheme.colorScheme.surfaceContainerHighest
                            ) {
                                Text(
                                    text = "$metaLabel: $value",
                                    style = MaterialTheme.typography.labelMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                                )
                            }
                        }
                    }
                }

                if (subject.isNotBlank()) {
                    Surface(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(16.dp),
                        color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.72f)
                    ) {
                        Column(modifier = Modifier.padding(14.dp)) {
                            Text(
                                text = "Subject",
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.76f)
                            )
                            Spacer(Modifier.height(4.dp))
                            Text(
                                text = subject,
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                                color = MaterialTheme.colorScheme.onSecondaryContainer
                            )
                        }
                    }
                }

                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(18.dp),
                    color = MaterialTheme.colorScheme.surfaceContainerLow,
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
                ) {
                    Column(
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 14.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        body.forEach { paragraph ->
                            Text(
                                text = paragraph,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                        }
                        if (signature.isNotEmpty()) {
                            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.6f))
                            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                signature.forEachIndexed { index, line ->
                                    Text(
                                        text = line,
                                        style = if (index == 0) {
                                            MaterialTheme.typography.bodyMedium
                                        } else {
                                            MaterialTheme.typography.bodyMedium.copy(
                                                fontWeight = if (index == 1) FontWeight.SemiBold else FontWeight.Normal
                                            )
                                        },
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                }
                            }
                        }
                    }
                }

                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    if (subject.isNotBlank()) {
                        OutlinedButton(
                            onClick = { clipboard.setText(AnnotatedString(subject)) },
                            modifier = Modifier.semantics {
                                contentDescription = "Copy email subject"
                                role = Role.Button
                            }
                        ) {
                            Text("Copy Subject")
                        }
                    }
                    if (copyBody.isNotBlank()) {
                        Button(
                            onClick = { clipboard.setText(AnnotatedString(copyBody)) },
                            modifier = Modifier.semantics {
                                contentDescription = "Copy email body"
                                role = Role.Button
                            }
                        ) {
                            Text("Copy Email")
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun RenderText(props: Map<String, Any?>, modifier: Modifier = Modifier) {
    val rawText = (
        props["text"]
            ?: props["title"]
            ?: props["label"]
            ?: props["content"]
            ?: props["value"]
        )
        ?.toString()
        .orEmpty()
    if (rawText.isBlank()) return
    parseFencedCodeBlock(rawText)?.let { codeBlock ->
        RenderCodeBlock(codeBlock = codeBlock, modifier = modifier)
        return
    }
    inferCodeBlockFromPlainText(rawText, props)?.let { codeBlock ->
        RenderCodeBlock(codeBlock = codeBlock, modifier = modifier)
        return
    }
    if (looksLikeFormulaText(rawText)) {
        RenderFormula(
            props = mapOf(
                "latex" to rawText,
                "title" to props["formulaTitle"]
            ),
            modifier = modifier
        )
        return
    }
    val markdown = parseSupportedMarkdownText(rawText)
    val rawVariant = (
        props["variant"]?.toString()
            ?: props["typography"]?.toString()
        )
        ?.lowercase()
        .orEmpty()
    val impliedHeadingVariant = when (markdown.headingLevel) {
        1 -> "h1"
        2 -> "h2"
        3 -> "h3"
        else -> ""
    }
    val normalizedVariantSource = if (rawVariant.isBlank()) impliedHeadingVariant else rawVariant
    val variant = when {
        normalizedVariantSource == "h1" || normalizedVariantSource.contains("headline-large") -> "h1"
        normalizedVariantSource == "h2" || normalizedVariantSource.contains("headline") || normalizedVariantSource.contains("title-large") -> "h2"
        normalizedVariantSource == "h3" || normalizedVariantSource.contains("title") || normalizedVariantSource.contains("subtitle") || normalizedVariantSource.contains("heading") -> "h3"
        normalizedVariantSource.contains("caption") || normalizedVariantSource.contains("label") || normalizedVariantSource.contains("body-small") -> "caption"
        normalizedVariantSource.contains("chip") -> "chip"
        else -> normalizedVariantSource
    }
    val horizontalPadding = asFlatSpacingDp(props["textPaddingHorizontal"])
        ?: asFlatSpacingDp(props["paddingHorizontal"])
        ?: LocalFlatSpecTextHorizontalPadding.current
    val primaryTextColor = MaterialTheme.colorScheme.onSurface
    when (variant) {
        "h1" -> Text(
            text = markdown.content,
            style = MaterialTheme.typography.headlineMedium.copy(fontWeight = FontWeight.Bold),
            color = primaryTextColor,
            modifier = modifier
                .padding(horizontal = horizontalPadding, vertical = 8.dp)
                .accessibilitySemantics(props = props, isHeading = true)
        )
        "h2" -> Text(
            text = markdown.content,
            style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
            color = primaryTextColor,
            modifier = modifier
                .padding(horizontal = horizontalPadding, vertical = 6.dp)
                .accessibilitySemantics(props = props, isHeading = true)
        )
        "h3" -> Text(
            text = markdown.content,
            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
            color = primaryTextColor,
            modifier = modifier
                .padding(horizontal = horizontalPadding, vertical = 4.dp)
                .accessibilitySemantics(props = props, isHeading = true)
        )
        "caption", "label" -> Text(
            text = markdown.content,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = modifier
                .padding(horizontal = horizontalPadding, vertical = 2.dp)
                .accessibilitySemantics(props = props)
        )
        "chip" -> Surface(
            shape = RoundedCornerShape(16.dp),
            color = MaterialTheme.colorScheme.secondaryContainer,
            modifier = modifier
                .padding(2.dp)
                .accessibilitySemantics(props = props, fallbackLabel = rawText)
        ) {
            Text(
                text = markdown.content,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSecondaryContainer,
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
            )
        }
        else -> Text(
            text = markdown.content,
            style = MaterialTheme.typography.bodyMedium,
            color = primaryTextColor,
            modifier = modifier
                .padding(horizontal = horizontalPadding, vertical = 2.dp)
                .accessibilitySemantics(props = props)
        )
    }
}

private data class FencedCodeBlock(
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
private fun RenderFormula(props: Map<String, Any?>, modifier: Modifier = Modifier) {
    val rawFormula = (
        props["latex"]
            ?: props["formula"]
            ?: props["text"]
            ?: props["value"]
        )
        ?.toString()
        .orEmpty()
    val normalizedFormula = normalizeFormulaText(rawFormula)
    if (normalizedFormula.isBlank()) return
    val title = props["title"]?.toString()?.trim().orEmpty()
    val subtitle = props["subtitle"]?.toString()?.trim().orEmpty()
    val result = props["result"]?.toString()?.trim().orEmpty()
    val displayFormula = plainBracketFractionToLatex(normalizedFormula)
    val segments = parseFormulaSegments(displayFormula)
    Surface(
        shape = RoundedCornerShape(18.dp),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.64f),
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = LocalFlatSpecTextHorizontalPadding.current, vertical = 6.dp)
            .semantics(mergeDescendants = true) {
                contentDescription = "Formula ${readableFormulaText(displayFormula)}"
            }
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            if (title.isNotBlank()) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
            if (subtitle.isNotBlank()) {
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Surface(
                shape = RoundedCornerShape(14.dp),
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.78f),
                modifier = Modifier.fillMaxWidth()
            ) {
                val scrollState = rememberScrollState()
                RenderFormulaExpression(
                    displayFormula = displayFormula,
                    segments = segments,
                    scrollState = scrollState
                )
            }
            if (result.isNotBlank()) {
                Surface(
                    shape = RoundedCornerShape(999.dp),
                    color = MaterialTheme.colorScheme.primaryContainer
                ) {
                    Text(
                        text = result,
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 7.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun RenderFormulaExpression(
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
private fun FormulaFractionView(
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

@Composable
private fun RenderCodeBlock(codeBlock: FencedCodeBlock, modifier: Modifier = Modifier) {
    if (codeBlock.code.isBlank() && codeBlock.title.isNullOrBlank()) return
    val isConsole = codeBlock.isConsole ||
        codeBlock.language.equals("console", ignoreCase = true) ||
        codeBlock.language.equals("terminal", ignoreCase = true) ||
        codeBlock.language.equals("shell", ignoreCase = true) ||
        codeBlock.language.equals("bash", ignoreCase = true)
    val label = codeBlock.title?.trim()?.takeIf { it.isNotBlank() }
        ?: codeBlock.language?.trim()?.takeIf { it.isNotBlank() }
        ?: if (isConsole) "Console" else "Code"
    val containerColor = if (isConsole) Color(0xFF0B1020) else Color(0xFF111827)
    val headerColor = if (isConsole) Color(0xFF67E8F9) else Color(0xFFA7F3D0)
    val bodyColor = Color(0xFFE5E7EB)
    Surface(
        shape = RoundedCornerShape(16.dp),
        color = containerColor,
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = LocalFlatSpecTextHorizontalPadding.current, vertical = 6.dp)
            .semantics(mergeDescendants = true) {
                contentDescription = buildString {
                    append(if (isConsole) "Console log" else "Code block")
                    label.takeIf { it.isNotBlank() }?.let { language ->
                        append(", ")
                        append(language)
                    }
                    codeBlock.code.trim().takeIf { it.isNotBlank() }?.let { code ->
                        append(". ")
                        append(code)
                    }
                }
            }
    ) {
        Column(modifier = Modifier.fillMaxWidth().padding(12.dp)) {
            if (label.isNotBlank()) {
                Text(
                    text = label.uppercase(),
                    style = MaterialTheme.typography.labelSmall,
                    color = headerColor,
                    modifier = Modifier.padding(bottom = 6.dp)
                )
            }
            val codeTextModifier = if (isConsole) {
                Modifier.fillMaxWidth()
            } else {
                Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
            }
            Text(
                text = codeBlock.code.ifBlank { " " },
                style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                color = bodyColor,
                softWrap = isConsole,
                modifier = codeTextModifier
            )
        }
    }
}

private fun codeBlockFromProps(
    props: Map<String, Any?>,
    defaultLanguage: String? = null,
    isConsole: Boolean = false
): FencedCodeBlock {
    val rawContent = listOf("code", "text", "content", "value", "output", "log", "logs")
        .firstNotNullOfOrNull { key -> props[key]?.toString()?.takeIf { it.isNotBlank() } }
        .orEmpty()
    val parsed = parseFencedCodeBlock(rawContent)
    val language = props["language"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
        ?: props["lang"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
        ?: props["syntax"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
        ?: parsed?.language
        ?: defaultLanguage
    val title = props["title"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
        ?: props["label"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
    val console = isConsole ||
        props["kind"]?.toString()?.contains("console", ignoreCase = true) == true ||
        props["role"]?.toString()?.contains("console", ignoreCase = true) == true ||
        language.equals("console", ignoreCase = true) ||
        language.equals("terminal", ignoreCase = true)
    return FencedCodeBlock(
        code = parsed?.code ?: rawContent,
        language = language,
        title = title,
        isConsole = console
    )
}

private data class SupportedMarkdownText(
    val content: AnnotatedString,
    val headingLevel: Int?
)

private fun parseFencedCodeBlock(raw: String): FencedCodeBlock? {
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

private fun inferCodeBlockFromPlainText(raw: String, props: Map<String, Any?>): FencedCodeBlock? {
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

private data class LatexGroup(val value: String, val nextIndex: Int)

private fun readLatexGroup(text: String, start: Int): LatexGroup? {
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

private data class FormulaToken(val value: String, val nextIndex: Int)

private fun readFormulaScriptToken(text: String, start: Int): FormulaToken {
    if (start >= text.length) return FormulaToken("", start)
    if (text[start] == '{') {
        readLatexGroup(text, start)?.let { return FormulaToken(it.value, it.nextIndex) }
    }
    return FormulaToken(text[start].toString(), start + 1)
}

private fun readLatexCommand(text: String, start: Int): FormulaToken {
    var index = start
    while (index < text.length && text[index].isLetter()) index++
    return FormulaToken(text.substring(start, index), index)
}

private fun latexCommandReplacement(command: String): String? = when (command) {
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

private fun readableFormulaText(raw: String): String =
    normalizeFormulaText(raw)
        .replace("\\frac", " fraction ")
        .replace("{", " ")
        .replace("}", " ")
        .replace(Regex("""\s+"""), " ")
        .trim()

private fun parseSupportedMarkdownText(raw: String): SupportedMarkdownText {
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

private val LEADING_LABEL_REGEX = Regex("^(\\s*)([A-Za-z][A-Za-z0-9 ./()&+\\-]{0,40})([:;])(\\s*.*)$")

private fun parseBoldMarkdown(raw: String): AnnotatedString {
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

private fun AnnotatedString.Builder.appendMarkdownBoldSpans(segment: String) {
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

@Composable
private fun RenderImage(
    props: Map<String, Any?>,
    onOpenUrl: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    val resolveAssetUrl = LocalFlatSpecAssetResolver.current
    val rawUrl = resolveMediaUrlCandidate(props, IMAGE_PROP_KEYS)
    val resolvedUrl = resolveAssetUrl(rawUrl)
    val url = resolveCoilMediaModel(resolvedUrl)
    if (url.isBlank()) return
    val fallbackUrl = deriveImageFallbackUrl(resolvedUrl, props)
        ?.let(resolveAssetUrl)
        ?.let(::resolveCoilMediaModel)
        .orEmpty()
    val contentScale = resolveImageScale(props)
    val context = LocalContext.current
    val imageLoader = remember(context) {
        ImageLoader.Builder(context)
            .components { add(SvgDecoder.Factory()) }
            .build()
    }
    val actionUrl = extractMediaUrlToken(props["actionUrl"]).orEmpty()
    val widthDp = asDp(props["width"])
    val heightDp = asDp(props["height"])
    val aspectRatio = parseAspectRatio(props["aspectRatio"]) ?: if (heightDp == null) 16f / 9f else null
    var imageModifier = modifier
    imageModifier = if (widthDp != null) imageModifier.width(widthDp) else imageModifier.fillMaxWidth()
    if (heightDp != null) {
        imageModifier = imageModifier.height(heightDp)
    }
    if (aspectRatio != null) {
        imageModifier = imageModifier.aspectRatio(aspectRatio)
    }
    imageModifier = imageModifier.clip(RoundedCornerShape(12.dp))
    val clickableModifier = if (actionUrl.isNotBlank()) {
        imageModifier.clickable(
            role = Role.Button,
            onClickLabel = accessibilityString(props, "onClickLabel", "actionLabel") ?: "Open image"
        ) { onOpenUrl(actionUrl) }
    } else {
        imageModifier
    }
    val generatedRestaurantVisual = parseGeneratedRestaurantVisual(resolvedUrl)
        ?: parseGeneratedRestaurantVisual(url)
    if (generatedRestaurantVisual != null) {
        RenderGeneratedRestaurantVisual(
            visual = generatedRestaurantVisual,
            modifier = clickableModifier,
            imageLabel = accessibilityLabel(props, "${generatedRestaurantVisual.title} restaurant image")
        )
        return
    }
    var failed by remember(url) { mutableStateOf(false) }
    var activeUrl by remember(url) { mutableStateOf(url) }
    var fallbackAttempted by remember(url) { mutableStateOf(false) }
    val imageLabel = accessibilityLabel(props, props["alt"]?.toString() ?: "Image")
    val placeholderBg = MaterialTheme.colorScheme.surfaceContainerHighest

    Box(
        modifier = clickableModifier
            .background(placeholderBg)
            .accessibilitySemantics(
                props = props,
                fallbackLabel = imageLabel,
                semanticRole = Role.Button.takeIf { actionUrl.isNotBlank() },
                state = "Image unavailable".takeIf { failed },
                mergeDescendants = true
            ),
        contentAlignment = Alignment.Center
    ) {
        if (failed) {
            Icon(
                imageVector = Icons.Filled.Image,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(24.dp)
            )
        } else {
            AsyncImage(
                model = ImageRequest.Builder(context)
                    .data(activeUrl)
                    .crossfade(true)
                    .allowHardware(false)
                    .build(),
                imageLoader = imageLoader,
                contentDescription = null,
                contentScale = contentScale,
                modifier = Modifier.fillMaxSize(),
                onSuccess = { failed = false },
                onError = {
                    val failedUrl = activeUrl
                    val shouldTryFallback = !fallbackAttempted &&
                        fallbackUrl.isNotBlank() &&
                        !fallbackUrl.equals(failedUrl, ignoreCase = true)
                    if (shouldTryFallback) {
                        fallbackAttempted = true
                        activeUrl = fallbackUrl
                        failed = false
                        Log.w(
                            FLAT_SPEC_RENDERER_TAG,
                            "RenderImage failed for URL '$failedUrl', retrying with fallback '$fallbackUrl'."
                        )
                    } else {
                        failed = true
                        Log.w(
                            FLAT_SPEC_RENDERER_TAG,
                            "RenderImage failed for URL '$failedUrl'."
                        )
                    }
                }
            )
        }
    }
}

private fun parseGeneratedRestaurantVisual(raw: String): GeneratedRestaurantVisual? {
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
private fun RenderGeneratedRestaurantVisual(
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

private fun restaurantVisualPalette(seed: String): Pair<Color, Color> {
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

@Composable
private fun RenderIcon(
    props: Map<String, Any?>,
    modifier: Modifier = Modifier
) {
    val resolveAssetUrl = LocalFlatSpecAssetResolver.current
    val rawUrl = resolveMediaUrlCandidate(props, ICON_PROP_KEYS)
    val url = resolveCoilMediaModel(resolveAssetUrl(rawUrl))
    if (url.isBlank()) return
    val context = LocalContext.current
    val imageLoader = remember(context) {
        ImageLoader.Builder(context)
            .components { add(SvgDecoder.Factory()) }
            .build()
    }
    val iconSize = resolveIconSize(props)
    val basePadding = asFlatSpacingDp(props["padding"]) ?: asFlatSpacingDp(props["iconPadding"]) ?: 4.dp
    val horizontalPadding =
        asFlatSpacingDp(props["paddingHorizontal"]) ?: asFlatSpacingDp(props["iconPaddingHorizontal"]) ?: basePadding
    val verticalPadding =
        asFlatSpacingDp(props["paddingVertical"]) ?: asFlatSpacingDp(props["iconPaddingVertical"]) ?: basePadding
    val iconModifier = modifier
        .padding(horizontal = horizontalPadding, vertical = verticalPadding)
        .size(iconSize)
    var failed by remember(url) { mutableStateOf(false) }
    val iconDescription = if (isAccessibilityDecorative(props)) {
        null
    } else {
        accessibilityLabel(props)
    }
    if (failed) {
        Icon(
            imageVector = Icons.Filled.Image,
            contentDescription = iconDescription?.let { "$it unavailable" },
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = iconModifier
        )
    } else {
        AsyncImage(
            model = ImageRequest.Builder(context)
                .data(url)
                .crossfade(false)
                .allowHardware(false)
                .build(),
            imageLoader = imageLoader,
            contentDescription = iconDescription,
            contentScale = ContentScale.Fit,
            colorFilter = ColorFilter.tint(MaterialTheme.colorScheme.onSurface),
            modifier = iconModifier,
            onSuccess = { failed = false },
            onError = { failed = true }
        )
    }
}

@Composable
private fun RenderButton(
    props: Map<String, Any?>,
    onMap: Map<String, Any?>?,
    repeatScope: RepeatScope?,
    onAction: (Any?, RepeatScope?) -> Int,
    modifier: Modifier = Modifier
) {
    val label = props["label"]?.toString().orEmpty().ifBlank { "Open" }
    val variant = props["variant"]?.toString()?.lowercase().orEmpty()
    val actionCandidate = onMap?.get("press") ?: onMap?.get("click") ?: onMap?.get("tap")
    val onClick: () -> Unit = if (actionCandidate == null) {
        {}
    } else {
        { onAction(actionCandidate, repeatScope) }
    }
    val buttonModifier = modifier
        .padding(horizontal = 16.dp, vertical = 4.dp)
        .let { base ->
            if (label.length > 20) base.fillMaxWidth() else base
        }
        .accessibilitySemantics(
            props = props,
            fallbackLabel = label,
            semanticRole = Role.Button,
            state = "Disabled".takeIf { actionCandidate == null }
        )
    if (variant == "borderless" || variant == "text" || variant == "outlined") {
        OutlinedButton(
            onClick = onClick,
            enabled = actionCandidate != null,
            modifier = buttonModifier
        ) {
            Text(text = label)
        }
    } else {
        Button(
            onClick = onClick,
            enabled = actionCandidate != null,
            modifier = buttonModifier
        ) {
            Text(text = label)
        }
    }
}

@Composable
private fun RenderDivider(modifier: Modifier = Modifier) {
    HorizontalDivider(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
        color = MaterialTheme.colorScheme.outlineVariant
    )
}

@Composable
private fun RenderTabs(
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
    val tabs = props["tabs"] as? List<*> ?: return
    if (tabs.isEmpty()) return

    val activeTabId = props["activeTabId"]?.toString().orEmpty()
    val initialIndex = tabs.indexOfFirst { tab ->
        val map = tab as? Map<*, *> ?: return@indexOfFirst false
        val tabChild = map["child"]?.toString()
            ?: map["content"]?.toString()
            ?: map["id"]?.toString()
            ?: map["element"]?.toString()
        tabChild == activeTabId
    }.takeIf { it >= 0 } ?: 0

    var selectedIndex by remember(tabs, activeTabId) { mutableIntStateOf(initialIndex) }

    Column(modifier = modifier.fillMaxWidth()) {
        ScrollableTabRow(selectedTabIndex = selectedIndex) {
            tabs.forEachIndexed { index, tabAny ->
                val tab = tabAny as? Map<*, *> ?: return@forEachIndexed
                val title = (tab["title"]?.toString() ?: tab["label"]?.toString()).orEmpty().ifBlank { "Tab ${index + 1}" }
                Tab(
                    selected = selectedIndex == index,
                    onClick = { selectedIndex = index },
                    text = { Text(title) }
                )
            }
        }
        Spacer(Modifier.height(8.dp))
        val selectedTab = tabs.getOrNull(selectedIndex) as? Map<*, *>
        val childId = (
            selectedTab?.get("child")?.toString()
                ?: selectedTab?.get("content")?.toString()
                ?: selectedTab?.get("id")?.toString()
                ?: selectedTab?.get("element")?.toString()
            ).orEmpty()
        if (childId.isNotBlank()) {
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
    }
}

@Composable
private fun RenderModal(
    props: Map<String, Any?>,
    children: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    onAction: (Any?, RepeatScope?) -> Int,
    activePath: Set<String>,
    modifier: Modifier = Modifier
) {
    val title = props["title"]?.toString().orEmpty().ifBlank { "Modal" }
    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp, vertical = 4.dp)
            .accessibilitySemantics(
                props = props,
                fallbackLabel = title
            ),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.fillMaxWidth().padding(8.dp)) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier
                    .padding(horizontal = 8.dp, vertical = 4.dp)
                    .semantics { heading() }
            )
            val ids = buildList {
                props["trigger"]?.toString()?.takeIf { it.isNotBlank() }?.let(::add)
                props["content"]?.toString()?.takeIf { it.isNotBlank() }?.let(::add)
                addAll(children)
            }.distinct()
            ids.forEach { childId ->
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
        }
    }
}

private fun bindPathFromValueExpression(value: Any?, repeatScope: RepeatScope?): String? {
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

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun RenderTextField(
    props: Map<String, Any?>,
    onSetState: (String, Any?) -> Unit,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    modifier: Modifier = Modifier
) {
    val computedFunctions = LocalFlatSpecComputedFunctions.current
    val label = props["label"]?.toString().orEmpty().ifBlank { "Input" }
    val bindPath = bindPathFromValueExpression(props["value"], repeatScope) ?: props["statePath"]?.toString()
    val value = FlatExprResolver.resolveString(props["value"], state, repeatScope, computedFunctions)
    var localValue by remember(label) { mutableStateOf(value) }
    val textValue = if (!bindPath.isNullOrBlank()) value else localValue
    val bringIntoViewRequester = remember { BringIntoViewRequester() }
    val coroutineScope = rememberCoroutineScope()

    OutlinedTextField(
        value = textValue,
        onValueChange = { next ->
            if (!bindPath.isNullOrBlank()) {
                onSetState(bindPath, next)
            } else {
                localValue = next
            }
        },
        label = { Text(label) },
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
            .bringIntoViewRequester(bringIntoViewRequester)
            .onFocusChanged { focusState ->
                if (focusState.isFocused) {
                    coroutineScope.launch {
                        bringIntoViewRequester.bringIntoView()
                    }
                }
            }
    )
}

@Composable
private fun RenderCheckBox(
    props: Map<String, Any?>,
    onSetState: (String, Any?) -> Unit,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    modifier: Modifier = Modifier
) {
    val computedFunctions = LocalFlatSpecComputedFunctions.current
    val label = props["label"]?.toString().orEmpty().ifBlank { "Option" }
    val bindPath = bindPathFromValueExpression(props["value"], repeatScope) ?: props["statePath"]?.toString()
    val checked = FlatExprResolver.resolveBoolean(props["value"], state, repeatScope, computedFunctions)
    var localChecked by remember(label) { mutableStateOf(checked) }
    val isChecked = if (!bindPath.isNullOrBlank()) checked else localChecked

    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
            .toggleable(
                value = isChecked,
                role = Role.Checkbox,
                onValueChange = { next ->
                    if (!bindPath.isNullOrBlank()) {
                        onSetState(bindPath, next)
                    } else {
                        localChecked = next
                    }
                }
            )
            .accessibilitySemantics(
                props = props,
                fallbackLabel = label,
                state = if (isChecked) "Checked" else "Not checked",
                mergeDescendants = true
            ),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Checkbox(
            checked = isChecked,
            onCheckedChange = null
        )
        Text(text = label, style = MaterialTheme.typography.bodyMedium)
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun RenderChoicePicker(
    props: Map<String, Any?>,
    onSetState: (String, Any?) -> Unit,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    modifier: Modifier = Modifier
) {
    val computedFunctions = LocalFlatSpecComputedFunctions.current
    val label = props["label"]?.toString().orEmpty().ifBlank { "Choose" }
    val bindPath = bindPathFromValueExpression(props["value"], repeatScope) ?: props["statePath"]?.toString()
    val resolvedValue = FlatExprResolver.resolve(props["value"], state, repeatScope, computedFunctions)
    val selected = when (resolvedValue) {
        is List<*> -> resolvedValue.mapNotNull { it?.toString() }.toSet()
        is String -> setOf(resolvedValue)
        else -> emptySet()
    }
    val options = (props["options"] as? List<*>)?.mapNotNull { optionAny ->
        val optionMap = optionAny as? Map<*, *> ?: return@mapNotNull null
        val optionLabel = optionMap["label"]?.toString().orEmpty()
        val optionValue = optionMap["value"]?.toString().orEmpty()
        if (optionValue.isBlank()) null else optionLabel.ifBlank { optionValue } to optionValue
    }.orEmpty()

    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
    ) {
        Text(text = label, style = MaterialTheme.typography.labelLarge)
        Spacer(Modifier.height(6.dp))
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            options.forEach { (optionLabel, optionValue) ->
                val active = selected.contains(optionValue)
                OutlinedButton(
                    onClick = {
                        val next = if (active) selected - optionValue else selected + optionValue
                        if (!bindPath.isNullOrBlank()) {
                            onSetState(bindPath, next.toList())
                        }
                    },
                    modifier = Modifier.semantics {
                        contentDescription = optionLabel
                        stateDescription = if (active) "Selected" else "Not selected"
                    }
                ) {
                    Text(optionLabel)
                }
            }
        }
    }
}

@Composable
private fun RenderSlider(
    props: Map<String, Any?>,
    onSetState: (String, Any?) -> Unit,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    modifier: Modifier = Modifier
) {
    val computedFunctions = LocalFlatSpecComputedFunctions.current
    val label = props["label"]?.toString().orEmpty()
    val bindPath = bindPathFromValueExpression(props["value"], repeatScope) ?: props["statePath"]?.toString()
    val min = (props["min"] as? Number)?.toFloat() ?: 0f
    val max = (props["max"] as? Number)?.toFloat() ?: 100f
    val resolved = FlatExprResolver.resolve(props["value"], state, repeatScope, computedFunctions)
    val initial = (resolved as? Number)?.toFloat() ?: min
    var localValue by remember(label) { mutableStateOf(initial.coerceIn(min, max)) }
    val current = if (!bindPath.isNullOrBlank()) {
        ((FlatSpecParser.getAtPath(state, bindPath) as? Number)?.toFloat() ?: initial).coerceIn(min, max)
    } else {
        localValue
    }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
    ) {
        if (label.isNotBlank()) {
            Text("$label: ${current.roundToInt()}", style = MaterialTheme.typography.labelLarge)
            Spacer(Modifier.height(4.dp))
        }
        Slider(
            value = current,
            valueRange = min..max,
            modifier = Modifier.semantics {
                contentDescription = label.ifBlank { "Slider" }
                stateDescription = "${current.roundToInt()} of ${max.roundToInt()}"
            },
            onValueChange = { next ->
                if (!bindPath.isNullOrBlank()) {
                    onSetState(bindPath, next)
                } else {
                    localValue = next
                }
            }
        )
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun RenderDateTimeInput(
    props: Map<String, Any?>,
    onSetState: (String, Any?) -> Unit,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    modifier: Modifier = Modifier
) {
    val computedFunctions = LocalFlatSpecComputedFunctions.current
    val label = props["label"]?.toString().orEmpty().ifBlank { "Date/Time" }
    val bindPath = bindPathFromValueExpression(props["value"], repeatScope) ?: props["statePath"]?.toString()
    val resolvedValue = FlatExprResolver.resolveString(props["value"], state, repeatScope, computedFunctions)
    var localValue by remember(label) { mutableStateOf(resolvedValue) }
    val value = if (!bindPath.isNullOrBlank()) resolvedValue else localValue
    val bringIntoViewRequester = remember { BringIntoViewRequester() }
    val coroutineScope = rememberCoroutineScope()

    OutlinedTextField(
        value = value,
        onValueChange = { next ->
            if (!bindPath.isNullOrBlank()) {
                onSetState(bindPath, next)
            } else {
                localValue = next
            }
        },
        label = { Text(label) },
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
            .bringIntoViewRequester(bringIntoViewRequester)
            .onFocusChanged { focusState ->
                if (focusState.isFocused) {
                    coroutineScope.launch {
                        bringIntoViewRequester.bringIntoView()
                    }
                }
            }
    )
}

@Composable
private fun RenderVideo(
    props: Map<String, Any?>,
    onOpenUrl: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    val url = props["url"]?.toString().orEmpty()
    if (url.isBlank()) return
    val label = accessibilityLabel(props, "Video")
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
            .clip(RoundedCornerShape(8.dp))
            .clickable(
                role = Role.Button,
                onClickLabel = accessibilityString(props, "onClickLabel", "actionLabel") ?: "Open video"
            ) { onOpenUrl(url) }
            .accessibilitySemantics(
                props = props,
                fallbackLabel = label,
                semanticRole = Role.Button,
                mergeDescendants = true
            ),
        color = MaterialTheme.colorScheme.surfaceVariant
    ) {
        Text(
            text = "Video",
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.padding(16.dp)
        )
    }
}

@Composable
private fun RenderAudioPlayer(
    props: Map<String, Any?>,
    onOpenUrl: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    val url = props["url"]?.toString().orEmpty()
    if (url.isBlank()) return
    val description = props["description"]?.toString().orEmpty().ifBlank { "Audio" }
    val label = accessibilityLabel(props, description)
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
            .clip(RoundedCornerShape(8.dp))
            .clickable(
                role = Role.Button,
                onClickLabel = accessibilityString(props, "onClickLabel", "actionLabel") ?: "Open audio"
            ) { onOpenUrl(url) }
            .accessibilitySemantics(
                props = props,
                fallbackLabel = label,
                semanticRole = Role.Button,
                mergeDescendants = true
            ),
        color = MaterialTheme.colorScheme.surfaceVariant
    ) {
        Text(
            text = description,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.padding(16.dp)
        )
    }
}
