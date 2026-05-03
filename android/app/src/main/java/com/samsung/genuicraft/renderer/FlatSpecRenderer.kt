package com.samsung.genuicraft.renderer

import android.content.Intent
import android.content.res.Configuration
import android.net.Uri
import android.util.Log
import androidx.compose.foundation.clickable
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
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
import androidx.compose.material.icons.filled.FlightTakeoff
import androidx.compose.material.icons.filled.Image
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
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
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
import com.samsung.genuicraft.renderer.native.intents.flight.NativeFlightSemantics
import com.samsung.genuicraft.renderer.native.intents.flight.NativeFlightUiRenderer
import com.samsung.genuicraft.renderer.native.intents.weather.NativeWeatherSemantics
import com.samsung.genuicraft.renderer.native.intents.weather.NativeWeatherUiRenderer
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

private data class WatchEntry(
    val key: String,
    val statePath: String,
    val actionBinding: Any?
)

internal enum class FlatTableRenderMode {
    TABLE,
    TABLE_HORIZONTAL_SCROLL,
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
private val CARD_FIRST_TABLE_DOMAINS = setOf("weather", "flight", "booking", "schedule", "status", "playlist")
private val SUPPORTED_TABLE_DOMAINS = CARD_FIRST_TABLE_DOMAINS + setOf("generic", "comparison")

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
                state = state
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

    private fun interpolate(template: String, state: Map<String, Any?>): String {
        return Regex("""\$\{([^}]+)\}""").replace(template) { match ->
            val rawPath = match.groupValues.getOrNull(1).orEmpty()
            val path = normalizePointer(rawPath)
            FlatSpecParser.getAtPath(state, path)?.toString().orEmpty()
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
        "card" -> RenderCard(props, children, elements, state, repeatScope, repeatedChildScopes, onOpenUrl, onSetState, onAction, activePath, modifier)
        "table" -> RenderDirectTable(props, state, onOpenUrl, modifier)
        "chart", "barchart", "bar_chart" -> RenderChart(props, state, modifier)
        "text" -> RenderText(props, modifier)
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
    if (repeatedChildScopes == null) {
        children.forEach { childId ->
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
        children.forEach { childId ->
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
    return if (token in PLAYLIST_TABLE_DOMAIN_ALIASES) "playlist" else token
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
            token.contains("track title") ||
            token.contains("song title")
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
            token.contains("tempo")
    }
    return hasTrackTitle && (hasTrackNumber || hasArtist || hasMusicMetadata)
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
            element.children.size >= 2
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
    val shape = detectTableShape(
        headers = headers,
        rows = collectResolvedTableRows(
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
        ),
        domain = domain
    )
    val cardMappingStatus = when (domain) {
        "comparison" -> "pending_runtime_mapping"
        in CARD_FIRST_TABLE_DOMAINS -> "pending_runtime_mapping"
        else -> "not_applicable"
    }
    val comparisonCardsPreferred = domain == "comparison" && shouldPreferComparisonCards(headers, compactScreen)
    val renderMode = when {
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
            "file:///android_asset/${normalized.removePrefix("../")}"
        normalized.startsWith("./assets/") ->
            "file:///android_asset/${normalized.removePrefix("./")}"
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
    val hasDayColumn = first.contains("day") || first.contains("date")
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
    return hasDayColumn && (activityColumns >= 2 || hasDiningColumn)
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
    val day = row.getOrNull(0).orEmpty().trim().ifBlank { "Day" }
    val sections = (1 until maxOf(headers.size, row.size)).mapNotNull { index ->
        val value = row.getOrNull(index).orEmpty().trim()
        if (value.isBlank()) null else tableHeaderLabel(headers, index) to value
    }
    if (sections.isEmpty()) return

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
            sections.forEach { (label, value) ->
                ItinerarySectionBlock(label = label, value = value)
            }
        }
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
    containerColor = MaterialTheme.colorScheme.surfaceContainerLow
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
                elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
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
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
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
                elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
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

    val cellCount = maxOf(headers.size, row.size)
    val cells = (0 until cellCount).map { index ->
        Triple(index, tableHeaderLabel(headers, index), row.getOrNull(index).orEmpty().trim())
    }.filter { (_, _, value) -> value.isNotBlank() }
    if (cells.isEmpty()) return

    val promoteSecond = cells.size > 1 &&
        shouldPromoteTimelineSecondTitle(cells[0].second, cells[1].second)
    val titleCellIndex = if (promoteSecond) 1 else 0
    val badgeCell = if (promoteSecond) cells.firstOrNull() else null
    val titleCell = cells.getOrNull(titleCellIndex) ?: cells.first()
    val titleValue = formatTimelineTitle(titleCell.second, titleCell.third)
    val bodyCells = cells.filterNot { (index, _, _) ->
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
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
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
                FeatureMatrixField(feature = label, value = value)
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
            val actionUrl = row.firstOrNull(::isLikelyHttpUrl)
            val bodyIndexes = headers.indices.filterNot { index ->
                index == primaryIndex || index in inferredHighlightIndexes || row.getOrNull(index).orEmpty().isBlank()
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
                    Text(
                        text = parseBoldMarkdown(title),
                        style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                        color = MaterialTheme.colorScheme.onSurface
                    )
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
                        Button(onClick = { onOpenUrl(actionUrl) }) {
                            Text("Open")
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
    val tableModifier = applyStackModifier(modifier, props, "vertical")
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
        shape = RoundedCornerShape(16.dp)
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
    when (variant) {
        "h1" -> Text(
            text = markdown.content,
            style = MaterialTheme.typography.headlineMedium.copy(fontWeight = FontWeight.Bold),
            modifier = modifier
                .padding(horizontal = horizontalPadding, vertical = 8.dp)
                .accessibilitySemantics(props = props, isHeading = true)
        )
        "h2" -> Text(
            text = markdown.content,
            style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
            modifier = modifier
                .padding(horizontal = horizontalPadding, vertical = 6.dp)
                .accessibilitySemantics(props = props, isHeading = true)
        )
        "h3" -> Text(
            text = markdown.content,
            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
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
            modifier = modifier
                .padding(horizontal = horizontalPadding, vertical = 2.dp)
                .accessibilitySemantics(props = props)
        )
    }
}

private data class FencedCodeBlock(
    val code: String,
    val language: String? = null
)

@Composable
private fun RenderCodeBlock(codeBlock: FencedCodeBlock, modifier: Modifier = Modifier) {
    Surface(
        shape = RoundedCornerShape(10.dp),
        color = MaterialTheme.colorScheme.surfaceVariant,
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = LocalFlatSpecTextHorizontalPadding.current, vertical = 4.dp)
            .semantics(mergeDescendants = true) {
                contentDescription = buildString {
                    append("Code block")
                    codeBlock.language?.trim()?.takeIf { it.isNotBlank() }?.let { language ->
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
            val languageLabel = codeBlock.language?.trim().orEmpty()
            if (languageLabel.isNotBlank()) {
                Text(
                    text = languageLabel.uppercase(),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(bottom = 6.dp)
                )
            }
            Text(
                text = codeBlock.code.ifBlank { " " },
                style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                color = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
            )
        }
    }
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
    return FencedCodeBlock(code = code, language = languageToken)
}

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
