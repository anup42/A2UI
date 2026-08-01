package com.samsung.genuicraft.renderer.flat.parse

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
import com.samsung.genuicraft.renderer.flat.model.*

// moved from FlatSpecRenderer.kt (FlatSpecParser)
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
                    // Preserve the JSON numeric representation used by
                    // Kotlin equality. `3` is Long while `3.0`/`3e0` is
                    // Double; collapsing both to Long makes renderer
                    // `eq` disagree with Python and with Kotlin itself.
                    val lexical = primitive.toString()
                    if (lexical.contains('.') ||
                        lexical.contains('e', ignoreCase = true)
                    ) {
                        primitive.asDouble
                    } else {
                        primitive.asLong
                    }
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

    /** Removes an object key or list entry; explicit null values remain distinct. */
    fun removeAtPath(stateStore: MutableMap<String, Any?>, path: String): Boolean {
        val tokens = pointerTokens(path)
        if (tokens.isEmpty()) return false
        val topKey = tokens.first()
        if (tokens.size == 1) {
            val existed = stateStore.containsKey(topKey)
            if (existed) stateStore.remove(topKey)
            return existed
        }
        if (!stateStore.containsKey(topKey)) return false
        val result = removeInNode(stateStore[topKey], tokens.drop(1))
        if (result.removed) {
            stateStore[topKey] = result.value
        }
        return result.removed
    }

    private data class RemoveResult(val value: Any?, val removed: Boolean)

    private fun removeInNode(node: Any?, tokens: List<String>): RemoveResult {
        if (tokens.isEmpty()) return RemoveResult(node, false)
        val head = tokens.first()
        val tail = tokens.drop(1)
        val index = head.toIntOrNull()
        if (index != null) {
            val source = node as? List<*> ?: return RemoveResult(node, false)
            if (index !in source.indices) return RemoveResult(node, false)
            val list = source.toMutableList()
            if (tail.isEmpty()) {
                list.removeAt(index)
                return RemoveResult(list, true)
            }
            val child = removeInNode(list[index], tail)
            if (!child.removed) return RemoveResult(node, false)
            list[index] = child.value
            return RemoveResult(list, true)
        }

        val source = node as? Map<*, *> ?: return RemoveResult(node, false)
        val map = source.entries.associate { (key, value) -> key.toString() to value }.toMutableMap()
        if (!map.containsKey(head)) return RemoveResult(node, false)
        if (tail.isEmpty()) {
            map.remove(head)
            return RemoveResult(map, true)
        }
        val child = removeInNode(map[head], tail)
        if (!child.removed) return RemoveResult(node, false)
        map[head] = child.value
        return RemoveResult(map, true)
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
