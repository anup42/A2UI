package com.samsung.genuicraft.sdk.internal.renderer.flat.expr

import com.samsung.genuicraft.sdk.internal.renderer.*
import com.samsung.genuicraft.sdk.internal.pipeline.LiteralTextCodec

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
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.*

// moved from FlatSpecRenderer.kt (FlatExprResolver)
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
            is String -> LiteralTextCodec.decode(value)?.let(::FlatLiteralText)
                ?: resolveInlineTemplateString(value, state, repeatScope)
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
