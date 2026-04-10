package com.samsung.genuicraft.renderer

import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.clickable
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.relocation.BringIntoViewRequester
import androidx.compose.foundation.relocation.bringIntoViewRequester
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.HorizontalDivider
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
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil.ImageLoader
import coil.compose.AsyncImage
import coil.decode.SvgDecoder
import coil.request.ImageRequest
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import kotlinx.coroutines.launch
import kotlin.math.roundToInt
import java.util.UUID

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

typealias FlatComputedFunction = (Map<String, Any?>) -> Any?

private val LocalFlatSpecAssetResolver = staticCompositionLocalOf<(String) -> String> { { raw -> raw } }
private val LocalFlatSpecComputedFunctions = staticCompositionLocalOf<Map<String, FlatComputedFunction>> { emptyMap() }
private const val WATCH_ACTION_BUDGET = 32

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
        val type = obj.get("type")?.takeIf { it.isJsonPrimitive }?.asString ?: return null
        val props = mutableMapOf<String, Any?>()
        obj.getAsJsonObject("props")?.entrySet()?.forEach { (key, value) ->
            props[key] = toKotlin(value)
        }

        val children = obj.getAsJsonArray("children")
            ?.mapNotNull { child -> child.takeIf { it.isJsonPrimitive }?.asString }
            ?: emptyList()

        val repeat = obj.getAsJsonObject("repeat")?.let { repeatObj ->
            val statePath = repeatObj.get("statePath")
                ?.takeIf { it.isJsonPrimitive }
                ?.asString
                ?: return@let null
            RepeatConfig(
                statePath = statePath,
                key = repeatObj.get("key")?.takeIf { it.isJsonPrimitive }?.asString
            )
        }

        val visible = obj.get("visible")?.let(::toKotlin)
        val on = obj.getAsJsonObject("on")
            ?.entrySet()
            ?.associate { (key, value) -> key to toKotlin(value) }
        val watch = obj.getAsJsonObject("watch")
            ?.entrySet()
            ?.associate { (key, value) -> key to toKotlin(value) }

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
        return Regex("""\$\{([^}]+)}""").replace(template) { match ->
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

    val repeatedChildScopes: List<RepeatScope>? = element.repeat?.let { repeat ->
        val items = (FlatSpecParser.getAtPath(state, repeat.statePath) as? List<*>).orEmpty()
        items.mapIndexed { index, item ->
            RepeatScope(
                item = item,
                index = index,
                basePath = combinePointerPath(repeat.statePath, index.toString())
            )
        }
    }

    val resolvedProps = element.props.mapValues { (_, value) ->
        FlatExprResolver.resolve(value, state, repeatScope, computedFunctions)
    }

    RenderByType(
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
        "stack" -> RenderStack(props, children, elements, state, repeatScope, repeatedChildScopes, onOpenUrl, onSetState, onAction, activePath, modifier)
        "row", "column" -> return
        "list" -> RenderList(children, elements, state, repeatScope, repeatedChildScopes, onOpenUrl, onSetState, onAction, activePath, modifier)
        "card" -> RenderCard(props, children, elements, state, repeatScope, repeatedChildScopes, onOpenUrl, onSetState, onAction, activePath, modifier)
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

private fun stackDirection(props: Map<String, Any?>): String {
    return props["direction"]?.toString()?.trim()?.lowercase().orEmpty().ifBlank { "vertical" }
}

private fun stackGap(props: Map<String, Any?>): androidx.compose.ui.unit.Dp {
    return when (props["gap"]?.toString()?.trim()?.lowercase()) {
        "none" -> 0.dp
        "sm" -> 4.dp
        "md" -> 8.dp
        "lg" -> 12.dp
        "xl" -> 16.dp
        else -> 8.dp
    }
}

private fun applyStackModifier(
    base: Modifier,
    props: Map<String, Any?>,
    direction: String
): Modifier {
    var out = base
    val marginAll = asDp(props["margin"])
    val marginHorizontal = asDp(props["marginHorizontal"]) ?: marginAll
    val marginVertical = asDp(props["marginVertical"]) ?: marginAll
    if (marginHorizontal != null || marginVertical != null) {
        out = out.padding(
            horizontal = marginHorizontal ?: 0.dp,
            vertical = marginVertical ?: 0.dp
        )
    }

    val paddingAll = asDp(props["padding"])
    val paddingHorizontal = asDp(props["paddingHorizontal"]) ?: paddingAll
    val paddingVertical = asDp(props["paddingVertical"]) ?: paddingAll
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

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun RenderStack(
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
    val direction = stackDirection(props)
    val gap = stackGap(props)
    val align = props["align"]?.toString()?.trim()?.lowercase().orEmpty()
    val justify = props["justify"]?.toString()?.trim()?.lowercase().orEmpty()
    val wrap = props["wrap"]?.toString()?.trim()?.lowercase() == "wrap"
    val stackModifier = applyStackModifier(modifier, props, direction)

    if (direction == "horizontal") {
        val horizontalArrangement: Arrangement.Horizontal = when (justify) {
            "center" -> if (gap > 0.dp) Arrangement.spacedBy(gap, Alignment.CenterHorizontally) else Arrangement.Center
            "end" -> if (gap > 0.dp) Arrangement.spacedBy(gap, Alignment.End) else Arrangement.End
            "between" -> Arrangement.SpaceBetween
            "around" -> Arrangement.SpaceAround
            else -> if (gap > 0.dp) Arrangement.spacedBy(gap) else Arrangement.Start
        }
        if (wrap) {
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
            return
        }
        val verticalAlignment = when (align) {
            "center" -> Alignment.CenterVertically
            "end" -> Alignment.Bottom
            else -> Alignment.Top
        }
        Row(
            modifier = stackModifier,
            horizontalArrangement = horizontalArrangement,
            verticalAlignment = verticalAlignment
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
    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.fillMaxWidth()) {
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

@Composable
private fun RenderText(props: Map<String, Any?>, modifier: Modifier = Modifier) {
    val text = props["text"]?.toString().orEmpty()
    if (text.isBlank()) return
    val variant = props["variant"]?.toString()?.lowercase().orEmpty()
    when (variant) {
        "h1" -> Text(
            text = text,
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            modifier = modifier.padding(horizontal = 16.dp, vertical = 8.dp)
        )
        "h2" -> Text(
            text = text,
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold,
            modifier = modifier.padding(horizontal = 16.dp, vertical = 6.dp)
        )
        "h3" -> Text(
            text = text,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            modifier = modifier.padding(horizontal = 16.dp, vertical = 4.dp)
        )
        "caption", "label" -> Text(
            text = text,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = modifier.padding(horizontal = 16.dp, vertical = 2.dp)
        )
        "chip" -> Surface(
            shape = RoundedCornerShape(16.dp),
            color = MaterialTheme.colorScheme.secondaryContainer,
            modifier = modifier.padding(2.dp)
        ) {
            Text(
                text = text,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSecondaryContainer,
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
            )
        }
        else -> Text(
            text = text,
            style = MaterialTheme.typography.bodyMedium,
            modifier = modifier.padding(horizontal = 16.dp, vertical = 2.dp)
        )
    }
}

@Composable
private fun RenderImage(
    props: Map<String, Any?>,
    onOpenUrl: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    val resolveAssetUrl = LocalFlatSpecAssetResolver.current
    val url = resolveAssetUrl(props["url"]?.toString()?.trim().orEmpty())
    if (url.isBlank()) return
    val fit = props["fit"]?.toString()?.lowercase().orEmpty()
    val isCover = fit == "cover"
    val context = LocalContext.current
    val imageLoader = remember(context) {
        ImageLoader.Builder(context)
            .components { add(SvgDecoder.Factory()) }
            .build()
    }
    val actionUrl = props["actionUrl"]?.toString().orEmpty()
    val imageModifier = modifier
        .fillMaxWidth()
        .aspectRatio(16f / 9f)
        .clip(RoundedCornerShape(topStart = 12.dp, topEnd = 12.dp))
    val clickableModifier = if (actionUrl.isNotBlank()) {
        imageModifier.clickable { onOpenUrl(actionUrl) }
    } else {
        imageModifier
    }

    AsyncImage(
        model = ImageRequest.Builder(context).data(url).crossfade(true).build(),
        imageLoader = imageLoader,
        contentDescription = props["alt"]?.toString()
            ?: (props["accessibility"] as? Map<*, *>)?.get("label")?.toString(),
        contentScale = if (isCover) ContentScale.Crop else ContentScale.Fit,
        modifier = clickableModifier
    )
}

@Composable
private fun RenderIcon(
    props: Map<String, Any?>,
    modifier: Modifier = Modifier
) {
    val resolveAssetUrl = LocalFlatSpecAssetResolver.current
    val url = resolveAssetUrl(props["name"]?.toString()?.trim().orEmpty())
    if (url.isBlank()) return
    val context = LocalContext.current
    val imageLoader = remember(context) {
        ImageLoader.Builder(context)
            .components { add(SvgDecoder.Factory()) }
            .build()
    }
    AsyncImage(
        model = ImageRequest.Builder(context).data(url).crossfade(false).build(),
        imageLoader = imageLoader,
        contentDescription = null,
        contentScale = ContentScale.Fit,
        modifier = modifier.size(24.dp)
    )
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
    val buttonModifier = modifier.padding(horizontal = 16.dp, vertical = 4.dp)
    if (variant == "borderless" || variant == "text" || variant == "outlined") {
        OutlinedButton(onClick = onClick, modifier = buttonModifier) {
            Text(text = label)
        }
    } else {
        Button(onClick = onClick, modifier = buttonModifier) {
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
        (tab as? Map<*, *>)?.get("child")?.toString() == activeTabId
    }.takeIf { it >= 0 } ?: 0

    var selectedIndex by remember(tabs, activeTabId) { mutableIntStateOf(initialIndex) }

    Column(modifier = modifier.fillMaxWidth()) {
        ScrollableTabRow(selectedTabIndex = selectedIndex) {
            tabs.forEachIndexed { index, tabAny ->
                val tab = tabAny as? Map<*, *> ?: return@forEachIndexed
                val title = tab["title"]?.toString().orEmpty().ifBlank { "Tab ${index + 1}" }
                Tab(
                    selected = selectedIndex == index,
                    onClick = { selectedIndex = index },
                    text = { Text(title) }
                )
            }
        }
        Spacer(Modifier.height(8.dp))
        val selectedTab = tabs.getOrNull(selectedIndex) as? Map<*, *>
        val childId = selectedTab?.get("child")?.toString().orEmpty()
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
    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp, vertical = 4.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.fillMaxWidth().padding(8.dp)) {
            Text(
                text = props["title"]?.toString().orEmpty().ifBlank { "Modal" },
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
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
            .padding(horizontal = 16.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Checkbox(
            checked = isChecked,
            onCheckedChange = { next ->
                if (!bindPath.isNullOrBlank()) {
                    onSetState(bindPath, next)
                } else {
                    localChecked = next
                }
            }
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
                    }
                ) {
                    Text(if (active) "$optionLabel*" else optionLabel)
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
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
            .clip(RoundedCornerShape(8.dp))
            .clickable { onOpenUrl(url) },
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
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
            .clip(RoundedCornerShape(8.dp))
            .clickable { onOpenUrl(url) },
        color = MaterialTheme.colorScheme.surfaceVariant
    ) {
        Text(
            text = description,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.padding(16.dp)
        )
    }
}
