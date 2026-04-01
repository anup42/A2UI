package com.samsung.genuicraft.renderer

import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
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
import kotlin.math.roundToInt

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
    val on: Map<String, Any?>? = null
)

data class RepeatConfig(
    val statePath: String,
    val key: String? = null
)

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

        return FlatElement(
            type = type,
            props = props,
            children = children,
            repeat = repeat,
            visible = visible,
            on = on
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

object FlatExprResolver {

    fun resolve(value: Any?, state: Map<String, Any?>, item: Map<String, Any?>?): Any? {
        if (value !is Map<*, *>) return value
        @Suppress("UNCHECKED_CAST")
        val expr = value as Map<String, Any?>
        return when {
            expr.containsKey("\$item") -> item?.get(expr["\$item"]?.toString().orEmpty())
            expr.containsKey("\$state") -> FlatSpecParser.getAtPath(state, expr["\$state"]?.toString().orEmpty())
            expr.containsKey("\$bindState") -> FlatSpecParser.getAtPath(state, expr["\$bindState"]?.toString().orEmpty())
            expr.containsKey("\$cond") -> {
                val condition = evaluateCondition(expr["\$cond"], state, item)
                if (condition) resolve(expr["\$then"], state, item) else resolve(expr["\$else"], state, item)
            }
            expr.containsKey("\$template") -> interpolate(expr["\$template"]?.toString().orEmpty(), state)
            expr.containsKey("literalString") -> expr["literalString"]
            expr.containsKey("literalNumber") -> expr["literalNumber"]
            expr.containsKey("literalBoolean") -> expr["literalBoolean"]
            else -> value
        }
    }

    fun resolveString(value: Any?, state: Map<String, Any?>, item: Map<String, Any?>?): String {
        return resolve(value, state, item)?.toString().orEmpty()
    }

    fun resolveBoolean(value: Any?, state: Map<String, Any?>, item: Map<String, Any?>?): Boolean {
        return when (val resolved = resolve(value, state, item)) {
            is Boolean -> resolved
            is Number -> resolved.toInt() != 0
            is String -> resolved.equals("true", ignoreCase = true)
            else -> false
        }
    }

    fun evaluateVisible(visible: Any?, state: Map<String, Any?>, item: Map<String, Any?>?): Boolean {
        if (visible == null) return true
        return evaluateCondition(visible, state, item)
    }

    private fun evaluateCondition(condition: Any?, state: Map<String, Any?>, item: Map<String, Any?>?): Boolean {
        if (condition == null) return true
        if (condition is Boolean) return condition
        if (condition is List<*>) return condition.all { evaluateCondition(it, state, item) }
        if (condition !is Map<*, *>) return true
        @Suppress("UNCHECKED_CAST")
        val expr = condition as Map<String, Any?>

        if (expr.containsKey("\$and")) {
            val list = expr["\$and"] as? List<*> ?: return true
            return list.all { evaluateCondition(it, state, item) }
        }
        if (expr.containsKey("\$or")) {
            val list = expr["\$or"] as? List<*> ?: return false
            return list.any { evaluateCondition(it, state, item) }
        }

        val rawValue: Any? = when {
            expr.containsKey("\$state") -> FlatSpecParser.getAtPath(state, expr["\$state"]?.toString().orEmpty())
            expr.containsKey("\$item") -> item?.get(expr["\$item"]?.toString().orEmpty())
            expr.containsKey("value") -> resolve(expr["value"], state, item)
            else -> null
        }

        val result = when {
            expr.containsKey("eq") -> rawValue?.toString() == expr["eq"]?.toString()
            expr.containsKey("neq") -> rawValue?.toString() != expr["neq"]?.toString()
            expr.containsKey("gt") -> toDouble(rawValue) > toDouble(expr["gt"])
            expr.containsKey("gte") -> toDouble(rawValue) >= toDouble(expr["gte"])
            expr.containsKey("lt") -> toDouble(rawValue) < toDouble(expr["lt"])
            expr.containsKey("lte") -> toDouble(rawValue) <= toDouble(expr["lte"])
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
        return Regex("""\$\{(/[^}]+)}""").replace(template) { match ->
            val path = match.groupValues.getOrNull(1).orEmpty()
            FlatSpecParser.getAtPath(state, path)?.toString().orEmpty()
        }
    }
}

@Composable
fun FlatSpecContent(
    spec: FlatSpec,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val stateStore = remember(spec) {
        mutableStateMapOf<String, Any?>().apply { putAll(spec.state) }
    }

    val onOpenUrl: (String) -> Unit = { url ->
        runCatching { context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url))) }
    }
    val onSetState: (String, Any?) -> Unit = { path, value ->
        FlatSpecParser.setAtPath(stateStore, path, value)
    }

    RenderElement(
        elementId = spec.root,
        elements = spec.elements,
        state = stateStore,
        itemContext = null,
        onOpenUrl = onOpenUrl,
        onSetState = onSetState,
        activePath = emptySet(),
        modifier = modifier
    )
}

@Composable
private fun RenderElement(
    elementId: String,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    itemContext: Map<String, Any?>?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    activePath: Set<String>,
    modifier: Modifier = Modifier
) {
    if (elementId in activePath) return
    val element = elements[elementId] ?: return
    if (!FlatExprResolver.evaluateVisible(element.visible, state, itemContext)) return

    if (element.repeat != null) {
        val rawItems = FlatSpecParser.getAtPath(state, element.repeat.statePath)
        val items = rawItems as? List<*> ?: emptyList<Any?>()
        items.forEach { rawItem ->
            val mapped = when (rawItem) {
                is Map<*, *> -> rawItem.entries.associate { (k, v) -> k.toString() to v }
                else -> mapOf("value" to rawItem)
            }
            val resolvedProps = element.props.mapValues { (_, value) ->
                FlatExprResolver.resolve(value, state, mapped)
            }
            RenderByType(
                type = element.type,
                props = resolvedProps,
                children = element.children,
                onMap = element.on,
                elements = elements,
                state = state,
                itemContext = mapped,
                onOpenUrl = onOpenUrl,
                onSetState = onSetState,
                activePath = activePath + elementId,
                modifier = modifier
            )
        }
        return
    }

    val resolvedProps = element.props.mapValues { (_, value) ->
        FlatExprResolver.resolve(value, state, itemContext)
    }

    RenderByType(
        type = element.type,
        props = resolvedProps,
        children = element.children,
        onMap = element.on,
        elements = elements,
        state = state,
        itemContext = itemContext,
        onOpenUrl = onOpenUrl,
        onSetState = onSetState,
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
    itemContext: Map<String, Any?>?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    activePath: Set<String>,
    modifier: Modifier = Modifier
) {
    when (type.lowercase()) {
        "column" -> RenderColumn(children, elements, state, itemContext, onOpenUrl, onSetState, activePath, modifier)
        "row" -> RenderRow(props, children, elements, state, itemContext, onOpenUrl, onSetState, activePath, modifier)
        "list" -> RenderList(children, elements, state, itemContext, onOpenUrl, onSetState, activePath, modifier)
        "card" -> RenderCard(props, children, elements, state, itemContext, onOpenUrl, onSetState, activePath, modifier)
        "text" -> RenderText(props, modifier)
        "image" -> RenderImage(props, onOpenUrl, modifier)
        "icon" -> RenderIcon(props, modifier)
        "button" -> RenderButton(props, onMap, onOpenUrl, onSetState, state, itemContext, modifier)
        "divider" -> RenderDivider(modifier)
        "tabs" -> RenderTabs(props, elements, state, itemContext, onOpenUrl, onSetState, activePath, modifier)
        "modal" -> RenderModal(props, children, elements, state, itemContext, onOpenUrl, onSetState, activePath, modifier)
        "textfield" -> RenderTextField(props, onSetState, state, itemContext, modifier)
        "checkbox" -> RenderCheckBox(props, onSetState, state, itemContext, modifier)
        "choicepicker" -> RenderChoicePicker(props, onSetState, state, itemContext, modifier)
        "slider" -> RenderSlider(props, onSetState, state, itemContext, modifier)
        "datetimeinput" -> RenderDateTimeInput(props, onSetState, state, itemContext, modifier)
        "video" -> RenderVideo(props, onOpenUrl, modifier)
        "audioplayer" -> RenderAudioPlayer(props, onOpenUrl, modifier)
        else -> if (children.isNotEmpty()) {
            RenderColumn(children, elements, state, itemContext, onOpenUrl, onSetState, activePath, modifier)
        }
    }
}

@Composable
private fun RenderColumn(
    children: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    itemContext: Map<String, Any?>?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    activePath: Set<String>,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        children.forEach { childId ->
            RenderElement(
                elementId = childId,
                elements = elements,
                state = state,
                itemContext = itemContext,
                onOpenUrl = onOpenUrl,
                onSetState = onSetState,
                activePath = activePath
            )
        }
    }
}

@Composable
private fun RenderRow(
    props: Map<String, Any?>,
    children: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    itemContext: Map<String, Any?>?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    activePath: Set<String>,
    modifier: Modifier = Modifier
) {
    val arrangement = when (props["justify"]?.toString()?.lowercase()) {
        "center" -> Arrangement.Center
        "end" -> Arrangement.End
        "spacebetween", "space-between" -> Arrangement.SpaceBetween
        else -> Arrangement.spacedBy(8.dp)
    }
    Row(
        modifier = modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState()),
        horizontalArrangement = arrangement
    ) {
        children.forEach { childId ->
            RenderElement(
                elementId = childId,
                elements = elements,
                state = state,
                itemContext = itemContext,
                onOpenUrl = onOpenUrl,
                onSetState = onSetState,
                activePath = activePath
            )
        }
    }
}

@Composable
private fun RenderList(
    children: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    itemContext: Map<String, Any?>?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    activePath: Set<String>,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        children.forEach { childId ->
            RenderElement(
                elementId = childId,
                elements = elements,
                state = state,
                itemContext = itemContext,
                onOpenUrl = onOpenUrl,
                onSetState = onSetState,
                activePath = activePath
            )
        }
    }
}

@Composable
private fun RenderCard(
    props: Map<String, Any?>,
    children: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    itemContext: Map<String, Any?>?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
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
            allChildren.forEach { childId ->
                RenderElement(
                    elementId = childId,
                    elements = elements,
                    state = state,
                    itemContext = itemContext,
                    onOpenUrl = onOpenUrl,
                    onSetState = onSetState,
                    activePath = activePath
                )
            }
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
    val url = props["url"]?.toString()?.trim().orEmpty()
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
        contentDescription = props["alt"]?.toString(),
        contentScale = if (isCover) ContentScale.Crop else ContentScale.Fit,
        modifier = clickableModifier
    )
}

@Composable
private fun RenderIcon(
    props: Map<String, Any?>,
    modifier: Modifier = Modifier
) {
    val url = props["name"]?.toString()?.trim().orEmpty()
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
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    state: Map<String, Any?>,
    itemContext: Map<String, Any?>?,
    modifier: Modifier = Modifier
) {
    val label = props["label"]?.toString().orEmpty().ifBlank { "Open" }
    val variant = props["variant"]?.toString()?.lowercase().orEmpty()
    val onClick = resolveActionHandler(
        actionCandidate = props["action"] ?: onMap?.get("press"),
        state = state,
        itemContext = itemContext,
        onOpenUrl = onOpenUrl,
        onSetState = onSetState
    )
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

private fun resolveActionHandler(
    actionCandidate: Any?,
    state: Map<String, Any?>,
    itemContext: Map<String, Any?>?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit
): () -> Unit {
    val action = toStringKeyMap(actionCandidate) ?: return {}

    val functionCall = extractFunctionCall(action) ?: return {}

    val call = (functionCall["call"] ?: functionCall["action"] ?: functionCall["name"])
        ?.toString()
        .orEmpty()
        .trim()
        .lowercase()
    val args = extractActionArgs(functionCall)

    return when (call) {
        "openurl" -> {
            val raw = resolveFirstArg(args, state, itemContext, "url", "href", "link", "targetUrl")
            if (raw.isNotBlank()) ({ onOpenUrl(raw) }) else ({})
        }

        "setstate" -> {
            val path = resolveFirstArg(
                args,
                state,
                itemContext,
                "path",
                "statePath",
                "\$state",
                "bindState",
                "\$bindState"
            )
            val value = FlatExprResolver.resolve(args["value"], state, itemContext)
            if (path.isNotBlank()) ({ onSetState(path, value) }) else ({})
        }

        else -> ({})
    }
}

private fun toStringKeyMap(value: Any?): Map<String, Any?>? {
    val map = value as? Map<*, *> ?: return null
    return map.entries.associate { (k, v) -> k.toString() to v }
}

private fun extractFunctionCall(action: Map<String, Any?>): Map<String, Any?>? {
    if (action["functionCall"] is Map<*, *>) {
        return toStringKeyMap(action["functionCall"])
    }
    if (action["call"] is String || action["action"] is String || action["name"] is String) {
        return action
    }
    val event = toStringKeyMap(action["event"])
    if (event != null) {
        return extractFunctionCall(event)
    }
    listOf("onClick", "click", "tap", "press", "onPress", "onSelect", "select").forEach { key ->
        val nested = toStringKeyMap(action[key]) ?: return@forEach
        extractFunctionCall(nested)?.let { return it }
    }
    return null
}

private fun extractActionArgs(functionCall: Map<String, Any?>): Map<String, Any?> {
    val args = toStringKeyMap(functionCall["args"])
    if (args != null) {
        return args
    }
    val params = toStringKeyMap(functionCall["params"])
    if (params != null) {
        return params
    }
    val contextEntries = functionCall["context"] as? List<*> ?: return emptyMap()
    val out = linkedMapOf<String, Any?>()
    contextEntries.forEach { entry ->
        val context = toStringKeyMap(entry) ?: return@forEach
        val key = context["key"]?.toString().orEmpty()
        if (key.isNotBlank()) {
            out[key] = context["value"]
        }
    }
    return out
}

private fun resolveFirstArg(
    args: Map<String, Any?>,
    state: Map<String, Any?>,
    itemContext: Map<String, Any?>?,
    vararg keys: String
): String {
    keys.forEach { key ->
        val resolved = FlatExprResolver.resolve(args[key], state, itemContext)
            ?.toString()
            .orEmpty()
            .trim()
        if (resolved.isNotBlank()) {
            return resolved
        }
    }
    return ""
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
    itemContext: Map<String, Any?>?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
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
                itemContext = itemContext,
                onOpenUrl = onOpenUrl,
                onSetState = onSetState,
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
    itemContext: Map<String, Any?>?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
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
                    itemContext = itemContext,
                    onOpenUrl = onOpenUrl,
                    onSetState = onSetState,
                    activePath = activePath
                )
            }
        }
    }
}

private fun bindPathFromValueExpression(value: Any?): String? {
    if (value !is Map<*, *>) return null
    val bindState = value["\$bindState"]?.toString()?.takeIf { it.isNotBlank() }
    if (bindState != null) {
        return bindState
    }
    return value["\$state"]?.toString()?.takeIf { it.isNotBlank() }
}

@Composable
private fun RenderTextField(
    props: Map<String, Any?>,
    onSetState: (String, Any?) -> Unit,
    state: Map<String, Any?>,
    itemContext: Map<String, Any?>?,
    modifier: Modifier = Modifier
) {
    val label = props["label"]?.toString().orEmpty().ifBlank { "Input" }
    val bindPath = bindPathFromValueExpression(props["value"]) ?: props["statePath"]?.toString()
    val value = FlatExprResolver.resolveString(props["value"], state, itemContext)
    var localValue by remember(label) { mutableStateOf(value) }
    val textValue = if (!bindPath.isNullOrBlank()) value else localValue

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
    )
}

@Composable
private fun RenderCheckBox(
    props: Map<String, Any?>,
    onSetState: (String, Any?) -> Unit,
    state: Map<String, Any?>,
    itemContext: Map<String, Any?>?,
    modifier: Modifier = Modifier
) {
    val label = props["label"]?.toString().orEmpty().ifBlank { "Option" }
    val bindPath = bindPathFromValueExpression(props["value"]) ?: props["statePath"]?.toString()
    val checked = FlatExprResolver.resolveBoolean(props["value"], state, itemContext)
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
    itemContext: Map<String, Any?>?,
    modifier: Modifier = Modifier
) {
    val label = props["label"]?.toString().orEmpty().ifBlank { "Choose" }
    val bindPath = bindPathFromValueExpression(props["value"]) ?: props["statePath"]?.toString()
    val resolvedValue = FlatExprResolver.resolve(props["value"], state, itemContext)
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
    itemContext: Map<String, Any?>?,
    modifier: Modifier = Modifier
) {
    val label = props["label"]?.toString().orEmpty()
    val bindPath = bindPathFromValueExpression(props["value"]) ?: props["statePath"]?.toString()
    val min = (props["min"] as? Number)?.toFloat() ?: 0f
    val max = (props["max"] as? Number)?.toFloat() ?: 100f
    val resolved = FlatExprResolver.resolve(props["value"], state, itemContext)
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
    itemContext: Map<String, Any?>?,
    modifier: Modifier = Modifier
) {
    val label = props["label"]?.toString().orEmpty().ifBlank { "Date/Time" }
    val bindPath = bindPathFromValueExpression(props["value"]) ?: props["statePath"]?.toString()
    val resolvedValue = FlatExprResolver.resolveString(props["value"], state, itemContext)
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
