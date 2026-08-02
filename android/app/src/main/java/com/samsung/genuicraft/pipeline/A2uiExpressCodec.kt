package com.samsung.genuicraft.pipeline

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonNull
import com.google.gson.JsonObject
import com.google.gson.JsonPrimitive

/**
 * Dependency-free implementation of the pinned A2UI Express grammar subset
 * used by GenUICraft. It preserves custom component props and metadata through
 * `_props`, `_children`, `_repeat`, `_visible`, `_on`, and `_watch`.
 */
internal object A2uiExpressCodec {
    private const val OPEN = "<a2ui>"
    private const val CLOSE = "</a2ui>"

    fun looksLike(text: String?): Boolean {
        val value = text?.trim().orEmpty()
        return value.contains(OPEN) || value.lineSequence().any { it.trimStart().startsWith("root=") || it.trimStart().startsWith("root =") }
    }

    fun encode(flatSpec: JsonObject): String {
        val source = FlatSpecIdRewriter.rewrite(flatSpec, shorten = true, reserveRoot = true)
        val lines = mutableListOf(OPEN)
        source.get("state")?.takeIf { it.isJsonObject && it.asJsonObject.size() > 0 }
            ?.let { lines += "$/=${it}" }
        source.get("elements")?.takeIf { it.isJsonObject }?.asJsonObject?.entrySet()?.forEach { (id, raw) ->
            if (!raw.isJsonObject) return@forEach
            val element = raw.asJsonObject
            val props = element.get("props")?.takeIf { it.isJsonObject }?.asJsonObject?.deepCopy() ?: JsonObject()
            val children = element.get("children")?.takeIf { it.isJsonArray }?.asJsonArray?.deepCopy() ?: JsonArray()
            val rawType = element.string("type") ?: "Stack"
            val type = if (rawType == "Stack") {
                when (props.string("direction")?.lowercase()) {
                    "horizontal", "row" -> {
                        props.remove("direction")
                        "Row"
                    }
                    "vertical", "column" -> {
                        props.remove("direction")
                        "Column"
                    }
                    else -> rawType
                }
            } else rawType
            val positional = GenUiA2uiCatalog.positional[type].orEmpty()
            val values = positional.map { key -> if (key == "children") children else props.get(key) }
            var last = -1
            values.forEachIndexed { index, value -> if (value != null && !value.isJsonNull && !(value.isJsonArray && value.asJsonArray.size() == 0)) last = index }
            val args = mutableListOf<String>()
            val consumed = mutableSetOf<String>()
            for (index in 0..last) {
                val key = positional[index]
                val value = values[index]
                if (value == null || value.isJsonNull || (value.isJsonArray && value.asJsonArray.size() == 0)) {
                    args += "_"
                } else {
                    args += if (key == "children") componentReferences(value.asJsonArray) else expressValue(value)
                    consumed += key
                }
            }
            val leftover = JsonObject()
            props.entrySet().filter { it.key !in consumed }.forEach { (key, value) -> leftover.add(key, value.deepCopy()) }
            if (leftover.size() > 0) args += "_props=$leftover"
            if (children.size() > 0 && "children" !in consumed) args += "_children=$children"
            val remainingEvents = JsonObject()
            element.get("on")?.takeIf { it.isJsonObject }?.asJsonObject?.entrySet()?.forEach { (event, action) ->
                val keyword = eventKeyword(event)
                val expression = actionExpression(action)
                if (keyword != null && expression != null) args += "$keyword=$expression"
                else remainingEvents.add(event, action.deepCopy())
            }
            if (remainingEvents.size() > 0) args += "_on=$remainingEvents"
            listOf("repeat" to "_repeat", "visible" to "_visible", "watch" to "_watch").forEach { (full, short) ->
                element.get(full)?.takeUnless { it.isJsonNull || (it.isJsonObject && it.asJsonObject.size() == 0) }
                    ?.let { args += "$short=$it" }
            }
            lines += "$id=$type(${args.joinToString(",")})"
        }
        lines += CLOSE
        return lines.joinToString("\n")
    }

    fun decode(text: String): JsonObject {
        val state = JsonObject()
        val elements = JsonObject()
        var inlineCounter = 0

        fun materialize(id: String, call: Expr.Call): String {
            val expressComponent = call.name
            val positional = GenUiA2uiCatalog.positional[expressComponent]
                ?: error("Unknown A2UI Express component '$expressComponent'.")
            val component = when (expressComponent) {
                "Row" -> "Stack"
                "Column" -> "Stack"
                else -> expressComponent
            }
            val props = JsonObject()
            if (expressComponent == "Row") props.addProperty("direction", "horizontal")
            if (expressComponent == "Column") props.addProperty("direction", "vertical")
            val children = JsonArray()
            val directEvents = JsonObject()
            call.args.forEachIndexed { index, expr ->
                if (expr === Expr.Skipped) return@forEachIndexed
                require(index < positional.size) { "Too many positional args for $expressComponent." }
                val key = positional[index]
                if (key == "children") {
                    val array = expr as? Expr.ArrayValue ?: error("$expressComponent.children must be an array.")
                    array.items.forEach { child ->
                        when (child) {
                            is Expr.Ref -> children.add(child.name)
                            is Expr.Literal -> {
                                require(child.value.isJsonPrimitive && child.value.asJsonPrimitive.isString) {
                                    "$expressComponent.children must contain component references."
                                }
                                children.add(child.value.asString)
                            }
                            is Expr.Call -> {
                                inlineCounter += 1
                                val childId = "${id}_i$inlineCounter"
                                materialize(childId, child)
                                children.add(childId)
                            }
                            else -> error("Unsupported inline child expression.")
                        }
                    }
                } else {
                    props.add(key, expr.toJson())
                }
            }
            val metadataKeys = setOf("_props", "_children", "_repeat", "_visible", "_on", "_watch")
            call.kwargs["_props"]?.let { expr ->
                val extra = expr.toJson()
                require(extra.isJsonObject) { "_props must be an object." }
                extra.asJsonObject.entrySet().forEach { (key, value) -> props.add(key, value.deepCopy()) }
            }
            call.kwargs["_children"]?.let { expr ->
                val value = expr.toJson()
                require(value.isJsonArray) { "_children must be an array." }
                while (children.size() > 0) children.remove(0)
                value.asJsonArray.forEach { child ->
                    require(child.isJsonPrimitive && child.asJsonPrimitive.isString) { "_children values must be strings." }
                    children.add(child.asString)
                }
            }
            call.kwargs.filterKeys { it !in metadataKeys }.forEach { (key, expr) ->
                val eventName = eventName(key)
                if (eventName != null) {
                    directEvents.add(eventName, actionFromExpr(expr))
                } else if (key == "children") {
                    val value = expr.toJson()
                    require(value.isJsonArray) { "children must be an array." }
                    while (children.size() > 0) children.remove(0)
                    value.asJsonArray.forEach { child -> children.add(child.asString) }
                } else {
                    props.add(key, expr.toJson())
                }
            }
            val element = JsonObject().apply {
                addProperty("type", component)
                add("props", props)
                add("children", children)
            }
            if (directEvents.size() > 0) element.add("on", directEvents)
            call.kwargs["_repeat"]?.let { element.add("repeat", it.toJson()) }
            call.kwargs["_visible"]?.let { element.add("visible", it.toJson()) }
            call.kwargs["_on"]?.let { expression ->
                val normalized = actionMapFromExpr(expression)
                val on = element.get("on")?.takeIf { it.isJsonObject }?.asJsonObject ?: JsonObject()
                normalized.entrySet().forEach { (key, value) -> on.add(key, value) }
                element.add("on", on)
            }
            call.kwargs["_watch"]?.let { element.add("watch", actionMapFromExpr(it)) }
            elements.add(id, element)
            return id
        }

        statements(text).forEach { statement ->
            val (lhs, rhs) = splitAssignment(statement)
            val parser = Parser(rhs)
            val value = parser.parseValue()
            parser.requireComplete()
            if (lhs.startsWith("$")) {
                val json = value.toJson()
                if (lhs == "$" || lhs == "$/") {
                    require(json.isJsonObject) { "Root state assignment must be an object." }
                    json.asJsonObject.entrySet().forEach { (key, item) -> state.add(key, item.deepCopy()) }
                } else {
                    setStatePath(state, lhs, json)
                }
            } else {
                val call = value as? Expr.Call ?: error("A2UI Express assignment '$lhs' must be a component call.")
                materialize(lhs, call)
            }
        }
        require(elements.has("root")) { "A2UI Express requires reserved root component." }
        return JsonObject().apply {
            addProperty("root", "root")
            add("state", state)
            add("elements", elements)
        }
    }

    private val actionPositional = mapOf(
        "openUrl" to listOf("url"),
        "setState" to listOf("statePath", "value"),
        "pushState" to listOf("statePath", "value", "clearStatePath"),
        "removeState" to listOf("statePath", "index"),
        "validateForm" to listOf("statePath", "resultStatePath"),
        "emitEvent" to listOf("name", "context", "wantResponse", "responsePath"),
    )

    private fun componentReferences(value: JsonArray): String = value.joinToString(
        prefix = "[",
        postfix = "]",
        separator = ",",
    ) { child ->
        require(child.isJsonPrimitive && child.asJsonPrimitive.isString) {
            "Express component references must be strings."
        }
        val id = child.asString
        require(id.matches(Regex("[A-Za-z_][A-Za-z0-9_]*"))) { "Invalid Express component id '$id'." }
        id
    }

    private fun expressValue(value: JsonElement): String = when {
        value.isJsonNull -> "null"
        value.isJsonPrimitive && value.asJsonPrimitive.isString -> {
            val raw = value.asString
            if (raw.matches(Regex("\\$(?:/[A-Za-z0-9_.-]+)+"))) raw else value.toString()
        }
        value.isJsonArray -> value.asJsonArray.joinToString(separator = ",", prefix = "[", postfix = "]") { expressValue(it) }
        value.isJsonObject -> value.asJsonObject.entrySet().joinToString(separator = ",", prefix = "{", postfix = "}") { (key, item) ->
            val renderedKey = if (key.matches(Regex("[A-Za-z_][A-Za-z0-9_]*"))) key else JsonPrimitive(key).toString()
            "$renderedKey:${expressValue(item)}"
        }
        else -> value.toString()
    }

    private fun eventKeyword(event: String): String? {
        val parts = Regex("[A-Za-z0-9]+").findAll(event).map { it.value }.toList()
        if (parts.isEmpty()) return null
        val suffix = parts.joinToString("") { it.replaceFirstChar(Char::uppercaseChar) }
        return "on$suffix".takeIf { it.matches(Regex("[A-Za-z_][A-Za-z0-9_]*")) }
    }

    private fun eventName(keyword: String): String? = keyword
        .takeIf { it.startsWith("on") && it.length > 2 }
        ?.substring(2)
        ?.replaceFirstChar(Char::lowercaseChar)

    private fun actionExpression(value: JsonElement): String? {
        if (value.isJsonArray) {
            val expressions = value.asJsonArray.map { actionExpression(it) ?: return null }
            return expressions.joinToString(separator = ",", prefix = "[", postfix = "]")
        }
        if (!value.isJsonObject) return null
        val action = value.asJsonObject.string("action") ?: return null
        val positional = actionPositional[action] ?: return null
        val params = value.asJsonObject.get("params")?.takeIf { it.isJsonObject }?.asJsonObject ?: JsonObject()
        var last = -1
        positional.forEachIndexed { index, key -> if (params.has(key)) last = index }
        val args = mutableListOf<String>()
        val consumed = mutableSetOf<String>()
        for (index in 0..last) {
            val key = positional[index]
            val raw = params.get(key)
            if (raw == null) args += "_" else {
                args += expressValue(raw)
                consumed += key
            }
        }
        params.entrySet().filter { it.key !in consumed }.forEach { (key, raw) ->
            if (!key.matches(Regex("[A-Za-z_][A-Za-z0-9_]*"))) return null
            args += "$key=${expressValue(raw)}"
        }
        val function = if (action == "emitEvent") "Event" else action
        return "$function(${args.joinToString(",")})"
    }

    private fun actionFromExpr(expression: Expr): JsonElement {
        if (expression is Expr.ArrayValue) {
            require(expression.items.isNotEmpty()) { "Action list must not be empty." }
            return JsonArray().apply { expression.items.forEach { add(actionFromExpr(it)) } }
        }
        val literal = expression.toJson()
        if (literal.isJsonObject && literal.asJsonObject.string("action") != null) return literal
        val call = expression as? Expr.Call ?: error("Event binding must be an action call.")
        val action = if (call.name.equals("Event", ignoreCase = true)) {
            "emitEvent"
        } else {
            actionPositional.keys.firstOrNull { it.equals(call.name, ignoreCase = true) }
                ?: error("Unknown Express action '${call.name}'.")
        }
        val positional = actionPositional.getValue(action)
        require(call.args.size <= positional.size) { "Too many positional parameters for action ${call.name}." }
        val params = JsonObject()
        call.args.forEachIndexed { index, expr ->
            if (expr !== Expr.Skipped) params.add(positional[index], expr.toJson())
        }
        call.kwargs.forEach { (key, expr) ->
            require(!params.has(key)) { "Duplicate action parameter '$key'." }
            params.add(key, expr.toJson())
        }
        return JsonObject().apply {
            addProperty("action", action)
            add("params", params)
        }
    }

    private fun actionMapFromExpr(expression: Expr): JsonObject {
        val map = expression as? Expr.ObjectValue ?: error("Action map must be an object.")
        return JsonObject().apply {
            map.entries.forEach { (key, value) -> add(key, actionFromExpr(value)) }
        }
    }

    private fun setStatePath(state: JsonObject, path: String, value: JsonElement) {
        val parts = path.removePrefix("$/").removePrefix("$").split('/').filter { it.isNotBlank() }
        if (parts.isEmpty()) return
        var current = state
        parts.dropLast(1).forEach { part ->
            val next = current.get(part)?.takeIf { it.isJsonObject }?.asJsonObject ?: JsonObject().also { current.add(part, it) }
            current = next
        }
        current.add(parts.last(), value.deepCopy())
    }

    private fun statements(text: String): List<String> {
        val body = if (text.contains(OPEN)) text.substringAfter(OPEN).substringBefore(CLOSE) else text
        val out = mutableListOf<String>()
        val buffer = StringBuilder()
        var depth = 0
        var inString = false
        var escaped = false
        body.forEach { ch ->
            if (inString) {
                buffer.append(ch)
                when {
                    escaped -> escaped = false
                    ch == '\\' -> escaped = true
                    ch == '"' -> inString = false
                }
                return@forEach
            }
            when (ch) {
                '"' -> { inString = true; buffer.append(ch) }
                '(', '[', '{' -> { depth += 1; buffer.append(ch) }
                ')', ']', '}' -> { depth -= 1; buffer.append(ch) }
                '\n', ';' -> if (depth == 0) {
                    buffer.toString().trim().takeIf { it.isNotEmpty() && !it.startsWith("#") && !it.startsWith("//") }?.let(out::add)
                    buffer.clear()
                } else buffer.append(ch)
                else -> buffer.append(ch)
            }
        }
        buffer.toString().trim().takeIf { it.isNotEmpty() }?.let(out::add)
        return out
    }

    private fun splitAssignment(statement: String): Pair<String, String> {
        var depth = 0
        var inString = false
        var escaped = false
        statement.forEachIndexed { index, ch ->
            if (inString) {
                when {
                    escaped -> escaped = false
                    ch == '\\' -> escaped = true
                    ch == '"' -> inString = false
                }
            } else {
                when (ch) {
                    '"' -> inString = true
                    '(', '[', '{' -> depth += 1
                    ')', ']', '}' -> depth -= 1
                    '=' -> if (depth == 0) return statement.substring(0, index).trim() to statement.substring(index + 1).trim()
                }
            }
        }
        error("A2UI Express statement is not an assignment: ${statement.take(80)}")
    }

    private sealed interface Expr {
        fun toJson(): JsonElement

        data class Literal(val value: JsonElement) : Expr { override fun toJson(): JsonElement = value.deepCopy() }
        data class Ref(val name: String) : Expr { override fun toJson(): JsonElement = JsonPrimitive(name) }
        data class ArrayValue(val items: List<Expr>) : Expr {
            override fun toJson(): JsonElement = JsonArray().apply { items.forEach { add(it.toJson()) } }
        }
        data class ObjectValue(val entries: Map<String, Expr>) : Expr {
            override fun toJson(): JsonElement = JsonObject().apply { entries.forEach { (key, value) -> add(key, value.toJson()) } }
        }
        data class Call(val name: String, val args: List<Expr>, val kwargs: Map<String, Expr>) : Expr {
            override fun toJson(): JsonElement = JsonObject().apply {
                addProperty("call", name)
                if (args.isNotEmpty()) add("args", JsonArray().apply { args.forEach { add(it.toJson()) } })
                if (kwargs.isNotEmpty()) add("kwargs", JsonObject().apply { kwargs.forEach { (key, value) -> add(key, value.toJson()) } })
            }
        }
        object Skipped : Expr { override fun toJson(): JsonElement = JsonNull.INSTANCE }
    }

    private class Parser(private val source: String) {
        private var index = 0
        fun requireComplete() { skipWhitespace(); require(index == source.length) { "Trailing A2UI Express syntax at $index." } }
        fun parseValue(): Expr {
            skipWhitespace()
            if (index >= source.length) error("Unexpected end of A2UI Express expression.")
            return when (source[index]) {
                '"' -> Expr.Literal(JsonPrimitive(parseString()))
                '[' -> parseArray()
                '{' -> parseObject()
                '$' -> Expr.Literal(JsonPrimitive(parsePath()))
                '?' -> parseCheck()
                '-', in '0'..'9' -> parseNumber()
                else -> parseIdentifierValue()
            }
        }
        private fun parseArray(): Expr.ArrayValue {
            consume('['); val items = mutableListOf<Expr>(); skipWhitespace()
            if (peek() != ']') while (true) {
                items += parseValue(); skipWhitespace()
                if (peek() == ',') { consume(','); continue }
                break
            }
            consume(']'); return Expr.ArrayValue(items)
        }
        private fun parseObject(): Expr.ObjectValue {
            consume('{'); val entries = linkedMapOf<String, Expr>(); skipWhitespace()
            if (peek() != '}') while (true) {
                val key = if (peek() == '"') parseString() else parseIdentifier()
                consume(':'); entries[key] = parseValue(); skipWhitespace()
                if (peek() == ',') { consume(','); continue }
                break
            }
            consume('}'); return Expr.ObjectValue(entries)
        }
        private fun parseCheck(): Expr.Call {
            consume('?'); val name = parseIdentifier(); val args = mutableListOf<Expr>()
            skipWhitespace(); if (peek() == '(') {
                consume('('); skipWhitespace()
                if (peek() != ')') while (true) {
                    args += parseValue(); skipWhitespace()
                    if (peek() == ',') { consume(','); continue }
                    break
                }
                consume(')')
            }
            return Expr.Call("?$name", args, emptyMap())
        }
        private fun parseNumber(): Expr.Literal {
            skipWhitespace(); val start = index
            if (peek() == '-') index += 1
            while (index < source.length && source[index].isDigit()) index += 1
            if (index < source.length && source[index] == '.') {
                index += 1; while (index < source.length && source[index].isDigit()) index += 1
            }
            val token = source.substring(start, index)
            return Expr.Literal(if (token.contains('.')) JsonPrimitive(token.toDouble()) else JsonPrimitive(token.toLong()))
        }
        private fun parseIdentifierValue(): Expr {
            val name = parseIdentifier()
            return when (name) {
                "true" -> Expr.Literal(JsonPrimitive(true))
                "false" -> Expr.Literal(JsonPrimitive(false))
                "null" -> Expr.Literal(JsonNull.INSTANCE)
                "_" -> Expr.Skipped
                else -> {
                    skipWhitespace()
                    if (peek() != '(') Expr.Ref(name) else parseCall(name)
                }
            }
        }
        private fun parseCall(name: String): Expr.Call {
            consume('('); val args = mutableListOf<Expr>(); val kwargs = linkedMapOf<String, Expr>(); skipWhitespace()
            if (peek() != ')') while (true) {
                val saved = index
                val named = runCatching {
                    val candidate = parseIdentifier(); skipWhitespace()
                    if (peek() == '=') candidate else null
                }.getOrNull()
                if (named != null) {
                    consume('='); kwargs[named] = parseValue()
                } else {
                    index = saved; args += parseValue()
                }
                skipWhitespace(); if (peek() == ',') { consume(','); continue }
                break
            }
            consume(')'); return Expr.Call(name, args, kwargs)
        }
        private fun parseString(): String {
            skipWhitespace(); val start = index
            index = start + 1; val out = StringBuilder(); var escaped = false
            while (index < source.length) {
                val ch = source[index++]
                if (escaped) {
                    out.append(when (ch) { 'n' -> '\n'; 'r' -> '\r'; 't' -> '\t'; else -> ch }); escaped = false
                } else if (ch == '\\') escaped = true
                else if (ch == '"') return out.toString()
                else out.append(ch)
            }
            error("Unterminated A2UI Express string.")
        }
        private fun parsePath(): String {
            skipWhitespace(); val start = index; index += 1
            while (index < source.length && (source[index].isLetterOrDigit() || source[index] in "_/.-")) index += 1
            return source.substring(start, index)
        }
        private fun parseIdentifier(): String {
            skipWhitespace(); val start = index
            require(index < source.length && (source[index].isLetter() || source[index] == '_')) { "Expected identifier at $index." }
            index += 1
            while (index < source.length && (source[index].isLetterOrDigit() || source[index] == '_')) index += 1
            return source.substring(start, index)
        }
        private fun skipWhitespace() { while (index < source.length && source[index].isWhitespace()) index += 1 }
        private fun peek(): Char? { skipWhitespace(); return source.getOrNull(index) }
        private fun consume(expected: Char) { skipWhitespace(); require(source.getOrNull(index) == expected) { "Expected '$expected' at $index." }; index += 1 }
    }

    private fun JsonObject.string(key: String): String? =
        get(key)?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
}
