package com.samsung.genuicraft.sdk.internal.pipeline

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject

/**
 * Narrow generated-only repair of direct Table rows whose ordered duplicate-member
 * signature exactly matches the ordered explicit columns. This never edits state.
 */
internal object A2uiDuplicateTableRecovery {
    data class Prepared(val input: String, val changes: List<String>)
    private sealed interface Literal {
        data class Scalar(val value: JsonElement) : Literal
        data class Array(val values: List<Literal>) : Literal
        data class Members(val values: List<Pair<String, Literal>>) : Literal
    }
    private data class Declaration(val id: String, val start: Int, val end: Int, val call: String)
    private val declaration = Regex("(?m)(?:^|(?<=;))[\\t ]*([A-Za-z_][A-Za-z0-9_]*)[\\t ]*=[\\t ]*([A-Za-z_][A-Za-z0-9_]*)[\\t ]*\\(")
    // Inventory identities before validating RHS calls. Malformed/non-call definitions still
    // conflict; scanning all top-level assignment tokens also covers comment-prefixed statements.
    private val assignment = Regex("(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)[\\t ]*=[\\t ]*")
    private val binding = Regex("^(?:\\$[/A-Za-z_]|@source(?:\\.|$))")
    private val metadata = setOf("domain", "preferredPresentation", "presentation", "title", "subtitle")
    private val containers = setOf("Stack", "Card", "List")

    fun prepare(input: String): Prepared {
        if (input.length > A2uiExpressCodec.MAX_INPUT_CHARS) return Prepared(input, emptyList())
        val mask = outside(input)
        val declarationScope = topLevel(input, mask) ?: return Prepared(input, emptyList())
        val assignments = assignment.findAll(input).filter { match ->
            val start = match.groups[1]!!.range.first
            mask[start] && declarationScope[start]
        }.toList()
        if (assignments.size > A2uiExpressCodec.MAX_STATEMENTS) return Prepared(input, emptyList())
        val duplicates = assignments.groupingBy { it.groupValues[1] }.eachCount().filterValues { it > 1 }.keys
        val declarations = declaration.findAll(input).filter { match ->
            val start = match.groups[1]!!.range.first
            mask[start] && declarationScope[start]
        }.mapNotNull { match ->
            val start = match.groups[2]!!.range.first
            callEnd(input, start, mask)?.let { end ->
                // Do not ignore unparsed properties/expressions after an apparently closed call.
                if (!completeTail(input, end)) return@let null
                Declaration(match.groupValues[1], match.range.first, end, input.substring(start, end))
            }
        }.toList()
        if (declarations.size > A2uiExpressCodec.MAX_STATEMENTS) return Prepared(input, emptyList())
        val projected = linkedMapOf<String, String>()
        declarations.filter { it.id !in duplicates }.forEach { d ->
            if (Regex("^Table\\s*\\(").containsMatchIn(d.call)) {
                project(d.call)?.let { projected[d.id] = it }
            }
        }
        if (projected.isEmpty()) return Prepared(input, emptyList())
        val elements = linkedMapOf<String, JsonObject>()
        declarations.filter { it.id !in duplicates }.forEach { d ->
            val call = projected[d.id] ?: d.call
            runCatching {
                A2uiExpressCodec.decodeStatements(listOf(d.id + "=" + call), requireRoot = false)
                    .getAsJsonObject("elements").getAsJsonObject(d.id)
            }.getOrNull()?.let { elements[d.id] = it }
        }
        val safe = linkedSetOf<String>()
        fun visit(id: String, depth: Int) {
            if (depth >= A2uiExpressCodec.MAX_EXPRESSION_DEPTH || id in safe) return
            val element = elements[id] ?: return
            if (listOf("visible", "repeat", "on", "watch").any(element::has) ||
                hasBinding(element.get("props"))) return
            safe += id
            // Only static containment proves a table's original placement. Do not promote orphans,
            // modal content, selectable tab content or an opaque/guarded ancestor.
            if (element.get("type")?.asString in containers) {
                element.getAsJsonArray("children")?.forEach { visit(it.asString, depth + 1) }
            }
        }
        visit("root", 0)
        val edits = declarations.filter { it.id in projected && it.id in safe }
        if (edits.isEmpty()) return Prepared(input, emptyList())
        val output = StringBuilder(input)
        edits.asReversed().forEach { d -> output.replace(d.start, d.end, d.id + "=" + projected.getValue(d.id)) }
        return Prepared(output.toString(), edits.map {
            "Projected complete ordered duplicate-key Table '" + it.id +
                "' to its emitted column labels and exact positional cell literals; state was not edited."
        })
    }

    private fun project(call: String): String? = runCatching {
        val reader = Reader(call)
        require(reader.name() == "Table")
        reader.character('(')
        val args = linkedMapOf<String, Literal>()
        val order = mutableListOf<String>()
        if (!reader.at(')')) while (true) {
            val key = reader.name()
            require(key in metadata || key in setOf("columns", "rows"))
            require(!args.containsKey(key))
            reader.character('=')
            args[key] = reader.literal()
            order += key
            if (!reader.at(',')) break
            reader.character(',')
            if (reader.at(')')) break
        }
        reader.character(')')
        reader.complete()
        val columns = (args["columns"] as? Literal.Array)?.values ?: error("Explicit columns required.")
        require(columns.isNotEmpty())
        val keys = mutableListOf<String>()
        val labels = mutableListOf<String>()
        columns.forEach { column ->
            val members = (column as? Literal.Members)?.values ?: error("Explicit keyed labels required.")
            require(members.size == 2 && members.map { it.first }.toSet() == setOf("key", "label"))
            val key = scalar(members.single { it.first == "key" }.second)
            val label = scalar(members.single { it.first == "label" }.second)
            require(key.isJsonPrimitive && key.asJsonPrimitive.isString && key.asString.isNotBlank())
            require(label.isJsonPrimitive && label.asJsonPrimitive.isString && label.asString.isNotBlank())
            require(!hasDynamicText(key.asString) && !hasDynamicText(label.asString))
            keys += key.asString
            labels += label.asString
        }
        require(keys.toSet().size < keys.size) // Unique maps already have a strict interpretation.
        require(labels.map(String::trim).toSet().size == labels.size)
        val rows = (args["rows"] as? Literal.Array)?.values ?: error("Literal direct rows required.")
        require(rows.isNotEmpty())
        val positional = JsonArray()
        rows.forEach { row ->
            val members = (row as? Literal.Members)?.values ?: error("Ordered member rows required.")
            require(members.map { it.first } == keys)
            positional.add(JsonArray().apply { members.forEach { (_, value) -> add(scalar(value)) } })
        }
        val labelArray = JsonArray().apply { labels.forEach(::add) }
        val rendered = order.map { key ->
            val value = when (key) {
                "columns" -> labelArray
                "rows" -> positional
                else -> scalar(args.getValue(key)).also {
                    require(it.isJsonPrimitive && it.asJsonPrimitive.isString)
                }
            }
            key + "=" + value
        }
        val result = "Table(" + rendered.joinToString(",") + ")"
        val graph = A2uiExpressCodec.decode("<a2ui>\nroot=" + result + "\n</a2ui>")
        require(A2uiCanonicalGraph.validate(graph).isValid)
        result
    }.getOrNull()

    private fun scalar(value: Literal): JsonElement {
        val literal = (value as? Literal.Scalar)?.value ?: error("Nested or expression cells are ambiguous.")
        require(literal.isJsonPrimitive || literal.isJsonNull)
        if (literal.isJsonPrimitive && literal.asJsonPrimitive.isString) {
            require(!hasDynamicText(literal.asString))
        }
        return literal.deepCopy()
    }

    private fun hasBinding(value: JsonElement?): Boolean = when {
        value == null -> false
        value.isJsonPrimitive && value.asJsonPrimitive.isString -> hasDynamicText(value.asString)
        value.isJsonObject -> value.asJsonObject.entrySet().any { (key, child) ->
            // Runtime expression maps include $item/$bindItem/$index/$cond/$template/$computed.
            // Any reserved dollar key is opaque here, including future expression extensions.
            key.startsWith("\$") || key in setOf("statePath", "rowsPath", "dataPath") || hasBinding(child)
        }
        value.isJsonArray -> value.asJsonArray.any(::hasBinding)
        else -> false
    }

    private fun hasDynamicText(value: String): Boolean = binding.containsMatchIn(value) ||
        value.contains("{{") || value.contains("}}") || value.contains("\${") ||
        value.contains("\$item") || value.contains("\$index")

    private fun completeTail(input: String, start: Int): Boolean {
        var index = start
        while (index < input.length) {
            if (input[index] in " \t\r") { index++; continue }
            if (input.startsWith("/*", index)) {
                val end = input.indexOf("*/", index + 2)
                if (end < 0) return false
                index = end + 2; continue
            }
            return input[index] in "\n;" || input[index] == '#' ||
                input.startsWith("//", index) || input.startsWith("</a2ui>", index)
        }
        return true
    }

    private fun topLevel(input: String, mask: BooleanArray): BooleanArray? {
        val result = BooleanArray(input.length)
        val stack = java.util.ArrayDeque<Char>()
        for (index in input.indices) {
            result[index] = stack.isEmpty()
            if (!mask[index]) continue
            when (val char = input[index]) {
                '(', '[', '{' -> {
                    if (stack.size >= A2uiExpressCodec.MAX_EXPRESSION_DEPTH) return null
                    stack.addLast(char)
                }
                ')', ']', '}' -> {
                    val expected = when (stack.peekLast()) { '(' -> ')'; '[' -> ']'; '{' -> '}'; else -> null }
                    if (char != expected) return null // No identity inventory is reliable after a scope mismatch.
                    stack.removeLast()
                }
            }
        }
        return result.takeIf { stack.isEmpty() }
    }

    /** Pair-preserving literal reader used only inside a declared Table call, never state. */
    private class Reader(private val source: String) {
        private var index = 0
        private var nodes = 0
        private fun whitespace() { while (index < source.length && source[index].isWhitespace()) index++ }
        fun at(char: Char): Boolean { whitespace(); return source.getOrNull(index) == char }
        fun character(char: Char) { require(at(char)); index++ }
        fun complete() { whitespace(); require(index == source.length) }
        fun name(): String {
            whitespace()
            val start = index
            require(source.getOrNull(index)?.let { it.isLetter() || it == '_' } == true)
            index++
            while (index < source.length && (source[index].isLetterOrDigit() || source[index] == '_')) index++
            return source.substring(start, index)
        }
        fun literal(depth: Int = 0): Literal {
            whitespace()
            require(depth < A2uiExpressCodec.MAX_EXPRESSION_DEPTH && ++nodes <= 4_096)
            return when (source.getOrNull(index)) {
                '[' -> {
                    index++
                    val values = mutableListOf<Literal>()
                    if (!at(']')) while (true) {
                        values += literal(depth + 1)
                        if (!at(',')) break
                        character(',')
                        if (at(']')) break
                    }
                    character(']')
                    Literal.Array(values)
                }
                '{' -> {
                    index++
                    val values = mutableListOf<Pair<String, Literal>>()
                    if (!at('}')) while (true) {
                        whitespace()
                        val key = if (source.getOrNull(index) == '"' || source.startsWith("r\"", index, ignoreCase = true)) {
                            val value = literal(depth + 1) as? Literal.Scalar ?: error("Literal key required.")
                            require(value.value.isJsonPrimitive && value.value.asJsonPrimitive.isString)
                            value.value.asString
                        } else name()
                        character(':')
                        values += key to literal(depth + 1)
                        if (!at(',')) break
                        character(',')
                        if (at('}')) break
                    }
                    character('}')
                    Literal.Members(values)
                }
                else -> {
                    val (value, consumed) = A2uiExpressCodec.literalPrefix(source.substring(index))
                    require(value.isJsonPrimitive || value.isJsonNull)
                    index += consumed
                    Literal.Scalar(value)
                }
            }
        }
    }

    /** Typed complete-call scan; quoted strings/comments cannot open or close structural scopes. */
    private fun callEnd(input: String, start: Int, mask: BooleanArray): Int? {
        val stack = java.util.ArrayDeque<Char>()
        var open = false
        for (index in start until input.length) {
            if (!mask[index]) continue
            when (val char = input[index]) {
                '(', '[', '{' -> {
                    if (stack.size >= A2uiExpressCodec.MAX_EXPRESSION_DEPTH) return null
                    stack.addLast(char); open = true
                }
                ')', ']', '}' -> {
                    val expected = when (stack.peekLast()) { '(' -> ')'; '[' -> ']'; '{' -> '}'; else -> null }
                    if (char != expected) return null
                    stack.removeLast()
                    if (open && stack.isEmpty()) return index + 1
                }
            }
        }
        return null
    }

    private fun outside(input: String): BooleanArray {
        val mask = BooleanArray(input.length) { true }
        var index = 0
        while (index < input.length) {
            val start = index
            val raw = input.startsWith("r\"", index, ignoreCase = true) &&
                (index == 0 || (!input[index - 1].isLetterOrDigit() && input[index - 1] != '_'))
            val quote = index + if (raw) 1 else 0
            if (input.getOrNull(quote) == '"') {
                val delimiter = if (input.startsWith("\"\"\"", quote)) "\"\"\"" else "\""
                index = quote + delimiter.length
                var escaped = false
                while (index < input.length) {
                    if (input.startsWith(delimiter, index) && (raw || !escaped)) {
                        index += delimiter.length; break
                    }
                    val char = input[index++]
                    if (!raw) { if (escaped) escaped = false else if (char == '\\') escaped = true }
                }
                for (i in start until index) mask[i] = false
            } else if (input.startsWith("/*", index)) {
                index = input.indexOf("*/", index + 2).let { if (it < 0) input.length else it + 2 }
                for (i in start until index) mask[i] = false
            } else if (input.startsWith("//", index) || input[index] == '#') {
                index = input.indexOf('\n', index).let { if (it < 0) input.length else it }
                for (i in start until index) mask[i] = false
            } else index++
        }
        return mask
    }
}
