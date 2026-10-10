package com.samsung.genuicraft.sdk.internal.pipeline

import com.google.gson.JsonArray
import com.google.gson.JsonElement

/** Completed rows from unfinished state literals and named inline Table calls. */
internal object A2uiExpressPartialState {
    /**
     * A provisional inline Table built only from closed columns and closed row literals.
     * The returned assignment is parsed again by the production component codec; it is never
     * committed as a model statement. Every other completed named argument is retained verbatim
     * so visibility/repeat guards cannot be bypassed by the provisional projection.
     */
    fun inlineTable(statement: String): String? {
        val (lhs, rhs) = runCatching { A2uiExpressCodec.splitAssignment(statement) }.getOrNull()
            ?: return null
        if (lhs.startsWith('$') || !lhs.matches(Regex("[A-Za-z_][A-Za-z0-9_]*"))) return null
        var index = rhs.indexOfFirst { !it.isWhitespace() }
        if (index < 0 || !rhs.startsWith("Table", index)) return null
        index += "Table".length
        index = rhs.skipWhitespace(index)
        if (rhs.getOrNull(index) != '(') return null
        index += 1
        val reader = Reader(rhs, emptySet())
        val keys = mutableSetOf<String>()
        var columns: JsonArray? = null
        var rows: JsonArray? = null
        val closedArgs = linkedMapOf<String, String>()
        while (index < rhs.length) {
            index = rhs.skipWhitespace(index)
            if (rhs.getOrNull(index) == ')') return null // The complete call takes the normal path.
            val beginning = index
            if (rhs.getOrNull(index)?.isLetter() != true && rhs.getOrNull(index) != '_') break
            index += 1
            while (rhs.getOrNull(index)?.let { it.isLetterOrDigit() || it == '_' } == true) index += 1
            val key = rhs.substring(beginning, index)
            if (!keys.add(key)) return null
            index = rhs.skipWhitespace(index)
            if (rhs.getOrNull(index) != '=') break
            index = rhs.skipWhitespace(index + 1)
            if (key == "rows") {
                val parsed = reader.rows(index, allowClosedCompositeAtEnd = true, compositeOnly = true) ?: break
                if (parsed.values.size() > 0) rows = parsed.values
                index = parsed.end ?: break
            } else {
                val valueStart = index
                val parsed = runCatching { A2uiExpressCodec.literalPrefix(rhs.substring(index)) }.getOrNull() ?: break
                val end = rhs.skipWhitespace(index + parsed.second)
                val boundary = rhs.getOrNull(end)
                if (boundary !in listOf(',', ')', null)) break
                // Numbers and bare scalar prefixes may still grow before a separator arrives.
                if (boundary == null && !rhs.selfDelimitedValue(valueStart)) break
                if (key == "columns") columns = parsed.first.takeIf { it.isJsonArray }?.asJsonArray
                else closedArgs[key] = rhs.substring(valueStart, index + parsed.second).trim()
                index = end
            }
            index = rhs.skipWhitespace(index)
            when (rhs.getOrNull(index)) {
                ',' -> index += 1
                ')' -> return null // Full statement must retain all its generated arguments.
                null -> break
                else -> break
            }
        }
        val readyColumns = columns?.takeIf { it.size() > 0 } ?: return null
        val readyRows = rows?.takeIf { it.size() > 0 } ?: return null
        val knownArgs = closedArgs.entries.joinToString("") { (key, value) -> ",$key=$value" }
        return "$lhs=Table(columns=$readyColumns,rows=$readyRows$knownArgs)"
    }

    fun arrays(statement: String, wantedPaths: Set<String>): Map<String, JsonArray> {
        if (wantedPaths.isEmpty() || !statement.trimStart().startsWith('$')) return emptyMap()
        val (lhs, rhs) = runCatching { A2uiExpressCodec.splitAssignment(statement) }.getOrNull()
            ?: return emptyMap()
        val base = when {
            lhs == "$" || lhs == "$/" -> ""
            lhs.startsWith("$/") -> lhs.removePrefix("$")
            else -> return emptyMap()
        }
        val reader = Reader(rhs, wantedPaths)
        reader.walk(0, base, 0)
        return reader.arrays
    }

    private fun String.skipWhitespace(start: Int): Int {
        var index = start
        while (index < length && this[index].isWhitespace()) index += 1
        return index
    }

    private fun String.selfDelimitedValue(start: Int): Boolean =
        getOrNull(start) in listOf('"', '[', '{') || startsWith("r\"", start, ignoreCase = true)

    private data class Rows(val values: JsonArray, val end: Int?)

    private class Reader(private val text: String, private val wanted: Set<String>) {
        val arrays = linkedMapOf<String, JsonArray>()

        private fun whitespace(start: Int): Int {
            var index = start
            while (index < text.length && text[index].isWhitespace()) index += 1
            return index
        }

        private fun literal(start: Int): Pair<JsonElement, Int>? =
            runCatching { A2uiExpressCodec.literalPrefix(text.substring(start)) }
                .getOrNull()?.let { (value, length) -> value to (start + length) }

        fun rows(start: Int, allowClosedCompositeAtEnd: Boolean = false, compositeOnly: Boolean = false): Rows? {
            if (text.getOrNull(start) != '[') return null
            val values = JsonArray()
            var index = whitespace(start + 1)
            if (text.getOrNull(index) == ']') return Rows(values, index + 1)
            while (index < text.length) {
                val parsed = literal(index) ?: break
                if (compositeOnly && !parsed.first.isJsonArray && !parsed.first.isJsonObject) break
                val next = whitespace(parsed.second)
                val boundary = text.getOrNull(next)
                if (boundary != ',' && boundary != ']') {
                    // A closed composite row cannot gain more cells/fields in a valid continuation.
                    if (allowClosedCompositeAtEnd && boundary == null &&
                        (parsed.first.isJsonArray || parsed.first.isJsonObject)) {
                        require(values.size() < A2uiExpressCodec.MAX_ELEMENTS) { "Streaming row limit." }
                        values.add(parsed.first)
                    }
                    break
                }
                require(values.size() < A2uiExpressCodec.MAX_ELEMENTS) { "Streaming row limit." }
                values.add(parsed.first)
                if (boundary == ']') return Rows(values, next + 1)
                index = whitespace(next + 1)
                if (text.getOrNull(index) == ']') return Rows(values, index + 1)
            }
            return Rows(values, null)
        }

        fun walk(start: Int, path: String, depth: Int): Int? {
            require(depth < A2uiExpressCodec.MAX_EXPRESSION_DEPTH) { "Express state nesting limit." }
            var index = whitespace(start)
            if (index >= text.length) return null
            if (text[index] == '[' && path in wanted) {
                val parsed = rows(index) ?: return null
                if (parsed.values.size() > 0 || parsed.end != null) arrays[path] = parsed.values
                return parsed.end
            }
            if (text[index] != '{') return literal(index)?.second
            index = whitespace(index + 1)
            val keys = mutableSetOf<String>()
            if (text.getOrNull(index) == '}') return index + 1
            while (index < text.length) {
                val key: String
                if (text[index] == '"' || text.startsWith("r\"", index, ignoreCase = true)) {
                    val parsed = literal(index) ?: return null
                    if (!parsed.first.isJsonPrimitive || !parsed.first.asJsonPrimitive.isString) return null
                    key = parsed.first.asString
                    index = whitespace(parsed.second)
                } else {
                    val beginning = index
                    if (!text[index].isLetter() && text[index] != '_') return null
                    index += 1
                    while (index < text.length && (text[index].isLetterOrDigit() || text[index] == '_')) index += 1
                    key = text.substring(beginning, index)
                    index = whitespace(index)
                }
                // A repeated key is a definite error, even while the object is still streaming.
                require(keys.add(key)) { "Duplicate streaming state key '$key'." }
                if (text.getOrNull(index) != ':') return null
                val token = key.replace("~", "~0").replace("/", "~1")
                val end = walk(index + 1, "$path/$token", depth + 1) ?: return null
                index = whitespace(end)
                when (text.getOrNull(index)) {
                    '}' -> return index + 1
                    ',' -> {
                        index = whitespace(index + 1)
                        if (text.getOrNull(index) == '}') return index + 1
                    }
                    else -> return null
                }
            }
            return null
        }
    }
}
