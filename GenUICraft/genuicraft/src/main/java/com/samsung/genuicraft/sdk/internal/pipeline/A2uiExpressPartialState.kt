package com.samsung.genuicraft.sdk.internal.pipeline

import com.google.gson.JsonArray
import com.google.gson.JsonElement

/**
 * Completed rows from an unfinished state literal, restricted to known Table/Chart bindings.
 * Each row crosses the production literal parser and must be followed by ',' or ']'.
 * No component signature, column, visibility, or presentation is inferred from state data.
 */
internal object A2uiExpressPartialState {
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

        fun walk(start: Int, path: String, depth: Int): Int? {
            require(depth < A2uiExpressCodec.MAX_EXPRESSION_DEPTH) { "Express state nesting limit." }
            var index = whitespace(start)
            if (index >= text.length) return null
            if (text[index] == '[' && path in wanted) {
                val rows = JsonArray()
                index = whitespace(index + 1)
                if (text.getOrNull(index) == ']') {
                    arrays[path] = rows
                    return index + 1
                }
                while (index < text.length) {
                    val parsed = literal(index) ?: break
                    val next = whitespace(parsed.second)
                    val boundary = text.getOrNull(next)
                    if (boundary != ',' && boundary != ']') break
                    require(rows.size() < A2uiExpressCodec.MAX_ELEMENTS) { "Streaming row limit." }
                    rows.add(parsed.first)
                    if (boundary == ']') {
                        arrays[path] = rows
                        return next + 1
                    }
                    index = whitespace(next + 1)
                    if (text.getOrNull(index) == ']') {
                        arrays[path] = rows
                        return index + 1
                    }
                }
                if (rows.size() > 0) arrays[path] = rows
                return null
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
