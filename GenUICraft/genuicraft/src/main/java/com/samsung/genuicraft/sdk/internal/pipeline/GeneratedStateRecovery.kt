package com.samsung.genuicraft.sdk.internal.pipeline

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonNull
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.google.gson.JsonPrimitive

/** Reads complete literal values inside damaged generated state; never completes a quoted value. */
internal object GeneratedStateRecovery {
    data class Result(val state: JsonObject, val assignments: Int, val partialAssignments: Int)

    fun recover(statements: List<String>): Result {
        val state = JsonObject()
        var assignments = 0
        var partial = 0
        statements.filter { it.trimStart().startsWith('$') }.forEach { statement ->
            var line = statement.trim()
            line = when {
                line.startsWith("$/{") -> "$/=" + line.removePrefix("$/")
                Regex("^\\$/[A-Za-z0-9_/]+:").containsMatchIn(line) -> line.replaceFirst(":", "=")
                else -> line
            }
            val split = line.indexOf('=')
            if (split < 0) return@forEach
            fun decode(assignment: String) = runCatching {
                A2uiExpressCodec.decode("<a2ui>\n$assignment\nroot=Text(\"state parser sentinel\")\n</a2ui>")
                    .getAsJsonObject("state")
            }.getOrNull()
            val exact = decode(line)
            val parsed = exact ?: LiteralReader(line.substring(split + 1)).read()?.let { value ->
                decode(line.substring(0, split + 1) + value)
            } ?: return@forEach
            if (parsed.size() == 0) return@forEach
            merge(state, parsed)
            assignments += 1
            if (exact == null) partial += 1
        }
        return Result(state, assignments, partial)
    }

    private fun merge(target: JsonObject, source: JsonObject) {
        source.entrySet().forEach { (key, value) ->
            val old = target.get(key)
            if (old?.isJsonObject == true && value.isJsonObject) merge(old.asJsonObject, value.asJsonObject)
            else target.add(key, value.deepCopy())
        }
    }

    private class LiteralReader(private val text: String) {
        private var index = 0
        private var nodes = 0

        fun read(depth: Int = 0): JsonElement? {
            whitespace()
            if (index >= text.length || depth > 64 || ++nodes > 4_096) return null
            return when (text[index]) {
                '{' -> objectValue(depth + 1)
                '[' -> arrayValue(depth + 1)
                '"' -> quoted()?.let(::JsonPrimitive)
                else -> {
                    val start = index
                    while (index < text.length && !text[index].isWhitespace() && text[index] !in ",]}\n") index++
                    val token = text.substring(start, index)
                    when {
                        token == "null" -> JsonNull.INSTANCE
                        token == "true" || token == "false" -> JsonPrimitive(token.toBoolean())
                        Regex("-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?").matches(token) ->
                            runCatching { JsonParser.parseString(token) }.getOrNull()
                        else -> null
                    }
                }
            }
        }

        private fun objectValue(depth: Int): JsonObject? {
            index++
            val value = JsonObject()
            while (index < text.length && nodes <= 4_096) {
                separators()
                if (index >= text.length) break
                if (text[index] == '}') { index++; return value }
                val start = index
                val key = if (text[index] == '"') quoted() else {
                    while (index < text.length && (text[index].isLetterOrDigit() || text[index] == '_')) index++
                    text.substring(start, index).takeIf { it.isNotEmpty() }
                }
                whitespace()
                if (key != null && index < text.length && text[index] == ':') {
                    index++
                    read(depth)?.let { value.add(key, it) }
                }
                if (index == start) index++
                recoverBoundary('}')
            }
            return value.takeIf { it.size() > 0 }
        }

        private fun arrayValue(depth: Int): JsonArray? {
            index++
            val value = JsonArray()
            while (index < text.length && nodes <= 4_096) {
                separators()
                if (index >= text.length) break
                if (text[index] == ']') { index++; return value }
                val start = index
                read(depth)?.let(value::add)
                if (index == start) index++
                recoverBoundary(']')
            }
            return value.takeIf { it.size() > 0 }
        }

        private fun quoted(): String? {
            val start = index++
            var escaped = false
            while (index < text.length) {
                val char = text[index++]
                when {
                    escaped -> escaped = false
                    char == '\\' -> escaped = true
                    char == '"' -> return runCatching {
                        JsonParser.parseString(text.substring(start, index)).asString
                    }.getOrNull()
                }
            }
            return null
        }

        // Skip a damaged member without promoting nested keys/values into its parent scope.
        private fun recoverBoundary(close: Char) {
            whitespace()
            var depth = 0
            while (index < text.length) {
                val char = text[index]
                if (depth == 0 && (char == ',' || char == close)) return
                when (char) {
                    '"' -> { quoted(); continue }
                    '{', '[' -> depth++
                    '}', ']' -> { if (depth == 0) return; depth-- }
                }
                index++
            }
        }

        private fun whitespace() { while (index < text.length && text[index].isWhitespace()) index++ }
        private fun separators() { while (index < text.length && (text[index].isWhitespace() || text[index] == ',')) index++ }
    }
}
