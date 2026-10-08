package com.samsung.genuicraft.sdk.internal.pipeline

/** Scans only the uncommitted suffix. An unfinished token is retained, never repaired. */
internal object A2uiExpressStreamScanner {
    data class Scan(
        val statements: List<String>,
        val nextOffset: Int,
        val tail: String,
        val closed: Boolean,
    )

    fun scan(source: String, offset: Int): Scan {
        val statements = mutableListOf<String>()
        val buffer = StringBuilder()
        val delimiters = java.util.ArrayDeque<Char>()
        var index = offset
        var nextOffset = offset
        var delimiter: String? = null
        var raw = false
        var escaped = false
        fun flush(end: Int) {
            buffer.toString().trim().takeIf(String::isNotEmpty)?.let(statements::add)
            buffer.clear()
            nextOffset = end
        }
        while (index < source.length) {
            val stringDelimiter = delimiter
            if (stringDelimiter != null) {
                if (source.startsWith(stringDelimiter, index) && (raw || !escaped)) {
                    buffer.append(stringDelimiter)
                    index += stringDelimiter.length
                    delimiter = null
                    escaped = false
                } else {
                    val character = source[index++]
                    buffer.append(character)
                    if (!raw) {
                        if (escaped) escaped = false else if (character == '\\') escaped = true
                    }
                }
                continue
            }
            when {
                source.startsWith("</a2ui>", index) -> {
                    require(delimiters.isEmpty()) { "Document closed inside an unfinished expression." }
                    flush(index)
                    require(source.substring(index + 7).isBlank()) { "Trailing Express content." }
                    return Scan(statements, source.length, "", true)
                }
                source.startsWith("<a2ui>", index) -> error("A second Express document is not allowed.")
                source[index] == '<' && "</a2ui>".startsWith(source.substring(index)) -> break
                source.startsWith("/*", index) -> {
                    val end = source.indexOf("*/", index + 2)
                    if (end < 0) break
                    index = end + 2
                }
                source.startsWith("//", index) || source[index] == '#' -> {
                    val end = source.indexOf('\n', index)
                    if (end < 0) break
                    index = end
                }
                source.startsWith("r\"", index, ignoreCase = true) || source[index] == '"' -> {
                    raw = source[index] != '"'
                    if (raw) buffer.append(source[index++])
                    delimiter = if (source.startsWith("\"\"\"", index)) "\"\"\"" else "\""
                    buffer.append(delimiter)
                    index += delimiter.length
                    escaped = false
                }
                source[index] in "([{" -> {
                    require(delimiters.size < A2uiExpressCodec.MAX_EXPRESSION_DEPTH) { "Express nesting limit." }
                    delimiters.addLast(source[index])
                    buffer.append(source[index++])
                }
                source[index] in ")]}" -> {
                    val expected = when (source[index]) { ')' -> '('; ']' -> '['; else -> '{' }
                    require(delimiters.pollLast() == expected) { "Unbalanced Express delimiter." }
                    buffer.append(source[index++])
                }
                source[index] in "\n;" && delimiters.isEmpty() -> {
                    index += 1
                    flush(index)
                }
                else -> buffer.append(source[index++])
            }
        }
        return Scan(statements, nextOffset, buffer.toString(), false)
    }
}
