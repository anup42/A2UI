package com.samsung.genuicraft.sdk.provider

import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec

internal data class GenUiRepetitionStop(
    val repeatLimit: Int,
    val detail: String,
)

/**
 * Stops actual structural decode repetition, rather than ascending unique ids.
 * Quoted literals/comments are opaque; state rows retain their valid multiplicity.
 * Native token/deadline and terminal compiler size limits remain independent.
 */
internal class GenUiOutputRepetitionGuard(
    private val repeatLimit: Int = DEFAULT_REPEAT_LIMIT,
) {
    init { require(repeatLimit >= 2) { "repeatLimit must be at least 2." } }

    fun inspect(cumulativeText: String): GenUiRepetitionStop? {
        // Scan only the compiler-permitted prefix; oversized input still fails terminal compile.
        val raw = cumulativeText.take(A2uiExpressCodec.MAX_INPUT_CHARS)
        val code = outsideLiterals(raw)
        if (code.contains("</a2ui>", ignoreCase = true)) return null
        referenceListStart.findAll(code).toList().asReversed().forEach { match ->
            val references = parseReferenceList(raw, code, match.range.last + 1)
            repeatedReference(references.orEmpty())?.let { return it }
        }
        return repeatedDefinition(code) ?: repeatedTopLevelContent(code)
    }

    private fun repeatedReference(references: List<String>): GenUiRepetitionStop? {
        val counts = mutableMapOf<String, Int>()
        references.forEach { reference ->
            val count = (counts[reference] ?: 0) + 1
            counts[reference] = count
            if (count >= repeatLimit) return GenUiRepetitionStop(
                repeatLimit,
                "A2UI child reference '$reference' was emitted $repeatLimit times in one component list.",
            )
        }
        return null
    }

    private fun repeatedDefinition(code: String): GenUiRepetitionStop? {
        val counts = mutableMapOf<String, Int>()
        definitionStart.findAll(code).forEach { match ->
            val id = match.groupValues[1]
            val count = (counts[id] ?: 0) + 1
            counts[id] = count
            if (count >= repeatLimit) return GenUiRepetitionStop(
                repeatLimit, "A2UI component definition '$id' was emitted $repeatLimit times.",
            )
        }
        return null
    }

    /** Mask while retaining offsets/newlines, so apparent source code cannot become structure. */
    private fun outsideLiterals(text: String): String {
        val result = text.toCharArray()
        var delimiter: String? = null
        var rawString = false
        var escaped = false
        var index = 0
        fun mask(start: Int, end: Int) {
            for (cursor in start until end) {
                if (text[cursor] != '\n' && text[cursor] != '\r') result[cursor] = ' '
            }
        }
        while (index < text.length) {
            val closing = delimiter
            if (closing != null) {
                if (text.startsWith(closing, index) && (rawString || !escaped)) {
                    mask(index, index + closing.length)
                    index += closing.length
                    delimiter = null
                    rawString = false
                    escaped = false
                } else {
                    val char = text[index]
                    mask(index, index + 1)
                    if (!rawString) {
                        if (escaped) escaped = false
                        else if (char == '\\') escaped = true
                    }
                    index++
                }
                continue
            }
            val commentEnd = commentEnd(text, index)
            if (commentEnd != null) {
                mask(index, commentEnd)
                index = commentEnd
            } else if (isStringStart(text, index)) {
                rawString = text[index] != '"'
                val quoteIndex = index + if (rawString) 1 else 0
                val opening = if (text.startsWith("\"\"\"", quoteIndex)) "\"\"\"" else "\""
                delimiter = opening
                val end = quoteIndex + opening.length
                mask(index, end)
                index = end
            } else {
                index++
            }
        }
        return String(result)
    }

    private fun isStringStart(text: String, index: Int): Boolean =
        text.getOrNull(index) == '"' ||
            (text.getOrNull(index)?.lowercaseChar() == 'r' && text.getOrNull(index + 1) == '"')

    /** Matches compiler trivia, including an unfinished comment in a streaming prefix. */
    private fun commentEnd(text: String, index: Int): Int? = when {
        text.startsWith("/*", index) -> text.indexOf("*/", index + 2).let { if (it < 0) text.length else it + 2 }
        text.startsWith("//", index) || text.getOrNull(index) == '#' ->
            text.indexOf('\n', index).let { if (it < 0) text.length else it }
        else -> null
    }

    private fun skipTrivia(text: String, start: Int): Int {
        var index = start
        while (index < text.length) {
            if (text[index].isWhitespace()) index++
            else index = commentEnd(text, index) ?: return index
        }
        return index
    }

    /** Only bare top-level output is content here; values inside calls/state stay opaque. */
    private fun repeatedTopLevelContent(code: String): GenUiRepetitionStop? {
        val completeLines = mutableMapOf<String, Int>()
        var depth = 0
        var lineDepth = 0
        var lineStart = 0
        var lastWord: String? = null
        var wordCount = 0
        var lastAlphabeticValue: Long? = null
        var alphabeticRun = 0
        var alphabeticStart: String? = null
        var index = 0
        while (index < code.length) {
            val char = code[index]
            if (code.startsWith("<a2ui>", index, ignoreCase = true)) {
                index += "<a2ui>".length
                lastWord = null
                wordCount = 0
                lastAlphabeticValue = null
                alphabeticRun = 0
                continue
            }
            if (depth == 0 && (char.isLetter() || char == '_')) {
                val start = index++
                while (index < code.length && (code[index].isLetterOrDigit() || code[index] == '_')) index++
                val word = code.substring(start, index)
                var next = index
                while (next < code.length && code[next].isWhitespace()) next++
                if (code.getOrNull(next) !in setOf('=', '(')) {
                    wordCount = if (word == lastWord) wordCount + 1 else 1
                    lastWord = word
                    if (wordCount >= repeatLimit) return GenUiRepetitionStop(
                        repeatLimit, "A2UI bare top-level content '$word' repeated $repeatLimit times.",
                    )
                    val current = alphabeticIdValue(word)
                    alphabeticRun = if (current != null && lastAlphabeticValue != null &&
                        current == lastAlphabeticValue!! + 1L) alphabeticRun + 1 else if (current != null) 1 else 0
                    if (alphabeticRun == 1) alphabeticStart = word
                    lastAlphabeticValue = current
                    if (alphabeticRun >= repeatLimit) return GenUiRepetitionStop(
                        repeatLimit,
                        "A2UI bare top-level identifiers '$alphabeticStart' through '$word' continued for $repeatLimit sequential words.",
                    )
                } else {
                    lastWord = null
                    wordCount = 0
                    lastAlphabeticValue = null
                    alphabeticRun = 0
                }
                continue
            }
            when (char) {
                '(', '[', '{' -> { depth++; lastWord = null; wordCount = 0; lastAlphabeticValue = null; alphabeticRun = 0 }
                ')', ']', '}' -> { depth = (depth - 1).coerceAtLeast(0); lastWord = null; wordCount = 0; lastAlphabeticValue = null; alphabeticRun = 0 }
                '\n' -> {
                    if (lineDepth == 0 && depth == 0) {
                        val line = code.substring(lineStart, index).trim().replace(Regex("\\s+"), " ")
                        if (line.isNotEmpty() && line != "<a2ui>" && !line.startsWith('$')) {
                            val count = (completeLines[line] ?: 0) + 1
                            completeLines[line] = count
                            if (count >= repeatLimit) return GenUiRepetitionStop(
                                repeatLimit, "A2UI top-level output fragment repeated $repeatLimit times: ${line.take(80)}",
                            )
                        }
                    }
                    lineStart = index + 1
                    lineDepth = depth
                }
            }
            index++
        }
        return null
    }

    private fun alphabeticIdValue(raw: String): Long? {
        val value = raw.lowercase(java.util.Locale.ROOT)
        if (value.isEmpty() || value.length > 6 || value.any { it !in 'a'..'z' }) return null
        var result = 0L
        value.forEach { result = result * 26L + (it - 'a' + 1) }
        return result
    }

    /** Real child arrays accept bare/string ids and inline calls; source strings stay opaque. */
    private fun parseReferenceList(text: String, code: String, contentStart: Int): List<String> {
        val references = mutableListOf<String>()
        var index = contentStart
        var expectingReference = true
        while (index < text.length) {
            index = skipTrivia(text, index)
            if (index >= text.length) break
            if (isStringStart(text, index)) {
                if (!expectingReference) return references
                val parsed = runCatching { A2uiExpressCodec.literalPrefix(text.substring(index)) }.getOrNull()
                    ?: return references
                val value = parsed.first
                if (!value.isJsonPrimitive || !value.asJsonPrimitive.isString || value.asString.isBlank()) return references
                references += value.asString
                index += parsed.second
                expectingReference = false
                continue
            }
            when (text[index]) {
                ']' -> return references
                ',' -> {
                    if (expectingReference) return references
                    expectingReference = true
                    index++
                }
                else -> {
                    if (!expectingReference) return references
                    if (text[index] == '@') index++
                    if (index >= text.length || !(text[index].isLetter() || text[index] == '_')) return references
                    val start = index++
                    while (index < text.length && (text[index].isLetterOrDigit() || text[index] == '_')) index++
                    val identifier = text.substring(start, index)
                    val next = skipTrivia(text, index)
                    if (text.getOrNull(next) == '(') {
                        // Skip only the inline call, using the already-masked code for balance.
                        var depth = 0
                        var cursor = next
                        while (cursor < code.length) {
                            if (code[cursor] == '(') depth++
                            if (code[cursor] == ')') {
                                depth--
                                if (depth == 0) break
                            }
                            cursor++
                        }
                        if (cursor >= code.length) return references
                        index = cursor + 1
                    } else references += identifier
                    expectingReference = false
                }
            }
        }
        return references
    }

    companion object {
        const val DEFAULT_REPEAT_LIMIT = 20
        private val referenceListStart = Regex(
            """(?i)(?:\bchildren\s*=\s*|\b(?:column|row|stack|card|grid|section)\s*\(\s*)\[""",
        )
        private val definitionStart = Regex("""(?m)(?:^|;)[\t ]*([A-Za-z_][A-Za-z0-9_]*)[\t ]*=[\t ]*[A-Za-z_][A-Za-z0-9_]*[\t ]*\(""")
    }
}
