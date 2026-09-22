package com.samsung.genuicraft.sdk.provider

import java.util.Locale

internal data class GenUiRepetitionStop(
    val repeatLimit: Int,
    val detail: String,
)

/**
 * Detects model decode loops inside A2UI component-reference lists.
 *
 * Generated state values are deliberately ignored: repeated table rows and prose can be valid.
 * A list is stopped when one component reference is emitted [repeatLimit] times, or when the
 * model keeps inventing sequential alphabetic ids (a, b, ... or aa, ab, ...) for that many
 * references. Both patterns produce an invalid or impractically large component graph and are
 * safe to hand to the normal generated-output recovery pipeline as a partial document.
 */
internal class GenUiOutputRepetitionGuard(
    private val repeatLimit: Int = DEFAULT_REPEAT_LIMIT,
) {
    init {
        require(repeatLimit >= 2) { "repeatLimit must be at least 2." }
    }

    fun inspect(cumulativeText: String): GenUiRepetitionStop? {
        if (cumulativeText.contains("</a2ui>", ignoreCase = true)) return null
        return referenceListStart.findAll(cumulativeText).toList().asReversed().firstNotNullOfOrNull { match ->
            val references = parseReferenceList(cumulativeText, match.range.last + 1) ?: return@firstNotNullOfOrNull null
            repeatedReference(references) ?: sequentialReferenceRun(references)
        }
    }

    private fun repeatedReference(references: List<String>): GenUiRepetitionStop? {
        val counts = linkedMapOf<String, Int>()
        references.forEach { reference ->
            val count = (counts[reference] ?: 0) + 1
            counts[reference] = count
            if (count >= repeatLimit) {
                return GenUiRepetitionStop(
                    repeatLimit,
                    "A2UI child reference '$reference' was emitted $repeatLimit times in one component list.",
                )
            }
        }
        return null
    }

    private fun sequentialReferenceRun(references: List<String>): GenUiRepetitionStop? {
        var runStart = 0
        var runLength = 1
        for (index in 1 until references.size) {
            val previous = alphabeticIdValue(references[index - 1])
            val current = alphabeticIdValue(references[index])
            if (previous != null && current == previous + 1L) {
                if (runLength == 1) runStart = index - 1
                runLength += 1
                if (runLength >= repeatLimit) {
                    return GenUiRepetitionStop(
                        repeatLimit,
                        "A2UI child identifiers '${references[runStart]}' through '${references[index]}' " +
                            "continued for $repeatLimit sequential references.",
                    )
                }
            } else {
                runLength = 1
                runStart = index
            }
        }
        return null
    }

    private fun alphabeticIdValue(raw: String): Long? {
        val value = raw.lowercase(Locale.ROOT)
        if (value.isEmpty() || value.length > MAX_ALPHABETIC_ID_LENGTH || value.any { it !in 'a'..'z' }) return null
        var result = 0L
        value.forEach { result = result * 26L + (it - 'a' + 1) }
        return result
    }

    /** Returns null when the bracket contains values rather than bare component references. */
    private fun parseReferenceList(text: String, contentStart: Int): List<String>? {
        val references = mutableListOf<String>()
        var index = contentStart
        var expectingReference = true
        while (index < text.length) {
            while (index < text.length && text[index].isWhitespace()) index += 1
            if (index >= text.length) break
            when (text[index]) {
                ']' -> return references
                ',' -> {
                    if (expectingReference) return null
                    expectingReference = true
                    index += 1
                }
                else -> {
                    if (!expectingReference) return null
                    if (text[index] == '@') index += 1
                    if (index >= text.length || !(text[index].isLetter() || text[index] == '_')) return null
                    val start = index
                    index += 1
                    while (index < text.length && (text[index].isLetterOrDigit() || text[index] == '_')) index += 1
                    references += text.substring(start, index)
                    expectingReference = false
                }
            }
        }
        return references
    }

    companion object {
        const val DEFAULT_REPEAT_LIMIT = 20
        private const val MAX_ALPHABETIC_ID_LENGTH = 6
        private val referenceListStart = Regex(
            """(?i)(?:\bchildren\s*=\s*|\b(?:column|row|stack|card|grid|section)\s*\(\s*)\[""",
        )
    }
}
