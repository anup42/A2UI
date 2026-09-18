package com.samsung.genuicraft.sdk.internal.renderer.native

import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.isSpecified
import androidx.compose.ui.unit.sp
import java.nio.charset.Charset
import java.time.LocalDate
import java.time.LocalTime
import java.time.format.DateTimeFormatter
import java.util.Locale

internal object NativeTextFormatter {
    private val INLINE_LABEL_REGEX =
        Regex("""(?:^|(?<=[.!?)]\s)|(?<=[\u2022\-]\s))([\p{L}\p{N}][\p{L}\p{N}/&()' -]{0,84}?)([:;\uFF1A\uFF1B])""")
    private val LEADING_LABEL_REGEX =
        Regex("""^\s*([\u2022\-*]\s*)?([^:;\uFF1A\uFF1B\n]{1,70}?)([:;\uFF1A\uFF1B])\s*(.+)$""")
    private val MARKDOWN_HEADING_LINE_REGEX = Regex("""^#{1,6}\s+(.+)$""")
    private val MARKDOWN_NUMBERED_HEADING_LINE_REGEX = Regex("""^(\d+)[.)]\s+(.+)$""")
    private val URL_REGEX = Regex(
        """(?i)(?:https?://|//)[^\s<>\]]+|(?<![@\w])(?:www\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:[/?#][^\s<>\]]*)?"""
    )
    private val ISO_DATE_TIME_REGEX =
        Regex("""\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2})?(?:\.\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?)?""")
    private val ISO_TIME_PREFIX_REGEX = Regex("""[T\s](\d{2}):(\d{2})""")
    private val READABLE_DATE_FORMATTER = DateTimeFormatter.ofPattern("MMM d", Locale.US)
    private val READABLE_TIME_FORMATTER = DateTimeFormatter.ofPattern("h:mm a", Locale.US)

    fun containsMarkdownInlineFormatting(text: String): Boolean {
        return text.contains("**") ||
            text.contains("__") ||
            text.contains('`') ||
            Regex("""(^|[^\p{L}\p{N}])_[^_\n]+_($|[^\p{L}\p{N}])""").containsMatchIn(text)
    }

    fun containsMarkdownHeading(text: String): Boolean =
        text.lineSequence().any { line ->
            val trimmed = line.trimStart()
            MARKDOWN_HEADING_LINE_REGEX.matches(trimmed) || isLikelyMarkdownNumberedHeading(trimmed)
        }

    fun parseMarkdownWithHeadings(
        text: String,
        baseStyle: TextStyle
    ): AnnotatedString {
        return buildAnnotatedString {
            val lines = text.split('\n')
            lines.forEachIndexed { index, rawLine ->
                val trimmedStart = rawLine.trimStart()
                val hashHeading = MARKDOWN_HEADING_LINE_REGEX.find(trimmedStart)
                val numberedHeading = if (hashHeading == null && isLikelyMarkdownNumberedHeading(trimmedStart)) {
                    MARKDOWN_NUMBERED_HEADING_LINE_REGEX.find(trimmedStart)
                } else {
                    null
                }

                val headingText = when {
                    hashHeading != null -> hashHeading.groupValues[1].trim()
                    numberedHeading != null -> numberedHeading.groupValues[2].trim()
                    else -> ""
                }

                if (headingText.isNotEmpty()) {
                    val headingLevel = when {
                        hashHeading != null -> trimmedStart.takeWhile { it == '#' }.length.coerceIn(1, 6)
                        else -> 3
                    }
                    val start = length
                    append(parseInlineMarkdown(headingText))
                    addStyle(
                        style = SpanStyle(
                            fontWeight = FontWeight.W700,
                            fontSize = markdownHeadingFontSize(baseStyle.fontSize, headingLevel)
                        ),
                        start = start,
                        end = length
                    )
                } else {
                    append(parseInlineMarkdown(rawLine))
                }
                if (index != lines.lastIndex) {
                    append('\n')
                }
            }
        }
    }

    fun markdownHeadingFontSize(baseSize: TextUnit, level: Int): TextUnit {
        val scale = when (level.coerceIn(1, 6)) {
            1 -> 1.30f
            2 -> 1.24f
            3 -> 1.18f
            4 -> 1.14f
            5 -> 1.10f
            else -> 1.08f
        }
        if (!baseSize.isSpecified) {
            return 20.sp * scale
        }
        return baseSize * scale
    }

    fun isLikelyMarkdownNumberedHeading(line: String): Boolean {
        val match = MARKDOWN_NUMBERED_HEADING_LINE_REGEX.matchEntire(line.trim()) ?: return false
        val content = match.groupValues[2].trim()
        if (content.isBlank()) {
            return false
        }
        if (content.length > 72 || content.contains("|")) {
            return false
        }
        if (content.endsWith(".") || content.endsWith("?") || content.endsWith("!")) {
            return false
        }
        val wordCount = content.split(Regex("""\s+""")).count { it.isNotBlank() }
        return wordCount in 1..10
    }

    fun parseInlineMarkdown(text: String): AnnotatedString {
        val markdownParsed = buildAnnotatedString {
            var cursor = 0
            while (cursor < text.length) {
                if (text[cursor] == '\\' && cursor + 1 < text.length) {
                    val escaped = text[cursor + 1]
                    if (escaped == '*' || escaped == '_' || escaped == '`' || escaped == '\\') {
                        append(escaped)
                        cursor += 2
                        continue
                    }
                }

                if (text.startsWith("**", cursor) || text.startsWith("__", cursor)) {
                    val marker = if (text.startsWith("**", cursor)) "**" else "__"
                    val close = findBalancedMarkerEnd(text, cursor, marker)
                    if (close > cursor + marker.length) {
                        val content = text.substring(cursor + marker.length, close)
                        pushStyle(SpanStyle(fontWeight = FontWeight.W700))
                        append(content)
                        pop()
                        cursor = close + marker.length
                        continue
                    }
                    append(marker)
                    cursor += marker.length
                    continue
                }

                if (text[cursor] == '`') {
                    val close = findBalancedMarkerEnd(text, cursor, "`")
                    if (close > cursor + 1) {
                        val content = text.substring(cursor + 1, close)
                        pushStyle(
                            SpanStyle(
                                fontFamily = FontFamily.Monospace,
                                fontWeight = FontWeight.Medium
                            )
                        )
                        append(content)
                        pop()
                        cursor = close + 1
                        continue
                    }
                    append('`')
                    cursor += 1
                    continue
                }

                if (text[cursor] == '_') {
                    val close = findItalicUnderscoreEnd(text, cursor)
                    if (close > cursor + 1) {
                        val content = text.substring(cursor + 1, close)
                        pushStyle(SpanStyle(fontStyle = FontStyle.Italic))
                        append(content)
                        pop()
                        cursor = close + 1
                        continue
                    }
                }

                append(text[cursor])
                cursor += 1
            }
        }

        val labelRanges = findLeadingLabelRanges(markdownParsed.text)
        if (labelRanges.isEmpty()) {
            return markdownParsed
        }

        return buildAnnotatedString {
            append(markdownParsed)
            labelRanges.forEach { range ->
                addStyle(
                    style = SpanStyle(fontWeight = FontWeight.W700),
                    start = range.first,
                    end = range.last + 1
                )
            }
        }
    }

    fun sanitizeDisplayText(text: String, preserveMarkdown: Boolean = false): String {
        // A source-backed UI must retain caveats, timestamps and ordinary words. The reference
        // demo removed example/as-of text and shortened ISO dates; those edits lose source facts.
        if (preserveMarkdown || text.isBlank()) return text
        return text.replace(Regex("(?m)^\\s{0,3}#{1,6}\\s*"), "")
            .replace("**", "")
            .trim()
    }

    fun normalizeMojibakeText(value: String): String {
        if (value.isBlank()) {
            return value
        }
        val candidates = linkedSetOf(value)
        var frontier = listOf(value)
        repeat(2) {
            frontier = frontier.flatMap { candidate ->
                if (!hasMojibakeMarker(candidate)) {
                    emptyList()
                } else {
                    listOfNotNull(
                        decodeMojibakeCandidate(candidate, Charset.forName("windows-1252")),
                        decodeMojibakeCandidate(candidate, Charsets.ISO_8859_1)
                    )
                }
            }.distinct()
            candidates += frontier
        }
        val normalized = candidates.minByOrNull(::mojibakeScore) ?: value
        return normalized
            .replace("\u00C2\u00B0", "\u00B0")
            .replace("\u00C2\u00A0", " ")
            .replace("\u00E2\u0082\u00AC", "\u20AC")
            .replace("\u00E2\u0082\u00B9", "\u20B9")
            .replace("\u00E2\u0080\u00A2", "\u2022")
            .replace("\u00E2\u0080\u0093", "\u2013")
            .replace("\u00E2\u0080\u0094", "\u2014")
            .replace("\u00E2\u0080\u0098", "\u2018")
            .replace("\u00E2\u0080\u0099", "\u2019")
            .replace("\u00E2\u0080\u009C", "\u201C")
            .replace("\u00E2\u0080\u009D", "\u201D")
            .replace("\u013D", "\u00A5")
    }

    fun isLikelyLeadingLabel(label: String, value: String): Boolean {
        if (label.isBlank() || value.isBlank()) {
            return false
        }
        val normalized = label.trim()
        val lower = normalized.lowercase(Locale.US)
        if (lower in setOf("http", "https", "file", "content")) {
            return false
        }
        if (!normalized.any { it.isLetter() }) {
            return false
        }
        if (normalized.length > 60) {
            return false
        }
        return true
    }

    fun parseLeadingLabelValue(text: String): LeadingLabelValue? {
        val match = LEADING_LABEL_REGEX.find(text) ?: return null
        val prefix = match.groups[1]?.value.orEmpty()
        val label = match.groups[2]?.value?.trim().orEmpty()
        val delimiter = match.groups[3]?.value.orEmpty()
        val value = match.groups[4]?.value?.trim().orEmpty()
        if (!isLikelyLeadingLabel(label, value)) {
            return null
        }
        return LeadingLabelValue(
            prefix = prefix,
            label = label,
            delimiter = delimiter,
            value = value
        )
    }

    fun containsUrlLikeToken(value: String): Boolean = URL_REGEX.containsMatchIn(value)

    private fun formatReadableIsoDateTimes(text: String): String {
        if (!ISO_DATE_TIME_REGEX.containsMatchIn(text)) {
            return text
        }
        val urlRanges = URL_REGEX.findAll(text).map { match -> match.range }.toList()
        return ISO_DATE_TIME_REGEX.replace(text) { match ->
            if (!shouldFormatIsoDateMatch(text, match.range, urlRanges)) {
                return@replace match.value
            }
            val date = runCatching { LocalDate.parse(match.value.take(10)) }.getOrNull()
                ?: return@replace match.value
            val dateText = date.format(READABLE_DATE_FORMATTER)
            val timeMatch = ISO_TIME_PREFIX_REGEX.find(match.value)
            if (timeMatch == null) {
                dateText
            } else {
                val hour = timeMatch.groupValues[1].toIntOrNull()
                val minute = timeMatch.groupValues[2].toIntOrNull()
                val timeText = if (hour != null && minute != null) {
                    runCatching { LocalTime.of(hour, minute).format(READABLE_TIME_FORMATTER) }.getOrNull()
                } else {
                    null
                }
                if (timeText == null) dateText else "$dateText, $timeText"
            }
        }
    }

    private fun shouldFormatIsoDateMatch(
        text: String,
        range: IntRange,
        urlRanges: List<IntRange>
    ): Boolean {
        if (urlRanges.any { urlRange -> range.first <= urlRange.last && range.last >= urlRange.first }) {
            return false
        }
        val previous = text.getOrNull(range.first - 1)
        val next = text.getOrNull(range.last + 1)
        if (previous != null && (previous.isLetterOrDigit() || previous == '_' || previous == '/' || previous == '\\' || previous == '.')) {
            return false
        }
        if (next != null && (next.isLetterOrDigit() || next == '_' || next == '/' || next == '\\')) {
            return false
        }
        return true
    }

    private fun findBalancedMarkerEnd(text: String, start: Int, marker: String): Int {
        var search = start + marker.length
        while (search < text.length) {
            val idx = text.indexOf(marker, search)
            if (idx < 0) {
                return -1
            }
            if (idx > start + marker.length) {
                val content = text.substring(start + marker.length, idx)
                if (content.any { !it.isWhitespace() }) {
                    return idx
                }
            }
            search = idx + marker.length
        }
        return -1
    }

    private fun findItalicUnderscoreEnd(text: String, start: Int): Int {
        if (start < 0 || start >= text.length || text[start] != '_') {
            return -1
        }
        if (start > 0 && text[start - 1].isLetterOrDigit()) {
            return -1
        }

        var search = start + 1
        while (search < text.length) {
            val idx = text.indexOf('_', search)
            if (idx < 0) {
                return -1
            }
            if (idx <= start + 1) {
                search = idx + 1
                continue
            }
            if (idx + 1 < text.length && text[idx + 1].isLetterOrDigit()) {
                search = idx + 1
                continue
            }
            val content = text.substring(start + 1, idx)
            if (content.any { !it.isWhitespace() }) {
                return idx
            }
            search = idx + 1
        }
        return -1
    }

    private fun findLeadingLabelRanges(text: String): List<IntRange> {
        val ranges = mutableListOf<IntRange>()
        var offset = 0
        text.split('\n').forEach { line ->
            val trimmed = line.trim()
            if (trimmed.isNotEmpty() && !containsUrlLikeToken(trimmed)) {
                LEADING_LABEL_REGEX.find(line)?.let { match ->
                    val label = match.groups[2]?.value?.trim().orEmpty()
                    val valueText = match.groups[4]?.value?.trim().orEmpty()
                    val labelStart = match.groups[2]?.range?.first ?: match.range.first
                    val delimiterEnd = match.groups[3]?.range?.last ?: match.range.last
                    if (isLikelyLeadingLabel(label, valueText)) {
                        ranges += (offset + labelStart)..(offset + delimiterEnd)
                    }
                }
                INLINE_LABEL_REGEX.findAll(line).forEach { match ->
                    val label = match.groups[1]?.value?.trim().orEmpty()
                    val matchStart = match.range.first
                    val valueText = line.substring(match.range.last + 1).trimStart()
                    val range = (offset + match.range.first)..(offset + match.range.last)
                    if (valueText.isNotEmpty() &&
                        isLikelyLeadingLabel(label, valueText) &&
                        isSentenceBoundary(line, matchStart) &&
                        ranges.none { existing -> range.first <= existing.last && range.last >= existing.first }
                    ) {
                        ranges += range
                    }
                }
            }
            offset += line.length + 1
        }
        return ranges
    }

    private fun isSentenceBoundary(line: String, start: Int): Boolean {
        if (start <= 0) {
            return true
        }
        val prev = line[start - 1]
        if (prev == '\u2022' || prev == '-') {
            return true
        }
        if (prev != ' ') {
            return false
        }
        val prevNonSpaceIndex = line.substring(0, start - 1).indexOfLast { !it.isWhitespace() }
        if (prevNonSpaceIndex < 0) {
            return true
        }
        return line[prevNonSpaceIndex] in charArrayOf('.', '!', '?', '\u2022', '-', ')')
    }

    private fun mojibakeScore(value: String): Int {
        if (value.isBlank()) {
            return 0
        }
        val markers = listOf(
            "\u00C3\u0192",
            "\u00C3\u201A",
            "\u00C3",
            "\u00C2",
            "\u00E2\u0080\u00A2",
            "\u00E2\u0082\u00AC",
            "\u00E2\u0082\u00B9",
            "\u00E2\u0080\u0093",
            "\u00E2\u0080\u0094",
            "\u00E2\u0080\u0098",
            "\u00E2\u0080\u0099",
            "\u00E2\u0080\u009C",
            "\u00E2\u0080\u009D",
            "\u00EF\u00BF\u00BD",
            "\uFFFD"
        )
        return markers.sumOf { marker -> countOccurrences(value, marker) }
    }

    private fun hasMojibakeMarker(value: String): Boolean = mojibakeScore(value) > 0

    private fun decodeMojibakeCandidate(value: String, charset: Charset): String? =
        runCatching {
            String(value.toByteArray(charset), Charsets.UTF_8)
        }.getOrNull()

    private fun countOccurrences(value: String, needle: String): Int {
        if (needle.isEmpty()) {
            return 0
        }
        var count = 0
        var cursor = 0
        while (true) {
            val index = value.indexOf(needle, cursor)
            if (index < 0) {
                break
            }
            count++
            cursor = index + needle.length
        }
        return count
    }
}
