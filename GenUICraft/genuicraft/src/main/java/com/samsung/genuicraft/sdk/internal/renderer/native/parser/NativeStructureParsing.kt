package com.samsung.genuicraft.sdk.internal.renderer.native.parser

import com.samsung.genuicraft.sdk.internal.renderer.native.NativePayloadParser
import com.samsung.genuicraft.sdk.internal.renderer.native.ParsedButton
import com.samsung.genuicraft.sdk.internal.renderer.native.StepEntry

internal object NativeStructureParsing {
    private val OPTION_LINE_REGEX =
        Regex("""^Option\s+\d+\s*:\s*(.+?)\s*\|\s*(.+)$""", RegexOption.IGNORE_CASE)
    private val BULLET_PREFIX_REGEX = Regex("""^\s*[\-\*\u2022]\s+""")
    private val BUTTON_LINE_REGEX =
        Regex("""^(?:Action:\s*)?\[Button:\s*(.+?)\]\s*(.+)$""", RegexOption.IGNORE_CASE)
    private val ACTION_BRACKET_LABEL_REGEX =
        Regex("""^Action:\s*\[(.+?)\]\s*:?\s*(.+)$""", RegexOption.IGNORE_CASE)
    private val BUTTON_MARKDOWN_LINK_REGEX =
        Regex("""^(?:Action:\s*)?\[Button:\s*(.+?)\]\s*\(\s*(.+?)\s*\)\s*$""", RegexOption.IGNORE_CASE)
    private val NUMBERED_STEP_REGEX = Regex("""^(\d+)\.\s+(.+)$""")
    private val BULLET_LINE_REGEX = Regex("""^\s*([\-*\u2022])\s+(.+)$""")
    private val URL_REGEX = Regex(
        """(?i)(?:https?://|//)[^\s<>\]]+|(?<![@\w])(?:www\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:[/?#][^\s<>\]]*)?"""
    )

    fun collectNumberedSteps(
        lines: List<String>,
        startIndex: Int,
        isStructuredBoundary: (String) -> Boolean
    ): Pair<List<StepEntry>, Int>? {
        val firstLine = lines[startIndex].trim()
        if (NUMBERED_STEP_REGEX.matchEntire(firstLine) == null) {
            return null
        }

        val steps = mutableListOf<StepEntry>()
        var cursor = startIndex
        while (cursor < lines.size) {
            while (cursor < lines.size && lines[cursor].trim().isEmpty()) {
                cursor++
            }
            if (cursor >= lines.size) {
                break
            }

            val current = lines[cursor].trim()
            val match = NUMBERED_STEP_REGEX.matchEntire(current) ?: break
            val stepTitle = "${match.groupValues[1]}. ${match.groupValues[2].trim().removeSuffix(":")}"
            cursor++

            val detailLines = mutableListOf<String>()
            while (cursor < lines.size) {
                val candidate = lines[cursor].trim()
                if (candidate.isEmpty()) {
                    cursor++
                    if (detailLines.isNotEmpty()) {
                        break
                    }
                    continue
                }
                if (NUMBERED_STEP_REGEX.matchEntire(candidate) != null || isStructuredBoundary(candidate)) {
                    break
                }
                detailLines += candidate
                cursor++
            }

            steps += StepEntry(
                title = stepTitle,
                details = detailLines.joinToString(" ")
            )
        }

        if (steps.size < 2) {
            return null
        }
        return steps to cursor
    }

    fun collectTableRows(
        lines: List<String>,
        startIndex: Int,
        containsUrlLikeToken: (String) -> Boolean,
        isPlaceholderTableStringRow: (List<String>) -> Boolean
    ): Pair<List<List<String>>, Int>? {
        if (!isTableLikeLine(lines[startIndex].trim(), containsUrlLikeToken)) {
            return null
        }

        val rows = mutableListOf<List<String>>()
        var cursor = startIndex
        while (cursor < lines.size) {
            val candidate = lines[cursor].trim()
            if (!isTableLikeLine(candidate, containsUrlLikeToken)) break
            rows += splitTableCells(candidate)
            cursor++
        }
        val filteredRows = rows.filterIndexed { index, row ->
            index == 0 || !isPlaceholderTableStringRow(row)
        }
        if (filteredRows.size < 2) {
            return null
        }
        return filteredRows to cursor
    }

    fun parseOptionLine(line: String): Pair<String, String>? {
        val match = OPTION_LINE_REGEX.find(line) ?: return null
        val title = match.groupValues[1].trim()
        val details = match.groupValues[2].trim()
        if (title.isEmpty() || details.isEmpty()) {
            return null
        }
        return title to details
    }

    fun parseButtonLine(line: String, sanitizeUrlToken: (String) -> String): ParsedButton? {
        val normalized = BULLET_PREFIX_REGEX.replace(line.trim(), "")
        val markdownLink = BUTTON_MARKDOWN_LINK_REGEX.find(normalized)
        val (label, trailing) = if (markdownLink != null) {
            markdownLink.groupValues[1].trim() to markdownLink.groupValues[2].trim()
        } else {
            val buttonMatch = BUTTON_LINE_REGEX.find(normalized)
            val actionMatch = ACTION_BRACKET_LABEL_REGEX.find(normalized)
            when {
                buttonMatch != null ->
                    buttonMatch.groupValues[1].trim() to buttonMatch.groupValues[2].trim()
                actionMatch != null ->
                    actionMatch.groupValues[1].trim() to actionMatch.groupValues[2].trim()
                else -> return null
            }
        }
        val rawUrlToken = URL_REGEX.find(trailing)?.value ?: trailing
        val url = NativePayloadParser.canonicalizeNetworkUrlToken(sanitizeUrlToken(rawUrlToken))
        if (label.isBlank() || url.isBlank()) {
            return null
        }
        return ParsedButton(label, url)
    }

    fun isTableLikeLine(line: String, containsUrlLikeToken: (String) -> Boolean): Boolean {
        if (containsUrlLikeToken(line)) {
            return false
        }
        return splitTableCells(line).size >= 3
    }

    fun splitTableCells(line: String): List<String> =
        line.split('|').map { it.trim() }.filter { it.isNotEmpty() }

    fun sanitizeUrlToken(value: String): String =
        value.trim().trim('"', '\'', '<', '>', '`').trimEnd('.', ',', ';', ')', ']', '}')

    fun isBulletListLine(line: String): Boolean =
        BULLET_LINE_REGEX.matches(line.trim())

    fun extractBulletListItem(line: String): String? =
        BULLET_LINE_REGEX.matchEntire(line.trim())
            ?.groupValues
            ?.getOrNull(2)
            ?.trim()
            ?.takeIf { it.isNotBlank() }

    fun stripLeadingBulletMarker(line: String): String =
        extractBulletListItem(line) ?: line.trim()

    fun isNumberedStepLine(line: String): Boolean =
        NUMBERED_STEP_REGEX.matchEntire(line) != null
}
