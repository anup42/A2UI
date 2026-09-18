package com.samsung.genuicraft.sdk.internal.renderer.native.parser

import com.samsung.genuicraft.sdk.internal.renderer.native.LeadingLabelValue

internal object NativeBlockHeuristics {
    fun isPlaceholderListEntry(
        value: String,
        parseLeadingLabelValue: (String) -> LeadingLabelValue?,
        isPlaceholderTableCellValue: (String) -> Boolean
    ): Boolean {
        val normalized = value
            .replace(Regex("""^[\u2022•\-*]\s*"""), "")
            .trim()
        if (isPlaceholderTableCellValue(normalized)) {
            return true
        }
        val leadingLabel = parseLeadingLabelValue(normalized) ?: return false
        return isPlaceholderTableCellValue(leadingLabel.value)
    }

    fun looksLikeSectionHeading(
        line: String,
        parseLeadingLabelValue: (String) -> LeadingLabelValue?,
        isBulletListLine: (String) -> Boolean,
        containsUrlLikeToken: (String) -> Boolean
    ): Boolean {
        if (line.length > 80) {
            return false
        }
        if (Regex("""^day\s+\d+\s*:""", RegexOption.IGNORE_CASE).containsMatchIn(line)) {
            return true
        }
        if (parseLeadingLabelValue(line) != null) {
            return false
        }
        if (isBulletListLine(line) || line.endsWith(".") || line.endsWith("?") || line.endsWith("!")) {
            return false
        }
        if (line.contains('|') || containsUrlLikeToken(line)) {
            return false
        }
        if (line.contains("[Button:", ignoreCase = true)) {
            return false
        }
        return line.any { it.isLetter() }
    }

    fun looksLikeStandaloneLinkLine(
        value: String,
        containsUrlLikeToken: (String) -> Boolean,
        isInlineMediaLine: (String) -> Boolean,
        isMediaMarkerHeading: (String) -> Boolean,
        splitTableCells: (String) -> List<String>
    ): Boolean {
        val line = value.trim()
        if (line.isBlank() || !containsUrlLikeToken(line)) {
            return false
        }
        if (isInlineMediaLine(line) || isMediaMarkerHeading(line)) {
            return false
        }
        if (line.length > 260) {
            return false
        }
        if (line.contains('|') && splitTableCells(line).size >= 3) {
            return false
        }
        return true
    }

    fun isStructuredBoundary(
        line: String,
        isBulletListLine: (String) -> Boolean,
        isInlineMediaLine: (String) -> Boolean,
        isMediaMarkerHeading: (String) -> Boolean,
        parseButtonLine: (String) -> Boolean,
        parseOptionLine: (String) -> Boolean,
        isNumberedStepLine: (String) -> Boolean,
        isTableLikeLine: (String) -> Boolean,
        looksLikeSectionHeading: (String) -> Boolean
    ): Boolean {
        if (isBulletListLine(line)) {
            return true
        }
        if (isInlineMediaLine(line) || isMediaMarkerHeading(line)) {
            return true
        }
        if (parseButtonLine(line) || parseOptionLine(line) || isNumberedStepLine(line)) {
            return true
        }
        if (isTableLikeLine(line) || looksLikeSectionHeading(line)) {
            return true
        }
        return false
    }
}
