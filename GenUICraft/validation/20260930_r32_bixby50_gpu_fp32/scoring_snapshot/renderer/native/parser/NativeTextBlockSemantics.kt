package com.samsung.genuicraft.renderer.native.parser

import com.samsung.genuicraft.renderer.native.ParsedMediaEntry
import com.samsung.genuicraft.renderer.native.intents.flight.NativeFlightSemantics

internal object NativeTextBlockSemantics {
    fun assignIconsToPrimary(
        primary: List<ParsedMediaEntry>,
        icons: List<ParsedMediaEntry>
    ): List<List<ParsedMediaEntry>> {
        if (primary.isEmpty()) {
            return emptyList()
        }

        val buckets = MutableList(primary.size) { mutableListOf<ParsedMediaEntry>() }
        val unmatchedContextual = mutableListOf<ParsedMediaEntry>()
        val primaryKeys = primary.map { normalizeMatchText(it.label) }

        icons.forEach { icon ->
            val iconKey = normalizeMatchText(icon.label)
            if (iconKey.isBlank()) {
                return@forEach
            }

            val matchedIndex = primaryKeys.indexOfFirst { key ->
                key.isNotBlank() && (key.contains(iconKey) || iconKey.contains(key))
            }
            if (matchedIndex >= 0) {
                buckets[matchedIndex] += icon
                return@forEach
            }

            if (isContextualInlineIconLabel(icon.label)) {
                unmatchedContextual += icon
            }
        }

        unmatchedContextual.forEachIndexed { index, icon ->
            buckets[index % buckets.size] += icon
        }
        return buckets
    }

    fun isBoilerplateContextHeading(text: String): Boolean {
        val normalized = normalizeMatchText(text.removeSuffix(":"))
        if (normalized.isBlank()) {
            return false
        }
        if (normalized in setOf("information context", "context", "background context", "background information")) {
            return true
        }
        if (normalized.contains("information context")) {
            return true
        }
        return normalized.contains("assumption") ||
            normalized.contains("travel planning") ||
            normalized.startsWith("a note") ||
            normalized.contains("background")
    }

    fun isBoilerplateContextParagraph(text: String): Boolean {
        val normalized = normalizeMatchText(text)
        if (text.length < 120) {
            return false
        }
        return normalized.contains("subject to change") ||
            normalized.contains("following table") ||
            normalized.contains("for informational purposes") ||
            normalized.contains("comparison of") ||
            normalized.contains("the following tables")
    }

    private fun isContextualInlineIconLabel(label: String): Boolean {
        val normalized = normalizeMatchText(label)
        return normalized in setOf(
            "airline",
            "time",
            "clock",
            "calendar",
            "date",
            "fire",
            "basket",
            "meal",
            "schedule"
        )
    }

    private fun normalizeMatchText(value: String): String {
        return NativeFlightSemantics.normalizeMatchText(value)
    }
}
