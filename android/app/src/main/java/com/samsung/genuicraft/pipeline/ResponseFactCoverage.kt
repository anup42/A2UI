package com.samsung.genuicraft.pipeline

import com.google.gson.JsonElement

/**
 * Extracts high-confidence facts from a Stage 2 response and verifies that
 * their values remain in user-visible FlatSpec content. This deliberately
 * ignores URLs and action metadata so a carrier name in a tracking URL, for
 * example, cannot hide an omitted visible carrier field.
 */
internal object ResponseFactCoverage {
    data class Fact(val label: String, val value: String)

    fun missingFacts(sourceResponseText: String, canonicalJson: JsonElement): List<Fact> {
        val facts = extractFacts(sourceResponseText)
        if (facts.isEmpty()) return emptyList()

        val visibleText = collectVisibleStrings(canonicalJson).joinToString(" ")
        val visibleCompact = compact(visibleText)
        val visibleTokens = tokens(visibleText).toSet()
        return facts.filterNot { fact ->
            val factCompact = compact(fact.value)
            val factTokens = significantTokens(fact.value)
            (factCompact.length >= MIN_COMPACT_FACT_LENGTH && visibleCompact.contains(factCompact)) ||
                (factTokens.isNotEmpty() && factTokens.all(visibleTokens::contains))
        }
    }

    fun failureReason(sourceResponseText: String, canonicalJson: JsonElement): String? {
        val missing = missingFacts(sourceResponseText, canonicalJson)
        if (missing.isEmpty()) return null
        return "Stage 3 omitted authoritative response facts: " +
            missing.take(MAX_FACTS_IN_REASON).joinToString("; ") { "${it.label}=${it.value}" }
    }

    private fun extractFacts(text: String): List<Fact> {
        val facts = linkedMapOf<String, Fact>()

        fun addFact(rawLabel: String, rawValue: String) {
            if (facts.size >= MAX_EXTRACTED_FACTS) return
            val label = cleanMarkdown(rawLabel).trim().take(MAX_LABEL_CHARS)
            val value = cleanMarkdown(rawValue).trim().take(MAX_VALUE_CHARS)
            val normalizedLabel = compact(label)
            if (label.isBlank() || value.isBlank() || normalizedLabel in NON_FACT_LABELS) return
            if (value.contains("http://", ignoreCase = true) ||
                value.contains("https://", ignoreCase = true) ||
                value.contains("[Button", ignoreCase = true)
            ) return
            if (significantTokens(value).isEmpty()) return
            facts.putIfAbsent("$normalizedLabel|${compact(value)}", Fact(label, value))
        }

        BOLD_LABEL_VALUE.findAll(text).forEach { match ->
            addFact(match.groupValues[1], match.groupValues[2])
        }
        MARKDOWN_HEADING.findAll(text).forEach { match ->
            addFact("Heading", match.groupValues[1])
        }
        INLINE_BOLD_VALUE.findAll(text).forEach { match ->
            val value = match.groupValues[1].trim()
            if (!value.endsWith(':')) addFact("Emphasized fact", value)
        }
        REFERENCE_IDENTIFIER.findAll(text).forEach { match ->
            val value = match.groupValues[2]
            if (value.any(Char::isDigit)) addFact(match.groupValues[1], value)
        }
        PLAIN_LABEL_VALUE.findAll(text).forEach { match ->
            addFact(match.groupValues[1], match.groupValues[2])
        }
        text.lineSequence().forEach { line ->
            val trimmed = line.trim()
            if (!trimmed.startsWith('|') || !trimmed.endsWith('|')) return@forEach
            val cells = trimmed.split('|').drop(1).dropLast(1).map(::cleanMarkdown)
            if (cells.size < 2 || cells.any { TABLE_SEPARATOR.matches(it.trim()) }) return@forEach
            val first = compact(cells.first())
            val second = compact(cells[1])
            if (first in TABLE_HEADER_LABELS && second in TABLE_HEADER_VALUES) return@forEach
            addFact(cells.first(), cells.drop(1).joinToString(" | "))
        }
        return facts.values.toList()
    }

    private fun collectVisibleStrings(canonicalJson: JsonElement): List<String> {
        if (!canonicalJson.isJsonObject) return emptyList()
        val root = canonicalJson.asJsonObject
        val visible = mutableListOf<String>()
        root.get("state")?.let { collectVisibleValue(it, null, visible) }
        root.get("elements")?.takeIf { it.isJsonObject }?.asJsonObject
            ?.entrySet()
            ?.forEach { (_, rawElement) ->
                rawElement.takeIf { it.isJsonObject }?.asJsonObject
                    ?.get("props")
                    ?.let { collectVisibleValue(it, null, visible) }
            }
        return visible
    }

    private fun collectVisibleValue(element: JsonElement, key: String?, output: MutableList<String>) {
        if (key?.let(::compact) in NON_VISIBLE_PROP_KEYS) return
        when {
            element.isJsonPrimitive && element.asJsonPrimitive.isString -> {
                val value = element.asString.trim()
                if (value.isNotBlank() &&
                    !value.startsWith("http://", ignoreCase = true) &&
                    !value.startsWith("https://", ignoreCase = true)
                ) output += value
            }
            element.isJsonArray -> element.asJsonArray.forEach { collectVisibleValue(it, key, output) }
            element.isJsonObject -> element.asJsonObject.entrySet().forEach { (childKey, child) ->
                collectVisibleValue(child, childKey, output)
            }
        }
    }

    private fun cleanMarkdown(value: String): String = value
        .trim()
        .trim('*', '_', '`', ' ')
        .replace(Regex("\\s+"), " ")

    private fun compact(value: String): String = value
        .lowercase()
        .filter(Char::isLetterOrDigit)

    private fun tokens(value: String): List<String> = TOKEN.findAll(value.lowercase())
        .map { it.value }
        .toList()

    private fun significantTokens(value: String): List<String> = tokens(value)
        .filter { token ->
            token !in FACT_STOP_WORDS && (token.length >= 2 || token.all(Char::isDigit))
        }

    private val BOLD_LABEL_VALUE = Regex(
        "(?m)^\\s*\\*\\*([^*:\\r\\n]{2,80}):\\*\\*\\s*(.+?)\\s*$"
    )
    private val MARKDOWN_HEADING = Regex(
        "(?m)^\\s{0,3}#{1,6}\\s+(.+?)\\s*#*\\s*$"
    )
    private val INLINE_BOLD_VALUE = Regex(
        "\\*\\*([^*\\r\\n]{2,120})\\*\\*"
    )
    private val REFERENCE_IDENTIFIER = Regex(
        "(?i)\\b(order|booking|reservation|confirmation|tracking|reference|case|ticket)\\s*" +
            "(?:number|no\\.?|id)?\\s*(?:[:#-]\\s*)?([a-z0-9][a-z0-9-]{2,})\\b"
    )
    private val PLAIN_LABEL_VALUE = Regex(
        "(?m)^\\s*(?:[-*]\\s*)?([A-Za-z][A-Za-z0-9 /_()'-]{1,60}):\\s*(.+?)\\s*$"
    )
    private val TABLE_SEPARATOR = Regex("^:?-{3,}:?$")
    private val TOKEN = Regex("[\\p{L}\\p{N}]+")
    private val NON_FACT_LABELS = setOf(
        "action", "actions", "media", "image", "icon", "source", "sources", "url", "link", "links",
        "note", "notes", "tag", "tags", "quickaction", "quickactions"
    )
    private val TABLE_HEADER_LABELS = setOf("detail", "field", "label", "attribute", "key")
    private val TABLE_HEADER_VALUES = setOf("value", "details", "description")
    private val NON_VISIBLE_PROP_KEYS = setOf(
        "url", "uri", "src", "href", "action", "params", "statepath", "path", "asset",
        "variant", "direction", "orientation", "domain", "preferredpresentation", "presentation"
    )
    private val FACT_STOP_WORDS = setOf(
        "a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "the", "to", "with", "is", "are"
    )
    private const val MIN_COMPACT_FACT_LENGTH = 3
    private const val MAX_EXTRACTED_FACTS = 24
    private const val MAX_FACTS_IN_REASON = 8
    private const val MAX_LABEL_CHARS = 80
    private const val MAX_VALUE_CHARS = 180
}
