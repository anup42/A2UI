package com.samsung.genuicraft.sdk.internal.pipeline

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import java.util.Locale

/**
 * Promotes a narrow class of literal, repeated entity lists to the renderer's existing entity cards.
 * This changes presentation only: one authored item becomes one row, in the same order, and every
 * character after the entity delimiter stays in the Details cell. No price, time, or provider fields
 * are inferred from the prose.
 */
internal object EntityListPresentation {
    data class Result(val graph: JsonObject, val convertedLists: Int)

    private const val MAX_ITEMS = 24
    private const val MAX_ITEM_LENGTH = 1_200

    private enum class Delimiter { COLON, DASH, NARRATIVE }
    private data class EntityItem(
        val title: String,
        val details: String,
        val delimiter: Delimiter,
        val leadingAttribute: Boolean,
    )

    // The renderer's two-column entity route requires compact first-column values (<= 44 chars).
    private val boldPrefix = Regex("""^\*\*([^*\r\n]{2,44})\*\*(.*)$""")
    private val attribute = Regex(
        """(?:₹|Rs\.?|INR\b|\$|€|£)\s*\d|\b(?:[01]?\d|2[0-3]):[0-5]\d\b|\b\d+(?:\.\d+)?\s*(?:dB|hours?|mins?|minutes?|GB|km)\b""",
        RegexOption.IGNORE_CASE,
    )
    private val leadingAttribute = Regex(
        """^(?:\*\*)?(?:(?:from|about|around|at|starting(?:\s+at)?|up to)\s+)?(?:(?:₹|Rs\.?|INR\b|\$|€|£)\s*\d|(?:[01]?\d|2[0-3]):[0-5]\d\b|\d+(?:\.\d+)?\s*(?:dB|hours?|mins?|minutes?|GB|km)\b)""",
        RegexOption.IGNORE_CASE,
    )
    private val genericTitleWords = setOf(
        "action", "advice", "airline", "best", "budget", "case", "cheapest", "cost", "costs",
        "date", "day", "example", "fare", "fares", "fastest", "flight", "flights", "friday",
        "highest", "important", "item", "lowest", "monday", "note", "notes", "option", "point",
        "price", "prices", "reminder", "rule", "saturday", "scenario", "step", "steps",
        "sunday", "summary", "task", "thursday", "time", "tip", "tips", "today", "tomorrow",
        "tuesday", "value", "wednesday", "warning",
    )
    private val adviceOpeners = setOf(
        "ask", "avoid", "book", "bring", "buy", "check", "choose", "compare", "confirm",
        "consider", "expect", "get", "keep", "look", "pay", "plan", "prefer", "remember",
        "review", "save", "try", "use", "verify", "wait",
    )

    fun normalize(graph: JsonObject): Result {
        val elements = graph.get("elements")?.takeIf { it.isJsonObject }?.asJsonObject
            ?: return Result(graph, 0)
        var output: JsonObject? = null
        var count = 0
        elements.entrySet().forEach { (id, raw) ->
            if (!raw.isJsonObject) return@forEach
            val replacement = tableFor(raw.asJsonObject) ?: return@forEach
            if (output == null) output = graph.deepCopy()
            output!!.getAsJsonObject("elements").add(id, replacement)
            count += 1
        }
        return Result(output ?: graph, count)
    }

    private fun tableFor(element: JsonObject): JsonObject? {
        if (element.get("type")?.takeIf { it.isJsonPrimitive }?.asString != "List") return null
        // Do not change stateful/interactive/repeated lists or lists with rich children or styling.
        if (element.keySet() != setOf("type", "props", "children")) return null
        val children = element.get("children")?.takeIf { it.isJsonArray }?.asJsonArray ?: return null
        if (children.size() != 0) return null
        val props = element.get("props")?.takeIf { it.isJsonObject }?.asJsonObject ?: return null
        if (props.keySet() != setOf("items")) return null
        val items = props.get("items")?.takeIf { it.isJsonArray }?.asJsonArray ?: return null
        if (items.size() !in 2..MAX_ITEMS) return null
        val parsed = items.map { raw ->
            if (!raw.isJsonPrimitive || !raw.asJsonPrimitive.isString) return null
            val text = raw.asString
            if (text.length !in 12..MAX_ITEM_LENGTH || LiteralTextCodec.decode(text) != null ||
                text.contains("${'$'}{") || text.contains("${'$'}/") || text.contains("${'$'}state.") ||
                text.contains("${'$'}item") || text.contains("${'$'}index")) return null
            parseItem(text) ?: return null
        }
        if (parsed.map { it.title.lowercase(Locale.ROOT) }.distinct().size < 2) return null
        if (parsed.count { it.leadingAttribute } < maxOf(2, (parsed.size + 1) / 2)) return null
        if (parsed.count { it.delimiter == Delimiter.COLON } < 2 &&
            parsed.count { it.delimiter == Delimiter.DASH } < 2) return null
        if (parsed.count { it.delimiter != Delimiter.NARRATIVE } < (parsed.size + 1) / 2) return null

        val rows = JsonArray().apply {
            parsed.forEach { item ->
                add(JsonArray().apply {
                    add(item.title)
                    add(item.details)
                })
            }
        }
        return JsonObject().apply {
            addProperty("type", "Table")
            add("props", JsonObject().apply {
                add("columns", JsonArray().apply { add("Option"); add("Details") })
                add("rows", rows)
                addProperty("domain", "comparison")
                addProperty("preferredPresentation", "cards")
                addProperty("primaryColumn", "Option")
            })
            add("children", JsonArray())
        }
    }

    private fun parseItem(text: String): EntityItem? {
        val bold = boldPrefix.matchEntire(text)
        val title: String
        val details: String
        val delimiter: Delimiter
        if (bold != null) {
            title = bold.groupValues[1]
            val suffix = bold.groupValues[2]
            when {
                suffix.startsWith(": ") -> {
                    details = suffix.substring(2)
                    delimiter = Delimiter.COLON
                }
                suffix.startsWith(" — ") || suffix.startsWith(" – ") -> {
                    details = suffix.substring(3)
                    delimiter = Delimiter.DASH
                }
                suffix.startsWith(" lists ") || suffix.startsWith(" shows ") ||
                    suffix.startsWith(" offers ") -> {
                    details = suffix.substring(1)
                    delimiter = Delimiter.NARRATIVE
                }
                else -> return null
            }
        } else {
            val delimiters = listOf(": " to Delimiter.COLON, " — " to Delimiter.DASH, " – " to Delimiter.DASH)
            val split = delimiters.mapNotNull { (marker, kind) ->
                text.indexOf(marker).takeIf { it in 2..44 }?.let { Triple(it, marker, kind) }
            }.minByOrNull { it.first } ?: return null
            title = text.substring(0, split.first)
            details = text.substring(split.first + split.second.length)
            delimiter = split.third
            if (!isPlainName(title)) return null
        }
        if (!isPlausibleTitle(title) || details.isBlank() || !attribute.containsMatchIn(details)) return null
        return EntityItem(title, details, delimiter, leadingAttribute.containsMatchIn(details))
    }

    private fun isPlausibleTitle(title: String): Boolean {
        if (title != title.trim() || title.length !in 2..44 || title.count(Char::isLetter) < 3 ||
            title.contains('[') || title.contains(']') || title.contains(':') ||
            title.contains('?') || title.contains('!')) return false
        val words = title.split(Regex("""\s+"""))
        if (words.size !in 1..8) return false
        val lower = words.map { it.lowercase(Locale.ROOT).trim('.', ',') }
        if (lower.any { it in genericTitleWords }) return false
        if (lower.first() in adviceOpeners) return false
        return true
    }

    private fun isPlainName(title: String): Boolean {
        val words = title.split(Regex("""\s+"""))
        if (words.isEmpty() || words.size > 8) return false
        // Unmarked labels need the additional signal of a proper name, rather than an advice heading.
        return words.all { word ->
            val firstLetter = word.firstOrNull(Char::isLetter) ?: return@all word.any(Char::isDigit)
            firstLetter.isUpperCase() || word.lowercase(Locale.ROOT) in setOf("and", "of", "the", "&")
        }
    }
}
