package com.samsung.genuicraft.pipeline

import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser

internal object PipelineJsonExtractor {
    /** Legacy JSON extraction used only for debug/import compatibility. */
    enum class ExpressRepairMode {
        GENERAL,
        ON_DEVICE_LITERT
    }

    fun extractJsonElement(text: String): JsonElement? {
        val cleaned = text.trim()
        if (cleaned.isEmpty()) {
            return null
        }

        val candidates = linkedSetOf<String>()
        extractFencedBlock(cleaned)?.let { candidates += it }
        if (cleaned.startsWith("[") || cleaned.startsWith("{")) {
            candidates += cleaned
        }
        val syntaxRepaired = repairCommonJsonSyntax(cleaned)
        if (syntaxRepaired != cleaned) {
            extractFencedBlock(syntaxRepaired)?.let { candidates += it }
            if (syntaxRepaired.startsWith("[") || syntaxRepaired.startsWith("{")) {
                candidates += syntaxRepaired
            }
            candidates += collectBalancedCandidates(syntaxRepaired, '[', ']')
            candidates += collectBalancedCandidates(syntaxRepaired, '{', '}')
        }
        candidates += collectBalancedCandidates(cleaned, '[', ']')
        candidates += collectBalancedCandidates(cleaned, '{', '}')

        val parsedCandidates = candidates.mapNotNull { candidate ->
            runCatching { JsonParser.parseString(candidate) }.getOrNull()
        }
        if (parsedCandidates.isEmpty()) {
            return null
        }

        return parsedCandidates.maxWithOrNull(
            compareBy<JsonElement> { flatSpecShapePriority(it) }
                .thenBy(::scoreJsonCandidate)
        )
    }

    fun extractFencedBlock(text: String): String? {
        val start = text.indexOf("```")
        if (start < 0) {
            return null
        }
        val end = text.indexOf("```", startIndex = start + 3)
        if (end <= start) {
            return null
        }
        val block = text.substring(start + 3, end).trim()
        return if (block.startsWith("json", ignoreCase = true)) {
            block.removePrefix("json").trim()
        } else {
            block
        }
    }

    fun findFirstBalanced(text: String, open: Char, close: Char): String? {
        var index = text.indexOf(open)
        while (index >= 0) {
            val candidate = balancedSubstring(text, index, open, close)
            if (candidate != null) {
                return candidate
            }
            index = text.indexOf(open, startIndex = index + 1)
        }
        return null
    }

    fun balancedSubstring(text: String, start: Int, open: Char, close: Char): String? {
        var depth = 0
        var inString = false
        var escape = false

        for (i in start until text.length) {
            val ch = text[i]
            if (escape) {
                escape = false
                continue
            }
            if (ch == '\\' && inString) {
                escape = true
                continue
            }
            if (ch == '"') {
                inString = !inString
                continue
            }
            if (inString) {
                continue
            }
            if (ch == open) {
                depth += 1
            } else if (ch == close) {
                depth -= 1
                if (depth == 0) {
                    return text.substring(start, i + 1)
                }
            }
        }
        return null
    }

    private fun collectBalancedCandidates(text: String, open: Char, close: Char): List<String> {
        val out = mutableListOf<String>()
        var index = text.indexOf(open)
        while (index >= 0) {
            balancedSubstring(text, index, open, close)?.let { out += it }
            index = text.indexOf(open, startIndex = index + 1)
        }
        return out
    }

    private fun repairCommonJsonSyntax(text: String): String {
        var out = text
        repeat(8) {
            val next = STRAY_COMMA_TOKEN.replace(out, ",")
            if (next == out) {
                return@repeat
            }
            out = next
        }
        out = CHILDREN_EMPTY_ARRAY_TRAILING_QUOTE.replace(out) { match ->
            match.groupValues[1]
        }
        out = MISSING_TABLE_COLUMN_LABEL_KEY.replace(out) { match ->
            "\"key\":\"${match.groupValues[1]}\",\"label\":\"${match.groupValues[2]}\""
        }
        out = MISSING_KEY_PROP_COLON.replace(out) { match ->
            "\"key\":\"${match.groupValues[1]}\",\"label\""
        }
        out = MISSING_TABLE_ROWS_ARRAY_CLOSE_BEFORE_PROP.replace(out) { match ->
            val rowsPrefix = match.groupValues[1] + match.groupValues[2]
            if (rowsPrefix.trimEnd().endsWith("]")) {
                match.value
            } else {
                "$rowsPrefix]],${match.groupValues[3]}"
            }
        }
        repeat(6) {
            val next = MISSING_COMMA_BETWEEN_STRING_PROPS.replace(out) { match ->
                "${match.groupValues[1]},${match.groupValues[2]}"
            }
            if (next == out) {
                return@repeat
            }
            out = next
        }
        return closeUnbalancedJsonSuffix(out)
    }

    private fun closeUnbalancedJsonSuffix(text: String): String {
        val trimmed = text.trim()
        if (trimmed.firstOrNull() !in setOf('{', '[')) return text
        val stack = mutableListOf<Char>()
        var inString = false
        var escaped = false
        trimmed.forEach { ch ->
            if (inString) {
                when {
                    escaped -> escaped = false
                    ch == '\\' -> escaped = true
                    ch == '"' -> inString = false
                }
                return@forEach
            }
            when (ch) {
                '"' -> inString = true
                '{', '[' -> stack += ch
                '}' -> {
                    if (stack.lastOrNull() != '{') return text
                    stack.removeAt(stack.lastIndex)
                }
                ']' -> {
                    if (stack.lastOrNull() != '[') return text
                    stack.removeAt(stack.lastIndex)
                }
            }
        }
        if (inString || stack.isEmpty() || stack.size > MAX_JSON_SUFFIX_CLOSERS) return text
        return buildString(trimmed.length + stack.size) {
            append(trimmed)
            stack.asReversed().forEach { open -> append(if (open == '{') '}' else ']') }
        }
    }

    private fun flatSpecShapePriority(element: JsonElement): Int {
        if (!element.isJsonObject) {
            return 0
        }
        val obj = element.asJsonObject
        return when {
            FlatSpecContract.looksLikeFlatSpec(element) -> 3
            obj.has("genui_json") || obj.has("payload") -> 2
            obj.has("messages") -> 1
            else -> 0
        }
    }

    private fun scoreJsonCandidate(element: JsonElement): Int {
        var score = 0
        if (FlatSpecContract.looksLikeFlatSpec(element)) {
            score += 260
        }
        val ingest = FlatSpecIngestor.ingest(element, FlatSpecIngestMode.STRICT)
        val spec = when (ingest) {
            is FlatSpecIngestResult.CanonicalFlatSpec -> ingest.canonicalJson
            is FlatSpecIngestResult.GenuineLegacyPayload -> ingest.migratedFlatSpec
            is FlatSpecIngestResult.RejectedPayload -> null
        }
        if (spec != null) {
            val elements = spec?.getAsJsonObject("elements")
            val size = elements?.size() ?: 0
            score += 400
            score += size.coerceAtMost(80)
            val rootId = spec?.get("root")
                ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                ?.asString
                .orEmpty()
            if (rootId.isNotBlank() && elements?.has(rootId) == true) {
                score += 60
            }
        } else {
            val normalized = FlatSpecContract.normalizeToFlatSpec(element, compatibilityHeaderInference = false)
            if (normalized.spec != null) {
                score += 160
            } else if (element.isJsonArray) {
                score += 80
            } else if (element.isJsonObject) {
                score += 40
            }
        }

        if (element.isJsonObject) {
            val obj = element.asJsonObject
            if (obj.has("genui_json") || obj.has("messages") || obj.has("payload")) {
                score += 50
            }
        }
        score += element.toString().length.coerceAtMost(4000) / 200
        return score
    }

    private val MISSING_TABLE_COLUMN_LABEL_KEY = Regex(
        "\"key\"\\s*:\\s*\"([^\"]+)\"\\s*:\\s*\"([^\"]+)\""
    )
    private val MISSING_KEY_PROP_COLON = Regex(
        "\"key([A-Za-z0-9_ -]+)\"\\s*,\\s*\"label\""
    )
    private val MISSING_COMMA_BETWEEN_STRING_PROPS = Regex(
        "(\"[^\"]+\"\\s*:\\s*\"[^\"]*\")\\s*:\\s*(\"[^\"]+\"\\s*:)"
    )
    private val MISSING_TABLE_ROWS_ARRAY_CLOSE_BEFORE_PROP = Regex(
        """("rows"\s*:\s*\[\s*\[[\s\S]*?)(.)\u005D\s*\}\s*,\s*("(?:domain|preferredPresentation|presentation|primaryColumn|highlightColumns)"\s*:)"""
    )
    private val STRAY_COMMA_TOKEN = Regex(",\\s*,")
    private val CHILDREN_EMPTY_ARRAY_TRAILING_QUOTE = Regex(
        "(\"children\"\\s*:\\s*\\[\\])\"(?=\\s*[,}])"
    )
    private const val MAX_JSON_SUFFIX_CLOSERS = 4

    fun buildExpressRepairPrompt(
        rawText: String,
        failureReason: String? = null,
        mode: ExpressRepairMode = ExpressRepairMode.GENERAL
    ): String {
        val reasonLine = failureReason?.trim()?.takeIf { it.isNotEmpty() }?.let {
            "Failure reason: $it\n"
        }.orEmpty()
        return (
            "The previous output does not satisfy the required A2UI Express contract.\n" +
            reasonLine +
                "Return ONLY one complete <a2ui>...</a2ui> block with one assignment per line.\n" +
                "Use the pinned catalog/profile and preserve the complete rich UI.\n\n" +
                "Rules:\n" +
                "- Use `root=Column([content])` or another catalog container.\n" +
                "- Every child reference must resolve to an assignment or inline component.\n" +
                "- Use explicit named properties for sparse/ambiguous values; never use opaque bags.\n" +
                "- For tables, preserve all columns/rows with the pinned Table signature and domain metadata.\n" +
                "- Every event value must be an action call such as `openUrl(...)` or `Event(...)`.\n" +
                "- Never return JSON, HTML, CSS, type/props objects, or markdown fences.\n\n" +
                "Example skeleton:\n" +
                "<a2ui>\nroot=Column([content])\ncontent=Text(\"Result\",\"h2\")\n</a2ui>\n\n" +
                "Repair mode: ${mode.name}.\n" +
                "Original output:\n${rawText.trim()}"
            )
    }
}
