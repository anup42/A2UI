package com.samsung.genuicraft.pipeline

import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser

internal object PipelineJsonExtractor {
    enum class FlatSpecRepairMode {
        GENERAL,
        ON_DEVICE_LITERT
    }

    fun extractJsonElement(text: String): JsonElement? {
        val cleaned = text.trim()
        if (cleaned.isEmpty()) {
            return null
        }

        // A complete Compact IR envelope must outrank any nested arrays or
        // element objects, even when its graph still needs strict repair.
        runCatching { JsonParser.parseString(cleaned) }.getOrNull()
            ?.takeIf(CompactIrCodec::looksLike)
            ?.let(::normalizeCompactEnvelope)
            ?.let { return it }
        extractFencedBlock(cleaned)
            ?.let { runCatching { JsonParser.parseString(it) }.getOrNull() }
            ?.takeIf(CompactIrCodec::looksLike)
            ?.let(::normalizeCompactEnvelope)
            ?.let { return it }

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
        }.map(::normalizeCompactEnvelope)
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

    /**
     * Small on-device models occasionally close `e` before emitting a final
     * referenced element, most often `action`. Preserve the generated graph by
     * moving only unmistakable Compact element objects into `e`; leave every
     * other unknown key untouched so strict validation can still reject it.
     */
    private fun normalizeCompactEnvelope(element: JsonElement): JsonElement {
        if (!CompactIrCodec.looksLike(element)) return element
        val source = element.asJsonObject
        source.get("e")?.takeIf { it.isJsonObject }?.asJsonObject
            ?: return element
        val normalized = source.deepCopy()
        val normalizedElements = normalized.getAsJsonObject("e")
        var changed = false

        normalized.entrySet()
            .filter { (key, value) ->
                key !in COMPACT_TOP_LEVEL_KEYS &&
                    value.isJsonObject &&
                    value.asJsonObject.get("t")?.let { it.isJsonPrimitive && it.asString.isNotBlank() } == true &&
                    !normalizedElements.has(key)
            }
            .toList()
            .forEach { (key, value) ->
            normalizedElements.add(key, value.deepCopy())
            normalized.remove(key)
            changed = true
        }

        normalizedElements.entrySet().forEach { (_, raw) ->
            if (!raw.isJsonObject) return@forEach
            val item = raw.asJsonObject
            COMPACT_MISPLACED_PROP_KEYS.forEach { key ->
                val misplaced = item.get(key) ?: return@forEach
                val props = item.get("p")?.takeIf { it.isJsonObject }?.asJsonObject
                    ?: JsonObject().also { item.add("p", it) }
                if (!props.has(key)) props.add(key, misplaced.deepCopy())
                item.remove(key)
                changed = true
            }
            val props = item.get("p")?.takeIf { it.isJsonObject }?.asJsonObject
            val misplacedAction = props?.get("o")
            if (!item.has("o") && misplacedAction?.isJsonObject == true) {
                item.add("o", misplacedAction.deepCopy())
                props.remove("o")
                changed = true
            }
        }
        return if (changed) normalized else element
    }

    private fun flatSpecShapePriority(element: JsonElement): Int {
        if (!element.isJsonObject) {
            return 0
        }
        val obj = element.asJsonObject
        return when {
            CompactIrCodec.looksLike(element) -> 4
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
    private val COMPACT_TOP_LEVEL_KEYS = setOf("v", "r", "s", "e")
    private val COMPACT_MISPLACED_PROP_KEYS = setOf(
        "columns",
        "rows",
        "statePath",
        "domain",
        "preferredPresentation",
        "presentation",
        "primaryColumn",
        "highlightColumns",
        "variant",
    )
    private const val MAX_JSON_SUFFIX_CLOSERS = 4

    fun buildFlatSpecRepairPrompt(
        rawText: String,
        failureReason: String? = null,
        mode: FlatSpecRepairMode = FlatSpecRepairMode.GENERAL
    ): String {
        if (mode == FlatSpecRepairMode.ON_DEVICE_LITERT) {
            return buildOnDeviceFlatSpecRepairPrompt(rawText, failureReason)
        }
        val reasonLine = failureReason?.trim()?.takeIf { it.isNotEmpty() }?.let {
            "Failure reason: $it\n"
        }.orEmpty()
        return (
            "The previous output does not satisfy the required flat-spec contract.\n" +
                reasonLine +
                "Return ONLY one valid JSON object with this shape:\n" +
                "{\"root\":\"<id>\",\"state\":{...},\"elements\":{...}}\n\n" +
                "Rules:\n" +
                "- `root` must reference an existing key in `elements`.\n" +
                "- `elements` must be a non-empty object with at least 2 entries (root container + content).\n" +
                "- Do NOT return `{}` and do NOT return `\"elements\": {}`.\n" +
                "- Every element must contain `type`, `props`, and `children`.\n" +
                "- Every id in `children` must exist in `elements`.\n" +
                "- For table data, prefer a compact `Table` element with empty `children`.\n" +
                "- Set `Table.props.columns` and `Table.props.statePath` (or inline `rows` when needed).\n" +
                "- Keep row data in state arrays; do not expand to one element id per row/cell.\n" +
                "- Set `Table.props.domain` (`weather|flight|generic`) and `Table.props.preferredPresentation` (`cards|table`).\n" +
                "- Weather/climate outputs must include a dedicated metrics table section.\n" +
                "- Prefer a Stack root container with direction set.\n" +
                "- Return a complete, renderable flat-spec even when source content is brief.\n" +
                "- Return JSON only, no markdown.\n\n" +
                "Required minimum skeleton (adapt ids/content as needed):\n" +
                "{\"root\":\"root\",\"state\":{},\"elements\":{\"root\":{\"type\":\"Stack\",\"props\":{\"direction\":\"vertical\"},\"children\":[\"content\"]},\"content\":{\"type\":\"Text\",\"props\":{\"text\":\"...\"},\"children\":[]}}}\n\n" +
                "Original output:\n${rawText.trim()}"
            )
    }

    private fun buildOnDeviceFlatSpecRepairPrompt(rawText: String, failureReason: String?): String {
        val reasonLine = failureReason?.trim()?.takeIf { it.isNotEmpty() }?.let {
            "Failure reason: $it\n"
        }.orEmpty()
        return (
            "Repair the previous on-device Gemma output into ONE valid GenUICraft flat-spec JSON object.\n" +
                reasonLine +
                "Return JSON only. No markdown, no explanations.\n\n" +
                "Required top-level shape exactly:\n" +
                "{\"root\":\"root\",\"state\":{\"rows\":[]},\"elements\":{...}}\n\n" +
                "Hard repair rules:\n" +
                "- Top-level keys must be only root, state, elements.\n" +
                "- Move element-like top-level objects into elements, or drop them if not referenced.\n" +
                "- root must exist in elements.\n" +
                "- Every child id must exist in elements; remove missing child references instead of preserving them.\n" +
                "- Use only these element ids for data screens: root, title, summary, summaryText, table.\n" +
                "- root children must be exactly [\"title\",\"summary\",\"table\"] for data screens.\n" +
                "- Drop actions, sources, icons, images, buttons, currentWeather, nextDays, and forecast section ids.\n" +
                "- Normalize id mismatches such as Table/ForecastTable/forecastTable/nextDays to one child id named table.\n" +
                "- Every element must include type, props object, and children array.\n" +
                "- Put children only in the element-level children array. Do not put child lists inside props.\n" +
                "- JSON syntax must be strict: no comma-only lines, no duplicate commas, no trailing commas, and no quote after an empty children array.\n" +
                "- Allowed types in on-device repair: Stack, Card, Text, Table.\n" +
                "- For weather, flight, booking, schedule, and status data: use one compact Table backed by state.rows.\n" +
                "- Row keys must be unique inside each row; do not repeat label/value keys.\n" +
                "- Every Table column key must exist in every row object; normalize rain/rainChance and label/value mismatches.\n" +
                "- Do not expand table rows/cells into forecastRow, dayRow, cell, or per-column element trees.\n" +
                "- Preserve numbers, dates, units, and labels from the source as much as possible.\n\n" +
                "Minimum valid compact output pattern:\n" +
                "{\"root\":\"root\",\"state\":{\"rows\":[{\"label\":\"Example\",\"value\":\"Value\"}]},\"elements\":{\"root\":{\"type\":\"Stack\",\"props\":{\"direction\":\"vertical\",\"gap\":\"md\"},\"children\":[\"title\",\"summary\",\"table\"]},\"title\":{\"type\":\"Text\",\"props\":{\"text\":\"Result\",\"variant\":\"h2\"},\"children\":[]},\"summary\":{\"type\":\"Card\",\"props\":{},\"children\":[\"summaryText\"]},\"summaryText\":{\"type\":\"Text\",\"props\":{\"text\":\"Short summary.\"},\"children\":[]},\"table\":{\"type\":\"Table\",\"props\":{\"columns\":[{\"key\":\"label\",\"label\":\"Label\"},{\"key\":\"value\",\"label\":\"Value\"}],\"statePath\":\"/rows\",\"domain\":\"generic\",\"preferredPresentation\":\"cards\",\"primaryColumn\":\"label\",\"highlightColumns\":[\"value\"]},\"children\":[]}}}\n\n" +
                "Previous output:\n${rawText.trim()}"
            )
    }
}
