package com.samsung.genuicraft.sdk.internal.pipeline

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject

/**
 * General recovery for malformed model-produced Express.
 *
 * This layer consumes only bytes already present in the generated DSL. It can normalize catalog
 * spelling/layout syntax, prune broken graph edges, retain independently valid component calls, or
 * literalize a syntactically valid generated state object. It never reads the source response and
 * never invents answer text. Source fidelity remains a separate acceptance gate in [GenUiCompiler].
 */
internal object A2uiExpressGeneralRepair {
    data class Candidate(val express: String, val changes: List<String>)

    private const val OPEN = "<a2ui>"
    private const val CLOSE = "</a2ui>"
    private const val MAX_CALLS = 64
    private const val MAX_LINES = 2_048
    private const val MAX_LITERAL_ITEMS = 64
    private const val MAX_LITERAL_CHARS = 20_000
    private const val MAX_LITERAL_LENGTH = 1_500
    private val components = GenUiA2uiCatalog.positional.keys
    private val containerComponents = setOf("Card", "Column", "Row", "Stack", "List")
    private val gapTokens = setOf("none", "sm", "md", "lg", "xl")
    private val alignTokens = setOf("start", "center", "end", "stretch")
    private val componentCallPattern = Regex(
        "\\b(?:${components.joinToString("|") { Regex.escape(it) }})\\s*\\(",
        RegexOption.IGNORE_CASE,
    )

    fun candidates(input: String): List<Candidate> {
        val values = linkedMapOf<String, Candidate>()
        var complete: Candidate? = null
        normalizedCompleteDocument(input)?.let { (document, normalizationChanges) ->
            repairDecodedGraph(document, normalizationChanges)?.let { candidate ->
                if (!hasUnresolvedDataBindings(candidate.express)) complete = candidate
            }
        }
        // A pruned graph can leave data in unreachable state. Prefer the combined recovery when
        // state was generated; retain the original layout for complete literal-only graphs.
        complete?.takeIf { A2uiExpressCodec.decode(it.express).getAsJsonObject("state").size() == 0 }
            ?.let { values.putIfAbsent(it.express, it) }
        // These are complementary sources of generated content, not competing candidates. Returning
        // the first valid leaf used to discard every state row and the other damaged components.
        salvageGeneratedContent(input)?.let { candidate -> values.putIfAbsent(candidate.express, candidate) }
        complete?.let { values.putIfAbsent(it.express, it) }
        return values.values.toList()
    }

    fun hasUnresolvedDataBindings(express: String): Boolean {
        val graph = runCatching { A2uiExpressCodec.decode(express) }.getOrNull() ?: return false
        val state = graph.getAsJsonObject("state")
        return graph.getAsJsonObject("elements").entrySet().any { (_, raw) ->
            val element = raw.asJsonObject
            val props = element.getAsJsonObject("props")
            element.get("type")?.asString in setOf("Table", "Chart") && props.get("rows")?.isJsonArray != true &&
                listOf("statePath", "rowsPath", "dataPath").any { key ->
                    props.get(key)?.takeIf { it.isJsonPrimitive }?.asString?.let { path ->
                        stateAtPath(state, path)?.isJsonArray != true
                    } == true
                }
        }
    }

    private fun normalizedCompleteDocument(input: String): Pair<String, List<String>>? {
        var value = input.removePrefix("\uFEFF").trim()
        val changes = mutableListOf<String>()
        exactFenceBody(value)?.let {
            value = it.trim()
            changes += "Removed an exact Markdown code-fence wrapper."
        }
        when {
            value.startsWith("<a2a2ui>") -> {
                value = OPEN + value.removePrefix("<a2a2ui>")
                changes += "Normalized duplicated A2UI opening tag."
            }
            value.startsWith("< 2ui>") -> {
                value = OPEN + value.removePrefix("< 2ui>")
                changes += "Normalized corrupted A2UI opening tag."
            }
        }
        if (!value.startsWith(OPEN)) return null
        value = value.replace(Regex("</a2ui(?=</a2ui>\\s*$)"), "").also {
            if (it != value) changes += "Removed malformed duplicate closing-tag prefix."
        }
        val close = closingTagOutsideStrings(value)
        value = when {
            close >= 0 -> {
                val end = close + CLOSE.length
                if (value.substring(end).isNotBlank()) changes += "Discarded bytes after the first complete A2UI envelope."
                value.substring(0, end)
            }
            lexicallyBalancedBody(value.removePrefix(OPEN)) -> {
                changes += "Appended a missing closing tag to a lexically complete body."
                "$value\n$CLOSE"
            }
            else -> return null
        }
        val normalized = normalizeSyntax(value, changes)
        return normalized to changes.distinct()
    }

    private fun repairDecodedGraph(document: String, initialChanges: List<String>): Candidate? {
        val graph = runCatching { A2uiExpressCodec.decode(document) }.getOrNull() ?: return null
        val changes = initialChanges.toMutableList()
        val elements = graph.getAsJsonObject("elements") ?: return null
        val ids = elements.keySet().toSet()
        elements.entrySet().forEach { (id, raw) ->
            val children = raw.asJsonObject.getAsJsonArray("children") ?: JsonArray().also { raw.asJsonObject.add("children", it) }
            val kept = JsonArray()
            val local = linkedSetOf<String>()
            children.forEach { child ->
                val target = child.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
                when {
                    target == null || target.isBlank() -> changes += "Removed invalid child reference from '$id'."
                    target !in ids -> changes += "Removed dangling child '$target' from '$id'."
                    target == id -> changes += "Removed self-reference '$target' from '$id'."
                    !local.add(target) -> changes += "Removed duplicate child '$target' from '$id'."
                    else -> kept.add(target)
                }
            }
            raw.asJsonObject.add("children", kept)
        }

        val root = graph.get("root")?.asString ?: return null
        val claimed = linkedSetOf(root)
        fun claim(id: String, stack: MutableSet<String>) {
            val element = elements.getAsJsonObject(id) ?: return
            val children = element.getAsJsonArray("children") ?: return
            val kept = JsonArray()
            children.forEach { child ->
                val target = child.asString
                if (target in stack || target in claimed) {
                    changes += "Removed repeated or cyclic child '$target' from '$id'."
                } else {
                    kept.add(target)
                    claimed += target
                    stack += target
                    claim(target, stack)
                    stack -= target
                }
            }
            element.add("children", kept)
        }
        claim(root, linkedSetOf(root))
        val rootElement = elements.getAsJsonObject(root) ?: return null
        if (rootElement.get("type")?.asString == "Stack") {
            val rootChildren = rootElement.getAsJsonArray("children")
            elements.keySet().toList().forEach { id ->
                if (id != root && id !in claimed) {
                    rootChildren.add(id)
                    claimed += id
                    changes += "Attached previously unreachable generated component '$id' to root."
                    claim(id, linkedSetOf(root, id))
                }
            }
        }
        if (!hasVisibleContent(graph)) return null
        if (!A2uiCanonicalGraph.validate(graph).isValid) return null
        val canonical = runCatching { A2uiExpressCodec.encode(graph) }.getOrNull() ?: return null
        return Candidate(canonical, changes.distinct() + "Recovered a complete generated DSL graph without source fallback.")
    }

    private fun salvageGeneratedContent(input: String): Candidate? {
        val body = repairBody(input) ?: return null
        if (body.lineSequence().take(MAX_LINES + 1).count() > MAX_LINES) return null
        val statements = recoveryStatements(body)
        val recovered = GeneratedStateRecovery.recover(statements)
        val accepted = mutableListOf<JsonObject>()
        val canonicalSeen = linkedSetOf<String>()
        val changes = mutableListOf<String>()
        graphFromGeneratedState(recovered.state)?.let(accepted::add)
        var recoveredCalls = 0
        var damagedCalls = 0
        statements.forEach { statement ->
            var line = statement.trim()
            if (line.startsWith('$')) {
                return@forEach
            }
            line = line.replace(Regex("^[*@]+(?=[A-Za-z_])"), "")
            val call = extractBalancedCall(line)
            val normalized = call?.let { normalizeSyntax(it, changes) }
            val graph = normalized?.let {
                runCatching { A2uiExpressCodec.decode("$OPEN\nroot=$it\n$CLOSE") }.getOrNull()
            }
            graph?.getAsJsonObject("elements")?.entrySet()?.forEach { (_, raw) ->
                val element = raw.asJsonObject
                val props = element.getAsJsonObject("props")
                if (element.get("type")?.asString in setOf("Table", "Chart") && props.get("rows")?.isJsonArray == true) {
                    listOf("statePath", "rowsPath", "dataPath").forEach(props::remove)
                }
            }
            // State has already been materialized in full above. Keeping an isolated bound table
            // here would either duplicate its rows or leave an empty table with a dangling path.
            if (graph == null || recoveredCalls >= MAX_CALLS || hasStateBindings(graph) ||
                !A2uiCanonicalGraph.validate(graph).isValid || !hasVisibleContent(graph)) {
                damagedCalls += 1
                return@forEach
            }
            val canonical = runCatching { A2uiExpressCodec.encode(graph) }.getOrNull() ?: run {
                damagedCalls += 1
                return@forEach
            }
            if (canonicalSeen.add(canonical)) {
                accepted += graph
                recoveredCalls += 1
            }
        }
        // Loose literals from damaged calls are noisy and often duplicate the structured state or
        // surviving components. Do not append them as an "Additional recovered text" section.
        // If no structured content survived at all, literal-only recovery remains the last resort.
        if (accepted.isEmpty()) return salvageVisibleStringLiterals(input)
        val merged = mergeRecoveredGraphs(accepted) ?: return null
        val canonical = runCatching { A2uiExpressCodec.encode(merged) }.getOrNull() ?: return null
        return Candidate(
            canonical,
            changes.distinct() + listOf(
                "Recovered ${recovered.assignments} generated state assignment(s), including ${recovered.partialAssignments} damaged assignment(s) with complete literal values.",
                "Materialized all ${recovered.state.size()} recovered state field(s), preserving row order, repeated rows, and generated values.",
                "Combined $recoveredCalls valid generated component call(s) with recovered state; inspected $damagedCalls damaged or state-bound call(s) for additional readable fragments.",
                "Best-effort generated-output recovery; unresolved structure is not a complete or source-verified answer. Source text was not used.",
            ),
        )
    }

    /** Resynchronize at the next assignment even when the previous expression was truncated. */
    private fun recoveryStatements(body: String): List<String> {
        val starts = Regex("(?m)^[\\t ]*(?:\\$(?:/[^\\s=:{]*)?[\\t ]*(?:=|:|(?=\\{))|[*@]*[A-Za-z_][A-Za-z0-9_]*[\\t ]*=)")
            .findAll(body).map { it.range.first }.toList()
        return starts.mapIndexed { index, start -> body.substring(start, starts.getOrNull(index + 1) ?: body.length).trim() }
    }

    private fun hasStateBindings(graph: JsonObject): Boolean {
        fun bound(value: JsonElement): Boolean = when {
            value.isJsonObject -> value.asJsonObject.entrySet().any { (key, child) ->
                key in setOf("statePath", "rowsPath", "dataPath", "repeat", "visible") || bound(child)
            }
            value.isJsonArray -> value.asJsonArray.any(::bound)
            value.isJsonPrimitive && value.asJsonPrimitive.isString -> value.asString.startsWith("$/")
            else -> false
        }
        return bound(graph.getAsJsonObject("elements"))
    }

    private fun stateAtPath(state: JsonElement, path: String): JsonElement? {
        var value: JsonElement = state
        val pointer = path.removePrefix("$")
        if (!pointer.startsWith('/')) return null
        for (part in pointer.removePrefix("/").split('/')) {
            val key = part.replace("~1", "/").replace("~0", "~")
            value = when {
                value.isJsonObject -> value.asJsonObject.get(key)
                value.isJsonArray -> key.toIntOrNull()?.takeIf { it in 0 until value.asJsonArray.size() }?.let { value.asJsonArray.get(it) }
                else -> null
            } ?: return null
        }
        return value
    }

    private fun graphFromGeneratedState(state: JsonObject): JsonObject? {
        val elements = JsonObject()
        val rootChildren = JsonArray()
        var index = 0
        state.entrySet().forEach { (key, value) ->
            val id = "s${index++}"
            val element = stateElement(key, value) ?: return@forEach
            elements.add(id, element)
            rootChildren.add(id)
        }
        if (rootChildren.size() == 0) return null
        elements.add("root", JsonObject().apply {
            addProperty("type", "Stack")
            add("props", JsonObject().apply {
                addProperty("direction", "vertical")
                addProperty("gap", "md")
            })
            add("children", rootChildren)
        })
        return JsonObject().apply {
            addProperty("root", "root")
            add("state", JsonObject())
            add("elements", elements)
        }
    }

    /** Last-resort generated-output salvage for a program whose graph and state grammar are lost. */
    private fun salvageVisibleStringLiterals(input: String): Candidate? {
        val body = repairBody(input) ?: return null
        val values = linkedSetOf<String>()
        var retainedCharacters = 0
        var ignored = 0
        // A broken quote in one assignment must not swallow every later assignment's strings.
        val ranges = Regex("[^\\r\\n]+").findAll(body).flatMap { line ->
            val offset = line.range.first
            stringRanges(line.value).map { (it.first + offset)..(it.last + offset) }
        }
        ranges.forEach { range ->
            if (values.size >= MAX_LITERAL_ITEMS || retainedCharacters >= MAX_LITERAL_CHARS) {
                ignored += 1
                return@forEach
            }
            if (range.last !in body.indices || body[range.last] != '"') {
                ignored += 1
                return@forEach
            }
            var next = range.last + 1
            while (next < body.length && body[next].isWhitespace()) next += 1
            if (next < body.length && body[next] == ':') {
                ignored += 1 // Quoted object key, not a visible value.
                return@forEach
            }
            val prefix = body.substring(0, range.first).takeLast(96)
            if (Regex(
                    "\\b(?:gap|align|alignment|justify|variant|tone|domain|preferredPresentation|presentation|statePath|rowsPath|dataPath|icon|fit|direction|width|height|flex)\\s*=\\s*$|\\bkey\\s*:\\s*$",
                    RegexOption.IGNORE_CASE,
                ).containsMatchIn(prefix)) {
                ignored += 1 // Layout/binding metadata remains metadata even when its value is corrupt.
                return@forEach
            }
            val decoded = runCatching { com.google.gson.JsonParser.parseString(body.substring(range)).asString }.getOrNull()
            if (decoded == null) {
                ignored += 1
                return@forEach
            }
            if (!isUsefulVisibleLiteral(decoded) || retainedCharacters + decoded.length > MAX_LITERAL_CHARS) {
                ignored += 1
                return@forEach
            }
            if (values.add(decoded)) retainedCharacters += decoded.length
        }
        if (values.isEmpty()) return null
        val graph = JsonObject().apply {
            addProperty("root", "root")
            add("state", JsonObject())
            add("elements", JsonObject().apply {
                add("root", JsonObject().apply {
                    addProperty("type", "List")
                    add("props", JsonObject().apply {
                        add("items", JsonArray().apply { values.forEach(::add) })
                    })
                    add("children", JsonArray())
                })
            })
        }
        if (!A2uiCanonicalGraph.validate(graph).isValid || !hasVisibleContent(graph)) return null
        val canonical = runCatching { A2uiExpressCodec.encode(graph) }.getOrNull() ?: return null
        return Candidate(
            canonical,
            listOf(
                "Salvaged ${values.size} additional complete generated string literal value(s) from damaged structure.",
                "Ignored $ignored key, style, corrupt, duplicate, or over-budget string literal(s).",
                "Rebuilt a generated-literal List; source text was not used.",
            ),
        )
    }

    private fun isUsefulVisibleLiteral(value: String): Boolean {
        val normalized = value.trim()
        if (normalized.length !in 2..MAX_LITERAL_LENGTH) return false
        if (normalized.startsWith(':') || normalized.startsWith(',') || '=' in normalized || '{' in normalized || '}' in normalized) return false
        if (normalized.startsWith('/') || normalized.startsWith("$/")) return false
        if ('_' in normalized && normalized.none(Char::isWhitespace)) return false
        if (componentCallPattern.containsMatchIn(normalized)) return false
        if (normalized.lowercase() in setOf(
                "h1", "h2", "h3", "h4", "h5", "h6", "sg", "sg1", "body", "caption",
                "primary", "secondary", "generic", "cards", "table", "vertical", "horizontal",
                "none", "sm", "md", "lg", "xl", "start", "center", "end", "stretch",
            )) return false
        if (Regex("(.)\\1{31,}").containsMatchIn(normalized)) return false
        if (normalized.length > 200 && normalized.toSet().size <= 8) return false
        val lettersOrDigits = normalized.count(Char::isLetterOrDigit)
        return lettersOrDigits >= 2 && lettersOrDigits * 5 >= normalized.length
    }

    private fun stateElement(key: String, value: JsonElement): JsonObject? {
        val title = key.replace('_', ' ').replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
        val props = JsonObject()
        val type = when {
            value.isJsonArray && value.asJsonArray.size() > 0 && value.asJsonArray.all { it.isJsonObject } -> {
                val columns = linkedSetOf<String>()
                value.asJsonArray.forEach { row -> row.asJsonObject.keySet().forEach(columns::add) }
                if (columns.isEmpty()) return null
                props.addProperty("title", title)
                props.add("columns", JsonArray().apply { columns.forEach(::add) })
                props.add("rows", JsonArray().apply {
                    value.asJsonArray.forEach { row ->
                        add(JsonArray().apply { columns.forEach { column -> add(displayValue(row.asJsonObject.get(column))) } })
                    }
                })
                props.addProperty("domain", recoveredTableDomain(key, columns))
                props.addProperty("preferredPresentation", "cards")
                "Table"
            }
            value.isJsonArray -> {
                val items = JsonArray()
                value.asJsonArray.forEach { items.add(displayValue(it)) }
                if (items.size() == 0) return null
                props.add("items", items)
                "List"
            }
            value.isJsonObject -> {
                val rows = JsonArray()
                value.asJsonObject.entrySet().forEach { (field, item) ->
                    rows.add(JsonArray().apply { add(field); add(displayValue(item)) })
                }
                if (rows.size() == 0) return null
                props.addProperty("title", title)
                props.add("columns", JsonArray().apply { add("Field"); add("Value") })
                props.add("rows", rows)
                props.addProperty("domain", "generic")
                props.addProperty("preferredPresentation", "cards")
                "Table"
            }
            else -> {
                val text = "$title: ${displayValue(value)}"
                if (text.isBlank()) return null
                props.addProperty("text", text)
                "Text"
            }
        }
        return JsonObject().apply {
            addProperty("type", type)
            add("props", props)
            add("children", JsonArray())
        }
    }

    /**
     * Generated state keys often add a descriptive suffix to a renderer target, for example
     * `flight_options_schedule`. Match a known target anywhere in the normalized key so recovery
     * retains that semantic route. Flight-shaped columns are a fallback for generic state names.
     */
    private fun recoveredTableDomain(key: String, columns: Collection<String>): String {
        val normalizedKey = normalizeTargetName(key)
        recoveredStateDomainTargets.forEach { (domain, targets) ->
            if (targets.any(normalizedKey::contains)) return domain
        }

        val normalizedColumns = columns.map(::normalizeTargetName)
        val hasCarrier = normalizedColumns.any { column ->
            listOf("airline", "carrier", "flight", "flight_number").any(column::contains)
        }
        val hasDeparture = normalizedColumns.any { column ->
            listOf("departure", "depart", "origin").any(column::contains)
        }
        val hasArrival = normalizedColumns.any { column ->
            listOf("arrival", "arrive", "destination").any(column::contains)
        }
        return if (hasCarrier && hasDeparture && hasArrival) "flight" else "generic"
    }

    private fun normalizeTargetName(value: String): String = value
        .lowercase(java.util.Locale.ROOT)
        .replace(Regex("[^a-z0-9]+"), "_")
        .trim('_')

    private fun displayValue(value: JsonElement?): String = when {
        value == null || value.isJsonNull -> ""
        value.isJsonPrimitive -> value.asJsonPrimitive.asString
        else -> value.toString()
    }

    private val recoveredStateDomainTargets = linkedMapOf(
        // Keep more specific targets before generic schedule/status substrings.
        "flight" to listOf("flight_options", "flight_schedule", "flight_results", "flight_itinerary"),
        "weather" to listOf("weather_forecast", "weather_outlook", "weather_conditions"),
        "booking" to listOf("booking_options", "hotel_options", "restaurant_options"),
        "comparison" to listOf("comparison", "compare_options", "feature_matrix"),
        "status" to listOf("delivery_status", "order_status", "service_status", "status_updates"),
        "schedule" to listOf("schedule", "agenda", "itinerary"),
    )

    private fun mergeRecoveredGraphs(graphs: List<JsonObject>): JsonObject? {
        val elements = JsonObject()
        val rootChildren = JsonArray()
        graphs.forEachIndexed { index, graph ->
            val sourceElements = graph.getAsJsonObject("elements") ?: return@forEachIndexed
            val ids = sourceElements.keySet().associateWith { old -> if (old == "root") "c$index" else "c${index}_$old" }
            sourceElements.entrySet().forEach { (oldId, raw) ->
                val copy = raw.asJsonObject.deepCopy()
                rewriteReferences(copy, ids)
                elements.add(ids.getValue(oldId), copy)
            }
            rootChildren.add(ids.getValue("root"))
        }
        if (rootChildren.size() == 0) return null
        elements.add("root", JsonObject().apply {
            addProperty("type", "Stack")
            add("props", JsonObject().apply {
                addProperty("direction", "vertical")
                addProperty("gap", "md")
            })
            add("children", rootChildren)
        })
        return JsonObject().apply {
            addProperty("root", "root")
            add("state", JsonObject())
            add("elements", elements)
        }.takeIf { A2uiCanonicalGraph.validate(it).isValid }
    }

    private fun rewriteReferences(element: JsonObject, ids: Map<String, String>) {
        element.getAsJsonArray("children")?.let { children ->
            val rewritten = JsonArray()
            children.forEach { child -> rewritten.add(ids[child.asString] ?: child.asString) }
            element.add("children", rewritten)
        }
        val props = element.getAsJsonObject("props") ?: return
        listOf("trigger", "content", "child", "template", "itemTemplate").forEach { key ->
            props.get(key)?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString?.let { value ->
                ids[value]?.let { props.addProperty(key, it) }
            }
        }
        props.getAsJsonArray("tabs")?.forEach { raw ->
            if (!raw.isJsonObject) return@forEach
            val tab = raw.asJsonObject
            listOf("child", "content", "id", "element").forEach { key ->
                tab.get(key)?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString?.let { value ->
                    ids[value]?.let { tab.addProperty(key, it) }
                }
            }
        }
        element.getAsJsonObject("repeat")?.let { repeat ->
            listOf("template", "itemTemplate", "child").forEach { key ->
                repeat.get(key)?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString?.let { value ->
                    ids[value]?.let { repeat.addProperty(key, it) }
                }
            }
        }
    }

    private fun hasVisibleContent(graph: JsonObject): Boolean {
        val elements = graph.getAsJsonObject("elements") ?: return false
        return elements.entrySet().any { (_, raw) ->
            val element = raw.asJsonObject
            val type = element.get("type")?.asString
            val props = element.getAsJsonObject("props") ?: JsonObject()
            when (type) {
                "Text" -> props.get("text")?.asString?.isNotBlank() == true
                "Table" -> props.get("rows")?.takeIf { it.isJsonArray }?.asJsonArray?.size()?.let { it > 0 } == true ||
                    listOf("statePath", "rowsPath", "dataPath").any { key ->
                        props.get(key)?.takeIf { it.isJsonPrimitive }?.asString?.let { path ->
                            stateAtPath(graph.getAsJsonObject("state"), path)?.takeIf { it.isJsonArray }?.asJsonArray?.size()?.let { it > 0 }
                        } == true
                    }
                "List", "Checklist" -> props.getAsJsonArray("items")?.size()?.let { it > 0 } == true
                "Card" -> listOf("title", "subtitle").any { props.get(it)?.asString?.isNotBlank() == true }
                "Image", "Video", "AudioPlayer", "Button", "Alert", "EmailPreview", "Chart", "CodeBlock", "ConsoleLog", "Formula" -> props.size() > 0
                else -> false
            }
        }
    }

    private fun repairBody(input: String): String? {
        var value = input.removePrefix("\uFEFF").trim()
        exactFenceBody(value)?.let { value = it.trim() }
        value = when {
            value.startsWith("<a2a2ui>") -> OPEN + value.removePrefix("<a2a2ui>")
            value.startsWith("< 2ui>") -> OPEN + value.removePrefix("< 2ui>")
            else -> value
        }
        val open = value.indexOf(OPEN)
        if (open < 0) return null
        val start = open + OPEN.length
        val close = closingTagOutsideStrings(value, start).let { if (it < 0) value.length else it }
        return value.substring(start, close)
    }

    private fun closingTagOutsideStrings(value: String, start: Int = 0): Int {
        val ranges = stringRanges(value)
        var index = value.indexOf(CLOSE, start)
        while (index >= 0 && ranges.any { index in it }) index = value.indexOf(CLOSE, index + CLOSE.length)
        return index
    }

    private fun extractBalancedCall(line: String): String? {
        val match = Regex("(?:^|=)\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*\\(").findAll(line).firstOrNull() ?: return null
        val start = match.groups[1]?.range?.first ?: return null
        var depth = 0
        var inString = false
        var escaped = false
        var sawOpen = false
        for (index in start until line.length) {
            val char = line[index]
            if (inString) {
                when {
                    escaped -> escaped = false
                    char == '\\' -> escaped = true
                    char == '"' -> inString = false
                }
                continue
            }
            when (char) {
                '"' -> inString = true
                '(', '[', '{' -> { depth += 1; sawOpen = true }
                ')', ']', '}' -> {
                    depth -= 1
                    if (depth < 0) return null
                    if (sawOpen && depth == 0) return line.substring(start, index + 1)
                }
            }
        }
        return null
    }

    private fun normalizeSyntax(input: String, changes: MutableList<String>): String {
        var value = input
        val componentRegex = Regex("\\b([A-Za-z_][A-Za-z0-9_]*)\\s*(?=\\()")
        value = replaceOutsideStrings(value, componentRegex) { match ->
            val raw = match.groupValues[1]
            val canonical = components.singleOrNull { it.equals(raw, ignoreCase = true) }
                ?: raw.removeSuffix("s").takeIf { raw.endsWith("s", ignoreCase = true) }
                    ?.let { singular -> components.singleOrNull { it.equals(singular, ignoreCase = true) } }
                ?: raw
            if (canonical != raw) changes += "Canonicalized component '$raw' to '$canonical'."
            match.value.replaceRange(match.groups[1]!!.range.relativeTo(match.range), canonical)
        }
        value = replaceOutsideStrings(value, Regex("\\b(Gap|Alignment|Children)\\s*(?==)", RegexOption.IGNORE_CASE)) { match ->
            val canonical = when (match.groupValues[1].lowercase()) {
                "gap" -> "gap"
                "alignment" -> "align"
                else -> "children"
            }
            if (match.groupValues[1] != canonical) changes += "Canonicalized property '${match.groupValues[1]}' to '$canonical'."
            match.value.replace(match.groupValues[1], canonical)
        }
        value = replaceOutsideStrings(value, Regex("\\bchildren\\s*=\\s*([A-Za-z_][A-Za-z0-9_]*)\\b")) { match ->
            changes += "Wrapped scalar children reference '${match.groupValues[1]}' in an array."
            "children=[${match.groupValues[1]}]"
        }
        value = normalizeHybridTableColumns(value, changes)
        value = replaceOutsideStrings(
            value,
            Regex("\\b(Card|Column|Row|Stack|List)\\s*\\(\\s*((?:[A-Za-z_][A-Za-z0-9_]*\\s*,\\s*)*[A-Za-z_][A-Za-z0-9_]*)\\s*(?=[,)])"),
        ) { match ->
            val refs = match.groupValues[2].split(',').map(String::trim)
            changes += "Wrapped ${refs.size} positional child reference(s) for ${match.groupValues[1]}."
            "${match.groupValues[1]}([${refs.joinToString(",")} ]"
        }
        value = replaceOutsideStrings(
            value,
            Regex("\\b(Column|Row|Stack)\\s*\\(\\s*children\\s*=\\s*(\\[[^]\\r\\n]*])\\s*,\\s*(\"(?:\\\\.|[^\"])*\"|-?\\d+(?:\\.\\d+)?)"),
        ) { match ->
            changes += "Rebound positional layout value after named children to gap."
            "${match.groupValues[1]}(children=${match.groupValues[2]},gap=${match.groupValues[3]}"
        }
        value = normalizeLayoutValue(value, "gap", gapTokens, "md", changes)
        value = normalizeLayoutValue(value, "align", alignTokens, "start", changes)
        return value
    }

    /**
     * Recovers the narrow malformed form emitted by the trained model:
     * columns=[{Name:"Name","Area","Price"}]. The identifier must exactly repeat the first
     * label, so the transformation does not infer a key or rewrite any visible column label.
     */
    private fun normalizeHybridTableColumns(input: String, changes: MutableList<String>): String {
        val string = "\"(?:\\\\.|[^\"\\\\])*\""
        val regex = Regex("\\bcolumns\\s*=\\s*\\[\\{([A-Za-z_][A-Za-z0-9_]*)\\s*:\\s*($string)((?:\\s*,\\s*$string)+)\\s*\\}\\]", RegexOption.IGNORE_CASE)
        return replaceOutsideStrings(input, regex, allowQuotedValue = true) { match ->
            val identifier = match.groupValues[1]
            val labels = Regex(string).findAll(match.groupValues[2] + match.groupValues[3]).map { it.value }.toList()
            val first = labels.firstOrNull()?.removeSurrounding("\"")
            if (first != identifier) {
                match.value
            } else {
                changes += "Normalized a hybrid Table columns object whose first key exactly repeated its first label."
                "columns=[${labels.joinToString(",")}]"
            }
        }
    }

    private fun normalizeLayoutValue(
        input: String,
        property: String,
        allowed: Set<String>,
        fallback: String,
        changes: MutableList<String>,
    ): String = replaceOutsideStrings(
        input,
        Regex("\\b$property\\s*=\\s*(\"(?:\\\\.|[^\"])*\"|-?\\d+(?:\\.\\d+)?)"),
        allowQuotedValue = true,
    ) { match ->
        val raw = match.groupValues[1]
        val token = raw.removeSurrounding("\"").trim().lowercase()
        if (token in allowed) match.value else {
            changes += "Normalized invalid $property value '$token' to '$fallback'."
            "$property=\"$fallback\""
        }
    }

    private fun replaceOutsideStrings(
        input: String,
        regex: Regex,
        allowQuotedValue: Boolean = false,
        transform: (MatchResult) -> String,
    ): String {
        val ranges = stringRanges(input)
        val matches = regex.findAll(input).filter { match ->
            ranges.none { range -> match.range.first in range || (!allowQuotedValue && match.range.last in range) }
        }.toList()
        if (matches.isEmpty()) return input
        val out = StringBuilder(input)
        matches.asReversed().forEach { match -> out.replace(match.range.first, match.range.last + 1, transform(match)) }
        return out.toString()
    }

    private fun stringRanges(input: String): List<IntRange> {
        val ranges = mutableListOf<IntRange>()
        var start = -1
        var escaped = false
        input.forEachIndexed { index, char ->
            if (start >= 0) {
                when {
                    escaped -> escaped = false
                    char == '\\' -> escaped = true
                    char == '"' -> { ranges += start..index; start = -1 }
                }
            } else if (char == '"') start = index
        }
        if (start >= 0) ranges += start until input.length
        return ranges
    }

    private fun lexicallyBalancedBody(value: String): Boolean {
        var depth = 0
        var inString = false
        var escaped = false
        value.forEach { char ->
            if (inString) {
                when {
                    escaped -> escaped = false
                    char == '\\' -> escaped = true
                    char == '"' -> inString = false
                }
            } else when (char) {
                '"' -> inString = true
                '(', '[', '{' -> depth += 1
                ')', ']', '}' -> if (--depth < 0) return false
            }
        }
        return depth == 0 && !inString
    }

    private fun exactFenceBody(value: String): String? =
        Regex("(?s)^```(?:a2ui|express)?\\s*\\n(.*)\\n```$").matchEntire(value)?.groupValues?.get(1)

    private fun IntRange.relativeTo(outer: IntRange): IntRange =
        (first - outer.first)..(last - outer.first)
}
