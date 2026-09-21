package com.samsung.genuicraft.sdk

import com.google.gson.Gson
import com.google.gson.JsonElement
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
import com.samsung.genuicraft.sdk.internal.pipeline.CanonicalGraphIdRewriter
import com.samsung.genuicraft.sdk.internal.pipeline.LiteralTextCodec

/**
 * Keeps source values out of the model's copy loop. The model generates the layout; this class
 * supplies typed literal values before normal Express compilation. It never supplies a fallback
 * layout. Unrecognized Markdown stays in a source block which the model must place in its layout.
 * Potential interpolation/state syntax is escaped as a literal string using the catalog's explicit
 * wire-compatible literal convention, rather than being evaluated by the renderer.
 */
internal class SourceBindings private constructor(
    val blocks: List<Map<String, Any>>,
    private val values: Map<String, JsonElement>,
) {
    private data class Binding(val block: Int, val kind: String, val role: String, val value: JsonElement)

    fun expand(program: String): String {
        require(program.trim().startsWith("<a2ui>") && program.trim().endsWith("</a2ui>")) {
            "Source-bound output must be one complete <a2ui> program."
        }
        // Parse first: inserted source can never become code, keys, identifiers, or another binding.
        val graph = A2uiExpressCodec.decode(program)
        require(graph.getAsJsonObject("state").size() == 0) { "Source-bound layouts cannot define state." }
        val elements = graph.getAsJsonObject("elements")
        val bindings = linkedMapOf<String, Binding>()
        blocks.forEachIndexed { index, block ->
            val kind = block.getValue("kind") as String
            bindingRoles.forEach { role ->
                (block[role] as? String)?.takeIf(values::containsKey)?.let { token ->
                    bindings[token] = Binding(index, kind, role, values.getValue(token))
                }
            }
        }
        val used = mutableSetOf<String>()
        val visited = mutableSetOf<String>()
        var nextBlock = 0

        fun visit(id: String, depth: Int) {
            require(depth <= 64) { "Source-bound layout nesting exceeds 64 levels." }
            require(visited.add(id)) { "Every component must be reachable exactly once; repeated or cyclic component '$id'." }
            val element = elements.get(id)?.takeIf { it.isJsonObject }?.asJsonObject
                ?: throw IllegalArgumentException("Missing source-bound component '$id'.")
            require(element.keySet().all { it in setOf("type", "props", "children") }) {
                "Source-bound layouts cannot define actions, visibility, repeats, or watch expressions."
            }
            val type = element.get("type").asString
            require(type in allowedProperties) { "Unsupported source-bound component '$type'." }
            val props = element.getAsJsonObject("props")
            val childValues = element.getAsJsonArray("children")
            if (type !in setOf("Stack", "Card")) {
                require(childValues.size() == 0) { "$type cannot contain child components in a source-bound layout." }
            }
            val fields = linkedMapOf<String, Binding>()
            props.entrySet().forEach { (key, raw) ->
                require(key in allowedProperties.getValue(type)) { "Unsupported source-bound property $type.$key." }
                if (key in contentProperties.getValue(type)) {
                    require(raw.isJsonPrimitive && raw.asJsonPrimitive.isString) {
                        "$type.$key must be one complete quoted source binding, not copied text or an expression."
                    }
                    val token = raw.asString
                    val binding = bindings[token]
                        ?: throw IllegalArgumentException("$type.$key must use a known source binding; received '$token'.")
                    require(accepts(type, key, binding.kind, binding.role)) {
                        "Source ${binding.kind}.${binding.role} cannot be placed in $type.$key."
                    }
                    require(used.add(token)) { "Use every source binding exactly once; repeated $token." }
                    fields[key] = binding
                } else {
                    require(raw.isJsonPrimitive && raw.asJsonPrimitive.isString &&
                        raw.asString in layoutValues.getValue(key)) {
                        "Unsupported source-bound layout value for $type.$key."
                    }
                }
            }
            require(requiredContent[type].orEmpty().all(fields::containsKey)) {
                "$type is missing required source-bound content."
            }
            if (type == "Text" && fields["text"]?.kind == "heading") {
                val variant = props.get("variant")?.asString?.lowercase()
                if (variant !in headingVariants) {
                    props.addProperty("variant", "heading")
                }
            }
            if (fields.isNotEmpty()) {
                val sourceIndices = fields.values.map { it.block }.toSet()
                require(sourceIndices.size == 1) { "Keep each source block together; $type mixes bindings from different blocks." }
                val blockIndex = sourceIndices.single()
                require(blockIndex == nextBlock) { "Preserve source block order: expected block $nextBlock, received $blockIndex." }
                val expectedRoles = bindings.values.filter { it.block == blockIndex }.map { it.role }.toSet()
                require(fields.values.map { it.role }.toSet() == expectedRoles) {
                    "Keep all fields of source block $blockIndex on one component (including table columns/rows or code/title)."
                }
                nextBlock++
                fields.forEach { (key, binding) -> props.add(key, LiteralTextCodec.protectValue(binding.value)) }
            } else if (type == "Divider") {
                require(nextBlock < blocks.size && blocks[nextBlock]["kind"] == "divider") {
                    "Divider must correspond to the next source divider."
                }
                nextBlock++
            }
            childValues.forEach { child ->
                require(child.isJsonPrimitive && child.asJsonPrimitive.isString) { "Children must be component identifiers." }
                visit(child.asString, depth + 1)
            }
        }

        require(elements.size() <= 1024) { "Source-bound layout exceeds 1,024 components." }
        visit(graph.get("root").asString, 0)
        require(visited.size == elements.size()) { "Every source-bound component must be reachable from root; orphan elements are not allowed." }
        require(used == values.keys && nextBlock == blocks.size) {
            "Use every source block exactly once and in order. Missing bindings: ${(values.keys - used).take(20)}"
        }
        val expanded = A2uiExpressCodec.encode(graph)
        require(A2uiExpressCodec.decode(expanded) == CanonicalGraphIdRewriter.rewrite(graph, shorten = true, reserveRoot = true)) {
            "Source content cannot round-trip through the Express literal codec."
        }
        return expanded
    }

    companion object {
        private const val PREFIX = "@source."
        private val gson = Gson()
        private val bindingRoles = setOf("text", "items", "columns", "rows", "code", "title")
        private val contentProperties = mapOf(
            "Stack" to emptySet(), "Card" to setOf("title"), "Text" to setOf("text"),
            "List" to setOf("items"), "Table" to setOf("columns", "rows"),
            "CodeBlock" to setOf("code", "title"), "Divider" to emptySet(),
        )
        private val allowedProperties = mapOf(
            "Stack" to setOf("direction", "gap", "align", "justify", "wrap"),
            "Card" to setOf("title"), "Text" to setOf("text", "variant"), "List" to setOf("items"),
            "Table" to setOf("columns", "rows", "domain", "preferredPresentation"),
            "CodeBlock" to setOf("code", "title"), "Divider" to emptySet(),
        )
        private val requiredContent = mapOf(
            "Text" to setOf("text"), "List" to setOf("items"),
            "Table" to setOf("columns", "rows"), "CodeBlock" to setOf("code"),
        )
        private val layoutValues = mapOf(
            "direction" to setOf("vertical", "horizontal"), "gap" to setOf("none", "sm", "md", "lg", "xl"),
            "align" to setOf("start", "center", "end", "stretch"),
            "justify" to setOf("start", "center", "end", "between", "around", "evenly", "spaceBetween", "spaceAround", "spaceEvenly"),
            "wrap" to setOf("wrap", "nowrap"),
            "variant" to setOf("body", "caption", "title", "heading", "headline", "h1", "h2", "h3", "h4", "h5", "h6"),
            "domain" to setOf("weather", "flight", "booking", "schedule", "status", "comparison", "generic"),
            "preferredPresentation" to setOf("cards", "table", "auto"),
        )
        private val headingVariants = setOf("title", "heading", "headline", "h1", "h2", "h3", "h4", "h5", "h6")
        private val heading = Regex("^ {0,3}#{1,6}[ \\t]+(.+?)\\s*$")
        private val orderedListItem = Regex("^( {0,3})\\d+[.)][ \\t]+(.+)$")
        private val listItem = Regex("^( {0,3})(?:[-+*]|\\d+[.)])[ \\t]+(.+)$")
        private val fence = Regex("^ {0,3}(`{3,}|~{3,})(.*)$")
        private val separator = Regex(":?-{3,}:?")
        private val rule = Regex("^ {0,3}(?:-{3,}|(?:\\*\\s*){3,}|(?:_\\s*){3,})$")

        private fun accepts(type: String, key: String, kind: String, role: String): Boolean = when (kind) {
            "heading" -> role == "text" && (type == "Card" && key == "title" || type == "Text" && key == "text")
            "paragraph" -> type == "Text" && key == "text" && role == "text"
            "list" -> type == "List" && key == "items" && role == "items"
            "table" -> type == "Table" && key == role && role in setOf("columns", "rows")
            "code" -> type == "CodeBlock" && key == role && role in setOf("code", "title")
            else -> false
        }

        fun from(text: String): SourceBindings {
            val values = linkedMapOf<String, JsonElement>()
            val blocks = mutableListOf<Map<String, Any>>()
            fun bind(value: Any): String {
                require(values.size < 512) { "Input has too many source fields (maximum 512)." }
                val name = PREFIX + alphabeticId(values.size)
                val json = gson.toJsonTree(value)
                values[name] = json
                return name
            }
            fun paragraph(value: String) {
                blocks += linkedMapOf("kind" to "paragraph", "text" to bind(value), "value" to value)
            }
            val lines = text.replace("\r\n", "\n").replace('\r', '\n').split('\n')
            var index = 0
            while (index < lines.size) {
                val line = lines[index]
                if (line.isBlank()) { index++; continue }
                val codeStart = fence.matchEntire(line)
                if (codeStart != null) {
                    val marker = codeStart.groupValues[1]
                    val close = ((index + 1) until lines.size).firstOrNull { candidate ->
                        val candidateLine = lines[candidate]
                        val trimmed = candidateLine.trim()
                        candidateLine.takeWhile { it == ' ' }.length <= 3 &&
                            trimmed.length >= marker.length && trimmed.all { it == marker.first() }
                    }
                    if (close != null) {
                        val code = lines.subList(index + 1, close).joinToString("\n")
                        val info = codeStart.groupValues[2].trim()
                        blocks += linkedMapOf<String, Any>("kind" to "code", "code" to bind(code), "value" to code).apply {
                            if (info.isNotEmpty()) { put("title", bind(info)); put("titleValue", info) }
                        }
                        index = close + 1
                        continue
                    }
                    // An unclosed fence makes the remainder ambiguous; retain it as source text.
                    paragraph(lines.subList(index, lines.size).joinToString("\n"))
                    break
                }
                val title = heading.matchEntire(line)
                if (title != null) {
                    // A closing hash sequence is markup only when preceded by whitespace (C# stays C#).
                    val value = title.groupValues[1].replace(Regex("[ \\t]+#+[ \\t]*$"), "")
                    blocks += linkedMapOf("kind" to "heading", "text" to bind(value), "value" to value)
                    index++
                    continue
                }
                val table = tableAt(lines, index)
                if (table != null) {
                    blocks += linkedMapOf(
                        "kind" to "table", "columns" to bind(table.columns), "rows" to bind(table.rows),
                        "value" to mapOf("columns" to table.columns, "rows" to table.rows),
                    )
                    index = table.end
                    continue
                }
                if (rule.matches(line)) {
                    blocks += mapOf("kind" to "divider")
                    index++
                    continue
                }
                if (orderedListItem.matches(line)) {
                    // Numeric markers can carry authored meaning (years, legal clauses, non-1 starts).
                    // Preserve the complete lines instead of replacing them with generic bullets.
                    val start = index++
                    while (index < lines.size && orderedListItem.matches(lines[index])) index++
                    paragraph(lines.subList(start, index).joinToString("\n"))
                    continue
                }
                if (listItem.matches(line)) {
                    val items = mutableListOf<String>()
                    while (index < lines.size) {
                        val item = listItem.matchEntire(lines[index]) ?: break
                        items += item.groupValues[1] + item.groupValues[2]
                        index++
                    }
                    blocks += linkedMapOf("kind" to "list", "items" to bind(items), "value" to items)
                    continue
                }
                val start = index++
                while (index < lines.size && lines[index].isNotBlank() &&
                    !heading.matches(lines[index]) && !listItem.matches(lines[index]) &&
                    !fence.matches(lines[index]) && !rule.matches(lines[index]) && tableAt(lines, index) == null
                ) index++
                paragraph(lines.subList(start, index).joinToString("\n"))
            }
            return SourceBindings(blocks, values)
        }

        private data class SourceTable(val columns: List<String>, val rows: List<List<String>>, val end: Int)

        private fun tableAt(lines: List<String>, start: Int): SourceTable? {
            if (start + 1 >= lines.size || '|' !in lines[start]) return null
            val columns = cells(lines[start]) ?: return null
            val separators = cells(lines[start + 1]) ?: return null
            if (columns.size < 2 || separators.size != columns.size || separators.any { !separator.matches(it) }) return null
            val rows = mutableListOf<List<String>>()
            var end = start + 2
            while (end < lines.size && lines[end].isNotBlank() && '|' in lines[end]) {
                val row = cells(lines[end]) ?: return null
                // A malformed/ambiguous table is preserved verbatim, not partially consumed.
                if (row.size != columns.size) return null
                rows += row
                end++
            }
            if (rows.isEmpty()) return null
            return SourceTable(columns, rows, end)
        }

        private fun cells(line: String): List<String>? {
            val text = line.trim()
            val cells = mutableListOf<String>()
            val cell = StringBuilder()
            var codeTicks = 0
            var index = 0
            while (index < text.length) {
                val ch = text[index]
                when {
                    ch == '\\' -> {
                        // Preserve escaped data and honor backslash parity before a delimiter.
                        cell.append(ch)
                        if (index + 1 < text.length) { index++; cell.append(text[index]) }
                    }
                    ch == '`' -> {
                        val count = text.substring(index).takeWhile { it == '`' }.length
                        codeTicks = if (codeTicks == count) 0 else if (codeTicks == 0) count else codeTicks
                        repeat(count) { cell.append('`') }
                        index += count - 1
                    }
                    ch == '|' && codeTicks == 0 -> { cells += cell.toString().trim(); cell.setLength(0) }
                    else -> cell.append(ch)
                }
                index++
            }
            if (codeTicks != 0) return null
            cells += cell.toString().trim()
            if (text.startsWith('|')) cells.removeAt(0)
            if (text.endsWith('|') && cells.lastOrNull() == "") cells.removeAt(cells.lastIndex)
            return cells
        }

        private fun alphabeticId(index: Int): String {
            var n = index + 1
            var result = ""
            while (n > 0) { n--; result = ('a' + n % 26) + result; n /= 26 }
            return result
        }
    }
}
