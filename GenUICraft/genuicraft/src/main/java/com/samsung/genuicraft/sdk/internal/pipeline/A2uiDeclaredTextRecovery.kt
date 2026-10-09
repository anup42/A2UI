package com.samsung.genuicraft.sdk.internal.pipeline

import com.google.gson.JsonElement
import com.google.gson.JsonObject

/**
 * Generated-only parser islands for declared Text calls. This never consults source text.
 * Unsafe/ambiguous declarations are removed from this candidate so literal-only fallback
 * cannot turn their actions, guards or bindings into unconditional answer prose.
 */
internal object A2uiDeclaredTextRecovery {
    data class Prepared(val input: String, val changes: List<String>, val allowLiteralOnlyFallback: Boolean = true)
    private data class Definition(
        val id: String, val start: Int, val end: Int, val statement: String,
        val separatorBoundary: Boolean, val ambiguousStart: Boolean,
    )
    private data class TextCall(val body: JsonElement, val variant: JsonElement?, val source: JsonElement?)
    private data class Context(val references: Set<String>, val guarded: Boolean)
    private data class Lexed(val outside: BooleanArray, val syntax: String, val closedComments: Boolean)

    private val header = Regex("([A-Za-z_][A-Za-z0-9_]*|\\$(?:/[^\\s=]*)?)\\s*=")
    private val callHeader = Regex("\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*\\(")
    private val identifier = Regex("[A-Za-z_][A-Za-z0-9_]*")
    private val citationLiteral = Regex("(?:\\[[0-9]+])+")
    private val bindingPrefix = Regex("^(?:\\$[/A-Za-z_]|@source(?:\\.|$))")
    private val inlineBinding = Regex(
        """[$]\{[^}]+\}|\{\{\s*[$]item[./][^}]+?\s*\}\}|(?<![$])\{\s*[$]item[./][^}]+?\s*\}|(?<![$])\{\s*[$]index\s*(?:\+\s*1)?\s*\}""",
    )
    private val bindingKeys = setOf(
        "\$state", "\$bindState", "\$item", "\$bindItem", "\$index", "\$cond", "\$template", "\$computed",
        "statePath", "rowsPath", "dataPath",
    )
    private val metadata = setOf("visible", "repeat", "on", "watch")
    private val guardedTypes = setOf("Modal", "Tabs")
    private val gap = Regex("\\*\\*(none|sm|md|lg|xl)\\*\\*")

    fun prepare(input: String, maxCalls: Int, maxLiteralChars: Int, maxLiteralLength: Int): Prepared {
        val trimmed = input.trim()
        if (input.length > A2uiExpressCodec.MAX_INPUT_CHARS ||
            !trimmed.startsWith("<a2ui>") || !trimmed.endsWith("</a2ui>")) return Prepared(input, emptyList())
        val body = trimmed.substring("<a2ui>".length, trimmed.length - "</a2ui>".length)
        val lexed = lex(body)
        if (!lexed.closedComments) return Prepared(input, emptyList(), allowLiteralOnlyFallback = false)
        val definitions = definitions(body, lexed) ?: return Prepared(input, emptyList())
        val duplicateIds = definitions.filterNot { it.id.startsWith('$') }
            .groupingBy { it.id }.eachCount().filterValues { it > 1 }.keys
        val knownIds = definitions.map { it.id }.filterNot { it.startsWith('$') }.toSet()
        val parsed = linkedMapOf<String, TextCall>()
        val strict = linkedMapOf<String, Map<String, JsonObject>>()
        val textIds = linkedSetOf<String>()
        val changes = mutableListOf<String>()
        definitions.filterNot { it.id.startsWith('$') }.forEach { definition ->
            val rhs = definition.statement.substringAfter('=').trimStart()
            val type = callHeader.find(rhs)?.takeIf { it.range.first == 0 }?.groupValues?.get(1)
            decodeElements(definition.statement)?.let { strict[definition.id] = it }
            if (type.equals("Text", ignoreCase = true)) {
                textIds += definition.id
                if (definition.id !in strict && definition.id !in duplicateIds && !definition.ambiguousStart) {
                    parseText(rhs)?.let { parsed[definition.id] = it }
                }
            }
        }
        val contexts = linkedMapOf<String, Context>()
        fun addContext(id: String, context: Context) {
            val previous = contexts[id]
            // A separately declared ID can collide with a codec-generated inline ID. It cannot
            // establish one unambiguous path, even when both declarations look individually safe.
            contexts[id] = if (previous == null) context else Context(previous.references + context.references, true)
        }
        definitions.filterNot { it.id.startsWith('$') }.forEach { definition ->
            val elements = if (definition.id in duplicateIds) null else (
                strict[definition.id] ?: decodeElements(normalizeGapForContext(definition.statement))
                    ?: parsed[definition.id]?.let { decodeElements(textStatement(definition.id, it)) }
                )
            if (elements != null) {
                // Keep every materialized inline node: its metadata and reference edges determine
                // whether a named Text is guarded, rather than an independent static orphan.
                elements.forEach { (id, element) ->
                    addContext(id, Context(RendererReferenceSemantics.references(element).map { it.targetId }.toSet(),
                        metadata.any(element::has) || element.get("type")?.asString in guardedTypes ||
                            hasBinding(element.get("props"))))
                }
            } else {
                // An opaque ancestor cannot prove that its descendants are unconditional.
                addContext(definition.id, Context(possibleReferences(definition.statement, knownIds) - definition.id, true))
            }
        }
        val referenced = contexts.values.flatMap { it.references }.toSet()
        val safe = linkedSetOf<String>()
        fun visit(id: String, depth: Int) {
            if (depth >= A2uiExpressCodec.MAX_EXPRESSION_DEPTH || id in safe) return
            val context = contexts[id] ?: return
            if (context.guarded) return
            safe += id
            context.references.forEach { visit(it, depth + 1) }
        }
        visit("root", 0)
        val rootStatic = "root" in safe
        val reserved = (knownIds + contexts.keys).toMutableSet()
        fun fresh(base: String): String {
            var id = base
            var suffix = 0
            while (!reserved.add(id)) id = base + "_" + (++suffix)
            return id
        }
        val replacements = linkedMapOf<Int, String>()
        var recovered = 0
        var characters = 0
        var blocked = false
        definitions.filter { it.id in textIds && (it.id !in strict || it.id in duplicateIds) }.forEach { definition ->
            val call = parsed[definition.id]
            val eligible = call != null && definition.id !in duplicateIds &&
                (definition.id in safe || (rootStatic && definition.id !in referenced))
            val size = call?.let { it.body.asString.length + (it.source?.asString?.length ?: 0) } ?: 0
            if (!eligible || call == null || recovered >= maxCalls ||
                call.body.asString.length > maxLiteralLength ||
                (call.source?.asString?.length ?: 0) > maxLiteralLength ||
                characters + size > maxLiteralChars) {
                replacements[definition.start] = "// Omitted unsafe or ambiguous declared Text: " + definition.id
                blocked = true
                changes += "Did not promote unsafe, guarded, bound, duplicate or over-budget Text '" + definition.id + "'."
            } else {
                replacements[definition.start] = if (call.source?.asString.isNullOrEmpty()) {
                    textStatement(definition.id, call)
                } else {
                    val bodyId = fresh(definition.id + "_recovered_body")
                    val citationId = fresh(definition.id + "_recovered_citation")
                    definition.id + "=Column([" + bodyId + "," + citationId + "],gap=\"sm\")\n" +
                        textStatement(bodyId, call) + "\n" +
                        citationId + "=Text(" + call.source + ",variant=\"caption\")"
                }
                characters += size
                recovered++
                changes += "Recovered exact declared Text '" + definition.id + "' and its complete citation literal."
            }
        }
        if (replacements.isEmpty()) return Prepared(input, emptyList())
        val rewritten = StringBuilder()
        var cursor = 0
        definitions.forEach { definition ->
            val replacement = replacements[definition.start]
            if (replacement == null && !definition.separatorBoundary) return@forEach
            rewritten.append(lexed.syntax, cursor, definition.start)
            rewritten.append(replacement ?: definition.statement)
            rewritten.append('\n')
            cursor = definition.end
        }
        rewritten.append(lexed.syntax, cursor, body.length)
        return Prepared("<a2ui>\n" + rewritten.toString().trim() + "\n</a2ui>", changes.distinct(),
            allowLiteralOnlyFallback = !blocked)
    }

    /** Decode context without requiring referenced children to be present in this one statement. */
    private fun decodeElements(statement: String): Map<String, JsonObject>? = runCatching {
        A2uiExpressCodec.decodeStatements(listOf(statement), requireRoot = false)
            .getAsJsonObject("elements").entrySet().associate { (id, element) -> id to element.asJsonObject }
    }.getOrNull()

    private fun textStatement(id: String, call: TextCall): String =
        id + "=Text(" + call.body + (call.variant?.let { ",variant=" + it } ?: "") + ")"

    /** Only fully understood literal tails are allowed; no expression or metadata is discarded. */
    private fun parseText(rhs: String): TextCall? = runCatching {
        val opening = callHeader.matchAt(rhs, 0) ?: error("Missing declared call.")
        require(opening.groupValues[1].equals("Text", ignoreCase = true))
        var index = opening.range.last + 1
        fun whitespace() { while (index < rhs.length && rhs[index].isWhitespace()) index++ }
        fun literal(): JsonElement {
            whitespace()
            val (value, consumed) = A2uiExpressCodec.literalPrefix(rhs.substring(index))
            require(value.isJsonPrimitive && value.asJsonPrimitive.isString)
            require(!isBoundString(value.asString))
            index += consumed
            return value
        }
        val value = literal()
        var variant: JsonElement? = null
        var source: JsonElement? = null
        var named = false
        whitespace()
        while (rhs.getOrNull(index) == ',') {
            index++; whitespace()
            if (rhs.getOrNull(index) == '"' || rhs.startsWith("r\"", index, ignoreCase = true)) {
                require(!named && variant == null)
                variant = literal()
            } else {
                val key = identifier.matchAt(rhs, index) ?: error("Unsupported tail.")
                index = key.range.last + 1; whitespace()
                val separator = rhs.getOrNull(index)
                require(separator == '=' || (key.value == "source" && separator == ':'))
                index++; named = true
                when (key.value) {
                    "variant" -> { require(variant == null); variant = literal() }
                    "source" -> {
                        require(source == null)
                        source = literal()
                        require(source.asString.isEmpty() || citationLiteral.matches(source.asString))
                    }
                    else -> error("Actions, guards, bindings and unknown metadata are not literal Text.")
                }
            }
            whitespace()
        }
        require(rhs.getOrNull(index)?.let { it in ")}]" } == true)
        index++; whitespace()
        require(index == rhs.length)
        TextCall(value, variant, source)
    }.getOrNull()

    /** Context-only normalization; production candidate gap syntax is not rewritten here. */
    private fun normalizeGapForContext(statement: String): String {
        val outside = outside(statement)
        val result = StringBuilder(statement)
        gap.findAll(statement).filter { outside[it.range.first] }.toList().asReversed().forEach {
            result.replace(it.range.first, it.range.last + 1, "\"" + it.groupValues[1] + "\"")
        }
        return result.toString()
    }

    private fun hasBinding(value: JsonElement?): Boolean = when {
        value == null -> false
        value.isJsonPrimitive && value.asJsonPrimitive.isString -> isBoundString(value.asString)
        value.isJsonObject -> value.asJsonObject.entrySet().any { (key, child) ->
            key in bindingKeys || hasBinding(child)
        }
        value.isJsonArray -> value.asJsonArray.any(::hasBinding)
        else -> false
    }

    private fun isBoundString(value: String): Boolean =
        bindingPrefix.containsMatchIn(value) || inlineBinding.containsMatchIn(value)

    private fun possibleReferences(statement: String, ids: Set<String>): Set<String> {
        val mask = outside(statement)
        val result = linkedSetOf<String>()
        identifier.findAll(statement).filter { mask[it.range.first] && it.value in ids }.forEach { result += it.value }
        var index = 0
        while (index < statement.length) {
            if (!mask[index] && (index == 0 || mask[index - 1]) &&
                (statement[index] == '"' || statement.startsWith("r\"", index, ignoreCase = true))) {
                runCatching { A2uiExpressCodec.literalPrefix(statement.substring(index)) }.getOrNull()?.let { (literal, consumed) ->
                    if (literal.isJsonPrimitive && literal.asJsonPrimitive.isString && literal.asString in ids) result += literal.asString
                    index += consumed
                } ?: run { index++ }
            } else index++
        }
        return result
    }

    /** Assignment boundaries are read by a typed lexer, including proven shallow wrong Text closers. */
    private fun definitions(body: String, lexed: Lexed): List<Definition>? {
        val mask = lexed.outside
        val syntax = lexed.syntax
        val definitions = mutableListOf<Definition>()
        var cursor = 0
        var ambiguous = false
        while (cursor < body.length) {
            while (cursor < body.length && (syntax[cursor].isWhitespace() || !mask[cursor])) cursor++
            if (cursor >= body.length) break
            val match = header.matchAt(syntax, cursor)
            if (match == null) {
                cursor = body.indexOf('\n', cursor).let { if (it < 0) body.length else it + 1 }
                ambiguous = true
                continue
            }
            if (definitions.size >= A2uiExpressCodec.MAX_STATEMENTS) return null
            val start = cursor
            val rhsStart = match.range.last + 1
            val call = callHeader.matchAt(syntax, rhsStart)
            val text = call?.groupValues?.get(1)?.equals("Text", ignoreCase = true) == true
            val stack = java.util.ArrayDeque<Char>()
            var end = rhsStart
            var separator = false
            var nextAmbiguous = false
            var statementEnd = body.length
            while (end < body.length) {
                if (!mask[end]) { end++; continue }
                val ch = syntax[end]
                if (ch == '\n' || ch == ';') {
                    var next = end + 1
                    while (next < body.length && syntax[next].isWhitespace()) next++
                    if (stack.isEmpty() || (next < body.length && mask[next] && header.matchAt(syntax, next) != null)) {
                        statementEnd = end
                        separator = ch == ';'
                        nextAmbiguous = stack.isNotEmpty()
                        end++
                        break
                    }
                }
                if (ch in "([{") stack.addLast(ch)
                else if (ch in ")]}") {
                    val expected = when (stack.peekLast()) { '(' -> ')'; '[' -> ']'; '{' -> '}'; else -> null }
                    if (ch == expected) stack.removeLast()
                    else if (text && stack.size == 1 && stack.peekLast() == '(') stack.removeLast()
                    else nextAmbiguous = true
                }
                if (ch == ',' && stack.isEmpty()) {
                    var next = end + 1
                    while (next < body.length && syntax[next].isWhitespace()) next++
                    if (next < body.length && mask[next] && header.matchAt(syntax, next) != null) {
                        statementEnd = end
                        separator = true
                        end++
                        break
                    }
                }
                end++
            }
            definitions += Definition(match.groupValues[1], start, end,
                syntax.substring(start, statementEnd).trim(), separator, ambiguous)
            cursor = end
            ambiguous = nextAmbiguous
        }
        return definitions
    }

    /** Production quote forms and comments stay opaque; an incomplete quote hides the rest. */
    private fun outside(input: String): BooleanArray = lex(input).outside

    /** Comment-free syntax preserves offsets and all quoted bytes, including raw/triple literals. */
    private fun lex(input: String): Lexed {
        val mask = BooleanArray(input.length) { true }
        val syntax = input.toCharArray()
        var closedComments = true
        fun comment(start: Int, end: Int) {
            for (i in start until end) {
                mask[i] = false
                if (syntax[i] != '\n' && syntax[i] != '\r') syntax[i] = ' '
            }
        }
        var index = 0
        while (index < input.length) {
            val start = index
            val raw = input.startsWith("r\"", index, ignoreCase = true) &&
                (index == 0 || (!input[index - 1].isLetterOrDigit() && input[index - 1] != '_'))
            val quote = index + if (raw) 1 else 0
            if (input.getOrNull(quote) == '"') {
                val delimiter = if (input.startsWith("\"\"\"", quote)) "\"\"\"" else "\""
                index = quote + delimiter.length
                var escaped = false
                var closed = false
                while (index < input.length) {
                    if (input.startsWith(delimiter, index) && (raw || !escaped)) {
                        index += delimiter.length; closed = true; break
                    }
                    val ch = input[index++]
                    if (!raw) {
                        if (escaped) escaped = false else if (ch == '\\') escaped = true
                    }
                }
                for (i in start until index) mask[i] = false
                if (!closed) break
            } else if (input.startsWith("/*", index)) {
                val closing = input.indexOf("*/", index + 2)
                if (closing < 0) closedComments = false
                index = if (closing < 0) input.length else closing + 2
                comment(start, index)
            } else if (input.startsWith("//", index) || input[index] == '#') {
                index = input.indexOf('\n', index).let { if (it < 0) input.length else it }
                comment(start, index)
            } else index++
        }
        return Lexed(mask, String(syntax), closedComments)
    }
}
