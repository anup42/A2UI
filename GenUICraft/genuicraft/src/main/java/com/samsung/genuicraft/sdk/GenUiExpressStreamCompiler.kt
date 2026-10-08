package com.samsung.genuicraft.sdk

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiCanonicalGraph
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressPartialState
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressStreamScanner
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiWireCodec
import com.samsung.genuicraft.sdk.internal.pipeline.RendererReferenceSemantics

internal data class GenUiStreamPreview(val document: GenUiDocument, val readyComponentCount: Int)

/**
 * One compiler per generation attempt. Input is the exact cumulative provider snapshot.
 * Pending syntax and graph edges remain in the model program; only a separate, validated
 * projection is emitted. Final acceptance and generated-output repair belong to the caller.
 */
internal class GenUiExpressStreamCompiler {
    private var previousRaw = ""
    private var offset: Int? = null
    private val statements = mutableListOf<String>()
    private val assignedIds = mutableSetOf<String>()
    private var statementCount = 0
    private var graph = emptyGraph()
    private var previousWire: String? = null
    private var failed = false

    /** Preview errors must never escape into a native generation callback. */
    fun accept(rawText: String): GenUiStreamPreview? {
        if (failed || rawText == previousRaw) return null
        return try {
            require(rawText.length <= A2uiExpressCodec.MAX_INPUT_CHARS) { "Express stream size limit." }
            require(rawText.startsWith(previousRaw)) { "A cumulative Express stream changed its prefix." }
            previousRaw = rawText
            if (offset == null) {
                val start = rawText.indexOfFirst { !it.isWhitespace() }
                if (start < 0) return null
                val prefix = rawText.substring(start)
                if ("<a2ui>".startsWith(prefix)) return null
                require(prefix.startsWith("<a2ui>")) { "Unsupported Express stream prefix." }
                offset = start + 6
            }
            val scan = A2uiExpressStreamScanner.scan(rawText, requireNotNull(offset))
            scan.statements.forEach(::consume)
            offset = scan.nextOffset
            if (!graph.getAsJsonObject("elements").has("root")) return null
            val candidate = graph.deepCopy()
            ensureAcyclic(candidate)
            val arrayPaths = boundArrayPaths(candidate)
            A2uiExpressPartialState.arrays(scan.tail, arrayPaths).forEach { (path, rows) ->
                setPath(candidate.getAsJsonObject("state"), path, rows)
            }
            val projection = project(candidate) ?: return null
            require(A2uiCanonicalGraph.validate(projection).isValid) { "Invalid Express preview projection." }
            val wire = A2uiWireCodec.encode(projection, shortenIds = false).toString()
            if (wire == previousWire) return null
            val document = GenUiDocument(
                express = A2uiExpressCodec.encode(projection, shortenIds = false),
                a2uiJson = wire,
            )
            previousWire = wire
            GenUiStreamPreview(document, projection.getAsJsonObject("elements").size())
        } catch (_: Exception) {
            failed = true
            null
        }
    }

    private fun consume(statement: String) {
        statementCount += 1
        require(statementCount <= A2uiExpressCodec.MAX_STATEMENTS) { "Express statement limit." }
        val assignment = runCatching { A2uiExpressCodec.splitAssignment(statement) }.getOrNull() ?: return
        if (!assignment.first.startsWith('$')) {
            require(assignedIds.add(assignment.first)) { "Duplicate Express component id." }
            // An unsupported closed component is still pending for its parents. Valid siblings
            // can continue, without interpreting its literals or replacing its structure.
            val result = runCatching {
                A2uiExpressCodec.decodeStatements(statements + statement, requireRoot = false)
            }
            result.exceptionOrNull()?.let { error ->
                val reason = error.message.orEmpty()
                if (reason.startsWith("Duplicate") || reason.contains("more than") || reason.contains("nesting")) throw error
            }
            val parsed = result.getOrNull() ?: return
            graph = parsed
        } else {
            graph = A2uiExpressCodec.decodeStatements(statements + statement, requireRoot = false)
        }
        statements += statement
    }

    private fun ensureAcyclic(source: JsonObject) {
        val elements = source.getAsJsonObject("elements")
        val visiting = mutableSetOf<String>()
        val visited = mutableSetOf<String>()
        fun visit(id: String, depth: Int) {
            if (!elements.has(id) || id in visited) return
            require(depth <= A2uiExpressCodec.MAX_EXPRESSION_DEPTH && visiting.add(id)) {
                "Express stream cycle or reference depth limit."
            }
            RendererReferenceSemantics.references(elements.getAsJsonObject(id)).forEach { visit(it.targetId, depth + 1) }
            visiting.remove(id)
            visited.add(id)
        }
        visit("root", 1)
    }

    private fun boundArrayPaths(source: JsonObject): Set<String> {
        val elements = source.getAsJsonObject("elements")
        val paths = mutableSetOf<String>()
        val visited = mutableSetOf<String>()
        fun visit(id: String) {
            if (!visited.add(id) || !elements.has(id)) return
            val element = elements.getAsJsonObject(id)
            val props = element.getAsJsonObject("props")
            if (element.text("type") in setOf("Table", "Chart")) {
                val rows = props.get("rows")
                if (rows?.isJsonArray != true) {
                    listOf("statePath", "rowsPath", "dataPath").forEach { key ->
                        props.text(key)?.let(::pointer)?.let(paths::add)
                    }
                    bindingPath(rows)?.let(paths::add)
                }
            }
            RendererReferenceSemantics.references(element).forEach { visit(it.targetId) }
        }
        visit("root")
        return paths
    }

    private data class Branch(val elements: JsonObject, val hasContent: Boolean)

    private fun project(source: JsonObject): JsonObject? {
        val elements = source.getAsJsonObject("elements")
        val state = source.getAsJsonObject("state")
        val memo = mutableMapOf<Pair<String, Boolean>, Branch?>()
        var computations = 0
        fun branch(id: String, itemScope: Boolean = false): Branch? {
            val key = id to itemScope
            if (memo.containsKey(key)) return memo[key]
            // Cache pending/hidden results too. The acyclic input check makes this sentinel safe.
            memo[key] = null
            require(++computations <= A2uiExpressCodec.MAX_ELEMENTS * 2) { "Express projection work limit." }
            val original = elements.get(id)?.takeIf { it.isJsonObject }?.asJsonObject ?: return null
            val visible = original.get("visible")
            if (visible?.isJsonPrimitive == true && visible.asJsonPrimitive.isBoolean && !visible.asBoolean) return null
            if (!bindingsReady(original, state, itemScope)) return null
            if (visible?.isJsonObject == true && visible.asJsonObject.keySet().all {
                    it in setOf("path", "\$state", "\$bindState")
                }) {
                val resolved = bindingPath(visible)?.let { atPath(state, it) }
                if (resolved?.isJsonPrimitive == true && resolved.asJsonPrimitive.isBoolean && !resolved.asBoolean) return null
            }
            val repeat = original.get("repeat")?.takeIf { it.isJsonObject }?.asJsonObject
            if (repeat != null) {
                val rows = repeat.text("statePath")?.let { atPath(state, it) }
                if (rows?.isJsonArray != true || rows.asJsonArray.size() == 0) return null
            }
            val type = original.text("type")
            val props = original.getAsJsonObject("props")
            if (type in setOf("Table", "Chart") && !tableDataReady(props, state)) return null
            val atomic = repeat != null || type in setOf("Tabs", "Modal")
            val children = JsonArray()
            val projected = JsonObject()
            var hasContent = ownContent(type, props)
            val childBranches = mutableMapOf<String, Branch?>()
            fun resolve(target: String): Branch? = childBranches.getOrPut(target) { branch(target, itemScope || repeat != null) }
            original.getAsJsonArray("children").forEach { child ->
                val ready = resolve(child.asString)
                if (ready == null && atomic) return null
                if (ready != null) {
                    children.add(child.asString)
                    ready.elements.entrySet().forEach { (key, value) -> projected.add(key, value) }
                    hasContent = hasContent || ready.hasContent
                }
            }
            RendererReferenceSemantics.references(original).filter { it.kind != "child" }.forEach { reference ->
                val ready = resolve(reference.targetId) ?: return null
                ready.elements.entrySet().forEach { (key, value) -> projected.add(key, value) }
                hasContent = hasContent || ready.hasContent
            }
            val element = original.deepCopy().apply { add("children", children) }
            projected.add(id, element)
            val local = JsonObject().apply {
                addProperty("root", id)
                add("state", state)
                add("elements", projected)
            }
            if (!A2uiCanonicalGraph.validate(local, requireReservedRoot = false).isValid) return null
            return Branch(projected, hasContent).also { memo[key] = it }
        }
        val root = branch("root") ?: return null
        if (!root.hasContent) return null
        return JsonObject().apply {
            addProperty("root", "root")
            add("state", state.deepCopy())
            add("elements", root.elements)
        }
    }

    private fun tableDataReady(props: JsonObject, state: JsonObject): Boolean {
        val rows = props.get("rows")
        if (rows != null) return rows.isJsonArray || bindingPath(rows)?.let { atPath(state, it)?.isJsonArray } == true
        val path = listOf("statePath", "rowsPath", "dataPath").firstNotNullOfOrNull { props.text(it) }
        return path == null || atPath(state, path)?.isJsonArray == true
    }

    private fun bindingsReady(value: JsonElement?, state: JsonObject, itemScope: Boolean): Boolean {
        if (value == null || value.isJsonNull) return true
        if (value.isJsonArray) return value.asJsonArray.all { bindingsReady(it, state, itemScope) }
        if (value.isJsonPrimitive) {
            if (!value.asJsonPrimitive.isString) return true
            val text = value.asString
            if (text == "$" || text.startsWith("$/") || text.startsWith("\$state.")) return atPath(state, text) != null
            if ("\$item" in text || "\$index" in text) return itemScope
            return true
        }
        val obj = value.asJsonObject
        bindingPath(obj)?.let { if (atPath(state, it) == null) return false }
        if (obj.has("\$computed")) return false // No host function registry is available at this compiler boundary.
        if ((obj.has("\$item") || obj.has("\$bindItem") || obj.has("\$index")) && !itemScope) return false
        return obj.entrySet().all { (key, child) ->
            when {
                key in setOf("statePath", "rowsPath", "dataPath") && child.isJsonPrimitive && child.asJsonPrimitive.isString ->
                    atPath(state, child.asString) != null
                key == "watch" && child.isJsonObject -> child.asJsonObject.entrySet().all { (path, action) ->
                    atPath(state, path) != null && bindingsReady(action, state, itemScope)
                }
                else -> bindingsReady(child, state, itemScope)
            }
        }
    }

    private fun ownContent(type: String?, props: JsonObject): Boolean = when (type) {
        "Stack", "Divider", "Tabs", "Modal" -> false
        "Card" -> listOf("title", "subtitle").any { !props.text(it).isNullOrBlank() }
        "List" -> props.get("items")?.takeIf { it.isJsonArray }?.asJsonArray?.size()?.let { it > 0 } == true
        else -> true
    }

    private fun bindingPath(value: JsonElement?): String? {
        if (value?.isJsonPrimitive == true && value.asJsonPrimitive.isString) {
            val text = value.asString
            return text.takeIf { it == "$" || it.startsWith("$/") || it.startsWith("\$state.") }?.let(::pointer)
        }
        if (value?.isJsonObject != true) return null
        val obj = value.asJsonObject
        return listOf("\$state", "\$bindState", "path").firstNotNullOfOrNull { obj.text(it) }?.let(::pointer)
    }

    private fun pointer(path: String): String? = when {
        path == "$" || path == "/" -> "/"
        path.startsWith("$/") -> path.removePrefix("$")
        path.startsWith("\$state.") -> "/" + path.removePrefix("\$state.").replace('.', '/')
        path.startsWith('/') -> path
        else -> null
    }

    private fun atPath(state: JsonObject, path: String): JsonElement? {
        val normalized = pointer(path) ?: return null
        if (normalized == "/") return state
        var current: JsonElement = state
        for (part in normalized.split('/').drop(1)) {
            val key = part.replace("~1", "/").replace("~0", "~")
            current = when {
                current.isJsonObject -> current.asJsonObject.get(key)
                current.isJsonArray -> key.toIntOrNull()?.let { current.asJsonArray.getOrNull(it) }
                else -> null
            } ?: return null
        }
        return current.takeUnless { it.isJsonNull }
    }

    private fun setPath(state: JsonObject, path: String, value: JsonElement) {
        val parts = path.split('/').drop(1).map { it.replace("~1", "/").replace("~0", "~") }
        if (parts.isEmpty()) return
        var current = state
        parts.dropLast(1).forEach { key ->
            current = current.get(key)?.takeIf { it.isJsonObject }?.asJsonObject
                ?: JsonObject().also { current.add(key, it) }
        }
        current.add(parts.last(), value.deepCopy())
    }

    private fun JsonArray.getOrNull(index: Int): JsonElement? = if (index in 0 until size()) get(index) else null
    private fun JsonObject.text(key: String): String? = get(key)?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString

    private fun emptyGraph(): JsonObject = JsonObject().apply {
        addProperty("root", "root")
        add("state", JsonObject())
        add("elements", JsonObject())
    }
}
