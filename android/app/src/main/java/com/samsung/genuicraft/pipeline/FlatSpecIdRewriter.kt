package com.samsung.genuicraft.pipeline

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonPrimitive

/** Deterministic element-id rewrite driven by the renderer reference inventory. */
internal object FlatSpecIdRewriter {
    fun rewrite(flatSpec: JsonObject, shorten: Boolean = true, reserveRoot: Boolean = false): JsonObject {
        val elements = flatSpec.get("elements")?.takeIf { it.isJsonObject }?.asJsonObject
            ?: return flatSpec.deepCopy()
        val rootId = flatSpec.get("root")?.takeIf { it.isJsonPrimitive }?.asString.orEmpty()
        if (rootId.isBlank() || !elements.has(rootId)) return flatSpec.deepCopy()

        val order = mutableListOf<String>()
        val seen = mutableSetOf<String>()
        fun walk(id: String) {
            if (!elements.has(id) || !seen.add(id)) return
            order += id
            elements.get(id)?.takeIf { it.isJsonObject }?.asJsonObject?.let { element ->
                FlatSpecReferenceSemantics.references(element).forEach { walk(it.targetId) }
            }
        }
        walk(rootId)
        elements.entrySet().forEach { walk(it.key) }

        val mapping = linkedMapOf<String, String>()
        if (shorten) {
            mapping[rootId] = "root"
            order.filter { it != rootId }.forEachIndexed { index, id -> mapping[id] = shortName(index) }
        } else {
            order.forEach { mapping[it] = it }
            if (reserveRoot && rootId != "root") {
                if (mapping.containsKey("root")) {
                    var replacementIndex = 1
                    var replacement = "root_$replacementIndex"
                    while (mapping.containsValue(replacement)) {
                        replacementIndex += 1
                        replacement = "root_$replacementIndex"
                    }
                    mapping["root"] = replacement
                }
                mapping[rootId] = "root"
            }
        }

        val rewritten = JsonObject()
        order.forEach { oldId ->
            val element = elements.get(oldId)?.takeIf { it.isJsonObject }?.asJsonObject?.deepCopy()
                ?: return@forEach
            FlatSpecReferenceSemantics.references(element).forEach { reference ->
                mapping[reference.targetId]?.let { replacement ->
                    setReference(element, reference.sourcePath, replacement)
                }
            }
            rewritten.add(mapping.getValue(oldId), element)
        }
        return JsonObject().apply {
            addProperty("root", mapping.getValue(rootId))
            add(
                "state",
                flatSpec.get("state")?.takeIf { it.isJsonObject }?.deepCopy() ?: JsonObject(),
            )
            add("elements", rewritten)
        }
    }

    private fun shortName(index: Int): String {
        var current = index
        val out = StringBuilder()
        do {
            out.insert(0, ('a'.code + (current % 26)).toChar())
            current = current / 26 - 1
        } while (current >= 0)
        return out.toString()
    }

    private fun setReference(root: JsonObject, path: String, replacement: String) {
        val tokens = pathTokens(path)
        require(tokens.isNotEmpty()) { "Renderer reference path is empty." }
        var current: JsonElement = root
        tokens.dropLast(1).forEach { token -> current = child(current, token, path) }
        when (val last = tokens.last()) {
            is PathToken.Key -> current.asJsonObject.addProperty(last.value, replacement)
            is PathToken.Index -> current.asJsonArray.set(last.value, JsonPrimitive(replacement))
        }
    }

    private fun child(current: JsonElement, token: PathToken, path: String): JsonElement = when (token) {
        is PathToken.Key -> current.takeIf { it.isJsonObject }?.asJsonObject?.get(token.value)
        is PathToken.Index -> current.takeIf { it.isJsonArray }?.asJsonArray?.get(token.value)
    } ?: error("Renderer reference path '$path' does not exist.")

    private fun pathTokens(path: String): List<PathToken> {
        val tokens = mutableListOf<PathToken>()
        val key = StringBuilder()
        var index = 0
        fun flushKey() {
            if (key.isNotEmpty()) {
                tokens += PathToken.Key(key.toString())
                key.clear()
            }
        }
        while (index < path.length) {
            when (path[index]) {
                '.' -> {
                    flushKey()
                    index += 1
                }
                '[' -> {
                    flushKey()
                    val end = path.indexOf(']', startIndex = index)
                    require(end > index + 1) { "Invalid renderer reference path '$path'." }
                    tokens += PathToken.Index(path.substring(index + 1, end).toInt())
                    index = end + 1
                }
                else -> {
                    key.append(path[index])
                    index += 1
                }
            }
        }
        flushKey()
        return tokens
    }

    private sealed interface PathToken {
        data class Key(val value: String) : PathToken
        data class Index(val value: Int) : PathToken
    }
}
