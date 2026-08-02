package com.samsung.genuicraft.pipeline

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject

/** Lossless Compact IR v2 codec. Empty/default syntax is restored on decode. */
internal object CompactIrCodec {
    const val VERSION: String = "gci2"

    fun looksLike(payload: JsonElement?): Boolean = payload?.takeIf { it.isJsonObject }
        ?.asJsonObject?.get("v")?.takeIf { it.isJsonPrimitive }?.asString == VERSION

    fun decode(payload: JsonElement): JsonObject {
        require(payload.isJsonObject) { "Compact IR payload must be an object." }
        val source = payload.asJsonObject
        require(source.get("v")?.asString == VERSION) { "Compact IR version must be '$VERSION'." }
        val root = source.string("r")?.trim().orEmpty()
        require(root.isNotEmpty()) { "Compact IR r must be a non-empty string." }
        val compactElements = source.get("e")?.takeIf { it.isJsonObject }?.asJsonObject
            ?: error("Compact IR e must be an object.")
        require(compactElements.size() > 0) { "Compact IR e must be non-empty." }
        val state = source.get("s") ?: JsonObject()
        require(state.isJsonObject) { "Compact IR s must be an object." }
        val elements = JsonObject()
        compactElements.entrySet().forEach { (id, raw) ->
            require(id.isNotBlank() && raw.isJsonObject) { "Compact IR element '$id' must be an object." }
            val item = raw.asJsonObject
            val typeToken = item.string("t")?.trim().orEmpty()
            require(typeToken.isNotEmpty()) { "Compact IR element '$id' requires t." }
            val props = item.get("p") ?: JsonObject()
            require(props.isJsonObject) { "Compact IR element '$id' p must be an object." }
            val children = item.get("c") ?: JsonArray()
            require(children.isJsonArray) { "Compact IR element '$id' c must be an array." }
            children.asJsonArray.forEach { child ->
                require(child.isJsonPrimitive && child.asJsonPrimitive.isString && child.asString.isNotBlank()) {
                    "Compact IR element '$id' c must contain non-empty strings."
                }
            }
            val element = JsonObject()
            val propsOut = props.deepCopy().asJsonObject
            when (typeToken) {
                "Row" -> {
                    element.addProperty("type", "Stack")
                    if (!propsOut.has("direction")) propsOut.addProperty("direction", "horizontal")
                }
                "Column" -> {
                    element.addProperty("type", "Stack")
                    if (!propsOut.has("direction")) propsOut.addProperty("direction", "vertical")
                }
                else -> element.addProperty("type", typeToken)
            }
            element.add("props", propsOut)
            element.add("children", children.deepCopy())
            listOf("x" to "repeat", "o" to "on", "w" to "watch").forEach { (short, full) ->
                item.get(short)?.let { value ->
                    require(value.isJsonObject) { "Compact IR element '$id' $short must be an object." }
                    element.add(full, value.deepCopy())
                }
            }
            item.get("z")?.let { element.add("visible", it.deepCopy()) }
            val unknown = item.entrySet().map { it.key }.toSet() - setOf("t", "p", "c", "x", "z", "o", "w")
            require(unknown.isEmpty()) { "Compact IR element '$id' has unknown keys ${unknown.sorted()}." }
            elements.add(id, element)
        }
        val unknownTop = source.entrySet().map { it.key }.toSet() - setOf("v", "r", "s", "e")
        require(unknownTop.isEmpty()) { "Compact IR has unknown keys ${unknownTop.sorted()}." }
        return JsonObject().apply {
            addProperty("root", root)
            add("state", state.deepCopy())
            add("elements", elements)
        }
    }

    fun encode(flatSpec: JsonObject): JsonObject {
        val source = FlatSpecIdRewriter.rewrite(flatSpec, shorten = true)
        val out = JsonObject().apply {
            addProperty("v", VERSION)
            addProperty("r", source.string("root") ?: "root")
        }
        source.get("state")?.takeIf { it.isJsonObject && it.asJsonObject.size() > 0 }
            ?.let { out.add("s", it.deepCopy()) }
        val compactElements = JsonObject()
        source.get("elements")?.takeIf { it.isJsonObject }?.asJsonObject?.entrySet()?.forEach { (id, raw) ->
            if (!raw.isJsonObject) return@forEach
            val element = raw.asJsonObject
            val props = element.get("props")?.takeIf { it.isJsonObject }?.asJsonObject?.deepCopy() ?: JsonObject()
            val rawType = element.string("type").orEmpty()
            val compactType = if (rawType == "Stack") {
                when (props.string("direction")?.lowercase()) {
                    "horizontal", "row" -> { props.remove("direction"); "Row" }
                    "vertical", "column" -> { props.remove("direction"); "Column" }
                    else -> rawType
                }
            } else rawType
            val item = JsonObject().apply { addProperty("t", compactType) }
            if (props.size() > 0) item.add("p", props)
            element.get("children")?.takeIf { it.isJsonArray && it.asJsonArray.size() > 0 }
                ?.let { item.add("c", it.deepCopy()) }
            listOf("repeat" to "x", "on" to "o", "watch" to "w").forEach { (full, short) ->
                element.get(full)?.takeIf { it.isJsonObject && it.asJsonObject.size() > 0 }
                    ?.let { item.add(short, it.deepCopy()) }
            }
            element.get("visible")?.let { item.add("z", it.deepCopy()) }
            compactElements.add(id, item)
        }
        out.add("e", compactElements)
        return out
    }

    private fun JsonObject.string(key: String): String? =
        get(key)?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
}
