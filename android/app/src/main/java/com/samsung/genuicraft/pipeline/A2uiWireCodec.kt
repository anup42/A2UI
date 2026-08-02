package com.samsung.genuicraft.pipeline

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject

/** Custom-catalog A2UI v1 wire conversion for the native renderer graph. */
internal object A2uiWireCodec {
    private const val VERSION = "v1.0"

    fun looksLike(payload: JsonElement?): Boolean {
        if (payload == null || payload.isJsonNull) return false
        if (payload.isJsonObject) return isV1Message(payload.asJsonObject)
        return payload.takeIf { it.isJsonArray }?.asJsonArray?.let { messages ->
            messages.size() > 0 && messages.all { it.isJsonObject && isV1Message(it.asJsonObject) }
        } == true
    }

    fun decode(payload: JsonElement): JsonObject {
        val messages = when {
            payload.isJsonObject -> listOf(payload.asJsonObject)
            payload.isJsonArray && payload.asJsonArray.size() > 0 -> payload.asJsonArray.map {
                require(it.isJsonObject) { "A2UI wire messages must be objects." }
                it.asJsonObject
            }
            else -> error("A2UI wire payload must be an object or non-empty message array.")
        }
        val elements = JsonObject()
        var state = JsonObject()
        var surfaceId: String? = null
        var rootId: String? = null
        var deleted = false

        messages.forEach { message ->
            require(message.string("version") == VERSION) { "A2UI wire version must be $VERSION." }
            when {
                message.has("createSurface") -> {
                    val create = message.objectValue("createSurface")
                    surfaceId = validateSurface(create, surfaceId)
                    create.string("catalogId")?.let {
                        require(it == GenUiA2uiCatalog.CATALOG_ID) {
                            "A2UI createSurface.catalogId must be ${GenUiA2uiCatalog.CATALOG_ID}."
                        }
                    }
                    create.string("rootId")?.let {
                        require(it.isNotBlank()) { "A2UI createSurface.rootId must be non-empty." }
                        rootId = it
                    }
                    create.get("dataModel")?.let {
                        require(it.isJsonObject) { "A2UI createSurface.dataModel must be an object." }
                        state = it.asJsonObject.deepCopy()
                    }
                    create.get("components")?.let { decodeComponents(it, elements) }
                    deleted = false
                }
                message.has("updateComponents") -> {
                    val update = message.objectValue("updateComponents")
                    surfaceId = validateSurface(update, surfaceId)
                    decodeComponents(update.get("components"), elements)
                }
                message.has("updateDataModel") -> {
                    val update = message.objectValue("updateDataModel")
                    surfaceId = validateSurface(update, surfaceId)
                    require(update.has("value")) { "A2UI updateDataModel requires value." }
                    setDataPath(state, update.string("path"), update.get("value").deepCopy())
                }
                message.has("deleteSurface") -> {
                    val delete = message.objectValue("deleteSurface")
                    surfaceId = validateSurface(delete, surfaceId)
                    elements.entrySet().map { it.key }.forEach(elements::remove)
                    state = JsonObject()
                    deleted = true
                }
                else -> error("Unsupported A2UI v1 wire message.")
            }
        }
        require(!deleted) { "A2UI surface was deleted before conversion." }
        require(elements.size() > 0) { "A2UI wire payload did not produce any components." }
        val resolvedRoot = rootId ?: if (elements.has("root")) "root" else elements.entrySet().first().key
        require(elements.has(resolvedRoot)) { "A2UI rootId '$resolvedRoot' does not exist in components." }
        return JsonObject().apply {
            addProperty("root", resolvedRoot)
            add("state", state)
            add("elements", elements)
        }
    }

    fun encode(flatSpec: JsonObject): JsonObject {
        val source = FlatSpecIdRewriter.rewrite(flatSpec, shorten = true)
        val components = JsonArray()
        source.get("elements")?.takeIf { it.isJsonObject }?.asJsonObject?.entrySet()?.forEach { (id, raw) ->
            if (!raw.isJsonObject) return@forEach
            val element = raw.asJsonObject
            val component = JsonObject().apply {
                addProperty("id", id)
                addProperty("component", element.string("type") ?: "Stack")
            }
            element.get("props")?.takeIf { it.isJsonObject }?.asJsonObject?.entrySet()?.forEach { (key, value) ->
                component.add(key, value.deepCopy())
            }
            listOf("children", "repeat", "visible", "on", "watch").forEach { key ->
                element.get(key)?.takeUnless {
                    it.isJsonNull || (it.isJsonArray && it.asJsonArray.size() == 0) ||
                        (it.isJsonObject && it.asJsonObject.size() == 0)
                }?.let { component.add(key, it.deepCopy()) }
            }
            components.add(component)
        }
        val create = JsonObject().apply {
            addProperty("surfaceId", "default_surface")
            addProperty("catalogId", GenUiA2uiCatalog.CATALOG_ID)
            addProperty("rootId", source.string("root") ?: "root")
            add("components", components)
            source.get("state")?.takeIf { it.isJsonObject && it.asJsonObject.size() > 0 }
                ?.let { add("dataModel", it.deepCopy()) }
        }
        return JsonObject().apply {
            addProperty("version", VERSION)
            add("createSurface", create)
        }
    }

    private fun decodeComponents(raw: JsonElement?, elements: JsonObject) {
        val components = raw?.takeIf { it.isJsonArray }?.asJsonArray
            ?: error("A2UI components must be an array.")
        require(components.size() > 0) { "A2UI components must be non-empty." }
        val seen = mutableSetOf<String>()
        components.forEach { item ->
            require(item.isJsonObject) { "A2UI component must be an object." }
            val component = item.asJsonObject
            val id = component.string("id")?.trim().orEmpty()
            val type = component.string("component")?.trim().orEmpty()
            require(id.isNotEmpty() && seen.add(id)) { "A2UI component id must be non-empty and unique per update." }
            require(GenUiA2uiCatalog.positional.containsKey(type)) { "Unsupported A2UI component '$type'." }
            val children = component.get("children") ?: JsonArray()
            require(children.isJsonArray) { "A2UI component '$id' children must be an array." }
            children.asJsonArray.forEach { child ->
                require(child.isJsonPrimitive && child.asJsonPrimitive.isString && child.asString.isNotBlank()) {
                    "A2UI component '$id' children must be non-empty strings."
                }
            }
            val metadata = setOf("id", "component", "children", "repeat", "visible", "on", "watch")
            val props = JsonObject()
            component.entrySet().filter { it.key !in metadata }.forEach { (key, value) ->
                props.add(key, value.deepCopy())
            }
            val canonicalType = when (type) {
                "Row", "Column" -> "Stack"
                else -> type
            }
            if (type == "Row" && !props.has("direction")) props.addProperty("direction", "horizontal")
            if (type == "Column" && !props.has("direction")) props.addProperty("direction", "vertical")
            val element = JsonObject().apply {
                addProperty("type", canonicalType)
                add("props", props)
                add("children", children.deepCopy())
            }
            listOf("repeat", "on", "watch").forEach { key ->
                component.get(key)?.let {
                    require(it.isJsonObject) { "A2UI component '$id' $key must be an object." }
                    element.add(key, it.deepCopy())
                }
            }
            component.get("visible")?.let { element.add("visible", it.deepCopy()) }
            elements.add(id, element)
        }
    }

    private fun validateSurface(message: JsonObject, current: String?): String {
        val value = message.string("surfaceId")?.trim().orEmpty()
        require(value.isNotEmpty()) { "A2UI message requires a non-empty surfaceId." }
        require(current == null || current == value) { "A2UI conversion supports one surface per payload." }
        return value
    }

    private fun setDataPath(state: JsonObject, path: String?, value: JsonElement) {
        if (path.isNullOrEmpty() || path == "/") {
            require(value.isJsonObject) { "Root A2UI data-model update must be an object." }
            state.entrySet().map { it.key }.forEach(state::remove)
            value.asJsonObject.entrySet().forEach { (key, item) -> state.add(key, item.deepCopy()) }
            return
        }
        require(path.startsWith('/')) { "A2UI updateDataModel.path must be a JSON pointer." }
        val parts = path.split('/').drop(1).map { it.replace("~1", "/").replace("~0", "~") }
        var current = state
        parts.dropLast(1).forEach { part ->
            val next = current.get(part)?.takeIf { it.isJsonObject }?.asJsonObject
                ?: JsonObject().also { current.add(part, it) }
            current = next
        }
        current.add(parts.last(), value)
    }

    private fun isV1Message(message: JsonObject): Boolean =
        message.string("version") == VERSION &&
            listOf("createSurface", "updateComponents", "updateDataModel", "deleteSurface").count(message::has) == 1

    private fun JsonObject.objectValue(key: String): JsonObject =
        get(key)?.takeIf { it.isJsonObject }?.asJsonObject ?: error("A2UI $key must be an object.")

    private fun JsonObject.string(key: String): String? =
        get(key)?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
}
