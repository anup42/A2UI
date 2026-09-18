package com.samsung.genuicraft.sdk.internal.pipeline

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonPrimitive

/** Internal compiler for the pinned upstream A2UI v0.9 wire envelopes. */
internal object A2uiWireCodec {
    private const val VERSION = "v0.9"

    fun looksLike(payload: JsonElement?): Boolean {
        if (payload == null || payload.isJsonNull) return false
        if (payload.isJsonObject) return isMessage(payload.asJsonObject)
        return payload.takeIf { it.isJsonArray }?.asJsonArray?.let { messages ->
            messages.size() > 0 && messages.all { it.isJsonObject && isMessage(it.asJsonObject) }
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
        var deleted = false

        messages.forEach { message ->
            require(message.string("version") == VERSION) { "A2UI wire version must be $VERSION." }
            when {
                message.has("createSurface") -> {
                    val create = message.objectValue("createSurface")
                    surfaceId = validateSurface(create, surfaceId)
                    require(create.string("catalogId") == GenUiA2uiCatalog.CATALOG_ID) {
                        "A2UI createSurface.catalogId must be ${GenUiA2uiCatalog.CATALOG_ID}."
                    }
                    require(!create.has("rootId") && !create.has("components") && !create.has("dataModel")) {
                        "A2UI v0.9 createSurface may not contain rootId, components, or dataModel."
                    }
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
                    setDataPath(
                        state,
                        update.string("path"),
                        if (update.has("value")) update.get("value").deepCopy() else null,
                    )
                }
                message.has("deleteSurface") -> {
                    val delete = message.objectValue("deleteSurface")
                    surfaceId = validateSurface(delete, surfaceId)
                    elements.entrySet().map { it.key }.forEach(elements::remove)
                    state = JsonObject()
                    deleted = true
                }
                else -> error("Unsupported A2UI v0.9 wire message.")
            }
        }
        require(!deleted) { "A2UI surface was deleted before conversion." }
        require(elements.size() > 0) { "A2UI wire payload did not produce any components." }
        require(elements.has("root")) { "A2UI payload must contain a component with id 'root'." }
        return JsonObject().apply {
            addProperty("root", "root")
            add("state", state)
            add("elements", elements)
        }
    }

    fun encode(canonicalGraph: JsonObject): JsonArray {
        val source = CanonicalGraphIdRewriter.rewrite(canonicalGraph, shorten = true)
        val components = JsonArray()
        source.get("elements")?.takeIf { it.isJsonObject }?.asJsonObject?.entrySet()?.forEach { (id, raw) ->
            if (!raw.isJsonObject) return@forEach
            val element = raw.asJsonObject
            val component = JsonObject().apply {
                addProperty("id", id)
                addProperty("component", element.string("type") ?: "Stack")
            }
            element.get("props")?.takeIf { it.isJsonObject }?.asJsonObject?.entrySet()?.forEach { (key, value) ->
                component.add(key, lowerBindings(value.deepCopy()))
            }
            val children = element.get("children")?.takeIf { it.isJsonArray }?.asJsonArray ?: JsonArray()
            val repeat = element.getAsJsonObjectOrNull("repeat")
            val childList = lowerChildList(children, repeat)
            if (childList.isJsonArray && childList.asJsonArray.size() == 0) {
                // Omit an empty optional ChildList.
            } else {
                component.add("children", childList)
            }
            listOf("visible", "on", "watch").forEach { key ->
                element.get(key)?.takeUnless {
                    it.isJsonNull || (it.isJsonArray && it.asJsonArray.size() == 0) ||
                        (it.isJsonObject && it.asJsonObject.size() == 0)
                }?.let {
                    component.add(key, lowerBindings(it.deepCopy(), actionMap = key == "on" || key == "watch"))
                }
            }
            components.add(component)
        }
        require((source.string("root") ?: "root") == "root") {
            "Compiled A2UI root must be deterministically rewritten to id 'root'."
        }
        return JsonArray().apply {
            add(JsonObject().apply {
                addProperty("version", VERSION)
                add("createSurface", JsonObject().apply {
                    addProperty("surfaceId", "default_surface")
                    addProperty("catalogId", GenUiA2uiCatalog.CATALOG_ID)
                    addProperty("sendDataModel", false)
                })
            })
            add(JsonObject().apply {
                addProperty("version", VERSION)
                add("updateComponents", JsonObject().apply {
                    addProperty("surfaceId", "default_surface")
                    add("components", components)
                })
            })
            source.get("state")?.takeIf { it.isJsonObject && it.asJsonObject.size() > 0 }?.let { state ->
                add(JsonObject().apply {
                    addProperty("version", VERSION)
                    add("updateDataModel", JsonObject().apply {
                        addProperty("surfaceId", "default_surface")
                        addProperty("path", "/")
                        add("value", lowerBindings(state.deepCopy()))
                    })
                })
            }
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
            val (children, repeat) = decodeChildList(component.get("children"), id)
            val metadata = setOf("id", "component", "children", "visible", "on", "watch")
            val props = JsonObject()
            component.entrySet().filter { it.key !in metadata }.forEach { (key, value) ->
                require(GenUiA2uiCatalog.isAllowedProperty(type, key)) {
                    "A2UI component '$id' has unsupported property '$key'."
                }
                props.add(key, raiseBindings(value.deepCopy()))
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
                add("children", children)
                repeat?.let { add("repeat", it) }
            }
            listOf("on", "watch").forEach { key ->
                component.get(key)?.let {
                    require(it.isJsonObject) { "A2UI component '$id' $key must be an object." }
                    element.add(key, raiseBindings(it.deepCopy(), actionMap = true))
                }
            }
            component.get("visible")?.let { element.add("visible", raiseBindings(it.deepCopy())) }
            require(!elements.has(id)) { "Duplicate A2UI component id across updates: $id" }
            elements.add(id, element)
        }
    }

    private fun decodeChildList(value: JsonElement?, elementId: String): Pair<JsonArray, JsonObject?> {
        if (value == null || value.isJsonNull) return JsonArray() to null
        if (value.isJsonArray) {
            value.asJsonArray.forEach { child ->
                require(child.isJsonPrimitive && child.asJsonPrimitive.isString && child.asString.isNotBlank()) {
                    "A2UI component '$elementId' children must be non-empty strings."
                }
            }
            return value.asJsonArray.deepCopy() to null
        }
        require(value.isJsonObject) { "A2UI component '$elementId' children must be a ChildList." }
        val dynamic = value.asJsonObject
        require(dynamic.keySet().all { it in setOf("componentId", "path", "key") }) {
            "A2UI component '$elementId' dynamic children must contain componentId and path."
        }
        val template = dynamic.string("componentId")?.trim().orEmpty()
        val path = dynamic.string("path")?.trim().orEmpty()
        require(template.isNotEmpty() && path.isNotEmpty()) {
            "A2UI component '$elementId' dynamic children require componentId and path."
        }
        return JsonArray().apply { add(template) } to JsonObject().apply {
            addProperty("statePath", path)
            addProperty("template", template)
            dynamic.string("key")?.takeIf { it.isNotBlank() }?.let { addProperty("key", it) }
        }
    }

    private fun lowerChildList(children: JsonArray, repeat: JsonObject?): JsonElement {
        if (repeat != null && repeat.size() > 0) {
            val template = repeat.string("template")
                ?: repeat.string("itemTemplate")
                ?: repeat.string("child")
                ?: children.takeIf { it.size() > 0 }?.get(0)
                    ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                    ?.asString
            val path = repeat.string("statePath") ?: repeat.string("path")
            require(!template.isNullOrBlank() && !path.isNullOrBlank()) {
                "A2UI repeat requires template/itemTemplate/child and statePath."
            }
            return JsonObject().apply {
                addProperty("componentId", template)
                addProperty("path", jsonPointerPath(path))
                repeat.string("key")?.takeIf { it.isNotBlank() }?.let { addProperty("key", it) }
            }
        }
        return children.deepCopy()
    }

    private fun validateSurface(message: JsonObject, current: String?): String {
        val value = message.string("surfaceId")?.trim().orEmpty()
        require(value.isNotEmpty()) { "A2UI message requires a non-empty surfaceId." }
        require(current == null || current == value) { "A2UI conversion supports one surface per payload." }
        return value
    }

    private fun setDataPath(state: JsonObject, path: String?, value: JsonElement?) {
        if (path.isNullOrEmpty() || path == "/") {
            state.entrySet().map { it.key }.forEach(state::remove)
            if (value != null) {
                require(value.isJsonObject) { "Root A2UI data-model update must be an object." }
                value.asJsonObject.entrySet().forEach { (key, item) -> state.add(key, raiseBindings(item.deepCopy())) }
            }
            return
        }
        require(path.startsWith('/')) { "A2UI updateDataModel.path must be a JSON pointer." }
        val parts = path.split('/').drop(1).map { it.replace("~1", "/").replace("~0", "~") }
        require(parts.isNotEmpty() && parts.last().isNotEmpty()) { "A2UI updateDataModel.path must name a value." }
        var current = state
        parts.dropLast(1).forEach { part ->
            val next = current.get(part)?.takeIf { it.isJsonObject }?.asJsonObject
                ?: JsonObject().also { current.add(part, it) }
            current = next
        }
        if (value == null) current.remove(parts.last())
        else current.add(parts.last(), raiseBindings(value.deepCopy()))
    }

    private fun jsonPointerPath(value: String): String {
        val path = value.trim()
        require(path.isNotEmpty()) { "A2UI dynamic path must be non-empty." }
        if (path == "$") return "/"
        if (path.startsWith("$/")) return "/${path.substring(2)}"
        if (path.startsWith("\$state.")) return "/${path.substring(7).replace('.', '/')}"
        if (path.startsWith('/')) return path
        if (path.startsWith('$')) return "/${path.substring(1).trimStart('/').replace('.', '/')}"
        return path
    }

    private fun lowerBindings(value: JsonElement, actionMap: Boolean = false): JsonElement {
        if (value.isJsonPrimitive && value.asJsonPrimitive.isString) {
            val text = value.asString
            if (text == "$" || text.startsWith("$/") || text.startsWith("\$state.")) {
                return JsonObject().apply { addProperty("path", jsonPointerPath(text)) }
            }
            return value
        }
        if (value.isJsonArray) return JsonArray().also { out -> value.asJsonArray.forEach { out.add(lowerBindings(it.deepCopy(), actionMap)) } }
        if (!value.isJsonObject) return value
        if (actionMap && value.asJsonObject.string("action") != null) return lowerAction(value.asJsonObject)
        return JsonObject().also { out -> value.asJsonObject.entrySet().forEach { (key, item) -> out.add(key, lowerBindings(item.deepCopy(), actionMap)) } }
    }

    private fun lowerAction(value: JsonObject): JsonObject {
        val action = value.string("action") ?: error("A2UI action name is required.")
        val params = value.getAsJsonObjectOrNull("params") ?: JsonObject()
        if (action == "emitEvent") {
            val name = params.string("name")?.takeIf { it.isNotBlank() } ?: error("emitEvent requires name.")
            return JsonObject().apply {
                add("event", JsonObject().apply {
                    addProperty("name", name)
                    val context = JsonObject()
                    params.getAsJsonObjectOrNull("context")?.entrySet()?.forEach { (key, item) -> context.add(key, lowerBindings(item.deepCopy())) }
                    if (context.size() > 0) add("context", context)
                    listOf("wantResponse", "responsePath").forEach { key ->
                        params.get(key)?.let { add(key, lowerBindings(it.deepCopy())) }
                    }
                })
            }
        }
        return JsonObject().apply {
            add("functionCall", JsonObject().apply {
                addProperty("call", action)
                val args = JsonObject()
                params.entrySet().forEach { (key, item) -> args.add(key, lowerBindings(item.deepCopy())) }
                if (args.size() > 0) add("args", args)
            })
        }
    }

    private fun raiseBindings(value: JsonElement, actionMap: Boolean = false): JsonElement {
        if (value.isJsonObject) {
            val obj = value.asJsonObject
            if (obj.keySet() == setOf("path") && obj.string("path") != null) return obj.deepCopy()
            if (actionMap && (obj.has("event") || obj.has("functionCall"))) return raiseAction(obj)
            return JsonObject().also { out -> obj.entrySet().forEach { (key, item) -> out.add(key, raiseBindings(item.deepCopy(), actionMap)) } }
        }
        if (value.isJsonArray) return JsonArray().also { out -> value.asJsonArray.forEach { out.add(raiseBindings(it.deepCopy(), actionMap)) } }
        return value
    }

    private fun raiseAction(value: JsonObject): JsonObject {
        value.getAsJsonObjectOrNull("event")?.let { event ->
            return JsonObject().apply {
                addProperty("action", "emitEvent")
                add("params", JsonObject().also { params ->
                    params.add("name", event.get("name")?.deepCopy() ?: JsonPrimitive(""))
                    event.getAsJsonObjectOrNull("context")?.let { context ->
                        params.add("context", JsonObject().also { out ->
                            context.entrySet().forEach { (key, item) -> out.add(key, raiseBindings(item.deepCopy())) }
                        })
                    }
                    listOf("wantResponse", "responsePath").forEach { key ->
                        event.get(key)?.let { params.add(key, raiseBindings(it.deepCopy())) }
                    }
                })
            }
        }
        val call = value.getAsJsonObjectOrNull("functionCall") ?: error("A2UI action must be an event or functionCall.")
        val name = call.string("call")?.takeIf { it.isNotBlank() } ?: error("A2UI functionCall.call is required.")
        return JsonObject().apply {
            addProperty("action", name)
            add("params", JsonObject().also { params -> call.getAsJsonObjectOrNull("args")?.entrySet()?.forEach { (key, item) -> params.add(key, raiseBindings(item.deepCopy())) } })
        }
    }

    private fun isMessage(message: JsonObject): Boolean =
        message.string("version") == VERSION &&
            listOf("createSurface", "updateComponents", "updateDataModel", "deleteSurface").count(message::has) == 1

    private fun JsonObject.objectValue(key: String): JsonObject =
        get(key)?.takeIf { it.isJsonObject }?.asJsonObject ?: error("A2UI $key must be an object.")

    private fun JsonObject.getAsJsonObjectOrNull(key: String): JsonObject? =
        get(key)?.takeIf { it.isJsonObject }?.asJsonObject

    private fun JsonObject.string(key: String): String? =
        get(key)?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
}
