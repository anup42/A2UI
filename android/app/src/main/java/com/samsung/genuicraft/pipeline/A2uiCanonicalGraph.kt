package com.samsung.genuicraft.pipeline

import com.google.gson.JsonElement
import com.google.gson.JsonObject

/** Strict validator for the graph produced by the active Express/wire codecs. */
internal object A2uiCanonicalGraph {
    data class ValidationResult(val isValid: Boolean, val error: String? = null)

    private val allowedElementFields = setOf("type", "props", "children", "repeat", "visible", "on", "watch")
    private val actionParams = mapOf(
        "openUrl" to setOf("url"),
        "setState" to setOf("statePath", "value"),
        "pushState" to setOf("statePath", "value", "clearStatePath"),
        "removeState" to setOf("statePath", "index"),
        "validateForm" to setOf("statePath", "resultStatePath"),
        "emitEvent" to setOf("name", "context", "wantResponse", "responsePath"),
    )
    private val requiredActionParams = mapOf(
        "openUrl" to setOf("url"),
        "setState" to setOf("statePath", "value"),
        "pushState" to setOf("statePath", "value"),
        "removeState" to setOf("statePath", "index"),
        "validateForm" to emptySet(),
        "emitEvent" to setOf("name"),
    )
    private val repeatFields = setOf("statePath", "key", "template", "itemTemplate", "child")

    fun validate(graph: JsonObject, requireReservedRoot: Boolean = true): ValidationResult {
        val root = graph.get("root")?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
            ?: return invalid("Canonical graph requires a string root.")
        if (requireReservedRoot && root != "root") return invalid("Canonical graph root must be the reserved 'root' id.")
        if (graph.entrySet().any { it.key !in setOf("root", "state", "elements") }) {
            return invalid("Canonical graph contains unsupported top-level fields.")
        }
        if (graph.get("state")?.isJsonObject != true) return invalid("Canonical graph state must be an object.")
        val elements = graph.get("elements")?.takeIf { it.isJsonObject }?.asJsonObject
            ?: return invalid("Canonical graph elements must be an object.")
        if (elements.size() == 0 || !elements.has(root)) return invalid("Canonical graph root element is missing.")
        val ids = elements.entrySet().map { it.key }.toSet()

        elements.entrySet().forEach { (id, raw) ->
            if (!raw.isJsonObject) return invalid("Element '$id' must be an object.")
            val element = raw.asJsonObject
            if (element.entrySet().any { it.key !in allowedElementFields }) {
                return invalid("Element '$id' contains unsupported fields.")
            }
            val type = element.get("type")?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
                ?: return invalid("Element '$id' has no canonical type.")
            if (type !in GenUiA2uiCatalog.positional.keys) return invalid("Element '$id' has unsupported type '$type'.")
            val props = element.get("props")?.takeIf { it.isJsonObject }?.asJsonObject
                ?: return invalid("Element '$id' props must be an object.")
            props.entrySet().forEach { (key, _) ->
                if (!GenUiA2uiCatalog.isAllowedProperty(type, key)) {
                    return invalid("Element '$id' has unsupported property '$key'.")
                }
            }
            if (type == "List" && props.has("items")) {
                val items = props.get("items")
                if (!items.isJsonArray) return invalid("Element '$id' List.items must be an array; use children for components.")
                val fields = setOf("text", "label", "title", "name", "content", "value", "description", "url", "href", "link", "source")
                items.asJsonArray.forEach { item ->
                    val text = item.isJsonPrimitive && item.asJsonPrimitive.isString
                    val data = item.isJsonObject && item.asJsonObject.size() > 0 && item.asJsonObject.entrySet().all {
                        it.key in fields && it.value.isJsonPrimitive && it.value.asJsonPrimitive.isString
                    }
                    if (!text && !data) return invalid("Element '$id' List.items contains unsupported data; use children for component calls.")
                }
            }
            if (type == "Stack") {
                props.get("direction")?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString?.let {
                    if (it !in setOf("vertical", "horizontal")) return invalid("Element '$id' has invalid Stack.direction.")
                }
                props.get("gap")?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString?.let {
                    if (it !in setOf("none", "sm", "md", "lg", "xl")) return invalid("Element '$id' has invalid Stack.gap.")
                }
            }
            val children = element.get("children")?.takeIf { it.isJsonArray }?.asJsonArray
                ?: return invalid("Element '$id' children must be an array.")
            children.forEach { child ->
                if (!child.isJsonPrimitive || !child.asJsonPrimitive.isString || child.asString.isBlank()) {
                    return invalid("Element '$id' contains an invalid child reference.")
                }
            }
            element.get("repeat")?.let { repeat ->
                if (!repeat.isJsonObject || repeat.asJsonObject.get("statePath")?.asString?.isBlank() != false) {
                    return invalid("Element '$id' repeat.statePath is required.")
                }
                val unknown = repeat.asJsonObject.keySet() - repeatFields
                if (unknown.isNotEmpty()) return invalid("Element '$id' repeat contains unsupported fields: ${unknown.sorted()}.")
            }
            validateActions(element.get("on"), "Element '$id'.on")?.let { return invalid(it) }
            val watch = element.get("watch")
            if (watch != null) {
                if (!watch.isJsonObject) return invalid("Element '$id'.watch must be an object.")
                watch.asJsonObject.entrySet().forEach { (path, action) ->
                    if (!path.startsWith("/") || path.length == 1) return invalid("Element '$id'.watch key '$path' is not a JSON pointer.")
                    validateAction(action, "Element '$id'.watch.$path")?.let { return invalid(it) }
                }
            }
            RendererReferenceSemantics.references(element).forEach { reference ->
                if (reference.targetId !in ids) return invalid("Element '$id' references missing '${reference.targetId}'.")
            }
        }

        val visiting = linkedSetOf<String>()
        val visited = linkedSetOf<String>()
        fun visit(id: String): String? {
            if (id in visiting) return id
            if (id in visited) return null
            visiting += id
            val element = elements.get(id)?.takeIf { it.isJsonObject }?.asJsonObject
            element?.let { RendererReferenceSemantics.references(it).forEach { ref -> visit(ref.targetId)?.let { return it } } }
            visiting -= id
            visited += id
            return null
        }
        visit(root)?.let { return invalid("Canonical graph contains reachable reference cycle at '$it'.") }
        return ValidationResult(true)
    }

    private fun validateActions(value: JsonElement?, context: String): String? {
        if (value == null) return null
        if (!value.isJsonObject) return "$context must be an object."
        value.asJsonObject.entrySet().forEach { (name, action) ->
            validateAction(action, "$context.$name")?.let { return it }
        }
        return null
    }

    private fun validateAction(value: JsonElement, context: String): String? {
        if (value.isJsonArray) {
            if (value.asJsonArray.size() == 0) return "$context action list must not be empty."
            value.asJsonArray.forEach { validateAction(it, context)?.let { return it } }
            return null
        }
        if (!value.isJsonObject) return "$context must be an action object."
        val action = value.asJsonObject.get("action")?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
            ?: return "$context has no action name."
        val allowed = actionParams[action] ?: return "$context uses unknown action '$action'."
        val params = value.asJsonObject.get("params")?.takeIf { it.isJsonObject }?.asJsonObject
            ?: return "$context.params must be an object."
        params.entrySet().forEach { (key, _) -> if (key !in allowed) return "$context.params has unsupported '$key'." }
        val missing = requiredActionParams[action].orEmpty() - params.keySet()
        if (missing.isNotEmpty()) return "$context.params is missing required ${missing.sorted()}."
        return null
    }

    private fun invalid(message: String) = ValidationResult(false, message)
}
