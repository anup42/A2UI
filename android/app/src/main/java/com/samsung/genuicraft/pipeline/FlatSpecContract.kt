package com.samsung.genuicraft.pipeline

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject

internal object FlatSpecContract {

    data class NormalizeResult(
        val spec: JsonObject?,
        val convertedFromLegacy: Boolean,
        val error: String? = null
    )

    data class ValidationResult(
        val isValid: Boolean,
        val error: String? = null
    )

    data class CoerceResult(
        val spec: JsonObject?,
        val convertedFromLegacy: Boolean,
        val error: String?
    ) {
        val isValid: Boolean get() = spec != null && error == null
    }

    private val allowedTypes = setOf(
        "stack",
        "list",
        "card",
        "text",
        "image",
        "icon",
        "video",
        "audioplayer",
        "divider",
        "button",
        "tabs",
        "modal",
        "textfield",
        "checkbox",
        "choicepicker",
        "slider",
        "datetimeinput"
    )

    fun looksLikeFlatSpec(json: JsonElement?): Boolean {
        if (json == null || !json.isJsonObject) return false
        val obj = json.asJsonObject
        return obj.has("root") && obj.has("elements")
    }

    fun coerceAndValidate(json: JsonElement?): CoerceResult {
        if (json == null) {
            return CoerceResult(spec = null, convertedFromLegacy = false, error = "Stage 3 output is null.")
        }
        val normalized = normalizeToFlatSpec(json)
        val spec = normalized.spec ?: return CoerceResult(
            spec = null,
            convertedFromLegacy = normalized.convertedFromLegacy,
            error = normalized.error ?: "Could not normalize Stage 3 output to flat spec."
        )
        val validation = validateFlatSpec(spec)
        if (!validation.isValid) {
            return CoerceResult(
                spec = null,
                convertedFromLegacy = normalized.convertedFromLegacy,
                error = validation.error ?: "Flat spec validation failed."
            )
        }
        return CoerceResult(
            spec = spec,
            convertedFromLegacy = normalized.convertedFromLegacy,
            error = null
        )
    }

    fun normalizeToFlatSpec(json: JsonElement): NormalizeResult {
        if (looksLikeFlatSpec(json)) {
            return NormalizeResult(spec = canonicalizeFlatSpec(json.asJsonObject), convertedFromLegacy = false)
        }

        val unwrapped = unwrapKnownContainers(json)
        if (unwrapped != null) {
            if (looksLikeFlatSpec(unwrapped)) {
                return NormalizeResult(spec = canonicalizeFlatSpec(unwrapped.asJsonObject), convertedFromLegacy = true)
            }
            if (unwrapped.isJsonArray) {
                val converted = convertLegacyMessages(unwrapped.asJsonArray)
                return if (converted != null) {
                    NormalizeResult(spec = converted, convertedFromLegacy = true)
                } else {
                    NormalizeResult(
                        spec = null,
                        convertedFromLegacy = true,
                        error = "Wrapped legacy payload could not be converted."
                    )
                }
            }
            if (unwrapped.isJsonObject) {
                val converted = convertLegacyMessages(JsonArray().apply { add(unwrapped) })
                if (converted != null) {
                    return NormalizeResult(spec = converted, convertedFromLegacy = true)
                }
            }
        }

        if (json.isJsonArray) {
            val converted = convertLegacyMessages(json.asJsonArray)
            return if (converted != null) {
                NormalizeResult(spec = converted, convertedFromLegacy = true)
            } else {
                NormalizeResult(
                    spec = null,
                    convertedFromLegacy = true,
                    error = "Legacy message array could not be converted."
                )
            }
        }

        if (json.isJsonObject) {
            val converted = convertLegacyMessages(JsonArray().apply { add(json) })
            if (converted != null) {
                return NormalizeResult(spec = converted, convertedFromLegacy = true)
            }
        }

        return NormalizeResult(
            spec = null,
            convertedFromLegacy = false,
            error = "Unsupported JSON shape for flat spec."
        )
    }

    fun validateFlatSpec(spec: JsonObject): ValidationResult {
        val root = spec.get("root")
        if (root == null || !root.isJsonPrimitive || !root.asJsonPrimitive.isString || root.asString.isBlank()) {
            return ValidationResult(false, "Missing or invalid root id.")
        }

        val elementsNode = spec.get("elements")
        if (elementsNode == null || !elementsNode.isJsonObject) {
            return ValidationResult(false, "Missing or invalid elements object.")
        }

        val elements = elementsNode.asJsonObject
        if (elements.size() == 0) {
            return ValidationResult(false, "Elements map is empty.")
        }
        if (!elements.has(root.asString)) {
            return ValidationResult(false, "Root id '${root.asString}' does not exist in elements.")
        }

        val ids = elements.entrySet().map { it.key }.toSet()
        elements.entrySet().forEach { (id, rawElement) ->
            if (!rawElement.isJsonObject) {
                return ValidationResult(false, "Element '$id' must be an object.")
            }
            val element = rawElement.asJsonObject
            val type = element.get("type")
            if (type == null || !type.isJsonPrimitive || !type.asJsonPrimitive.isString) {
                return ValidationResult(false, "Element '$id' is missing a valid type.")
            }
            if (!allowedTypes.contains(type.asString.lowercase())) {
                return ValidationResult(false, "Element '$id' has unsupported type '${type.asString}'.")
            }

            val props = element.get("props")
            if (props == null || !props.isJsonObject) {
                return ValidationResult(false, "Element '$id' must define props as an object.")
            }
            val propsObject = props.asJsonObject
            if (propsObject.has("action")) {
                return ValidationResult(
                    false,
                    "Element '$id' uses legacy props.action. Use on.<event> action bindings instead."
                )
            }
            if (propsObject.has("className")) {
                return ValidationResult(
                    false,
                    "Element '$id' uses unsupported props.className."
                )
            }
            if (type.asString.equals("stack", ignoreCase = true)) {
                val stackError = validateStackProps(propsObject, id)
                if (stackError != null) {
                    return ValidationResult(false, stackError)
                }
            }

            val children = element.get("children")
            if (children == null || !children.isJsonArray) {
                return ValidationResult(false, "Element '$id' must define children as an array.")
            }
            children.asJsonArray.forEach { child ->
                if (!child.isJsonPrimitive || !child.asJsonPrimitive.isString) {
                    return ValidationResult(false, "Element '$id' contains a non-string child reference.")
                }
                val childId = child.asString
                if (!ids.contains(childId)) {
                    return ValidationResult(false, "Element '$id' references missing child '$childId'.")
                }
            }

            val repeat = element.get("repeat")
            if (repeat != null && !repeat.isJsonNull) {
                if (!repeat.isJsonObject) {
                    return ValidationResult(false, "Element '$id' repeat must be an object.")
                }
                val statePath = repeat.asJsonObject.get("statePath")
                if (statePath == null || !statePath.isJsonPrimitive || !statePath.asJsonPrimitive.isString || statePath.asString.isBlank()) {
                    return ValidationResult(false, "Element '$id' repeat.statePath is required.")
                }
            }

            val on = element.get("on")
            if (on != null && !on.isJsonNull) {
                if (!on.isJsonObject) {
                    return ValidationResult(false, "Element '$id' on must be an object.")
                }
                on.asJsonObject.entrySet().forEach { (eventName, actionValue) ->
                    val actionError = validateActionCandidate(
                        actionValue,
                        "Element '$id' on.$eventName"
                    )
                    if (actionError != null) {
                        return ValidationResult(false, actionError)
                    }
                }
            }

            val watch = element.get("watch")
            if (watch != null && !watch.isJsonNull) {
                if (!watch.isJsonObject) {
                    return ValidationResult(false, "Element '$id' watch must be an object.")
                }
                watch.asJsonObject.entrySet().forEach { (statePath, actionValue) ->
                    if (statePath.isBlank() || !statePath.startsWith("/")) {
                        return ValidationResult(
                            false,
                            "Element '$id' watch key '$statePath' must be a non-empty JSON pointer path."
                        )
                    }
                    val actionError = validateActionCandidate(
                        actionValue,
                        "Element '$id' watch.$statePath"
                    )
                    if (actionError != null) {
                        return ValidationResult(false, actionError)
                    }
                }
            }
        }
        return ValidationResult(true)
    }

    private fun validateActionCandidate(candidate: JsonElement, context: String): String? {
        if (candidate.isJsonNull) {
            return "$context cannot be null."
        }
        if (candidate.isJsonArray) {
            val array = candidate.asJsonArray
            if (array.size() == 0) {
                return "$context action array cannot be empty."
            }
            array.forEachIndexed { index, action ->
                val actionError = validateSingleActionBinding(
                    action,
                    "$context[$index]"
                )
                if (actionError != null) {
                    return actionError
                }
            }
            return null
        }
        return validateSingleActionBinding(candidate, context)
    }

    private fun validateSingleActionBinding(binding: JsonElement, context: String): String? {
        if (!binding.isJsonObject) {
            return "$context must be an object action binding."
        }
        val obj = binding.asJsonObject
        if (obj.has("functionCall")) {
            return "$context uses legacy functionCall. Use {\"action\":\"...\",\"params\":{...}}."
        }
        val action = obj.get("action")
        if (action == null || !action.isJsonPrimitive || !action.asJsonPrimitive.isString || action.asString.isBlank()) {
            return "$context action must be a non-empty string."
        }
        val params = obj.get("params")
        if (params != null && !params.isJsonNull && !params.isJsonObject) {
            return "$context params must be an object when present."
        }
        return null
    }

    private fun validateStackProps(props: JsonObject, id: String): String? {
        val directionError = validateStringTokenProp(
            props = props,
            prop = "direction",
            allowed = setOf("horizontal", "vertical"),
            context = "Element '$id' Stack"
        )
        if (directionError != null) return directionError

        val gapError = validateStringTokenProp(
            props = props,
            prop = "gap",
            allowed = setOf("none", "sm", "md", "lg", "xl"),
            context = "Element '$id' Stack"
        )
        if (gapError != null) return gapError

        val alignError = validateStringTokenProp(
            props = props,
            prop = "align",
            allowed = setOf("start", "center", "end", "stretch"),
            context = "Element '$id' Stack"
        )
        if (alignError != null) return alignError

        val justifyError = validateStringTokenProp(
            props = props,
            prop = "justify",
            allowed = setOf("start", "center", "end", "between", "around"),
            context = "Element '$id' Stack"
        )
        if (justifyError != null) return justifyError

        val wrapError = validateStringTokenProp(
            props = props,
            prop = "wrap",
            allowed = setOf("nowrap", "wrap"),
            context = "Element '$id' Stack"
        )
        if (wrapError != null) return wrapError

        val numberProps = listOf(
            "padding",
            "paddingHorizontal",
            "paddingVertical",
            "margin",
            "marginHorizontal",
            "marginVertical",
            "width",
            "height",
            "flex"
        )
        numberProps.forEach { prop ->
            val numericError = validateNumericProp(
                props = props,
                prop = prop,
                context = "Element '$id' Stack"
            )
            if (numericError != null) return numericError
        }
        return null
    }

    private fun validateStringTokenProp(
        props: JsonObject,
        prop: String,
        allowed: Set<String>,
        context: String
    ): String? {
        val value = props.get(prop) ?: return null
        if (value.isJsonObject) return null
        if (!value.isJsonPrimitive || !value.asJsonPrimitive.isString) {
            return "$context props.$prop must be one of: ${allowed.joinToString(", ")}."
        }
        val token = value.asString.trim().lowercase()
        if (!allowed.contains(token)) {
            return "$context props.$prop '$token' is unsupported. Allowed: ${allowed.joinToString(", ")}."
        }
        return null
    }

    private fun validateNumericProp(
        props: JsonObject,
        prop: String,
        context: String
    ): String? {
        val value = props.get(prop) ?: return null
        if (value.isJsonObject) return null
        if (!value.isJsonPrimitive || !value.asJsonPrimitive.isNumber) {
            return "$context props.$prop must be numeric."
        }
        return null
    }

    fun buildFallbackFlatSpec(stage2Response: String): JsonObject {
        val textValue = stage2Response.trim().ifBlank { "No content generated." }
        val rootId = "root"
        val textId = "text_1"
        return JsonObject().apply {
            addProperty("root", rootId)
            add("state", JsonObject())
            add("elements", JsonObject().apply {
                add(rootId, JsonObject().apply {
                    addProperty("type", "Stack")
                    add("props", JsonObject().apply {
                        addProperty("direction", "vertical")
                        addProperty("gap", "md")
                    })
                    add("children", JsonArray().apply { add(textId) })
                })
                add(textId, JsonObject().apply {
                    addProperty("type", "Text")
                    add("props", JsonObject().apply {
                        addProperty("variant", "body")
                        addProperty("text", textValue)
                    })
                    add("children", JsonArray())
                })
            })
        }
    }

    private fun canonicalizeFlatSpec(raw: JsonObject): JsonObject {
        val root = raw.get("root")?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
        val sourceElements = raw.get("elements")?.takeIf { it.isJsonObject }?.asJsonObject
        val canonicalElements = JsonObject()

        sourceElements?.entrySet()?.forEach { (id, elementValue) ->
            if (!elementValue.isJsonObject) return@forEach
            val element = elementValue.asJsonObject.deepCopy()
            if (!element.has("props") || !element.get("props").isJsonObject) {
                element.add("props", JsonObject())
            }
            if (!element.has("children") || !element.get("children").isJsonArray) {
                element.add("children", JsonArray())
            }
            canonicalElements.add(id, element)
        }

        val fallbackRoot = canonicalElements.entrySet().firstOrNull()?.key ?: "root"
        val resolvedRoot = if (!root.isNullOrBlank() && canonicalElements.has(root)) root else fallbackRoot
        return JsonObject().apply {
            addProperty("root", resolvedRoot)
            add(
                "state",
                raw.get("state")?.takeIf { it.isJsonObject }?.asJsonObject?.deepCopy() ?: JsonObject()
            )
            add("elements", canonicalElements)
        }
    }

    private fun unwrapKnownContainers(json: JsonElement): JsonElement? {
        if (!json.isJsonObject) return null
        val obj = json.asJsonObject

        obj.get("genui_json")?.let { return it }
        obj.get("messages")?.let { return it }
        obj.get("payload")?.let { payload ->
            if (payload.isJsonArray) return payload
            if (payload.isJsonObject) {
                payload.asJsonObject.get("messages")?.let { return it }
                payload.asJsonObject.get("genui_json")?.let { return it }
            }
        }
        return null
    }

    private fun convertLegacyMessages(messages: JsonArray): JsonObject? {
        val componentsArray = extractLegacyComponents(messages) ?: return null

        val elements = JsonObject()
        var rootId: String? = null
        componentsArray.forEach { component ->
            if (!component.isJsonObject) return@forEach
            val obj = component.asJsonObject
            val id = obj.get("id")
                ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                ?.asString
                ?: return@forEach
            val type = obj.get("component")
                ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                ?.asString
                ?: obj.get("type")
                    ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                    ?.asString
                ?: return@forEach
            val normalizedType = normalizeLegacyType(type)

            if (rootId == null && id == "root") {
                rootId = id
            }

            val props = JsonObject()
            val children = JsonArray()

            obj.entrySet().forEach { (key, value) ->
                when (key) {
                    "id", "component", "type", "children" -> Unit
                    "child" -> {
                        props.add(key, value.deepCopy())
                        if (value.isJsonPrimitive && value.asJsonPrimitive.isString) {
                            addUnique(children, value.asString)
                        }
                    }
                    else -> props.add(key, value.deepCopy())
                }
            }
            if (normalizedType == "Stack") {
                normalizeLegacyStackProps(type, props)
            }

            obj.get("children")?.takeIf { it.isJsonArray }?.asJsonArray?.forEach { child ->
                if (child.isJsonPrimitive && child.asJsonPrimitive.isString) {
                    addUnique(children, child.asString)
                }
            }

            elements.add(id, JsonObject().apply {
                addProperty("type", normalizedType)
                add("props", props)
                add("children", children)
            })
        }

        if (elements.size() == 0) {
            return null
        }

        val resolvedRoot = when {
            !rootId.isNullOrBlank() && elements.has(rootId) -> rootId
            elements.has("root") -> "root"
            else -> elements.entrySet().first().key
        }

        return JsonObject().apply {
            addProperty("root", resolvedRoot)
            add("state", JsonObject())
            add("elements", elements)
        }
    }

    private fun extractLegacyComponents(messages: JsonArray): JsonArray? {
        messages.forEach { message ->
            if (!message.isJsonObject) return@forEach
            val obj = message.asJsonObject
            val update = obj.getAsJsonObject("updateComponents")
            if (update != null) {
                val components = update.get("components")
                if (components != null && components.isJsonArray) {
                    return components.asJsonArray
                }
            }
            val directComponents = obj.get("components")
            if (directComponents != null && directComponents.isJsonArray) {
                return directComponents.asJsonArray
            }
        }

        val looksLikeComponentArray = messages.all {
            it.isJsonObject &&
                it.asJsonObject.get("id")?.let { id -> id.isJsonPrimitive && id.asJsonPrimitive.isString } == true &&
                (
                    it.asJsonObject.get("component")?.let { c -> c.isJsonPrimitive && c.asJsonPrimitive.isString } == true ||
                        it.asJsonObject.get("type")?.let { t -> t.isJsonPrimitive && t.asJsonPrimitive.isString } == true
                    )
        }
        return if (looksLikeComponentArray) messages else null
    }

    private fun addUnique(array: JsonArray, value: String) {
        val hasValue = array.any { it.isJsonPrimitive && it.asJsonPrimitive.isString && it.asString == value }
        if (!hasValue) {
            array.add(value)
        }
    }

    private fun normalizeLegacyType(type: String): String = when (type.trim().lowercase()) {
        "column", "row" -> "Stack"
        else -> type
    }

    private fun normalizeLegacyStackProps(originalType: String, props: JsonObject) {
        val normalizedOriginal = originalType.trim().lowercase()
        val defaultDirection = when (normalizedOriginal) {
            "row" -> "horizontal"
            "column" -> "vertical"
            else -> null
        }
        if (defaultDirection != null && !props.has("direction")) {
            props.addProperty("direction", defaultDirection)
        }

        val align = props.get("align")
            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.let(::mapLegacyAlignToken)
        if (!align.isNullOrBlank()) {
            props.addProperty("align", align)
        }

        val justify = props.get("justify")
            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.let(::mapLegacyJustifyToken)
        if (!justify.isNullOrBlank()) {
            props.addProperty("justify", justify)
        }

        val wrap = props.get("wrap")
            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.let(::mapLegacyWrapToken)
        if (!wrap.isNullOrBlank()) {
            props.addProperty("wrap", wrap)
        }
    }

    private fun mapLegacyAlignToken(raw: String): String {
        return when (raw.trim().lowercase()) {
            "center", "middle" -> "center"
            "end", "flexend", "right", "bottom" -> "end"
            "stretch" -> "stretch"
            else -> "start"
        }
    }

    private fun mapLegacyJustifyToken(raw: String): String {
        return when (raw.trim().lowercase()) {
            "center" -> "center"
            "end", "flexend", "right", "bottom" -> "end"
            "between", "spacebetween", "space-between" -> "between"
            "around", "spacearound", "space-around", "spaceevenly", "space-evenly", "evenly" -> "around"
            else -> "start"
        }
    }

    private fun mapLegacyWrapToken(raw: String): String {
        return when (raw.trim().lowercase()) {
            "wrap" -> "wrap"
            else -> "nowrap"
        }
    }
}
