package com.samsung.genuicraft.pipeline

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.samsung.genuicraft.renderer.flat.capability.GeneratedRendererCapabilities
import com.samsung.genuicraft.security.SafeContentPolicy
import java.util.Locale
import com.samsung.genuicraft.renderer.flat.legacy.*

internal object FlatSpecContract {

    data class TableDiagnostics(
        val tableDetected: Boolean = false,
        val columns: Int = 0,
        val rows: Int = 0,
        val tableDomain: String = "none",
        val preferredPresentation: String = "table",
        val presentationChosen: String = "table",
        val renderMode: String = "none",
        val cardMappingStatus: String = "not_applicable",
        val mappingWarnings: List<String> = emptyList(),
        val canonicalizationRewrites: List<String> = emptyList(),
        val irElementCount: Int = 0,
        val irByteSize: Int = 0,
        val compactionApplied: Boolean = false,
        val removedFieldCount: Int = 0
    )

    data class NormalizeResult(
        val spec: JsonObject?,
        val convertedFromLegacy: Boolean,
        val error: String? = null,
        val warnings: List<String> = emptyList(),
        val tableDiagnostics: TableDiagnostics = TableDiagnostics()
    )

    data class ValidationResult(
        val isValid: Boolean,
        val error: String? = null
    )

    data class CoerceResult(
        val spec: JsonObject?,
        val convertedFromLegacy: Boolean,
        val error: String?,
        val warnings: List<String> = emptyList(),
        val tableDiagnostics: TableDiagnostics = TableDiagnostics()
    ) {
        val isValid: Boolean get() = spec != null && error == null
    }

    private val allowedTypes = GeneratedRendererCapabilities.typeAliases.keys
    private val cardFirstTableDomains = GeneratedRendererCapabilities.cardFirstTableDomains
    private val supportedTableDomains = GeneratedRendererCapabilities.tableDomains

    fun looksLikeFlatSpec(json: JsonElement?): Boolean {
        if (json == null || !json.isJsonObject) return false
        val obj = json.asJsonObject
        return obj.has("root") && obj.has("elements")
    }

    fun coerceAndValidate(
        json: JsonElement?,
        compatibilityHeaderInference: Boolean = false
    ): CoerceResult {
        if (json == null) {
            return CoerceResult(spec = null, convertedFromLegacy = false, error = "Stage 3 output is null.")
        }
        val normalized = normalizeToFlatSpec(json, compatibilityHeaderInference)
        val spec = normalized.spec ?: return CoerceResult(
            spec = null,
            convertedFromLegacy = normalized.convertedFromLegacy,
            error = normalized.error ?: "Could not normalize Stage 3 output to flat spec.",
            warnings = normalized.warnings,
            tableDiagnostics = normalized.tableDiagnostics
        )
        val validation = validateFlatSpec(spec)
        if (!validation.isValid) {
            return CoerceResult(
                spec = null,
                convertedFromLegacy = normalized.convertedFromLegacy,
                error = validation.error ?: "Flat spec validation failed.",
                warnings = normalized.warnings,
                tableDiagnostics = normalized.tableDiagnostics
            )
        }
        return CoerceResult(
            spec = spec,
            convertedFromLegacy = normalized.convertedFromLegacy,
            error = null,
            warnings = normalized.warnings,
            tableDiagnostics = normalized.tableDiagnostics
        )
    }

    fun normalizeToFlatSpec(
        json: JsonElement,
        compatibilityHeaderInference: Boolean = false
    ): NormalizeResult {
        if (looksLikeFlatSpec(json)) {
            val canonical = canonicalizeFlatSpec(json.asJsonObject, compatibilityHeaderInference)
            return NormalizeResult(
                spec = canonical.spec,
                convertedFromLegacy = false,
                warnings = canonical.rewrites,
                tableDiagnostics = canonical.tableDiagnostics
            )
        }

        val unwrapped = unwrapKnownContainers(json)
        if (unwrapped != null) {
            if (looksLikeFlatSpec(unwrapped)) {
                val canonical = canonicalizeFlatSpec(unwrapped.asJsonObject, compatibilityHeaderInference)
                return NormalizeResult(
                    spec = canonical.spec,
                    convertedFromLegacy = true,
                    warnings = canonical.rewrites,
                    tableDiagnostics = canonical.tableDiagnostics
                )
            }
            if (unwrapped.isJsonArray) {
                val converted = convertLegacyMessages(unwrapped.asJsonArray)
                return if (converted != null) {
                    val canonical = canonicalizeFlatSpec(converted, compatibilityHeaderInference)
                    NormalizeResult(
                        spec = canonical.spec,
                        convertedFromLegacy = true,
                        warnings = canonical.rewrites,
                        tableDiagnostics = canonical.tableDiagnostics
                    )
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
                    val canonical = canonicalizeFlatSpec(converted, compatibilityHeaderInference)
                    return NormalizeResult(
                        spec = canonical.spec,
                        convertedFromLegacy = true,
                        warnings = canonical.rewrites,
                        tableDiagnostics = canonical.tableDiagnostics
                    )
                }
            }
        }

        if (json.isJsonArray) {
            val converted = convertLegacyMessages(json.asJsonArray)
            return if (converted != null) {
                val canonical = canonicalizeFlatSpec(converted, compatibilityHeaderInference)
                NormalizeResult(
                    spec = canonical.spec,
                    convertedFromLegacy = true,
                    warnings = canonical.rewrites,
                    tableDiagnostics = canonical.tableDiagnostics
                )
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
                val canonical = canonicalizeFlatSpec(converted, compatibilityHeaderInference)
                return NormalizeResult(
                    spec = canonical.spec,
                    convertedFromLegacy = true,
                    warnings = canonical.rewrites,
                    tableDiagnostics = canonical.tableDiagnostics
                )
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
        val state = spec.get("state")?.takeIf { it.isJsonObject }?.asJsonObject ?: JsonObject()
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
            val mediaError = validateMediaProps(
                type = type.asString,
                props = propsObject,
                state = state,
                id = id
            )
            if (mediaError != null) {
                return ValidationResult(false, mediaError)
            }
            val actionUrlError = validateActionUrlProps(
                type = type.asString,
                props = propsObject,
                id = id
            )
            if (actionUrlError != null) {
                return ValidationResult(false, actionUrlError)
            }

            val children = element.get("children")
            if (children == null || !children.isJsonArray) {
                return ValidationResult(false, "Element '$id' must define children as an array.")
            }
            children.asJsonArray.forEach { child ->
                if (!child.isJsonPrimitive || !child.asJsonPrimitive.isString) {
                    return ValidationResult(false, "Element '$id' contains a non-string child reference.")
                }
            }
            FlatSpecReferenceSemantics.references(element).forEach { reference ->
                if (!ids.contains(reference.targetId)) {
                    val noun = if (reference.kind == "child") "child" else "element"
                    return ValidationResult(
                        false,
                        "Element '$id' references missing $noun '${reference.targetId}' at ${reference.sourcePath}."
                    )
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
        val visiting = linkedSetOf<String>()
        val visited = linkedSetOf<String>()
        fun visit(elementId: String): String? {
            if (elementId in visiting) return elementId
            if (elementId in visited) return null
            visiting += elementId
            val element = elements.get(elementId)?.takeIf { it.isJsonObject }?.asJsonObject
            element?.let {
                FlatSpecReferenceSemantics.references(it).forEach { reference ->
                    visit(reference.targetId)?.let { cycle -> return cycle }
                }
            }
            visiting -= elementId
            visited += elementId
            return null
        }
        visit(root.asString)?.let { cycle ->
            return ValidationResult(false, "Renderer reference cycle is reachable from root through element '$cycle'.")
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
        val unsupportedKeys = obj.entrySet().map { it.key }.toSet() - setOf("action", "params")
        if (unsupportedKeys.isNotEmpty()) {
            return "$context contains unsupported action fields ${unsupportedKeys.sorted()}."
        }
        val actionKey = action.asString.trim().lowercase(Locale.US)
        if (actionKey !in GeneratedRendererCapabilities.actionRuntimeKeys) {
            return "$context uses unsupported action '${action.asString}'. Supported: ${GeneratedRendererCapabilities.actionNames.sorted()}."
        }
        val params = obj.get("params")
        if (params != null && !params.isJsonNull && !params.isJsonObject) {
            return "$context params must be an object when present."
        }
        val paramsObject = params?.takeIf { it.isJsonObject }?.asJsonObject ?: JsonObject()
        val missingRequired = GeneratedRendererCapabilities.actionRequired[actionKey]
            .orEmpty()
            .filterNot(paramsObject::has)
        if (missingRequired.isNotEmpty()) {
            return "$context action '${action.asString}' requires params ${missingRequired.sorted()}."
        }
        GeneratedRendererCapabilities.actionRequiredAny[actionKey].orEmpty().forEach { alternatives ->
            val hasAlternative = alternatives.any { key ->
                val value = paramsObject.get(key)
                value != null && !value.isJsonNull &&
                    (!value.isJsonPrimitive || !value.asJsonPrimitive.isString || value.asString.isNotBlank())
            }
            if (!hasAlternative) {
                return "$context action '${action.asString}' requires one of params ${alternatives.sorted()}."
            }
        }
        if (actionKey in GeneratedRendererCapabilities.safeUrlActions) {
            val url = firstStringProp(paramsObject, "url", "href", "link", "targetUrl")
            if (url != null && SafeContentPolicy.sanitizeActionUrl(url) == null) {
                return "$context openUrl contains unsafe URL."
            }
        }
        if (actionKey == "removestate") {
            val index = paramsObject.get("index")
            val numericIndex = index
                ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isNumber }
                ?.asDouble
            if (numericIndex == null || numericIndex < 0.0 || numericIndex % 1.0 != 0.0) {
                return "$context removeState params.index must be a non-negative integer."
            }
        }
        return null
    }

    private data class TableMediaColumn(
        val index: Int,
        val key: String,
        val label: String
    )

    private fun validateMediaProps(
        type: String,
        props: JsonObject,
        state: JsonObject,
        id: String
    ): String? {
        if (type.equals("image", ignoreCase = true)) {
            val imageUrl = firstStringProp(props, "url", "src", "image", "source", "name")
            if (imageUrl != null && !SafeContentPolicy.isSafeMediaUrl(imageUrl, SafeContentPolicy.MediaKind.IMAGE)) {
                return if (SafeContentPolicy.isIconOnlyMediaUrl(imageUrl)) {
                    "Element '$id' Image source points to icon/vector media. Use Icon instead of Image."
                } else {
                    "Element '$id' Image source is not allowed by the safe media policy."
                }
            }
        }
        if (type.equals("icon", ignoreCase = true)) {
            val iconUrl = firstStringProp(props, "name", "icon", "source", "url", "src")
            if (iconUrl != null &&
                (SafeContentPolicy.looksLikeUrl(iconUrl) || SafeContentPolicy.isLocalAssetUrl(iconUrl)) &&
                !SafeContentPolicy.isSafeMediaUrl(iconUrl, SafeContentPolicy.MediaKind.ICON)
            ) {
                return "Element '$id' Icon source is not allowed by the safe media policy."
            }
        }
        if (type.equals("video", ignoreCase = true)) {
            val videoUrl = firstStringProp(props, "url", "src", "source")
            if (videoUrl != null && !SafeContentPolicy.isSafeMediaUrl(videoUrl, SafeContentPolicy.MediaKind.VIDEO)) {
                return "Element '$id' Video source is not allowed by the safe media policy."
            }
        }
        if (type.equals("audioplayer", ignoreCase = true)) {
            val audioUrl = firstStringProp(props, "url", "src", "source")
            if (audioUrl != null && !SafeContentPolicy.isSafeMediaUrl(audioUrl, SafeContentPolicy.MediaKind.AUDIO)) {
                return "Element '$id' AudioPlayer source is not allowed by the safe media policy."
            }
        }
        if (type.equals("table", ignoreCase = true)) {
            return validateTableImageColumns(props, state, id)
        }
        return null
    }

    private fun validateTableImageColumns(
        props: JsonObject,
        state: JsonObject,
        id: String
    ): String? {
        val imageColumns = tableImageColumns(props)
        if (imageColumns.isEmpty()) return null
        val rows = tableRows(props, state)
        rows.forEach { row ->
            imageColumns.forEach { column ->
                val value = tableCell(row, column)?.trim().orEmpty()
                if (value.isNotBlank() && !SafeContentPolicy.isSafeMediaUrl(value, SafeContentPolicy.MediaKind.IMAGE)) {
                    return if (SafeContentPolicy.isIconOnlyMediaUrl(value)) {
                        "Element '$id' Table image column '${column.label}' uses icon/vector media. Use an icon column or omit the image."
                    } else {
                        "Element '$id' Table image column '${column.label}' uses media that is not allowed by the safe media policy."
                    }
                }
            }
        }
        return null
    }

    private fun validateActionUrlProps(
        type: String,
        props: JsonObject,
        id: String
    ): String? {
        if (type.equals("image", ignoreCase = true) ||
            type.equals("icon", ignoreCase = true) ||
            type.equals("video", ignoreCase = true) ||
            type.equals("audioplayer", ignoreCase = true) ||
            type.equals("table", ignoreCase = true)
        ) {
            return null
        }
        listOf("url", "actionUrl", "bookingUrl", "buttonUrl", "sourceUrl", "targetUrl", "href", "link").forEach { key ->
            val value = firstStringProp(props, key) ?: return@forEach
            if (SafeContentPolicy.looksLikeUrl(value) && SafeContentPolicy.sanitizeActionUrl(value) == null) {
                return "Element '$id' prop '$key' contains unsafe URL."
            }
        }
        return null
    }

    private fun tableImageColumns(props: JsonObject): List<TableMediaColumn> {
        val columns = props.get("columns")?.takeIf { it.isJsonArray }?.asJsonArray ?: return emptyList()
        return columns.mapIndexedNotNull { index, entry ->
            val column = entry.takeIf { it.isJsonObject }?.asJsonObject ?: return@mapIndexedNotNull null
            val key = column.get("key")?.takeIf { it.isJsonPrimitive }?.asString.orEmpty()
            val label = column.get("label")?.takeIf { it.isJsonPrimitive }?.asString.orEmpty()
            if (isImageColumnToken(key) || isImageColumnToken(label)) {
                TableMediaColumn(
                    index = index,
                    key = key.ifBlank { label },
                    label = label.ifBlank { key.ifBlank { "image" } }
                )
            } else {
                null
            }
        }
    }

    private fun tableRows(props: JsonObject, state: JsonObject): JsonArray {
        props.get("rows")?.takeIf { it.isJsonArray }?.asJsonArray?.let { return it }
        val statePath = props.get("statePath")
            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            .orEmpty()
        return resolveStateElement(state, statePath)
            ?.takeIf { it.isJsonArray }
            ?.asJsonArray
            ?: JsonArray()
    }

    private fun tableCell(row: JsonElement, column: TableMediaColumn): String? {
        return when {
            row.isJsonObject -> {
                val obj = row.asJsonObject
                listOf(column.key, column.label)
                    .filter { it.isNotBlank() }
                    .firstNotNullOfOrNull { key ->
                        obj.get(key)?.let(::jsonPrimitiveString)
                    }
            }
            row.isJsonArray -> row.asJsonArray.getOrNull(column.index)?.let(::jsonPrimitiveString)
            else -> null
        }
    }

    private fun JsonArray.getOrNull(index: Int): JsonElement? =
        if (index in 0 until size()) get(index) else null

    private fun firstStringProp(props: JsonObject, vararg keys: String): String? =
        keys.firstNotNullOfOrNull { key -> props.get(key)?.let(::jsonPrimitiveString) }

    private fun jsonPrimitiveString(value: JsonElement): String? {
        return value
            .takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.trim()
            ?.takeIf { it.isNotBlank() }
    }

    private fun isImageColumnToken(raw: String): Boolean {
        val token = raw
            .trim()
            .lowercase(Locale.US)
            .replace(Regex("[^a-z0-9]+"), " ")
            .trim()
        return token in setOf(
            "image",
            "photo",
            "picture",
            "thumbnail",
            "hero image",
            "image url",
            "photo url",
            "media image"
        )
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

    private data class FallbackAction(
        val id: String,
        val label: String,
        val eventName: String,
        val reference: String,
        val sourceLine: Int,
    )

    private data class FallbackIcon(
        val id: String,
        val reference: String,
        val sourceLine: Int,
    )

    private data class FallbackTable(
        val columns: List<String>,
        val rows: List<List<String>>,
        val title: String,
        val sourceLines: Set<Int>,
    )

    private val FALLBACK_REFERENCE = Regex("""\{\{u\d+\}\}|(?:https?://|//)\S+""")
    private val FALLBACK_BUTTON_LINE = Regex(
        """(?i)^\s*(?:Action:\s*)?\[Button:\s*(.+?)]\s*(.*?)\s*$"""
    )
    private val FALLBACK_LABEL_REFERENCE_LINE = Regex(
        """^\s*(?:[-*]\s*)?([^:\r\n]{2,100}):\s*(\S.*?)\s*$"""
    )
    private val FALLBACK_MEDIA_ICON_LINE = Regex(
        """(?i)^\s*Media:\s*Icon\s*=\s*(\{\{u\d+\}\}|(?:https?://|//)\S+)\s*$"""
    )
    private val FALLBACK_MEDIA_LINE = Regex(
        """(?i)^\s*Media:\s*(?:Icon|Image|Photo)\s*=.*$"""
    )
    private val FALLBACK_ICON_LABELS = setOf(
        "airplane", "alarm", "airline", "bookmark star", "briefcase", "calculator", "camera",
        "camera video", "cash coin", "celebration", "clock", "dining", "dice", "fire", "inventory box",
        "map", "players", "rain cloud", "schedule", "stabilization", "status alert", "sun", "thermometer",
        "time", "weight", "basket", "bell", "upc scan",
    )

    fun buildFallbackFlatSpec(stage2Response: String): JsonObject {
        val fallbackActions = extractFallbackActions(stage2Response)
        val fallbackIcons = extractFallbackIcons(stage2Response)
        val fallbackTables = extractFallbackMarkdownTables(stage2Response)
        val textValue = buildFallbackDisplayText(
            response = stage2Response,
            actions = fallbackActions,
            icons = fallbackIcons,
            tables = fallbackTables,
        ).ifBlank { "No content generated." }
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
                    add("children", JsonArray().apply {
                        add(textId)
                        fallbackIcons.forEach { add(it.id) }
                        fallbackTables.indices.forEach { add("fallback_table_${it + 1}") }
                        fallbackActions.forEach { add(it.id) }
                    })
                })
                add(textId, JsonObject().apply {
                    addProperty("type", "Text")
                    add("props", JsonObject().apply {
                        addProperty("variant", "body")
                        addProperty("text", textValue)
                    })
                    add("children", JsonArray())
                })
                fallbackIcons.forEach { icon ->
                    add(icon.id, JsonObject().apply {
                        addProperty("type", "Icon")
                        add("props", JsonObject().apply {
                            addProperty("name", "inline_icon")
                            addProperty("url", icon.reference)
                            addProperty("size", "sm")
                            addProperty("tint", "gray")
                        })
                        add("children", JsonArray())
                    })
                }
                fallbackTables.forEachIndexed { index, table ->
                    val tableDomain = inferTableDomainFromHeaders(table.columns)
                    add("fallback_table_${index + 1}", JsonObject().apply {
                        addProperty("type", "Table")
                        add("props", JsonObject().apply {
                            add("columns", JsonArray().apply { table.columns.forEach { add(it) } })
                            add("rows", JsonArray().apply {
                                table.rows.forEach { row ->
                                    add(JsonArray().apply { row.forEach { add(it) } })
                                }
                            })
                            addProperty("title", table.title)
                            addProperty("domain", tableDomain)
                            addProperty(
                                "preferredPresentation",
                                if (tableDomain in cardFirstTableDomains) "cards" else "table"
                            )
                        })
                        add("children", JsonArray())
                    })
                }
                fallbackActions.forEach { action ->
                    add(action.id, JsonObject().apply {
                        addProperty("type", "Button")
                        add("props", JsonObject().apply {
                            addProperty("label", action.label)
                            addProperty("variant", "primary")
                        })
                        add("on", JsonObject().apply {
                            add("press", JsonObject().apply {
                                addProperty("action", "emitEvent")
                                add("params", JsonObject().apply {
                                    addProperty("name", action.eventName)
                                    add("context", JsonObject().apply {
                                        addProperty("url", action.reference)
                                    })
                                    addProperty("wantResponse", true)
                                    addProperty("responsePath", "/result")
                                })
                            })
                        })
                        add("children", JsonArray())
                    })
                }
            })
        }
    }

    private fun extractFallbackActions(response: String): List<FallbackAction> {
        val lines = response.replace("\r\n", "\n").lines()
        val actions = mutableListOf<FallbackAction>()
        var inQuickActions = false

        lines.forEachIndexed { index, rawLine ->
            val line = rawLine.trim()
            if (line.isBlank()) return@forEachIndexed
            val heading = line
                .replace(Regex("""^#{1,6}\s*"""), "")
                .trim()
                .trimEnd(':')
                .trim()
            if (heading.equals("Quick Actions", ignoreCase = true)) {
                inQuickActions = true
                return@forEachIndexed
            }
            if (inQuickActions && isFallbackSectionHeading(heading)) {
                inQuickActions = false
            }

            val buttonMatch = FALLBACK_BUTTON_LINE.matchEntire(line)
            if (buttonMatch != null) {
                val label = buttonMatch.groupValues[1].trim()
                val reference = FALLBACK_REFERENCE.find(buttonMatch.groupValues[2])
                    ?.value
                    ?.trimEnd('>', ']')
                    ?.takeIf { it.isNotBlank() }
                if (!reference.isNullOrBlank()) {
                    actions += FallbackAction(
                        id = "fallback_action_${actions.size + 1}",
                        label = label.ifBlank { "Continue" },
                        eventName = fallbackEventName(label, actions.size + 1),
                        reference = reference,
                        sourceLine = index,
                    )
                }
                return@forEachIndexed
            }

            if (!inQuickActions) return@forEachIndexed
            val labelReference = FALLBACK_LABEL_REFERENCE_LINE.matchEntire(line) ?: return@forEachIndexed
            val label = labelReference.groupValues[1].trim()
            val reference = FALLBACK_REFERENCE.find(labelReference.groupValues[2])
                ?.value
                ?.trimEnd('>', ']')
                ?.takeIf { it.isNotBlank() }
                ?: return@forEachIndexed
            actions += FallbackAction(
                id = "fallback_action_${actions.size + 1}",
                label = label,
                eventName = fallbackEventName(label, actions.size + 1),
                reference = reference,
                sourceLine = index,
            )
        }
        return actions.distinctBy { it.reference to it.label }
    }

    private fun extractFallbackIcons(response: String): List<FallbackIcon> {
        val lines = response.replace("\r\n", "\n").lines()
        val icons = mutableListOf<FallbackIcon>()
        var inIcons = false
        var inSources = false

        fun addIcon(reference: String, sourceLine: Int) {
            val normalized = reference.trimEnd('>', ']')
            if (normalized.isBlank() || icons.any { it.reference == normalized }) return
            icons += FallbackIcon(
                id = "fallback_icon_${icons.size + 1}",
                reference = normalized,
                sourceLine = sourceLine,
            )
        }

        lines.forEachIndexed { index, rawLine ->
            val line = rawLine.trim()
            if (line.isBlank()) return@forEachIndexed
            FALLBACK_MEDIA_ICON_LINE.matchEntire(line)?.let { match ->
                addIcon(match.groupValues[1], index)
                return@forEachIndexed
            }
            val heading = line
                .replace(Regex("""^#{1,6}\s*"""), "")
                .trim()
                .trimEnd(':')
                .trim()
            if (heading.equals("Icons", ignoreCase = true)) {
                inSources = false
                inIcons = true
                return@forEachIndexed
            }
            if (heading.equals("Sources", ignoreCase = true) ||
                heading.equals("Source", ignoreCase = true) ||
                heading.equals("References", ignoreCase = true)
            ) {
                inSources = true
                inIcons = false
                return@forEachIndexed
            }
            if (inSources) return@forEachIndexed
            val labelReference = FALLBACK_LABEL_REFERENCE_LINE.matchEntire(line) ?: return@forEachIndexed
            val label = labelReference.groupValues[1].trim()
            val reference = FALLBACK_REFERENCE.find(labelReference.groupValues[2])
                ?.value
                ?.trimEnd('>', ']')
                ?: return@forEachIndexed
            if (inIcons || isFallbackIconLabel(label) || reference.contains("/icons/", ignoreCase = true)) {
                addIcon(reference, index)
            }
        }
        return icons
    }

    private fun fallbackEventName(label: String, index: Int): String = label
        .lowercase(Locale.US)
        .replace(Regex("[^a-z0-9]+"), "_")
        .trim('_')
        .ifBlank { "action_$index" }

    private fun buildFallbackDisplayText(
        response: String,
        actions: List<FallbackAction>,
        icons: List<FallbackIcon>,
        tables: List<FallbackTable>,
    ): String {
        val lines = response.replace("\r\n", "\n").lines()
        val removedTableLines = tables.flatMap { it.sourceLines }.toSet()
        val removedActionLines = actions.map { it.sourceLine }.toSet()
        val removedIconLines = icons.map { it.sourceLine }.toSet()
        val output = mutableListOf<String>()
        var inIcons = false
        var inSources = false

        lines.forEachIndexed { index, rawLine ->
            val line = rawLine.trim()
            val heading = line
                .replace(Regex("""^#{1,6}\s*"""), "")
                .trim()
                .trimEnd(':')
                .trim()
            if (heading.equals("Sources", ignoreCase = true) ||
                heading.equals("Source", ignoreCase = true) ||
                heading.equals("References", ignoreCase = true)
            ) {
                inSources = true
                return@forEachIndexed
            }
            if (inSources && heading.equals("Icons", ignoreCase = true)) {
                inSources = false
                inIcons = true
                return@forEachIndexed
            }
            if (inSources || index in removedTableLines || index in removedActionLines || index in removedIconLines) {
                return@forEachIndexed
            }
            if (FALLBACK_MEDIA_LINE.matches(line)) return@forEachIndexed
            if (isFallbackBlankLabelLine(line)) return@forEachIndexed
            if (heading.equals("Icons", ignoreCase = true)) {
                inIcons = true
                return@forEachIndexed
            }
            if (inIcons && FALLBACK_LABEL_REFERENCE_LINE.matches(line)) return@forEachIndexed
            if (heading.equals("Quick Actions", ignoreCase = true)) return@forEachIndexed
            output += rawLine.trimEnd()
                .replace(Regex("""^\s*#{1,6}\s+"""), "")
                .replace("**", "")
        }

        return output.joinToString("\n")
            .replace(Regex("\n{3,}"), "\n\n")
            .trim()
    }

    private fun isFallbackSectionHeading(value: String): Boolean {
        if (value.isBlank()) return false
        return value.equals("Sources", ignoreCase = true) ||
            value.equals("Source", ignoreCase = true) ||
            value.equals("References", ignoreCase = true) ||
            value.equals("Icons", ignoreCase = true) ||
            value.equals("Quick Actions", ignoreCase = true) ||
            value.endsWith("Overview", ignoreCase = true) ||
            value.endsWith("Context", ignoreCase = true) ||
            value.startsWith("Step-by-Step", ignoreCase = true) ||
            value.startsWith("Detailed", ignoreCase = true) ||
            value.startsWith("Technical", ignoreCase = true) ||
            value.startsWith("Flight Comparison", ignoreCase = true) ||
            value.startsWith("Gross Income Breakdown", ignoreCase = true)
    }

    private fun isFallbackIconLabel(value: String): Boolean =
        value.trim().lowercase(Locale.US) in FALLBACK_ICON_LABELS

    private fun isFallbackBlankLabelLine(value: String): Boolean {
        val match = Regex("""^\s*(?:[-*]\s*)?([^:\r\n]{2,100}):\s*$""").matchEntire(value)
            ?: return false
        return match.groupValues[1].trim().lowercase(Locale.US) !in setOf(
            "ingredients",
            "instructions",
            "sources",
            "source",
            "references",
            "quick actions",
            "icons",
            "notes",
            "steps",
        )
    }

    private fun extractFallbackMarkdownTables(response: String): List<FallbackTable> {
        val lines = response.replace("\r\n", "\n").lines()
        val tables = mutableListOf<FallbackTable>()
        var index = 0
        while (index < lines.size) {
            if (fallbackTableCells(lines[index]) == null) {
                index++
                continue
            }
            val start = index
            val candidateLines = mutableListOf<Int>()
            while (index < lines.size && fallbackTableCells(lines[index]) != null) {
                candidateLines += index
                index++
            }
            if (candidateLines.size < 2) continue

            val parsed = candidateLines.mapNotNull { lineIndex ->
                fallbackTableCells(lines[lineIndex])?.let { lineIndex to it }
            }
            val headerPair = parsed.firstOrNull { !isFallbackTableSeparator(it.second) } ?: continue
            val columns = headerPair.second
            val rows = parsed
                .dropWhile { it.first <= headerPair.first }
                .filterNot { isFallbackTableSeparator(it.second) }
                .map { it.second }
                .filter { it.size == columns.size }
            if (rows.isEmpty()) continue

            val title = lines.subList(0, start)
                .asReversed()
                .firstOrNull { candidate ->
                    val trimmed = candidate.trim()
                    trimmed.isNotBlank() &&
                        fallbackTableCells(trimmed) == null &&
                        !FALLBACK_MEDIA_LINE.matches(trimmed) &&
                        !isFallbackBlankLabelLine(trimmed) &&
                        !trimmed.contains("http://", ignoreCase = true) &&
                        !trimmed.contains("https://", ignoreCase = true) &&
                        !trimmed.endsWith('.') &&
                        trimmed.length <= 80
                }
                ?.trim()
                ?.replace(Regex("""^\s*#{1,6}\s+"""), "")
                ?.replace("**", "")
                ?: "Details"
            tables += FallbackTable(
                columns = columns,
                rows = rows,
                title = title,
                sourceLines = candidateLines.toSet(),
            )
        }
        return tables
    }

    private fun fallbackTableCells(rawLine: String): List<String>? {
        val line = rawLine.trim()
        if (!line.contains('|')) return null
        val parts = line.split('|').toMutableList()
        if (line.startsWith('|')) parts.removeAt(0)
        if (line.endsWith('|') && parts.isNotEmpty()) parts.removeAt(parts.lastIndex)
        val cells = parts.map { it.trim().replace("**", "") }
        return cells.takeIf { it.size >= 2 && it.all(String::isNotBlank) }
    }

    private fun isFallbackTableSeparator(cells: List<String>): Boolean =
        cells.all { it.matches(Regex(":?-{3,}:?")) }

    private data class CanonicalizationResult(
        val spec: JsonObject,
        val rewrites: List<String>,
        val tableDiagnostics: TableDiagnostics
    )

    private data class CanonicalizationStats(
        var compactionApplied: Boolean = false,
        var removedFieldCount: Int = 0
    )

    private data class TableCandidate(
        val tableElementId: String,
        val columns: Int,
        val rows: Int,
        val domain: String,
        val preferredPresentation: String
    )

    private data class StaticTableCompactionPlan(
        val tableElementId: String,
        val headerRowId: String,
        val staticRowIds: List<String>,
        val columns: Int
    )

    private fun canonicalizeFlatSpec(
        raw: JsonObject,
        compatibilityHeaderInference: Boolean
    ): CanonicalizationResult {
        val rewrites = mutableListOf<String>()
        val stats = CanonicalizationStats()
        val root = raw.get("root")?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
        val sourceElements = raw.get("elements")?.takeIf { it.isJsonObject }?.asJsonObject
        val canonicalElements = JsonObject()

        sourceElements?.entrySet()?.forEach { (id, elementValue) ->
            if (!elementValue.isJsonObject) {
                rewrites += "Dropped non-object element '$id'."
                return@forEach
            }
            val element = elementValue.asJsonObject.deepCopy()
            if (!element.has("props") || !element.get("props").isJsonObject) {
                element.add("props", JsonObject())
                rewrites += "Element '$id': inserted empty props object."
            }
            if (!element.has("children") || !element.get("children").isJsonArray) {
                element.add("children", JsonArray())
                rewrites += "Element '$id': inserted empty children array."
            }
            canonicalizeElementAliases(id, element, rewrites)
            canonicalElements.add(id, element)
        }

        val fallbackRoot = canonicalElements.entrySet().firstOrNull()?.key ?: "root"
        val resolvedRoot = if (!root.isNullOrBlank() && canonicalElements.has(root)) {
            root
        } else {
            if (!root.isNullOrBlank()) {
                rewrites += "Root '$root' missing in elements. Rewritten to '$fallbackRoot'."
            }
            fallbackRoot
        }

        val stateObject = raw.get("state")
            ?.takeIf { it.isJsonObject }
            ?.asJsonObject
            ?.deepCopy()
            ?: JsonObject()

        compactStaticTablesToRepeat(
            elements = canonicalElements,
            state = stateObject,
            rewrites = rewrites,
            stats = stats
        )
        applyTableDomainDefaults(
            elements = canonicalElements,
            rewrites = rewrites,
            compatibilityHeaderInference = compatibilityHeaderInference
        )
        alignTableColumnsWithRows(
            elements = canonicalElements,
            state = stateObject,
            rewrites = rewrites
        )
        stats.removedFieldCount += pruneRedundantElementPayload(
            elements = canonicalElements,
            rewrites = rewrites
        )
        stats.removedFieldCount += pruneUnreachableElements(
            rootId = resolvedRoot,
            elements = canonicalElements,
            rewrites = rewrites
        )

        val spec = JsonObject().apply {
            addProperty("root", resolvedRoot)
            add("state", stateObject)
            add("elements", canonicalElements)
        }
        val tableDiagnostics = buildTableDiagnostics(
            spec = spec,
            rewrites = rewrites,
            stats = stats,
            compatibilityHeaderInference = compatibilityHeaderInference
        )
        return CanonicalizationResult(
            spec = spec,
            rewrites = rewrites.toList(),
            tableDiagnostics = tableDiagnostics
        )
    }

    private fun compactStaticTablesToRepeat(
        elements: JsonObject,
        state: JsonObject,
        rewrites: MutableList<String>,
        stats: CanonicalizationStats
    ) {
        val plans = elements.entrySet()
            .mapNotNull { (elementId, _) -> planStaticTableCompaction(elementId, elements) }
        plans.forEach { plan ->
            val tableElement = elements.get(plan.tableElementId)?.takeIf { it.isJsonObject }?.asJsonObject ?: return@forEach
            val headerRowElement = elements.get(plan.headerRowId)?.takeIf { it.isJsonObject }?.asJsonObject ?: return@forEach
            val headerChildren = extractChildIds(headerRowElement)
            if (headerChildren.isEmpty()) return@forEach

            val firstRowId = plan.staticRowIds.firstOrNull() ?: return@forEach
            val firstRow = elements.get(firstRowId)?.takeIf { it.isJsonObject }?.asJsonObject ?: return@forEach
            val firstRowChildren = extractChildIds(firstRow)
            if (firstRowChildren.isEmpty()) return@forEach

            val rowArray = JsonArray()
            val columns = maxOf(plan.columns, 2)
            var rowIndex = 0
            val sampleCellElements = mutableListOf<JsonObject?>()
            repeat(columns) { sampleCellElements += null }
            var canCompact = true
            plan.staticRowIds.forEach { rowId ->
                val rowElement = elements.get(rowId)?.takeIf { it.isJsonObject }?.asJsonObject
                if (rowElement == null) {
                    canCompact = false
                    return@forEach
                }
                val rowChildren = extractChildIds(rowElement)
                if (rowChildren.isEmpty()) {
                    canCompact = false
                    return@forEach
                }
                val rowObj = JsonObject()
                rowObj.addProperty("row_id", "r_${rowIndex + 1}")
                for (columnIndex in 0 until columns) {
                    val cellId = rowChildren.getOrNull(columnIndex)
                    val cellElement = cellId
                        ?.let(elements::get)
                        ?.takeIf { it.isJsonObject }
                        ?.asJsonObject
                    if (sampleCellElements[columnIndex] == null && cellElement != null) {
                        sampleCellElements[columnIndex] = cellElement
                    }
                    val cellValue = extractCompactionCellValue(cellElement)
                    if (cellValue == null) {
                        canCompact = false
                        break
                    }
                    rowObj.add("c${columnIndex + 1}", cellValue)
                }
                if (!canCompact) {
                    return@forEach
                }
                rowArray.add(rowObj)
                rowIndex += 1
            }

            if (!canCompact || rowArray.size() == 0) {
                return@forEach
            }

            val stateKey = uniqueStateKey(state, "${plan.tableElementId}_rows")
            state.add(stateKey, rowArray)
            val bodyId = uniqueElementId(elements, "${plan.tableElementId}_body_rows")
            val rowTemplateId = uniqueElementId(elements, "${plan.tableElementId}_row_template")
            val rowTemplateChildren = JsonArray()
            for (columnIndex in 0 until columns) {
                val cellId = uniqueElementId(elements, "${rowTemplateId}_c${columnIndex + 1}")
                rowTemplateChildren.add(cellId)
                elements.add(
                    cellId,
                    buildTemplateCellElement(
                        prototype = sampleCellElements.getOrNull(columnIndex),
                        columnKey = "c${columnIndex + 1}"
                    )
                )
            }

            val rowTemplateElement = JsonObject().apply {
                addProperty("type", "Stack")
                add("props", firstRow.get("props")?.takeIf { it.isJsonObject }?.asJsonObject?.deepCopy() ?: JsonObject())
                add("children", rowTemplateChildren)
            }
            val rowTemplateProps = rowTemplateElement.getAsJsonObject("props")
            if (!rowTemplateProps.has("direction")) {
                rowTemplateProps.addProperty("direction", "horizontal")
            }
            elements.add(rowTemplateId, rowTemplateElement)

            val bodyElement = JsonObject().apply {
                addProperty("type", "Stack")
                add("props", JsonObject().apply {
                    addProperty("direction", "vertical")
                })
                add(
                    "repeat",
                    JsonObject().apply {
                        addProperty("statePath", "/$stateKey")
                        addProperty("key", "row_id")
                    }
                )
                add("children", JsonArray().apply { add(rowTemplateId) })
            }
            elements.add(bodyId, bodyElement)

            tableElement.add(
                "children",
                JsonArray().apply {
                    add(plan.headerRowId)
                    add(bodyId)
                }
            )

            rewrites += "Element '${plan.tableElementId}': compacted ${plan.staticRowIds.size} static table rows into repeat state '/$stateKey'."
            stats.compactionApplied = true
        }
    }

    private fun planStaticTableCompaction(
        tableElementId: String,
        elements: JsonObject
    ): StaticTableCompactionPlan? {
        val container = elements.get(tableElementId)?.takeIf { it.isJsonObject }?.asJsonObject ?: return null
        val childIds = extractChildIds(container)
        if (childIds.size < 3) return null

        val headerEntry = childIds
            .mapNotNull { childId ->
                val element = elements.get(childId)?.takeIf { it.isJsonObject }?.asJsonObject ?: return@mapNotNull null
                childId to element
            }
            .firstOrNull { (_, element) ->
                isRowLikeElement(element) && !element.has("repeat") && extractChildIds(element).size >= 2
            } ?: return null

        val repeatedBodyExists = childIds
            .filter { it != headerEntry.first }
            .any { childId ->
                val element = elements.get(childId)?.takeIf { it.isJsonObject }?.asJsonObject ?: return@any false
                element.has("repeat")
            }
        if (repeatedBodyExists) return null

        val staticRows = childIds
            .filter { it != headerEntry.first }
            .mapNotNull { childId ->
                val element = elements.get(childId)?.takeIf { it.isJsonObject }?.asJsonObject ?: return@mapNotNull null
                if (isRowLikeElement(element) && !element.has("repeat")) childId else null
            }
        if (staticRows.size < 2) return null

        val headerColumns = extractChildIds(headerEntry.second).size
        if (headerColumns < 2) return null
        val staticColumns = staticRows.maxOfOrNull { rowId ->
            val row = elements.get(rowId)?.takeIf { it.isJsonObject }?.asJsonObject ?: return@maxOfOrNull 0
            extractChildIds(row).size
        } ?: 0
        val columns = maxOf(headerColumns, staticColumns)
        if (columns < 2) return null

        return StaticTableCompactionPlan(
            tableElementId = tableElementId,
            headerRowId = headerEntry.first,
            staticRowIds = staticRows,
            columns = columns
        )
    }

    private fun extractCompactionCellValue(cellElement: JsonObject?): JsonElement? {
        if (cellElement == null) return null
        val props = cellElement.get("props")?.takeIf { it.isJsonObject }?.asJsonObject ?: return null
        val keys = listOf("text", "label", "title", "value", "content")
        val value = keys.firstNotNullOfOrNull { key ->
            props.get(key)?.takeIf { !it.isJsonNull }
        }
        return value?.deepCopy()
    }

    private fun buildTemplateCellElement(
        prototype: JsonObject?,
        columnKey: String
    ): JsonObject {
        val out = prototype?.deepCopy() ?: JsonObject()
        if (!out.has("type") || out.get("type")?.takeIf { it.isJsonPrimitive }?.asString?.isBlank() != false) {
            out.addProperty("type", "Text")
        }
        val props = out.get("props")?.takeIf { it.isJsonObject }?.asJsonObject ?: JsonObject().also { out.add("props", it) }
        val targetProp = listOf("text", "label", "title", "value", "content")
            .firstOrNull { props.has(it) }
            ?: "text"
        props.add(
            targetProp,
            JsonObject().apply {
                addProperty("\$item", columnKey)
            }
        )
        out.remove("repeat")
        out.remove("on")
        out.remove("watch")
        out.add("children", JsonArray())
        return out
    }

    private fun uniqueStateKey(state: JsonObject, base: String): String {
        val normalizedBase = base
            .trim()
            .replace(Regex("[^A-Za-z0-9_]+"), "_")
            .trim('_')
            .ifBlank { "table_rows" }
        if (!state.has(normalizedBase)) return normalizedBase
        var index = 2
        while (state.has("${normalizedBase}_$index")) {
            index += 1
        }
        return "${normalizedBase}_$index"
    }

    private fun uniqueElementId(elements: JsonObject, base: String): String {
        val normalizedBase = base
            .trim()
            .replace(Regex("[^A-Za-z0-9_]+"), "_")
            .trim('_')
            .ifBlank { "element" }
        if (!elements.has(normalizedBase)) return normalizedBase
        var index = 2
        while (elements.has("${normalizedBase}_$index")) {
            index += 1
        }
        return "${normalizedBase}_$index"
    }

    private fun applyTableDomainDefaults(
        elements: JsonObject,
        rewrites: MutableList<String>,
        compatibilityHeaderInference: Boolean
    ) {
        elements.entrySet()
            .mapNotNull { (elementId, _) ->
                detectTableCandidate(elementId, elements, JsonObject(), compatibilityHeaderInference)
            }
            .forEach { candidate ->
                val tableElement = elements.get(candidate.tableElementId)?.takeIf { it.isJsonObject }?.asJsonObject ?: return@forEach
                val props = tableElement.get("props")?.takeIf { it.isJsonObject }?.asJsonObject ?: JsonObject().also {
                    tableElement.add("props", it)
                }
                val rawDomain = props.get("domain")
                    ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                    ?.asString
                val normalizedDomain = normalizeManifestTableDomain(rawDomain)
                if (normalizedDomain !in supportedTableDomains) {
                    props.addProperty("domain", candidate.domain)
                    rewrites += "Element '${candidate.tableElementId}': set props.domain=${candidate.domain}."
                } else if (!rawDomain.equals(normalizedDomain, ignoreCase = true)) {
                    props.addProperty("domain", normalizedDomain)
                    rewrites += "Element '${candidate.tableElementId}': normalized props.domain to $normalizedDomain."
                }
                val normalizedPresentation = props.get("preferredPresentation")
                    ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                    ?.asString
                    ?.trim()
                    ?.lowercase()
                if (normalizedPresentation !in setOf("cards", "table")) {
                    props.addProperty("preferredPresentation", candidate.preferredPresentation)
                    rewrites += "Element '${candidate.tableElementId}': set props.preferredPresentation=${candidate.preferredPresentation}."
                }
            }
    }

    private fun alignTableColumnsWithRows(
        elements: JsonObject,
        state: JsonObject,
        rewrites: MutableList<String>
    ) {
        elements.entrySet().forEach { (elementId, elementValue) ->
            val element = elementValue.takeIf { it.isJsonObject }?.asJsonObject ?: return@forEach
            val type = element.get("type")
                ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                ?.asString
                ?.trim()
                ?.lowercase()
                .orEmpty()
            if (type != "table") return@forEach

            val props = element.get("props")?.takeIf { it.isJsonObject }?.asJsonObject ?: return@forEach
            val rowKeys = firstTableRowKeys(props, state)
            if (rowKeys.size < 2) return@forEach

            val columns = props.get("columns")?.takeIf { it.isJsonArray }?.asJsonArray
            val columnObjects = columns
                ?.mapNotNull { it.takeIf { value -> value.isJsonObject }?.asJsonObject }
                .orEmpty()
            val declaredKeys = columnObjects.mapNotNull { column ->
                column.get("key")
                    ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                    ?.asString
                    ?.trim()
                    ?.takeIf { it.isNotBlank() }
            }
            val matchingKeys = declaredKeys.count { it in rowKeys }
            val shouldRewriteColumns = columns == null || columns.size() < 2 || matchingKeys == 0
            if (shouldRewriteColumns) {
                val rewrittenColumns = JsonArray()
                rowKeys.forEachIndexed { index, rowKey ->
                    val label = columnObjects
                        .getOrNull(index)
                        ?.get("label")
                        ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                        ?.asString
                        ?.trim()
                        ?.takeIf { it.isNotBlank() }
                        ?: humanizeTableColumnKey(rowKey)
                    rewrittenColumns.add(
                        JsonObject().apply {
                            addProperty("key", rowKey)
                            addProperty("label", label)
                        }
                    )
                }
                props.add("columns", rewrittenColumns)
                rewrites += "Element '$elementId': aligned table columns to row keys ${rowKeys.joinToString(prefix = "[", postfix = "]")}."
            }

            val primaryColumn = props.get("primaryColumn")
                ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                ?.asString
                ?.trim()
                .orEmpty()
            if (primaryColumn !in rowKeys) {
                props.addProperty("primaryColumn", rowKeys.first())
                rewrites += "Element '$elementId': set props.primaryColumn=${rowKeys.first()} to match table rows."
            }

            val existingHighlights = props.get("highlightColumns")
                ?.takeIf { it.isJsonArray }
                ?.asJsonArray
                ?.mapNotNull { value ->
                    value.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                        ?.asString
                        ?.trim()
                        ?.takeIf { it.isNotBlank() }
                }
                .orEmpty()
            val validHighlights = existingHighlights.filter { it in rowKeys && it != rowKeys.first() }
            if (existingHighlights.isNotEmpty() && validHighlights.size != existingHighlights.size) {
                props.add(
                    "highlightColumns",
                    JsonArray().apply {
                        val replacements = validHighlights.ifEmpty { rowKeys.drop(1).take(2) }
                        replacements.forEach(::add)
                    }
                )
                rewrites += "Element '$elementId': filtered props.highlightColumns to match table rows."
            } else if (existingHighlights.isEmpty() && props.has("preferredPresentation")) {
                props.add(
                    "highlightColumns",
                    JsonArray().apply { rowKeys.drop(1).take(2).forEach(::add) }
                )
            }
        }
    }

    private fun firstTableRowKeys(
        props: JsonObject,
        state: JsonObject
    ): List<String> {
        val firstObject = tableRows(props, state)
            .firstOrNull { it.isJsonObject }
            ?.asJsonObject
            ?: return emptyList()
        val keys = firstObject.entrySet()
            .map { it.key.trim() }
            .filter { it.isNotBlank() }
        if (keys.size <= 2) return keys
        val nonTechnicalKeys = keys.filterNot { it in setOf("id", "_id", "row_id", "key") }
        return nonTechnicalKeys.ifEmpty { keys }
    }

    private fun humanizeTableColumnKey(key: String): String {
        val spaced = key
            .replace(Regex("([a-z])([A-Z])"), "$1 $2")
            .replace(Regex("[_\\-]+"), " ")
            .trim()
        return spaced
            .split(Regex("\\s+"))
            .filter { it.isNotBlank() }
            .joinToString(" ") { token ->
                token.replaceFirstChar { char ->
                    if (char.isLowerCase()) char.titlecase(Locale.US) else char.toString()
                }
            }
            .ifBlank { key }
    }

    private fun pruneRedundantElementPayload(
        elements: JsonObject,
        rewrites: MutableList<String>
    ): Int {
        var removed = 0
        elements.entrySet().forEach { (elementId, elementValue) ->
            if (!elementValue.isJsonObject) return@forEach
            val element = elementValue.asJsonObject
            val props = element.get("props")?.takeIf { it.isJsonObject }?.asJsonObject ?: return@forEach
            removed += prunePropsObject(elementId, element, props, rewrites)
            removed += normalizeChildrenArray(elementId, element, rewrites)
            if (element.has("on")) {
                val on = element.get("on")
                if (on == null || on.isJsonNull || !on.isJsonObject || on.asJsonObject.size() == 0) {
                    element.remove("on")
                    removed += 1
                    rewrites += "Element '$elementId': removed empty on bindings."
                }
            }
            if (element.has("watch")) {
                val watch = element.get("watch")
                if (watch == null || watch.isJsonNull || !watch.isJsonObject || watch.asJsonObject.size() == 0) {
                    element.remove("watch")
                    removed += 1
                    rewrites += "Element '$elementId': removed empty watch bindings."
                }
            }
        }
        return removed
    }

    private fun prunePropsObject(
        elementId: String,
        element: JsonObject,
        props: JsonObject,
        rewrites: MutableList<String>
    ): Int {
        var removed = 0
        val keys = props.entrySet().map { it.key }
        keys.forEach { key ->
            val value = props.get(key)
            val shouldRemove = when {
                value == null || value.isJsonNull -> true
                value.isJsonPrimitive && value.asJsonPrimitive.isString && value.asString.trim().isEmpty() -> true
                value.isJsonObject && value.asJsonObject.size() == 0 -> true
                value.isJsonArray && value.asJsonArray.size() == 0 -> true
                else -> false
            }
            if (shouldRemove) {
                props.remove(key)
                removed += 1
            }
        }
        val normalizedType = element.get("type")
            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.trim()
            ?.lowercase()
            .orEmpty()
        if (normalizedType == "stack") {
            removed += removeDefaultProp(props, "gap", "md")
            removed += removeDefaultProp(props, "align", "start")
            removed += removeDefaultProp(props, "justify", "start")
            removed += removeDefaultProp(props, "wrap", "nowrap")
        }
        if (normalizedType == "text") {
            removed += removeDefaultProp(props, "variant", "body")
        }
        if (removed > 0) {
            rewrites += "Element '$elementId': pruned $removed non-essential prop field(s)."
        }
        return removed
    }

    private fun removeDefaultProp(
        props: JsonObject,
        key: String,
        defaultValue: String
    ): Int {
        val value = props.get(key) ?: return 0
        if (!value.isJsonPrimitive || !value.asJsonPrimitive.isString) return 0
        if (!value.asString.equals(defaultValue, ignoreCase = true)) return 0
        props.remove(key)
        return 1
    }

    private fun normalizeChildrenArray(
        elementId: String,
        element: JsonObject,
        rewrites: MutableList<String>
    ): Int {
        val children = element.get("children")?.takeIf { it.isJsonArray }?.asJsonArray ?: return 0
        val deduped = linkedSetOf<String>()
        var removed = 0
        children.forEach { child ->
            val childId = child.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString?.trim().orEmpty()
            if (childId.isBlank() || !deduped.add(childId)) {
                removed += 1
            }
        }
        if (removed > 0) {
            element.add(
                "children",
                JsonArray().apply {
                    deduped.forEach { add(it) }
                }
            )
            rewrites += "Element '$elementId': removed $removed duplicate/blank child reference(s)."
        }
        return removed
    }

    private fun pruneUnreachableElements(
        rootId: String,
        elements: JsonObject,
        rewrites: MutableList<String>
    ): Int {
        val reachable = linkedSetOf<String>()
        fun walk(elementId: String) {
            if (!elements.has(elementId) || !reachable.add(elementId)) return
            val element = elements.get(elementId)?.takeIf { it.isJsonObject }?.asJsonObject ?: return
            extractChildIds(element).forEach(::walk)
        }
        walk(rootId)
        if (reachable.isEmpty()) return 0
        val allIds = elements.entrySet().map { it.key }
        val toRemove = allIds.filter { it !in reachable }
        toRemove.forEach(elements::remove)
        if (toRemove.isNotEmpty()) {
            rewrites += "Removed ${toRemove.size} unreachable element(s) after canonicalization."
        }
        return toRemove.size
    }

    private fun canonicalizeElementAliases(
        elementId: String,
        element: JsonObject,
        rewrites: MutableList<String>
    ) {
        val props = element.getAsJsonObject("props")
        val children = element.getAsJsonArray("children")

        val rawType = element.get("type")
            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.trim()
            ?.lowercase(Locale.US)
        val compatibilityDirection = rawType
            ?.let(GeneratedRendererCapabilities.compatibilityTypeDirections::get)
        if (rawType != null && compatibilityDirection != null) {
            element.addProperty("type", "Stack")
            if (!props.has("direction")) {
                props.addProperty("direction", compatibilityDirection)
            }
            rewrites += "Element '$elementId': normalized compatibility type '$rawType' to Stack/$compatibilityDirection."
        }

        if (element.get("type")?.asString?.equals("stack", ignoreCase = true) == true) {
            val spacingTokens = mapOf(
                "none" to 0,
                "xs" to 2,
                "sm" to 4,
                "md" to 8,
                "lg" to 12,
                "xl" to 16
            )
            listOf(
                "padding",
                "paddingHorizontal",
                "paddingVertical",
                "margin",
                "marginHorizontal",
                "marginVertical"
            ).forEach { prop ->
                val token = props.get(prop)
                    ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                    ?.asString
                    ?.trim()
                    ?.lowercase(Locale.US)
                val numeric = spacingTokens[token] ?: return@forEach
                props.addProperty(prop, numeric)
                rewrites += "Element '$elementId': normalized props.$prop spacing token '$token' to $numeric."
            }
        }

        if (props.has("repeat")) {
            if (!element.has("repeat")) {
                val canonicalRepeat = canonicalizeRepeat(
                    value = props.get("repeat"),
                    elementId = elementId,
                    sourceLabel = "props.repeat",
                    rewrites = rewrites
                )
                if (canonicalRepeat != null) {
                    element.add("repeat", canonicalRepeat)
                    rewrites += "Element '$elementId': hoisted props.repeat to repeat."
                }
            } else {
                rewrites += "Element '$elementId': removed duplicate props.repeat (element.repeat kept)."
            }
            props.remove("repeat")
        }

        if (element.has("repeat")) {
            val canonicalRepeat = canonicalizeRepeat(
                value = element.get("repeat"),
                elementId = elementId,
                sourceLabel = "repeat",
                rewrites = rewrites
            )
            if (canonicalRepeat == null) {
                element.remove("repeat")
                rewrites += "Element '$elementId': dropped invalid repeat object."
            } else {
                element.add("repeat", canonicalRepeat)
            }
        }

        listOf("template", "itemTemplate", "child").forEach { aliasKey ->
            addTemplateChildAlias(
                elementId = elementId,
                aliasKey = "props.$aliasKey",
                aliasValue = props.get(aliasKey),
                children = children,
                rewrites = rewrites
            )
            props.remove(aliasKey)
        }
        listOf("template", "itemTemplate", "child").forEach { aliasKey ->
            addTemplateChildAlias(
                elementId = elementId,
                aliasKey = aliasKey,
                aliasValue = element.get(aliasKey),
                children = children,
                rewrites = rewrites
            )
            element.remove(aliasKey)
        }

        promoteStyleAliasesToTopLevel(
            elementId = elementId,
            props = props,
            rewrites = rewrites
        )
        rewritePropAlias(props, "distribution", "mainAxisAlignment", elementId, rewrites)
        rewritePropAlias(props, "verticalAlign", "verticalAlignment", elementId, rewrites)
        rewritePropAlias(props, "horizontalAlign", "horizontalAlignment", elementId, rewrites)
        rewritePropAlias(props, "justifyContent", "mainAxisAlignment", elementId, rewrites)
        rewritePropAlias(props, "alignItems", "crossAxisAlignment", elementId, rewrites)
    }

    private fun canonicalizeRepeat(
        value: JsonElement?,
        elementId: String,
        sourceLabel: String,
        rewrites: MutableList<String>
    ): JsonObject? {
        if (value == null || value.isJsonNull) return null
        if (!value.isJsonObject) {
            rewrites += "Element '$elementId': $sourceLabel is not an object and was removed."
            return null
        }
        val repeatObj = value.asJsonObject
        val rawStatePath = repeatObj.get("statePath")
            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.trim()
            ?.takeIf { it.isNotBlank() }
            ?: repeatObj.get("path")
                ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                ?.asString
                ?.trim()
                ?.takeIf { it.isNotBlank() }
                ?.also { rewrites += "Element '$elementId': normalized $sourceLabel.path to repeat.statePath." }
        if (rawStatePath.isNullOrBlank()) {
            rewrites += "Element '$elementId': $sourceLabel is missing statePath/path and was removed."
            return null
        }
        return JsonObject().apply {
            addProperty("statePath", normalizePointer(rawStatePath))
            repeatObj.get("key")
                ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                ?.asString
                ?.trim()
                ?.takeIf { it.isNotBlank() }
                ?.let { addProperty("key", it) }
        }
    }

    private fun addTemplateChildAlias(
        elementId: String,
        aliasKey: String,
        aliasValue: JsonElement?,
        children: JsonArray,
        rewrites: MutableList<String>
    ) {
        if (aliasValue == null || aliasValue.isJsonNull) return
        if (!aliasValue.isJsonPrimitive || !aliasValue.asJsonPrimitive.isString) {
            rewrites += "Element '$elementId': ignored non-string alias '$aliasKey'."
            return
        }
        val childId = aliasValue.asString.trim()
        if (childId.isBlank()) {
            rewrites += "Element '$elementId': ignored blank alias '$aliasKey'."
            return
        }
        if (children.any { it.isJsonPrimitive && it.asJsonPrimitive.isString && it.asString == childId }) {
            rewrites += "Element '$elementId': removed redundant alias '$aliasKey' (already in children)."
            return
        }
        children.add(childId)
        rewrites += "Element '$elementId': normalized alias '$aliasKey' into children."
    }

    private fun promoteStyleAliasesToTopLevel(
        elementId: String,
        props: JsonObject,
        rewrites: MutableList<String>
    ) {
        val style = props.get("style")?.takeIf { it.isJsonObject }?.asJsonObject ?: return
        style.entrySet().forEach { (styleKey, styleValue) ->
            val normalizedKey = normalizeLayoutAliasKey(styleKey) ?: return@forEach
            if (!props.has(normalizedKey)) {
                props.add(normalizedKey, styleValue.deepCopy())
                rewrites += "Element '$elementId': promoted props.style.$styleKey to props.$normalizedKey."
            }
        }
    }

    private fun rewritePropAlias(
        props: JsonObject,
        sourceKey: String,
        targetKey: String,
        elementId: String,
        rewrites: MutableList<String>
    ) {
        if (!props.has(sourceKey)) return
        if (!props.has(targetKey)) {
            props.add(targetKey, props.get(sourceKey).deepCopy())
            rewrites += "Element '$elementId': normalized props.$sourceKey to props.$targetKey."
        } else {
            rewrites += "Element '$elementId': ignored props.$sourceKey because props.$targetKey already exists."
        }
        props.remove(sourceKey)
    }

    private fun normalizeLayoutAliasKey(key: String): String? {
        return when (key.trim()) {
            "distribution" -> "mainAxisAlignment"
            "verticalAlign" -> "verticalAlignment"
            "horizontalAlign" -> "horizontalAlignment"
            "justifyContent" -> "mainAxisAlignment"
            "alignItems" -> "crossAxisAlignment"
            "spacing" -> "gap"
            "direction", "gap", "align", "justify", "wrap",
            "padding", "paddingHorizontal", "paddingVertical",
            "margin", "marginHorizontal", "marginVertical",
            "width", "height", "flex",
            "mainAxisAlignment", "crossAxisAlignment",
            "verticalAlignment", "horizontalAlignment" -> key
            else -> null
        }
    }

    private fun buildTableDiagnostics(
        spec: JsonObject,
        rewrites: List<String>,
        stats: CanonicalizationStats,
        compatibilityHeaderInference: Boolean
    ): TableDiagnostics {
        val elements = spec.getAsJsonObject("elements") ?: return TableDiagnostics(
            canonicalizationRewrites = rewrites,
            irByteSize = spec.toString().toByteArray(Charsets.UTF_8).size,
            compactionApplied = stats.compactionApplied,
            removedFieldCount = stats.removedFieldCount
        )
        val state = spec.getAsJsonObject("state") ?: JsonObject()
        val elementCount = elements.size()
        val byteSize = spec.toString().toByteArray(Charsets.UTF_8).size

        val bestCandidate = elements.entrySet()
            .mapNotNull { (elementId, _) ->
                detectTableCandidate(
                    tableElementId = elementId,
                    elements = elements,
                    state = state,
                    compatibilityHeaderInference = compatibilityHeaderInference
                )
            }
            .maxByOrNull { candidate ->
                (candidate.columns * candidate.rows.coerceAtLeast(1))
            }

        if (bestCandidate == null) {
            return TableDiagnostics(
                tableDetected = false,
                columns = 0,
                rows = 0,
                tableDomain = "none",
                preferredPresentation = "table",
                presentationChosen = "table",
                renderMode = "none",
                cardMappingStatus = "not_applicable",
                canonicalizationRewrites = rewrites,
                irElementCount = elementCount,
                irByteSize = byteSize,
                compactionApplied = stats.compactionApplied,
                removedFieldCount = stats.removedFieldCount
            )
        }
        val renderMode = when {
            bestCandidate.domain in cardFirstTableDomains -> "cards_first"
            bestCandidate.domain == "comparison" && bestCandidate.columns >= 3 -> "cards_first"
            bestCandidate.domain == "generic" && bestCandidate.columns >= 4 -> "horizontal_scroll_table"
            else -> "table"
        }
        val cardMappingStatus = when {
            bestCandidate.domain == "comparison" -> "pending_runtime_mapping"
            bestCandidate.domain !in cardFirstTableDomains -> "not_applicable"
            else -> "pending_runtime_mapping"
        }
        val mappingWarnings = mutableListOf<String>()
        if (bestCandidate.domain in cardFirstTableDomains &&
            bestCandidate.rows == 0
        ) {
            mappingWarnings += "Domain '${bestCandidate.domain}' requested cards but table rows are empty."
        }
        return TableDiagnostics(
            tableDetected = true,
            columns = bestCandidate.columns,
            rows = bestCandidate.rows,
            tableDomain = bestCandidate.domain,
            preferredPresentation = bestCandidate.preferredPresentation,
            presentationChosen = bestCandidate.preferredPresentation,
            renderMode = renderMode,
            cardMappingStatus = cardMappingStatus,
            mappingWarnings = mappingWarnings,
            canonicalizationRewrites = rewrites,
            irElementCount = elementCount,
            irByteSize = byteSize,
            compactionApplied = stats.compactionApplied,
            removedFieldCount = stats.removedFieldCount
        )
    }

    private fun detectTableCandidate(
        tableElementId: String,
        elements: JsonObject,
        state: JsonObject,
        compatibilityHeaderInference: Boolean
    ): TableCandidate? {
        val container = elements.get(tableElementId)?.takeIf { it.isJsonObject }?.asJsonObject ?: return null
        val type = container.get("type")
            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.trim()
            ?.lowercase()
            .orEmpty()
        val props = container.get("props")?.takeIf { it.isJsonObject }?.asJsonObject ?: JsonObject()

        if (type == "table") {
            val headerLabels = extractTableColumnLabelsFromProps(props)
            val columns = headerLabels.size
            if (columns < 2) return null
            val rows = resolveTableRowCountFromProps(props, state)
            val intent = inferTableIntent(props, headerLabels, compatibilityHeaderInference)
            return TableCandidate(
                tableElementId = tableElementId,
                columns = columns,
                rows = rows,
                domain = intent.domain,
                preferredPresentation = intent.preferredPresentation
            )
        }

        val childIds = container.get("children")
            ?.takeIf { it.isJsonArray }
            ?.asJsonArray
            ?.mapNotNull { child ->
                child.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
            }
            .orEmpty()
        if (childIds.size < 2) return null

        val headerEntry = childIds
            .mapNotNull { childId ->
                val element = elements.get(childId)?.takeIf { it.isJsonObject }?.asJsonObject ?: return@mapNotNull null
                childId to element
            }
            .firstOrNull { (_, element) ->
                isRowLikeElement(element) && !element.has("repeat") && extractChildIds(element).size >= 2
            } ?: return null

        val headerChildren = extractChildIds(headerEntry.second)
        if (headerChildren.size < 2) return null

        val repeatedBodyEntry = childIds
            .asSequence()
            .filter { it != headerEntry.first }
            .mapNotNull { childId ->
                val element = elements.get(childId)?.takeIf { it.isJsonObject }?.asJsonObject ?: return@mapNotNull null
                childId to element
            }
            .firstOrNull { (_, element) ->
                val repeat = element.get("repeat")
                val templateId = extractChildIds(element).firstOrNull()
                val template = templateId
                    ?.let { elements.get(it) }
                    ?.takeIf { it.isJsonObject }
                    ?.asJsonObject
                repeat != null && repeat.isJsonObject && template != null && isRowLikeElement(template)
            }

        val headerLabels = headerChildren.map { childId ->
            extractTextLabel(elements.get(childId)?.takeIf { it.isJsonObject }?.asJsonObject)
        }
        val intent = inferTableIntent(props, headerLabels, compatibilityHeaderInference)

        if (repeatedBodyEntry != null) {
            val bodyElement = repeatedBodyEntry.second
            val rowTemplateId = extractChildIds(bodyElement).firstOrNull()
            val rowTemplate = rowTemplateId
                ?.let { elements.get(it) }
                ?.takeIf { it.isJsonObject }
                ?.asJsonObject
            val repeat = bodyElement.getAsJsonObject("repeat")
            val repeatStatePath = repeat
                ?.get("statePath")
                ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                ?.asString
                ?.trim()
                .orEmpty()
            val rowCount = if (repeatStatePath.isBlank()) 0 else resolveStateArraySize(state, repeatStatePath)
            val templateColumns = rowTemplate?.let { extractChildIds(it).size } ?: 0
            val columns = maxOf(headerChildren.size, templateColumns)
            if (columns < 2) return null
            return TableCandidate(
                tableElementId = tableElementId,
                columns = columns,
                rows = rowCount,
                domain = intent.domain,
                preferredPresentation = intent.preferredPresentation
            )
        }

        val staticRows = childIds
            .filter { it != headerEntry.first }
            .mapNotNull { childId ->
                val element = elements.get(childId)?.takeIf { it.isJsonObject }?.asJsonObject ?: return@mapNotNull null
                if (isRowLikeElement(element)) childId to element else null
            }
        if (staticRows.isEmpty()) return null
        val staticColumns = staticRows.maxOf { (_, rowElement) -> extractChildIds(rowElement).size }
        val columns = maxOf(headerChildren.size, staticColumns)
        if (columns < 2) return null
        return TableCandidate(
            tableElementId = tableElementId,
            columns = columns,
            rows = staticRows.size,
            domain = intent.domain,
            preferredPresentation = intent.preferredPresentation
        )
    }

    private data class TableIntent(
        val domain: String,
        val preferredPresentation: String
    )

    private fun inferTableIntent(
        props: JsonObject,
        headerLabels: List<String>,
        compatibilityHeaderInference: Boolean
    ): TableIntent {
        val explicitDomain = normalizeManifestTableDomain(
            props.get("domain")
                ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                ?.asString
        )?.takeIf { it in supportedTableDomains }
        val inferredFromHeaders = if (compatibilityHeaderInference) {
            inferTableDomainFromHeaders(headerLabels)
        } else {
            "generic"
        }
        val inferredDomain = when {
            explicitDomain != null -> explicitDomain
            else -> inferredFromHeaders
        }
        val explicitPresentation = props.get("preferredPresentation")
            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.trim()
            ?.lowercase()
            ?.takeIf { it in setOf("cards", "table") }
        val preferredPresentation = when {
            explicitPresentation == null -> if (inferredDomain in cardFirstTableDomains) "cards" else "table"
            else -> explicitPresentation
        }
        return TableIntent(
            domain = inferredDomain,
            preferredPresentation = preferredPresentation
        )
    }

    private fun normalizeManifestTableDomain(raw: String?): String? {
        val token = raw?.trim()?.lowercase(Locale.US)?.takeIf { it.isNotBlank() } ?: return null
        return GeneratedRendererCapabilities.tableDomainAliases[token] ?: token
    }

    private fun extractTableColumnLabelsFromProps(props: JsonObject): List<String> {
        val columns = props.get("columns")?.takeIf { it.isJsonArray }?.asJsonArray ?: return emptyList()
        return columns.mapNotNull { entry ->
            when {
                entry.isJsonPrimitive && entry.asJsonPrimitive.isString -> {
                    entry.asString.trim().takeIf { it.isNotBlank() }
                }
                entry.isJsonObject -> {
                    val obj = entry.asJsonObject
                    val label = obj.get("label")
                        ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                        ?.asString
                        ?.trim()
                        .orEmpty()
                    if (label.isNotBlank()) {
                        label
                    } else {
                        obj.get("key")
                            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                            ?.asString
                            ?.trim()
                            ?.takeIf { it.isNotBlank() }
                    }
                }
                else -> null
            }
        }
    }

    private fun resolveTableRowCountFromProps(
        props: JsonObject,
        state: JsonObject
    ): Int {
        val rows = props.get("rows")
        if (rows != null && rows.isJsonArray) {
            return rows.asJsonArray.size()
        }
        val statePath = props.get("statePath")
            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.trim()
            .orEmpty()
        return if (statePath.isBlank()) 0 else resolveStateArraySize(state, statePath)
    }

    private fun extractChildIds(element: JsonObject): List<String> =
        FlatSpecReferenceSemantics.references(element).map { it.targetId }

    private fun isRowLikeElement(element: JsonObject): Boolean {
        val type = element.get("type")
            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.trim()
            ?.lowercase()
            .orEmpty()
        if (type == "row") return true
        if (type != "stack") return false
        val props = element.getAsJsonObject("props") ?: return false
        val direction = (
            props.get("direction")?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
                ?: props.get("orientation")?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
                ?: props.get("axis")?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
            )
            ?.trim()
            ?.lowercase()
            .orEmpty()
        return direction in setOf("horizontal", "row", "x")
    }

    private fun extractTextLabel(element: JsonObject?): String {
        if (element == null) return ""
        val type = element.get("type")
            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.trim()
            ?.lowercase()
            .orEmpty()
        if (type != "text") return ""
        val props = element.getAsJsonObject("props") ?: return ""
        val candidates = listOf("text", "label", "title", "value", "content")
        for (key in candidates) {
            val value = props.get(key)
            if (value != null && value.isJsonPrimitive && value.asJsonPrimitive.isString) {
                val text = value.asString.trim()
                if (text.isNotBlank()) return text
            }
        }
        return ""
    }

    private fun isWeatherHeaderLabel(label: String): Boolean {
        if (label.isBlank()) return false
        val token = label.lowercase()
        val keywords = listOf(
            "weather",
            "temp",
            "temperature",
            "condition",
            "high",
            "low",
            "max",
            "min",
            "humidity",
            "wind",
            "rain",
            "precip",
            "forecast",
            "uv",
            "feels"
        )
        return keywords.any { keyword -> token.contains(keyword) }
    }

    private fun isFlightHeaderLabel(label: String): Boolean {
        if (label.isBlank()) return false
        val token = label.lowercase()
        val keywords = listOf(
            "airline",
            "carrier",
            "flight",
            "depart",
            "departure",
            "arrival",
            "arrive",
            "duration",
            "fare",
            "price",
            "cost",
            "stops",
            "layover"
        )
        return keywords.any { keyword -> token.contains(keyword) }
    }

    private fun isStrongFlightHeaderLabel(label: String): Boolean {
        if (label.isBlank()) return false
        val token = label.lowercase()
        val keywords = listOf(
            "airline",
            "carrier",
            "flight",
            "depart",
            "departure",
            "arrival",
            "arrive",
            "takeoff",
            "landing",
            "origin",
            "destination"
        )
        return keywords.any { keyword -> token.contains(keyword) }
    }

    private fun isBookingEntityHeaderLabel(label: String): Boolean {
        if (label.isBlank()) return false
        val token = label.lowercase()
        val keywords = listOf(
            "hotel",
            "property",
            "provider",
            "option",
            "listing",
            "vendor",
            "airline",
            "plan",
            "package",
            "name"
        )
        return keywords.any { keyword -> token.contains(keyword) }
    }

    private fun isBookingValueHeaderLabel(label: String): Boolean {
        if (label.isBlank()) return false
        val token = label.lowercase()
        val keywords = listOf(
            "price",
            "cost",
            "fare",
            "rate",
            "night",
            "duration",
            "room",
            "amenity",
            "wifi",
            "rating",
            "review",
            "book",
            "reserve",
            "deal",
            "url",
            "link"
        )
        return keywords.any { keyword -> token.contains(keyword) }
    }

    private fun isScheduleHeaderLabel(label: String): Boolean {
        if (label.isBlank()) return false
        val token = label.lowercase()
        val timeKeywords = listOf("time", "date", "day", "slot", "start", "end")
        val eventKeywords = listOf("event", "activity", "agenda", "session", "task", "title", "stop", "location")
        return timeKeywords.any { token.contains(it) } || eventKeywords.any { token.contains(it) }
    }

    private fun isStatusHeaderLabel(label: String): Boolean {
        if (label.isBlank()) return false
        val token = label.lowercase()
        val keywords = listOf(
            "status",
            "state",
            "stage",
            "progress",
            "eta",
            "updated",
            "resolved",
            "tracking",
            "phase"
        )
        return keywords.any { keyword -> token.contains(keyword) }
    }

    private fun isComparisonFeatureHeader(label: String): Boolean {
        if (label.isBlank()) return false
        val token = label.lowercase()
        val keywords = listOf("feature", "metric", "criteria", "attribute", "spec", "dimension", "parameter")
        return keywords.any { keyword -> token.contains(keyword) }
    }

    private fun inferTableDomainFromHeaders(headerLabels: List<String>): String {
        val weatherSignals = headerLabels.count(::isWeatherHeaderLabel)
        val flightSignals = headerLabels.count(::isFlightHeaderLabel)
        val strongFlightSignals = headerLabels.count(::isStrongFlightHeaderLabel)
        val bookingEntitySignals = headerLabels.count(::isBookingEntityHeaderLabel)
        val bookingValueSignals = headerLabels.count(::isBookingValueHeaderLabel)
        val scheduleSignals = headerLabels.count(::isScheduleHeaderLabel)
        val statusSignals = headerLabels.count(::isStatusHeaderLabel)
        val featureLike = isComparisonFeatureHeader(headerLabels.firstOrNull().orEmpty())
        return when {
            weatherSignals >= 2 -> "weather"
            strongFlightSignals >= 1 && flightSignals >= 2 -> "flight"
            bookingEntitySignals >= 1 && bookingValueSignals >= 2 -> "booking"
            scheduleSignals >= 2 && statusSignals >= 1 -> "status"
            scheduleSignals >= 2 -> "schedule"
            featureLike && headerLabels.size >= 3 -> "comparison"
            else -> "generic"
        }
    }

    private fun resolveStateArraySize(state: JsonObject, path: String): Int {
        val current = resolveStateElement(state, path) ?: return 0
        return if (current.isJsonArray) current.asJsonArray.size() else 0
    }

    private fun resolveStateElement(state: JsonObject, path: String): JsonElement? {
        val tokens = pointerTokens(path)
        if (tokens.isEmpty()) return null
        var current: JsonElement = state
        tokens.forEach { token ->
            current = when {
                current.isJsonObject -> current.asJsonObject.get(token) ?: return null
                current.isJsonArray -> {
                    val index = token.toIntOrNull() ?: return null
                    val array = current.asJsonArray
                    if (index < 0 || index >= array.size()) return null
                    array[index]
                }
                else -> return null
            }
        }
        return current
    }

    private fun pointerTokens(path: String): List<String> {
        val trimmed = path.trim()
        if (trimmed.isBlank() || trimmed == "/") return emptyList()
        return trimmed
            .removePrefix("/")
            .split('/')
            .filter { it.isNotEmpty() }
            .map(::decodePointerToken)
    }

    private fun decodePointerToken(token: String): String {
        return token.replace("~1", "/").replace("~0", "~")
    }

    private fun normalizePointer(path: String): String {
        val trimmed = path.trim()
        if (trimmed.isBlank()) return ""
        return if (trimmed.startsWith('/')) trimmed else "/$trimmed"
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
