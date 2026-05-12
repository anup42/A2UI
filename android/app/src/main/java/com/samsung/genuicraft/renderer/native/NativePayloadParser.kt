package com.samsung.genuicraft.renderer.native

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.GenUiNativeRenderer
import com.samsung.genuicraft.security.SafeContentPolicy
import java.io.File
import java.util.Locale

internal object NativePayloadParser {
    private val URL_REGEX = Regex(
        """(?i)(?:https?://|//)[^\s<>\]]+|(?<![@\w])(?:www\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:[/?#][^\s<>\]]*)?"""
    )

    fun parseJsonOrJsonl(rawInput: String, warnings: MutableList<String>): JsonElement {
        val trimmed = rawInput.trim()
        require(trimmed.isNotEmpty()) { "Input is empty." }

        if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
            runCatching { JsonParser.parseString(trimmed) }.onSuccess { return it }
        }

        val firstLine = rawInput.lineSequence().firstOrNull { it.trim().isNotEmpty() }
            ?: error("Input is empty.")
        warnings += "Detected JSONL input. Rendering only the first non-empty line."
        return JsonParser.parseString(firstLine)
    }

    fun extractMessages(root: JsonElement): JsonArray? {
        if (root.isJsonArray) {
            return root.asJsonArray
        }
        if (!root.isJsonObject) {
            return null
        }

        val obj = root.asJsonObject
        if (looksLikeMessageObject(obj)) {
            return JsonArray().apply { add(obj.deepCopy()) }
        }

        val keys = listOf("genui_json", "messages", "payload", "a2ui_json")
        for (key in keys) {
            val value = obj.get(key) ?: continue
            if (value.isJsonArray) {
                return value.asJsonArray
            }
            if (value.isJsonObject) {
                val nested = value.asJsonObject
                if (looksLikeMessageObject(nested)) {
                    return JsonArray().apply { add(nested.deepCopy()) }
                }
            }
        }
        return null
    }

    fun extractSurfaces(
        messages: JsonArray,
        warnings: MutableList<String>
    ): List<GenUiNativeRenderer.SurfaceState> {
        val direct = messages.mapNotNull { it.asJsonObjectOrNull() }
        if (direct.isNotEmpty() && direct.all { it.hasString("id") && it.hasString("component") }) {
            val normalized = normalizeComponents(messages, warnings)
            return buildSurfaceStates("@default", normalized, rootHint = "root")
        }

        val order = mutableListOf<String>()
        val componentsBySurface = linkedMapOf<String, MutableList<JsonObject>>()
        val rootBySurface = mutableMapOf<String, String>()

        fun ensureSurface(surfaceId: String) {
            if (!componentsBySurface.containsKey(surfaceId)) {
                componentsBySurface[surfaceId] = mutableListOf()
                order += surfaceId
            }
        }

        messages.forEach { element ->
            val message = element.asJsonObjectOrNull() ?: return@forEach

            if (message.getAsJsonObjectOrNull("updateDataModel") != null) {
                warnings += "Ignored updateDataModel message (render-only pipeline)."
            }
            if (message.getAsJsonObjectOrNull("deleteSurface") != null) {
                warnings += "Ignored deleteSurface message (render-only pipeline)."
            }

            message.getAsJsonObjectOrNull("createSurface")?.let { create ->
                create.getString("surfaceId")?.let { ensureSurface(it) }
            }

            message.getAsJsonObjectOrNull("beginRendering")?.let { begin ->
                val surfaceId = begin.getString("surfaceId") ?: "@default"
                ensureSurface(surfaceId)
                begin.getString("root")?.let { rootBySurface[surfaceId] = it }
            }

            message.getAsJsonObjectOrNull("updateComponents")?.let { update ->
                val surfaceId = update.getString("surfaceId") ?: "@default"
                val componentArray = update.getAsJsonArrayOrNull("components") ?: JsonArray()
                ensureSurface(surfaceId)
                componentsBySurface[surfaceId] = normalizeComponents(componentArray, warnings).toMutableList()
            }

            message.getAsJsonObjectOrNull("surfaceUpdate")?.let { update ->
                val surfaceId = update.getString("surfaceId") ?: "@default"
                val componentArray = update.getAsJsonArrayOrNull("components") ?: JsonArray()
                ensureSurface(surfaceId)
                componentsBySurface[surfaceId] = normalizeComponents(componentArray, warnings).toMutableList()
            }
        }

        return order.flatMap { surfaceId ->
            val components = componentsBySurface[surfaceId] ?: return@flatMap emptyList()
            buildSurfaceStates(surfaceId, components, rootBySurface[surfaceId])
        }
    }

    fun readChildren(component: JsonObject): List<String> {
        val element = component.get("children") ?: return emptyList()
        if (element.isJsonPrimitive && element.asJsonPrimitive.isString) {
            return listOf(element.asString)
        }
        if (element.isJsonArray) {
            return element.asJsonArray.mapNotNull { it.asStringOrNull() }
        }
        if (!element.isJsonObject) {
            return emptyList()
        }

        val explicit = element.asJsonObject.getAsJsonArrayOrNull("explicitList") ?: return emptyList()
        return explicit.mapNotNull { it.asStringOrNull() }
    }

    fun readDynamicString(element: JsonElement?, normalizer: (String) -> String): String {
        if (element == null || element.isJsonNull) {
            return ""
        }
        if (element.isJsonPrimitive) {
            val primitive = element.asJsonPrimitive
            return when {
                primitive.isString -> normalizer(primitive.asString)
                primitive.isBoolean -> primitive.asBoolean.toString()
                primitive.isNumber -> primitive.asNumber.toString()
                else -> primitive.toString()
            }
        }
        if (!element.isJsonObject) {
            return normalizer(element.toString())
        }

        val obj = element.asJsonObject
        obj.getString("literalString")?.let { return normalizer(it) }
        obj.getString("path")?.let { return "" }
        obj.get("literalNumber")?.let { return normalizer(it.toString().trim('"')) }
        obj.get("literalBoolean")?.let { return normalizer(it.toString().trim('"')) }
        return normalizer(obj.toString())
    }

    fun resolveAssetUrl(raw: String, sourceDir: File?): String {
        val normalized = canonicalizeNetworkUrlToken(raw.replace("\\", "/").trim())
        if (sourceDir == null) {
            return when {
                normalized.startsWith("../assets/") -> "/" + normalized.removePrefix("../")
                normalized.startsWith("./assets/") -> "/" + normalized.removePrefix("./")
                normalized.startsWith("assets/") -> "/$normalized"
                else -> normalized
            }
        }

        val relative = when {
            normalized.startsWith("/assets/") -> normalized.removePrefix("/")
            normalized.startsWith("./assets/") -> normalized.removePrefix("./")
            normalized.startsWith("../assets/") -> normalized
            normalized.startsWith("assets/") -> normalized
            else -> return normalized
        }

        val candidateVariants = buildList {
            add(relative)
            if (normalized.startsWith("../assets/")) {
                add(normalized.removePrefix("../"))
            }
        }.distinct()
        for (candidateVariant in candidateVariants) {
            val localFile = runCatching {
                File(sourceDir, candidateVariant).toPath().normalize().toFile()
            }.getOrNull() ?: continue
            if (localFile.exists()) {
                return runCatching { localFile.toURI().toString() }.getOrElse { normalized }
            }
        }

        return when {
            normalized.startsWith("../assets/") -> "/" + normalized.removePrefix("../")
            normalized.startsWith("./assets/") -> "/" + normalized.removePrefix("./")
            normalized.startsWith("assets/") -> "/$normalized"
            normalized.startsWith("/assets/") -> normalized
            else -> normalized
        }
    }

    fun canonicalizeNetworkUrlToken(value: String): String {
        val trimmed = value.trim()
        if (trimmed.isBlank()) {
            return trimmed
        }
        return normalizeHttpUrlCandidate(trimmed) ?: trimmed
    }

    fun normalizeHttpUrlCandidate(value: String): String? {
        val trimmed = value.trim().trim('"', '\'')
        if (trimmed.isBlank()) {
            return null
        }
        if (
            trimmed.startsWith("/") ||
            trimmed.startsWith("assets/", ignoreCase = true) ||
            trimmed.startsWith("../assets/", ignoreCase = true) ||
            trimmed.startsWith("./assets/", ignoreCase = true) ||
            trimmed.startsWith("file:", ignoreCase = true) ||
            trimmed.startsWith("content:", ignoreCase = true)
        ) {
            return null
        }

        val pathCandidate = trimmed.trimEnd('.', ',', ';', ')', ']', '}')
        if (!URL_REGEX.matches(pathCandidate) && !pathCandidate.startsWith("https://", ignoreCase = true) && !pathCandidate.startsWith("//")) {
            return null
        }
        return SafeContentPolicy.sanitizeActionUrl(pathCandidate)
    }

    fun isLikelyPublicDomainHost(host: String): Boolean =
        SafeContentPolicy.isLikelyPublicDomainHost(host)

    fun jsonPrimitive(value: String): JsonElement =
        JsonParser.parseString("\"${escapeJsonString(value)}\"")

    fun escapeJsonString(value: String): String {
        return value
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
    }

    fun buildSurfaceStates(
        surfaceId: String,
        components: List<JsonObject>,
        rootHint: String?
    ): List<GenUiNativeRenderer.SurfaceState> {
        if (components.isEmpty()) {
            return emptyList()
        }

        val index = linkedMapOf<String, JsonObject>()
        components.forEach { component ->
            component.getString("id")?.let { id -> index[id] = component }
        }
        if (index.isEmpty()) {
            return emptyList()
        }

        val rootId = when {
            rootHint != null && index.containsKey(rootHint) -> rootHint
            index.containsKey("root") -> "root"
            else -> index.keys.first()
        }

        return listOf(GenUiNativeRenderer.SurfaceState(surfaceId, rootId, index))
    }

    fun normalizeComponents(components: JsonArray, warnings: MutableList<String>): List<JsonObject> {
        val normalizedById = linkedMapOf<String, JsonObject>()
        var generatedIdCounter = 0

        fun nextGeneratedId(prefix: String): String {
            generatedIdCounter += 1
            return "${prefix}_${generatedIdCounter}"
        }

        fun addNode(node: JsonObject) {
            val id = node.getString("id") ?: return
            if (!normalizedById.containsKey(id)) {
                normalizedById[id] = node
            }
        }

        fun canonicalComponentType(rawType: String?): String? {
            val token = rawType?.trim()?.lowercase(Locale.US) ?: return null
            return when (token) {
                "column" -> "Column"
                "row" -> "Row"
                "list" -> "List"
                "card" -> "Card"
                "text" -> "Text"
                "image" -> "Image"
                "icon" -> "Icon"
                "video" -> "Video"
                "audioplayer", "audio", "audio-player" -> "AudioPlayer"
                "table" -> "Table"
                "button" -> "Button"
                "tabs", "tab", "tabgroup" -> "Tabs"
                "modal", "dialog" -> "Modal"
                "divider" -> "Divider"
                "textfield", "input", "text-field" -> "TextField"
                "checkbox", "check-box" -> "CheckBox"
                "choicepicker", "choice-picker", "select", "picker" -> "ChoicePicker"
                "slider", "range" -> "Slider"
                "datetimeinput", "date-time-input", "datetime", "dateinput", "timeinput" -> "DateTimeInput"
                else -> rawType?.trim()
            }
        }

        fun normalizeSingleComponent(raw: JsonObject): JsonObject? {
            val out = if (raw.hasString("component")) {
                raw.deepCopy().asJsonObject
            } else {
                convertV08Component(raw, warnings)?.deepCopy()?.asJsonObject ?: return null
            }

            val canonicalType = canonicalComponentType(out.getString("component"))
            if (!canonicalType.isNullOrBlank()) {
                out.addProperty("component", canonicalType)
            }

            if (!out.hasString("id")) {
                val prefix = canonicalType?.lowercase(Locale.US)?.ifBlank { "component" } ?: "component"
                out.addProperty("id", nextGeneratedId(prefix))
            }

            fun normalizeChildElement(child: JsonElement): String? {
                if (child.isJsonPrimitive && child.asJsonPrimitive.isString) {
                    return child.asString.trim().takeIf { it.isNotBlank() }
                }
                if (!child.isJsonObject) {
                    return null
                }
                val normalizedChild = normalizeSingleComponent(child.asJsonObject) ?: return null
                val childId = normalizedChild.getString("id") ?: return null
                addNode(normalizedChild)
                return childId
            }

            out.get("child")?.let { child ->
                if (child.isJsonObject) {
                    val childId = normalizeChildElement(child)
                    if (!childId.isNullOrBlank()) {
                        out.addProperty("child", childId)
                    } else {
                        out.remove("child")
                    }
                }
            }

            out.get("children")?.let { children ->
                val childIds = JsonArray()
                when {
                    children.isJsonPrimitive && children.asJsonPrimitive.isString -> {
                        normalizeChildElement(children)?.let { childIds.add(it) }
                    }

                    children.isJsonArray -> {
                        children.asJsonArray.forEach { child ->
                            normalizeChildElement(child)?.let { childIds.add(it) }
                        }
                    }

                    children.isJsonObject -> {
                        val explicit = children.asJsonObject.getAsJsonArrayOrNull("explicitList")
                        explicit?.forEach { child ->
                            normalizeChildElement(child)?.let { childIds.add(it) }
                        }
                    }
                }
                out.add("children", childIds)
            }

            return out
        }

        components.forEach { element ->
            val raw = element.asJsonObjectOrNull() ?: return@forEach
            val normalized = normalizeSingleComponent(raw) ?: return@forEach
            addNode(normalized)
        }

        return normalizedById.values.toList()
    }

    fun convertV08Component(raw: JsonObject, warnings: MutableList<String>): JsonObject? {
        val id = raw.getString("id") ?: return null
        val typed = raw.getAsJsonObjectOrNull("component") ?: return null
        val entry = typed.entrySet().firstOrNull() ?: return null

        val type = entry.key
        val payload = entry.value.asJsonObjectOrNull()
        val out = JsonObject()
        out.addProperty("id", id)
        out.addProperty("component", type)

        when (type) {
            "Text" -> {
                out.add("text", payload?.get("text") ?: jsonPrimitive(""))
                payload?.get("usageHint")?.let { out.add("variant", it) }
                payload?.get("weight")?.let { out.add("weight", it) }
            }

            "Image" -> {
                out.add("url", payload?.get("url") ?: jsonPrimitive(""))
                payload?.get("usageHint")?.let { out.add("variant", it) }
                payload?.get("fit")?.let { out.add("fit", it) }
            }

            "Icon" -> {
                out.add("name", payload?.get("name") ?: jsonPrimitive(""))
            }

            "Column", "Row", "List" -> {
                payload?.get("children")?.let { out.add("children", normalizeChildrenValue(it)) }
                payload?.get("alignment")?.let { out.add("align", it) }
                payload?.get("distribution")?.let { out.add("justify", it) }
                payload?.get("direction")?.let { out.add("direction", it) }
            }

            "Button" -> {
                payload?.get("child")?.let { out.add("child", it) }
                if (payload?.get("primary")?.asBooleanOrNull() == true) {
                    out.addProperty("variant", "primary")
                } else {
                    out.addProperty("variant", "borderless")
                }

                val action = payload?.getAsJsonObjectOrNull("action")
                if (action != null) {
                    val hasModernAction = action.getAsJsonObjectOrNull("functionCall") != null ||
                        action.get("event") != null
                    if (hasModernAction) {
                        out.add("action", action.deepCopy())
                    } else {
                        val name = action.getString("name")
                        if (!name.isNullOrBlank()) {
                            val fn = JsonObject()
                            fn.addProperty("call", name)
                            val args = JsonObject()
                            action.getAsJsonArrayOrNull("context")?.forEach { ctx ->
                                val ctxObj = ctx.asJsonObjectOrNull() ?: return@forEach
                                val key = ctxObj.getString("key") ?: return@forEach
                                ctxObj.get("value")?.let { args.add(key, it) }
                            }
                            fn.add("args", args)
                            val actionOut = JsonObject()
                            actionOut.add("functionCall", fn)
                            out.add("action", actionOut)
                        }
                    }
                }
            }

            "Divider" -> {
                payload?.get("axis")?.let { out.add("axis", it) }
            }

            "Card" -> {
                payload?.get("child")?.let { out.add("child", it) }
            }

            else -> {
                warnings += "Unsupported v0.8 component type during normalization: $type"
                return null
            }
        }

        return out
    }

    fun normalizeChildrenValue(children: JsonElement): JsonElement {
        if (children.isJsonArray) {
            return children.asJsonArray
        }
        if (!children.isJsonObject) {
            return JsonArray()
        }
        val obj = children.asJsonObject
        return obj.getAsJsonArrayOrNull("explicitList") ?: JsonArray()
    }

    private fun JsonElement.asJsonObjectOrNull(): JsonObject? =
        if (isJsonObject) asJsonObject else null

    private fun JsonElement.asStringOrNull(): String? =
        if (isJsonPrimitive && asJsonPrimitive.isString) asString else null

    private fun JsonObject.getAsJsonArrayOrNull(key: String): JsonArray? {
        val value = get(key) ?: return null
        return if (value.isJsonArray) value.asJsonArray else null
    }

    private fun JsonObject.getAsJsonObjectOrNull(key: String): JsonObject? {
        val value = get(key) ?: return null
        return if (value.isJsonObject) value.asJsonObject else null
    }

    private fun JsonObject.getString(key: String): String? {
        val value = get(key) ?: return null
        return if (value.isJsonPrimitive && value.asJsonPrimitive.isString) value.asString else null
    }

    private fun JsonElement.asBooleanOrNull(): Boolean? {
        if (!isJsonPrimitive || !asJsonPrimitive.isBoolean) {
            return null
        }
        return asBoolean
    }

    private fun JsonObject.hasString(key: String): Boolean = getString(key) != null

    private fun looksLikeMessageObject(obj: JsonObject): Boolean {
        return obj.has("createSurface") ||
            obj.has("beginRendering") ||
            obj.has("updateComponents") ||
            obj.has("surfaceUpdate") ||
            obj.has("updateDataModel") ||
            obj.has("deleteSurface")
    }
}
