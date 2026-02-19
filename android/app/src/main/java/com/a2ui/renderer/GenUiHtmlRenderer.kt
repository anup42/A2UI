package com.a2ui.renderer

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.File
import java.util.Locale

object GenUiHtmlRenderer {
    data class RenderResult(
        val html: String,
        val warnings: List<String>
    )

    private data class SurfaceState(
        val surfaceId: String,
        val components: List<JsonObject>,
        val rootHint: String?
    )

    fun render(rawInput: String, sourceDir: File? = null): RenderResult {
        val warnings = mutableListOf<String>()
        val parsed = parseJsonOrJsonl(rawInput, warnings)
        val messages = extractMessages(parsed)
            ?: return RenderResult(
                html = renderErrorPage("No renderable GenUI payload found."),
                warnings = warnings + "Input did not contain genui_json/messages/payload."
            )

        val surfaces = extractSurfaces(messages, warnings)
        if (surfaces.isEmpty()) {
            return RenderResult(
                html = renderErrorPage("No renderable components found."),
                warnings = warnings + "Messages were parsed, but no component arrays were found."
            )
        }

        val body = surfaces.joinToString(separator = "\n") { surface ->
            renderSurface(surface, sourceDir, warnings)
        }
        return RenderResult(
            html = buildHtml(body),
            warnings = warnings
        )
    }

    fun renderErrorPage(message: String): String {
        val escaped = escapeHtml(message)
        return buildHtml("<section class=\"surface\"><p class=\"warning\">$escaped</p></section>")
    }

    private fun parseJsonOrJsonl(rawInput: String, warnings: MutableList<String>): JsonElement {
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

    private fun extractMessages(root: JsonElement): JsonArray? {
        if (root.isJsonArray) {
            return root.asJsonArray
        }
        if (!root.isJsonObject) {
            return null
        }

        val obj = root.asJsonObject
        val keys = listOf("genui_json", "messages", "payload", "a2ui_json")
        for (key in keys) {
            val arr = obj.getAsJsonArrayOrNull(key)
            if (arr != null) {
                return arr
            }
        }
        return null
    }

    private fun extractSurfaces(messages: JsonArray, warnings: MutableList<String>): List<SurfaceState> {
        val direct = messages.mapNotNull { it.asJsonObjectOrNull() }
        if (direct.isNotEmpty() && direct.all { it.hasString("id") && it.hasString("component") }) {
            return listOf(SurfaceState("@default", direct, rootHint = "root"))
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

            message.getAsJsonObjectOrNull("createSurface")?.let { createSurface ->
                val surfaceId = createSurface.getString("surfaceId")
                if (surfaceId != null) {
                    ensureSurface(surfaceId)
                }
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

        return order.mapNotNull { surfaceId ->
            val components = componentsBySurface[surfaceId]
            if (components.isNullOrEmpty()) {
                null
            } else {
                SurfaceState(surfaceId, components, rootBySurface[surfaceId])
            }
        }
    }

    private fun normalizeComponents(components: JsonArray, warnings: MutableList<String>): List<JsonObject> {
        val normalized = mutableListOf<JsonObject>()
        components.forEach { element ->
            val raw = element.asJsonObjectOrNull() ?: return@forEach
            if (raw.hasString("component")) {
                normalized += raw
                return@forEach
            }

            val converted = convertV08Component(raw, warnings)
            if (converted != null) {
                normalized += converted
            }
        }
        return normalized
    }

    private fun convertV08Component(raw: JsonObject, warnings: MutableList<String>): JsonObject? {
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
                payload?.get("children")?.let { children ->
                    out.add("children", normalizeChildrenValue(children))
                }
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
                val name = action?.getString("name")
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

    private fun normalizeChildrenValue(children: JsonElement): JsonElement {
        if (children.isJsonArray) {
            return children.asJsonArray
        }
        if (!children.isJsonObject) {
            return JsonArray()
        }
        val childrenObj = children.asJsonObject
        return childrenObj.getAsJsonArrayOrNull("explicitList") ?: JsonArray()
    }

    private fun renderSurface(
        surface: SurfaceState,
        sourceDir: File?,
        warnings: MutableList<String>
    ): String {
        val index = linkedMapOf<String, JsonObject>()
        surface.components.forEach { component ->
            val id = component.getString("id") ?: return@forEach
            index[id] = component
        }

        if (index.isEmpty()) {
            return "<section class=\"surface\"><p class=\"warning\">No components in ${escapeHtml(surface.surfaceId)}</p></section>"
        }

        val rootId = when {
            surface.rootHint != null && index.containsKey(surface.rootHint) -> surface.rootHint
            index.containsKey("root") -> "root"
            else -> index.keys.first()
        }

        val content = renderComponent(
            id = rootId,
            index = index,
            sourceDir = sourceDir,
            activePath = mutableSetOf(),
            warnings = warnings
        )
        return """
            <section class="surface">
                <h2 class="surface-title">${escapeHtml(surface.surfaceId)}</h2>
                $content
            </section>
        """.trimIndent()
    }

    private fun renderComponent(
        id: String,
        index: Map<String, JsonObject>,
        sourceDir: File?,
        activePath: MutableSet<String>,
        warnings: MutableList<String>
    ): String {
        val component = index[id] ?: return "<div class=\"warning\">Missing component: ${escapeHtml(id)}</div>"
        if (!activePath.add(id)) {
            return "<div class=\"warning\">Cycle detected at ${escapeHtml(id)}</div>"
        }

        val type = component.getString("component") ?: "Unknown"
        val rendered = when (type) {
            "Column" -> renderContainer(component, index, sourceDir, activePath, warnings, "column")
            "Row" -> renderContainer(component, index, sourceDir, activePath, warnings, "row")
            "List" -> {
                val direction = component.getString("direction")?.lowercase(Locale.US)
                val layout = if (direction == "horizontal") "row" else "column"
                renderContainer(component, index, sourceDir, activePath, warnings, layout)
            }

            "Card" -> renderCard(component, index, sourceDir, activePath, warnings)
            "Text" -> renderText(component)
            "Image" -> renderImage(component, sourceDir, className = "image")
            "Icon" -> renderImage(component, sourceDir, className = "icon")
            "Divider" -> "<hr class=\"divider\" />"
            "Button" -> renderButton(component, index, sourceDir)
            "Tabs" -> renderTabs(component, index, sourceDir, activePath, warnings)
            else -> {
                warnings += "Unsupported component type: $type"
                "<div class=\"warning\">Unsupported component: ${escapeHtml(type)}</div>"
            }
        }

        activePath.remove(id)
        return rendered
    }

    private fun renderContainer(
        component: JsonObject,
        index: Map<String, JsonObject>,
        sourceDir: File?,
        activePath: MutableSet<String>,
        warnings: MutableList<String>,
        layoutClass: String
    ): String {
        val children = readChildren(component)
        val inner = children.joinToString(separator = "\n") { childId ->
            val weight = index[childId]?.getAsNumberOrNull("weight")
            val childHtml = renderComponent(childId, index, sourceDir, activePath, warnings)
            if (layoutClass == "row" && weight != null && weight > 0.0) {
                """<div class="cell" style="flex:${formatNumber(weight)} 1 0%;">$childHtml</div>"""
            } else {
                childHtml
            }
        }

        return """<div class="$layoutClass">$inner</div>"""
    }

    private fun renderCard(
        component: JsonObject,
        index: Map<String, JsonObject>,
        sourceDir: File?,
        activePath: MutableSet<String>,
        warnings: MutableList<String>
    ): String {
        val childId = component.getString("child")
        val content = if (childId != null) {
            renderComponent(childId, index, sourceDir, activePath, warnings)
        } else {
            val children = readChildren(component)
            children.joinToString(separator = "\n") { child ->
                renderComponent(child, index, sourceDir, activePath, warnings)
            }
        }
        return """<div class="card">$content</div>"""
    }

    private fun renderText(component: JsonObject): String {
        val variant = (component.getString("variant") ?: "body").lowercase(Locale.US)
        val cls = when (variant) {
            "h1" -> "text h1"
            "h2" -> "text h2"
            "h3" -> "text h3"
            "h4" -> "text h4"
            "caption" -> "text caption"
            else -> "text body"
        }
        val text = escapeHtml(readDynamicString(component.get("text"))).replace("\n", "<br/>")
        return "<p class=\"$cls\">$text</p>"
    }

    private fun renderImage(component: JsonObject, sourceDir: File?, className: String): String {
        val raw = if (className == "icon") {
            readDynamicString(component.get("name"))
        } else {
            readDynamicString(component.get("url"))
        }
        if (raw.isBlank()) {
            return "<div class=\"warning\">Missing $className source</div>"
        }
        val resolved = resolveAssetUrl(raw, sourceDir)
        return """<img class="$className" src="${escapeAttr(resolved)}" alt="$className" />"""
    }

    private fun renderButton(
        component: JsonObject,
        index: Map<String, JsonObject>,
        sourceDir: File?
    ): String {
        val variant = (component.getString("variant") ?: "primary").lowercase(Locale.US)
        val childId = component.getString("child")
        val label = if (childId != null) {
            val child = index[childId]
            if (child != null && child.getString("component") == "Text") {
                readDynamicString(child.get("text")).ifBlank { "Open" }
            } else {
                "Open"
            }
        } else {
            "Open"
        }

        val rawUrl = extractOpenUrl(component)
        val buttonClass = if (variant == "borderless") "button borderless" else "button primary"
        val escapedLabel = escapeHtml(label)
        if (rawUrl.isNullOrBlank()) {
            return """<button class="$buttonClass" disabled>$escapedLabel</button>"""
        }
        val resolvedUrl = resolveAssetUrl(rawUrl, sourceDir)
        return """<a class="$buttonClass" href="${escapeAttr(resolvedUrl)}">$escapedLabel</a>"""
    }

    private fun renderTabs(
        component: JsonObject,
        index: Map<String, JsonObject>,
        sourceDir: File?,
        activePath: MutableSet<String>,
        warnings: MutableList<String>
    ): String {
        val tabs = component.getAsJsonArrayOrNull("tabs") ?: component.getAsJsonArrayOrNull("items")
        if (tabs == null || tabs.size() == 0) {
            return "<div class=\"tabs\"></div>"
        }

        val content = tabs.joinToString(separator = "\n") { tab ->
            val tabObj = tab.asJsonObjectOrNull() ?: return@joinToString ""
            val title = escapeHtml(readDynamicString(tabObj.get("title")).ifBlank { "Tab" })
            val childId = tabObj.getString("child")
            val childHtml = if (childId != null) {
                renderComponent(childId, index, sourceDir, activePath, warnings)
            } else {
                "<div class=\"warning\">Missing tab child</div>"
            }
            """<div class="tab"><h3 class="tab-title">$title</h3>$childHtml</div>"""
        }
        return """<div class="tabs">$content</div>"""
    }

    private fun readChildren(component: JsonObject): List<String> {
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
        val obj = element.asJsonObject
        val explicit = obj.getAsJsonArrayOrNull("explicitList")
        if (explicit != null) {
            return explicit.mapNotNull { it.asStringOrNull() }
        }
        return emptyList()
    }

    private fun extractOpenUrl(component: JsonObject): String? {
        val action = component.getAsJsonObjectOrNull("action") ?: return null
        val functionCall = action.getAsJsonObjectOrNull("functionCall") ?: return null
        val call = functionCall.getString("call") ?: return null
        if (!call.equals("openUrl", ignoreCase = true)) {
            return null
        }
        val args = functionCall.getAsJsonObjectOrNull("args") ?: return null
        return readDynamicString(args.get("url")).ifBlank { null }
    }

    private fun readDynamicString(element: JsonElement?): String {
        if (element == null || element.isJsonNull) {
            return ""
        }
        if (element.isJsonPrimitive) {
            val primitive = element.asJsonPrimitive
            return when {
                primitive.isString -> primitive.asString
                primitive.isBoolean -> primitive.asBoolean.toString()
                primitive.isNumber -> primitive.asNumber.toString()
                else -> primitive.toString()
            }
        }
        if (!element.isJsonObject) {
            return element.toString()
        }
        val obj = element.asJsonObject
        obj.getString("literalString")?.let { return it }
        obj.getString("path")?.let { return it }
        obj.get("literalNumber")?.let { return it.toString().trim('"') }
        obj.get("literalBoolean")?.let { return it.toString().trim('"') }
        return obj.toString()
    }

    private fun resolveAssetUrl(raw: String, sourceDir: File?): String {
        val normalized = raw.replace("\\", "/")
        if (sourceDir == null) {
            return normalized
        }

        val relative = when {
            normalized.startsWith("/assets/") -> normalized.removePrefix("/")
            normalized.startsWith("./assets/") -> normalized.removePrefix("./")
            normalized.startsWith("../assets/") -> normalized.removePrefix("../")
            normalized.startsWith("assets/") -> normalized
            else -> return normalized
        }

        return try {
            File(sourceDir, relative).toURI().toString()
        } catch (_: Exception) {
            normalized
        }
    }

    private fun buildHtml(body: String): String {
        return """
            <!doctype html>
            <html lang="en">
            <head>
              <meta charset="utf-8" />
              <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
              <title>A2UI Android Renderer</title>
              <style>
                :root {
                  --bg: #f7f8fa;
                  --card: #ffffff;
                  --text: #1d1f24;
                  --muted: #5f6673;
                  --line: #dde2ea;
                  --primary: #1f6feb;
                }
                * { box-sizing: border-box; }
                body {
                  margin: 0;
                  padding: 16px;
                  background: linear-gradient(180deg, #f9fafc 0%, #f3f5f9 100%);
                  color: var(--text);
                  font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                }
                .surface {
                  margin: 0 auto 20px;
                  max-width: 980px;
                  background: var(--card);
                  border: 1px solid var(--line);
                  border-radius: 16px;
                  padding: 16px;
                  box-shadow: 0 10px 30px rgba(20, 33, 61, 0.08);
                }
                .surface-title {
                  margin: 0 0 12px;
                  font-size: 12px;
                  letter-spacing: 0.08em;
                  text-transform: uppercase;
                  color: var(--muted);
                }
                .column { display: flex; flex-direction: column; gap: 10px; }
                .row { display: flex; flex-direction: row; gap: 10px; flex-wrap: wrap; align-items: center; }
                .cell { min-width: 0; }
                .text { margin: 0; line-height: 1.45; word-break: break-word; }
                .text.h1 { font-size: 32px; font-weight: 700; }
                .text.h2 { font-size: 26px; font-weight: 700; }
                .text.h3 { font-size: 20px; font-weight: 700; }
                .text.h4 { font-size: 16px; font-weight: 700; }
                .text.body { font-size: 15px; font-weight: 400; }
                .text.caption { font-size: 13px; color: var(--muted); }
                .divider { border: 0; border-top: 1px solid var(--line); margin: 6px 0; }
                .card {
                  border: 1px solid var(--line);
                  border-radius: 12px;
                  padding: 12px;
                  background: #fcfdff;
                }
                .image {
                  max-width: 100%;
                  height: auto;
                  border-radius: 10px;
                  border: 1px solid #ebeff6;
                }
                .icon { width: 26px; height: 26px; object-fit: contain; }
                .button {
                  display: inline-block;
                  padding: 10px 14px;
                  border-radius: 10px;
                  border: 1px solid transparent;
                  text-decoration: none;
                  font-size: 14px;
                  font-weight: 600;
                }
                .button.primary {
                  background: var(--primary);
                  color: #ffffff;
                }
                .button.borderless {
                  background: transparent;
                  border-color: var(--line);
                  color: var(--primary);
                }
                .button:disabled {
                  color: #8892a3;
                  border-color: #ccd3df;
                  background: #f0f3f8;
                }
                .warning {
                  margin: 0;
                  color: #b3261e;
                  font-size: 14px;
                  font-weight: 600;
                }
                .tabs { display: flex; flex-direction: column; gap: 12px; }
                .tab { border: 1px dashed #c8d4e8; border-radius: 10px; padding: 10px; }
                .tab-title { margin: 0 0 8px; font-size: 14px; color: #455a7d; }
              </style>
            </head>
            <body>
              $body
            </body>
            </html>
        """.trimIndent()
    }

    private fun escapeHtml(value: String): String {
        return value
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\"", "&quot;")
            .replace("'", "&#39;")
    }

    private fun escapeAttr(value: String): String = escapeHtml(value)

    private fun formatNumber(value: Double): String {
        val asLong = value.toLong()
        return if (asLong.toDouble() == value) asLong.toString() else value.toString()
    }

    private fun jsonPrimitive(value: String): JsonElement =
        JsonParser.parseString("\"${escapeJsonString(value)}\"")

    private fun escapeJsonString(value: String): String {
        return value
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
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

    private fun JsonObject.getAsNumberOrNull(key: String): Double? {
        val value = get(key) ?: return null
        if (!value.isJsonPrimitive || !value.asJsonPrimitive.isNumber) {
            return null
        }
        return value.asDouble
    }

    private fun JsonElement.asBooleanOrNull(): Boolean? {
        if (!isJsonPrimitive || !asJsonPrimitive.isBoolean) {
            return null
        }
        return asBoolean
    }

    private fun JsonObject.hasString(key: String): Boolean = getString(key) != null
}
