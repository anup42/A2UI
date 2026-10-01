package com.samsung.genuicraft

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.renderer.native.NativePayloadParser
import com.samsung.genuicraft.renderer.native.NativeTextFormatter
import java.io.File
import java.nio.charset.Charset
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

    private data class ParsedButton(
        val label: String,
        val url: String
    )

    private data class ParsedLogo(
        val label: String,
        val url: String
    )

    private data class ParsedMediaEntry(
        val label: String,
        val url: String,
        val iconLike: Boolean
    )

    private data class ChoiceOption(
        val label: String,
        val value: String
    )

    private enum class ComponentActionKind {
        OpenUrl,
        ShowMessage,
        ShowSurface
    }

    private data class ComponentAction(
        val kind: ComponentActionKind,
        val value: String
    )

    private data class BookingOption(
        val title: String,
        val details: String,
        val button: ParsedButton?,
        val logo: ParsedLogo? = null
    )

    private data class LeadingLabelValue(
        val prefix: String,
        val label: String,
        val delimiter: String,
        val value: String
    )

    private val OPTION_LINE_REGEX =
        Regex("""^Option\s+\d+\s*:\s*(.+?)\s*\|\s*(.+)$""", RegexOption.IGNORE_CASE)

    private val BULLET_PREFIX_REGEX = Regex("""^\s*[\-\*\u2022]\s+""")
    private val BUTTON_LINE_REGEX =
        Regex("""^(?:Action:\s*)?\[Button:\s*(.+?)\]\s*(.+)$""", RegexOption.IGNORE_CASE)
    private val BUTTON_MARKDOWN_LINK_REGEX =
        Regex("""^(?:Action:\s*)?\[Button:\s*(.+?)\]\s*\(\s*(.+?)\s*\)\s*$""", RegexOption.IGNORE_CASE)

    private val URL_REGEX = Regex("""https?://[^\s<>()]+""", RegexOption.IGNORE_CASE)
    private val TABLE_PLACEHOLDER_CELL_REGEX = Regex("""^[:\-\u2013\u2014]+$""")
    private val BOLD_REGEX = Regex("""\*\*(.+?)\*\*""")
    private val BULLET_LINE_REGEX = Regex("""^\s*([\-*\u2022])\s+(.+)$""")
    private val LEADING_LABEL_REGEX =
        Regex("""^\s*([•\-*]\s*)?([^:;：；\n]{1,70}?)([:;：；])\s*(.+)$""")
    private val INLINE_LABEL_REGEX =
        Regex("""(?:^|(?<=[.!?)]\s)|(?<=[\u2022\-]\s))([\p{L}\p{N}][\p{L}\p{N}/&()' -]{0,84}?)([:;\uFF1A\uFF1B])""")

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

            if (message.getAsJsonObjectOrNull("updateDataModel") != null) {
                warnings += "Ignored updateDataModel message (render-only pipeline)."
            }
            if (message.getAsJsonObjectOrNull("deleteSurface") != null) {
                warnings += "Ignored deleteSurface message (render-only pipeline)."
            }

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
            "Text" -> renderText(component, sourceDir)
            "Image" -> renderImage(component, sourceDir, className = "image")
            "Icon" -> renderImage(component, sourceDir, className = "icon")
            "Video" -> renderVideo(component, sourceDir)
            "AudioPlayer" -> renderAudioPlayer(component, sourceDir)
            "Divider" -> "<hr class=\"divider\" />"
            "Button" -> renderButton(component, index, sourceDir)
            "Tabs" -> renderTabs(component, index, sourceDir, activePath, warnings)
            "Modal" -> renderModal(component, index, sourceDir, activePath, warnings)
            "TextField" -> renderTextField(component)
            "CheckBox" -> renderCheckBox(component)
            "ChoicePicker" -> renderChoicePicker(component)
            "Slider" -> renderSlider(component)
            "DateTimeInput" -> renderDateTimeInput(component)
            else -> {
                warnings += "Unsupported component type: $type"
                "<div class=\"warning\">Unsupported component: ${escapeHtml(type)}</div>"
            }
        }

        val withAction = applyComponentActionWrapper(
            component = component,
            renderedHtml = rendered,
            sourceDir = sourceDir,
            warnings = warnings
        )
        activePath.remove(id)
        return withAction
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
        maybeRenderRowsAsTable(children, index, sourceDir)?.let { return it }
        maybeRenderTextList(component, children, index, sourceDir)?.let { return it }

        val inner = children.joinToString(separator = "\n") { childId ->
            val weight = index[childId]?.getAsNumberOrNull("weight")
            val childHtml = renderComponent(childId, index, sourceDir, activePath, warnings)
            if (layoutClass == "row" && weight != null && weight > 0.0) {
                """<div class="cell" style="flex:${formatNumber(weight)} 1 0%;">$childHtml</div>"""
            } else {
                childHtml
            }
        }
        val style = buildFlexStyle(component, layoutClass)
        return """<div class="$layoutClass"$style>$inner</div>"""
    }

    private fun maybeRenderRowsAsTable(
        children: List<String>,
        index: Map<String, JsonObject>,
        sourceDir: File?
    ): String? {
        if (children.size < 2) {
            return null
        }

        val rowComponents = mutableListOf<JsonObject>()
        for (childId in children) {
            val child = index[childId] ?: return null
            if (child.getString("component") != "Row") {
                return null
            }
            rowComponents += child
        }

        val rowCells = rowComponents.map { readChildren(it) }
        if (rowCells.any { it.isEmpty() }) {
            return null
        }

        val columnCount = rowCells.first().size
        if (columnCount < 2 || rowCells.any { it.size != columnCount }) {
            return null
        }

        val textRows = mutableListOf<List<JsonObject>>()
        var hasWeight = false
        for (cellIds in rowCells) {
            val cells = mutableListOf<JsonObject>()
            for (cellId in cellIds) {
                val cell = index[cellId] ?: return null
                if (cell.getString("component") != "Text") {
                    return null
                }
                if ((cell.getAsNumberOrNull("weight") ?: 0.0) > 0.0) {
                    hasWeight = true
                }
                cells += cell
            }
            textRows += cells
        }

        val headerLike = textRows.first().all {
            val variant = (it.getString("variant") ?: "").lowercase(Locale.US)
            variant in setOf("h1", "h2", "h3", "h4")
        }
        if (!hasWeight && !headerLike) {
            return null
        }

        val headerCells = if (headerLike) textRows.first() else emptyList()
        val bodyRows = (if (headerLike) textRows.drop(1) else textRows)
            .filterNot { row -> isPlaceholderTableStringRow(row.map { readDynamicString(it.get("text")) }) }
        if (bodyRows.isEmpty()) {
            return null
        }

        val thead = if (headerCells.isNotEmpty()) {
            val headers = headerCells.joinToString(separator = "") { cell ->
                val label = formatInlineText(readDynamicString(cell.get("text")), sourceDir)
                "<th>$label</th>"
            }
            "<thead><tr>$headers</tr></thead>"
        } else {
            ""
        }

        val tbody = bodyRows.joinToString(separator = "") { row ->
            val cells = row.joinToString(separator = "") { cell ->
                val value = formatInlineText(
                    sanitizeTableCellDisplayValue(readDynamicString(cell.get("text"))),
                    sourceDir
                )
                "<td>$value</td>"
            }
            "<tr>$cells</tr>"
        }

        return """
            <div class="table-wrap">
              <table class="data-table component-table">
                $thead
                <tbody>$tbody</tbody>
              </table>
            </div>
        """.trimIndent()
    }

    private fun maybeRenderTextList(
        component: JsonObject,
        children: List<String>,
        index: Map<String, JsonObject>,
        sourceDir: File?
    ): String? {
        if (component.getString("component") != "List") {
            return null
        }
        val direction = component.getString("direction")?.lowercase(Locale.US)
        if (direction == "horizontal" || children.isEmpty()) {
            return null
        }

        val textItems = mutableListOf<JsonObject>()
        for (childId in children) {
            val child = index[childId] ?: return null
            if (child.getString("component") != "Text") {
                return null
            }
            textItems += child
        }
        if (textItems.size < 2) {
            return null
        }

        val itemsHtml = textItems.joinToString(separator = "") { item ->
            var value = readDynamicString(item.get("text")).trim()
            value = stripLeadingBulletMarker(value)
            "<li>${formatInlineText(value, sourceDir)}</li>"
        }
        return """<ul class="component-list">$itemsHtml</ul>"""
    }

    private fun buildFlexStyle(component: JsonObject, layoutClass: String): String {
        val styles = mutableListOf<String>()
        component.getString("align")?.lowercase(Locale.US)?.let { align ->
            mapAlignItems(align)?.let { styles += "align-items:$it" }
        }
        component.getString("justify")?.lowercase(Locale.US)?.let { justify ->
            mapJustifyContent(justify)?.let { styles += "justify-content:$it" }
        }
        if (layoutClass == "row" && styles.isEmpty()) {
            return ""
        }
        return if (styles.isEmpty()) "" else """ style="${styles.joinToString(";")}" """
    }

    private fun mapAlignItems(value: String): String? = when (value) {
        "start", "flex-start", "top" -> "flex-start"
        "center", "middle" -> "center"
        "end", "flex-end", "bottom" -> "flex-end"
        "stretch" -> "stretch"
        "baseline" -> "baseline"
        else -> null
    }

    private fun mapJustifyContent(value: String): String? = when (value) {
        "start", "flex-start", "left" -> "flex-start"
        "center" -> "center"
        "end", "flex-end", "right" -> "flex-end"
        "between", "space-between" -> "space-between"
        "around", "space-around" -> "space-around"
        "evenly", "space-evenly" -> "space-evenly"
        else -> null
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

    private fun renderText(component: JsonObject, sourceDir: File?): String {
        val variant = (component.getString("variant") ?: "body").lowercase(Locale.US)
        val cls = when (variant) {
            "h1" -> "text headline-xl"
            "h2" -> "text headline-lg"
            "h3" -> "text headline-md"
            "h4" -> "text title-lg"
            "caption" -> "text body-xs"
            else -> "text body-lg"
        }
        val rawText = readDynamicString(component.get("text"))
        if (rawText.isBlank()) {
            return "<p class=\"$cls\"></p>"
        }

        if (shouldRenderStructuredText(rawText, variant)) {
            val structured = renderStructuredText(rawText, sourceDir)
            return """<div class="$cls rich-text">$structured</div>"""
        }

        val text = formatInlineText(rawText, sourceDir).replace("\n", "<br/>")
        return "<p class=\"$cls\">$text</p>"
    }

    private fun shouldRenderStructuredText(rawText: String, variant: String): Boolean {
        if (variant in setOf("h1", "h2", "h3", "h4")) {
            return false
        }
        val normalized = rawText.replace("\r\n", "\n")
        val lines = normalized.lines().map { it.trim() }.filter { it.isNotEmpty() }
        if (lines.isEmpty()) {
            return false
        }
        val buttonSyntax = normalized.contains("[Button:", ignoreCase = true)
        val tableSyntax = lines.count { isTableLikeLine(it) } >= 2
        val listSyntax = lines.count { isBulletListLine(it) } >= 2
        val sectionBreaks = normalized.contains("\n\n")
        return buttonSyntax || tableSyntax || listSyntax || sectionBreaks
    }

    private fun renderStructuredText(rawText: String, sourceDir: File?): String {
        val lines = rawText.replace("\r\n", "\n").split('\n')
        val out = StringBuilder()
        var index = 0
        var renderedAny = false

        while (index < lines.size) {
            val line = lines[index].trim()
            if (line.isEmpty()) {
                index++
                continue
            }

            val bookingOptions = collectBookingOptions(lines, index)
            if (bookingOptions != null) {
                val (options, nextIndex) = bookingOptions
                if (options.isNotEmpty()) {
                    out.append(renderBookingOptions(options, sourceDir))
                    index = nextIndex
                    renderedAny = true
                    continue
                }
            }

            val tableRows = collectTableRows(lines, index)
            if (tableRows != null) {
                val (rows, nextIndex) = tableRows
                out.append(renderTable(rows, sourceDir))
                index = nextIndex
                renderedAny = true
                continue
            }

            val mediaEntries = collectMediaEntries(lines, index)
            if (mediaEntries != null) {
                val (entries, nextIndex) = mediaEntries
                if (entries.isNotEmpty()) {
                    out.append(renderMediaCards(entries, sourceDir))
                    renderedAny = true
                }
                index = nextIndex
                continue
            }

            if (isBulletListLine(line)) {
                val items = mutableListOf<String>()
                var cursor = index
                while (cursor < lines.size) {
                    val listLine = lines[cursor].trim()
                    if (!isBulletListLine(listLine)) {
                        break
                    }
                    val item = extractBulletListItem(listLine) ?: break
                    if (!isPlaceholderListEntry(item)) {
                        items += item
                    }
                    cursor++
                }
                if (items.isNotEmpty()) {
                    out.append("<ul class=\"text-list\">")
                    items.forEach { item ->
                        out.append("<li>${renderListItem(item, sourceDir)}</li>")
                    }
                    out.append("</ul>")
                    renderedAny = true
                }
                index = cursor
                continue
            }

            val button = parseButtonLine(line)
            if (button != null) {
                out.append("<div class=\"action-row\">${renderButtonAnchor(button, sourceDir, "primary")}</div>")
                index += 1
                renderedAny = true
                continue
            }

            if (!renderedAny) {
                out.append("<h3 class=\"rich-title\">${escapeHtml(line)}</h3>")
                index += 1
                renderedAny = true
                continue
            }

            if (looksLikeSectionHeading(line)) {
                out.append("<h4 class=\"rich-heading\">${escapeHtml(line)}</h4>")
                index += 1
                continue
            }

            val paragraphLines = mutableListOf<String>()
            var cursor = index
            while (cursor < lines.size) {
                val candidate = lines[cursor].trim()
                if (candidate.isEmpty() || isStructuredBoundary(candidate)) {
                    break
                }
                paragraphLines += candidate
                cursor++
            }

            if (paragraphLines.isNotEmpty()) {
                out.append(
                    "<p class=\"rich-paragraph\">" +
                        formatInlineText(paragraphLines.joinToString(" "), sourceDir) +
                        "</p>"
                )
                index = cursor
                continue
            }

            index += 1
        }

        return out.toString()
    }

    private fun collectBookingOptions(
        lines: List<String>,
        startIndex: Int
    ): Pair<List<BookingOption>, Int>? {
        val firstLine = lines[startIndex].trim()
        if (parseOptionLine(firstLine) == null) {
            return null
        }

        val options = mutableListOf<BookingOption>()
        var cursor = startIndex

        while (cursor < lines.size) {
            val current = lines[cursor].trim()
            if (current.isEmpty()) {
                cursor++
                continue
            }

            val parsed = parseOptionLine(current) ?: break
            var next = cursor + 1
            while (next < lines.size && lines[next].trim().isEmpty()) {
                next++
            }

            val button = if (next < lines.size) parseButtonLine(lines[next].trim()) else null
            options += BookingOption(
                title = parsed.first,
                details = parsed.second,
                button = button
            )

            cursor = if (button != null) next + 1 else cursor + 1
            while (cursor < lines.size && lines[cursor].trim().isEmpty()) {
                cursor++
            }

            if (cursor >= lines.size || parseOptionLine(lines[cursor].trim()) == null) {
                break
            }
        }

        val logoBlock = collectTrailingLogoBlock(lines, cursor)
        if (logoBlock != null) {
            val (logos, nextCursor) = logoBlock
            if (logos.isNotEmpty()) {
                return attachLogosToOptions(options, logos) to nextCursor
            }
        }

        return options to cursor
    }

    private fun collectTrailingLogoBlock(
        lines: List<String>,
        startIndex: Int
    ): Pair<List<ParsedLogo>, Int>? {
        var cursor = startIndex
        while (cursor < lines.size && lines[cursor].trim().isEmpty()) {
            cursor++
        }
        if (cursor >= lines.size) {
            return null
        }

        val heading = lines[cursor].trim()
        if (!heading.startsWith("Airline Logos", ignoreCase = true)) {
            return null
        }
        cursor += 1

        val logos = mutableListOf<ParsedLogo>()
        while (cursor < lines.size) {
            val line = lines[cursor].trim()
            if (line.isEmpty()) {
                cursor++
                continue
            }
            if (!isBulletListLine(line)) {
                break
            }

            val item = extractBulletListItem(line) ?: break
            val parts = item.split(":", limit = 2)
            if (parts.size == 2) {
                val label = parts[0].trim()
                val value = parts[1].trim()
                if (label.isNotEmpty() && looksLikeImagePath(value)) {
                    logos += ParsedLogo(label = label, url = value)
                }
            }
            cursor++
        }

        if (logos.isEmpty()) {
            return null
        }
        return logos to cursor
    }

    private fun attachLogosToOptions(
        options: List<BookingOption>,
        logos: List<ParsedLogo>
    ): List<BookingOption> {
        if (options.isEmpty() || logos.isEmpty()) {
            return options
        }

        val remaining = logos.toMutableList()
        return options.map { option ->
            val matched = matchLogoForOption(option, remaining)
            if (matched != null) {
                remaining.remove(matched)
                option.copy(logo = matched)
            } else {
                option
            }
        }
    }

    private fun matchLogoForOption(
        option: BookingOption,
        logos: List<ParsedLogo>
    ): ParsedLogo? {
        if (logos.isEmpty()) {
            return null
        }

        val haystack = normalizeMatchText(
            option.title + " " + (option.button?.label ?: "")
        )
        if (haystack.isBlank()) {
            return null
        }

        return logos.firstOrNull { logo ->
            val needle = normalizeMatchText(logo.label)
            needle.isNotBlank() && (haystack.contains(needle) || needle.contains(haystack))
        }
    }

    private fun normalizeMatchText(value: String): String {
        return value
            .lowercase(Locale.US)
            .replace(Regex("[^a-z0-9]+"), " ")
            .trim()
    }

    private fun renderBookingOptions(options: List<BookingOption>, sourceDir: File?): String {
        val cards = options.joinToString(separator = "") { option ->
            val logo = option.logo?.let {
                val logoUrl = resolveAssetUrl(it.url, sourceDir)
                """
                    <div class="booking-logo-wrap">
                      <img class="booking-logo" src="${escapeAttr(logoUrl)}" alt="${escapeAttr(it.label)} logo" />
                    </div>
                """.trimIndent()
            } ?: ""
            val action = option.button?.let {
                "<div class=\"booking-action\">${renderButtonAnchor(it, sourceDir, "primary")}</div>"
            } ?: ""
            """
                <article class="booking-card">
                  $logo
                  <h5 class="booking-title">${escapeHtml(option.title)}</h5>
                  <p class="booking-body">${formatInlineText(option.details, sourceDir)}</p>
                  $action
                </article>
            """.trimIndent()
        }
        return """<div class="booking-grid">$cards</div>"""
    }

    private fun collectMediaEntries(
        lines: List<String>,
        startIndex: Int
    ): Pair<List<ParsedMediaEntry>, Int>? {
        var cursor = startIndex
        val entries = mutableListOf<ParsedMediaEntry>()
        var markerHeading: String? = null

        while (cursor < lines.size) {
            val line = lines[cursor].trim()
            if (line.isEmpty()) {
                cursor++
                continue
            }

            if (isMediaMarkerHeading(line)) {
                markerHeading = line.lowercase(Locale.US)
                cursor++
                continue
            }

            val mediaEntry = parseMediaEntryLine(line)
            if (mediaEntry != null) {
                entries += mediaEntry
                cursor++
                continue
            }
            break
        }

        if (entries.isEmpty()) {
            return null
        }

        if (markerHeading?.startsWith("icons") == true &&
            entries.all { it.iconLike && isGenericUtilityIconLabel(it.label) }
        ) {
            // Skip generic booking utility icon lists like Airline/Time/Calendar.
            return emptyList<ParsedMediaEntry>() to cursor
        }

        return entries to cursor
    }

    private fun isGenericUtilityIconLabel(label: String): Boolean {
        val normalized = normalizeMatchText(label)
        return normalized in setOf("airline", "time", "calendar")
    }

    private fun shouldShowMediaLabel(label: String): Boolean {
        val normalized = normalizeMatchText(label)
        if (normalized.isBlank()) {
            return false
        }
        if (normalized in setOf("image", "icon", "photo", "logo", "media")) {
            return false
        }
        return !Regex("""^(image|icon|photo|logo|media)\s*[:#-]?\s*\d*$""", RegexOption.IGNORE_CASE)
            .matches(normalized)
    }

    private fun isMediaMarkerHeading(line: String): Boolean {
        val normalized = line.trim().lowercase(Locale.US)
        return normalized in setOf(
            "icons:",
            "icon:",
            "assets:",
            "asset:",
            "logos:",
            "logo:",
            "images:",
            "image:",
            "airline logos:"
        )
    }

    private fun parseMediaEntryLine(line: String): ParsedMediaEntry? {
        val normalized = stripLeadingBulletMarker(line)
        val parts = normalized.split(":", limit = 2)
        if (parts.size != 2) {
            return null
        }
        val label = parts[0].trim()
        val value = parts[1].trim()
        if (label.isEmpty() || !looksLikeImagePath(value)) {
            return null
        }

        val lowerLabel = label.lowercase(Locale.US)
        val lowerValue = value.lowercase(Locale.US)
        val iconLike = lowerLabel.contains("icon") ||
            lowerValue.endsWith(".svg") && !lowerLabel.contains("logo") && !lowerValue.contains("logo")

        return ParsedMediaEntry(
            label = label,
            url = value,
            iconLike = iconLike
        )
    }

    private fun renderMediaCards(entries: List<ParsedMediaEntry>, sourceDir: File?): String {
        val primary = entries.filterNot { it.iconLike }
        val icons = entries.filter { it.iconLike }

        val cards = if (primary.isNotEmpty()) {
            val iconBuckets = MutableList(primary.size) { mutableListOf<ParsedMediaEntry>() }
            icons.forEachIndexed { index, icon ->
                iconBuckets[index % primary.size] += icon
            }

            primary.mapIndexed { index, entry ->
                val resolved = resolveAssetUrl(entry.url, sourceDir)
                val inlineIcons = iconBuckets[index]
                val iconStrip = if (inlineIcons.isEmpty()) {
                    ""
                } else {
                    val iconHtml = inlineIcons.joinToString(separator = "") { icon ->
                        val iconUrl = resolveAssetUrl(icon.url, sourceDir)
                        """
                            <span class="media-inline-icon-wrap">
                              <img class="media-inline-icon" src="${escapeAttr(iconUrl)}" alt="${escapeAttr(icon.label)}" title="${escapeAttr(icon.label)}" />
                            </span>
                        """.trimIndent()
                    }
                    """<div class="media-icon-row">$iconHtml</div>"""
                }
                val labelHtml = if (shouldShowMediaLabel(entry.label)) {
                    """<p class="media-label">${escapeHtml(entry.label)}</p>"""
                } else {
                    ""
                }

                """
                    <article class="media-card">
                      <div class="media-preview-wrap media-preview-wrap-image">
                        <img class="media-preview" src="${escapeAttr(resolved)}" alt="${escapeAttr(entry.label)}" />
                      </div>
                      $labelHtml
                      $iconStrip
                    </article>
                """.trimIndent()
            }.joinToString(separator = "")
        } else {
            entries.joinToString(separator = "") { entry ->
                val resolved = resolveAssetUrl(entry.url, sourceDir)
                val labelHtml = if (shouldShowMediaLabel(entry.label)) {
                    """<p class="media-label">${escapeHtml(entry.label)}</p>"""
                } else {
                    ""
                }
                """
                    <article class="media-card">
                      <div class="media-preview-wrap media-preview-wrap-icon">
                        <img class="media-preview media-preview-icon" src="${escapeAttr(resolved)}" alt="${escapeAttr(entry.label)}" />
                      </div>
                      $labelHtml
                    </article>
                """.trimIndent()
            }
        }

        return """<div class="media-grid">$cards</div>"""
    }

    private fun collectTableRows(
        lines: List<String>,
        startIndex: Int
    ): Pair<List<List<String>>, Int>? {
        if (!isTableLikeLine(lines[startIndex].trim())) {
            return null
        }

        val rows = mutableListOf<List<String>>()
        var cursor = startIndex
        while (cursor < lines.size) {
            val candidate = lines[cursor].trim()
            if (!isTableLikeLine(candidate)) {
                break
            }
            rows += splitTableCells(candidate)
            cursor++
        }

        val filteredRows = rows.filterIndexed { index, row ->
            index == 0 || !isPlaceholderTableStringRow(row)
        }
        if (filteredRows.size < 2) {
            return null
        }
        return filteredRows to cursor
    }

    private fun renderTable(rows: List<List<String>>, sourceDir: File?): String {
        if (rows.isEmpty()) return ""
        val header = rows.first()
        val body = rows.drop(1).filterNot { isPlaceholderTableStringRow(it) }
        if (body.isEmpty()) return ""
        val headHtml = header.joinToString(separator = "") { cell ->
            "<th>${formatInlineText(cell, sourceDir)}</th>"
        }
        val bodyHtml = body.joinToString(separator = "") { row ->
            val cells = row.joinToString(separator = "") { cell ->
                "<td>${formatInlineText(sanitizeTableCellDisplayValue(cell), sourceDir)}</td>"
            }
            "<tr>$cells</tr>"
        }
        return """
            <div class="table-wrap">
              <table class="data-table">
                <thead><tr>$headHtml</tr></thead>
                <tbody>$bodyHtml</tbody>
              </table>
            </div>
        """.trimIndent()
    }

    private fun renderListItem(item: String, sourceDir: File?): String {
        val leadingLabel = parseLeadingLabelValue(item)
        if (leadingLabel != null) {
            val label = leadingLabel.label
            val value = leadingLabel.value
            if (looksLikeImagePath(value)) {
                val imageUrl = resolveAssetUrl(value, sourceDir)
                return """
                    <span class="list-label">${escapeHtml(leadingLabel.prefix)}${escapeHtml(label)}${escapeHtml(leadingLabel.delimiter)}</span>
                    <img class="inline-logo" src="${escapeAttr(imageUrl)}" alt="${escapeAttr(label)}" />
                """.trimIndent()
            }
        }
        return formatInlineText(item, sourceDir)
    }

    private fun parseOptionLine(line: String): Pair<String, String>? {
        val match = OPTION_LINE_REGEX.find(line) ?: return null
        val title = match.groupValues[1].trim()
        val details = match.groupValues[2].trim()
        if (title.isEmpty() || details.isEmpty()) {
            return null
        }
        return title to details
    }

    private fun parseButtonLine(line: String): ParsedButton? {
        val normalized = BULLET_PREFIX_REGEX.replace(line.trim(), "")
        val markdownLink = BUTTON_MARKDOWN_LINK_REGEX.find(normalized)
        val (label, trailing) = if (markdownLink != null) {
            markdownLink.groupValues[1].trim() to markdownLink.groupValues[2].trim()
        } else {
            val match = BUTTON_LINE_REGEX.find(normalized) ?: return null
            match.groupValues[1].trim() to match.groupValues[2].trim()
        }
        val rawUrlToken = URL_REGEX.find(trailing)?.value ?: trailing
        val url = sanitizeUrlToken(rawUrlToken)
        if (label.isEmpty() || url.isEmpty()) {
            return null
        }
        return ParsedButton(label = label, url = url)
    }

    private fun renderButtonAnchor(button: ParsedButton, sourceDir: File?, style: String): String {
        val resolved = resolveAssetUrl(button.url, sourceDir)
        return """
            <a class="button $style" href="${escapeAttr(resolved)}">${escapeHtml(button.label)}</a>
        """.trimIndent()
    }

    private fun sanitizeUrlToken(value: String): String =
        value.trim().trim('"', '\'', '<', '>', '`').trimEnd('.', ',', ';', ')', ']', '}')

    private fun isTableLikeLine(line: String): Boolean {
        if (line.contains("http://", ignoreCase = true) || line.contains("https://", ignoreCase = true)) {
            return false
        }
        return splitTableCells(line).size >= 3
    }

    private fun splitTableCells(line: String): List<String> =
        line
            .split('|')
            .map { it.trim() }
            .filter { it.isNotEmpty() }

    private fun isPlaceholderTableStringRow(row: List<String>): Boolean {
        if (row.isEmpty()) return true
        return row.all(::isPlaceholderTableCellValue)
    }

    private fun sanitizeTableCellDisplayValue(value: String): String {
        val normalized = value.trim()
        return if (isPlaceholderTableCellValue(normalized)) "" else normalized
    }

    private fun isPlaceholderTableCellValue(value: String): Boolean {
        val normalized = value.trim()
        if (normalized.isEmpty()) {
            return true
        }
        val compact = normalized.replace(Regex("""\s+"""), "")
        if (TABLE_PLACEHOLDER_CELL_REGEX.matches(compact)) {
            return true
        }
        return when (normalized.lowercase(Locale.US)) {
            "na", "n/a", "null", "none", "not available" -> true
            else -> false
        }
    }

    private fun isPlaceholderListEntry(value: String): Boolean {
        val normalized = value
            .replace(Regex("""^[\u2022•\-*]\s*"""), "")
            .trim()
        if (isPlaceholderTableCellValue(normalized)) {
            return true
        }
        val leadingLabel = parseLeadingLabelValue(normalized) ?: return false
        return isPlaceholderTableCellValue(leadingLabel.value)
    }

    private fun looksLikeSectionHeading(line: String): Boolean {
        if (line.length > 80) {
            return false
        }
        if (isBulletListLine(line) || line.endsWith(".") || line.endsWith("?") || line.endsWith("!")) {
            return false
        }
        if (line.contains('|') || line.contains("http://", ignoreCase = true) || line.contains("https://", ignoreCase = true)) {
            return false
        }
        if (line.contains("[Button:", ignoreCase = true)) {
            return false
        }
        return line.any { it.isLetter() }
    }

    private fun isStructuredBoundary(line: String): Boolean {
        if (isBulletListLine(line)) {
            return true
        }
        if (parseButtonLine(line) != null || parseOptionLine(line) != null) {
            return true
        }
        if (isTableLikeLine(line) || looksLikeSectionHeading(line)) {
            return true
        }
        return false
    }

    private fun isBulletListLine(line: String): Boolean =
        BULLET_LINE_REGEX.matches(line.trim())

    private fun extractBulletListItem(line: String): String? =
        BULLET_LINE_REGEX.matchEntire(line.trim())
            ?.groupValues
            ?.getOrNull(2)
            ?.trim()
            ?.takeIf { it.isNotBlank() }

    private fun stripLeadingBulletMarker(line: String): String =
        extractBulletListItem(line) ?: line.trim()

    private fun looksLikeImagePath(value: String): Boolean {
        val normalized = value.trim().lowercase(Locale.US)
        return normalized.endsWith(".png") ||
            normalized.endsWith(".jpg") ||
            normalized.endsWith(".jpeg") ||
            normalized.endsWith(".svg") ||
            normalized.endsWith(".webp")
    }

    private fun formatInlineText(rawText: String, sourceDir: File?): String {
        val escaped = escapeHtml(
            NativeTextFormatter.sanitizeDisplayText(
                normalizeMojibakeText(rawText),
                preserveMarkdown = true
            )
        )
        val withLeadingLabels = emphasizeLeadingLabels(escaped)
        val withBold = BOLD_REGEX.replace(withLeadingLabels) { match ->
            "<strong>${match.groupValues[1]}</strong>"
        }
        return linkifyUrls(withBold, sourceDir)
    }

    private fun emphasizeLeadingLabels(text: String): String {
        val out = StringBuilder()
        val lines = text.split('\n')
        lines.forEachIndexed { lineIndex, line ->
            val ranges = mutableListOf<IntRange>()
            if (line.trim().isNotEmpty() && !URL_REGEX.containsMatchIn(line)) {
                LEADING_LABEL_REGEX.find(line)?.let { match ->
                    val label = match.groups[2]?.value?.trim().orEmpty()
                    val labelStart = match.groups[2]?.range?.first ?: match.range.first
                    val delimiterEnd = match.groups[3]?.range?.last ?: match.range.last
                    val valueText = match.groups[4]?.value?.trim().orEmpty()
                    if (isLikelyLeadingLabel(label, valueText)) {
                        ranges += labelStart..delimiterEnd
                    }
                }
                INLINE_LABEL_REGEX.findAll(line).forEach { match ->
                    val label = match.groups[1]?.value?.trim().orEmpty()
                    val labelStart = match.groups[1]?.range?.first ?: match.range.first
                    val delimiterEnd = match.groups[2]?.range?.last ?: match.range.last
                    val valueText = line.substring(delimiterEnd + 1).trimStart()
                    val range = labelStart..delimiterEnd
                    if (valueText.isEmpty() ||
                        !isLikelyLeadingLabel(label, valueText) ||
                        !isSentenceBoundary(line, labelStart) ||
                        ranges.any { existing -> range.first <= existing.last && range.last >= existing.first }
                    ) {
                        return@forEach
                    }
                    ranges += range
                }
            }

            if (ranges.isEmpty()) {
                out.append(line)
            } else {
                var cursor = 0
                ranges
                    .sortedBy { it.first }
                    .forEach { range ->
                        if (range.first < cursor) {
                            return@forEach
                        }
                        out.append(line.substring(cursor, range.first))
                        out.append("<strong>")
                        out.append(line.substring(range.first, range.last + 1))
                        out.append("</strong>")
                        cursor = range.last + 1
                    }
                out.append(line.substring(cursor))
            }

            if (lineIndex != lines.lastIndex) {
                out.append('\n')
            }
        }
        return out.toString()
    }

    private fun parseLeadingLabelValue(text: String): LeadingLabelValue? {
        val match = LEADING_LABEL_REGEX.find(text) ?: return null
        val prefix = match.groups[1]?.value.orEmpty()
        val label = match.groups[2]?.value?.trim().orEmpty()
        val delimiter = match.groups[3]?.value.orEmpty()
        val value = match.groups[4]?.value?.trim().orEmpty()
        if (!isLikelyLeadingLabel(label, value)) {
            return null
        }
        return LeadingLabelValue(
            prefix = prefix,
            label = label,
            delimiter = delimiter,
            value = value
        )
    }

    private fun isLikelyLeadingLabel(label: String, value: String): Boolean {
        if (label.isBlank() || value.isBlank()) {
            return false
        }
        val normalized = label.trim()
        val lower = normalized.lowercase(Locale.US)
        if (lower in setOf("http", "https", "file", "content")) {
            return false
        }
        if (!normalized.any { it.isLetter() }) {
            return false
        }
        if (normalized.length > 60) {
            return false
        }
        return true
    }

    private fun isSentenceBoundary(line: String, start: Int): Boolean {
        if (start <= 0) {
            return true
        }
        val prev = line[start - 1]
        if (prev == '•' || prev == '-') {
            return true
        }
        if (prev != ' ') {
            return false
        }
        val prevNonSpaceIndex = line.substring(0, start - 1).indexOfLast { !it.isWhitespace() }
        if (prevNonSpaceIndex < 0) {
            return true
        }
        return line[prevNonSpaceIndex] in charArrayOf('.', '!', '?', '•', '-', ')')
    }

    private fun linkifyUrls(escapedText: String, sourceDir: File?): String {
        val matches = URL_REGEX.findAll(escapedText).toList()
        if (matches.isEmpty()) {
            return escapedText
        }

        val out = StringBuilder()
        var cursor = 0
        for (match in matches) {
            out.append(escapedText.substring(cursor, match.range.first))
            val escapedUrl = match.value
            val rawUrl = escapedUrl.replace("&amp;", "&")
            val resolved = resolveAssetUrl(rawUrl, sourceDir)
            out.append("""<a class="inline-link" href="${escapeAttr(resolved)}">$escapedUrl</a>""")
            cursor = match.range.last + 1
        }
        out.append(escapedText.substring(cursor))
        return out.toString()
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
        if (className == "icon") {
            return """<img class="$className" src="${escapeAttr(resolved)}" alt="$className" />"""
        }

        val variant = (component.getString("variant") ?: "").lowercase(Locale.US)
        val fit = component.getString("fit")
            ?.lowercase(Locale.US)
            ?.replace("_", "")
            ?.replace("-", "")
            ?.replace(" ", "")
        val rawLower = raw.lowercase(Locale.US)
        val likelyLogo = variant.contains("logo") || rawLower.contains("logo")
        val raster = rawLower.endsWith(".png") ||
            rawLower.endsWith(".jpg") ||
            rawLower.endsWith(".jpeg") ||
            rawLower.endsWith(".webp")
        val useCover = when (fit) {
            "contain", "fit", "inside" -> false
            "cover", "crop", "fill", "fillbounds", "fillwidth", "fillheight" -> true
            else -> raster && !likelyLogo
        }
        val imageClass = if (useCover) "$className image-cover" else "$className image-contain"
        return """<img class="$imageClass" src="${escapeAttr(resolved)}" alt="$className" />"""
    }

    private fun renderVideo(component: JsonObject, sourceDir: File?): String {
        val raw = readDynamicString(component.get("url")).trim()
        if (raw.isBlank()) {
            return "<div class=\"warning\">Missing video source</div>"
        }
        val resolved = resolveAssetUrl(raw, sourceDir)
        return """
            <div class="media-player-wrap">
              <video class="media-player video-player" controls preload="metadata" src="${escapeAttr(resolved)}"></video>
              <a class="inline-link" href="${escapeAttr(resolved)}" target="_blank" rel="noopener noreferrer">Open video</a>
            </div>
        """.trimIndent()
    }

    private fun renderAudioPlayer(component: JsonObject, sourceDir: File?): String {
        val raw = readDynamicString(component.get("url")).trim()
        if (raw.isBlank()) {
            return "<div class=\"warning\">Missing audio source</div>"
        }
        val resolved = resolveAssetUrl(raw, sourceDir)
        val description = escapeHtml(sanitizeDisplayText(readDynamicString(component.get("description"))))
        val title = if (description.isBlank()) "" else "<p class=\"rich-paragraph\">$description</p>"
        return """
            <div class="media-player-wrap">
              $title
              <audio class="media-player audio-player" controls preload="metadata" src="${escapeAttr(resolved)}"></audio>
              <a class="inline-link" href="${escapeAttr(resolved)}" target="_blank" rel="noopener noreferrer">Open audio</a>
            </div>
        """.trimIndent()
    }

    private fun renderModal(
        component: JsonObject,
        index: Map<String, JsonObject>,
        sourceDir: File?,
        activePath: MutableSet<String>,
        warnings: MutableList<String>
    ): String {
        val triggerId = component.getString("trigger")
        val contentId = component.getString("content")
        val summary = escapeHtml(resolveComponentLabel(triggerId, index).ifBlank { "Open details" })
        val contentHtml = if (contentId.isNullOrBlank()) {
            "<div class=\"warning\">Missing modal content</div>"
        } else {
            renderComponent(contentId, index, sourceDir, activePath, warnings)
        }
        return """
            <details class="modal-box">
              <summary class="modal-trigger">$summary</summary>
              <div class="modal-content">$contentHtml</div>
            </details>
        """.trimIndent()
    }

    private fun renderTextField(component: JsonObject): String {
        val label = escapeHtml(sanitizeDisplayText(readDynamicString(component.get("label"))).ifBlank { "Input" })
        val value = escapeAttr(readDynamicString(component.get("value")))
        val variant = (component.getString("variant") ?: "shortText").lowercase(Locale.US)
        return when {
            variant.contains("longtext") -> """
                <label class="input-block">
                  <span class="input-label">$label</span>
                  <textarea class="text-area" rows="4">$value</textarea>
                </label>
            """.trimIndent()
            else -> {
                val inputType = when {
                    variant.contains("number") -> "number"
                    variant.contains("obscured") -> "password"
                    else -> "text"
                }
                """
                <label class="input-block">
                  <span class="input-label">$label</span>
                  <input class="text-input" type="$inputType" value="$value" />
                </label>
                """.trimIndent()
            }
        }
    }

    private fun renderCheckBox(component: JsonObject): String {
        val label = escapeHtml(sanitizeDisplayText(readDynamicString(component.get("label"))).ifBlank { "Option" })
        val checked = component.get("value")?.asBooleanOrNull()
            ?: parseBooleanLike(readDynamicString(component.get("value")))
            ?: false
        val checkedAttr = if (checked) " checked" else ""
        return """<label class="check-row"><input type="checkbox"$checkedAttr /><span>$label</span></label>"""
    }

    private fun renderChoicePicker(component: JsonObject): String {
        val componentId = component.getString("id").orEmpty()
        val label = escapeHtml(sanitizeDisplayText(readDynamicString(component.get("label"))))
        val options = readChoiceOptions(component)
        if (options.isEmpty()) {
            return "<div class=\"warning\">ChoicePicker has no options</div>"
        }

        val multiple = (component.getString("variant") ?: "mutuallyExclusive")
            .lowercase(Locale.US)
            .contains("multiple")
        val selected = readDynamicStringList(component.get("value")).toSet()
        val head = if (label.isBlank()) "" else "<p class=\"input-label\">$label</p>"
        val body = if (multiple) {
            options.joinToString("") { option ->
                val checked = if (selected.contains(option.value)) " checked" else ""
                """<label class="check-row"><input type="checkbox" value="${escapeAttr(option.value)}"$checked /><span>${escapeHtml(option.label)}</span></label>"""
            }
        } else {
            val radioName = if (componentId.isBlank()) "choice_picker" else "choice_picker_${escapeAttr(componentId)}"
            val selectedSingle = selected.firstOrNull() ?: options.first().value
            options.joinToString("") { option ->
                val checked = if (option.value == selectedSingle) " checked" else ""
                """<label class="check-row"><input type="radio" name="$radioName" value="${escapeAttr(option.value)}"$checked /><span>${escapeHtml(option.label)}</span></label>"""
            }
        }
        return """<div class="choice-picker">$head$body</div>"""
    }

    private fun renderSlider(component: JsonObject): String {
        val label = escapeHtml(sanitizeDisplayText(readDynamicString(component.get("label"))).ifBlank { "Value" })
        val min = component.getAsNumberOrNull("min") ?: 0.0
        val maxRaw = component.getAsNumberOrNull("max") ?: 100.0
        val max = if (maxRaw <= min) min + 1.0 else maxRaw
        val value = readDynamicNumber(component.get("value"))?.coerceIn(min, max) ?: min
        return """
            <label class="input-block">
              <span class="input-label">$label: ${escapeHtml(formatNumber(value))}</span>
              <input class="range-input" type="range" min="${escapeAttr(formatNumber(min))}" max="${escapeAttr(formatNumber(max))}" value="${escapeAttr(formatNumber(value))}" />
            </label>
        """.trimIndent()
    }

    private fun renderDateTimeInput(component: JsonObject): String {
        val label = escapeHtml(sanitizeDisplayText(readDynamicString(component.get("label"))).ifBlank { "Date/time" })
        val value = escapeAttr(readDynamicString(component.get("value")))
        val enableDate = component.get("enableDate")?.asBooleanOrNull() ?: true
        val enableTime = component.get("enableTime")?.asBooleanOrNull() ?: false
        val inputType = when {
            enableDate && enableTime -> "datetime-local"
            enableDate -> "date"
            enableTime -> "time"
            else -> "text"
        }
        return """
            <label class="input-block">
              <span class="input-label">$label</span>
              <input class="text-input" type="$inputType" value="$value" />
            </label>
        """.trimIndent()
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

    private fun resolveComponentLabel(componentId: String?, index: Map<String, JsonObject>): String {
        if (componentId.isNullOrBlank()) {
            return ""
        }
        val component = index[componentId] ?: return ""
        val type = component.getString("component") ?: return ""
        if (type.equals("Text", ignoreCase = true)) {
            return sanitizeDisplayText(readDynamicString(component.get("text")))
        }
        if (type.equals("Button", ignoreCase = true)) {
            val childId = component.getString("child")
            if (!childId.isNullOrBlank()) {
                val child = index[childId]
                if (child != null && child.getString("component").equals("Text", ignoreCase = true)) {
                    return sanitizeDisplayText(readDynamicString(child.get("text")))
                }
            }
        }
        return ""
    }

    private fun readChoiceOptions(component: JsonObject): List<ChoiceOption> {
        val options = component.getAsJsonArrayOrNull("options") ?: return emptyList()
        return options.mapNotNull { option ->
            val obj = option.asJsonObjectOrNull() ?: return@mapNotNull null
            val value = sanitizeDisplayText(readDynamicString(obj.get("value"))).ifBlank {
                sanitizeDisplayText(readDynamicString(obj.get("label")))
            }
            val label = sanitizeDisplayText(readDynamicString(obj.get("label"))).ifBlank { value }
            if (value.isBlank()) {
                null
            } else {
                ChoiceOption(label = label, value = value)
            }
        }
    }

    private fun readDynamicStringList(element: JsonElement?): List<String> {
        if (element == null || element.isJsonNull) {
            return emptyList()
        }
        if (element.isJsonArray) {
            return element.asJsonArray.map { readDynamicString(it).trim() }.filter { it.isNotBlank() }
        }
        val token = readDynamicString(element).trim()
        if (token.isBlank()) {
            return emptyList()
        }
        return token
            .split(',', ';', '|')
            .map { it.trim() }
            .filter { it.isNotBlank() }
    }

    private fun parseBooleanLike(value: String?): Boolean? {
        return when (value?.trim()?.lowercase(Locale.US)) {
            "true", "1", "yes", "on" -> true
            "false", "0", "no", "off" -> false
            else -> null
        }
    }

    private fun readDynamicNumber(element: JsonElement?): Double? {
        if (element == null || element.isJsonNull) {
            return null
        }
        if (element.isJsonPrimitive && element.asJsonPrimitive.isNumber) {
            return element.asDouble
        }
        val token = readDynamicString(element).trim().replace(",", "")
        return token.toDoubleOrNull()
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

    private fun applyComponentActionWrapper(
        component: JsonObject,
        renderedHtml: String,
        sourceDir: File?,
        warnings: MutableList<String>
    ): String {
        if (component.getString("component").equals("Button", ignoreCase = true)) {
            return renderedHtml
        }
        val action = extractComponentAction(component) ?: return renderedHtml
        return when (action.kind) {
            ComponentActionKind.OpenUrl -> {
                if (renderedHtml.contains("<a ", ignoreCase = true)) {
                    warnings += "Skipped wrapping nested link interaction for component with existing anchor."
                    return renderedHtml
                }
                val resolved = resolveAssetUrl(action.value, sourceDir)
                """<a class="component-action-link" href="${escapeAttr(resolved)}">$renderedHtml</a>"""
            }

            ComponentActionKind.ShowMessage,
            ComponentActionKind.ShowSurface -> {
                warnings += "Non-link interaction (${action.kind}) is supported in native renderer only."
                renderedHtml
            }
        }
    }

    private fun extractOpenUrl(component: JsonObject): String? =
        extractComponentAction(component)
            ?.takeIf { it.kind == ComponentActionKind.OpenUrl }
            ?.value

    private fun extractComponentAction(component: JsonObject): ComponentAction? {
        parseActionEnvelope(component.getAsJsonObjectOrNull("action"))?.let { return it }
        parseActionEnvelope(component.getAsJsonObjectOrNull("onClick"))?.let { return it }
        parseActionEvent(component.get("event"))?.let { return it }
        return null
    }

    private fun parseActionEnvelope(actionObject: JsonObject?): ComponentAction? {
        if (actionObject == null) {
            return null
        }
        parseFunctionCallAction(actionObject.getAsJsonObjectOrNull("functionCall"))?.let { return it }
        parseFunctionCallAction(actionObject)?.let { return it }
        parseActionEvent(actionObject.get("event"))?.let { return it }

        listOf("onClick", "click", "tap", "press", "onPress", "onSelect", "select").forEach { key ->
            parseActionEnvelope(actionObject.getAsJsonObjectOrNull(key))?.let { return it }
        }

        actionObject.getAsJsonArrayOrNull("events")?.forEach { eventEntry ->
            parseActionEvent(eventEntry)?.let { return it }
        }
        actionObject.getAsJsonArrayOrNull("handlers")?.forEach { eventEntry ->
            parseActionEvent(eventEntry)?.let { return it }
        }

        val callName = actionObject.getString("name")
            ?: actionObject.getString("call")
            ?: return null
        val args = actionObject.getAsJsonObjectOrNull("args")
            ?: readLegacyActionContextArgs(actionObject)
            ?: JsonObject()
        return parseActionFromCall(callName, args)
    }

    private fun parseActionEvent(event: JsonElement?): ComponentAction? {
        if (event == null || event.isJsonNull) {
            return null
        }
        if (event.isJsonArray) {
            event.asJsonArray.forEach { entry ->
                parseActionEvent(entry)?.let { return it }
            }
            return null
        }
        if (!event.isJsonObject) {
            return null
        }
        val eventObject = event.asJsonObject
        parseActionEnvelope(eventObject)?.let { return it }
        listOf("onClick", "click", "tap", "press", "onPress", "onSelect", "select").forEach { key ->
            parseActionEnvelope(eventObject.getAsJsonObjectOrNull(key))?.let { return it }
        }
        return null
    }

    private fun parseFunctionCallAction(functionCall: JsonObject?): ComponentAction? {
        if (functionCall == null) {
            return null
        }
        val callName = functionCall.getString("call")
            ?: functionCall.getString("name")
            ?: return null
        val args = functionCall.getAsJsonObjectOrNull("args")
            ?: readLegacyActionContextArgs(functionCall)
            ?: JsonObject()
        return parseActionFromCall(callName, args)
    }

    private fun readLegacyActionContextArgs(actionObject: JsonObject): JsonObject? {
        val contextEntries = actionObject.getAsJsonArrayOrNull("context") ?: return null
        if (contextEntries.size() == 0) {
            return null
        }
        val args = JsonObject()
        contextEntries.forEach { contextElement ->
            val contextObject = contextElement.asJsonObjectOrNull() ?: return@forEach
            val key = contextObject.getString("key") ?: return@forEach
            contextObject.get("value")?.let { args.add(key, it) }
        }
        return if (args.entrySet().isEmpty()) null else args
    }

    private fun parseActionFromCall(callName: String, args: JsonObject): ComponentAction? {
        val canonicalCall = callName.trim().lowercase(Locale.US)
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        return when (canonicalCall) {
            "openurl", "openlink", "launchurl", "browseurl" -> {
                readActionArgument(args, "url", "href", "link", "targetUrl")
                    ?.let { ComponentAction(ComponentActionKind.OpenUrl, it) }
            }

            "showmessage", "showtoast", "toast", "snackbar", "message" -> {
                readActionArgument(args, "message", "text", "title")
                    ?.let { ComponentAction(ComponentActionKind.ShowMessage, it) }
            }

            "showsurface", "opensurface", "navigatesurface", "switchsurface" -> {
                readActionArgument(args, "surfaceId", "id", "surface", "targetSurfaceId", "target")
                    ?.let { ComponentAction(ComponentActionKind.ShowSurface, it) }
            }

            else -> null
        }
    }

    private fun readActionArgument(args: JsonObject, vararg keys: String): String? {
        keys.forEach { key ->
            val value = readDynamicString(args.get(key)).trim()
            if (value.isNotBlank()) {
                return value
            }
        }
        return null
    }

    private fun readDynamicString(element: JsonElement?): String {
        if (element == null || element.isJsonNull) {
            return ""
        }
        if (element.isJsonPrimitive) {
            val primitive = element.asJsonPrimitive
            return when {
                primitive.isString -> normalizeMojibakeText(primitive.asString)
                primitive.isBoolean -> primitive.asBoolean.toString()
                primitive.isNumber -> primitive.asNumber.toString()
                else -> primitive.toString()
            }
        }
        if (!element.isJsonObject) {
            return normalizeMojibakeText(element.toString())
        }
        val obj = element.asJsonObject
        obj.getString("literalString")?.let { return normalizeMojibakeText(it) }
        obj.getString("path")?.let { return "" }
        obj.get("literalNumber")?.let { return normalizeMojibakeText(it.toString().trim('"')) }
        obj.get("literalBoolean")?.let { return normalizeMojibakeText(it.toString().trim('"')) }
        return normalizeMojibakeText(obj.toString())
    }

    private fun normalizeMojibakeText(value: String): String {
        if (value.isBlank()) {
            return value
        }
        val candidates = linkedSetOf(value)
        if (value.contains('Ã') || value.contains('Â') || value.contains('â')) {
            runCatching {
                String(value.toByteArray(Charset.forName("windows-1252")), Charsets.UTF_8)
            }.onSuccess { candidates += it }
            runCatching {
                String(value.toByteArray(Charsets.ISO_8859_1), Charsets.UTF_8)
            }.onSuccess { candidates += it }
        }
        val normalized = candidates.minByOrNull(::mojibakeScore) ?: value
        return normalized
            .replace("Â°", "°")
            .replace("Â₹", "₹")
            .replace("â‚¹", "₹")
            .replace("â€¢", "•")
    }

    private fun sanitizeDisplayText(value: String): String {
        return NativeTextFormatter.sanitizeDisplayText(value.replace("\u00A0", " "))
    }

    private fun mojibakeScore(value: String): Int {
        if (value.isBlank()) {
            return 0
        }
        val markers = listOf(
            "Ã", "Â", "â€¢", "â‚¹", "â€“", "â€”", "â€˜", "â€™", "â€œ", "â€�", "�"
        )
        return markers.sumOf { marker -> countOccurrences(value, marker) }
    }

    private fun countOccurrences(value: String, needle: String): Int {
        if (needle.isEmpty()) {
            return 0
        }
        var count = 0
        var cursor = 0
        while (true) {
            val index = value.indexOf(needle, cursor)
            if (index < 0) {
                break
            }
            count++
            cursor = index + needle.length
        }
        return count
    }

    private fun resolveAssetUrl(raw: String, sourceDir: File?): String {
        return NativePayloadParser.resolveAssetUrl(raw, sourceDir)
    }

    private fun buildHtml(body: String): String {
        return """
            <!doctype html>
            <html lang="en">
            <head>
              <meta charset="utf-8" />
              <meta name="viewport" content="width=device-width, initial-scale=1" />
              <title>GenUICraft</title>
              <style>
                :root {
                  --font-family: "One UI Sans GUI", "sec", "SamsungOne", "Segoe UI", sans-serif;
                  --font-light: 300;
                  --font-regular: 400;
                  --font-semibold: 600;
                  --font-bold: 700;

                  --type-headline-xl: 34px;
                  --type-headline-lg: 30px;
                  --type-headline-md: 26px;
                  --type-headline-sm: 21px;

                  --type-title-lg: 19px;
                  --type-title-md: 17px;
                  --type-title-sm: 15px;
                  --type-title-xs: 13px;

                  --type-body-xl: 18px;
                  --type-body-lg: 17px;
                  --type-body-md: 15px;
                  --type-body-sm: 14px;
                  --type-body-xs: 13px;
                  --type-body-2xs: 12px;
                  --type-body-3xs: 11px;

                  --sys-radius-sm: 8px;
                  --sys-radius-md: 12px;
                  --sys-radius-lg: 22px;
                  --sys-radius-xl: 26px;
                  --sys-radius-50: 999px;

                  --sys-border-sm: 0.5px;
                  --sys-border-md: 1px;
                  --sys-border-lg: 2px;

                  --sys-color-background: #F1F1F3;
                  --sys-color-background-variant: #FCFCFF;
                  --sys-color-surface: #FCFCFF;
                  --sys-color-surface-bright: #FCFCFF;
                  --sys-color-surface-brightest: #FCFCFF;
                  --sys-color-surface-variant: #E9E9EC;
                  --sys-color-surface-fixed: #4D4D52;
                  --sys-color-surface-fixed-variant: #010102;

                  --sys-color-surface-container-lowest: #FCFCFF;
                  --sys-color-surface-container-low: #F1F1F3;
                  --sys-color-surface-container: #E4E4E7;
                  --sys-color-surface-container-high: #CCCCCC;
                  --sys-color-surface-container-higher: #A3A3A7;
                  --sys-color-surface-container-highest: #848487;
                  --sys-color-surface-container-fixed: #FCFCFF;
                  --sys-color-surface-container-fixed-variant: #010102;

                  --sys-color-tone-on-tone: rgba(23, 23, 26, 0.05);
                  --sys-color-tone-on-tone-high: rgba(23, 23, 26, 0.10);

                  --sys-color-on-surface-container-highest: #010102;
                  --sys-color-on-surface-container-high: #252528;
                  --sys-color-on-surface-container: #4D4D52;
                  --sys-color-on-surface-container-low: #636368;
                  --sys-color-on-surface-container-lowest: #848487;
                  --sys-color-on-surface-container-fixed: #FCFCFF;
                  --sys-color-on-surface-container-fixed-variant: #010102;

                  --sys-color-outline-highest: #252528;
                  --sys-color-outline-high: #848487;
                  --sys-color-outline: #CCCCCC;
                  --sys-color-outline-low: #E4E4E7;
                  --sys-color-media-border: #B7B7BB;

                  --sys-color-primary: #387AFF;
                  --sys-color-primary-bright: #387AFF;
                  --sys-color-primary-high: #2E65D4;

                  --sys-color-functional-red: #D93E36;
                  --sys-color-functional-red-bright: #D93E36;
                  --sys-color-functional-green: #11A85F;
                  --sys-color-functional-green-bright: #11A85F;
                  --sys-color-functional-orange: #E65B17;
                  --sys-color-functional-orange-bright: #E65B17;

                  --shadow-key: rgba(0, 0, 0, 0.15);
                  --shadow-ambient: rgba(0, 0, 0, 0.04);
                  --elevation-sm: 0 4px 8px var(--shadow-key), 0 2px 4px var(--shadow-ambient);
                  --elevation-md: 0 12px 22px var(--shadow-key), 0 4px 10px var(--shadow-ambient);
                  --elevation-lg: 0 20px 34px var(--shadow-key), 0 6px 14px var(--shadow-ambient);
                  --elevation-xl: 0 32px 42px var(--shadow-key), 0 10px 20px var(--shadow-ambient);

                  --scrim: rgba(0, 0, 0, 0.60);
                  --state-overlay: rgba(0, 0, 0, 0.10);

                  --gb-percent: 0.50;
                  --gb-pivot-percent: 0.75;
                  --gb-radius: 300px;
                  --gb-saturation: 0.55;
                  --gb-curve-x1: 0.15;
                  --gb-curve-x2: 0.25;
                  --gb-tone-max-y: 180;

                  --bg-blur-thin: 22px;
                  --bg-blur-regular: 28px;
                  --bg-blur-thick: 35px;
                }

                @media (prefers-color-scheme: dark) {
                  :root {
                    --sys-color-background: #010102;
                    --sys-color-background-variant: #010102;
                    --sys-color-surface: #17171A;
                    --sys-color-surface-bright: #252528;
                    --sys-color-surface-brightest: #3A3A3D;
                    --sys-color-surface-variant: #17171A;
                    --sys-color-surface-fixed: #4D4D52;
                    --sys-color-surface-fixed-variant: #010102;

                    --sys-color-surface-container-lowest: #010102;
                    --sys-color-surface-container-low: #252528;
                    --sys-color-surface-container: #3A3A3D;
                    --sys-color-surface-container-high: #4D4D52;
                    --sys-color-surface-container-higher: #636368;
                    --sys-color-surface-container-highest: #848487;
                    --sys-color-surface-container-fixed: #FCFCFF;
                    --sys-color-surface-container-fixed-variant: #010102;

                    --sys-color-tone-on-tone: rgba(252, 252, 255, 0.10);
                    --sys-color-tone-on-tone-high: rgba(252, 252, 255, 0.15);

                    --sys-color-on-surface-container-highest: #FCFCFF;
                    --sys-color-on-surface-container-high: #E9E9EC;
                    --sys-color-on-surface-container: #CCCCCC;
                    --sys-color-on-surface-container-low: #B7B7BB;
                    --sys-color-on-surface-container-lowest: #A3A3A7;
                    --sys-color-on-surface-container-fixed: #FCFCFF;
                    --sys-color-on-surface-container-fixed-variant: #010102;

                    --sys-color-outline-highest: #E9E9EC;
                    --sys-color-outline-high: #A3A3A7;
                    --sys-color-outline: #636368;
                    --sys-color-outline-low: #3A3A3D;
                    --sys-color-media-border: #636368;

                    --sys-color-primary: #387AFF;
                    --sys-color-primary-bright: #578FFF;
                    --sys-color-primary-high: #578FFF;

                    --sys-color-functional-red: #D93E36;
                    --sys-color-functional-red-bright: #F56962;
                    --sys-color-functional-green: #11A85F;
                    --sys-color-functional-green-bright: #5AE0A0;
                    --sys-color-functional-orange: #E65B17;
                    --sys-color-functional-orange-bright: #FF8249;

                    --shadow-key: rgba(0, 0, 0, 0.35);
                    --shadow-ambient: rgba(0, 0, 0, 0.10);
                  }
                }

                * {
                  box-sizing: border-box;
                }

                html, body {
                  min-height: 100%;
                }

                body {
                  margin: 0;
                  padding: 18px;
                  background: linear-gradient(
                    180deg,
                    var(--sys-color-background-variant) 0%,
                    var(--sys-color-background) 100%
                  );
                  color: var(--sys-color-on-surface-container-highest);
                  font-family: var(--font-family);
                  font-size: var(--type-body-md);
                  font-weight: var(--font-regular);
                  line-height: normal;
                  letter-spacing: 0;
                  position: relative;
                  overflow-x: hidden;
                }

                body::before {
                  content: "";
                  position: fixed;
                  inset: 0;
                  pointer-events: none;
                  background: linear-gradient(
                    180deg,
                    var(--sys-color-background-variant) 0%,
                    var(--sys-color-background) 100%
                  );
                  transform: scale(1.10);
                  transform-origin: center;
                  filter: saturate(var(--gb-saturation)) blur(var(--gb-radius));
                  opacity: 0.20;
                  -webkit-mask-image: linear-gradient(
                    to bottom,
                    rgba(0, 0, 0, 0) 0%,
                    rgba(0, 0, 0, 1) 50%,
                    rgba(0, 0, 0, 1) 100%
                  );
                  mask-image: linear-gradient(
                    to bottom,
                    rgba(0, 0, 0, 0) 0%,
                    rgba(0, 0, 0, 1) 50%,
                    rgba(0, 0, 0, 1) 100%
                  );
                  transition: opacity 300ms cubic-bezier(0.15, 0, 0.25, 1);
                }

                body::after {
                  content: "";
                  position: fixed;
                  inset: 0;
                  pointer-events: none;
                  background: linear-gradient(
                    to bottom,
                    rgba(0, 0, 0, 0.00) 0%,
                    rgba(0, 0, 0, 0.06) 60%,
                    rgba(0, 0, 0, 0.10) 100%
                  );
                  opacity: 0.25;
                }

                .surface {
                  position: relative;
                  isolation: isolate;
                  margin: 0 auto 20px;
                  max-width: 980px;
                  background: var(--sys-color-surface);
                  border: var(--sys-border-md) solid var(--sys-color-outline);
                  border-radius: var(--sys-radius-xl);
                  padding: 18px;
                  box-shadow: var(--elevation-md);
                }

                .surface-title {
                  margin: 0 0 12px;
                  font-size: var(--type-title-xs);
                  font-weight: var(--font-semibold);
                  letter-spacing: 0.08em;
                  text-transform: uppercase;
                  color: var(--sys-color-on-surface-container-low);
                }

                .column {
                  display: flex;
                  flex-direction: column;
                  gap: 12px;
                }

                .row {
                  display: flex;
                  flex-direction: row;
                  gap: 10px;
                  flex-wrap: wrap;
                  align-items: center;
                }

                .cell {
                  min-width: 0;
                }

                .text {
                  margin: 0;
                  line-height: normal;
                  letter-spacing: 0;
                  word-break: break-word;
                  color: var(--sys-color-on-surface-container-high);
                }

                strong,
                b {
                  font-weight: var(--font-bold);
                  color: var(--sys-color-on-surface-container-highest);
                }

                .text.h1,
                .text.headline-xl {
                  font-size: var(--type-headline-xl);
                  font-weight: var(--font-bold);
                  line-height: 43px;
                  color: var(--sys-color-on-surface-container-highest);
                }

                .text.h2,
                .text.headline-lg {
                  font-size: var(--type-headline-lg);
                  font-weight: var(--font-semibold);
                  color: var(--sys-color-on-surface-container-highest);
                }

                .text.h3,
                .text.headline-md {
                  font-size: var(--type-headline-md);
                  font-weight: var(--font-semibold);
                  color: var(--sys-color-on-surface-container-highest);
                }

                .text.h4,
                .text.title-lg {
                  font-size: var(--type-title-lg);
                  font-weight: var(--font-bold);
                  color: var(--sys-color-on-surface-container-highest);
                }

                .text.body,
                .text.body-lg {
                  font-size: var(--type-body-lg);
                  font-weight: var(--font-regular);
                  color: var(--sys-color-on-surface-container-high);
                }

                .text.caption,
                .text.body-xs {
                  font-size: var(--type-body-xs);
                  font-weight: var(--font-regular);
                  color: var(--sys-color-on-surface-container-low);
                }

                .rich-text {
                  display: flex;
                  flex-direction: column;
                  gap: 12px;
                }

                .rich-title {
                  margin: 0;
                  font-size: var(--type-title-lg);
                  font-weight: var(--font-bold);
                  color: var(--sys-color-on-surface-container-highest);
                }

                .rich-heading {
                  margin: 8px 0 0;
                  font-size: var(--type-title-sm);
                  font-weight: var(--font-semibold);
                  color: var(--sys-color-on-surface-container-high);
                }

                .rich-paragraph {
                  margin: 0;
                  font-size: var(--type-body-md);
                  font-weight: var(--font-regular);
                  line-height: 1.5;
                  color: var(--sys-color-on-surface-container);
                }

                .inline-link {
                  color: var(--sys-color-primary);
                  text-decoration: none;
                  border-bottom: var(--sys-border-sm) dotted rgba(56, 122, 255, 0.65);
                }

                .inline-link:hover {
                  border-bottom-style: solid;
                }

                .media-player-wrap {
                  display: flex;
                  flex-direction: column;
                  gap: 8px;
                  padding: 10px;
                  border: var(--sys-border-md) solid var(--sys-color-outline);
                  border-radius: var(--sys-radius-lg);
                  background: rgba(252, 252, 255, 0.62);
                }

                .media-player {
                  width: 100%;
                  min-height: 44px;
                  border-radius: var(--sys-radius-md);
                }

                .modal-box {
                  border: var(--sys-border-md) solid var(--sys-color-outline);
                  border-radius: var(--sys-radius-lg);
                  background: rgba(252, 252, 255, 0.58);
                  padding: 8px 10px;
                }

                .modal-trigger {
                  cursor: pointer;
                  font-size: var(--type-body-md);
                  font-weight: var(--font-semibold);
                  color: var(--sys-color-primary);
                  list-style: none;
                }

                .modal-content {
                  margin-top: 8px;
                }

                .input-block {
                  display: flex;
                  flex-direction: column;
                  gap: 6px;
                  width: 100%;
                }

                .input-label {
                  font-size: var(--type-body-sm);
                  font-weight: var(--font-semibold);
                  color: var(--sys-color-on-surface-container-high);
                }

                .text-input,
                .text-area,
                .range-input {
                  width: 100%;
                  border: var(--sys-border-md) solid var(--sys-color-outline);
                  border-radius: var(--sys-radius-md);
                  background: rgba(252, 252, 255, 0.68);
                  color: var(--sys-color-on-surface-container-highest);
                  font: inherit;
                  padding: 8px 10px;
                }

                .text-area {
                  resize: vertical;
                }

                .check-row {
                  display: flex;
                  align-items: center;
                  gap: 8px;
                  min-height: 28px;
                  font-size: var(--type-body-sm);
                  color: var(--sys-color-on-surface-container-high);
                }

                .choice-picker {
                  display: flex;
                  flex-direction: column;
                  gap: 4px;
                }

                .text-list {
                  margin: 0;
                  padding-left: 20px;
                  display: grid;
                  gap: 10px;
                }

                .list-label {
                  font-weight: var(--font-semibold);
                  color: var(--sys-color-on-surface-container-high);
                }

                .inline-logo {
                  margin-left: 8px;
                  max-height: 24px;
                  max-width: 120px;
                  vertical-align: middle;
                }

                .table-wrap {
                  border: var(--sys-border-md) solid var(--sys-color-outline);
                  border-radius: var(--sys-radius-lg);
                  overflow-x: auto;
                  background: transparent;
                  box-shadow: var(--elevation-sm);
                  -webkit-backdrop-filter: blur(18px) saturate(1.08);
                  backdrop-filter: blur(18px) saturate(1.08);
                }

                .data-table {
                  width: 100%;
                  border-collapse: collapse;
                  min-width: 580px;
                  background: transparent;
                }

                .data-table thead th {
                  background: rgba(1, 1, 2, 0.07);
                  font-size: var(--type-body-2xs);
                  font-weight: var(--font-semibold);
                  text-transform: uppercase;
                  letter-spacing: 0.05em;
                  color: var(--sys-color-on-surface-container-high);
                }

                .data-table th,
                .data-table td {
                  padding: 10px 12px;
                  border-bottom: var(--sys-border-md) solid var(--sys-color-outline-low);
                  text-align: left;
                  vertical-align: top;
                }

                .data-table tbody td {
                  font-size: var(--type-body-sm);
                  font-weight: var(--font-regular);
                  color: var(--sys-color-on-surface-container);
                }

                .data-table tbody tr {
                  background: transparent;
                }

                .data-table tbody tr:nth-child(even) {
                  background: rgba(1, 1, 2, 0.04);
                }

                .data-table tbody tr:nth-child(odd) {
                  background: transparent;
                }

                @media (prefers-color-scheme: dark) {
                  .table-wrap {
                    background: transparent;
                  }

                  .data-table thead th {
                    background: rgba(252, 252, 255, 0.10);
                  }

                  .data-table tbody tr:nth-child(even) {
                    background: rgba(252, 252, 255, 0.06);
                  }
                }

                .component-table td,
                .component-table th {
                  min-width: 110px;
                }

                .action-row {
                  display: flex;
                  flex-wrap: wrap;
                  gap: 8px;
                }

                .booking-grid {
                  display: grid;
                  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                  gap: 12px;
                }

                .booking-card {
                  border: var(--sys-border-md) solid var(--sys-color-outline);
                  border-radius: var(--sys-radius-lg);
                  padding: 14px;
                  background: var(--sys-color-surface-container-low);
                  display: flex;
                  flex-direction: column;
                  gap: 10px;
                  box-shadow: var(--elevation-sm);
                }

                .media-grid {
                  display: grid;
                  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                  gap: 12px;
                }

                .media-card {
                  border: var(--sys-border-md) solid var(--sys-color-outline);
                  border-radius: var(--sys-radius-lg);
                  padding: 12px;
                  background: var(--sys-color-surface-container-low);
                  display: flex;
                  flex-direction: column;
                  gap: 8px;
                  box-shadow: var(--elevation-sm);
                }

                .media-preview-wrap {
                  border-radius: var(--sys-radius-md);
                  border: var(--sys-border-md) solid var(--sys-color-media-border);
                  background: var(--sys-color-surface);
                  display: flex;
                  align-items: center;
                  justify-content: center;
                  padding: 0;
                  overflow: hidden;
                }
                .media-preview-wrap-image {
                  width: 100%;
                  aspect-ratio: 3 / 2;
                }
                .media-preview-wrap-icon {
                  width: 52px;
                  height: 52px;
                  border-radius: var(--sys-radius-50);
                }
                .media-preview {
                  width: 100%;
                  height: 100%;
                  object-fit: cover;
                  display: block;
                }
                .media-preview-icon {
                  width: 32px;
                  height: 32px;
                  object-fit: contain;
                }

                .media-label {
                  margin: 0;
                  font-size: var(--type-body-xs);
                  color: var(--sys-color-on-surface-container-high);
                  font-weight: var(--font-semibold);
                  line-height: 1.35;
                }

                .media-icon-row {
                  display: flex;
                  flex-wrap: wrap;
                  gap: 6px;
                }

                .media-inline-icon-wrap {
                  width: 28px;
                  height: 28px;
                  border-radius: var(--sys-radius-50);
                  background: var(--sys-color-tone-on-tone-high);
                  border: var(--sys-border-md) solid var(--sys-color-media-border);
                  display: inline-flex;
                  align-items: center;
                  justify-content: center;
                }

                .media-inline-icon {
                  width: 16px;
                  height: 16px;
                  object-fit: contain;
                }

                .component-list {
                  margin: 0;
                  padding-left: 20px;
                  display: grid;
                  gap: 8px;
                }

                .component-list li {
                  font-size: var(--type-body-sm);
                  line-height: 1.45;
                  color: var(--sys-color-on-surface-container);
                }

                .booking-logo-wrap {
                  height: 26px;
                  display: flex;
                  align-items: center;
                }

                .booking-logo {
                  max-height: 24px;
                  max-width: 140px;
                  object-fit: contain;
                }

                .booking-title {
                  margin: 0;
                  font-size: var(--type-title-sm);
                  font-weight: var(--font-semibold);
                  color: var(--sys-color-on-surface-container-highest);
                }

                .booking-body {
                  margin: 0;
                  font-size: var(--type-body-sm);
                  font-weight: var(--font-regular);
                  color: var(--sys-color-on-surface-container);
                  line-height: 1.45;
                }

                .booking-action {
                  margin-top: 2px;
                }

                .divider {
                  border: 0;
                  border-top: var(--sys-border-md) solid var(--sys-color-outline-low);
                  margin: 6px 0;
                }

                .card {
                  border: var(--sys-border-md) solid var(--sys-color-outline);
                  border-radius: var(--sys-radius-lg);
                  padding: 14px;
                  background: var(--sys-color-surface-container-low);
                  box-shadow: var(--elevation-sm);
                }

                .component-action-link {
                  display: block;
                  text-decoration: none;
                  color: inherit;
                }

                .image {
                  width: 100%;
                  aspect-ratio: 3 / 2;
                  object-fit: cover;
                  border-radius: var(--sys-radius-md);
                  border: var(--sys-border-md) solid var(--sys-color-media-border);
                  display: block;
                  background: transparent;
                }

                .image.image-contain {
                  object-fit: contain;
                }

                .icon {
                  width: 26px;
                  height: 26px;
                  object-fit: contain;
                }

                .button {
                  display: inline-flex;
                  align-items: center;
                  justify-content: center;
                  min-height: 44px;
                  padding: 0 16px;
                  border-radius: var(--sys-radius-50);
                  border: var(--sys-border-md) solid transparent;
                  text-decoration: none;
                  font-size: var(--type-body-lg);
                  font-weight: var(--font-semibold);
                }

                .button.primary {
                  background: var(--sys-color-primary);
                  color: #FFFFFF;
                  box-shadow: var(--elevation-sm);
                }

                .button.borderless {
                  background: var(--sys-color-surface-container-low);
                  border-color: var(--sys-color-outline);
                  color: var(--sys-color-primary-high);
                }

                .button:disabled {
                  color: var(--sys-color-on-surface-container-lowest);
                  border-color: var(--sys-color-outline-low);
                  background: var(--sys-color-surface-container);
                }

                .warning {
                  margin: 0;
                  color: var(--sys-color-functional-red);
                  font-size: var(--type-body-sm);
                  font-weight: var(--font-semibold);
                }

                .tabs {
                  display: flex;
                  flex-direction: column;
                  gap: 12px;
                }

                .tab {
                  border: var(--sys-border-sm) dashed var(--sys-color-outline-high);
                  border-radius: var(--sys-radius-lg);
                  padding: 12px;
                  background: var(--sys-color-surface-container-low);
                }

                .tab-title {
                  margin: 0 0 8px;
                  font-size: var(--type-body-sm);
                  font-weight: var(--font-semibold);
                  color: var(--sys-color-on-surface-container-high);
                }

                @media (max-width: 480px) {
                  body {
                    padding: 10px;
                  }

                  .surface {
                    border-radius: var(--sys-radius-lg);
                    padding: 12px;
                    margin-bottom: 14px;
                  }

                  .row {
                    gap: 8px;
                  }

                  .button {
                    width: 100%;
                    min-height: 44px;
                  }

                  .booking-grid,
                  .media-grid {
                    grid-template-columns: 1fr;
                  }

                  .data-table {
                    min-width: 420px;
                  }

                  .data-table th,
                  .data-table td {
                    padding: 8px 10px;
                  }
                }

                @media (max-width: 410px) {
                  body {
                    padding: 12px;
                  }

                  .surface {
                    border-radius: var(--sys-radius-lg);
                    padding: 12px;
                  }

                  .data-table {
                    min-width: 380px;
                  }
                }

                @media (max-width: 320px) {
                  .booking-grid,
                  .media-grid {
                    grid-template-columns: 1fr;
                  }

                  .component-list li,
                  .text-list li {
                    white-space: normal;
                  }
                }
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

