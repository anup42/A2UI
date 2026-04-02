package com.samsung.genuicraft

import android.net.Uri
import android.widget.Toast
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.sizeIn
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentWidth
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Image
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import coil.ImageLoader
import coil.compose.AsyncImage
import coil.decode.SvgDecoder
import coil.request.ImageRequest
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.renderer.FlatSpec
import com.samsung.genuicraft.renderer.FlatSpecContent
import com.samsung.genuicraft.renderer.FlatSpecParser
import com.samsung.genuicraft.renderer.native.NativeActionParsing
import com.samsung.genuicraft.renderer.native.NativeFormComponents
import com.samsung.genuicraft.renderer.native.NativePayloadParser
import com.samsung.genuicraft.renderer.native.NativeTextFormatter
import com.samsung.genuicraft.renderer.native.*
import com.samsung.genuicraft.renderer.native.media.NativeMediaVisualUtils
import com.samsung.genuicraft.renderer.native.intents.NativeIntentRegistry
import com.samsung.genuicraft.renderer.native.parser.NativeBlockHeuristics
import com.samsung.genuicraft.renderer.native.parser.NativeBookingParsing
import com.samsung.genuicraft.renderer.native.parser.NativeMediaParsing
import com.samsung.genuicraft.renderer.native.parser.NativeSourceParsing
import com.samsung.genuicraft.renderer.native.parser.NativeStructureParsing
import com.samsung.genuicraft.renderer.native.parser.NativeTextBlockParser
import com.samsung.genuicraft.renderer.native.parser.NativeTextBlockSemantics
import com.samsung.genuicraft.renderer.native.intents.flight.NativeFlightSemantics
import com.samsung.genuicraft.renderer.native.intents.weather.NativeWeatherSemantics
import com.samsung.genuicraft.renderer.native.intents.weather.NativeWeatherUiRenderer
import com.samsung.genuicraft.renderer.native.table.NativeComponentTableExtractor
import com.samsung.genuicraft.renderer.native.table.NativeTableSemantics
import com.samsung.genuicraft.renderer.native.table.NativeTableUi
import com.samsung.genuicraft.renderer.native.text.NativeVariantStyleResolver
import java.io.File
import java.util.Locale
import kotlin.math.abs

object GenUiNativeRenderer {
    private val flatSpecWrapperKeys = listOf(
        "genui_json",
        "payload",
        "messages",
        "a2ui_json",
        "stage3_json",
        "stage3Json",
        "ir_json",
        "irJson",
        "flat_spec",
        "flatSpec"
    )

    data class RenderResult(
        val surfaces: List<SurfaceState>,
        val warnings: List<String>,
        val errorMessage: String? = null
    )

    data class SurfaceState(
        val surfaceId: String,
        val rootId: String,
        val components: Map<String, JsonObject>,
        /** Non-null when this surface uses the Phase 2+ flat-spec format. */
        val flatSpec: FlatSpec? = null
    )

    private enum class IconFallbackKind {
        Generic,
        Weather
    }

    fun render(rawInput: String, sourceDir: File?): RenderResult {
        val warnings = mutableListOf<String>()
        val parsed = try {
            NativePayloadParser.parseJsonOrJsonl(rawInput, warnings)
        } catch (exc: Exception) {
            return RenderResult(emptyList(), warnings, "Invalid payload: ${exc.message ?: exc.javaClass.simpleName}")
        }

        parseFlatSpecSurface(parsed, warnings)?.let { surface ->
            return RenderResult(listOf(surface), warnings)
        }

        val messages = NativePayloadParser.extractMessages(parsed)
            ?: return RenderResult(
                surfaces = emptyList(),
                warnings = warnings + "Input did not contain genui_json/messages/payload.",
                errorMessage = "No renderable GenUI payload found."
            )

        val surfaces = NativePayloadParser.extractSurfaces(messages, warnings)
        if (surfaces.isEmpty()) {
            return RenderResult(
                surfaces = emptyList(),
                warnings = warnings + "Messages were parsed, but no component arrays were found.",
                errorMessage = "No renderable components found."
            )
        }
        return RenderResult(surfaces, warnings)
    }

    private fun parseFlatSpecSurface(
        parsed: JsonElement,
        warnings: MutableList<String>
    ): SurfaceState? {
        if (FlatSpecParser.isFlatSpec(parsed)) {
            return buildFlatSpecSurface(parsed, warnings)
        }

        val embedded = findEmbeddedFlatSpec(parsed) ?: return null
        warnings += "Detected flat-spec IR inside record metadata and rendered the embedded payload."
        return buildFlatSpecSurface(embedded, warnings)
    }

    private fun buildFlatSpecSurface(
        parsed: JsonElement,
        warnings: MutableList<String>
    ): SurfaceState? {
        val flatSpec = FlatSpecParser.parse(parsed) ?: return null
        maybeBuildLegacySurfaceFromTextHeavyFlatSpec(flatSpec)?.let { legacySurface ->
            warnings += "Flat-spec contained text-heavy fallback content; rendered with structured text parser."
            return legacySurface
        }
        return SurfaceState(
            surfaceId = "flat_surface",
            rootId = flatSpec.root,
            components = emptyMap(),
            flatSpec = flatSpec
        )
    }

    private fun maybeBuildLegacySurfaceFromTextHeavyFlatSpec(flatSpec: FlatSpec): SurfaceState? {
        val reachableIds = collectReachableFlatElementIds(flatSpec)
        if (reachableIds.isEmpty()) {
            return null
        }

        val supportedTypes = setOf("column", "row", "list", "card", "text", "divider")
        val unsupportedReachable = reachableIds.any { id ->
            val type = flatSpec.elements[id]?.type?.trim()?.lowercase(Locale.US).orEmpty()
            type.isNotBlank() && type !in supportedTypes
        }
        if (unsupportedReachable) {
            return null
        }

        val textElements = reachableIds.mapNotNull { id ->
            val element = flatSpec.elements[id] ?: return@mapNotNull null
            if (!element.type.equals("text", ignoreCase = true)) {
                return@mapNotNull null
            }
            val textValue = element.props["text"]?.toString().orEmpty().trim()
            if (textValue.isBlank()) {
                return@mapNotNull null
            }
            id to element
        }
        if (textElements.size != 1) {
            return null
        }

        val textElement = textElements.first().second
        val textValue = textElement.props["text"]?.toString().orEmpty()
        if (!looksLikeStructuredRichText(textValue)) {
            return null
        }

        val variant = textElement.props["variant"]?.toString()?.trim().orEmpty().ifBlank { "body" }
        val rootId = "root"
        val textId = "text_1"
        val rootComponent = JsonObject().apply {
            addProperty("id", rootId)
            addProperty("component", "Column")
            add("children", JsonArray().apply { add(textId) })
        }
        val textComponent = JsonObject().apply {
            addProperty("id", textId)
            addProperty("component", "Text")
            addProperty("variant", variant)
            addProperty("text", textValue)
        }
        return SurfaceState(
            surfaceId = "flat_surface",
            rootId = rootId,
            components = linkedMapOf(
                rootId to rootComponent,
                textId to textComponent
            ),
            flatSpec = null
        )
    }

    private fun collectReachableFlatElementIds(flatSpec: FlatSpec): Set<String> {
        val visited = linkedSetOf<String>()
        fun walk(id: String) {
            if (!visited.add(id)) {
                return
            }
            val element = flatSpec.elements[id] ?: return
            element.children.forEach(::walk)
        }
        walk(flatSpec.root)
        return visited
    }

    private fun looksLikeStructuredRichText(value: String): Boolean {
        val normalized = value.replace("\r\n", "\n").trim()
        if (normalized.isEmpty()) {
            return false
        }
        val lines = normalized.lines().map { it.trim() }.filter { it.isNotEmpty() }
        if (lines.size < 2) {
            return false
        }
        return normalized.contains("\n\n") ||
            lines.any { line ->
                line.startsWith("#") ||
                    line.startsWith("Action:", ignoreCase = true) ||
                    line.startsWith("Source", ignoreCase = true) ||
                    line.startsWith("Media:", ignoreCase = true) ||
                    line.startsWith("Tags:", ignoreCase = true) ||
                    line.startsWith("- ") ||
                    line.startsWith("* ") ||
                    (line.contains('|') && line.count { it == '|' } >= 2)
            }
    }

    private fun findEmbeddedFlatSpec(
        element: JsonElement?,
        depth: Int = 0
    ): JsonElement? {
        if (element == null || element.isJsonNull || depth > 4) {
            return null
        }
        if (FlatSpecParser.isFlatSpec(element)) {
            return element
        }

        if (element.isJsonArray) {
            element.asJsonArray.forEach { child ->
                findEmbeddedFlatSpec(coerceEmbeddedJsonElement(child), depth + 1)?.let { return it }
            }
            return null
        }

        if (!element.isJsonObject) {
            return null
        }

        val obj = element.asJsonObject
        flatSpecWrapperKeys.forEach { key ->
            findEmbeddedFlatSpec(coerceEmbeddedJsonElement(obj.get(key)), depth + 1)?.let { return it }
        }
        return null
    }

    private fun coerceEmbeddedJsonElement(element: JsonElement?): JsonElement? {
        if (element == null || element.isJsonNull) {
            return null
        }
        if (!element.isJsonPrimitive || !element.asJsonPrimitive.isString) {
            return element
        }
        val raw = element.asString.trim()
        if (raw.isEmpty() || (!raw.startsWith("{") && !raw.startsWith("["))) {
            return null
        }
        return runCatching { JsonParser.parseString(raw) }.getOrNull()
    }

    @Composable
    fun Render(
        result: RenderResult,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit,
        modifier: Modifier = Modifier
    ) {
        if (result.errorMessage != null) {
            Surface(
                modifier = modifier.fillMaxSize(),
                color = genUiCardContainerColor(GenUiCardTone.Error),
                shape = RoundedCornerShape(GenUiTokens.RadiusXl)
            ) {
                Text(
                    text = result.errorMessage,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(16.dp)
                )
            }
            return
        }

        val (visibleSurfaces, runtimeMessage, onRuntimeAction) = rememberRuntimeState(result.surfaces)

        LazyColumn(
            modifier = modifier
                .fillMaxSize()
                .imePadding(),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            contentPadding = PaddingValues(bottom = 20.dp)
        ) {
            if (!runtimeMessage.isNullOrBlank()) {
                item("runtime_message") {
                    RuntimeMessageBanner(runtimeMessage)
                }
            }
            items(items = visibleSurfaces, key = { it.surfaceId }) { surface ->
                SurfaceCard(
                    surface = surface,
                    sourceDir = sourceDir,
                    onOpenExternalUrl = onOpenExternalUrl,
                    onRuntimeAction = onRuntimeAction
                )
            }
        }
    }

    @Composable
    fun RenderInline(
        result: RenderResult,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit,
        modifier: Modifier = Modifier
    ) {
        if (result.errorMessage != null) {
            Surface(
                modifier = modifier.fillMaxWidth(),
                color = genUiCardContainerColor(GenUiCardTone.Error),
                shape = RoundedCornerShape(GenUiTokens.RadiusXl)
            ) {
                Text(
                    text = result.errorMessage,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(16.dp)
                )
            }
            return
        }

        val (visibleSurfaces, runtimeMessage, onRuntimeAction) = rememberRuntimeState(result.surfaces)

        Column(
            modifier = modifier
                .fillMaxWidth()
                .imePadding(),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            if (!runtimeMessage.isNullOrBlank()) {
                RuntimeMessageBanner(runtimeMessage)
            }
            visibleSurfaces.forEach { surface ->
                SurfaceCard(
                    surface = surface,
                    sourceDir = sourceDir,
                    onOpenExternalUrl = onOpenExternalUrl,
                    onRuntimeAction = onRuntimeAction
                )
            }
        }
    }

    @Composable
    private fun rememberRuntimeState(
        surfaces: List<SurfaceState>
    ): Triple<List<SurfaceState>, String?, (NativeActionParsing.RuntimeAction) -> Unit> {
        val context = LocalContext.current
        val surfaceIds = remember(surfaces) { surfaces.map { it.surfaceId }.toSet() }
        var focusedSurfaceId by remember(surfaces) { mutableStateOf<String?>(null) }
        var runtimeMessage by remember(surfaces) { mutableStateOf<String?>(null) }

        LaunchedEffect(surfaceIds, focusedSurfaceId) {
            if (!focusedSurfaceId.isNullOrBlank() && focusedSurfaceId !in surfaceIds) {
                focusedSurfaceId = null
            }
        }

        val visibleSurfaces = if (focusedSurfaceId.isNullOrBlank()) {
            surfaces
        } else {
            surfaces.filter { it.surfaceId == focusedSurfaceId }
        }

        val onRuntimeAction: (NativeActionParsing.RuntimeAction) -> Unit = { action ->
            when (action) {
                is NativeActionParsing.RuntimeAction.ShowMessage -> {
                    val message = sanitizeDisplayText(action.message)
                    if (message.isNotBlank()) {
                        runtimeMessage = message
                        Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
                    }
                }

                is NativeActionParsing.RuntimeAction.ShowSurface -> {
                    val targetSurface = action.surfaceId.trim()
                    if (targetSurface.isNotBlank()) {
                        if (targetSurface in surfaceIds) {
                            focusedSurfaceId = targetSurface
                            runtimeMessage = "Showing surface: $targetSurface"
                        } else {
                            val message = "Surface not found: $targetSurface"
                            runtimeMessage = message
                            Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            }
        }

        return Triple(visibleSurfaces, runtimeMessage, onRuntimeAction)
    }

    @Composable
    private fun RuntimeMessageBanner(message: String) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(GenUiTokens.RadiusLg),
            colors = genUiCardColors(GenUiCardTone.Primary),
            border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
        ) {
            Text(
                text = message,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp)
            )
        }
    }

    @Composable
    private fun SurfaceCard(
        surface: SurfaceState,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit,
        onRuntimeAction: (NativeActionParsing.RuntimeAction) -> Unit
    ) {
        // Phase 2+: flat spec format — use new renderer
        if (surface.flatSpec != null) {
            Card(
                shape = RoundedCornerShape(GenUiTokens.RadiusXl),
                colors = genUiCardColors(GenUiCardTone.Neutral),
                elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationMd),
                border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor()),
                modifier = Modifier.fillMaxWidth()
            ) {
                FlatSpecContent(
                    spec = surface.flatSpec,
                    resolveAssetUrl = { raw -> NativePayloadParser.resolveAssetUrl(raw, sourceDir) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 8.dp)
                )
            }
            return
        }

        // Legacy format
        Card(
            shape = RoundedCornerShape(GenUiTokens.RadiusXl),
            colors = genUiCardColors(GenUiCardTone.Neutral),
            elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationMd),
            border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor()),
            modifier = Modifier
                .fillMaxWidth()
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                RenderComponent(
                    id = surface.rootId,
                    index = surface.components,
                    sourceDir = sourceDir,
                    onOpenExternalUrl = onOpenExternalUrl,
                    onRuntimeAction = onRuntimeAction,
                    activePath = emptySet()
                )
            }
        }
    }

    @Composable
    private fun RenderComponent(
        id: String,
        index: Map<String, JsonObject>,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit,
        onRuntimeAction: (NativeActionParsing.RuntimeAction) -> Unit,
        activePath: Set<String>
    ) {
        val component = index[id]
        if (component == null) {
            Text("Missing component: $id", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
            return
        }
        if (id in activePath) {
            Text("Cycle detected at $id", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
            return
        }

        val nextPath = activePath + id
        when (component.getString("component")?.trim()) {
            "Column" -> RenderContainer(component, index, sourceDir, onOpenExternalUrl, onRuntimeAction, nextPath, false, false)
            "Row" -> RenderContainer(component, index, sourceDir, onOpenExternalUrl, onRuntimeAction, nextPath, true, false)
            "List" -> {
                val isHorizontal = component.getString("direction")?.normalizeLayoutToken()?.contains("horizontal") == true
                RenderContainer(component, index, sourceDir, onOpenExternalUrl, onRuntimeAction, nextPath, isHorizontal, true)
            }

            "Card" -> RenderCardComponent(component, index, sourceDir, onOpenExternalUrl, onRuntimeAction, nextPath)
            "Text" -> RenderTextComponent(component, sourceDir, onOpenExternalUrl, onRuntimeAction)
            "Image" -> RenderImageComponent(component, sourceDir, onOpenExternalUrl, onRuntimeAction)
            "Icon" -> RenderIconComponent(component, sourceDir, onOpenExternalUrl, onRuntimeAction)
            "Video" -> RenderVideoComponent(component, sourceDir, onOpenExternalUrl)
            "AudioPlayer" -> RenderAudioPlayerComponent(component, sourceDir, onOpenExternalUrl)
            "Table" -> RenderTableComponent(component)
            "Divider" -> HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            "Button" -> RenderButtonComponent(component, index, sourceDir, onOpenExternalUrl, onRuntimeAction)
            "Tabs" -> RenderTabsComponent(component, index, sourceDir, onOpenExternalUrl, onRuntimeAction, nextPath)
            "Modal" -> NativeFormComponents.RenderModalComponent(
                component, index, sourceDir, onOpenExternalUrl, onRuntimeAction, nextPath
            ) { id, idx, sd, oeu, ora, ap -> RenderComponent(id, idx, sd, oeu, ora, ap) }
            "TextField" -> NativeFormComponents.RenderTextFieldComponent(component)
            "CheckBox" -> NativeFormComponents.RenderCheckBoxComponent(component)
            "ChoicePicker" -> NativeFormComponents.RenderChoicePickerComponent(component)
            "Slider" -> NativeFormComponents.RenderSliderComponent(component)
            "DateTimeInput" -> NativeFormComponents.RenderDateTimeInputComponent(component)
            else -> Text(
                text = "Unsupported component: ${component.getString("component") ?: "Unknown"}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error
            )
        }
    }

    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    private fun RenderContainer(
        component: JsonObject,
        index: Map<String, JsonObject>,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit,
        onRuntimeAction: (NativeActionParsing.RuntimeAction) -> Unit,
        activePath: Set<String>,
        rowLayout: Boolean,
        listComponent: Boolean
    ) {
        val children = readChildren(component)
        if (children.isEmpty()) {
            return
        }

        if (!rowLayout) {
            maybeExtractCardRowTable(children, index)?.let {
                RenderTableSpec(it)
                return
            }

            maybeExtractTable(children, index)?.let {
                RenderTableSpec(it)
                return
            }

            if (listComponent) {
                maybeExtractTextList(children, index)?.let { items ->
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        items.forEach { item ->
                            MarkdownText(
                                text = "\u2022 ${item.trim()}",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurface,
                                modifier = Modifier.fillMaxWidth()
                            )
                        }
                    }
                    return
                }
            }
        }

        val componentAction = NativeActionParsing.extractComponentAction(component)
        val actionModifier = NativeActionParsing.componentActionModifier(
            action = componentAction,
            sourceDir = sourceDir,
            onOpenExternalUrl = onOpenExternalUrl,
            onRuntimeAction = onRuntimeAction
        )
        val alignToken = component.getString("align")?.normalizeLayoutToken()

        // Chip row heuristic: a Row whose children are ALL short Text components → render as chip tags
        if (rowLayout) {
            val childObjs = children.mapNotNull { index[it] }
            val allShortTexts = childObjs.size in 1..8 &&
                childObjs.all { child ->
                    child.getString("component") == "Text" &&
                        child.getString("variant") == "chip" ||
                        (child.getString("component") == "Text" &&
                            (readDynamicString(child.get("text")).let { t ->
                                t.isNotBlank() && t.length <= 40 && !t.contains("http") &&
                                    !t.contains("://") && !t.contains("•")
                            }))
                }
            if (allShortTexts) {
                FlowRow(
                    modifier = actionModifier,
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    childObjs.forEach { child ->
                        val tagText = readDynamicString(child.get("text")).trim()
                        if (tagText.isNotBlank()) {
                            Surface(
                                shape = RoundedCornerShape(50),
                                color = MaterialTheme.colorScheme.secondaryContainer,
                                contentColor = MaterialTheme.colorScheme.onSecondaryContainer
                            ) {
                                Text(
                                    text = tagText,
                                    style = MaterialTheme.typography.labelSmall,
                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                )
                            }
                        }
                    }
                }
                return
            }
        }

        if (rowLayout) {
            Box(modifier = actionModifier) {
                RenderRowChildren(
                    children = children,
                    index = index,
                    sourceDir = sourceDir,
                    onOpenExternalUrl = onOpenExternalUrl,
                    onRuntimeAction = onRuntimeAction,
                    activePath = activePath,
                    alignToken = alignToken,
                    justifyToken = component.getString("justify")?.normalizeLayoutToken()
                )
            }
            return
        }

        val horizontalAlignment = when (alignToken) {
            "center", "middle" -> Alignment.CenterHorizontally
            "end", "flexend", "right", "bottom" -> Alignment.End
            else -> Alignment.Start
        }

        Column(
            modifier = NativeActionParsing.componentActionModifier(
                base = Modifier.fillMaxWidth(),
                action = componentAction,
                sourceDir = sourceDir,
                onOpenExternalUrl = onOpenExternalUrl,
                onRuntimeAction = onRuntimeAction
            ),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            horizontalAlignment = horizontalAlignment
        ) {
            var cursor = 0
            while (cursor < children.size) {
                val tableRun = maybeExtractTableRun(children, cursor, index)
                if (tableRun != null) {
                    val (tableSpec, consumed) = tableRun
                    RenderTableSpec(tableSpec)
                    cursor += consumed
                    continue
                }

                val childId = children[cursor]
                RenderComponent(childId, index, sourceDir, onOpenExternalUrl, onRuntimeAction, activePath)
                cursor++
            }
        }
    }

    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    private fun RenderRowChildren(
        children: List<String>,
        index: Map<String, JsonObject>,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit,
        onRuntimeAction: (NativeActionParsing.RuntimeAction) -> Unit,
        activePath: Set<String>,
        alignToken: String?,
        justifyToken: String?
    ) {
        parseInlineBulletRow(children, index)?.let { bulletRow ->
            MarkdownText(
                text = "${bulletRow.bullet} ${bulletRow.text}",
                style = NativeVariantStyleResolver.textStyleForVariant(bulletRow.variant),
                color = NativeVariantStyleResolver.textColorForVariant(bulletRow.variant),
                modifier = Modifier.fillMaxWidth()
            )
            return
        }

        val hasWeight = children.any { (index[it]?.getAsNumberOrNull("weight") ?: 0.0) > 0.0 }
        val verticalAlignment = when (alignToken) {
            "center", "middle" -> Alignment.CenterVertically
            "end", "flexend", "bottom" -> Alignment.Bottom
            else -> Alignment.Top
        }

        if (hasWeight) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = verticalAlignment
            ) {
                children.forEach { childId ->
                    val weight = (index[childId]?.getAsNumberOrNull("weight") ?: 0.0).toFloat().coerceAtLeast(0f)
                    Box(modifier = if (weight > 0f) Modifier.weight(weight) else Modifier) {
                        RenderComponent(childId, index, sourceDir, onOpenExternalUrl, onRuntimeAction, activePath)
                    }
                }
            }
            return
        }

        val horizontalArrangement = when (justifyToken) {
            "center" -> Arrangement.Center
            "end", "flexend", "right" -> Arrangement.End
            "between", "spacebetween" -> Arrangement.SpaceBetween
            "around", "spacearound" -> Arrangement.SpaceAround
            "evenly", "spaceevenly" -> Arrangement.SpaceEvenly
            else -> Arrangement.Start
        }
        val allButtons = children.all { childId ->
            index[childId]?.getString("component")?.equals("Button", ignoreCase = true) == true
        }
        val requiresFullWidth =
            justifyToken in setOf("between", "spacebetween", "around", "spacearound", "evenly", "spaceevenly")

        if (allButtons && children.size > 1) {
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = horizontalArrangement,
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                children.forEach { childId ->
                    RenderComponent(childId, index, sourceDir, onOpenExternalUrl, onRuntimeAction, activePath)
                }
            }
            return
        }

        Row(
            modifier = if (requiresFullWidth) Modifier.fillMaxWidth() else Modifier,
            horizontalArrangement = horizontalArrangement,
            verticalAlignment = verticalAlignment
        ) {
            children.forEachIndexed { childIndex, childId ->
                RenderComponent(childId, index, sourceDir, onOpenExternalUrl, onRuntimeAction, activePath)
                if (childIndex != children.lastIndex && horizontalArrangement == Arrangement.Start) {
                    Box(modifier = Modifier.width(10.dp))
                }
            }
        }
    }

    private fun parseInlineBulletRow(
        children: List<String>,
        index: Map<String, JsonObject>
    ): InlineBulletRow? {
        if (children.size != 2) {
            return null
        }

        val bulletNode = index[children[0]] ?: return null
        val textNode = index[children[1]] ?: return null
        if (bulletNode.getString("component") != "Text" || textNode.getString("component") != "Text") {
            return null
        }

        val bulletToken = readDynamicString(bulletNode.get("text")).trim()
        if (!bulletToken.matches(Regex("""^[\u2022\u25CF\u25E6\u00B7\u25AA*-]$"""))) {
            return null
        }

        val text = readDynamicString(textNode.get("text")).trim()
        if (text.isBlank()) {
            return null
        }

        val variant = (textNode.getString("variant") ?: "body").lowercase(Locale.US)
        val normalizedBullet = if (bulletToken == "-") "\u2022" else bulletToken
        return InlineBulletRow(
            bullet = normalizedBullet,
            text = text,
            variant = variant
        )
    }

    @Composable
    private fun RenderCardComponent(
        component: JsonObject,
        index: Map<String, JsonObject>,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit,
        onRuntimeAction: (NativeActionParsing.RuntimeAction) -> Unit,
        activePath: Set<String>
    ) {
        val componentAction = NativeActionParsing.extractComponentAction(component)
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .then(
                    NativeActionParsing.componentActionModifier(
                        action = componentAction,
                        sourceDir = sourceDir,
                        onOpenExternalUrl = onOpenExternalUrl,
                        onRuntimeAction = onRuntimeAction
                    )
                ),
            shape = RoundedCornerShape(GenUiTokens.RadiusLg),
            colors = genUiCardColors(GenUiCardTone.Neutral),
            elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
            border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                val childId = component.getString("child")
                if (childId != null) {
                    RenderComponent(childId, index, sourceDir, onOpenExternalUrl, onRuntimeAction, activePath)
                } else {
                    readChildren(component).forEach { child ->
                        RenderComponent(child, index, sourceDir, onOpenExternalUrl, onRuntimeAction, activePath)
                    }
                }
            }
        }
    }

    @Composable
    private fun RenderTextComponent(
        component: JsonObject,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit,
        onRuntimeAction: (NativeActionParsing.RuntimeAction) -> Unit
    ) {
        val variant = (component.getString("variant") ?: "body").lowercase(Locale.US)
        val rawText = readDynamicString(component.get("text"))
        val style = NativeVariantStyleResolver.textStyleForVariant(variant)
        val color = NativeVariantStyleResolver.textColorForVariant(variant)
        val actionModifier = NativeActionParsing.componentActionModifier(
            action = NativeActionParsing.extractComponentAction(component),
            sourceDir = sourceDir,
            onOpenExternalUrl = onOpenExternalUrl,
            onRuntimeAction = onRuntimeAction
        )

        if (rawText.isBlank()) {
            Text("", style = style, modifier = actionModifier)
            return
        }

        // Chip variant: render as a pill-shaped tag
        if (variant == "chip") {
            Surface(
                shape = RoundedCornerShape(50),
                color = MaterialTheme.colorScheme.secondaryContainer,
                contentColor = MaterialTheme.colorScheme.onSecondaryContainer,
                modifier = actionModifier
            ) {
                Text(
                    text = rawText,
                    style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                )
            }
            return
        }

        if (shouldUseStructuredBlocks(rawText, variant)) {
            val blocks = remember(rawText) { parseTextBlocks(rawText) }
            Column(modifier = actionModifier) {
                RenderTextBlocks(blocks, sourceDir, onOpenExternalUrl, style, color)
            }
            return
        }

        MarkdownText(
            text = rawText,
            style = style,
            color = color,
            modifier = actionModifier
        )
    }

    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    private fun RenderTextBlocks(
        blocks: List<TextBlock>,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit,
        baseStyle: TextStyle,
        baseColor: Color
    ) {
        val paragraphStyle = baseStyle.copy(lineHeight = baseStyle.fontSize * 1.5f)

        @Composable
        fun RenderSingleBlock(block: TextBlock) {
            when (block) {
                is TextBlock.Title -> MarkdownText(
                    text = block.text,
                    style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
                    color = MaterialTheme.colorScheme.onSurface
                )

                is TextBlock.Heading -> MarkdownText(
                    text = block.text,
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                    color = MaterialTheme.colorScheme.onSurface
                )

                is TextBlock.Paragraph -> MarkdownText(
                    text = block.text,
                    style = paragraphStyle,
                    color = baseColor
                )

                is TextBlock.Bullets -> {
                    val visibleItems = block.items.filterNot(::isPlaceholderListEntry)
                    if (visibleItems.isNotEmpty()) {
                        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            visibleItems.forEach { item ->
                                RenderListItem(item = "\u2022 ${item.trim()}", sourceDir = sourceDir)
                            }
                        }
                    }
                }

                is TextBlock.NumberedSteps -> {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        block.items.forEach { step ->
                            Card(
                                modifier = Modifier
                                    .fillMaxWidth(),
                                shape = RoundedCornerShape(GenUiTokens.RadiusLg),
                                colors = genUiCardColors(GenUiCardTone.Neutral),
                                elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                                border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                            ) {
                                Column(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(horizontal = 12.dp, vertical = 10.dp),
                                    verticalArrangement = Arrangement.spacedBy(6.dp)
                                ) {
                                    MarkdownText(
                                        text = step.title,
                                        style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                    if (step.details.isNotBlank()) {
                                        MarkdownText(
                                            text = step.details,
                                            style = paragraphStyle,
                                            color = baseColor
                                        )
                                    }
                                }
                            }
                        }
                    }
                }

                is TextBlock.Table -> RenderTextTable(block.rows)

                is TextBlock.Sources -> {
                    RenderSourceLinks(
                        links = block.links,
                        sourceDir = sourceDir,
                        onOpenExternalUrl = onOpenExternalUrl
                    )
                }

                is TextBlock.Actions -> {
                    FlowRow(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        block.actions.forEach { action ->
                            val resolved = resolveExternalUrl(action.url, sourceDir)
                            ExternalActionButton(action.label, resolved, onOpenExternalUrl)
                        }
                    }
                }

                is TextBlock.BookingCards -> {
                    RenderBookingCards(
                        options = block.options,
                        sourceDir = sourceDir,
                        onOpenExternalUrl = onOpenExternalUrl
                    )
                }

                is TextBlock.MediaCards -> {
                    val renderableEntries = block.entries.filterNot(::isPlaceholderMediaEntry)
                    if (renderableEntries.isNotEmpty()) {
                        RenderMediaCards(entries = renderableEntries, sourceDir = sourceDir)
                    }
                }

                is TextBlock.TagRow -> {
                    FlowRow(
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        block.tags.forEach { tag ->
                            Surface(
                                shape = RoundedCornerShape(50),
                                color = MaterialTheme.colorScheme.secondaryContainer,
                                contentColor = MaterialTheme.colorScheme.onSecondaryContainer
                            ) {
                                Text(
                                    text = tag,
                                    style = MaterialTheme.typography.labelSmall,
                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                )
                            }
                        }
                    }
                }
            }
        }

        fun isSimpleSectionBlock(block: TextBlock): Boolean = when (block) {
            is TextBlock.Paragraph,
            is TextBlock.Bullets,
            is TextBlock.Actions,
            is TextBlock.Sources,
            is TextBlock.MediaCards,
            is TextBlock.TagRow -> true
            else -> false
        }

        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            var index = 0
            while (index < blocks.size) {
                val block = blocks[index]
                val headingText = when (block) {
                    is TextBlock.Title -> block.text
                    is TextBlock.Heading -> block.text
                    else -> null
                }
                if (headingText != null && NativeTextBlockSemantics.isBoilerplateContextHeading(headingText)) {
                    val nextBlock = blocks.getOrNull(index + 1)
                    if (nextBlock is TextBlock.Paragraph && NativeTextBlockSemantics.isBoilerplateContextParagraph(nextBlock.text)) {
                        index += 2
                    } else {
                        index += 1
                    }
                    continue
                }

                if (block is TextBlock.Title || block is TextBlock.Heading) {
                    val sectionBlocks = mutableListOf<TextBlock>()
                    var cursor = index + 1
                    while (cursor < blocks.size) {
                        val candidate = blocks[cursor]
                        if (candidate is TextBlock.Title || candidate is TextBlock.Heading) {
                            break
                        }
                        if (!isSimpleSectionBlock(candidate)) {
                            break
                        }
                        sectionBlocks += candidate
                        cursor++
                    }

                    val titleText = when (block) {
                        is TextBlock.Title -> block.text
                        is TextBlock.Heading -> block.text
                        else -> ""
                    }
                    val isFlightLikeSection = looksLikeFlightSection(titleText, sectionBlocks)
                    val fallbackWeatherCondition = if (isFlightLikeSection) {
                        null
                    } else {
                        NativeWeatherSemantics.inferWeatherConditionFromSection(titleText, sectionBlocks)
                    }
                    val titleNormalized = NativeWeatherSemantics.normalizeWeatherText(titleText)
                    val titleHasCurrentMarker =
                        titleNormalized.contains("current") ||
                            titleNormalized.contains("currently") ||
                            titleNormalized.contains("now")
                    val sectionMissingTextBlocks = sectionBlocks.none {
                        it is TextBlock.Paragraph || it is TextBlock.Bullets
                    }
                    val shouldCollectWeatherFollowUp =
                        !isFlightLikeSection &&
                            (
                                NativeWeatherSemantics.isCurrentWeatherHeading(titleText) ||
                            (!fallbackWeatherCondition.isNullOrBlank() &&
                                (titleHasCurrentMarker || sectionMissingTextBlocks))
                                )
                    val (followUpWeatherLines, followUpConsumed) =
                        if (shouldCollectWeatherFollowUp) {
                            NativeWeatherSemantics.collectCurrentWeatherFollowUpLines(blocks, cursor)
                        } else {
                            emptyList<String>() to 0
                        }
                    val currentWeatherDetails = if (isFlightLikeSection) {
                        null
                    } else {
                        NativeWeatherSemantics.buildCurrentWeatherDetails(
                            title = titleText,
                            sectionBlocks = sectionBlocks,
                            fallbackCondition = fallbackWeatherCondition,
                            additionalLines = followUpWeatherLines
                        )
                    }
                    val resolvedCurrentWeatherDetails =
                        currentWeatherDetails ?: if (isFlightLikeSection) {
                            null
                        } else {
                            buildRelaxedCurrentWeatherDetails(
                                title = titleText,
                                sectionBlocks = sectionBlocks,
                                fallbackCondition = fallbackWeatherCondition,
                                additionalLines = followUpWeatherLines
                            ) ?: buildMinimalCurrentWeatherDetails(
                                title = titleText,
                                sectionBlocks = sectionBlocks,
                                fallbackCondition = fallbackWeatherCondition,
                                additionalLines = followUpWeatherLines
                            )
                        }
                    val hasVisualMedia = sectionBlocks.any { section ->
                        section is TextBlock.MediaCards &&
                            section.entries.any { entry ->
                                !entry.iconLike && !isPlaceholderMediaEntry(entry)
                            }
                    }
                    val hasIconGallery = sectionBlocks.any { section ->
                        section is TextBlock.MediaCards &&
                            section.entries.none { entry -> !entry.iconLike && !isPlaceholderMediaEntry(entry) } &&
                            section.entries.count { entry -> entry.iconLike && !isPlaceholderMediaEntry(entry) } >= 4
                    }
                    val shouldRenderSectionCard =
                        sectionBlocks.isNotEmpty() &&
                            (resolvedCurrentWeatherDetails != null || hasVisualMedia || hasIconGallery)
                    if (shouldRenderSectionCard) {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(GenUiTokens.RadiusLg),
                            colors = genUiCardColors(GenUiCardTone.Neutral),
                            elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                            border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                        ) {
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 12.dp, vertical = 12.dp),
                                verticalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                val titleStyle = if (block is TextBlock.Title) {
                                    MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold)
                                } else {
                                    MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold)
                                }
                                if (resolvedCurrentWeatherDetails != null) {
                                    RenderCurrentWeatherDetails(
                                        titleText = titleText,
                                        titleStyle = titleStyle,
                                        details = resolvedCurrentWeatherDetails,
                                        sourceDir = sourceDir
                                    )
                                } else {
                                    if (!fallbackWeatherCondition.isNullOrBlank()) {
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            WeatherConditionIcon(
                                                condition = fallbackWeatherCondition,
                                                size = 20.dp
                                            )
                                            Box(modifier = Modifier.weight(1f)) {
                                                MarkdownText(
                                                    text = titleText,
                                                    style = titleStyle,
                                                    color = MaterialTheme.colorScheme.onSurface
                                                )
                                            }
                                        }
                                    } else {
                                        MarkdownText(
                                            text = titleText,
                                            style = titleStyle,
                                            color = MaterialTheme.colorScheme.onSurface
                                        )
                                    }
                                }
                                sectionBlocks.forEach { sectionBlock ->
                                    if (sectionBlock is TextBlock.Paragraph &&
                                        isPlaceholderInlineToken(sectionBlock.text)
                                    ) {
                                        return@forEach
                                    }
                                    if (resolvedCurrentWeatherDetails != null &&
                                        (sectionBlock is TextBlock.Paragraph || sectionBlock is TextBlock.Bullets)
                                    ) {
                                        return@forEach
                                    }
                                    if (resolvedCurrentWeatherDetails != null &&
                                        sectionBlock is TextBlock.MediaCards &&
                                        sectionBlock.entries.all { it.iconLike || isPlaceholderMediaEntry(it) }
                                    ) {
                                        return@forEach
                                    }
                                    if (NativeWeatherSemantics.isCurrentWeatherHeading(titleText) &&
                                        sectionBlock is TextBlock.MediaCards &&
                                        sectionBlock.entries.all { it.iconLike || isPlaceholderMediaEntry(it) }
                                    ) {
                                        return@forEach
                                    }
                                    when (sectionBlock) {
                                        is TextBlock.Bullets -> {
                                            val sanitizedItems =
                                                sectionBlock.items.filterNot(::isPlaceholderInlineToken)
                                            if (sanitizedItems.isNotEmpty()) {
                                                RenderSingleBlock(TextBlock.Bullets(sanitizedItems))
                                            }
                                        }
                                        is TextBlock.MediaCards -> {
                                            val renderableEntries =
                                                sectionBlock.entries.filterNot(::isPlaceholderMediaEntry)
                                            if (renderableEntries.isNotEmpty()) {
                                                RenderMediaEntriesInline(
                                                    entries = renderableEntries,
                                                    sourceDir = sourceDir
                                                )
                                            }
                                        }

                                        else -> RenderSingleBlock(sectionBlock)
                                    }
                                }
                            }
                        }
                        index = cursor + followUpConsumed
                        continue
                    }
                }

                val titleParagraph = if (block is TextBlock.Title) {
                    blocks.getOrNull(index + 1) as? TextBlock.Paragraph
                } else {
                    null
                }
                if (titleParagraph != null && titleParagraph.text.length >= 180) {
                    val titleText = (block as TextBlock.Title).text
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        MarkdownText(
                            text = titleText,
                            style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
                            color = MaterialTheme.colorScheme.onSurface
                        )
                        MarkdownText(
                            text = titleParagraph.text,
                            style = paragraphStyle,
                            color = baseColor
                        )
                    }
                    index += 2
                    continue
                }

                if (block is TextBlock.Heading) {
                    val sectionBlocks = mutableListOf<TextBlock>()
                    var cursor = index + 1
                    sectionLoop@ while (cursor < blocks.size) {
                        when (val candidate = blocks[cursor]) {
                            is TextBlock.Paragraph -> {
                                sectionBlocks += candidate
                                cursor++
                            }

                            is TextBlock.Bullets -> {
                                sectionBlocks += candidate
                                cursor++
                            }

                            else -> break@sectionLoop
                        }
                    }
                    val hasLongParagraph =
                        sectionBlocks.any { it is TextBlock.Paragraph && it.text.length >= 120 }

                    if (sectionBlocks.isNotEmpty() && hasLongParagraph) {
                        Column(
                            modifier = Modifier.fillMaxWidth(),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            MarkdownText(
                                text = block.text,
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            sectionBlocks.forEach { sectionBlock ->
                                RenderSingleBlock(sectionBlock)
                            }
                        }
                        index = cursor
                        continue
                    }
                }

                RenderSingleBlock(block)
                index++
            }
        }
    }

    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    private fun RenderMediaEntriesInline(entries: List<ParsedMediaEntry>, sourceDir: File?) {
        if (entries.isEmpty()) {
            return
        }

        val primary = entries.filterNot { it.iconLike }
        val icons = entries.filter { it.iconLike }
        if (primary.isEmpty()) {
            val labeledIcons = icons.filter { shouldShowMediaLabel(it.label) }
            if (labeledIcons.isEmpty()) {
                return
            }
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                labeledIcons.forEach { icon ->
                    Row(
                        modifier = Modifier
                            .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                            .background(genUiCardContainerColor(GenUiCardTone.Neutral))
                            .border(
                                GenUiTokens.BorderMd,
                                genUiCardBorderColor(),
                                RoundedCornerShape(GenUiTokens.RadiusPill)
                            )
                            .padding(horizontal = 10.dp, vertical = 6.dp),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        MediaImage(
                            rawUrl = icon.url,
                            sourceDir = sourceDir,
                            modifier = Modifier.size(16.dp),
                            contentScale = ContentScale.Fit,
                            asIcon = true
                        )
                        Text(
                            text = sanitizeDisplayText(icon.label),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
            return
        }

        val iconBuckets = NativeTextBlockSemantics.assignIconsToPrimary(primary, icons)
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            primary.forEachIndexed { index, entry ->
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    MediaImage(
                        rawUrl = entry.url,
                        sourceDir = sourceDir,
                        modifier = mediaFrameModifier(),
                        contentScale = defaultImageScale(
                            rawUrl = entry.url,
                            fitValue = null,
                            defaultCoverForRaster = true
                        ),
                        asIcon = false
                    )

                    if (shouldShowMediaLabel(entry.label)) {
                        Text(
                            text = sanitizeDisplayText(entry.label),
                            style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold),
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }

                    val inlineIcons = iconBuckets.getOrElse(index) { emptyList() }
                    if (inlineIcons.isNotEmpty()) {
                        FlowRow(
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            inlineIcons.forEach { icon ->
                                Row(
                                    modifier = Modifier
                                        .sizeIn(maxWidth = 170.dp)
                                        .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                                        .background(genUiCardContainerColor(GenUiCardTone.Neutral))
                                        .border(
                                            GenUiTokens.BorderMd,
                                            genUiCardBorderColor(),
                                            RoundedCornerShape(GenUiTokens.RadiusPill)
                                        )
                                        .padding(horizontal = 8.dp, vertical = 5.dp),
                                    horizontalArrangement = Arrangement.spacedBy(5.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Box(
                                        modifier = Modifier.size(16.dp),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        MediaImage(
                                            rawUrl = icon.url,
                                            sourceDir = sourceDir,
                                            modifier = Modifier.size(14.dp),
                                            contentScale = ContentScale.Fit,
                                            asIcon = true
                                        )
                                    }
                                    if (shouldShowMediaLabel(icon.label)) {
                                        Text(
                                            text = sanitizeDisplayText(icon.label),
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            maxLines = 1,
                                            overflow = TextOverflow.Clip
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Composable
    private fun ExternalActionButton(
        label: String,
        url: String?,
        onOpenExternalUrl: (String) -> Unit
    ) {
        val displayLabel = sanitizeDisplayText(label).ifBlank { "Open" }
        Button(
            onClick = { url?.let(onOpenExternalUrl) },
            enabled = url != null,
            shape = RoundedCornerShape(GenUiTokens.RadiusPill)
        ) {
            Text(displayLabel, style = MaterialTheme.typography.labelLarge)
        }
    }

    @Composable
    private fun RenderListItem(item: String, sourceDir: File?) {
        val leadingLabel = parseLeadingLabelValue(item)
        if (leadingLabel != null) {
            val prefix = leadingLabel.prefix
            val label = leadingLabel.label
            val delimiter = leadingLabel.delimiter
            val value = leadingLabel.value
            if (looksLikeImagePath(value, labelHint = label)) {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Text(
                        text = sanitizeDisplayText("$prefix$label$delimiter"),
                        style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold),
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    MediaImage(
                        rawUrl = value,
                        sourceDir = sourceDir,
                        modifier = mediaFrameModifier(),
                        contentScale = defaultImageScale(
                            rawUrl = value,
                            fitValue = null,
                            defaultCoverForRaster = true
                        ),
                        asIcon = false
                    )
                }
                return
            }

            MarkdownText(
                text = item,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier.fillMaxWidth()
            )
            return
        }

        MarkdownText(
            text = item,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface,
            modifier = Modifier.fillMaxWidth()
        )
    }

    @Composable
    private fun RenderSourceLinks(
        links: List<ParsedButton>,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit
    ) {
        if (links.isEmpty()) {
            return
        }
        val dedupedLinks = links.distinctBy { canonicalSourceUrlToken(it.url) }

        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            MarkdownText(
                text = "Sources",
                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                color = MaterialTheme.colorScheme.onSurface
            )
            dedupedLinks.forEach { link ->
                val resolved = resolveExternalUrl(link.url, sourceDir)
                OutlinedButton(
                    onClick = { resolved?.let(onOpenExternalUrl) },
                    enabled = resolved != null,
                    shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                    border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                ) {
                    MarkdownText(
                        text = link.label,
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }
        }
    }

    @Composable
    private fun MarkdownText(
        text: String,
        style: TextStyle,
        color: Color,
        modifier: Modifier = Modifier,
        maxLines: Int = Int.MAX_VALUE,
        overflow: TextOverflow = TextOverflow.Clip
    ) {
        val displayText = remember(text) { sanitizeDisplayText(text, preserveMarkdown = true) }
        if (displayText.isBlank()) {
            return
        }
        val hasMarkdownInline = NativeTextFormatter.containsMarkdownInlineFormatting(displayText)
        val hasMarkdownHeading = NativeTextFormatter.containsMarkdownHeading(displayText)
        Text(
            text = remember(displayText, style) {
                if (hasMarkdownHeading) {
                    NativeTextFormatter.parseMarkdownWithHeadings(displayText, style)
                } else {
                    NativeTextFormatter.parseInlineMarkdown(displayText)
                }
            },
            style = if (hasMarkdownInline || hasMarkdownHeading) {
                style.copy(fontWeight = FontWeight.Normal)
            } else {
                style
            },
            color = color,
            modifier = modifier,
            maxLines = maxLines,
            overflow = overflow
        )
    }

    @Suppress("unused")
    private fun parseBoldMarkdown(text: String): AnnotatedString {
        // Kept for compatibility with existing reflection-based tests/callers.
        return NativeTextFormatter.parseInlineMarkdown(text)
    }

    private fun sanitizeDisplayText(text: String, preserveMarkdown: Boolean = false): String =
        NativeTextFormatter.sanitizeDisplayText(text, preserveMarkdown)

    private fun normalizeMojibakeText(value: String): String =
        NativeTextFormatter.normalizeMojibakeText(value)

    private fun parseLeadingLabelValue(text: String): LeadingLabelValue? {
        return NativeTextFormatter.parseLeadingLabelValue(text)
    }

    @Composable
    private fun RenderBookingCards(
        options: List<BookingOption>,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit
    ) {
        if (options.isEmpty()) {
            return
        }

        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            options.forEach { option ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth(),
                    shape = RoundedCornerShape(GenUiTokens.RadiusLg),
                    colors = genUiCardColors(GenUiCardTone.Primary),
                    elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                    border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            option.logo?.let { logo ->
                                MediaImage(
                                    rawUrl = logo.url,
                                    sourceDir = sourceDir,
                                    modifier = Modifier
                                        .width(84.dp)
                                        .height(24.dp),
                                    contentScale = ContentScale.Fit,
                                    asIcon = false
                                )
                            }

                            Text(
                                text = sanitizeDisplayText(option.title),
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                                color = MaterialTheme.colorScheme.onSurface,
                                modifier = Modifier.weight(1f)
                            )
                        }

                        Text(
                            text = sanitizeDisplayText(option.details),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )

                        option.button?.let { button ->
                            val resolved = resolveExternalUrl(button.url, sourceDir)
                            ExternalActionButton(
                                sanitizeDisplayText(button.label).ifBlank { "Open" },
                                resolved,
                                onOpenExternalUrl
                            )
                        }
                    }
                }
            }
        }
    }

    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    private fun RenderMediaCards(entries: List<ParsedMediaEntry>, sourceDir: File?) {
        if (entries.isEmpty()) {
            return
        }

        val primary = entries.filterNot { it.iconLike }
        val icons = entries.filter { it.iconLike }

        if (primary.isEmpty()) {
            val labeledIcons = icons.filter { shouldShowMediaLabel(it.label) }
            if (labeledIcons.isEmpty()) {
                return
            }
            Card(
                modifier = Modifier
                    .fillMaxWidth(),
                shape = RoundedCornerShape(GenUiTokens.RadiusLg),
                colors = genUiCardColors(GenUiCardTone.Neutral),
                elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
            ) {
                FlowRow(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(12.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    labeledIcons.forEach { icon ->
                        Row(
                            modifier = Modifier
                                .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                                .background(genUiCardContainerColor(GenUiCardTone.Neutral))
                                .border(
                                    GenUiTokens.BorderMd,
                                    genUiCardBorderColor(),
                                    RoundedCornerShape(GenUiTokens.RadiusPill)
                                )
                                .padding(horizontal = 10.dp, vertical = 6.dp),
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            MediaImage(
                                rawUrl = icon.url,
                                sourceDir = sourceDir,
                                modifier = Modifier.size(16.dp),
                                contentScale = ContentScale.Fit,
                                asIcon = true
                            )
                            Text(
                                text = sanitizeDisplayText(icon.label),
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }
            return
        }

        val iconBuckets = NativeTextBlockSemantics.assignIconsToPrimary(primary, icons)

        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            primary.forEachIndexed { index, entry ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth(),
                    shape = RoundedCornerShape(GenUiTokens.RadiusLg),
                    colors = genUiCardColors(GenUiCardTone.Neutral),
                    elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                    border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        MediaImage(
                            rawUrl = entry.url,
                            sourceDir = sourceDir,
                            modifier = mediaFrameModifier(),
                            contentScale = defaultImageScale(
                                rawUrl = entry.url,
                                fitValue = null,
                                defaultCoverForRaster = true
                            ),
                            asIcon = false
                        )
                        if (shouldShowMediaLabel(entry.label)) {
                            Text(
                                text = sanitizeDisplayText(entry.label),
                                style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold),
                                color = MaterialTheme.colorScheme.onSurface
                            )
                        }

                        val inlineIcons = iconBuckets[index]
                        if (inlineIcons.isNotEmpty()) {
                            FlowRow(
                                horizontalArrangement = Arrangement.spacedBy(6.dp),
                                verticalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                inlineIcons.forEach { icon ->
                                    Row(
                                        modifier = Modifier
                                            .sizeIn(maxWidth = 170.dp)
                                            .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                                            .background(genUiCardContainerColor(GenUiCardTone.Neutral))
                                            .border(
                                                GenUiTokens.BorderMd,
                                                genUiCardBorderColor(),
                                                RoundedCornerShape(GenUiTokens.RadiusPill)
                                            )
                                            .padding(horizontal = 8.dp, vertical = 5.dp),
                                        horizontalArrangement = Arrangement.spacedBy(5.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Box(
                                            modifier = Modifier.size(16.dp),
                                            contentAlignment = Alignment.Center
                                        ) {
                                            MediaImage(
                                                rawUrl = icon.url,
                                                sourceDir = sourceDir,
                                                modifier = Modifier.size(14.dp),
                                                contentScale = ContentScale.Fit,
                                                asIcon = true
                                            )
                                        }
                                        if (shouldShowMediaLabel(icon.label)) {
                                            Text(
                                                text = sanitizeDisplayText(icon.label),
                                                style = MaterialTheme.typography.labelSmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                maxLines = 1,
                                                overflow = TextOverflow.Clip
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Composable
    private fun RenderTextTable(rows: List<List<String>>) {
        if (rows.size < 2) {
            return
        }
        val header = rows.first()
        val body = rows.drop(1).filterNot { isPlaceholderTableStringRow(it) }
        if (body.isEmpty()) {
            return
        }

        val intentModel = NativeIntentRegistry.resolve(header, body)
        if (intentModel != null && NativeIntentRegistry.render(model = intentModel)) {
            return
        }

        val columnCount = rows.maxOf { it.size }.coerceAtLeast(2)
        val columnWidths = List(columnCount) { NativeTableSemantics.tableBaseCellWidth(columnCount) }
        val dark = isSystemInDarkTheme()
        val dividerColor = MaterialTheme.colorScheme.outlineVariant.copy(alpha = if (dark) 0.24f else 0.18f)

        Card(
            modifier = Modifier
                .fillMaxWidth(),
            shape = RoundedCornerShape(GenUiTokens.RadiusXl),
            colors = CardDefaults.cardColors(containerColor = genUiTableContainerColor()),
            elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
            border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
            ) {
                RenderPlainTableRow(
                    values = header,
                    columnCount = columnCount,
                    columnWidths = columnWidths,
                    isHeader = true,
                    rowIndex = 0
                )
                HorizontalDivider(color = dividerColor)
                body.forEachIndexed { index, row ->
                    RenderPlainTableRow(
                        values = row,
                        columnCount = columnCount,
                        columnWidths = columnWidths,
                        isHeader = false,
                        rowIndex = index
                    )
                    if (index != body.lastIndex) {
                        HorizontalDivider(color = dividerColor)
                    }
                }
            }
        }
    }

    @Composable
    private fun RenderTableComponent(component: JsonObject) {
        val columns = component.getAsJsonArrayOrNull("columns") ?: return
        if (columns.size() == 0) {
            return
        }

        val headers = mutableListOf<String>()
        val columnValues = mutableListOf<List<String>>()
        columns.forEach { columnElement ->
            val column = columnElement.asJsonObjectOrNull() ?: return@forEach
            headers += readDynamicString(column.get("header")).trim()
            val values = (column.getAsJsonArrayOrNull("data")
                ?: column.getAsJsonArrayOrNull("values")
                ?: JsonArray()).map { value ->
                readDynamicString(value).trim()
            }
            columnValues += values
        }

        if (headers.isEmpty()) {
            return
        }

        val rowCount = columnValues.maxOfOrNull { it.size } ?: 0
        if (rowCount == 0) {
            return
        }

        val rows = mutableListOf<List<String>>()
        rows += headers
        repeat(rowCount) { rowIndex ->
            rows += columnValues.map { values -> values.getOrNull(rowIndex).orEmpty() }
        }

        RenderTextTable(rows)
    }

    @Composable
    private fun RenderImageComponent(
        component: JsonObject,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit,
        onRuntimeAction: (NativeActionParsing.RuntimeAction) -> Unit
    ) {
        val rawUrl = readDynamicString(component.get("url"))
        if (rawUrl.isBlank()) {
            Text("Missing image source", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
            return
        }

        val variant = (component.getString("variant") ?: "").lowercase(Locale.US)
        val fit = component.getString("fit")
        val urlLower = rawUrl.lowercase(Locale.US)
        val likelyLogo = variant.contains("logo") || urlLower.contains("logo")
        val photoLikeImage = isPhotoLikeImageUrl(rawUrl)
        val mediumFeature = variant.contains("mediumfeature")
        val likelyIcon =
            variant.contains("icon") ||
                looksLikeCompactIconUrl(urlLower) ||
                (!likelyLogo &&
                    urlLower.endsWith(".svg") &&
                    !variant.contains("feature") &&
                    !variant.contains("thumbnail"))
        val inlineIconLike = likelyIcon || (mediumFeature && !photoLikeImage)
        val actionModifier = NativeActionParsing.componentActionModifier(
            action = NativeActionParsing.extractComponentAction(component),
            sourceDir = sourceDir,
            onOpenExternalUrl = onOpenExternalUrl,
            onRuntimeAction = onRuntimeAction
        )

        Box(modifier = actionModifier) {
            MediaImage(
                rawUrl = rawUrl,
                sourceDir = sourceDir,
                modifier = when {
                    inlineIconLike -> Modifier.size(28.dp)
                    likelyLogo -> Modifier
                        .width(90.dp)
                        .height(30.dp)
                    mediumFeature && photoLikeImage -> mediaFrameModifier()
                    variant.contains("feature") -> mediaFrameModifier()
                    variant.contains("thumbnail") -> mediaFrameModifier()
                    else -> mediaFrameModifier()
                },
                contentScale = defaultImageScale(
                    rawUrl = rawUrl,
                    fitValue = fit,
                    defaultCoverForRaster = photoLikeImage && !likelyLogo && !inlineIconLike
                ),
                asIcon = inlineIconLike
            )
        }
    }

    @Composable
    private fun mediaFrameModifier(): Modifier {
        val shape = RoundedCornerShape(GenUiTokens.RadiusMd)
        return Modifier
            .fillMaxWidth()
            .aspectRatio(3f / 2f)
            .clip(shape)
            .border(
                GenUiTokens.BorderMd,
                genUiMediaFrameBorderColor(),
                shape
            )
    }

    @Composable
    private fun RenderIconComponent(
        component: JsonObject,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit,
        onRuntimeAction: (NativeActionParsing.RuntimeAction) -> Unit
    ) {
        val raw = readDynamicString(component.get("name"))
        if (raw.isBlank()) {
            Text("Missing icon source", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
            return
        }
        val actionModifier = NativeActionParsing.componentActionModifier(
            action = NativeActionParsing.extractComponentAction(component),
            sourceDir = sourceDir,
            onOpenExternalUrl = onOpenExternalUrl,
            onRuntimeAction = onRuntimeAction
        )

        Box(modifier = actionModifier) {
            MediaImage(
                rawUrl = raw,
                sourceDir = sourceDir,
                modifier = Modifier.size(26.dp),
                contentScale = ContentScale.Fit,
                asIcon = true
            )
        }
    }

    @Composable
    private fun RenderVideoComponent(
        component: JsonObject,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit
    ) {
        val rawUrl = readDynamicString(component.get("url")).trim()
        if (rawUrl.isBlank()) {
            Text("Missing video source", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
            return
        }
        val externalUrl = resolveExternalUrl(rawUrl, sourceDir)
        Card(
            shape = RoundedCornerShape(GenUiTokens.RadiusLg),
            colors = genUiCardColors(GenUiCardTone.Neutral),
            border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "Video",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurface
                )
                OutlinedButton(
                    onClick = { externalUrl?.let(onOpenExternalUrl) },
                    enabled = externalUrl != null,
                    shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                    border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                ) {
                    Text("Open video", style = MaterialTheme.typography.labelLarge)
                }
            }
        }
    }

    @Composable
    private fun RenderAudioPlayerComponent(
        component: JsonObject,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit
    ) {
        val rawUrl = readDynamicString(component.get("url")).trim()
        if (rawUrl.isBlank()) {
            Text("Missing audio source", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
            return
        }
        val description = sanitizeDisplayText(readDynamicString(component.get("description"))).trim()
        val externalUrl = resolveExternalUrl(rawUrl, sourceDir)
        Card(
            shape = RoundedCornerShape(GenUiTokens.RadiusLg),
            colors = genUiCardColors(GenUiCardTone.Neutral),
            border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = if (description.isNotBlank()) description else "Audio",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurface
                )
                OutlinedButton(
                    onClick = { externalUrl?.let(onOpenExternalUrl) },
                    enabled = externalUrl != null,
                    shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                    border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                ) {
                    Text("Play audio", style = MaterialTheme.typography.labelLarge)
                }
            }
        }
    }

    @Composable
    private fun RenderButtonComponent(
        component: JsonObject,
        index: Map<String, JsonObject>,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit,
        onRuntimeAction: (NativeActionParsing.RuntimeAction) -> Unit
    ) {
        val variant = (component.getString("variant") ?: "primary").lowercase(Locale.US)
        val label = component.getString("child")
            ?.let { childId ->
                index[childId]
                    ?.takeIf { it.getString("component") == "Text" }
                    ?.let { readDynamicString(it.get("text")) }
            }
            ?.ifBlank { null }
            ?: "Open"
        val displayLabel = sanitizeDisplayText(label).ifBlank { "Open" }
        val componentAction = NativeActionParsing.extractComponentAction(component)
        val enabled = componentAction != null

        if (variant == "borderless") {
            OutlinedButton(
                onClick = {
                    NativeActionParsing.executeComponentAction(
                        action = componentAction,
                        sourceDir = sourceDir,
                        onOpenExternalUrl = onOpenExternalUrl,
                        onRuntimeAction = onRuntimeAction
                    )
                },
                enabled = enabled,
                shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
            ) {
                Text(displayLabel, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
            }
            return
        }

        Button(
            onClick = {
                NativeActionParsing.executeComponentAction(
                    action = componentAction,
                    sourceDir = sourceDir,
                    onOpenExternalUrl = onOpenExternalUrl,
                    onRuntimeAction = onRuntimeAction
                )
            },
            enabled = enabled,
            shape = RoundedCornerShape(GenUiTokens.RadiusPill)
        ) {
            Text(displayLabel, style = MaterialTheme.typography.labelLarge)
        }
    }

    @Composable
    private fun RenderTabsComponent(
        component: JsonObject,
        index: Map<String, JsonObject>,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit,
        onRuntimeAction: (NativeActionParsing.RuntimeAction) -> Unit,
        activePath: Set<String>
    ) {
        val tabs = component.getAsJsonArrayOrNull("tabs") ?: component.getAsJsonArrayOrNull("items")
        if (tabs == null || tabs.size() == 0) {
            return
        }
        val activeTabId = component.getString("activeTabId")

        data class TabItem(
            val title: String,
            val childId: String?
        )

        val tabItems = tabs.mapIndexed { tabIndex, tabElement ->
            val tab = tabElement.asJsonObjectOrNull()
            TabItem(
                title = sanitizeDisplayText(readDynamicString(tab?.get("title")))
                    .ifBlank { "Tab ${tabIndex + 1}" },
                childId = tab?.getString("child")
            )
        }
        if (tabItems.isEmpty()) {
            return
        }

        val initialIndex = tabItems.indexOfFirst { item -> item.childId == activeTabId }
            .takeIf { it >= 0 }
            ?: 0
        var selectedIndex by remember(component.getString("id"), activeTabId) {
            mutableIntStateOf(initialIndex)
        }
        if (selectedIndex !in tabItems.indices) {
            selectedIndex = initialIndex.coerceIn(tabItems.indices)
        }
        val selectedTab = tabItems[selectedIndex]

        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            ScrollableTabRow(selectedTabIndex = selectedIndex) {
                tabItems.forEachIndexed { index, item ->
                    Tab(
                        selected = selectedIndex == index,
                        onClick = { selectedIndex = index },
                        text = { Text(item.title) }
                    )
                }
            }

            Card(
                modifier = Modifier
                    .fillMaxWidth(),
                shape = RoundedCornerShape(GenUiTokens.RadiusLg),
                colors = genUiCardColors(GenUiCardTone.Primary),
                border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        selectedTab.title,
                        style = MaterialTheme.typography.titleSmall,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    val childId = selectedTab.childId
                    if (childId == null) {
                        Text(
                            "Missing tab child",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error
                        )
                    } else {
                        RenderComponent(childId, index, sourceDir, onOpenExternalUrl, onRuntimeAction, activePath)
                    }
                }
            }
        }
    }

    @Composable
    private fun MediaImage(
        rawUrl: String,
        sourceDir: File?,
        modifier: Modifier,
        contentScale: ContentScale,
        asIcon: Boolean,
        fallbackCondition: String? = null,
        iconFallbackKind: IconFallbackKind = IconFallbackKind.Generic,
        iconFallbackSize: Dp = 18.dp
    ) {
        val context = LocalContext.current
        val model = remember(rawUrl, sourceDir) { resolveImageModel(rawUrl, sourceDir) }
        val imageLoader = remember(context) {
            ImageLoader.Builder(context)
                .components { add(SvgDecoder.Factory()) }
                .build()
        }
        var failed by remember(rawUrl, sourceDir) { mutableStateOf(false) }

        val loadingBg = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = if (isSystemInDarkTheme()) 0.16f else 0.30f)
        val iconTintColor = if (asIcon) iconTintForAsset(rawUrl) else null

        if (failed && !asIcon) {
            // Collapse failed non-icon media instead of keeping an empty placeholder box.
            return
        }

        Box(
            modifier = modifier.background(if (asIcon) Color.Transparent else loadingBg),
            contentAlignment = Alignment.Center
        ) {
            if (failed) {
                if (asIcon) {
                    when (iconFallbackKind) {
                        IconFallbackKind.Weather -> {
                            // Keep weather icon slots informative even when remote icon fetch fails.
                            WeatherConditionIcon(
                                condition = fallbackCondition ?: inferWeatherConditionFromIconUrl(rawUrl) ?: "Cloudy",
                                size = iconFallbackSize
                            )
                        }
                        IconFallbackKind.Generic -> {
                            Icon(
                                imageVector = Icons.Filled.Image,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.size(iconFallbackSize)
                            )
                        }
                    }
                }
            } else {
                AsyncImage(
                    model = ImageRequest.Builder(context).data(model).crossfade(true).build(),
                    imageLoader = imageLoader,
                    contentDescription = null,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = contentScale,
                    colorFilter = iconTintColor?.let { ColorFilter.tint(it) },
                    onSuccess = { failed = false },
                    onError = { failed = true }
                )
            }
        }
    }

    private fun inferWeatherConditionFromIconUrl(rawUrl: String): String? {
        return NativeWeatherSemantics.inferWeatherConditionFromIconUrl(rawUrl)
    }

    @Composable
    private fun iconTintForAsset(rawUrl: String): Color {
        val key = rawUrl.lowercase(Locale.US)
        val scheme = MaterialTheme.colorScheme
        return when {
            key.contains("sun") || key.contains("clear") -> Color(0xFFFFB300)
            key.contains("rain") || key.contains("shower") || key.contains("water") -> Color(0xFF2E65D4)
            key.contains("cloud") || key.contains("fog") || key.contains("mist") -> Color(0xFF7A7A85)
            key.contains("wind") -> Color(0xFF25B871)
            key.contains("snow") || key.contains("ice") -> Color(0xFF82B1FF)
            key.contains("warn") || key.contains("alert") -> Color(0xFFE65B17)
            else -> scheme.primary
        }
    }

    @Composable
    private fun RenderTableSpec(spec: TableSpec) {
        val filteredRows = spec.rows.filterNot { isPlaceholderTableCellRow(it) }
        if (filteredRows.isEmpty()) {
            return
        }
        val normalizedSpec = if (filteredRows.size == spec.rows.size) spec else spec.copy(rows = filteredRows)

        val specHeader = normalizedSpec.header?.map { it.text }
        val specBody = normalizedSpec.rows.map { row -> row.map { it.text } }
        val inferredHeader = specBody.firstOrNull()
        val semanticHeader = specHeader ?: inferredHeader
        val semanticBody = when {
            specHeader != null -> specBody
            inferredHeader != null -> specBody.drop(1)
            else -> emptyList()
        }
        if (semanticHeader != null && semanticBody.isNotEmpty()) {
            val intentModel = NativeIntentRegistry.resolve(semanticHeader, semanticBody)
            if (intentModel != null && NativeIntentRegistry.render(model = intentModel)) {
                return
            }
        }

        val columnCount = listOfNotNull(normalizedSpec.header?.size, normalizedSpec.rows.maxOfOrNull { it.size })
            .maxOrNull()
            ?.coerceAtLeast(2)
            ?: return
        val hasExplicitWeights =
            normalizedSpec.header?.any { abs(it.weight - 1f) > 0.01f } == true ||
                normalizedSpec.rows.any { row -> row.any { abs(it.weight - 1f) > 0.01f } }
        val columnWidths = NativeTableSemantics.tableColumnWidths(normalizedSpec, columnCount, hasExplicitWeights)
        val dark = isSystemInDarkTheme()
        val dividerColor = MaterialTheme.colorScheme.outlineVariant.copy(alpha = if (dark) 0.24f else 0.18f)

        Card(
            modifier = Modifier
                .fillMaxWidth(),
            shape = RoundedCornerShape(GenUiTokens.RadiusXl),
            colors = CardDefaults.cardColors(containerColor = genUiTableContainerColor()),
            elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
            border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
            ) {
                normalizedSpec.header?.let { header ->
                    RenderWeightedRow(
                        cells = header,
                        columnCount = columnCount,
                        columnWidths = columnWidths,
                        isHeader = true,
                        rowIndex = 0
                    )
                    HorizontalDivider(color = dividerColor)
                }
                normalizedSpec.rows.forEachIndexed { index, row ->
                    RenderWeightedRow(
                        cells = row,
                        columnCount = columnCount,
                        columnWidths = columnWidths,
                        isHeader = false,
                        rowIndex = index
                    )
                    if (index != normalizedSpec.rows.lastIndex) {
                        HorizontalDivider(color = dividerColor)
                    }
                }
            }
        }
    }

    @Composable
    private fun RenderWeightedRow(
        cells: List<TableCell>,
        columnCount: Int,
        columnWidths: List<Dp>,
        isHeader: Boolean,
        rowIndex: Int
    ) {
        Row(
            modifier = Modifier
                .wrapContentWidth(align = Alignment.Start)
                .height(IntrinsicSize.Min)
                .background(NativeTableUi.tableRowBackground(isHeader = isHeader, rowIndex = rowIndex))
        ) {
            for (column in 0 until columnCount) {
                val cell = cells.getOrNull(column)
                val displayText = NativeTableSemantics.sanitizeTableCellDisplayValue(cell?.text.orEmpty())
                val style = if (isHeader) {
                    MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold)
                } else {
                    NativeVariantStyleResolver.textStyleForVariant(cell?.variant ?: "body")
                }
                val color = if (isHeader) {
                    MaterialTheme.colorScheme.onSurface
                } else {
                    NativeVariantStyleResolver.textColorForVariant(cell?.variant ?: "body")
                }
                val cellWidth = columnWidths.getOrElse(column) { NativeTableSemantics.tableBaseCellWidth(columnCount) }

                MarkdownText(
                    text = displayText,
                    style = style,
                    color = color,
                    maxLines = if (isHeader) 4 else Int.MAX_VALUE,
                    overflow = TextOverflow.Clip,
                    modifier = Modifier
                        .width(cellWidth)
                        .padding(horizontal = 12.dp, vertical = 10.dp)
                )

                if (column != columnCount - 1) {
                    NativeTableUi.TableCellDivider()
                }
            }
        }
    }

    @Composable
    private fun RenderPlainTableRow(
        values: List<String>,
        columnCount: Int,
        columnWidths: List<Dp>,
        isHeader: Boolean,
        rowIndex: Int
    ) {
        Row(
            modifier = Modifier
                .wrapContentWidth(align = Alignment.Start)
                .height(IntrinsicSize.Min)
                .background(NativeTableUi.tableRowBackground(isHeader = isHeader, rowIndex = rowIndex))
        ) {
            for (column in 0 until columnCount) {
                val value = NativeTableSemantics.sanitizeTableCellDisplayValue(values.getOrNull(column).orEmpty())
                MarkdownText(
                    text = value,
                    style = if (isHeader) {
                        MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold)
                    } else {
                        MaterialTheme.typography.bodySmall
                    },
                    color = MaterialTheme.colorScheme.onSurface,
                    maxLines = if (isHeader) 4 else Int.MAX_VALUE,
                    overflow = TextOverflow.Clip,
                    modifier = Modifier
                        .width(columnWidths.getOrElse(column) { NativeTableSemantics.tableBaseCellWidth(columnCount) })
                        .padding(horizontal = 12.dp, vertical = 10.dp)
                )

                if (column != columnCount - 1) {
                    NativeTableUi.TableCellDivider()
                }
            }
        }
    }

    private fun detectWeatherColumns(header: List<String>): WeatherTableColumns? {
        return NativeWeatherSemantics.detectWeatherColumns(header)
    }

    private fun detectFlightColumns(header: List<String>): FlightTableColumns? {
        return NativeFlightSemantics.detectFlightColumns(header)
    }

    private fun isPlaceholderMediaEntry(entry: ParsedMediaEntry): Boolean {
        val candidate = (entry.url.ifBlank { entry.label }).trim().lowercase(Locale.US)
        if (candidate.isBlank()) {
            return true
        }
        return candidate in setOf(
            "icon",
            "image",
            "url",
            "image_url",
            "icon_url",
            "<icon_url>",
            "<image_url>",
            "<url>",
            "na",
            "n/a",
            "none",
            "null",
            "--"
        ) || candidate.contains("placeholder")
    }

    private fun isPlaceholderInlineToken(value: String): Boolean {
        val normalized = value.trim().trim('\'', '"').lowercase(Locale.US)
        if (normalized.isBlank()) {
            return true
        }
        return normalized in setOf(
            "icon",
            "image",
            "url",
            "image_url",
            "icon_url",
            "<icon_url>",
            "<image_url>",
            "<url>",
            "--",
            "na",
            "n/a",
            "none",
            "null"
        ) || normalized.contains("placeholder")
    }

    private fun looksLikeFlightSection(
        title: String,
        sectionBlocks: List<TextBlock>
    ): Boolean {
        val titleText = sanitizeDisplayText(title)
        val textPool = buildString {
            append(titleText)
            sectionBlocks.forEach { block ->
                when (block) {
                    is TextBlock.Paragraph -> append(' ').append(block.text)
                    is TextBlock.Bullets -> block.items.forEach { append(' ').append(it) }
                    is TextBlock.MediaCards -> block.entries.forEach {
                        append(' ').append(it.label).append(' ').append(it.url)
                    }
                    else -> Unit
                }
            }
        }
        val normalizedTitle = NativeFlightSemantics.normalizeMatchText(titleText)
        val normalizedPool = NativeFlightSemantics.normalizeMatchText(textPool)
        val flightTokens = listOf(
            "flight",
            "airline",
            "fare",
            "price",
            "non stop",
            "stop",
            "departure",
            "arrival",
            "airport",
            "terminal",
            "boarding",
            "layover",
            "ticket",
            "blr",
            "lko"
        )
        val weatherTokens = listOf(
            "weather",
            "forecast",
            "temperature",
            "humidity",
            "wind",
            "cloud",
            "rain",
            "storm",
            "uv"
        )
        val titleFlightSignals = flightTokens.count { normalizedTitle.contains(it) }
        val bodyFlightSignals = flightTokens.count { normalizedPool.contains(it) }
        val weatherSignals = weatherTokens.count { normalizedPool.contains(it) }
        return (titleFlightSignals >= 1 || bodyFlightSignals >= 3) && weatherSignals <= 1
    }

    @Composable
    private fun WeatherConditionIcon(
        condition: String?,
        size: Dp
    ) {
        NativeWeatherUiRenderer.WeatherConditionIcon(
            condition = condition,
            size = size,
            sanitizeDisplayText = ::sanitizeDisplayText
        )
    }

    private fun buildRelaxedCurrentWeatherDetails(
        title: String,
        sectionBlocks: List<TextBlock>,
        fallbackCondition: String?,
        additionalLines: List<String> = emptyList()
    ): WeatherCurrentDetails? {
        if (!NativeWeatherSemantics.isCurrentWeatherHeading(title)) {
            return null
        }
        val textLines = buildList {
            sectionBlocks.forEach { block ->
                when (block) {
                    is TextBlock.Paragraph -> add(block.text)
                    is TextBlock.Bullets -> addAll(block.items)
                    else -> Unit
                }
            }
            addAll(additionalLines)
        }
        val merged = textLines.joinToString(" ").replace(Regex("\\s+"), " ").trim()
        if (merged.isBlank()) {
            return null
        }
        val normalizedMerged = NativeWeatherSemantics.normalizeTemperatureText(merged)
        val iconUrl = NativeWeatherSemantics.extractCurrentWeatherIconUrl(sectionBlocks)
        val temperature = NativeWeatherSemantics.extractTemperatureValue(normalizedMerged)
        val feelsLike = NativeWeatherSemantics.extractFeelsLikeValue(normalizedMerged)
        val humidity = NativeWeatherSemantics.extractHumidityValue(normalizedMerged)
        val wind = NativeWeatherSemantics.extractWindValue(normalizedMerged)
        val rainChance = NativeWeatherSemantics.extractRainChanceValue(normalizedMerged)
        val uvIndex = NativeWeatherSemantics.extractUvIndexValue(normalizedMerged)
        val condition = NativeWeatherSemantics.inferWeatherConditionFromTextPool(normalizedMerged)
            ?: fallbackCondition
        val summary = normalizedMerged
            .split(Regex("(?<=[.!?])\\s+"))
            .filter { it.isNotBlank() }
            .take(2)
            .joinToString(" ")
            .trim()
            .takeIf { it.isNotBlank() }

        val hasAnySignal = !iconUrl.isNullOrBlank() ||
            !condition.isNullOrBlank() ||
            !temperature.isNullOrBlank() ||
            !feelsLike.isNullOrBlank() ||
            !humidity.isNullOrBlank() ||
            !wind.isNullOrBlank() ||
            !rainChance.isNullOrBlank() ||
            !uvIndex.isNullOrBlank()
        if (!hasAnySignal) {
            return null
        }
        return WeatherCurrentDetails(
            iconUrl = iconUrl,
            condition = condition,
            temperature = temperature,
            feelsLike = feelsLike,
            humidity = humidity,
            wind = wind,
            rainChance = rainChance,
            uvIndex = uvIndex,
            summary = summary
        )
    }

    private fun buildMinimalCurrentWeatherDetails(
        title: String,
        sectionBlocks: List<TextBlock>,
        fallbackCondition: String?,
        additionalLines: List<String> = emptyList()
    ): WeatherCurrentDetails? {
        if (!NativeWeatherSemantics.isCurrentWeatherHeading(title)) {
            return null
        }
        val merged = buildString {
            sectionBlocks.forEach { block ->
                when (block) {
                    is TextBlock.Paragraph -> append(' ').append(block.text)
                    is TextBlock.Bullets -> block.items.forEach { append(' ').append(it) }
                    else -> Unit
                }
            }
            additionalLines.forEach { append(' ').append(it) }
        }.replace(Regex("\\s+"), " ").trim()
        if (merged.isBlank()) {
            return null
        }
        val normalized = NativeWeatherSemantics.normalizeTemperatureText(merged)
        val iconUrl = NativeWeatherSemantics.extractCurrentWeatherIconUrl(sectionBlocks)
        val temperature = NativeWeatherSemantics.extractTemperatureValue(normalized)
        val feelsLike = NativeWeatherSemantics.extractFeelsLikeValue(normalized)
        val humidity = NativeWeatherSemantics.extractHumidityValue(normalized)
        val wind = NativeWeatherSemantics.extractWindValue(normalized)
        val rainChance = NativeWeatherSemantics.extractRainChanceValue(normalized)
        val uvIndex = NativeWeatherSemantics.extractUvIndexValue(normalized)
        val condition = NativeWeatherSemantics.inferWeatherConditionFromTextPool(normalized)
            ?: fallbackCondition
        val summary = normalized.takeIf { it.isNotBlank() }
        return WeatherCurrentDetails(
            iconUrl = iconUrl,
            condition = condition,
            temperature = temperature,
            feelsLike = feelsLike,
            humidity = humidity,
            wind = wind,
            rainChance = rainChance,
            uvIndex = uvIndex,
            summary = summary
        )
    }

    @Composable
    @Suppress("UNUSED_PARAMETER")
    private fun RenderCurrentWeatherDetails(
        titleText: String,
        titleStyle: TextStyle,
        details: WeatherCurrentDetails,
        sourceDir: File?
    ) {
        NativeWeatherUiRenderer.RenderCurrentWeatherDetails(
            titleText = titleText,
            titleStyle = titleStyle,
            details = details,
            sanitizeDisplayText = ::sanitizeDisplayText,
            inferWeatherConditionFromIconUrl = ::inferWeatherConditionFromIconUrl,
            markdownText = { text, style, color ->
                MarkdownText(text = text, style = style, color = color)
            },
            weatherConditionIcon = { condition, size ->
                WeatherConditionIcon(condition = condition, size = size)
            }
        )
    }

    private fun maybeExtractTable(children: List<String>, index: Map<String, JsonObject>): TableSpec? {
        return NativeComponentTableExtractor.maybeExtractTable(
            children = children,
            index = index,
            readChildren = ::readChildren,
            getString = { obj, key -> obj.getString(key) },
            readDynamicString = ::readDynamicString,
            getAsNumberOrNull = { obj, key -> obj.getAsNumberOrNull(key) },
            detectWeatherColumns = ::detectWeatherColumns,
            detectFlightColumns = ::detectFlightColumns,
            isPlaceholderTableCellRow = ::isPlaceholderTableCellRow
        )
    }

    private fun maybeExtractTableRun(
        children: List<String>,
        startIndex: Int,
        index: Map<String, JsonObject>
    ): Pair<TableSpec, Int>? {
        return NativeComponentTableExtractor.maybeExtractTableRun(
            children = children,
            startIndex = startIndex,
            index = index,
            readChildren = ::readChildren,
            getString = { obj, key -> obj.getString(key) },
            readDynamicString = ::readDynamicString,
            getAsNumberOrNull = { obj, key -> obj.getAsNumberOrNull(key) },
            detectWeatherColumns = ::detectWeatherColumns,
            detectFlightColumns = ::detectFlightColumns,
            isPlaceholderTableCellRow = ::isPlaceholderTableCellRow
        )
    }

    private fun maybeExtractCardRowTable(children: List<String>, index: Map<String, JsonObject>): TableSpec? {
        return NativeComponentTableExtractor.maybeExtractCardRowTable(
            children = children,
            index = index,
            readChildren = ::readChildren,
            getString = { obj, key -> obj.getString(key) },
            readDynamicString = ::readDynamicString,
            getAsNumberOrNull = { obj, key -> obj.getAsNumberOrNull(key) },
            detectWeatherColumns = ::detectWeatherColumns,
            detectFlightColumns = ::detectFlightColumns,
            isPlaceholderTableCellRow = ::isPlaceholderTableCellRow
        )
    }

    private fun maybeExtractTextList(children: List<String>, index: Map<String, JsonObject>): List<String>? {
        return NativeComponentTableExtractor.maybeExtractTextList(
            children = children,
            index = index,
            getString = { obj, key -> obj.getString(key) },
            readDynamicString = ::readDynamicString
        )
    }

    private fun shouldUseStructuredBlocks(rawText: String, variant: String): Boolean {
        return NativeTextBlockParser.shouldUseStructuredBlocks(
            rawText = rawText,
            variant = variant,
            isTableLikeLine = ::isTableLikeLine,
            isBulletListLine = ::isBulletListLine,
            isInlineMediaLine = ::isInlineMediaLine,
            isMediaMarkerHeading = ::isMediaMarkerHeading,
            looksLikeStandaloneLinkLine = ::looksLikeStandaloneLinkLine,
            isSourceHeadingLine = NativeSourceParsing::isSourceHeadingLine
        )
    }

    private fun parseTextBlocks(rawText: String): List<TextBlock> {
        return NativeTextBlockParser.parseTextBlocks(
            rawText = rawText,
            collectSourceLinks = ::collectSourceLinks,
            collectBookingOptions = ::collectBookingOptions,
            collectTableRows = ::collectTableRows,
            collectMediaEntries = ::collectMediaEntries,
            collectNumberedSteps = ::collectNumberedSteps,
            isBulletListLine = ::isBulletListLine,
            extractBulletListItem = ::extractBulletListItem,
            isPlaceholderListEntry = ::isPlaceholderListEntry,
            parseButtonLine = ::parseButtonLine,
            looksLikeStandaloneLinkLine = ::looksLikeStandaloneLinkLine,
            parseSourceLinksFromLine = ::parseSourceLinksFromLine,
            isSourcePrefixedLinkLine = NativeSourceParsing::isSourcePrefixedLinkLine,
            looksLikeSectionHeading = ::looksLikeSectionHeading,
            isStructuredBoundary = ::isStructuredBoundary
        )
    }

    private fun collectSourceLinks(
        lines: List<String>,
        startIndex: Int,
        inSourcesSection: Boolean
    ): Pair<List<ParsedButton>, Int>? {
        return NativeSourceParsing.collectSourceLinks(
            lines = lines,
            startIndex = startIndex,
            inSourcesSection = inSourcesSection,
            looksLikeSectionHeading = ::looksLikeSectionHeading,
            parseSourceLinksFromLine = ::parseSourceLinksFromLine
        )
    }

    private fun parseSourceLinksFromLine(line: String): List<ParsedButton> {
        return NativeSourceParsing.parseSourceLinksFromLine(
            line = line,
            stripLeadingBulletMarker = ::stripLeadingBulletMarker,
            sanitizeDisplayText = { sanitizeDisplayText(it) },
            sanitizeUrlToken = ::sanitizeUrlToken,
            toExternalUrl = { raw -> toExternalUrl(raw) },
            containsUrlLikeToken = ::containsUrlLikeToken
        )
    }

    private fun collectBookingOptions(
        lines: List<String>,
        startIndex: Int
    ): Pair<List<BookingOption>, Int>? {
        return NativeBookingParsing.collectBookingOptions(
            lines = lines,
            startIndex = startIndex,
            parseOptionLine = ::parseOptionLine,
            parseButtonLine = ::parseButtonLine,
            isBulletListLine = ::isBulletListLine,
            extractBulletListItem = ::extractBulletListItem,
            looksLikeImagePath = ::looksLikeImagePath,
            normalizeMatchText = ::normalizeMatchText
        )
    }

    private fun normalizeMatchText(value: String): String {
        return NativeFlightSemantics.normalizeMatchText(value)
    }

    private fun collectMediaEntries(
        lines: List<String>,
        startIndex: Int
    ): Pair<List<ParsedMediaEntry>, Int>? {
        return NativeMediaParsing.collectMediaEntries(
            lines = lines,
            startIndex = startIndex,
            stripLeadingBulletMarker = ::stripLeadingBulletMarker,
            sanitizeUrlToken = ::sanitizeUrlToken,
            looksLikeImagePath = ::looksLikeImagePath,
            looksLikeCompactIconUrl = ::looksLikeCompactIconUrl
        )
    }

    private fun isMediaMarkerHeading(line: String): Boolean {
        return NativeMediaParsing.isMediaMarkerHeading(line)
    }

    private fun isInlineMediaLine(line: String): Boolean {
        return NativeMediaParsing.isInlineMediaLine(line)
    }

    private fun collectNumberedSteps(
        lines: List<String>,
        startIndex: Int
    ): Pair<List<StepEntry>, Int>? {
        return NativeStructureParsing.collectNumberedSteps(
            lines = lines,
            startIndex = startIndex,
            isStructuredBoundary = ::isStructuredBoundary
        )
    }

    private fun collectTableRows(lines: List<String>, startIndex: Int): Pair<List<List<String>>, Int>? {
        return NativeStructureParsing.collectTableRows(
            lines = lines,
            startIndex = startIndex,
            containsUrlLikeToken = ::containsUrlLikeToken,
            isPlaceholderTableStringRow = ::isPlaceholderTableStringRow
        )
    }

    private fun parseOptionLine(line: String): Pair<String, String>? {
        return NativeStructureParsing.parseOptionLine(line)
    }

    private fun parseButtonLine(line: String): ParsedButton? {
        return NativeStructureParsing.parseButtonLine(line, ::sanitizeUrlToken)
    }

    private fun sanitizeUrlToken(value: String): String =
        NativeStructureParsing.sanitizeUrlToken(value)

    private fun isTableLikeLine(line: String): Boolean {
        return NativeStructureParsing.isTableLikeLine(line, ::containsUrlLikeToken)
    }

    private fun splitTableCells(line: String): List<String> =
        NativeStructureParsing.splitTableCells(line)

    private fun isPlaceholderTableStringRow(row: List<String>): Boolean {
        return NativeTableSemantics.isPlaceholderTableStringRow(row)
    }

    private fun isPlaceholderTableCellRow(row: List<TableCell>): Boolean =
        NativeTableSemantics.isPlaceholderTableCellRow(row)

    private fun isPlaceholderTableCellValue(value: String): Boolean {
        return NativeTableSemantics.isPlaceholderTableCellValue(value)
    }

    private fun isPlaceholderListEntry(value: String): Boolean {
        return NativeBlockHeuristics.isPlaceholderListEntry(
            value = value,
            parseLeadingLabelValue = ::parseLeadingLabelValue,
            isPlaceholderTableCellValue = ::isPlaceholderTableCellValue
        )
    }

    private fun looksLikeSectionHeading(line: String): Boolean {
        return NativeBlockHeuristics.looksLikeSectionHeading(
            line = line,
            parseLeadingLabelValue = ::parseLeadingLabelValue,
            isBulletListLine = ::isBulletListLine,
            containsUrlLikeToken = ::containsUrlLikeToken
        )
    }

    private fun containsUrlLikeToken(value: String): Boolean =
        NativeTextFormatter.containsUrlLikeToken(value)

    private fun looksLikeStandaloneLinkLine(value: String): Boolean {
        return NativeBlockHeuristics.looksLikeStandaloneLinkLine(
            value = value,
            containsUrlLikeToken = ::containsUrlLikeToken,
            isInlineMediaLine = ::isInlineMediaLine,
            isMediaMarkerHeading = ::isMediaMarkerHeading,
            splitTableCells = ::splitTableCells
        )
    }

    private fun isStructuredBoundary(line: String): Boolean {
        return NativeBlockHeuristics.isStructuredBoundary(
            line = line,
            isBulletListLine = ::isBulletListLine,
            isInlineMediaLine = ::isInlineMediaLine,
            isMediaMarkerHeading = ::isMediaMarkerHeading,
            parseButtonLine = { parseButtonLine(it) != null },
            parseOptionLine = { parseOptionLine(it) != null },
            isNumberedStepLine = NativeStructureParsing::isNumberedStepLine,
            isTableLikeLine = ::isTableLikeLine,
            looksLikeSectionHeading = ::looksLikeSectionHeading
        )
    }

    private fun isBulletListLine(line: String): Boolean =
        NativeStructureParsing.isBulletListLine(line)

    private fun extractBulletListItem(line: String): String? =
        NativeStructureParsing.extractBulletListItem(line)

    private fun stripLeadingBulletMarker(line: String): String =
        NativeStructureParsing.stripLeadingBulletMarker(line)

    private fun looksLikeImagePath(value: String, labelHint: String? = null): Boolean {
        return NativeMediaVisualUtils.looksLikeImagePath(value, labelHint)
    }

    private fun looksLikeCompactIconUrl(value: String): Boolean {
        return NativeMediaVisualUtils.looksLikeCompactIconUrl(value)
    }

    private fun isPhotoLikeImageUrl(value: String): Boolean {
        return NativeMediaVisualUtils.isPhotoLikeImageUrl(value)
    }

    private fun shouldShowMediaLabel(label: String): Boolean {
        return NativeMediaVisualUtils.shouldShowMediaLabel(label, ::sanitizeDisplayText)
    }

    private fun defaultImageScale(
        rawUrl: String,
        fitValue: String?,
        defaultCoverForRaster: Boolean
    ): ContentScale {
        return NativeMediaVisualUtils.defaultImageScale(
            rawUrl = rawUrl,
            fitValue = fitValue,
            defaultCoverForRaster = defaultCoverForRaster,
            normalizeLayoutToken = { token -> token.normalizeLayoutToken() }
        )
    }

    private fun resolveExternalUrl(raw: String, sourceDir: File?): String? =
        toExternalUrl(NativePayloadParser.resolveAssetUrl(raw, sourceDir))

    private fun canonicalSourceUrlToken(raw: String): String {
        val normalized = raw.trim()
        val uri = runCatching { Uri.parse(normalized) }.getOrNull() ?: return normalized.lowercase(Locale.US)
        val scheme = (uri.scheme ?: "https").lowercase(Locale.US)
        val host = uri.host?.lowercase(Locale.US)?.removePrefix("www.").orEmpty()
        if (host.isBlank()) {
            return normalized.lowercase(Locale.US)
        }
        val path = uri.path.orEmpty().trimEnd('/')
        val query = uri.query.orEmpty().trim()
        return buildString {
            append(scheme)
            append("://")
            append(host)
            if (path.isNotBlank()) append(path)
            if (query.isNotBlank()) {
                append('?')
                append(query)
            }
        }
    }

    private fun resolveImageModel(raw: String, sourceDir: File?): String {
        val resolved = NativePayloadParser.resolveAssetUrl(raw, sourceDir).replace("\\", "/")
        return when {
            resolved.startsWith("/assets/") -> "file:///android_asset/${resolved.removePrefix("/assets/")}"
            resolved.startsWith("assets/") -> "file:///android_asset/${resolved.removePrefix("assets/")}"
            else -> resolved
        }
    }

    private fun toExternalUrl(value: String?): String? {
        if (value.isNullOrBlank()) {
            return null
        }
        val normalized = NativePayloadParser.canonicalizeNetworkUrlToken(value)
        val scheme = runCatching { Uri.parse(normalized).scheme?.lowercase(Locale.US) }.getOrNull()
        return if (scheme == "http" || scheme == "https") normalized else null
    }

    private fun readChildren(component: JsonObject): List<String> =
        NativePayloadParser.readChildren(component)

    private fun readDynamicString(element: JsonElement?): String =
        NativePayloadParser.readDynamicString(element, ::normalizeMojibakeText)

    private fun String.normalizeLayoutToken(): String {
        return lowercase(Locale.US)
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
    }

}

