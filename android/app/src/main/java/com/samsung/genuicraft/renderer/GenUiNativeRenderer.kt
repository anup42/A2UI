package com.samsung.genuicraft

import android.net.Uri
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
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
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
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
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
import androidx.compose.ui.text.style.TextAlign
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
import com.samsung.genuicraft.renderer.native.NativePayloadParser
import com.samsung.genuicraft.renderer.native.NativeTextFormatter
import com.samsung.genuicraft.renderer.native.*
import com.samsung.genuicraft.renderer.native.media.NativeMediaVisualUtils
import com.samsung.genuicraft.renderer.native.intents.NativeIntentRegistry
import com.samsung.genuicraft.renderer.native.intents.NativeIntentRenderModel
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
    data class RenderResult(
        val surfaces: List<SurfaceState>,
        val warnings: List<String>,
        val errorMessage: String? = null
    )

    data class SurfaceState(
        val surfaceId: String,
        val rootId: String,
        val components: Map<String, JsonObject>
    )

    private enum class IconFallbackKind {
        Generic,
        Weather
    }

    fun render(rawInput: String, sourceDir: File?): RenderResult {
        val warnings = mutableListOf<String>()
        val parsed = try {
            parseJsonOrJsonl(rawInput, warnings)
        } catch (exc: Exception) {
            return RenderResult(emptyList(), warnings, "Invalid payload: ${exc.message ?: exc.javaClass.simpleName}")
        }

        val messages = extractMessages(parsed)
            ?: return RenderResult(
                surfaces = emptyList(),
                warnings = warnings + "Input did not contain genui_json/messages/payload.",
                errorMessage = "No renderable GenUI payload found."
            )

        val surfaces = extractSurfaces(messages, warnings)
        if (surfaces.isEmpty()) {
            return RenderResult(
                surfaces = emptyList(),
                warnings = warnings + "Messages were parsed, but no component arrays were found.",
                errorMessage = "No renderable components found."
            )
        }
        return RenderResult(surfaces, warnings)
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

        LazyColumn(
            modifier = modifier.fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            items(items = result.surfaces, key = { it.surfaceId }) { surface ->
                SurfaceCard(surface, sourceDir, onOpenExternalUrl)
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

        Column(
            modifier = modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            result.surfaces.forEach { surface ->
                SurfaceCard(
                    surface = surface,
                    sourceDir = sourceDir,
                    onOpenExternalUrl = onOpenExternalUrl
                )
            }
        }
    }

    @Composable
    private fun SurfaceCard(
        surface: SurfaceState,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit
    ) {
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
            "Column" -> RenderContainer(component, index, sourceDir, onOpenExternalUrl, nextPath, false, false)
            "Row" -> RenderContainer(component, index, sourceDir, onOpenExternalUrl, nextPath, true, false)
            "List" -> {
                val isHorizontal = component.getString("direction")?.normalizeLayoutToken()?.contains("horizontal") == true
                RenderContainer(component, index, sourceDir, onOpenExternalUrl, nextPath, isHorizontal, true)
            }

            "Card" -> RenderCardComponent(component, index, sourceDir, onOpenExternalUrl, nextPath)
            "Text" -> RenderTextComponent(component, sourceDir, onOpenExternalUrl)
            "Image" -> RenderImageComponent(component, sourceDir)
            "Icon" -> RenderIconComponent(component, sourceDir)
            "Table" -> RenderTableComponent(component)
            "Divider" -> HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            "Button" -> RenderButtonComponent(component, index, sourceDir, onOpenExternalUrl)
            "Tabs" -> RenderTabsComponent(component, index, sourceDir, onOpenExternalUrl, nextPath)
            else -> Text(
                text = "Unsupported component: ${component.getString("component") ?: "Unknown"}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error
            )
        }
    }

    @Composable
    private fun RenderContainer(
        component: JsonObject,
        index: Map<String, JsonObject>,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit,
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

        val alignToken = component.getString("align")?.normalizeLayoutToken()
        if (rowLayout) {
            RenderRowChildren(
                children = children,
                index = index,
                sourceDir = sourceDir,
                onOpenExternalUrl = onOpenExternalUrl,
                activePath = activePath,
                alignToken = alignToken,
                justifyToken = component.getString("justify")?.normalizeLayoutToken()
            )
            return
        }

        val horizontalAlignment = when (alignToken) {
            "center", "middle" -> Alignment.CenterHorizontally
            "end", "flexend", "right", "bottom" -> Alignment.End
            else -> Alignment.Start
        }

        Column(
            modifier = Modifier.fillMaxWidth(),
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
                RenderComponent(childId, index, sourceDir, onOpenExternalUrl, activePath)
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
        activePath: Set<String>,
        alignToken: String?,
        justifyToken: String?
    ) {
        parseInlineBulletRow(children, index)?.let { bulletRow ->
            MarkdownText(
                text = "${bulletRow.bullet} ${bulletRow.text}",
                style = textStyleForVariant(bulletRow.variant),
                color = textColorForVariant(bulletRow.variant),
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
                        RenderComponent(childId, index, sourceDir, onOpenExternalUrl, activePath)
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
                    RenderComponent(childId, index, sourceDir, onOpenExternalUrl, activePath)
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
                RenderComponent(childId, index, sourceDir, onOpenExternalUrl, activePath)
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
        activePath: Set<String>
    ) {
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
                val childId = component.getString("child")
                if (childId != null) {
                    RenderComponent(childId, index, sourceDir, onOpenExternalUrl, activePath)
                } else {
                    readChildren(component).forEach { child ->
                        RenderComponent(child, index, sourceDir, onOpenExternalUrl, activePath)
                    }
                }
            }
        }
    }

    @Composable
    private fun RenderTextComponent(
        component: JsonObject,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit
    ) {
        val variant = (component.getString("variant") ?: "body").lowercase(Locale.US)
        val rawText = readDynamicString(component.get("text"))
        val style = textStyleForVariant(variant)
        val color = textColorForVariant(variant)

        if (rawText.isBlank()) {
            Text("", style = style)
            return
        }

        if (shouldUseStructuredBlocks(rawText, variant)) {
            val blocks = remember(rawText) { parseTextBlocks(rawText) }
            RenderTextBlocks(blocks, sourceDir, onOpenExternalUrl, style, color)
            return
        }

        MarkdownText(
            text = rawText,
            style = style,
            color = color
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
                    RenderMediaCards(entries = block.entries, sourceDir = sourceDir)
                }
            }
        }

        fun isSimpleSectionBlock(block: TextBlock): Boolean = when (block) {
            is TextBlock.Paragraph,
            is TextBlock.Bullets,
            is TextBlock.Actions,
            is TextBlock.Sources,
            is TextBlock.MediaCards -> true
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
                if (headingText != null && isBoilerplateContextHeading(headingText)) {
                    val nextBlock = blocks.getOrNull(index + 1)
                    if (nextBlock is TextBlock.Paragraph && isBoilerplateContextParagraph(nextBlock.text)) {
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
                    val fallbackWeatherCondition = inferWeatherConditionFromSection(titleText, sectionBlocks)
                    val (followUpWeatherLines, followUpConsumed) =
                        if (isCurrentWeatherHeading(titleText)) {
                            collectCurrentWeatherFollowUpLines(blocks, cursor)
                        } else {
                            emptyList<String>() to 0
                        }
                    val currentWeatherDetails = buildCurrentWeatherDetails(
                        title = titleText,
                        sectionBlocks = sectionBlocks,
                        fallbackCondition = fallbackWeatherCondition,
                        additionalLines = followUpWeatherLines
                    )
                    val hasVisualMedia = sectionBlocks.any { section ->
                        section is TextBlock.MediaCards && section.entries.any { entry -> !entry.iconLike }
                    }
                    val hasIconGallery = sectionBlocks.any { section ->
                        section is TextBlock.MediaCards &&
                            section.entries.none { entry -> !entry.iconLike } &&
                            section.entries.count { entry -> entry.iconLike } >= 4
                    }
                    val shouldRenderSectionCard =
                        sectionBlocks.isNotEmpty() &&
                            (currentWeatherDetails != null || hasVisualMedia || hasIconGallery)
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
                                if (currentWeatherDetails != null) {
                                    RenderCurrentWeatherDetails(
                                        titleText = titleText,
                                        titleStyle = titleStyle,
                                        details = currentWeatherDetails,
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
                                    if (currentWeatherDetails != null &&
                                        (sectionBlock is TextBlock.Paragraph || sectionBlock is TextBlock.Bullets)
                                    ) {
                                        return@forEach
                                    }
                                    if (currentWeatherDetails != null &&
                                        sectionBlock is TextBlock.MediaCards &&
                                        sectionBlock.entries.all { it.iconLike }
                                    ) {
                                        return@forEach
                                    }
                                    when (sectionBlock) {
                                        is TextBlock.MediaCards -> RenderMediaEntriesInline(
                                            entries = sectionBlock.entries,
                                            sourceDir = sourceDir
                                        )

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
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                icons.forEach { icon ->
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
                        if (shouldShowMediaLabel(icon.label)) {
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

        val iconBuckets = assignIconsToPrimary(primary, icons)
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

        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            links.forEach { link ->
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
        val hasMarkdownInline = containsMarkdownInlineFormatting(displayText)
        val hasMarkdownHeading = containsMarkdownHeading(displayText)
        Text(
            text = remember(displayText, style) {
                if (hasMarkdownHeading) {
                    parseMarkdownWithHeadings(displayText, style)
                } else {
                    parseInlineMarkdown(displayText)
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

    private fun containsMarkdownInlineFormatting(text: String): Boolean =
        NativeTextFormatter.containsMarkdownInlineFormatting(text)

    private fun containsMarkdownHeading(text: String): Boolean =
        NativeTextFormatter.containsMarkdownHeading(text)

    private fun parseMarkdownWithHeadings(
        text: String,
        baseStyle: TextStyle
    ): AnnotatedString = NativeTextFormatter.parseMarkdownWithHeadings(text, baseStyle)

    private fun parseInlineMarkdown(text: String): AnnotatedString =
        NativeTextFormatter.parseInlineMarkdown(text)

    @Suppress("unused")
    private fun parseBoldMarkdown(text: String): AnnotatedString {
        // Kept for compatibility with existing reflection-based tests/callers.
        return parseInlineMarkdown(text)
    }

    private fun sanitizeDisplayText(text: String, preserveMarkdown: Boolean = false): String =
        NativeTextFormatter.sanitizeDisplayText(text, preserveMarkdown)

    private fun normalizeMojibakeText(value: String): String =
        NativeTextFormatter.normalizeMojibakeText(value)

    private fun parseLeadingLabelValue(text: String): LeadingLabelValue? {
        return NativeTextFormatter.parseLeadingLabelValue(text)
    }

    private fun isLikelyLeadingLabel(label: String, value: String): Boolean =
        NativeTextFormatter.isLikelyLeadingLabel(label, value)

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
                    icons.forEach { icon ->
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

        val iconBuckets = assignIconsToPrimary(primary, icons)

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

        val intentModel = resolveIntentModel(header, body)
        if (intentModel != null && renderIntentModel(intentModel)) {
            return
        }

        val columnCount = rows.maxOf { it.size }.coerceAtLeast(2)
        val columnWidths = List(columnCount) { tableBaseCellWidth(columnCount) }
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
    private fun RenderImageComponent(component: JsonObject, sourceDir: File?) {
        val rawUrl = readDynamicString(component.get("url"))
        if (rawUrl.isBlank()) {
            Text("Missing image source", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
            return
        }

        val variant = (component.getString("variant") ?: "").lowercase(Locale.US)
        val fit = component.getString("fit")
        val urlLower = rawUrl.lowercase(Locale.US)
        val likelyLogo = variant.contains("logo") || urlLower.contains("logo")
        val rasterImage =
            urlLower.endsWith(".jpg") ||
                urlLower.endsWith(".jpeg") ||
                urlLower.endsWith(".png") ||
                urlLower.endsWith(".webp")
        val mediumFeature = variant.contains("mediumfeature")
        val likelyIcon =
            variant.contains("icon") ||
                looksLikeCompactIconUrl(urlLower) ||
                (!likelyLogo &&
                    urlLower.endsWith(".svg") &&
                    !variant.contains("feature") &&
                    !variant.contains("thumbnail"))
        val inlineIconLike = likelyIcon || (mediumFeature && !rasterImage)

        MediaImage(
            rawUrl = rawUrl,
            sourceDir = sourceDir,
            modifier = when {
                inlineIconLike -> Modifier.size(28.dp)
                likelyLogo -> Modifier
                    .width(90.dp)
                    .height(30.dp)
                mediumFeature && rasterImage -> mediaFrameModifier()
                variant.contains("feature") -> mediaFrameModifier()
                variant.contains("thumbnail") -> mediaFrameModifier()
                else -> mediaFrameModifier()
            },
            contentScale = defaultImageScale(
                rawUrl = rawUrl,
                fitValue = fit,
                defaultCoverForRaster = rasterImage && !likelyLogo && !inlineIconLike
            ),
            asIcon = inlineIconLike
        )
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
    private fun RenderIconComponent(component: JsonObject, sourceDir: File?) {
        val raw = readDynamicString(component.get("name"))
        if (raw.isBlank()) {
            Text("Missing icon source", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
            return
        }

        MediaImage(
            rawUrl = raw,
            sourceDir = sourceDir,
            modifier = Modifier.size(26.dp),
            contentScale = ContentScale.Fit,
            asIcon = true
        )
    }

    @Composable
    private fun RenderButtonComponent(
        component: JsonObject,
        index: Map<String, JsonObject>,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit
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

        val resolvedUrl = extractOpenUrl(component)
            ?.let { resolveAssetUrl(it, sourceDir) }
            ?.let { toExternalUrl(it) }

        if (variant == "borderless") {
            OutlinedButton(
                onClick = { resolvedUrl?.let(onOpenExternalUrl) },
                enabled = resolvedUrl != null,
                shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
            ) {
                Text(displayLabel, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
            }
            return
        }

        Button(
            onClick = { resolvedUrl?.let(onOpenExternalUrl) },
            enabled = resolvedUrl != null,
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
        activePath: Set<String>
    ) {
        val tabs = component.getAsJsonArrayOrNull("tabs") ?: component.getAsJsonArrayOrNull("items")
        if (tabs == null || tabs.size() == 0) {
            return
        }

        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            tabs.forEachIndexed { tabIndex, tabElement ->
                val tab = tabElement.asJsonObjectOrNull()
                val title = sanitizeDisplayText(readDynamicString(tab?.get("title"))).ifBlank { "Tab ${tabIndex + 1}" }
                val childId = tab?.getString("child")

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
                        Text(title, style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.onSurface)
                        if (childId == null) {
                            Text("Missing tab child", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
                        } else {
                            RenderComponent(childId, index, sourceDir, onOpenExternalUrl, activePath)
                        }
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
            val intentModel = resolveIntentModel(semanticHeader, semanticBody)
            if (intentModel != null && renderIntentModel(intentModel)) {
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
        val columnWidths = tableColumnWidths(normalizedSpec, columnCount, hasExplicitWeights)
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
                .background(tableRowBackground(isHeader = isHeader, rowIndex = rowIndex))
        ) {
            for (column in 0 until columnCount) {
                val cell = cells.getOrNull(column)
                val displayText = sanitizeTableCellDisplayValue(cell?.text.orEmpty())
                val style = if (isHeader) {
                    MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold)
                } else {
                    textStyleForVariant(cell?.variant ?: "body")
                }
                val color = if (isHeader) {
                    MaterialTheme.colorScheme.onSurface
                } else {
                    textColorForVariant(cell?.variant ?: "body")
                }
                val cellWidth = columnWidths.getOrElse(column) { tableBaseCellWidth(columnCount) }

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
                    TableCellDivider()
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
                .background(tableRowBackground(isHeader = isHeader, rowIndex = rowIndex))
        ) {
            for (column in 0 until columnCount) {
                val value = sanitizeTableCellDisplayValue(values.getOrNull(column).orEmpty())
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
                        .width(columnWidths.getOrElse(column) { tableBaseCellWidth(columnCount) })
                        .padding(horizontal = 12.dp, vertical = 10.dp)
                )

                if (column != columnCount - 1) {
                    TableCellDivider()
                }
            }
        }
    }

    private fun resolveIntentModel(
        header: List<String>,
        body: List<List<String>>
    ): NativeIntentRenderModel? {
        return NativeIntentRegistry.resolve(header, body)
    }

    @Composable
    private fun renderIntentModel(model: NativeIntentRenderModel): Boolean {
        return NativeIntentRegistry.render(model = model)
    }

    private fun detectWeatherColumns(header: List<String>): WeatherTableColumns? {
        return NativeWeatherSemantics.detectWeatherColumns(header)
    }

    private fun detectFlightColumns(header: List<String>): FlightTableColumns? {
        return NativeFlightSemantics.detectFlightColumns(header)
    }

    private fun resolveFlightColumns(
        detected: FlightTableColumns,
        body: List<List<String>>
    ): FlightTableColumns {
        return NativeFlightSemantics.resolveFlightColumns(detected, body)
    }

    private fun looksLikeTimeValue(value: String): Boolean {
        return NativeFlightSemantics.looksLikeTimeValue(value)
    }

    private fun looksLikeDurationValue(value: String): Boolean {
        return NativeFlightSemantics.looksLikeDurationValue(value)
    }

    private fun looksLikeFareValue(value: String): Boolean {
        return NativeFlightSemantics.looksLikeFareValue(value)
    }

    private fun looksLikeAirlineValue(value: String): Boolean {
        return NativeFlightSemantics.looksLikeAirlineValue(value)
    }

    private fun extractAirportCode(headerText: String): String? {
        return NativeFlightSemantics.extractAirportCode(headerText)
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

    private fun isCurrentWeatherHeading(title: String): Boolean {
        return NativeWeatherSemantics.isCurrentWeatherHeading(title)
    }

    private fun collectCurrentWeatherFollowUpLines(
        blocks: List<TextBlock>,
        startIndex: Int
    ): Pair<List<String>, Int> {
        return NativeWeatherSemantics.collectCurrentWeatherFollowUpLines(blocks, startIndex)
    }

    private fun isLikelyCurrentWeatherDetailLine(text: String): Boolean {
        return NativeWeatherSemantics.isLikelyCurrentWeatherDetailLine(text)
    }

    private fun inferWeatherConditionFromSection(
        title: String,
        sectionBlocks: List<TextBlock>
    ): String? {
        return NativeWeatherSemantics.inferWeatherConditionFromSection(title, sectionBlocks)
    }

    private fun buildCurrentWeatherDetails(
        title: String,
        sectionBlocks: List<TextBlock>,
        fallbackCondition: String?,
        additionalLines: List<String> = emptyList()
    ): WeatherCurrentDetails? {
        return NativeWeatherSemantics.buildCurrentWeatherDetails(
            title = title,
            sectionBlocks = sectionBlocks,
            fallbackCondition = fallbackCondition,
            additionalLines = additionalLines
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

    @Composable
    private fun TableCellDivider() {
        NativeTableUi.TableCellDivider()
    }

    @Composable
    private fun tableRowBackground(isHeader: Boolean, rowIndex: Int): Color {
        return NativeTableUi.tableRowBackground(isHeader, rowIndex)
    }

    private fun tableBaseCellWidth(columnCount: Int): Dp {
        return NativeTableSemantics.tableBaseCellWidth(columnCount)
    }

    private fun tableColumnWidths(spec: TableSpec, columnCount: Int, weightedLayout: Boolean): List<Dp> {
        return NativeTableSemantics.tableColumnWidths(spec, columnCount, weightedLayout)
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

    @Composable
    private fun textStyleForVariant(variant: String): TextStyle =
        NativeVariantStyleResolver.textStyleForVariant(variant)

    @Composable
    private fun textColorForVariant(variant: String): Color =
        NativeVariantStyleResolver.textColorForVariant(variant)

    private fun shouldUseStructuredBlocks(rawText: String, variant: String): Boolean {
        return NativeTextBlockParser.shouldUseStructuredBlocks(
            rawText = rawText,
            variant = variant,
            isTableLikeLine = ::isTableLikeLine,
            isBulletListLine = ::isBulletListLine,
            isInlineMediaLine = ::isInlineMediaLine,
            isMediaMarkerHeading = ::isMediaMarkerHeading
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

    private fun assignIconsToPrimary(
        primary: List<ParsedMediaEntry>,
        icons: List<ParsedMediaEntry>
    ): List<List<ParsedMediaEntry>> {
        return NativeTextBlockSemantics.assignIconsToPrimary(primary, icons)
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

    private fun sanitizeTableCellDisplayValue(value: String): String {
        return NativeTableSemantics.sanitizeTableCellDisplayValue(value)
    }

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

    private fun isBoilerplateContextHeading(text: String): Boolean {
        return NativeTextBlockSemantics.isBoilerplateContextHeading(text)
    }

    private fun isBoilerplateContextParagraph(text: String): Boolean {
        return NativeTextBlockSemantics.isBoilerplateContextParagraph(text)
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

    private fun shouldShowMediaLabel(label: String): Boolean {
        return NativeMediaVisualUtils.shouldShowMediaLabel(label, ::sanitizeDisplayText)
    }

    private fun isVectorImagePath(value: String): Boolean =
        NativeMediaVisualUtils.isVectorImagePath(value)

    private fun isRasterImagePath(value: String): Boolean {
        return NativeMediaVisualUtils.isRasterImagePath(value)
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
        toExternalUrl(resolveAssetUrl(raw, sourceDir))

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

    private fun resolveImageModel(raw: String, sourceDir: File?): String {
        val resolved = resolveAssetUrl(raw, sourceDir).replace("\\", "/")
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
        val normalized = canonicalizeNetworkUrlToken(value)
        val scheme = runCatching { Uri.parse(normalized).scheme?.lowercase(Locale.US) }.getOrNull()
        return if (scheme == "http" || scheme == "https") normalized else null
    }

    private fun parseJsonOrJsonl(rawInput: String, warnings: MutableList<String>): JsonElement =
        NativePayloadParser.parseJsonOrJsonl(rawInput, warnings)

    private fun extractMessages(root: JsonElement): JsonArray? =
        NativePayloadParser.extractMessages(root)

    private fun extractSurfaces(messages: JsonArray, warnings: MutableList<String>): List<SurfaceState> =
        NativePayloadParser.extractSurfaces(messages, warnings)

    private fun buildSurfaceStates(
        surfaceId: String,
        components: List<JsonObject>,
        rootHint: String?
    ): List<SurfaceState> =
        NativePayloadParser.buildSurfaceStates(surfaceId, components, rootHint)

    private fun normalizeComponents(components: JsonArray, warnings: MutableList<String>): List<JsonObject> =
        NativePayloadParser.normalizeComponents(components, warnings)

    private fun convertV08Component(raw: JsonObject, warnings: MutableList<String>): JsonObject? =
        NativePayloadParser.convertV08Component(raw, warnings)

    private fun normalizeChildrenValue(children: JsonElement): JsonElement =
        NativePayloadParser.normalizeChildrenValue(children)

    private fun readChildren(component: JsonObject): List<String> =
        NativePayloadParser.readChildren(component)

    private fun readDynamicString(element: JsonElement?): String =
        NativePayloadParser.readDynamicString(element, ::normalizeMojibakeText)

    private fun resolveAssetUrl(raw: String, sourceDir: File?): String =
        NativePayloadParser.resolveAssetUrl(raw, sourceDir)

    private fun canonicalizeNetworkUrlToken(value: String): String =
        NativePayloadParser.canonicalizeNetworkUrlToken(value)

    private fun normalizeHttpUrlCandidate(value: String): String? =
        NativePayloadParser.normalizeHttpUrlCandidate(value)

    private fun isLikelyPublicDomainHost(host: String): Boolean =
        NativePayloadParser.isLikelyPublicDomainHost(host)

    private fun String.normalizeLayoutToken(): String {
        return lowercase(Locale.US)
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
    }

    private fun jsonPrimitive(value: String): JsonElement =
        NativePayloadParser.jsonPrimitive(value)

    private fun escapeJsonString(value: String): String =
        NativePayloadParser.escapeJsonString(value)

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

