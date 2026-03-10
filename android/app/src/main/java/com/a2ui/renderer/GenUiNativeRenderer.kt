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
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.sizeIn
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
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

    private data class ParsedButton(val label: String, val url: String)
    private data class ParsedLogo(val label: String, val url: String)
    private data class ParsedMediaEntry(val label: String, val url: String, val iconLike: Boolean)
    private data class BookingOption(
        val title: String,
        val details: String,
        val button: ParsedButton?,
        val logo: ParsedLogo? = null
    )

    private data class TableCell(
        val text: String,
        val variant: String,
        val weight: Float
    )

    private data class TableSpec(
        val header: List<TableCell>?,
        val rows: List<List<TableCell>>
    )
    private data class StepEntry(
        val title: String,
        val details: String
    )
    private data class LeadingLabelValue(
        val prefix: String,
        val label: String,
        val delimiter: String,
        val value: String
    )
    private data class InlineBulletRow(
        val bullet: String,
        val text: String,
        val variant: String
    )

    private sealed interface TextBlock {
        data class Title(val text: String) : TextBlock
        data class Heading(val text: String) : TextBlock
        data class Paragraph(val text: String) : TextBlock
        data class Bullets(val items: List<String>) : TextBlock
        data class NumberedSteps(val items: List<StepEntry>) : TextBlock
        data class Table(val rows: List<List<String>>) : TextBlock
        data class Sources(val links: List<ParsedButton>) : TextBlock
        data class Actions(val actions: List<ParsedButton>) : TextBlock
        data class BookingCards(val options: List<BookingOption>) : TextBlock
        data class MediaCards(val entries: List<ParsedMediaEntry>) : TextBlock
    }

    private val OPTION_LINE_REGEX =
        Regex("""^Option\s+\d+\s*:\s*(.+?)\s*\|\s*(.+)$""", RegexOption.IGNORE_CASE)

    private val BUTTON_LINE_REGEX =
        Regex("""^(?:Action:\s*)?\[Button:\s*(.+?)\]\s*(\S+)\s*$""", RegexOption.IGNORE_CASE)
    private val NUMBERED_STEP_REGEX = Regex("""^(\d+)\.\s+(.+)$""")
    private val LEADING_LABEL_REGEX =
        Regex("""^\s*([•\-]\s*)?([^:;：；\n]{1,70}?)([:;：；])\s*(.+)$""")
    private val INLINE_LABEL_REGEX =
        Regex("""([A-Za-z][A-Za-z0-9/&()' \-]{0,60})([:;：；])""")

    private val URL_REGEX = Regex("""https?://[^\s<>\]]+""", RegexOption.IGNORE_CASE)

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
            modifier = Modifier.fillMaxWidth()
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
            maybeExtractTable(children, index)?.let {
                RenderTableSpec(it)
                return
            }

            if (listComponent) {
                maybeExtractTextList(children, index)?.let { items ->
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        items.forEach { item ->
                            MarkdownText(
                                text = "• ${item.trim()}",
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
            children.forEach { childId ->
                RenderComponent(childId, index, sourceDir, onOpenExternalUrl, activePath)
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
        if (!bulletToken.matches(Regex("""^[•●◦·▪*-]$"""))) {
            return null
        }

        val text = readDynamicString(textNode.get("text")).trim()
        if (text.isBlank()) {
            return null
        }

        val variant = (textNode.getString("variant") ?: "body").lowercase(Locale.US)
        val normalizedBullet = if (bulletToken == "-") "•" else bulletToken
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
            modifier = Modifier.fillMaxWidth(),
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
        val rawText = readDynamicString(component.get("text")).replace("â€¢", "•")
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
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        block.items.forEach { item ->
                            RenderListItem(item = "• ${item.trim()}", sourceDir = sourceDir)
                        }
                    }
                }

                is TextBlock.NumberedSteps -> {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        block.items.forEach { step ->
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

        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            var index = 0
            while (index < blocks.size) {
                val block = blocks[index]

                val titleParagraph = if (block is TextBlock.Title) {
                    blocks.getOrNull(index + 1) as? TextBlock.Paragraph
                } else {
                    null
                }
                if (titleParagraph != null && titleParagraph.text.length >= 180) {
                    val titleText = (block as TextBlock.Title).text
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
                                .padding(horizontal = 12.dp, vertical = 10.dp),
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
                                    .padding(horizontal = 12.dp, vertical = 10.dp),
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

    @Composable
    private fun ExternalActionButton(
        label: String,
        url: String?,
        onOpenExternalUrl: (String) -> Unit
    ) {
        Button(
            onClick = { url?.let(onOpenExternalUrl) },
            enabled = url != null,
            shape = RoundedCornerShape(GenUiTokens.RadiusPill)
        ) {
            Text(label, style = MaterialTheme.typography.labelLarge)
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
            if (looksLikeImagePath(value)) {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Text(
                        text = "$prefix$label$delimiter",
                        style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold),
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    MediaImage(
                        rawUrl = value,
                        sourceDir = sourceDir,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(72.dp)
                            .clip(RoundedCornerShape(GenUiTokens.RadiusMd))
                            .border(
                                GenUiTokens.BorderMd,
                                MaterialTheme.colorScheme.outlineVariant,
                                RoundedCornerShape(GenUiTokens.RadiusMd)
                            ),
                        contentScale = ContentScale.Fit,
                        asIcon = false
                    )
                }
                return
            }

            if (item.contains("**")) {
                MarkdownText(
                    text = item,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.fillMaxWidth()
                )
                return
            }

            Text(
                text = remember(prefix, label, delimiter, value) {
                    formatLabelValueText(prefix, label, delimiter, value)
                },
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
        val hasMarkdownBold = text.contains("**")
        Text(
            text = remember(text) { parseBoldMarkdown(text) },
            style = if (hasMarkdownBold) style.copy(fontWeight = FontWeight.Normal) else style,
            color = color,
            modifier = modifier,
            maxLines = maxLines,
            overflow = overflow
        )
    }

    private fun parseBoldMarkdown(text: String): AnnotatedString {
        val markdownParsed = buildAnnotatedString {
            var cursor = 0
            while (cursor < text.length) {
                val open = text.indexOf("**", cursor)
                if (open < 0) {
                    append(text.substring(cursor))
                    break
                }

                val close = text.indexOf("**", open + 2)
                if (close < 0) {
                    append(text.substring(cursor))
                    break
                }

                if (open > cursor) {
                    append(text.substring(cursor, open))
                }
                pushStyle(SpanStyle(fontWeight = FontWeight.W700))
                append(text.substring(open + 2, close))
                pop()
                cursor = close + 2
            }
        }

        val labelRanges = findLeadingLabelRanges(markdownParsed.text)
        if (labelRanges.isEmpty()) {
            return markdownParsed
        }

        return buildAnnotatedString {
            append(markdownParsed)
            labelRanges.forEach { range ->
                addStyle(
                    style = SpanStyle(fontWeight = FontWeight.W700),
                    start = range.first,
                    end = range.last + 1
                )
            }
        }
    }

    private fun formatLabelValueText(
        prefix: String,
        label: String,
        delimiter: String,
        value: String
    ): AnnotatedString {
        return buildAnnotatedString {
            if (prefix.isNotBlank()) {
                append(prefix)
            }
            pushStyle(SpanStyle(fontWeight = FontWeight.W700))
            append(label)
            append(delimiter)
            pop()
            append(' ')
            append(value)
        }
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

    private fun findLeadingLabelRanges(text: String): List<IntRange> {
        val ranges = mutableListOf<IntRange>()
        var offset = 0
        text.split('\n').forEach { line ->
            val trimmed = line.trim()
            if (trimmed.isNotEmpty() &&
                !trimmed.contains("http://", ignoreCase = true) &&
                !trimmed.contains("https://", ignoreCase = true)
            ) {
                INLINE_LABEL_REGEX.findAll(line).forEach { match ->
                    val label = match.groups[1]?.value?.trim().orEmpty()
                    val matchStart = match.range.first
                    val valueText = line.substring(match.range.last + 1).trimStart()
                    if (valueText.isNotEmpty() &&
                        isLikelyLeadingLabel(label, valueText) &&
                        isSentenceBoundary(line, matchStart)
                    ) {
                        ranges += (offset + match.range.first)..(offset + match.range.last)
                    }
                }
            }
            offset += line.length + 1
        }
        return ranges
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
                    modifier = Modifier.fillMaxWidth(),
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
                                text = option.title,
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                                color = MaterialTheme.colorScheme.onSurface,
                                modifier = Modifier.weight(1f)
                            )
                        }

                        Text(
                            text = option.details,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )

                        option.button?.let { button ->
                            val resolved = resolveExternalUrl(button.url, sourceDir)
                            ExternalActionButton(button.label, resolved, onOpenExternalUrl)
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
                modifier = Modifier.fillMaxWidth(),
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
                                text = icon.label,
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
                    modifier = Modifier.fillMaxWidth(),
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
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(132.dp)
                                .clip(RoundedCornerShape(GenUiTokens.RadiusMd))
                                .border(
                                    GenUiTokens.BorderMd,
                                    MaterialTheme.colorScheme.outlineVariant,
                                    RoundedCornerShape(GenUiTokens.RadiusMd)
                                ),
                            contentScale = ContentScale.Fit,
                            asIcon = false
                        )
                        Text(
                            text = entry.label,
                            style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold),
                            color = MaterialTheme.colorScheme.onSurface
                        )

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
                                        Text(
                                            text = icon.label,
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis
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
    private fun RenderTextTable(rows: List<List<String>>) {
        if (rows.size < 2) {
            return
        }
        val header = rows.first()
        val body = rows.drop(1)
        val columnCount = rows.maxOf { it.size }.coerceAtLeast(2)
        val columnWidths = List(columnCount) { tableBaseCellWidth(columnCount) }
        val dark = isSystemInDarkTheme()
        val dividerColor = MaterialTheme.colorScheme.outlineVariant.copy(alpha = if (dark) 0.24f else 0.18f)

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .border(GenUiTokens.BorderMd, genUiCardBorderColor(), RoundedCornerShape(GenUiTokens.RadiusXl))
                .clip(RoundedCornerShape(GenUiTokens.RadiusXl))
                .background(genUiCardContainerColor(GenUiCardTone.Neutral).copy(alpha = if (dark) 0.44f else 0.54f))
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

    @Composable
    private fun RenderImageComponent(component: JsonObject, sourceDir: File?) {
        val rawUrl = readDynamicString(component.get("url"))
        if (rawUrl.isBlank()) {
            Text("Missing image source", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
            return
        }

        val variant = (component.getString("variant") ?: "").lowercase(Locale.US)
        val fit = (component.getString("fit") ?: "contain").normalizeLayoutToken()
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
                mediumFeature && rasterImage -> Modifier
                    .fillMaxWidth()
                    .height(168.dp)
                    .clip(RoundedCornerShape(GenUiTokens.RadiusMd))
                    .border(
                        GenUiTokens.BorderMd,
                        MaterialTheme.colorScheme.outlineVariant,
                        RoundedCornerShape(GenUiTokens.RadiusMd)
                    )
                variant.contains("feature") -> Modifier
                    .fillMaxWidth()
                    .height(170.dp)
                    .clip(RoundedCornerShape(GenUiTokens.RadiusMd))
                    .border(
                        GenUiTokens.BorderMd,
                        MaterialTheme.colorScheme.outlineVariant,
                        RoundedCornerShape(GenUiTokens.RadiusMd)
                    )
                variant.contains("thumbnail") -> Modifier
                    .fillMaxWidth()
                    .height(92.dp)
                    .clip(RoundedCornerShape(GenUiTokens.RadiusMd))
                    .border(
                        GenUiTokens.BorderMd,
                        MaterialTheme.colorScheme.outlineVariant,
                        RoundedCornerShape(GenUiTokens.RadiusMd)
                    )
                else -> Modifier
                    .fillMaxWidth()
                    .height(if (urlLower.endsWith(".svg")) 56.dp else 150.dp)
                    .clip(RoundedCornerShape(GenUiTokens.RadiusMd))
                    .border(
                        GenUiTokens.BorderMd,
                        MaterialTheme.colorScheme.outlineVariant,
                        RoundedCornerShape(GenUiTokens.RadiusMd)
                    )
            },
            contentScale = if (fit == "cover") ContentScale.Crop else ContentScale.Fit,
            asIcon = inlineIconLike
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
                Text(label, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
            }
            return
        }

        Button(
            onClick = { resolvedUrl?.let(onOpenExternalUrl) },
            enabled = resolvedUrl != null,
            shape = RoundedCornerShape(GenUiTokens.RadiusPill)
        ) {
            Text(label, style = MaterialTheme.typography.labelLarge)
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
                val title = readDynamicString(tab?.get("title")).ifBlank { "Tab ${tabIndex + 1}" }
                val childId = tab?.getString("child")

                Card(
                    modifier = Modifier.fillMaxWidth(),
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
        asIcon: Boolean
    ) {
        val context = LocalContext.current
        val model = remember(rawUrl, sourceDir) { resolveImageModel(rawUrl, sourceDir) }
        val imageLoader = remember(context) {
            ImageLoader.Builder(context)
                .components { add(SvgDecoder.Factory()) }
                .build()
        }

        AsyncImage(
            model = ImageRequest.Builder(context).data(model).crossfade(true).build(),
            imageLoader = imageLoader,
            contentDescription = null,
            modifier = modifier,
            contentScale = contentScale
        )
    }

    @Composable
    private fun RenderTableSpec(spec: TableSpec) {
        val columnCount = listOfNotNull(spec.header?.size, spec.rows.maxOfOrNull { it.size })
            .maxOrNull()
            ?.coerceAtLeast(2)
            ?: return
        val hasExplicitWeights =
            spec.header?.any { abs(it.weight - 1f) > 0.01f } == true ||
                spec.rows.any { row -> row.any { abs(it.weight - 1f) > 0.01f } }
        val columnWidths = tableColumnWidths(spec, columnCount, hasExplicitWeights)
        val dark = isSystemInDarkTheme()
        val dividerColor = MaterialTheme.colorScheme.outlineVariant.copy(alpha = if (dark) 0.24f else 0.18f)

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .border(GenUiTokens.BorderMd, genUiCardBorderColor(), RoundedCornerShape(GenUiTokens.RadiusXl))
                .clip(RoundedCornerShape(GenUiTokens.RadiusXl))
                .background(genUiCardContainerColor(GenUiCardTone.Neutral).copy(alpha = if (dark) 0.44f else 0.54f))
                .horizontalScroll(rememberScrollState())
        ) {
            spec.header?.let { header ->
                RenderWeightedRow(
                    cells = header,
                    columnCount = columnCount,
                    columnWidths = columnWidths,
                    isHeader = true,
                    rowIndex = 0
                )
                HorizontalDivider(color = dividerColor)
            }
            spec.rows.forEachIndexed { index, row ->
                RenderWeightedRow(
                    cells = row,
                    columnCount = columnCount,
                    columnWidths = columnWidths,
                    isHeader = false,
                    rowIndex = index
                )
                if (index != spec.rows.lastIndex) {
                    HorizontalDivider(color = dividerColor)
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
                .fillMaxWidth()
                .height(IntrinsicSize.Min)
                .background(tableRowBackground(isHeader = isHeader, rowIndex = rowIndex))
        ) {
            for (column in 0 until columnCount) {
                val cell = cells.getOrNull(column)
                val displayText = cell?.text.orEmpty().trim()
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
                    maxLines = if (isHeader) 3 else 5,
                    overflow = TextOverflow.Ellipsis,
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
                .fillMaxWidth()
                .height(IntrinsicSize.Min)
                .background(tableRowBackground(isHeader = isHeader, rowIndex = rowIndex))
        ) {
            for (column in 0 until columnCount) {
                val value = values.getOrNull(column).orEmpty().trim()
                MarkdownText(
                    text = value,
                    style = if (isHeader) {
                        MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold)
                    } else {
                        MaterialTheme.typography.bodySmall
                    },
                    color = MaterialTheme.colorScheme.onSurface,
                    maxLines = if (isHeader) 3 else 5,
                    overflow = TextOverflow.Ellipsis,
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

    @Composable
    private fun TableCellDivider() {
        val dark = isSystemInDarkTheme()
        Box(
            modifier = Modifier
                .fillMaxHeight()
                .width(GenUiTokens.BorderSm)
                .background(MaterialTheme.colorScheme.outlineVariant.copy(alpha = if (dark) 0.24f else 0.18f))
        )
    }

    @Composable
    private fun tableRowBackground(isHeader: Boolean, rowIndex: Int): Color {
        val dark = isSystemInDarkTheme()
        return when {
            isHeader -> MaterialTheme.colorScheme.onSurface.copy(alpha = if (dark) 0.14f else 0.08f)
            rowIndex % 2 == 0 -> Color.Transparent
            else -> MaterialTheme.colorScheme.onSurface.copy(alpha = if (dark) 0.08f else 0.05f)
        }
    }

    private fun tableBaseCellWidth(columnCount: Int): Dp {
        return if (columnCount <= 3) 152.dp else 132.dp
    }

    private fun tableColumnWidths(spec: TableSpec, columnCount: Int, weightedLayout: Boolean): List<Dp> {
        val baseWidth = tableBaseCellWidth(columnCount)
        if (!weightedLayout) {
            return List(columnCount) { baseWidth }
        }

        val columnWeights = MutableList(columnCount) { 1f }
        fun accumulate(cells: List<TableCell>) {
            for (column in 0 until columnCount) {
                val weight = cells.getOrNull(column)?.weight?.coerceIn(0.6f, 3.5f) ?: 1f
                columnWeights[column] = maxOf(columnWeights[column], weight)
            }
        }

        spec.header?.let(::accumulate)
        spec.rows.forEach(::accumulate)
        return columnWeights.map { baseWidth * it.coerceIn(0.6f, 3.5f) }
    }

    private fun maybeExtractTable(children: List<String>, index: Map<String, JsonObject>): TableSpec? {
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

        var hasWeight = false
        val textRows = mutableListOf<List<TableCell>>()
        for (cellIds in rowCells) {
            val cells = mutableListOf<TableCell>()
            for (cellId in cellIds) {
                val cell = index[cellId] ?: return null
                if (cell.getString("component") != "Text") {
                    return null
                }

                val text = readDynamicString(cell.get("text"))
                val variant = (cell.getString("variant") ?: "body").lowercase(Locale.US)
                val weight = (cell.getAsNumberOrNull("weight") ?: 1.0).toFloat().coerceAtLeast(0.6f)
                if (weight > 1f) hasWeight = true
                cells += TableCell(text, variant, weight)
            }
            textRows += cells
        }

        val headerLikeVariants = setOf("h1", "h2", "h3", "h4", "h5", "h6")
        val headerLike = textRows.first().all { it.variant in headerLikeVariants } ||
            rowCells.first().all { it.startsWith("th-", ignoreCase = true) || it.startsWith("th_", ignoreCase = true) }
        if (!hasWeight && !headerLike) {
            return null
        }

        val header = if (headerLike) textRows.first() else null
        val body = if (headerLike) textRows.drop(1) else textRows
        if (body.isEmpty()) {
            return null
        }
        return TableSpec(header = header, rows = body)
    }

    private fun maybeExtractTextList(children: List<String>, index: Map<String, JsonObject>): List<String>? {
        if (children.size < 2) {
            return null
        }

        val items = mutableListOf<String>()
        for (childId in children) {
            val child = index[childId] ?: return null
            if (child.getString("component") != "Text") {
                return null
            }
            items += readDynamicString(child.get("text"))
        }
        return items
    }

    @Composable
    private fun textStyleForVariant(variant: String): TextStyle = when (variant.lowercase(Locale.US)) {
        "h1" -> MaterialTheme.typography.displayLarge
        "h2" -> MaterialTheme.typography.displayMedium
        "h3" -> MaterialTheme.typography.displaySmall
        "h4" -> MaterialTheme.typography.titleLarge
        "h5" -> MaterialTheme.typography.titleMedium
        "h6" -> MaterialTheme.typography.titleSmall
        "caption" -> MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Normal)
        else -> MaterialTheme.typography.bodyMedium
    }

    @Composable
    private fun textColorForVariant(variant: String): Color = when (variant.lowercase(Locale.US)) {
        "h1", "h2", "h3", "h4", "h5", "h6" -> MaterialTheme.colorScheme.onSurface
        "caption" -> MaterialTheme.colorScheme.onSurfaceVariant
        else -> MaterialTheme.colorScheme.onSurface
    }

    private fun shouldUseStructuredBlocks(rawText: String, variant: String): Boolean {
        if (variant in setOf("h1", "h2", "h3", "h4")) {
            return false
        }
        val normalized = rawText.replace("\r\n", "\n")
        val lines = normalized.lines().map { it.trim() }.filter { it.isNotEmpty() }
        if (lines.isEmpty()) {
            return false
        }

        val hasButtons = normalized.contains("[Button:", ignoreCase = true)
        val hasTable = lines.count { isTableLikeLine(it) } >= 2
        val hasBullets = lines.count { it.startsWith("- ") } >= 2
        return hasButtons || hasTable || hasBullets || normalized.contains("\n\n")
    }

    private fun parseTextBlocks(rawText: String): List<TextBlock> {
        val lines = rawText.replace("\r\n", "\n").split('\n')
        val blocks = mutableListOf<TextBlock>()
        var index = 0
        var renderedAny = false
        var inSourcesSection = false

        while (index < lines.size) {
            val line = lines[index].trim()
            if (line.isEmpty()) {
                index++
                continue
            }

            val sourceLinks = collectSourceLinks(lines, index, inSourcesSection)
            if (sourceLinks != null) {
                val (links, nextIndex) = sourceLinks
                if (links.isNotEmpty()) {
                    blocks += TextBlock.Sources(links)
                    renderedAny = true
                    index = nextIndex
                    continue
                }
            }

            val bookingOptions = collectBookingOptions(lines, index)
            if (bookingOptions != null) {
                val (options, nextIndex) = bookingOptions
                if (options.isNotEmpty()) {
                    blocks += TextBlock.BookingCards(options)
                    renderedAny = true
                    index = nextIndex
                    continue
                }
            }

            val tableRows = collectTableRows(lines, index)
            if (tableRows != null) {
                val (rows, nextIndex) = tableRows
                blocks += TextBlock.Table(rows)
                renderedAny = true
                index = nextIndex
                continue
            }

            val mediaEntries = collectMediaEntries(lines, index)
            if (mediaEntries != null) {
                val (entries, nextIndex) = mediaEntries
                if (entries.isNotEmpty()) {
                    blocks += TextBlock.MediaCards(entries)
                    renderedAny = true
                }
                index = nextIndex
                continue
            }

            val numberedSteps = collectNumberedSteps(lines, index)
            if (numberedSteps != null) {
                val (steps, nextIndex) = numberedSteps
                if (steps.isNotEmpty()) {
                    blocks += TextBlock.NumberedSteps(steps)
                    renderedAny = true
                    index = nextIndex
                    continue
                }
            }

            if (line.startsWith("- ")) {
                val items = mutableListOf<String>()
                var cursor = index
                while (cursor < lines.size) {
                    val l = lines[cursor].trim()
                    if (!l.startsWith("- ")) break
                    items += l.removePrefix("- ").trim()
                    cursor++
                }
                blocks += TextBlock.Bullets(items)
                renderedAny = true
                index = cursor
                continue
            }

            val actions = mutableListOf<ParsedButton>()
            var actionCursor = index
            while (actionCursor < lines.size) {
                val parsed = parseButtonLine(lines[actionCursor].trim()) ?: break
                actions += parsed
                actionCursor++
            }
            if (actions.isNotEmpty()) {
                blocks += TextBlock.Actions(actions)
                renderedAny = true
                index = actionCursor
                continue
            }

            if (!renderedAny) {
                blocks += TextBlock.Title(line)
                renderedAny = true
                index++
                continue
            }

            if (looksLikeSectionHeading(line)) {
                blocks += TextBlock.Heading(line)
                val normalizedHeading = line.lowercase(Locale.US).removeSuffix(":").trim()
                inSourcesSection =
                    normalizedHeading.startsWith("sources") || normalizedHeading.startsWith("references")
                index++
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
                blocks += TextBlock.Paragraph(paragraphLines.joinToString(" "))
                index = cursor
                continue
            }

            index++
        }

        return blocks
    }

    private fun collectSourceLinks(
        lines: List<String>,
        startIndex: Int,
        inSourcesSection: Boolean
    ): Pair<List<ParsedButton>, Int>? {
        val firstLine = lines[startIndex].trim()
        val firstLower = firstLine.lowercase(Locale.US)
        val sourcePrefixed = firstLower.startsWith("sources") || firstLower.startsWith("references")
        if (!inSourcesSection && !sourcePrefixed) {
            return null
        }

        var cursor = startIndex
        val links = mutableListOf<ParsedButton>()
        while (cursor < lines.size) {
            val line = lines[cursor].trim()
            if (line.isEmpty()) {
                cursor++
                continue
            }

            if (cursor != startIndex && looksLikeSectionHeading(line)) {
                break
            }

            val parsed = parseSourceLinksFromLine(line)
            if (parsed.isEmpty()) {
                if (links.isNotEmpty()) {
                    break
                }
                return null
            }
            links += parsed
            cursor++
        }

        if (links.isEmpty()) {
            return null
        }
        val distinct = links.distinctBy { it.url.lowercase(Locale.US) }
        return distinct to cursor
    }

    private fun parseSourceLinksFromLine(line: String): List<ParsedButton> {
        val normalized = line.removePrefix("- ").trim()
        val urlMatches = URL_REGEX.findAll(normalized).toList()
        if (urlMatches.isEmpty()) {
            return emptyList()
        }

        val links = mutableListOf<ParsedButton>()
        var cursor = 0
        urlMatches.forEachIndexed { index, match ->
            val rawUrl = sanitizeUrlToken(match.value)
            val labelChunk = normalized.substring(cursor, match.range.first).trim()
            var label = labelChunk.removeSuffix(":").trim().trim('|')
            if (index == 0) {
                label = label.replace(Regex("""^(sources?|references?)\s*:?\s*""", RegexOption.IGNORE_CASE), "")
            }
            if (label.isBlank()) {
                label = Uri.parse(rawUrl).host?.removePrefix("www.").orEmpty().ifBlank { "Source ${index + 1}" }
            }
            links += ParsedButton(
                label = label,
                url = rawUrl
            )
            cursor = match.range.last + 1
        }
        return links
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
        cursor++

        val logos = mutableListOf<ParsedLogo>()
        while (cursor < lines.size) {
            val line = lines[cursor].trim()
            if (line.isEmpty()) {
                cursor++
                continue
            }
            if (!line.startsWith("- ")) {
                break
            }

            val item = line.removePrefix("- ").trim()
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

        val haystack = normalizeMatchText(option.title + " " + (option.button?.label ?: ""))
        if (haystack.isBlank()) {
            return null
        }

        return logos.firstOrNull { logo ->
            val needle = normalizeMatchText(logo.label)
            needle.isNotBlank() && (haystack.contains(needle) || needle.contains(haystack))
        }
    }

    private fun assignIconsToPrimary(
        primary: List<ParsedMediaEntry>,
        icons: List<ParsedMediaEntry>
    ): List<List<ParsedMediaEntry>> {
        if (primary.isEmpty()) {
            return emptyList()
        }

        val buckets = MutableList(primary.size) { mutableListOf<ParsedMediaEntry>() }
        val unmatchedContextual = mutableListOf<ParsedMediaEntry>()
        val primaryKeys = primary.map { normalizeMatchText(it.label) }

        icons.forEach { icon ->
            val iconKey = normalizeMatchText(icon.label)
            if (iconKey.isBlank()) {
                return@forEach
            }

            val matchedIndex = primaryKeys.indexOfFirst { key ->
                key.isNotBlank() && (key.contains(iconKey) || iconKey.contains(key))
            }
            if (matchedIndex >= 0) {
                buckets[matchedIndex] += icon
                return@forEach
            }

            if (isContextualInlineIconLabel(icon.label)) {
                unmatchedContextual += icon
            }
        }

        unmatchedContextual.forEachIndexed { index, icon ->
            buckets[index % buckets.size] += icon
        }
        return buckets
    }

    private fun isContextualInlineIconLabel(label: String): Boolean {
        val normalized = normalizeMatchText(label)
        return normalized in setOf(
            "airline",
            "time",
            "clock",
            "calendar",
            "date",
            "fire",
            "basket",
            "meal",
            "schedule"
        )
    }

    private fun normalizeMatchText(value: String): String {
        return value
            .lowercase(Locale.US)
            .replace(Regex("[^a-z0-9]+"), " ")
            .trim()
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
            return emptyList<ParsedMediaEntry>() to cursor
        }

        return entries to cursor
    }

    private fun isGenericUtilityIconLabel(label: String): Boolean {
        val normalized = normalizeMatchText(label)
        return normalized in setOf("airline", "time", "clock", "calendar", "date")
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
        val normalized = line.removePrefix("- ").trim()
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

    private fun collectNumberedSteps(
        lines: List<String>,
        startIndex: Int
    ): Pair<List<StepEntry>, Int>? {
        val firstLine = lines[startIndex].trim()
        if (NUMBERED_STEP_REGEX.matchEntire(firstLine) == null) {
            return null
        }

        val steps = mutableListOf<StepEntry>()
        var cursor = startIndex
        while (cursor < lines.size) {
            while (cursor < lines.size && lines[cursor].trim().isEmpty()) {
                cursor++
            }
            if (cursor >= lines.size) {
                break
            }

            val current = lines[cursor].trim()
            val match = NUMBERED_STEP_REGEX.matchEntire(current) ?: break
            val stepTitle = "${match.groupValues[1]}. ${match.groupValues[2].trim().removeSuffix(":")}"
            cursor++

            val detailLines = mutableListOf<String>()
            while (cursor < lines.size) {
                val candidate = lines[cursor].trim()
                if (candidate.isEmpty()) {
                    cursor++
                    if (detailLines.isNotEmpty()) {
                        break
                    }
                    continue
                }
                if (NUMBERED_STEP_REGEX.matchEntire(candidate) != null || isStructuredBoundary(candidate)) {
                    break
                }
                detailLines += candidate
                cursor++
            }

            steps += StepEntry(
                title = stepTitle,
                details = detailLines.joinToString(" ")
            )
        }

        if (steps.size < 2) {
            return null
        }
        return steps to cursor
    }

    private fun collectTableRows(lines: List<String>, startIndex: Int): Pair<List<List<String>>, Int>? {
        if (!isTableLikeLine(lines[startIndex].trim())) {
            return null
        }

        val rows = mutableListOf<List<String>>()
        var cursor = startIndex
        while (cursor < lines.size) {
            val candidate = lines[cursor].trim()
            if (!isTableLikeLine(candidate)) break
            rows += splitTableCells(candidate)
            cursor++
        }
        if (rows.size < 2) {
            return null
        }
        return rows to cursor
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
        val match = BUTTON_LINE_REGEX.find(line) ?: return null
        val label = match.groupValues[1].trim()
        val url = sanitizeUrlToken(match.groupValues[2])
        if (label.isBlank() || url.isBlank()) {
            return null
        }
        return ParsedButton(label, url)
    }

    private fun sanitizeUrlToken(value: String): String =
        value.trim().trimEnd('.', ',', ';')

    private fun isTableLikeLine(line: String): Boolean {
        if (line.contains("http://", ignoreCase = true) || line.contains("https://", ignoreCase = true)) {
            return false
        }
        return splitTableCells(line).size >= 3
    }

    private fun splitTableCells(line: String): List<String> =
        line.split('|').map { it.trim() }.filter { it.isNotEmpty() }

    private fun looksLikeSectionHeading(line: String): Boolean {
        if (line.length > 80) {
            return false
        }
        if (line.startsWith("- ") || line.endsWith(".") || line.endsWith("?") || line.endsWith("!")) {
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
        if (line.startsWith("- ")) {
            return true
        }
        if (parseButtonLine(line) != null || parseOptionLine(line) != null || NUMBERED_STEP_REGEX.matchEntire(line) != null) {
            return true
        }
        if (isTableLikeLine(line) || looksLikeSectionHeading(line)) {
            return true
        }
        return false
    }

    private fun looksLikeImagePath(value: String): Boolean {
        val normalized = value.trim().lowercase(Locale.US)
        return normalized.endsWith(".png") ||
            normalized.endsWith(".jpg") ||
            normalized.endsWith(".jpeg") ||
            normalized.endsWith(".svg") ||
            normalized.endsWith(".webp")
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
        val scheme = runCatching { Uri.parse(value).scheme?.lowercase(Locale.US) }.getOrNull()
        return if (scheme == "http" || scheme == "https") value else null
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
            if (arr != null) return arr
        }
        return null
    }

    private fun extractSurfaces(messages: JsonArray, warnings: MutableList<String>): List<SurfaceState> {
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

    private fun buildSurfaceStates(
        surfaceId: String,
        components: List<JsonObject>,
        rootHint: String?
    ): List<SurfaceState> {
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

        return listOf(SurfaceState(surfaceId, rootId, index))
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
        val obj = children.asJsonObject
        return obj.getAsJsonArrayOrNull("explicitList") ?: JsonArray()
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

        val explicit = element.asJsonObject.getAsJsonArrayOrNull("explicitList") ?: return emptyList()
        return explicit.mapNotNull { it.asStringOrNull() }
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

    private fun String.normalizeLayoutToken(): String {
        return lowercase(Locale.US)
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
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

