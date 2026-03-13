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
import androidx.compose.material.icons.filled.AcUnit
import androidx.compose.material.icons.filled.Air
import androidx.compose.material.icons.filled.Cloud
import androidx.compose.material.icons.filled.DarkMode
import androidx.compose.material.icons.filled.FlightTakeoff
import androidx.compose.material.icons.filled.Grain
import androidx.compose.material.icons.filled.Thunderstorm
import androidx.compose.material.icons.filled.WbSunny
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
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.isSpecified
import androidx.compose.ui.unit.sp
import coil.ImageLoader
import coil.compose.AsyncImage
import coil.decode.SvgDecoder
import coil.request.ImageRequest
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.File
import java.nio.charset.Charset
import java.time.LocalDate
import java.time.LocalTime
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
    private data class WeatherTableColumns(
        val period: Int,
        val date: Int?,
        val condition: Int?,
        val high: Int?,
        val low: Int?,
        val temp: Int?,
        val precip: Int?,
        val wind: Int?,
        val humidity: Int?,
        val uv: Int?
    )

    private data class WeatherRow(
        val period: String,
        val date: String?,
        val condition: String?,
        val high: String?,
        val low: String?,
        val temp: String?,
        val metrics: List<Pair<String, String>>
    )
    private data class FlightTableColumns(
        val airline: Int,
        val depart: Int?,
        val arrive: Int?,
        val duration: Int?,
        val stops: Int?,
        val fare: Int?,
        val status: Int?
    )

    private data class FlightRow(
        val airline: String,
        val depart: String?,
        val originCode: String?,
        val arrive: String?,
        val destinationCode: String?,
        val duration: String?,
        val stops: String?,
        val fare: String?,
        val status: String?
    )
    private data class FlightPoint(
        val time: String?,
        val code: String?
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
    private data class WeatherCurrentDetails(
        val iconUrl: String?,
        val condition: String?,
        val temperature: String?,
        val feelsLike: String?,
        val humidity: String?,
        val wind: String?,
        val rainChance: String?,
        val uvIndex: String?,
        val summary: String?
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
        Regex("""^(?:Action:\s*)?\[Button:\s*(.+?)\]\s*(.+)$""", RegexOption.IGNORE_CASE)
    private val NUMBERED_STEP_REGEX = Regex("""^(\d+)\.\s+(.+)$""")
    private val INLINE_MEDIA_ASSIGNMENT_REGEX =
        Regex("""(?i)\b(Image|Icon)\s*=\s*(https?://\S+|/assets/\S+|assets/\S+|\S+)""")
    private val MARKDOWN_IMAGE_REGEX =
        Regex("""!\[([^\]]*)\]\((https?://[^\s)]+|//[^\s)]+|/assets/[^\s)]+|assets/[^\s)]+)\)""", RegexOption.IGNORE_CASE)
    private val MARKDOWN_SOURCE_LINK_REGEX =
        Regex(
            """\[(.+?)]\((https?://[^\s)]+|//[^\s)]+|www\.[^\s)]+|(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:[/?#][^\s)]*)?)\)""",
            RegexOption.IGNORE_CASE
        )
    private val BULLET_LINE_REGEX = Regex("""^\s*([\-*\u2022])\s+(.+)$""")
    private val LEADING_LABEL_REGEX =
        Regex("""^\s*([\u2022\-*]\s*)?([^:;\uFF1A\uFF1B\n]{1,70}?)([:;\uFF1A\uFF1B])\s*(.+)$""")
    private val INLINE_LABEL_REGEX =
        Regex("""([\p{L}\p{N}][\p{L}\p{N}/&()' .-]{0,84})([:;\uFF1A\uFF1B])""")

    private val URL_REGEX = Regex(
        """(?i)(?:https?://|//)[^\s<>\]]+|(?<![@\w])(?:www\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:[/?#][^\s<>\]]*)?"""
    )
    private val TABLE_PLACEHOLDER_CELL_REGEX = Regex("""^[:\-\u2013\u2014]+$""")
    private val HOST_LABEL_REGEX = Regex("""(?i)^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$""")

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
        val hasMarkdownBold = displayText.contains("**")
        val hasMarkdownHeading = containsMarkdownHeading(displayText)
        Text(
            text = remember(displayText, style) {
                if (hasMarkdownHeading) {
                    parseMarkdownWithHeadings(displayText, style)
                } else {
                    parseBoldMarkdown(displayText)
                }
            },
            style = if (hasMarkdownBold || hasMarkdownHeading) style.copy(fontWeight = FontWeight.Normal) else style,
            color = color,
            modifier = modifier,
            maxLines = maxLines,
            overflow = overflow
        )
    }

    private fun containsMarkdownHeading(text: String): Boolean =
        text.lineSequence().any { line -> line.trimStart().startsWith("###") }

    private fun parseMarkdownWithHeadings(
        text: String,
        baseStyle: TextStyle
    ): AnnotatedString {
        val headingSize = markdownHeadingFontSize(baseStyle.fontSize)
        return buildAnnotatedString {
            val lines = text.split('\n')
            lines.forEachIndexed { index, rawLine ->
                val trimmedStart = rawLine.trimStart()
                if (trimmedStart.startsWith("###")) {
                    val headingText = trimmedStart.removePrefix("###").trim()
                    if (headingText.isNotEmpty()) {
                        val start = length
                        append(parseBoldMarkdown(headingText))
                        addStyle(
                            style = SpanStyle(
                                fontWeight = FontWeight.W700,
                                fontSize = headingSize
                            ),
                            start = start,
                            end = length
                        )
                    }
                } else {
                    append(parseBoldMarkdown(rawLine))
                }
                if (index != lines.lastIndex) {
                    append('\n')
                }
            }
        }
    }

    private fun markdownHeadingFontSize(baseSize: TextUnit): TextUnit {
        if (!baseSize.isSpecified) {
            return 20.sp
        }
        return baseSize * 1.22f
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
                    append(text.substring(cursor).replace("**", ""))
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

    private fun sanitizeDisplayText(text: String, preserveMarkdown: Boolean = false): String {
        if (text.isBlank()) {
            return text
        }
        var cleaned = normalizeMojibakeText(text)
        if (!preserveMarkdown) {
            cleaned = cleaned.replace(Regex("(?m)^\\s{0,3}#{1,6}\\s*"), "")
        }
        cleaned = cleaned.replace("ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢", "â€¢")
        cleaned = cleaned.replace(Regex("(?i)\\bfor\\s+example\\b\\s*[:,-]?\\s*"), "")
        cleaned = cleaned.replace(Regex("(?i)\\((?:\\s*(?:example|sample|illustrative|demo)\\s*)\\)"), "")
        cleaned = cleaned.replace(Regex("(?i)\\b(?:example|sample|illustrative|demo)s?\\b\\s*:?"), "")
        cleaned = cleaned.replace(Regex("(?m)^[ \\t]*:[ \\t]*$"), "")
        cleaned = cleaned.replace(
            Regex("(?im)^\\s*(accessing|fetching|retrieving|querying|searching)\\s+live\\s+[^\\n]*$"),
            ""
        )
        cleaned = cleaned.replace(
            Regex("(?im)^\\s*(accessing|fetching|retrieving|querying|searching)\\s+[^\\n]*(weather|flight|price|stock|trend|news)[^\\n]*$"),
            ""
        )
        cleaned = cleaned.replace(
            Regex("(?im)^\\s*(?:data\\s+)?as\\s+of\\b[^\\n]*$"),
            ""
        )
        if (!preserveMarkdown) {
            cleaned = cleaned.replace("**", "")
        }
        cleaned = cleaned.replace(Regex("[ \\t]{2,}"), " ")
        cleaned = cleaned.replace(Regex(" *([,.;:])"), "$1")
        cleaned = cleaned.replace(Regex("\\n{3,}"), "\n\n")
        return cleaned.trim()
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
                !containsUrlLikeToken(trimmed)
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
        if (prev == '\u2022' || prev == '-') {
            return true
        }
        if (prev != ' ') {
            return false
        }
        val prevNonSpaceIndex = line.substring(0, start - 1).indexOfLast { !it.isWhitespace() }
        if (prevNonSpaceIndex < 0) {
            return true
        }
        return line[prevNonSpaceIndex] in charArrayOf('.', '!', '?', '\u2022', '-', ')')
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

        buildWeatherRows(header, body)?.let { weatherRows ->
            RenderWeatherRows(weatherRows)
            return
        }
        buildFlightRows(header, body)?.let { flightRows ->
            RenderFlightRows(flightRows)
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
                    // Keep icon slots informative even when remote icon fetch fails.
                    WeatherConditionIcon(
                        condition = fallbackCondition ?: inferWeatherConditionFromIconUrl(rawUrl) ?: "Cloudy",
                        size = iconFallbackSize
                    )
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
        val normalized = rawUrl.lowercase(Locale.US)
        extractWeatherApiIconCode(normalized)?.let { code ->
            val isNight = normalized.contains("/night/")
            mapWeatherApiIconCode(code, isNight = isNight)?.let { return it }
        }
        extractOpenWeatherIconCode(normalized)?.let { code ->
            mapOpenWeatherIconCode(code)?.let { return it }
        }
        return inferWeatherConditionFromTextPool(normalized)
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
            buildWeatherRows(semanticHeader, semanticBody)?.let { weatherRows ->
                RenderWeatherRows(weatherRows)
                return
            }
            buildFlightRows(semanticHeader, semanticBody)?.let { flightRows ->
                RenderFlightRows(flightRows)
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

    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    private fun RenderWeatherRows(rows: List<WeatherRow>) {
        if (rows.isEmpty()) {
            return
        }
        val orderedRows = remember(rows) { orderWeatherRows(rows) }
        val todayRow = orderedRows.first()
        val laterRows = orderedRows.drop(1)
        val todayTemperature = weatherTemperatureText(todayRow)
        val todayCondition = sanitizeDisplayText(todayRow.condition.orEmpty())
        val todayDate = sanitizeDisplayText(todayRow.date.orEmpty()).ifBlank { null }
        val todayLabel = if (isTodayWeatherRow(todayRow)) "Today" else sanitizeDisplayText(todayRow.period)

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(GenUiTokens.RadiusXl),
            colors = CardDefaults.cardColors(containerColor = genUiTableContainerColor()),
            elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
            border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 10.dp, vertical = 10.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
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
                            .padding(horizontal = 12.dp, vertical = 12.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(10.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(
                                modifier = Modifier.weight(1f),
                                horizontalArrangement = Arrangement.spacedBy(10.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                WeatherConditionIcon(
                                    condition = todayRow.condition,
                                    size = 32.dp
                                )
                                Column(
                                    verticalArrangement = Arrangement.spacedBy(2.dp)
                                ) {
                                    Text(
                                        text = todayLabel,
                                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                    todayDate?.let { dateValue ->
                                        Text(
                                            text = dateValue,
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }
                            }

                            if (todayTemperature.isNotBlank()) {
                                Text(
                                    text = sanitizeDisplayText(todayTemperature),
                                    style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Bold),
                                    color = MaterialTheme.colorScheme.onSurface,
                                    textAlign = TextAlign.End
                                )
                            }
                        }

                        if (todayCondition.isNotBlank()) {
                            Text(
                                text = todayCondition,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        if (todayRow.metrics.isNotEmpty()) {
                            FlowRow(
                                horizontalArrangement = Arrangement.spacedBy(6.dp),
                                verticalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                todayRow.metrics.forEach { (label, value) ->
                                    val chipLabel = sanitizeDisplayText(label)
                                    val chipValue = sanitizeDisplayText(value)
                                    if (chipLabel.isNotBlank() && chipValue.isNotBlank()) {
                                        Text(
                                            text = "$chipLabel $chipValue",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            modifier = Modifier
                                                .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                                                .background(genUiCardContainerColor(GenUiCardTone.Neutral))
                                                .border(
                                                    GenUiTokens.BorderMd,
                                                    genUiCardBorderColor(),
                                                    RoundedCornerShape(GenUiTokens.RadiusPill)
                                                )
                                                .padding(horizontal = 8.dp, vertical = 5.dp)
                                        )
                                    }
                                }
                            }
                        }
                    }
                }

                laterRows.forEach { row ->
                    val temperature = weatherTemperatureText(row)
                    val periodText = sanitizeDisplayText(row.period)
                    val dateText = sanitizeDisplayText(row.date.orEmpty()).ifBlank { null }
                    val conditionText = sanitizeDisplayText(row.condition.orEmpty()).ifBlank { null }
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
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(10.dp),
                                verticalAlignment = Alignment.Top
                            ) {
                                Row(
                                    modifier = Modifier.weight(1f),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    verticalAlignment = Alignment.Top
                                ) {
                                    WeatherConditionIcon(
                                        condition = row.condition,
                                        size = 26.dp
                                    )
                                    Column(
                                        verticalArrangement = Arrangement.spacedBy(2.dp)
                                    ) {
                                        Text(
                                            text = periodText,
                                            style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                                            color = MaterialTheme.colorScheme.onSurface
                                        )
                                        dateText?.let { dateValue ->
                                            Text(
                                                text = dateValue,
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                maxLines = 1,
                                                overflow = TextOverflow.Ellipsis
                                            )
                                        }
                                    }
                                }
                                if (temperature.isNotBlank()) {
                                    Text(
                                        text = sanitizeDisplayText(temperature),
                                        style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
                                        color = MaterialTheme.colorScheme.onSurface,
                                        textAlign = TextAlign.End
                                    )
                                }
                            }

                            conditionText?.let { condition ->
                                Text(
                                    text = condition,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }

                            if (row.metrics.isNotEmpty()) {
                                FlowRow(
                                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                                    verticalArrangement = Arrangement.spacedBy(6.dp)
                                ) {
                                    row.metrics.forEach { (label, value) ->
                                        val chipLabel = sanitizeDisplayText(label)
                                        val chipValue = sanitizeDisplayText(value)
                                        if (chipLabel.isNotBlank() && chipValue.isNotBlank()) {
                                            Text(
                                                text = "$chipLabel $chipValue",
                                                style = MaterialTheme.typography.labelSmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                modifier = Modifier
                                                    .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                                                    .background(genUiCardContainerColor(GenUiCardTone.Neutral))
                                                    .border(
                                                        GenUiTokens.BorderMd,
                                                        genUiCardBorderColor(),
                                                        RoundedCornerShape(GenUiTokens.RadiusPill)
                                                    )
                                                    .padding(horizontal = 8.dp, vertical = 5.dp)
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
    private fun RenderFlightRows(rows: List<FlightRow>) {
        if (rows.isEmpty()) {
            return
        }

        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            rows.forEach { row ->
                val airline = sanitizeDisplayText(row.airline)
                val fare = sanitizeDisplayText(row.fare.orEmpty()).ifBlank { null }
                val (fareValue, fareMeta) = splitFareDisplay(fare)
                val depart = sanitizeDisplayText(row.depart.orEmpty()).ifBlank { null }
                val arrive = sanitizeDisplayText(row.arrive.orEmpty()).ifBlank { null }
                val originCode = sanitizeDisplayText(row.originCode.orEmpty())
                    .ifBlank { extractAirportCode(depart.orEmpty()).orEmpty() }
                    .ifBlank { null }
                val destinationCode = sanitizeDisplayText(row.destinationCode.orEmpty())
                    .ifBlank { extractAirportCode(arrive.orEmpty()).orEmpty() }
                    .ifBlank { null }
                val departPoint = parseFlightPoint(depart, originCode)
                val arrivePoint = parseFlightPoint(arrive, destinationCode)
                val departDisplay = departPoint.time ?: departPoint.code ?: depart
                val arriveDisplay = arrivePoint.time ?: arrivePoint.code ?: arrive
                val duration = normalizeDurationLabel(row.duration)
                val stopLabel = normalizeStopLabel(row.stops, row.status, depart, arrive)
                val statusLabel = normalizeFlightStatus(row.status, stopLabel)
                val topStatus = statusLabel?.takeIf { looksLikePunctualityStatus(it) }
                val arrivalStatus = statusLabel?.takeUnless { looksLikePunctualityStatus(it) }
                val centerMeta = stopLabel ?: arrivalStatus?.takeIf { it.length <= 22 }
                val promoMeta = arrivalStatus?.takeIf { it.length > 22 }
                val hasTimeRow = departDisplay != null || arriveDisplay != null || duration != null

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
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(10.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(
                                modifier = Modifier.weight(1f),
                                horizontalArrangement = Arrangement.spacedBy(10.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                AirlineBadge(airline = airline)
                                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                    Text(
                                        text = airline,
                                        style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                    topStatus?.let { statusValue ->
                                        Text(
                                            text = statusValue,
                                            style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold),
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }
                            }
                            if (fareValue != null) {
                                Column(
                                    modifier = Modifier.sizeIn(minWidth = 84.dp),
                                    horizontalAlignment = Alignment.End,
                                    verticalArrangement = Arrangement.spacedBy(2.dp)
                                ) {
                                    Text(
                                        text = fareValue,
                                        style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
                                        color = MaterialTheme.colorScheme.onSurface,
                                        textAlign = TextAlign.End
                                    )
                                    fareMeta?.let { fareSuffix ->
                                        Text(
                                            text = fareSuffix,
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            textAlign = TextAlign.End
                                        )
                                    }
                                }
                            }
                        }

                        if (hasTimeRow) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(10.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                FlightTimeCell(
                                    title = departDisplay,
                                    subtitle = departPoint.code,
                                    align = TextAlign.Start,
                                    modifier = Modifier.weight(1f),
                                    emphasis = true
                                )
                                Column(
                                    modifier = Modifier.weight(1.1f),
                                    verticalArrangement = Arrangement.spacedBy(5.dp),
                                    horizontalAlignment = Alignment.CenterHorizontally
                                ) {
                                    duration?.let { durationValue ->
                                        Text(
                                            text = durationValue,
                                            style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                                            color = MaterialTheme.colorScheme.onSurface,
                                            textAlign = TextAlign.Center
                                        )
                                    }
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(horizontal = 2.dp),
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                                    ) {
                                        HorizontalDivider(
                                            modifier = Modifier.weight(1f),
                                            color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.75f)
                                        )
                                        Icon(
                                            imageVector = Icons.Filled.FlightTakeoff,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.onSurfaceVariant,
                                            modifier = Modifier.size(14.dp)
                                        )
                                        HorizontalDivider(
                                            modifier = Modifier.weight(1f),
                                            color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.75f)
                                        )
                                    }
                                    centerMeta?.let { meta ->
                                        FlightMetaChip(meta)
                                    }
                                }
                                FlightTimeCell(
                                    title = arriveDisplay,
                                    subtitle = arrivePoint.code,
                                    align = TextAlign.End,
                                    modifier = Modifier.weight(1f),
                                    emphasis = true
                                )
                            }
                            promoMeta?.let { promo ->
                                Text(
                                    text = promo,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.fillMaxWidth()
                                )
                            }
                        } else {
                            val meta = listOfNotNull(stopLabel, arrivalStatus, fareValue, fareMeta)
                            if (meta.isNotEmpty()) {
                                Text(
                                    text = meta.joinToString(" | "),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    private fun parseFlightPoint(value: String?, fallbackCode: String?): FlightPoint {
        val raw = sanitizeDisplayText(value.orEmpty()).trim()
        val fallback = sanitizeDisplayText(fallbackCode.orEmpty()).ifBlank { null }
        if (raw.isBlank()) {
            return FlightPoint(time = null, code = fallback)
        }

        val time = normalizeFlightTime(raw)
        val inlineCode = Regex("""\b([A-Z]{3})\b""")
            .find(raw.uppercase(Locale.US))
            ?.groupValues
            ?.getOrNull(1)
            ?.uppercase(Locale.US)
        return FlightPoint(
            time = time,
            code = inlineCode ?: fallback
        )
    }

    private fun normalizeFlightTime(value: String): String? {
        val cleaned = sanitizeDisplayText(value).trim()
        if (cleaned.isBlank()) {
            return null
        }
        val match = Regex("""\b(\d{1,2}:\d{2})(?:\s?(AM|PM))?(?:\+(\d+))?\b""", RegexOption.IGNORE_CASE)
            .find(cleaned)
            ?: return null
        val hhmm = match.groupValues[1]
        val suffix = match.groupValues.getOrNull(2).orEmpty().uppercase(Locale.US)
        val dayOffset = match.groupValues.getOrNull(3).orEmpty()
        val ampm = if (suffix.isBlank()) "" else " $suffix"
        val plus = if (dayOffset.isBlank()) "" else "+$dayOffset"
        return "$hhmm$ampm$plus"
    }

    private fun normalizeDurationLabel(value: String?): String? {
        val raw = sanitizeDisplayText(value.orEmpty()).trim()
        if (raw.isBlank()) {
            return null
        }
        val match = Regex("""(?i)(\d+)\s*h(?:ours?)?\s*(?:(\d+)\s*m(?:in(?:ute)?s?)?)?""").find(raw)
        if (match != null) {
            val hours = match.groupValues[1]
            val mins = match.groupValues.getOrNull(2).orEmpty()
            return if (mins.isBlank()) "${hours}h" else "${hours}h ${mins}m"
        }
        val minsOnly = Regex("""(?i)(\d+)\s*m(?:in(?:ute)?s?)?""").find(raw)?.groupValues?.getOrNull(1)
        if (!minsOnly.isNullOrBlank()) {
            return "${minsOnly}m"
        }
        return raw
    }

    @Composable
    private fun AirlineBadge(airline: String) {
        val accent = airlineAccentColor(airline)
        val code = airlineBadgeCode(airline)
        Box(
            modifier = Modifier
                .size(34.dp)
                .clip(RoundedCornerShape(10.dp))
                .background(accent.copy(alpha = 0.18f))
                .border(
                    width = GenUiTokens.BorderMd,
                    color = accent.copy(alpha = 0.45f),
                    shape = RoundedCornerShape(10.dp)
                ),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Filled.FlightTakeoff,
                contentDescription = null,
                tint = accent,
                modifier = Modifier.size(15.dp)
            )
            if (code.isNotBlank()) {
                Text(
                    text = code,
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                    color = accent,
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(bottom = 1.dp)
                )
            }
        }
    }

    @Composable
    private fun FlightMetaChip(text: String) {
        Text(
            text = text,
            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier
                .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                .background(genUiCardContainerColor(GenUiCardTone.Neutral))
                .border(
                    width = GenUiTokens.BorderMd,
                    color = genUiCardBorderColor(),
                    shape = RoundedCornerShape(GenUiTokens.RadiusPill)
                )
                .padding(horizontal = 8.dp, vertical = 3.dp)
        )
    }

    @Composable
    private fun FlightTimeCell(
        title: String?,
        subtitle: String?,
        align: TextAlign,
        modifier: Modifier = Modifier,
        emphasis: Boolean = false
    ) {
        Column(
            modifier = modifier,
            verticalArrangement = Arrangement.spacedBy(2.dp),
            horizontalAlignment = when (align) {
                TextAlign.End -> Alignment.End
                TextAlign.Center -> Alignment.CenterHorizontally
                else -> Alignment.Start
            }
        ) {
            if (!title.isNullOrBlank()) {
                Text(
                    text = title,
                    style = if (emphasis) {
                        MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                    } else {
                        MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold)
                    },
                    color = MaterialTheme.colorScheme.onSurface,
                    textAlign = align
                )
            }
            if (!subtitle.isNullOrBlank()) {
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = align
                )
            }
        }
    }

    private fun splitFareDisplay(fare: String?): Pair<String?, String?> {
        val raw = sanitizeDisplayText(fare.orEmpty()).trim()
        if (raw.isBlank()) {
            return null to null
        }
        val compact = raw.replace(Regex("""\s+"""), " ")
        val lower = compact.lowercase(Locale.US)
        val amount = Regex("""([\u20B9$\u20AC\u00A3]\s?\d[\d,]*(?:\.\d+)?)""").find(compact)?.groupValues?.getOrNull(1)
        val normalizedAmount = amount?.replace(Regex("""\s+"""), "")
        val suffix = when {
            lower.contains("/adult") || lower.contains("per adult") -> "/adult"
            lower.contains("/person") || lower.contains("per person") -> "/person"
            else -> null
        }
        if (!normalizedAmount.isNullOrBlank()) {
            return normalizedAmount to suffix
        }
        return when {
            lower.contains("/adult") -> compact.replace(Regex("""(?i)\s*/\s*adult"""), "").trim() to "/adult"
            lower.contains("per adult") -> compact.replace(Regex("""(?i)\s*per\s*adult"""), "").trim() to "/adult"
            lower.contains("/person") -> compact.replace(Regex("""(?i)\s*/\s*person"""), "").trim() to "/person"
            lower.contains("per person") -> compact.replace(Regex("""(?i)\s*per\s*person"""), "").trim() to "/person"
            else -> compact to null
        }
    }

    private fun normalizeStopLabel(
        rawStops: String?,
        rawStatus: String?,
        depart: String?,
        arrive: String?
    ): String? {
        canonicalizeStopLabel(rawStops)?.let { return it }
        canonicalizeStopLabel(rawStatus)?.let { return it }
        if (!depart.isNullOrBlank() && !arrive.isNullOrBlank()) {
            return "Non-stop"
        }
        return null
    }

    private fun canonicalizeStopLabel(value: String?): String? {
        val text = sanitizeDisplayText(value.orEmpty()).trim()
        if (text.isBlank()) {
            return null
        }
        val normalized = normalizeMatchText(text)
        return when {
            normalized.contains("non stop") || normalized.contains("nonstop") || normalized.contains("direct") ||
                normalized == "0 stop" || normalized == "0 stops" -> "Non-stop"

            Regex("""\b1\b.*\bstop""").containsMatchIn(normalized) || normalized.contains("one stop") -> "1 stop"
            Regex("""\b2\b.*\bstop""").containsMatchIn(normalized) || normalized.contains("two stop") -> "2 stops"
            normalized.contains("stop") || normalized.contains("layover") || normalized.contains("connection") -> text
            else -> null
        }
    }

    private fun normalizeFlightStatus(rawStatus: String?, stopLabel: String?): String? {
        val status = sanitizeDisplayText(rawStatus.orEmpty()).trim()
        if (status.isBlank()) {
            return null
        }
        val normalized = normalizeMatchText(status)
        if (normalized in setOf("direct", "non stop", "nonstop")) {
            return null
        }
        if (stopLabel != null && canonicalizeStopLabel(status) != null) {
            return null
        }
        return status
    }

    private fun looksLikePunctualityStatus(text: String): Boolean {
        val normalized = normalizeMatchText(text)
        return normalized.contains("on time") ||
            normalized.contains("punctual") ||
            Regex("""\b\d{1,3}\s*%""").containsMatchIn(text)
    }

    private fun airlineBadgeCode(airline: String): String {
        val normalized = normalizeMatchText(airline)
        val explicit = when {
            normalized.contains("indigo") -> "6E"
            normalized.contains("air india express") -> "IX"
            normalized == "air india" || normalized.startsWith("air india ") -> "AI"
            normalized.contains("akasa") -> "QP"
            normalized.contains("vistara") -> "UK"
            normalized.contains("spicejet") -> "SG"
            normalized.contains("emirates") -> "EK"
            normalized.contains("british airways") -> "BA"
            normalized.contains("qatar") -> "QR"
            else -> ""
        }
        if (explicit.isNotBlank()) {
            return explicit
        }

        val initials = airline
            .split(Regex("""[^A-Za-z0-9]+"""))
            .filter { it.isNotBlank() }
            .take(2)
            .map { token ->
                token.firstOrNull { it.isLetterOrDigit() }?.uppercaseChar()?.toString().orEmpty()
            }
            .joinToString("")
        return initials.take(2)
    }

    @Composable
    private fun airlineAccentColor(airline: String): Color {
        val normalized = normalizeMatchText(airline)
        return when {
            normalized.contains("indigo") -> Color(0xFF3F51B5)
            normalized.contains("akasa") -> Color(0xFF6D1B7B)
            normalized.contains("air india express") -> Color(0xFFE53935)
            normalized == "air india" || normalized.startsWith("air india ") -> Color(0xFFD32F2F)
            normalized.contains("vistara") -> Color(0xFF6A1B9A)
            normalized.contains("spicejet") -> Color(0xFFD84315)
            normalized.contains("emirates") -> Color(0xFFB71C1C)
            else -> MaterialTheme.colorScheme.primary
        }
    }

    private fun buildWeatherRows(
        header: List<String>,
        body: List<List<String>>
    ): List<WeatherRow>? {
        if (header.isEmpty() || body.isEmpty()) {
            return null
        }
        val columns = detectWeatherColumns(header) ?: return null
        val rows = body.mapNotNull { row ->
            val period = normalizeWeatherCell(readCell(row, columns.period)).orEmpty()
            if (period.isBlank()) {
                return@mapNotNull null
            }

            val metrics = buildList {
                normalizeWeatherCell(readCell(row, columns.precip))?.let { add("Precip" to it) }
                normalizeWeatherCell(readCell(row, columns.wind))?.let { add("Wind" to it) }
                normalizeWeatherCell(readCell(row, columns.humidity))?.let { add("Humidity" to it) }
                normalizeWeatherCell(readCell(row, columns.uv))?.let { add("UV" to it) }
            }

            WeatherRow(
                period = period,
                date = normalizeWeatherCell(readCell(row, columns.date)),
                condition = normalizeWeatherCell(readCell(row, columns.condition)),
                high = normalizeWeatherCell(readCell(row, columns.high)),
                low = normalizeWeatherCell(readCell(row, columns.low)),
                temp = normalizeWeatherCell(readCell(row, columns.temp)),
                metrics = metrics
            )
        }
        return rows.takeIf { it.isNotEmpty() }
    }

    private fun buildFlightRows(
        header: List<String>,
        body: List<List<String>>
    ): List<FlightRow>? {
        if (header.isEmpty() || body.isEmpty()) {
            return null
        }
        val detectedColumns = detectFlightColumns(header) ?: return null
        val columns = resolveFlightColumns(detectedColumns, body)
        val originCode = columns.depart?.let { extractAirportCode(header.getOrNull(it).orEmpty()) }
        val destinationCode = columns.arrive?.let { extractAirportCode(header.getOrNull(it).orEmpty()) }
        val rows = body.mapNotNull { row ->
            val airline = readCell(row, columns.airline).orEmpty()
            if (airline.isBlank()) {
                return@mapNotNull null
            }
            FlightRow(
                airline = airline,
                depart = readCell(row, columns.depart),
                originCode = originCode,
                arrive = readCell(row, columns.arrive),
                destinationCode = destinationCode,
                duration = readCell(row, columns.duration),
                stops = readCell(row, columns.stops),
                fare = readCell(row, columns.fare),
                status = readCell(row, columns.status)
            )
        }
        return rows.takeIf { it.isNotEmpty() }
    }

    private fun detectWeatherColumns(header: List<String>): WeatherTableColumns? {
        val normalized = header.map { normalizeWeatherHeader(it) }
        val weatherSignal = normalized.count { token ->
            token.contains("weather") ||
                token.contains("forecast") ||
                token.contains("condition") ||
                token.contains("temp") ||
                token.contains("high") ||
                token.contains("low") ||
                token.contains("precip") ||
                token.contains("rain") ||
                token.contains("humidity") ||
                token.contains("wind") ||
                token.contains("uv")
        }
        if (weatherSignal < 2) {
            return null
        }

        val period = findHeaderIndex(normalized, listOf("day", "time", "hour", "period", "date")) ?: return null
        val date = findHeaderIndex(normalized, listOf("date"), exclude = setOf(period))
        val condition = findHeaderIndex(normalized, listOf("condition", "forecast", "weather", "summary"), exclude = setOf(period))
        val temp = findHeaderIndex(
            normalized,
            listOf("temperature", "temp", "high low", "high low c", "high low f"),
            exclude = setOf(period) + listOfNotNull(date, condition)
        )
        val high = findHeaderIndex(normalized, listOf("high", "max"), exclude = setOf(period) + listOfNotNull(date, condition, temp))
        val low = findHeaderIndex(normalized, listOf("low", "min"), exclude = setOf(period) + listOfNotNull(date, condition, temp, high))
        val precip = findHeaderIndex(normalized, listOf("precip", "rain", "chance", "pop"), exclude = setOf(period))
        val wind = findHeaderIndex(normalized, listOf("wind"), exclude = setOf(period))
        val humidity = findHeaderIndex(normalized, listOf("humidity"), exclude = setOf(period))
        val uv = findHeaderIndex(normalized, listOf("uv"), exclude = setOf(period))

        val contentSignals = listOf(condition, temp, high, low, precip, wind, humidity, uv).count { it != null }
        if (contentSignals < 2) {
            return null
        }

        return WeatherTableColumns(
            period = period,
            date = date,
            condition = condition,
            high = high,
            low = low,
            temp = temp,
            precip = precip,
            wind = wind,
            humidity = humidity,
            uv = uv
        )
    }

    private fun detectFlightColumns(header: List<String>): FlightTableColumns? {
        val normalized = header.map { normalizeWeatherHeader(it) }
        val flightSignal = normalized.count { token ->
            token.contains("airline") ||
                token.contains("carrier") ||
                token.contains("flight") ||
                token.contains("depart") ||
                token.contains("arrival") ||
                token.contains("arrive") ||
                token.contains("duration") ||
                token.contains("fare") ||
                token.contains("price") ||
                token.contains("cost") ||
                token.contains("stop") ||
                token.contains("layover") ||
                token.contains("status") ||
                token.contains("time")
        }
        if (flightSignal < 2) {
            return null
        }

        val airline = findHeaderIndex(normalized, listOf("airline", "carrier", "operator", "flight", "route"))
            ?: return null
        val depart = findHeaderIndex(normalized, listOf("depart", "departure", "takeoff", "from", "origin"), exclude = setOf(airline))
        val arrive = findHeaderIndex(normalized, listOf("arrive", "arrival", "landing", "to", "destination"), exclude = setOf(airline) + listOfNotNull(depart))
        val duration = findHeaderIndex(normalized, listOf("duration", "travel time", "elapsed"), exclude = setOf(airline) + listOfNotNull(depart, arrive))
        val stops = findHeaderIndex(normalized, listOf("stop", "stops", "layover", "connection", "type"), exclude = setOf(airline) + listOfNotNull(depart, arrive, duration))
        val fare = findHeaderIndex(normalized, listOf("fare", "price", "cost", "amount", "rate"), exclude = setOf(airline) + listOfNotNull(depart, arrive, duration, stops))
        val status = findHeaderIndex(normalized, listOf("status", "on time", "punctual", "delay"), exclude = setOf(airline) + listOfNotNull(depart, arrive, duration, stops, fare))

        val contentSignals = listOf(depart, arrive, duration, stops, fare, status).count { it != null }
        if (contentSignals < 2) {
            return null
        }

        return FlightTableColumns(
            airline = airline,
            depart = depart,
            arrive = arrive,
            duration = duration,
            stops = stops,
            fare = fare,
            status = status
        )
    }

    private fun resolveFlightColumns(
        detected: FlightTableColumns,
        body: List<List<String>>
    ): FlightTableColumns {
        if (body.isEmpty()) {
            return detected
        }
        val columnCount = body.maxOfOrNull { it.size } ?: return detected
        val sampleRows = body.take(5)
        val indices = (0 until columnCount).toList()

        fun score(index: Int?, predicate: (String) -> Boolean): Float {
            if (index == null || index !in indices) return 0f
            val values = sampleRows.mapNotNull { row -> row.getOrNull(index)?.trim()?.takeIf { it.isNotBlank() } }
            if (values.isEmpty()) return 0f
            return values.count(predicate).toFloat() / values.size.toFloat()
        }

        fun bestIndex(
            candidates: List<Int>,
            predicate: (String) -> Boolean,
            exclude: Set<Int> = emptySet()
        ): Int? {
            return candidates
                .filterNot { it in exclude }
                .maxByOrNull { idx -> score(idx, predicate) }
                ?.takeIf { score(it, predicate) >= 0.5f }
        }

        val timeScore: (String) -> Boolean = { looksLikeTimeValue(it) }
        val durationScore: (String) -> Boolean = { looksLikeDurationValue(it) }
        val fareScore: (String) -> Boolean = { looksLikeFareValue(it) }
        val stopScore: (String) -> Boolean = { canonicalizeStopLabel(it) != null }
        val airlineScore: (String) -> Boolean = { looksLikeAirlineValue(it) }

        var airline = detected.airline
        var depart = detected.depart
        var arrive = detected.arrive
        var duration = detected.duration
        var stops = detected.stops
        var fare = detected.fare

        // If airline values look like times or fares, remap to a text-like column.
        val airlineLooksWrong = score(airline, timeScore) >= 0.5f || score(airline, fareScore) >= 0.5f
        if (airlineLooksWrong || score(airline, airlineScore) < 0.4f) {
            bestIndex(indices, airlineScore, exclude = setOfNotNull(depart, arrive, duration, stops, fare))
                ?.let { airline = it }
        }

        if (depart == null || score(depart, timeScore) < 0.5f) {
            depart = bestIndex(indices, timeScore, exclude = setOf(airline))
        }

        if (arrive == null || score(arrive, timeScore) < 0.5f || arrive == depart) {
            arrive = bestIndex(indices, timeScore, exclude = setOfNotNull(airline, depart))
        }

        if (duration == null || score(duration, durationScore) < 0.4f) {
            duration = bestIndex(indices, durationScore, exclude = setOfNotNull(airline, depart, arrive))
        }

        if (fare == null || score(fare, fareScore) < 0.4f) {
            fare = bestIndex(indices, fareScore, exclude = setOfNotNull(airline, depart, arrive, duration, stops))
        }

        if (stops == null || score(stops, stopScore) < 0.4f) {
            stops = bestIndex(indices, stopScore, exclude = setOfNotNull(airline, depart, arrive, duration, fare))
        }

        return detected.copy(
            airline = airline,
            depart = depart,
            arrive = arrive,
            duration = duration,
            stops = stops,
            fare = fare
        )
    }

    private fun looksLikeTimeValue(value: String): Boolean {
        val normalized = value.trim().uppercase(Locale.US)
        return Regex("""^\d{1,2}:\d{2}(\s?(AM|PM))?(\+\d+)?$""").matches(normalized)
    }

    private fun looksLikeDurationValue(value: String): Boolean {
        val normalized = value.trim().lowercase(Locale.US)
        return Regex("""\d+\s*h""").containsMatchIn(normalized) || Regex("""\d+\s*m""").containsMatchIn(normalized)
    }

    private fun looksLikeFareValue(value: String): Boolean {
        val normalized = value.trim()
        return normalized.contains("â‚¹") ||
            normalized.contains("rs", ignoreCase = true) ||
            Regex("""\bfrom\s*\d""", RegexOption.IGNORE_CASE).containsMatchIn(normalized) ||
            Regex("""\d[\d,]+""").containsMatchIn(normalized)
    }

    private fun looksLikeAirlineValue(value: String): Boolean {
        val normalized = value.trim()
        if (normalized.length < 3) return false
        if (looksLikeTimeValue(normalized) || looksLikeDurationValue(normalized) || looksLikeFareValue(normalized)) {
            return false
        }
        if (Regex("""^[A-Z]{3}$""").matches(normalized.uppercase(Locale.US))) {
            return false
        }
        return normalized.any { it.isLetter() } && !normalized.equals("non-stop", ignoreCase = true)
    }

    private fun findHeaderIndex(
        normalizedHeader: List<String>,
        keywords: List<String>,
        exclude: Set<Int> = emptySet()
    ): Int? {
        return normalizedHeader.indices.firstOrNull { index ->
            index !in exclude && keywords.any { key -> normalizedHeader[index].contains(key) }
        }
    }

    private fun normalizeWeatherHeader(value: String): String {
        return value
            .lowercase(Locale.US)
            .replace(Regex("[^a-z0-9]+"), " ")
            .trim()
    }

    private fun readCell(row: List<String>, index: Int?): String? {
        if (index == null || index !in row.indices) {
            return null
        }
        return row[index].trim().takeIf { it.isNotEmpty() }
    }

    private fun extractAirportCode(headerText: String): String? {
        if (headerText.isBlank()) {
            return null
        }
        val match = Regex("""\(([A-Za-z]{3})\)""").find(headerText)
        return match?.groupValues?.getOrNull(1)?.uppercase(Locale.US)
    }

    private fun weatherTemperatureText(row: WeatherRow): String {
        val temp = formatTemperatureCellValue(row.temp)
        if (!temp.isNullOrBlank()) {
            return temp
        }
        val high = formatTemperatureCellValue(row.high)
        val low = formatTemperatureCellValue(row.low)
        if (!high.isNullOrBlank() && !low.isNullOrBlank()) {
            return "$high / $low"
        }
        return high ?: low ?: ""
    }

    private fun formatTemperatureCellValue(value: String?): String? {
        val raw = normalizeWeatherCell(value) ?: return null
        var formatted = raw
            .replace("º", "°")
            .replace("℃", "°C")
            .replace("℉", "°F")
            .replace(
                Regex("""(?i)(-?\d{1,2}(?:\.\d+)?)\s*(?:°\s*)?([CF])\b""")
            ) { match ->
                val number = match.groupValues[1]
                val unit = match.groupValues[2].uppercase(Locale.US)
                "$number°$unit"
            }
        if (formatted.contains('°')) {
            return formatted
        }

        val rangePattern = Regex(
            """^\s*-?\d{1,2}(?:\.\d+)?\s*(?:/|–|-|to)\s*-?\d{1,2}(?:\.\d+)?\s*$""",
            RegexOption.IGNORE_CASE
        )
        if (rangePattern.matches(formatted)) {
            formatted = formatted.replace(Regex("""-?\d{1,2}(?:\.\d+)?""")) { match ->
                "${match.value}°"
            }
            return formatted
        }

        val single = Regex("""^\s*(-?\d{1,2}(?:\.\d+)?)\s*$""").matchEntire(formatted)
        if (single != null) {
            return "${single.groupValues[1]}°"
        }
        return formatted
    }

    @Composable
    private fun WeatherConditionIcon(
        condition: String?,
        size: Dp
    ) {
        val (icon, tint) = weatherConditionIconSpec(condition)
        Icon(
            imageVector = icon,
            contentDescription = sanitizeDisplayText(condition.orEmpty()).ifBlank { "Weather condition" },
            tint = tint,
            modifier = Modifier.size(size)
        )
    }

    private fun orderWeatherRows(rows: List<WeatherRow>): List<WeatherRow> {
        val todayIndex = rows.indexOfFirst { isTodayWeatherRow(it) }
        if (todayIndex <= 0) {
            return rows
        }
        return buildList {
            add(rows[todayIndex])
            rows.forEachIndexed { index, row ->
                if (index != todayIndex) {
                    add(row)
                }
            }
        }
    }

    private fun isTodayWeatherRow(row: WeatherRow): Boolean {
        val period = normalizeWeatherText(row.period)
        val date = normalizeWeatherText(row.date.orEmpty())
        if (period.contains("today") || date.contains("today")) {
            return true
        }
        val today = LocalDate.now()
        val fullDay = today.dayOfWeek.getDisplayName(java.time.format.TextStyle.FULL, Locale.US).lowercase(Locale.US)
        val shortDay = today.dayOfWeek.getDisplayName(java.time.format.TextStyle.SHORT, Locale.US).lowercase(Locale.US)
        return period.contains(fullDay) || period.contains(shortDay) || date.contains(fullDay) || date.contains(shortDay)
    }

    private fun normalizeWeatherText(value: String): String {
        return value.lowercase(Locale.US).replace(Regex("[^a-z0-9]+"), " ").trim()
    }

    private fun normalizeWeatherCell(value: String?): String? {
        val normalized = value?.trim().orEmpty()
        if (isPlaceholderTableCellValue(normalized)) {
            return null
        }
        return normalized
    }

    private fun isCurrentWeatherHeading(title: String): Boolean {
        val normalized = normalizeWeatherText(title)
        return normalized.contains("current condition") ||
            normalized.contains("current weather") ||
            normalized == "currently" ||
            normalized.contains("now")
    }

    private fun collectCurrentWeatherFollowUpLines(
        blocks: List<TextBlock>,
        startIndex: Int
    ): Pair<List<String>, Int> {
        val lines = mutableListOf<String>()
        var consumed = 0
        var cursor = startIndex
        while (cursor < blocks.size && consumed < 6) {
            val candidate = blocks[cursor]
            val text = when (candidate) {
                is TextBlock.Title -> candidate.text
                is TextBlock.Heading -> candidate.text
                is TextBlock.Paragraph -> candidate.text
                is TextBlock.Bullets -> candidate.items.joinToString(" ")
                else -> null
            }?.trim().orEmpty()
            if (text.isBlank()) {
                break
            }

            val normalized = normalizeWeatherText(text)
            val startsNewSection =
                normalized.contains("forecast") ||
                    normalized.contains("hourly") ||
                    normalized.contains("sources") ||
                    normalized.contains("quick action") ||
                    normalized.contains("recommendation")
            if (startsNewSection) {
                break
            }

            if (!isLikelyCurrentWeatherDetailLine(text)) {
                break
            }
            lines += text
            consumed++
            cursor++
        }
        return lines to consumed
    }

    private fun isLikelyCurrentWeatherDetailLine(text: String): Boolean {
        val normalized = text.lowercase(Locale.US)
        return Regex("""(?<!\d)-?\d{1,2}(?:\.\d+)?\s*°""").containsMatchIn(text) ||
            normalized.contains("feels like") ||
            normalized.contains("humidity") ||
            normalized.contains("wind") ||
            normalized.contains("uv index") ||
            normalized.contains("chance of rain") ||
            normalized.contains("precip") ||
            normalized in setOf(
                "clear",
                "sunny",
                "partly cloudy",
                "cloudy",
                "overcast",
                "rain",
                "rainy",
                "thunderstorm",
                "snow",
                "fog",
                "mist"
            )
    }

    private fun inferWeatherConditionFromSection(
        title: String,
        sectionBlocks: List<TextBlock>
    ): String? {
        sectionBlocks.forEach { block ->
            if (block is TextBlock.MediaCards) {
                block.entries.forEach { entry ->
                    if (entry.iconLike) {
                        inferWeatherConditionFromIconUrl(entry.url)?.let { return it }
                    }
                }
            }
        }

        val normalizedTitle = normalizeWeatherText(title)
        val weatherContext =
            normalizedTitle.contains("weather") ||
                normalizedTitle.contains("condition") ||
                normalizedTitle.contains("forecast")
        if (!weatherContext) {
            return null
        }

        val textPool = buildString {
            append(title)
            sectionBlocks.forEach { block ->
                when (block) {
                    is TextBlock.Paragraph -> {
                        append(' ')
                        append(block.text)
                    }

                    is TextBlock.Bullets -> {
                        block.items.forEach { item ->
                            append(' ')
                            append(item)
                        }
                    }

                    else -> Unit
                }
            }
        }.lowercase(Locale.US)
        return inferWeatherConditionFromTextPool(textPool)
    }

    private fun extractCurrentWeatherIconUrl(sectionBlocks: List<TextBlock>): String? {
        sectionBlocks.forEach { block ->
            if (block is TextBlock.MediaCards) {
                block.entries.firstOrNull { entry ->
                    val normalized = entry.url.trim().lowercase(Locale.US)
                    val hasUsableUrl = normalized.isNotBlank() && !isLikelyPlaceholderMediaToken(normalized)
                    hasUsableUrl && (entry.iconLike || looksLikeCompactIconUrl(normalized))
                }?.let { return it.url }
            }
        }
        return null
    }

    private fun inferWeatherConditionFromTextPool(textPool: String): String? {
        return when {
            textPool.contains("thunder") || textPool.contains("storm") || textPool.contains("lightning") -> "Thunderstorm"
            textPool.contains("snow") || textPool.contains("sleet") || textPool.contains("blizzard") -> "Snow"
            textPool.contains("rain") || textPool.contains("shower") || textPool.contains("drizzle") -> "Rain"
            textPool.contains("fog") || textPool.contains("mist") || textPool.contains("haze") -> "Fog"
            textPool.contains("cloud") || textPool.contains("overcast") || textPool.contains("partly cloudy") -> "Cloudy"
            textPool.contains("clear") -> "Clear"
            textPool.contains("sun") || textPool.contains("fair") -> "Sunny"
            textPool.contains("wind") || textPool.contains("breeze") -> "Windy"
            else -> null
        }
    }

    private fun buildCurrentWeatherDetails(
        title: String,
        sectionBlocks: List<TextBlock>,
        fallbackCondition: String?,
        additionalLines: List<String> = emptyList()
    ): WeatherCurrentDetails? {
        if (!isCurrentWeatherHeading(title)) {
            return null
        }

        val rawLines = mutableListOf<String>()
        sectionBlocks.forEach { block ->
            when (block) {
                is TextBlock.Paragraph -> rawLines += block.text
                is TextBlock.Bullets -> rawLines += block.items
                else -> Unit
            }
        }
        rawLines += additionalLines
        if (rawLines.isEmpty()) {
            return null
        }

        val merged = rawLines.joinToString(" ")
            .replace(Regex("""\s+"""), " ")
            .trim()
        if (merged.isBlank()) {
            return null
        }

        val iconUrl = extractCurrentWeatherIconUrl(sectionBlocks)
        val iconCondition = iconUrl?.let(::inferWeatherConditionFromIconUrl)
        val condition = iconCondition
            ?: inferWeatherConditionFromTextPool(merged.lowercase(Locale.US))
            ?: fallbackCondition
        val temperature = extractTemperatureValue(merged)
        val feelsLike = extractFeelsLikeValue(merged)
        val humidity = extractHumidityValue(merged)
        val wind = extractWindValue(merged)
        val rainChance = extractRainChanceValue(merged)
        val uvIndex = extractUvIndexValue(merged)
        val summary = merged
            .split(Regex("""(?<=[.!?])\s+"""))
            .filter { it.isNotBlank() }
            .take(2)
            .joinToString(" ")
            .trim()
            .takeIf { it.isNotBlank() }

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
        ).takeIf {
            !it.iconUrl.isNullOrBlank() ||
            !it.condition.isNullOrBlank() ||
                !it.temperature.isNullOrBlank() ||
                !it.feelsLike.isNullOrBlank() ||
                !it.humidity.isNullOrBlank() ||
                !it.wind.isNullOrBlank() ||
                !it.rainChance.isNullOrBlank() ||
                !it.uvIndex.isNullOrBlank()
        }
    }

    private fun extractTemperatureValue(text: String): String? {
        Regex("""(?<!\d)(-?\d{1,2}(?:\.\d+)?)\s*(?:°\s*([CF])|([CF]))\b""", RegexOption.IGNORE_CASE)
            .find(text)
            ?.let { match ->
                val value = match.groupValues[1]
                val unit = match.groupValues.getOrNull(2).orEmpty().ifBlank {
                    match.groupValues.getOrNull(3).orEmpty()
                }
                return formatTemperatureReading(value, unit)
            }

        Regex("""(?<!\d)(-?\d{1,2}(?:\.\d+)?)\s*°\b""")
            .find(text)
            ?.let { match ->
                return formatTemperatureReading(match.groupValues[1], null)
            }

        Regex("""(?i)\b(?:temperature|temp|currently|current)\b[^-\d]{0,16}(-?\d{1,2}(?:\.\d+)?)\s*(?:°\s*([CF])|([CF]))?""")
            .find(text)
            ?.let { match ->
                val value = match.groupValues[1]
                val unit = match.groupValues.getOrNull(2).orEmpty().ifBlank {
                    match.groupValues.getOrNull(3).orEmpty()
                }
                return formatTemperatureReading(value, unit)
            }

        return null
    }

    private fun extractFeelsLikeValue(text: String): String? {
        Regex("""(?i)\bfeels?\s+like\b[^-\d]{0,16}(-?\d{1,2}(?:\.\d+)?)\s*(?:°\s*([CF])|([CF]))?""")
            .find(text)
            ?.let { match ->
                val value = match.groupValues[1]
                val unit = match.groupValues.getOrNull(2).orEmpty().ifBlank {
                    match.groupValues.getOrNull(3).orEmpty()
                }
                return formatTemperatureReading(value, unit)
            }
        return null
    }

    private fun formatTemperatureReading(value: String, unit: String?): String {
        val cleanValue = value.trim()
        val cleanUnit = unit.orEmpty().trim().uppercase(Locale.US)
        return if (cleanUnit.isBlank()) "$cleanValue°" else "$cleanValue°$cleanUnit"
    }

    private fun extractHumidityValue(text: String): String? =
        Regex("""(?i)\bhumidity\b[^0-9]{0,12}(\d{1,3})\s*%?""")
            .find(text)
            ?.groupValues
            ?.getOrNull(1)
            ?.let { "${it}%" }

    private fun extractWindValue(text: String): String? {
        val explicit = Regex(
            """(?i)\bwinds?\b[^0-9]{0,24}(\d{1,3}(?:\.\d+)?)\s*(km/?h|kph|mph|m/s)"""
        ).find(text)
        if (explicit != null) {
            return "${explicit.groupValues[1]} ${explicit.groupValues[2]}"
        }
        val normalized = text.lowercase(Locale.US)
        return when {
            normalized.contains("wind is calm") || normalized.contains("winds are calm") || normalized.contains("calm wind") -> "Calm"
            normalized.contains("breezy") -> "Breezy"
            else -> null
        }
    }

    private fun extractRainChanceValue(text: String): String? {
        Regex("""(?i)(\d{1,3})\s*%\s*(?:chance of rain|chance of precipitation|rain chance|precip(?:itation)? chance)""")
            .find(text)
            ?.groupValues
            ?.getOrNull(1)
            ?.let { return "${it}%" }
        Regex("""(?i)\b(?:chance of rain|chance of precipitation|rain chance|precip(?:itation)? chance)\b[^0-9]{0,16}(\d{1,3})\s*%?""")
            .find(text)
            ?.groupValues
            ?.getOrNull(1)
            ?.let { return "${it}%" }
        return null
    }

    private fun extractUvIndexValue(text: String): String? =
        Regex("""(?i)\buv(?:\s+index)?\b[^0-9]{0,10}(\d{1,2}(?:\.\d+)?)""")
            .find(text)
            ?.groupValues
            ?.getOrNull(1)

    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    private fun RenderCurrentWeatherDetails(
        titleText: String,
        titleStyle: TextStyle,
        details: WeatherCurrentDetails,
        sourceDir: File?
    ) {
        val iconUrl = details.iconUrl?.trim().orEmpty()
        val iconBasedCondition = iconUrl
            .takeIf { it.isNotBlank() }
            ?.let(::inferWeatherConditionFromIconUrl)
        val heroCondition = iconBasedCondition ?: details.condition
        val conditionText = (details.condition ?: heroCondition)?.let(::sanitizeDisplayText).orEmpty()
        val temperatureText = details.temperature?.let(::sanitizeDisplayText).orEmpty()
        val feelsLikeText = details.feelsLike?.let(::sanitizeDisplayText).orEmpty()

        MarkdownText(
            text = titleText,
            style = titleStyle,
            color = MaterialTheme.colorScheme.onSurface
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            WeatherConditionIcon(
                condition = heroCondition,
                size = 124.dp
            )

            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                if (temperatureText.isNotBlank()) {
                    Text(
                        text = temperatureText,
                        style = MaterialTheme.typography.headlineLarge.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
                if (conditionText.isNotBlank()) {
                    Text(
                        text = conditionText,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
                if (feelsLikeText.isNotBlank()) {
                    Text(
                        text = "Feels like $feelsLikeText",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }

        val chips = buildList {
            details.humidity?.let { add("Humidity" to it) }
            details.wind?.let { add("Wind" to it) }
            details.rainChance?.let { add("Rain Chance" to it) }
            details.uvIndex?.let { add("UV Index" to it) }
        }
        if (chips.isNotEmpty()) {
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                chips.forEach { (label, value) ->
                    val chipValue = sanitizeDisplayText(value)
                    if (chipValue.isBlank()) return@forEach
                    Text(
                        text = "$label $chipValue",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier
                            .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                            .background(genUiCardContainerColor(GenUiCardTone.Neutral))
                            .border(
                                GenUiTokens.BorderMd,
                                genUiCardBorderColor(),
                                RoundedCornerShape(GenUiTokens.RadiusPill)
                            )
                            .padding(horizontal = 8.dp, vertical = 5.dp)
                    )
                }
            }
        }

        details.summary
            ?.let(::sanitizeDisplayText)
            ?.takeIf { it.isNotBlank() }
            ?.let { summary ->
                MarkdownText(
                    text = summary,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
    }

    private fun extractWeatherApiIconCode(normalizedUrl: String): Int? {
        return Regex("""/(\d{3,4})\.(png|jpg|jpeg|webp|svg)(?:[?#].*)?$""")
            .find(normalizedUrl)
            ?.groupValues
            ?.getOrNull(1)
            ?.toIntOrNull()
    }

    private fun mapWeatherApiIconCode(code: Int, isNight: Boolean): String? {
        return when (code) {
            113 -> if (isNight) "Clear" else "Sunny"
            116 -> "Partly Cloudy"
            119, 122 -> "Cloudy"
            143, 248, 260 -> "Fog"
            176, 263, 266, 281, 284, 293, 296, 299, 302, 305, 308, 311, 314, 353, 356, 359 -> "Rain"
            179, 182, 185, 317, 320, 323, 326, 329, 332, 335, 338, 362, 365, 368, 371, 374, 377 -> "Snow"
            200, 386, 389, 392, 395 -> "Thunderstorm"
            227, 230 -> "Blizzard"
            350 -> "Ice"
            else -> null
        }
    }

    private fun extractOpenWeatherIconCode(normalizedUrl: String): String? {
        return Regex("""/([0-9]{2}[dn])(?:@\dx)?\.(png|jpg|jpeg|webp|svg)(?:[?#].*)?$""")
            .find(normalizedUrl)
            ?.groupValues
            ?.getOrNull(1)
            ?.lowercase(Locale.US)
    }

    private fun mapOpenWeatherIconCode(code: String): String? {
        return when (code) {
            "01d" -> "Sunny"
            "01n" -> "Clear"
            "02d", "02n" -> "Partly Cloudy"
            "03d", "03n", "04d", "04n" -> "Cloudy"
            "09d", "09n", "10d", "10n" -> "Rain"
            "11d", "11n" -> "Thunderstorm"
            "13d", "13n" -> "Snow"
            "50d", "50n" -> "Fog"
            else -> null
        }
    }

    @Composable
    private fun weatherConditionIconSpec(condition: String?): Pair<ImageVector, Color> {
        val normalized = normalizeWeatherText(condition.orEmpty())
        return when {
            normalized.contains("thunder") || normalized.contains("storm") || normalized.contains("lightning") ->
                Icons.Filled.Thunderstorm to Color(0xFFE65B17)
            normalized.contains("snow") || normalized.contains("sleet") || normalized.contains("blizzard") ->
                Icons.Filled.AcUnit to Color(0xFF82B1FF)
            normalized.contains("rain") || normalized.contains("shower") || normalized.contains("drizzle") ->
                Icons.Filled.Grain to Color(0xFF2E65D4)
            normalized.contains("fog") || normalized.contains("mist") || normalized.contains("haze") ||
                normalized.contains("wind") || normalized.contains("breeze") ->
                Icons.Filled.Air to Color(0xFF25B871)
            normalized.contains("cloud") || normalized.contains("overcast") ->
                Icons.Filled.Cloud to Color(0xFF7A7A85)
            normalized.contains("clear") ->
                if (isNightWeatherContext(condition)) {
                    Icons.Filled.DarkMode to Color(0xFF7DA2FF)
                } else {
                    Icons.Filled.WbSunny to Color(0xFFFFB300)
                }
            normalized.contains("sun") ->
                Icons.Filled.WbSunny to Color(0xFFFFB300)
            else -> Icons.Filled.Cloud to MaterialTheme.colorScheme.primary
        }
    }

    private fun isNightWeatherContext(condition: String?): Boolean {
        val normalized = normalizeWeatherText(condition.orEmpty())
        if (normalized.contains("night") || normalized.contains("tonight") || normalized.contains("overnight") || normalized.contains("evening")) {
            return true
        }
        if (normalized.contains("day") || normalized.contains("today") || normalized.contains("afternoon") || normalized.contains("morning")) {
            return false
        }
        val hour = LocalTime.now().hour
        return hour < 6 || hour >= 18
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
            isHeader -> MaterialTheme.colorScheme.onSurface.copy(alpha = if (dark) 0.08f else 0.045f)
            rowIndex % 2 == 0 -> Color.Transparent
            else -> MaterialTheme.colorScheme.onSurface.copy(alpha = if (dark) 0.04f else 0.020f)
        }
    }

    private fun tableBaseCellWidth(columnCount: Int): Dp {
        return when {
            columnCount <= 2 -> 220.dp
            columnCount == 3 -> 196.dp
            columnCount == 4 -> 172.dp
            else -> 156.dp
        }
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
        val extracted = maybeExtractTableRun(children, 0, index) ?: return null
        val (tableSpec, consumed) = extracted
        if (consumed != children.size) {
            return null
        }
        return tableSpec
    }

    private fun maybeExtractTableRun(
        children: List<String>,
        startIndex: Int,
        index: Map<String, JsonObject>
    ): Pair<TableSpec, Int>? {
        if (startIndex !in children.indices) {
            return null
        }

        val rowComponents = mutableListOf<JsonObject>()
        var cursor = startIndex
        while (cursor < children.size) {
            val child = index[children[cursor]] ?: return null
            if (child.getString("component") != "Row") {
                break
            }
            rowComponents += child
            cursor++
        }
        if (rowComponents.size < 2) {
            return null
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
        val weatherHeaderLike = detectWeatherColumns(textRows.first().map { it.text }) != null
        val flightHeaderLike = detectFlightColumns(textRows.first().map { it.text }) != null
        if (!hasWeight && !headerLike && !weatherHeaderLike && !flightHeaderLike) {
            return null
        }

        val header = if (headerLike || weatherHeaderLike || flightHeaderLike) textRows.first() else null
        val body = if (header != null) textRows.drop(1) else textRows
        val filteredBody = body.filterNot { isPlaceholderTableCellRow(it) }
        if (filteredBody.isEmpty()) {
            return null
        }
        return TableSpec(header = header, rows = filteredBody) to (cursor - startIndex)
    }

    private fun maybeExtractCardRowTable(children: List<String>, index: Map<String, JsonObject>): TableSpec? {
        if (children.size < 2) {
            return null
        }

        val headerRowId = children.first()
        val headerCells = extractTableCellsFromRow(headerRowId, index) ?: return null
        if (headerCells.size < 2) {
            return null
        }

        val body = mutableListOf<List<TableCell>>()
        for (childId in children.drop(1)) {
            val child = index[childId] ?: return null
            val rowCells = when (child.getString("component")) {
                "Row" -> extractTableCellsFromRow(childId, index)
                "Card" -> {
                    val directChild = child.getString("child")
                    when {
                        !directChild.isNullOrBlank() -> extractTableCellsFromRow(directChild, index)
                        else -> {
                            val nested = readChildren(child)
                            if (nested.size == 1) extractTableCellsFromRow(nested.first(), index) else null
                        }
                    }
                }

                else -> null
            } ?: return null

            if (rowCells.size != headerCells.size) {
                return null
            }
            body += rowCells
        }

        val filteredBody = body.filterNot { isPlaceholderTableCellRow(it) }
        if (filteredBody.isEmpty()) {
            return null
        }

        val headerText = headerCells.map { it.text }
        val weatherHeaderLike = detectWeatherColumns(headerText) != null
        val flightHeaderLike = detectFlightColumns(headerText) != null
        if (!weatherHeaderLike && !flightHeaderLike) {
            return null
        }

        return TableSpec(
            header = headerCells,
            rows = filteredBody
        )
    }

    private fun extractTableCellsFromRow(
        rowId: String,
        index: Map<String, JsonObject>
    ): List<TableCell>? {
        val row = index[rowId] ?: return null
        if (row.getString("component") != "Row") {
            return null
        }

        val cellIds = readChildren(row)
        if (cellIds.isEmpty()) {
            return null
        }

        val cells = mutableListOf<TableCell>()
        cellIds.forEach { cellId ->
            val cell = index[cellId] ?: return null
            if (cell.getString("component") != "Text") {
                return null
            }

            cells += TableCell(
                text = readDynamicString(cell.get("text")),
                variant = (cell.getString("variant") ?: "body").lowercase(Locale.US),
                weight = (cell.getAsNumberOrNull("weight") ?: 1.0).toFloat().coerceAtLeast(0.6f)
            )
        }
        return cells
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
        val hasBullets = lines.count { isBulletListLine(it) } >= 2
        val hasMedia = lines.any { isInlineMediaLine(it) || isMediaMarkerHeading(it) }
        val hasMultiLineLayout = lines.size >= 3
        return hasButtons || hasTable || hasBullets || hasMedia || hasMultiLineLayout || normalized.contains("\n\n")
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

            if (isBulletListLine(line)) {
                val items = mutableListOf<String>()
                var cursor = index
                while (cursor < lines.size) {
                    val l = lines[cursor].trim()
                    if (!isBulletListLine(l)) break
                    val item = extractBulletListItem(l) ?: break
                    if (!isPlaceholderListEntry(item)) {
                        items += item
                    }
                    cursor++
                }
                if (items.isNotEmpty()) {
                    blocks += TextBlock.Bullets(items)
                    renderedAny = true
                }
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

            val inlineLinkButtons =
                if (looksLikeStandaloneLinkLine(line)) {
                    parseSourceLinksFromLine(line)
                } else {
                    emptyList()
                }
            if (inlineLinkButtons.isNotEmpty()) {
                blocks += if (inSourcesSection) {
                    TextBlock.Sources(inlineLinkButtons)
                } else {
                    TextBlock.Actions(inlineLinkButtons)
                }
                renderedAny = true
                index++
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
        val normalized = stripLeadingBulletMarker(line)
        if (normalized.isBlank()) {
            return emptyList()
        }

        val markdownLinks = MARKDOWN_SOURCE_LINK_REGEX
            .findAll(normalized)
            .mapNotNull { match ->
                val rawLabel = sanitizeDisplayText(match.groupValues[1]).trim()
                val normalizedUrl = toExternalUrl(sanitizeUrlToken(match.groupValues[2])) ?: return@mapNotNull null
                val label = if (isUsefulSourceLabel(rawLabel)) rawLabel else buildSourceLabelFromUrl(normalizedUrl, 0)
                ParsedButton(label = label, url = normalizedUrl)
            }
            .toList()
        if (markdownLinks.isNotEmpty()) {
            return markdownLinks.distinctBy { it.url.lowercase(Locale.US) }
        }

        val urlMatches = URL_REGEX.findAll(normalized).toList()
        if (urlMatches.isEmpty()) {
            return emptyList()
        }

        val links = mutableListOf<ParsedButton>()
        var cursor = 0
        urlMatches.forEachIndexed { index, match ->
            val rawUrl = sanitizeUrlToken(match.value)
            val normalizedUrl = toExternalUrl(rawUrl) ?: return@forEachIndexed
            val nextStart = urlMatches.getOrNull(index + 1)?.range?.first ?: normalized.length
            val labelChunk = normalized.substring(cursor, match.range.first).trim()
            val trailingChunk = normalized.substring((match.range.last + 1).coerceAtMost(normalized.length), nextStart).trim()

            var label = labelChunk.removeSuffix(":").trim().trim('|')
            if (index == 0) {
                label = label.replace(Regex("""^(sources?|references?)\s*:?\s*""", RegexOption.IGNORE_CASE), "")
            }
            label = label
                .replace(Regex("""^\s*\d+\s*[\).:\-–—]?\s*"""), "")
                .replace(Regex("""^\s*[-–—|:]\s*"""), "")
                .trim(' ', '-', '–', '—', '|', ':')

            if (label.isBlank() && trailingChunk.isNotBlank() && !containsUrlLikeToken(trailingChunk)) {
                label = trailingChunk
                    .replace(Regex("""^\s*[-–—|:\u2022]\s*"""), "")
                    .trim(' ', '-', '–', '—', '|', ':')
            }

            if (!isUsefulSourceLabel(label)) {
                label = buildSourceLabelFromUrl(normalizedUrl, index)
            }

            links += ParsedButton(
                label = label,
                url = normalizedUrl
            )
            cursor = nextStart
        }
        return links
    }

    private fun isUsefulSourceLabel(label: String): Boolean {
        val normalized = label.trim()
        if (normalized.isBlank()) {
            return false
        }
        if (containsUrlLikeToken(normalized)) {
            return false
        }
        if (Regex("""(?i)^(?:www\.)?(?:[a-z0-9-]+\.)+[a-z]{2,24}$""").matches(normalized)) {
            return false
        }
        val lower = normalized.lowercase(Locale.US)
        if (lower in setOf("source", "sources", "reference", "references", "link", "links")) {
            return false
        }
        return normalized.any { it.isLetter() }
    }

    private fun buildSourceLabelFromUrl(url: String, index: Int): String {
        val uri = runCatching { Uri.parse(url) }.getOrNull()
        val host = uri?.host?.removePrefix("www.")?.trim().orEmpty()
        if (host.isBlank()) {
            return "Source ${index + 1}"
        }
        val hostLabel = host
            .substringBefore('.')
            .replace('-', ' ')
            .replace('_', ' ')
            .split(Regex("""\s+"""))
            .filter { it.isNotBlank() }
            .joinToString(" ") { token ->
                token.lowercase(Locale.US).replaceFirstChar { ch ->
                    if (ch.isLowerCase()) ch.titlecase(Locale.US) else ch.toString()
                }
            }
            .ifBlank { host }
        val pathHint = uri?.pathSegments
            ?.firstOrNull { segment -> segment.length >= 3 && segment.any { it.isLetter() } }
            ?.replace('-', ' ')
            ?.replace('_', ' ')
            ?.split(Regex("""\s+"""))
            ?.filter { it.isNotBlank() }
            ?.take(3)
            ?.joinToString(" ") { token ->
                token.lowercase(Locale.US).replaceFirstChar { ch ->
                    if (ch.isLowerCase()) ch.titlecase(Locale.US) else ch.toString()
                }
            }
            ?.takeIf { it.isNotBlank() && !hostLabel.contains(it, ignoreCase = true) }

        return if (pathHint != null) "$hostLabel $pathHint" else hostLabel
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
            if (!isBulletListLine(line)) {
                break
            }

            val item = extractBulletListItem(line) ?: break
            val parts = item.split(":", limit = 2)
            if (parts.size == 2) {
                val label = parts[0].trim()
                val value = parts[1].trim()
                if (label.isNotEmpty() && looksLikeImagePath(value, labelHint = label)) {
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

            val inlineEntries = parseInlineMediaEntries(line)
            if (inlineEntries.isNotEmpty()) {
                entries += inlineEntries
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
            "media:",
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
        // Let parseInlineMediaEntries handle compound assignments like:
        // Media: Image=... Icon=...
        if (value.contains("Image=", ignoreCase = true) || value.contains("Icon=", ignoreCase = true)) {
            return null
        }
        if (label.isEmpty() || !looksLikeImagePath(value, labelHint = label)) {
            return null
        }

        val lowerLabel = label.lowercase(Locale.US)
        val lowerValue = value.lowercase(Locale.US)
        val iconLike = lowerLabel.contains("icon") ||
            looksLikeCompactIconUrl(lowerValue) ||
            (lowerValue.endsWith(".svg") && !lowerLabel.contains("logo") && !lowerValue.contains("logo")) ||
            (lowerValue.contains("/icon") && !lowerLabel.contains("logo") && !lowerValue.contains("logo"))

        return ParsedMediaEntry(
            label = label,
            url = value,
            iconLike = iconLike
        )
    }

    private fun parseInlineMediaEntries(line: String): List<ParsedMediaEntry> {
        if (!isInlineMediaLine(line)) {
            return emptyList()
        }

        val assignmentEntries = INLINE_MEDIA_ASSIGNMENT_REGEX
            .findAll(line)
            .mapNotNull { match ->
                val mediaType = match.groupValues[1].trim()
                val rawValue = sanitizeUrlToken(match.groupValues[2])
                if (!looksLikeImagePath(rawValue, labelHint = mediaType)) {
                    null
                } else {
                    val normalizedType = mediaType.lowercase(Locale.US)
                    val normalizedUrl = rawValue.lowercase(Locale.US)
                    val inlineIconLike = normalizedType.contains("icon") ||
                        looksLikeCompactIconUrl(normalizedUrl) ||
                        (normalizedUrl.endsWith(".svg") && !normalizedUrl.contains("logo")) ||
                        (normalizedUrl.contains("/icon") && !normalizedUrl.contains("logo"))
                    ParsedMediaEntry(
                        label = mediaType,
                        url = rawValue,
                        iconLike = inlineIconLike
                    )
                }
            }
            .toList()
        val markdownEntries = parseMarkdownImageEntries(line)
        return (assignmentEntries + markdownEntries)
            .distinctBy { "${it.label.lowercase(Locale.US)}|${it.url.lowercase(Locale.US)}" }
    }

    private fun parseMarkdownImageEntries(line: String): List<ParsedMediaEntry> {
        return MARKDOWN_IMAGE_REGEX
            .findAll(line)
            .mapNotNull { match ->
                val label = match.groupValues.getOrNull(1)?.trim().orEmpty().ifBlank { "Image" }
                val rawValue = sanitizeUrlToken(match.groupValues.getOrNull(2).orEmpty())
                if (!looksLikeImagePath(rawValue, labelHint = label)) {
                    null
                } else {
                    val normalizedLabel = label.lowercase(Locale.US)
                    val normalizedUrl = rawValue.lowercase(Locale.US)
                    val iconLike = normalizedLabel.contains("icon") ||
                        looksLikeCompactIconUrl(normalizedUrl) ||
                        (normalizedUrl.endsWith(".svg") && !normalizedUrl.contains("logo")) ||
                        (normalizedUrl.contains("/icon") && !normalizedUrl.contains("logo"))
                    ParsedMediaEntry(
                        label = label,
                        url = rawValue,
                        iconLike = iconLike
                    )
                }
            }
            .toList()
    }

    private fun isInlineMediaLine(line: String): Boolean {
        val trimmed = line.trim()
        if (MARKDOWN_IMAGE_REGEX.containsMatchIn(trimmed)) {
            return true
        }
        if (trimmed.startsWith("Media:", ignoreCase = true)) {
            return INLINE_MEDIA_ASSIGNMENT_REGEX.containsMatchIn(trimmed)
        }
        if (trimmed.startsWith("Image:", ignoreCase = true) || trimmed.startsWith("Icon:", ignoreCase = true)) {
            return true
        }
        return INLINE_MEDIA_ASSIGNMENT_REGEX.containsMatchIn(trimmed)
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
        val filteredRows = rows.filterIndexed { index, row ->
            index == 0 || !isPlaceholderTableStringRow(row)
        }
        if (filteredRows.size < 2) {
            return null
        }
        return filteredRows to cursor
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
        val trailing = match.groupValues[2].trim()
        val rawUrlToken = URL_REGEX.find(trailing)?.value ?: trailing
        val url = canonicalizeNetworkUrlToken(sanitizeUrlToken(rawUrlToken))
        if (label.isBlank() || url.isBlank()) {
            return null
        }
        return ParsedButton(label, url)
    }

    private fun sanitizeUrlToken(value: String): String =
        value.trim().trim('"', '\'', '<', '>', '`').trimEnd('.', ',', ';', ')', ']', '}')

    private fun isTableLikeLine(line: String): Boolean {
        if (containsUrlLikeToken(line)) {
            return false
        }
        return splitTableCells(line).size >= 3
    }

    private fun splitTableCells(line: String): List<String> =
        line.split('|').map { it.trim() }.filter { it.isNotEmpty() }

    private fun isPlaceholderTableStringRow(row: List<String>): Boolean {
        if (row.isEmpty()) return true
        return row.all(::isPlaceholderTableCellValue)
    }

    private fun isPlaceholderTableCellRow(row: List<TableCell>): Boolean =
        isPlaceholderTableStringRow(row.map { it.text })

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
        if (Regex("""^day\s+\d+\s*:""", RegexOption.IGNORE_CASE).containsMatchIn(line)) {
            return true
        }
        if (parseLeadingLabelValue(line) != null) {
            return false
        }
        if (isBulletListLine(line) || line.endsWith(".") || line.endsWith("?") || line.endsWith("!")) {
            return false
        }
        if (line.contains('|') || containsUrlLikeToken(line)) {
            return false
        }
        if (line.contains("[Button:", ignoreCase = true)) {
            return false
        }
        return line.any { it.isLetter() }
    }

    private fun containsUrlLikeToken(value: String): Boolean =
        URL_REGEX.containsMatchIn(value)

    private fun looksLikeStandaloneLinkLine(value: String): Boolean {
        val line = value.trim()
        if (line.isBlank() || !containsUrlLikeToken(line)) {
            return false
        }
        if (isInlineMediaLine(line) || isMediaMarkerHeading(line)) {
            return false
        }
        if (line.length > 260) {
            return false
        }
        if (line.contains('|') && splitTableCells(line).size >= 3) {
            return false
        }
        return true
    }

    private fun isBoilerplateContextHeading(text: String): Boolean {
        val normalized = normalizeMatchText(text.removeSuffix(":"))
        if (normalized.isBlank()) {
            return false
        }
        if (normalized in setOf("information context", "context", "background context", "background information")) {
            return true
        }
        if (normalized.contains("information context")) {
            return true
        }
        return normalized.contains("assumption") ||
            normalized.contains("travel planning") ||
            normalized.startsWith("a note") ||
            normalized.contains("background")
    }

    private fun isBoilerplateContextParagraph(text: String): Boolean {
        val normalized = normalizeMatchText(text)
        if (text.length < 120) {
            return false
        }
        return normalized.contains("subject to change") ||
            normalized.contains("following table") ||
            normalized.contains("for informational purposes") ||
            normalized.contains("comparison of") ||
            normalized.contains("the following tables")
    }

    private fun isStructuredBoundary(line: String): Boolean {
        if (isBulletListLine(line)) {
            return true
        }
        if (isInlineMediaLine(line) || isMediaMarkerHeading(line)) {
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

    private fun looksLikeImagePath(value: String, labelHint: String? = null): Boolean {
        val normalized = value.trim()
        if (normalized.isBlank() || isLikelyPlaceholderMediaToken(normalized)) {
            return false
        }
        val normalizedLower = normalized.lowercase(Locale.US)
        val pathWithoutQuery = normalizedLower.substringBefore('?').substringBefore('#')
        if (pathWithoutQuery.contains("<") || pathWithoutQuery.contains(">")) {
            return false
        }
        if (
            pathWithoutQuery.endsWith(".png") ||
            pathWithoutQuery.endsWith(".jpg") ||
            pathWithoutQuery.endsWith(".jpeg") ||
            pathWithoutQuery.endsWith(".svg") ||
            pathWithoutQuery.endsWith(".webp")
        ) {
            return true
        }

        val uri = runCatching { Uri.parse(normalized) }.getOrNull()
        val scheme = uri?.scheme?.lowercase(Locale.US)
        if (scheme == "http" || scheme == "https") {
            val host = uri.host?.lowercase(Locale.US).orEmpty()
            val path = uri.path.orEmpty().lowercase(Locale.US)
            if (
                host.contains("loremflickr.com") ||
                host.contains("picsum.photos") ||
                host.contains("placehold.co") ||
                host.contains("dummyimage.com") ||
                host.contains("cdn.jsdelivr.net") ||
                host.contains("raw.githubusercontent.com") ||
                host.contains("upload.wikimedia.org") ||
                host.contains("imgur.com") ||
                host.contains("gstatic.com") ||
                host.contains("twimg.com")
            ) {
                return true
            }
            if (
                path.contains("/icon") ||
                path.contains("/icons/") ||
                path.contains("/image") ||
                path.contains("/images/")
            ) {
                return true
            }
        }
        return false
    }

    private fun isLikelyPlaceholderMediaToken(value: String): Boolean {
        val normalized = value
            .trim()
            .trim('\'', '"')
            .lowercase(Locale.US)
        if (normalized.isBlank()) {
            return true
        }
        return normalized in setOf(
            "<image_url>",
            "<icon_url>",
            "<url>",
            "image_url",
            "icon_url",
            "url",
            "n/a",
            "na",
            "none",
            "null",
            "--"
        ) || normalized.contains("placeholder")
    }

    private fun looksLikeCompactIconUrl(value: String): Boolean {
        val normalized = value.lowercase(Locale.US)
        val iconPathLike = Regex("""(?:^|/)(?:icon|icons)(?:/|[-_.]|$)""")
            .containsMatchIn(normalized)
        return iconPathLike ||
            normalized.contains("weatherapi.com/weather/") ||
            normalized.contains("/weather/64x64/") ||
            normalized.contains("/weather/128x128/") ||
            normalized.contains("/weather/icons/") ||
            normalized.contains("openweathermap.org/img/wn/") ||
            normalized.contains("/img/wn/") ||
            Regex("""/\d{2}[dn](?:@\dx)?\.(png|webp|jpg|jpeg)(?:[?#].*)?$""").containsMatchIn(normalized)
    }

    private fun shouldShowMediaLabel(label: String): Boolean {
        val normalized = sanitizeDisplayText(label).lowercase(Locale.US)
        if (normalized.isBlank()) {
            return false
        }
        if (normalized in setOf("image", "icon", "photo", "logo", "media")) {
            return false
        }
        return !Regex("""^(image|icon|photo|logo|media)\s*[:#-]?\s*\d*$""", RegexOption.IGNORE_CASE)
            .matches(normalized)
    }

    private fun isVectorImagePath(value: String): Boolean =
        value.trim().lowercase(Locale.US).endsWith(".svg")

    private fun isRasterImagePath(value: String): Boolean {
        val normalized = value.trim().lowercase(Locale.US)
        return normalized.endsWith(".png") ||
            normalized.endsWith(".jpg") ||
            normalized.endsWith(".jpeg") ||
            normalized.endsWith(".webp")
    }

    private fun defaultImageScale(
        rawUrl: String,
        fitValue: String?,
        defaultCoverForRaster: Boolean
    ): ContentScale {
        val normalizedFit = fitValue?.normalizeLayoutToken()
        return when (normalizedFit) {
            "contain", "fit", "inside" -> ContentScale.Fit
            "cover", "crop", "fill", "fillbounds", "fillwidth", "fillheight" -> ContentScale.Crop
            else -> {
                if (defaultCoverForRaster && isRasterImagePath(rawUrl) && !isVectorImagePath(rawUrl)) {
                    ContentScale.Crop
                } else {
                    ContentScale.Fit
                }
            }
        }
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
                "table" -> "Table"
                "button" -> "Button"
                "tabs", "tab", "tabgroup" -> "Tabs"
                "divider" -> "Divider"
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
        obj.getString("path")?.let { return normalizeMojibakeText(it) }
        obj.get("literalNumber")?.let { return normalizeMojibakeText(it.toString().trim('"')) }
        obj.get("literalBoolean")?.let { return normalizeMojibakeText(it.toString().trim('"')) }
        return normalizeMojibakeText(obj.toString())
    }

    private fun resolveAssetUrl(raw: String, sourceDir: File?): String {
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

    private fun canonicalizeNetworkUrlToken(value: String): String {
        val trimmed = value.trim()
        if (trimmed.isBlank()) {
            return trimmed
        }
        return normalizeHttpUrlCandidate(trimmed) ?: trimmed
    }

    private fun normalizeHttpUrlCandidate(value: String): String? {
        val trimmed = value.trim().trim('"', '\'')
        if (trimmed.isBlank()) {
            return null
        }
        if (trimmed.startsWith("http://", ignoreCase = true) || trimmed.startsWith("https://", ignoreCase = true)) {
            return trimmed
        }
        if (trimmed.startsWith("//")) {
            return "https:$trimmed"
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
        if (!URL_REGEX.matches(pathCandidate)) {
            return null
        }
        val host = pathCandidate
            .removePrefix("www.")
            .substringBefore('/')
            .substringBefore('?')
            .substringBefore('#')
            .lowercase(Locale.US)
        if (!isLikelyPublicDomainHost(host)) {
            return null
        }
        return "https://$pathCandidate"
    }

    private fun isLikelyPublicDomainHost(host: String): Boolean {
        if (host.isBlank() || host.contains('_')) {
            return false
        }
        val labels = host.split('.').filter { it.isNotBlank() }
        if (labels.size < 2 || labels.any { !HOST_LABEL_REGEX.matches(it) }) {
            return false
        }
        val tld = labels.last().lowercase(Locale.US)
        if (!tld.all { it in 'a'..'z' } || tld.length !in 2..24) {
            return false
        }
        if (
            tld in setOf(
                "png", "jpg", "jpeg", "svg", "webp", "gif", "bmp", "ico",
                "json", "xml", "txt", "csv", "md", "pdf", "zip", "apk"
            )
        ) {
            return false
        }
        return true
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


