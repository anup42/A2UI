package com.samsung.genuicraft.renderer.flat.compose

import com.samsung.genuicraft.renderer.*

import android.content.Intent
import android.content.res.Configuration
import android.net.Uri
import android.util.Log
import androidx.compose.foundation.clickable
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.relocation.BringIntoViewRequester
import androidx.compose.foundation.relocation.bringIntoViewRequester
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.OpenInNew
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Directions
import androidx.compose.material.icons.filled.EventAvailable
import androidx.compose.material.icons.filled.FlightTakeoff
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.CalendarToday
import androidx.compose.material.icons.filled.Language
import androidx.compose.material.icons.filled.PlayCircle
import androidx.compose.material.icons.filled.Link
import androidx.compose.material.icons.filled.RateReview
import androidx.compose.material.icons.filled.Restaurant
import androidx.compose.material.icons.filled.Star
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.BaselineShift
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import coil.ImageLoader
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.samsung.genuicraft.GenUiTokens
import com.samsung.genuicraft.GenUiCardTone
import com.samsung.genuicraft.genUiCardContainerColor
import com.google.gson.GsonBuilder
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.samsung.genuicraft.renderer.native.NativeTextFormatter
import com.samsung.genuicraft.renderer.native.NativePayloadParser
import com.samsung.genuicraft.renderer.native.ParsedButton
import com.samsung.genuicraft.renderer.native.intents.flight.NativeFlightSemantics
import com.samsung.genuicraft.renderer.native.intents.flight.NativeFlightUiRenderer
import com.samsung.genuicraft.renderer.native.intents.weather.NativeWeatherSemantics
import com.samsung.genuicraft.renderer.native.intents.weather.NativeWeatherUiRenderer
import com.samsung.genuicraft.renderer.native.media.NativeMediaVisualUtils
import com.samsung.genuicraft.renderer.native.parser.NativeSourceParsing
import com.samsung.genuicraft.renderer.native.parser.NativeStructureParsing
import com.samsung.genuicraft.security.SafeContentPolicy
import kotlinx.coroutines.launch
import java.util.Locale
import kotlin.math.roundToInt
import java.util.UUID
import java.net.URI
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.DatePicker
import androidx.compose.material3.DatePickerDialog
import androidx.compose.material3.rememberDatePickerState
import androidx.compose.material3.TimePicker
import androidx.compose.material3.rememberTimePickerState
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.TextButton
import androidx.compose.material3.IconButton
import androidx.compose.material.icons.filled.Schedule
import java.time.Instant
import java.time.ZoneOffset
import com.samsung.genuicraft.renderer.flat.domain.*
import com.samsung.genuicraft.renderer.flat.parse.*
import com.samsung.genuicraft.renderer.flat.expr.*
import com.samsung.genuicraft.renderer.flat.runtime.*
import com.samsung.genuicraft.renderer.flat.model.*
import com.samsung.genuicraft.renderer.flat.compose.table.*

// moved from FlatSpecRenderer.kt (RenderStack)
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun RenderStack(
    elementId: String,
    props: Map<String, Any?>,
    children: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    repeatedChildScopes: List<RepeatScope>?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    onAction: (Any?, RepeatScope?) -> Int,
    activePath: Set<String>,
    modifier: Modifier = Modifier
) {
    val compactScreen = isFlatCompactScreenWidth(LocalConfiguration.current.screenWidthDp)
    val computedFunctions = LocalFlatSpecComputedFunctions.current
    extractFlatSourceSection(
        elementId = elementId,
        props = props,
        children = children,
        elements = elements,
        state = state,
        repeatScope = repeatScope,
        computedFunctions = computedFunctions,
        allowChildSourceCue = false
    )?.let { sourceSection ->
        RenderFlatSourceSection(
            section = sourceSection,
            onOpenUrl = onOpenUrl,
            modifier = modifier,
            wrapInCard = false,
            showTitle = sourceSection.title != null
        )
        return
    }
    val tableModel = if (repeatedChildScopes == null) {
        extractFlatTableModel(
            containerChildren = children,
            containerProps = props,
            elements = elements,
            state = state,
            compactScreen = compactScreen
        )
    } else {
        null
    }
    if (tableModel != null) {
        RenderTableLayout(
            tableModel = tableModel,
            elementId = elementId,
            props = props,
            elements = elements,
            state = state,
            repeatScope = repeatScope,
            onOpenUrl = onOpenUrl,
            onSetState = onSetState,
            onAction = onAction,
            activePath = activePath,
            modifier = modifier
        )
        return
    }

    val direction = stackDirection(props)
    val gap = stackGap(props)
    val align = (
        props["align"]?.toString()
            ?: props["crossAxisAlignment"]?.toString()
            ?: if (direction == "horizontal") props["verticalAlignment"]?.toString() else props["horizontalAlignment"]?.toString()
        )
        ?.trim()
        ?.lowercase()
        .orEmpty()
    val justify = (
        props["justify"]?.toString()
            ?: props["mainAxisAlignment"]?.toString()
            ?: if (direction == "horizontal") props["horizontalArrangement"]?.toString() else props["verticalArrangement"]?.toString()
        )
        ?.trim()
        ?.lowercase()
        .orEmpty()
    val wrap = props["wrap"]?.toString()?.trim()?.lowercase() == "wrap"
    val childElements = children.mapNotNull { childId -> elements[childId] }
    val allButtonChildren = childElements.isNotEmpty() &&
        childElements.all { child -> child.type.equals("button", ignoreCase = true) }
    val hasLongButtonLabel = childElements.any { child ->
        child.props["label"]?.toString()?.trim()?.length ?: 0 > 20
    }
    val autoWrapButtonRow = direction == "horizontal" &&
        compactScreen &&
        !wrap &&
        allButtonChildren &&
        children.size >= 2
    val hasImageChild = childElements.any { child ->
        child.type.equals("image", ignoreCase = true)
    }
    val hasLongTextChild = childElements.any { child ->
        if (!child.type.equals("text", ignoreCase = true)) return@any false
        val rawText = (
            child.props["text"]
                ?: child.props["title"]
                ?: child.props["label"]
                ?: child.props["content"]
                ?: child.props["value"]
            )
            ?.toString()
            .orEmpty()
        rawText.length >= 90
    }
    val forceVerticalMediaTextRow = direction == "horizontal" &&
        compactScreen &&
        !wrap &&
        children.size in 2..3 &&
        hasImageChild &&
        hasLongTextChild
    val compactIconTextRow = direction == "horizontal" &&
        compactScreen &&
        !wrap &&
        children.size == 2 &&
        childElements.getOrNull(0)?.type?.equals("icon", ignoreCase = true) == true &&
        childElements.getOrNull(1)?.type?.equals("text", ignoreCase = true) == true
    val forceVerticalCardRow = direction == "horizontal" &&
        compactScreen &&
        !wrap &&
        children.size >= 2 &&
        childElements.all { child -> child.type.equals("card", ignoreCase = true) }
    val compactTextButtonRow = direction == "horizontal" &&
        compactScreen &&
        !wrap &&
        children.size == 2 &&
        childElements.any { child -> child.type.equals("text", ignoreCase = true) } &&
        childElements.any { child -> child.type.equals("button", ignoreCase = true) }
    val forceVerticalButtonStack = autoWrapButtonRow && hasLongButtonLabel
    val stackModifier = applyStackModifier(modifier, props, direction)
    val childTextHorizontalPadding = if (hasHorizontalContainerPadding(props)) {
        0.dp
    } else {
        LocalFlatSpecTextHorizontalPadding.current
    }

    if (direction == "horizontal") {
        val horizontalArrangement: Arrangement.Horizontal = when (justify) {
            "center" -> if (gap > 0.dp) Arrangement.spacedBy(gap, Alignment.CenterHorizontally) else Arrangement.Center
            "end" -> if (gap > 0.dp) Arrangement.spacedBy(gap, Alignment.End) else Arrangement.End
            "between" -> Arrangement.SpaceBetween
            "around" -> Arrangement.SpaceAround
            else -> if (gap > 0.dp) Arrangement.spacedBy(gap) else Arrangement.Start
        }
        if (compactIconTextRow) {
            CompositionLocalProvider(LocalFlatSpecTextHorizontalPadding provides childTextHorizontalPadding) {
                Row(
                    modifier = stackModifier,
                    horizontalArrangement = Arrangement.spacedBy(gap),
                    verticalAlignment = Alignment.Top
                ) {
                    RenderElement(
                        elementId = children[0],
                        elements = elements,
                        state = state,
                        repeatScope = repeatScope,
                        onOpenUrl = onOpenUrl,
                        onSetState = onSetState,
                        onAction = onAction,
                        activePath = activePath
                    )
                    RenderElement(
                        elementId = children[1],
                        elements = elements,
                        state = state,
                        repeatScope = repeatScope,
                        onOpenUrl = onOpenUrl,
                        onSetState = onSetState,
                        onAction = onAction,
                        activePath = activePath,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
            return
        }
        if (forceVerticalButtonStack || forceVerticalCardRow) {
            CompositionLocalProvider(LocalFlatSpecTextHorizontalPadding provides childTextHorizontalPadding) {
                Column(
                    modifier = stackModifier,
                    verticalArrangement = Arrangement.spacedBy(gap)
                ) {
                    RenderChildren(
                        children = children,
                        elements = elements,
                        state = state,
                        repeatScope = repeatScope,
                        repeatedChildScopes = repeatedChildScopes,
                        onOpenUrl = onOpenUrl,
                        onSetState = onSetState,
                        onAction = onAction,
                        activePath = activePath
                    )
                }
            }
            return
        }
        if (compactTextButtonRow) {
            val textChildId = children.firstOrNull { childId ->
                elements[childId]?.type?.equals("text", ignoreCase = true) == true
            }
            val buttonChildId = children.firstOrNull { childId ->
                elements[childId]?.type?.equals("button", ignoreCase = true) == true
            }
            if (textChildId != null && buttonChildId != null) {
                CompositionLocalProvider(LocalFlatSpecTextHorizontalPadding provides childTextHorizontalPadding) {
                    Column(
                        modifier = stackModifier,
                        verticalArrangement = Arrangement.spacedBy(if (gap > 0.dp) gap else 6.dp)
                    ) {
                        RenderElement(
                            elementId = textChildId,
                            elements = elements,
                            state = state,
                            repeatScope = repeatScope,
                            onOpenUrl = onOpenUrl,
                            onSetState = onSetState,
                            onAction = onAction,
                            activePath = activePath
                        )
                        RenderElement(
                            elementId = buttonChildId,
                            elements = elements,
                            state = state,
                            repeatScope = repeatScope,
                            onOpenUrl = onOpenUrl,
                            onSetState = onSetState,
                            onAction = onAction,
                            activePath = activePath,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }
                return
            }
        }
        if (forceVerticalMediaTextRow) {
            CompositionLocalProvider(LocalFlatSpecTextHorizontalPadding provides childTextHorizontalPadding) {
                Column(
                    modifier = stackModifier,
                    verticalArrangement = Arrangement.spacedBy(gap)
                ) {
                    RenderChildren(
                        children = children,
                        elements = elements,
                        state = state,
                        repeatScope = repeatScope,
                        repeatedChildScopes = repeatedChildScopes,
                        onOpenUrl = onOpenUrl,
                        onSetState = onSetState,
                        onAction = onAction,
                        activePath = activePath
                    )
                }
            }
            return
        }
        if (wrap || autoWrapButtonRow) {
            CompositionLocalProvider(LocalFlatSpecTextHorizontalPadding provides childTextHorizontalPadding) {
                FlowRow(
                    modifier = stackModifier,
                    horizontalArrangement = horizontalArrangement,
                    verticalArrangement = Arrangement.spacedBy(gap)
                ) {
                    RenderChildren(
                        children = children,
                        elements = elements,
                        state = state,
                        repeatScope = repeatScope,
                        repeatedChildScopes = repeatedChildScopes,
                        onOpenUrl = onOpenUrl,
                        onSetState = onSetState,
                        onAction = onAction,
                        activePath = activePath
                    )
                }
            }
            return
        }
        val verticalAlignment = when (align) {
            "center" -> Alignment.CenterVertically
            "end" -> Alignment.Bottom
            else -> Alignment.Top
        }
        CompositionLocalProvider(LocalFlatSpecTextHorizontalPadding provides childTextHorizontalPadding) {
            Row(
                modifier = stackModifier,
                horizontalArrangement = horizontalArrangement,
                verticalAlignment = verticalAlignment
            ) {
                RenderRowChildren(
                    children = children,
                    elements = elements,
                    state = state,
                    repeatScope = repeatScope,
                    repeatedChildScopes = repeatedChildScopes,
                    onOpenUrl = onOpenUrl,
                    onSetState = onSetState,
                    onAction = onAction,
                    activePath = activePath
                )
            }
        }
        return
    }

    val verticalArrangement: Arrangement.Vertical = when (justify) {
        "center" -> if (gap > 0.dp) Arrangement.spacedBy(gap, Alignment.CenterVertically) else Arrangement.Center
        "end" -> if (gap > 0.dp) Arrangement.spacedBy(gap, Alignment.Bottom) else Arrangement.Bottom
        "between" -> Arrangement.SpaceBetween
        "around" -> Arrangement.SpaceAround
        else -> if (gap > 0.dp) Arrangement.spacedBy(gap) else Arrangement.Top
    }
    val horizontalAlignment = when (align) {
        "center" -> Alignment.CenterHorizontally
        "end" -> Alignment.End
        else -> Alignment.Start
    }
    CompositionLocalProvider(LocalFlatSpecTextHorizontalPadding provides childTextHorizontalPadding) {
        Column(
            modifier = stackModifier,
            verticalArrangement = verticalArrangement,
            horizontalAlignment = horizontalAlignment
        ) {
            RenderChildren(
                children = children,
                elements = elements,
                state = state,
                repeatScope = repeatScope,
                repeatedChildScopes = repeatedChildScopes,
                onOpenUrl = onOpenUrl,
                onSetState = onSetState,
                onAction = onAction,
                activePath = activePath
            )
        }
    }
}

// moved from FlatSpecRenderer.kt (RenderList)
@Composable
internal fun RenderList(
    children: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    repeatedChildScopes: List<RepeatScope>?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    onAction: (Any?, RepeatScope?) -> Int,
    activePath: Set<String>,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        RenderChildren(
            children = children,
            elements = elements,
            state = state,
            repeatScope = repeatScope,
            repeatedChildScopes = repeatedChildScopes,
            onOpenUrl = onOpenUrl,
            onSetState = onSetState,
            onAction = onAction,
            activePath = activePath
        )
    }
}

// moved from FlatSpecRenderer.kt (RenderCard)
@Composable
internal fun RenderCard(
    elementId: String,
    props: Map<String, Any?>,
    children: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    repeatedChildScopes: List<RepeatScope>?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    onAction: (Any?, RepeatScope?) -> Int,
    activePath: Set<String>,
    modifier: Modifier = Modifier
) {
    val allChildren = children.ifEmpty {
        props["child"]?.toString()?.let { listOf(it) } ?: emptyList()
    }
    val computedFunctions = LocalFlatSpecComputedFunctions.current
    extractFlatSourceSection(
        elementId = elementId,
        props = props,
        children = allChildren,
        elements = elements,
        state = state,
        repeatScope = repeatScope,
        computedFunctions = computedFunctions,
        allowChildSourceCue = true
    )?.let { sourceSection ->
        RenderFlatSourceSection(
            section = sourceSection,
            onOpenUrl = onOpenUrl,
            modifier = modifier,
            wrapInCard = true,
            showTitle = true
        )
        return
    }
    extractEmailPreviewPropsFromCard(allChildren, elements, state, repeatScope)?.let { emailProps ->
        RenderEmailPreview(emailProps, modifier)
        return
    }
    val hasExplicitCardPadding = props.containsKey("contentPadding") ||
        props.containsKey("contentPaddingHorizontal") ||
        props.containsKey("contentPaddingVertical") ||
        props.containsKey("padding") ||
        props.containsKey("paddingHorizontal") ||
        props.containsKey("paddingVertical")
    val defaultContentPadding = if (!hasExplicitCardPadding &&
        allChildren.size == 1 &&
        isPaddedContainerElement(elements[allChildren.first()])
    ) {
        0.dp
    } else {
        10.dp
    }
    val contentPaddingAll = asFlatSpacingDp(props["contentPadding"]) ?: asFlatSpacingDp(props["padding"])
    val contentPaddingHorizontal = asFlatSpacingDp(props["contentPaddingHorizontal"])
        ?: asFlatSpacingDp(props["paddingHorizontal"])
        ?: contentPaddingAll
        ?: defaultContentPadding
    val contentPaddingVertical = asFlatSpacingDp(props["contentPaddingVertical"])
        ?: asFlatSpacingDp(props["paddingVertical"])
        ?: contentPaddingAll
        ?: defaultContentPadding
    val declaredTitle = props["title"]?.toString()?.trim().orEmpty()
    val declaredSubtitle = props["subtitle"]?.toString()?.trim().orEmpty()
    val tone = props["tone"]?.toString()?.trim()?.lowercase().orEmpty()
    val declaredColors = when (tone) {
        "info" -> CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer,
            contentColor = MaterialTheme.colorScheme.onSecondaryContainer
        )
        "success" -> CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer,
            contentColor = MaterialTheme.colorScheme.onPrimaryContainer
        )
        "warning" -> CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.tertiaryContainer,
            contentColor = MaterialTheme.colorScheme.onTertiaryContainer
        )
        "error" -> CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer,
            contentColor = MaterialTheme.colorScheme.onErrorContainer
        )
        else -> flatSpecCardColors()
    }
    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
            .accessibilitySemantics(
                props = props,
                mergeDescendants = false
            ),
        colors = declaredColors,
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        shape = RoundedCornerShape(16.dp),
        border = flatSpecCardBorder()
    ) {
        CompositionLocalProvider(LocalFlatSpecTextHorizontalPadding provides 0.dp) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(
                        horizontal = contentPaddingHorizontal,
                        vertical = contentPaddingVertical
                    )
            ) {
                if (declaredTitle.isNotBlank()) {
                    Text(
                        text = declaredTitle,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.semantics { heading() }
                    )
                }
                if (declaredSubtitle.isNotBlank()) {
                    Text(
                        text = declaredSubtitle,
                        style = MaterialTheme.typography.bodyMedium,
                        color = LocalContentColor.current.copy(alpha = 0.78f),
                        modifier = Modifier.padding(top = if (declaredTitle.isNotBlank()) 2.dp else 0.dp)
                    )
                }
                if ((declaredTitle.isNotBlank() || declaredSubtitle.isNotBlank()) && allChildren.isNotEmpty()) {
                    Spacer(Modifier.size(8.dp))
                }
                RenderChildren(
                    children = allChildren,
                    elements = elements,
                    state = state,
                    repeatScope = repeatScope,
                    repeatedChildScopes = repeatedChildScopes,
                    onOpenUrl = onOpenUrl,
                    onSetState = onSetState,
                    onAction = onAction,
                    activePath = activePath
                )
            }
        }
    }
}

// moved from FlatSpecRenderer.kt (RenderText)
@Composable
internal fun RenderText(props: Map<String, Any?>, modifier: Modifier = Modifier) {
    val rawText = (
        props["text"]
            ?: props["title"]
            ?: props["label"]
            ?: props["content"]
            ?: props["value"]
        )
        ?.toString()
        .orEmpty()
    if (rawText.isBlank()) return
    parseFencedCodeBlock(rawText)?.let { codeBlock ->
        RenderCodeBlock(codeBlock = codeBlock, modifier = modifier)
        return
    }
    inferCodeBlockFromPlainText(rawText, props)?.let { codeBlock ->
        RenderCodeBlock(codeBlock = codeBlock, modifier = modifier)
        return
    }
    if (looksLikeFormulaText(rawText)) {
        RenderFormula(
            props = mapOf(
                "latex" to rawText,
                "title" to props["formulaTitle"]
            ),
            modifier = modifier
        )
        return
    }
    val markdown = parseSupportedMarkdownText(rawText)
    val rawVariant = (
        props["variant"]?.toString()
            ?: props["typography"]?.toString()
        )
        ?.lowercase()
        .orEmpty()
    val impliedHeadingVariant = when (markdown.headingLevel) {
        1 -> "h1"
        2 -> "h2"
        3 -> "h3"
        else -> ""
    }
    val normalizedVariantSource = if (rawVariant.isBlank()) impliedHeadingVariant else rawVariant
    val variant = when {
        normalizedVariantSource == "h1" || normalizedVariantSource.contains("headline-large") -> "h1"
        normalizedVariantSource == "h2" || normalizedVariantSource.contains("headline") || normalizedVariantSource.contains("title-large") -> "h2"
        normalizedVariantSource == "h3" || normalizedVariantSource.contains("title") || normalizedVariantSource.contains("subtitle") || normalizedVariantSource.contains("heading") -> "h3"
        normalizedVariantSource.contains("caption") || normalizedVariantSource.contains("label") || normalizedVariantSource.contains("body-small") -> "caption"
        normalizedVariantSource.contains("chip") -> "chip"
        else -> normalizedVariantSource
    }
    val horizontalPadding = asFlatSpacingDp(props["textPaddingHorizontal"])
        ?: asFlatSpacingDp(props["paddingHorizontal"])
        ?: LocalFlatSpecTextHorizontalPadding.current
    val primaryTextColor = MaterialTheme.colorScheme.onSurface
    when (variant) {
        "h1" -> Text(
            text = markdown.content,
            style = MaterialTheme.typography.headlineMedium.copy(fontWeight = FontWeight.Bold),
            color = primaryTextColor,
            modifier = modifier
                .padding(horizontal = horizontalPadding, vertical = 8.dp)
                .accessibilitySemantics(props = props, isHeading = true)
        )
        "h2" -> Text(
            text = markdown.content,
            style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
            color = primaryTextColor,
            modifier = modifier
                .padding(horizontal = horizontalPadding, vertical = 6.dp)
                .accessibilitySemantics(props = props, isHeading = true)
        )
        "h3" -> Text(
            text = markdown.content,
            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
            color = primaryTextColor,
            modifier = modifier
                .padding(horizontal = horizontalPadding, vertical = 4.dp)
                .accessibilitySemantics(props = props, isHeading = true)
        )
        "caption", "label" -> Text(
            text = markdown.content,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = modifier
                .padding(horizontal = horizontalPadding, vertical = 2.dp)
                .accessibilitySemantics(props = props)
        )
        "chip" -> Surface(
            shape = RoundedCornerShape(16.dp),
            color = MaterialTheme.colorScheme.secondaryContainer,
            modifier = modifier
                .padding(2.dp)
                .accessibilitySemantics(props = props, fallbackLabel = rawText)
        ) {
            Text(
                text = markdown.content,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSecondaryContainer,
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
            )
        }
        else -> Text(
            text = markdown.content,
            style = MaterialTheme.typography.bodyMedium,
            color = primaryTextColor,
            modifier = modifier
                .padding(horizontal = horizontalPadding, vertical = 2.dp)
                .accessibilitySemantics(props = props)
        )
    }
}

// moved from FlatSpecRenderer.kt (RenderDivider)
@Composable
internal fun RenderDivider(modifier: Modifier = Modifier) {
    HorizontalDivider(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
        color = MaterialTheme.colorScheme.outlineVariant
    )
}
