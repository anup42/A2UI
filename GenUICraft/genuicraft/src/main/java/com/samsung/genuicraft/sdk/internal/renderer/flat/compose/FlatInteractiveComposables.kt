package com.samsung.genuicraft.sdk.internal.renderer.flat.compose

import com.samsung.genuicraft.sdk.internal.renderer.*

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
import com.samsung.genuicraft.sdk.internal.theme.GenUiTokens
import com.samsung.genuicraft.sdk.internal.theme.GenUiCardTone
import com.samsung.genuicraft.sdk.internal.theme.genUiCardContainerColor
import com.google.gson.GsonBuilder
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.samsung.genuicraft.sdk.internal.renderer.native.NativeTextFormatter
import com.samsung.genuicraft.sdk.internal.renderer.native.NativePayloadParser
import com.samsung.genuicraft.sdk.internal.renderer.native.ParsedButton
import com.samsung.genuicraft.sdk.internal.renderer.native.intents.flight.NativeFlightSemantics
import com.samsung.genuicraft.sdk.internal.renderer.native.intents.flight.NativeFlightUiRenderer
import com.samsung.genuicraft.sdk.internal.renderer.native.intents.weather.NativeWeatherSemantics
import com.samsung.genuicraft.sdk.internal.renderer.native.intents.weather.NativeWeatherUiRenderer
import com.samsung.genuicraft.sdk.internal.renderer.native.media.NativeMediaVisualUtils
import com.samsung.genuicraft.sdk.internal.renderer.native.parser.NativeSourceParsing
import com.samsung.genuicraft.sdk.internal.renderer.native.parser.NativeStructureParsing
import com.samsung.genuicraft.sdk.internal.security.SafeContentPolicy
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
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.parse.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.expr.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.runtime.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.legacy.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.*

// moved from FlatSpecRenderer.kt (RenderButton)
@Composable
internal fun RenderButton(
    props: Map<String, Any?>,
    onMap: Map<String, Any?>?,
    repeatScope: RepeatScope?,
    onAction: (Any?, RepeatScope?) -> Int,
    modifier: Modifier = Modifier
) {
    val label = (props["label"] ?: props["text"])?.toString().orEmpty().ifBlank { "Open" }
    val icon = props["icon"]?.toString()?.trim().orEmpty()
    val variant = props["variant"]?.toString()?.lowercase().orEmpty()
    val actionCandidate = onMap?.get("press") ?: onMap?.get("click") ?: onMap?.get("tap")
    val onClick: () -> Unit = if (actionCandidate == null) {
        {}
    } else {
        { onAction(actionCandidate, repeatScope) }
    }
    val buttonModifier = modifier
        .padding(horizontal = 16.dp, vertical = 4.dp)
        .let { base ->
            if (label.length > 20) base.fillMaxWidth() else base
        }
        .accessibilitySemantics(
            props = props,
            fallbackLabel = label,
            semanticRole = Role.Button,
            state = "Disabled".takeIf { actionCandidate == null }
        )
    if (variant == "borderless" || variant == "text" || variant == "outlined") {
        OutlinedButton(
            onClick = onClick,
            enabled = actionCandidate != null,
            modifier = buttonModifier
        ) {
            if (icon.isNotBlank()) {
                RenderIcon(
                    props = mapOf("name" to icon, "decorative" to true, "size" to 18),
                    modifier = Modifier.size(18.dp)
                )
                Spacer(Modifier.width(6.dp))
            }
            Text(text = label)
        }
    } else {
        Button(
            onClick = onClick,
            enabled = actionCandidate != null,
            modifier = buttonModifier
        ) {
            if (icon.isNotBlank()) {
                RenderIcon(
                    props = mapOf("name" to icon, "decorative" to true, "size" to 18),
                    modifier = Modifier.size(18.dp)
                )
                Spacer(Modifier.width(6.dp))
            }
            Text(text = label)
        }
    }
}

// moved from FlatSpecRenderer.kt (RenderTabs)
@Composable
internal fun RenderTabs(
    props: Map<String, Any?>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    onAction: (Any?, RepeatScope?) -> Int,
    activePath: Set<String>,
    modifier: Modifier = Modifier
) {
    val tabs = props["tabs"] as? List<*> ?: return
    if (tabs.isEmpty()) return

    val activeTabId = props["activeTabId"]?.toString().orEmpty()
    val initialIndex = tabs.indexOfFirst { tab ->
        val map = tab as? Map<*, *> ?: return@indexOfFirst false
        val tabChild = map["child"]?.toString()
            ?: map["content"]?.toString()
            ?: map["id"]?.toString()
            ?: map["element"]?.toString()
        tabChild == activeTabId
    }.takeIf { it >= 0 } ?: 0

    var selectedIndex by remember(tabs, activeTabId) { mutableIntStateOf(initialIndex) }

    Column(modifier = modifier.fillMaxWidth()) {
        ScrollableTabRow(selectedTabIndex = selectedIndex) {
            tabs.forEachIndexed { index, tabAny ->
                val tab = tabAny as? Map<*, *> ?: return@forEachIndexed
                val title = (tab["title"]?.toString() ?: tab["label"]?.toString()).orEmpty().ifBlank { "Tab ${index + 1}" }
                Tab(
                    selected = selectedIndex == index,
                    onClick = { selectedIndex = index },
                    text = { Text(title) }
                )
            }
        }
        Spacer(Modifier.height(8.dp))
        val selectedTab = tabs.getOrNull(selectedIndex) as? Map<*, *>
        val childId = (
            selectedTab?.get("child")?.toString()
                ?: selectedTab?.get("content")?.toString()
                ?: selectedTab?.get("id")?.toString()
                ?: selectedTab?.get("element")?.toString()
            ).orEmpty()
        if (childId.isNotBlank()) {
            RenderElement(
                elementId = childId,
                elements = elements,
                state = state,
                repeatScope = repeatScope,
                onOpenUrl = onOpenUrl,
                onSetState = onSetState,
                onAction = onAction,
                activePath = activePath
            )
        }
    }
}

// moved from FlatSpecRenderer.kt (RenderModal)
@Composable
internal fun RenderModal(
    props: Map<String, Any?>,
    children: List<String>,
    elements: Map<String, FlatElement>,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    onOpenUrl: (String) -> Unit,
    onSetState: (String, Any?) -> Unit,
    onAction: (Any?, RepeatScope?) -> Int,
    activePath: Set<String>,
    modifier: Modifier = Modifier
) {
    val title = props["title"]?.toString().orEmpty().ifBlank { "Modal" }
    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp, vertical = 4.dp)
            .accessibilitySemantics(
                props = props,
                fallbackLabel = title
            ),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.fillMaxWidth().padding(8.dp)) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier
                    .padding(horizontal = 8.dp, vertical = 4.dp)
                    .semantics { heading() }
            )
            val ids = buildList {
                props["trigger"]?.toString()?.takeIf { it.isNotBlank() }?.let(::add)
                props["content"]?.toString()?.takeIf { it.isNotBlank() }?.let(::add)
                addAll(children)
            }.distinct()
            ids.forEach { childId ->
                RenderElement(
                    elementId = childId,
                    elements = elements,
                    state = state,
                    repeatScope = repeatScope,
                    onOpenUrl = onOpenUrl,
                    onSetState = onSetState,
                    onAction = onAction,
                    activePath = activePath
                )
            }
        }
    }
}
