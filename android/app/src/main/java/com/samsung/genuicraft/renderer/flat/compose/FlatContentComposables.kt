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
import com.samsung.genuicraft.renderer.flat.compose.*
import com.samsung.genuicraft.renderer.flat.legacy.*
import com.samsung.genuicraft.renderer.flat.model.*

// moved from FlatSpecRenderer.kt (RenderFormula)
@Composable
internal fun RenderFormula(props: Map<String, Any?>, modifier: Modifier = Modifier) {
    val rawFormula = (
        props["latex"]
            ?: props["formula"]
            ?: props["text"]
            ?: props["value"]
        )
        ?.toString()
        .orEmpty()
    val normalizedFormula = normalizeFormulaText(rawFormula)
    if (normalizedFormula.isBlank()) return
    val title = props["title"]?.toString()?.trim().orEmpty()
    val subtitle = props["subtitle"]?.toString()?.trim().orEmpty()
    val result = props["result"]?.toString()?.trim().orEmpty()
    val displayFormula = plainBracketFractionToLatex(normalizedFormula)
    val segments = parseFormulaSegments(displayFormula)
    Surface(
        shape = RoundedCornerShape(18.dp),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.64f),
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = LocalFlatSpecTextHorizontalPadding.current, vertical = 6.dp)
            .semantics(mergeDescendants = true) {
                contentDescription = "Formula ${readableFormulaText(displayFormula)}"
            }
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            if (title.isNotBlank()) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
            if (subtitle.isNotBlank()) {
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Surface(
                shape = RoundedCornerShape(14.dp),
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.78f),
                modifier = Modifier.fillMaxWidth()
            ) {
                val scrollState = rememberScrollState()
                RenderFormulaExpression(
                    displayFormula = displayFormula,
                    segments = segments,
                    scrollState = scrollState
                )
            }
            if (result.isNotBlank()) {
                Surface(
                    shape = RoundedCornerShape(999.dp),
                    color = MaterialTheme.colorScheme.primaryContainer
                ) {
                    Text(
                        text = result,
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 7.dp)
                    )
                }
            }
        }
    }
}

// moved from FlatSpecRenderer.kt (RenderCodeBlock)
@Composable
internal fun RenderCodeBlock(codeBlock: FencedCodeBlock, modifier: Modifier = Modifier) {
    if (codeBlock.code.isBlank() && codeBlock.title.isNullOrBlank()) return
    val isConsole = codeBlock.isConsole ||
        codeBlock.language.equals("console", ignoreCase = true) ||
        codeBlock.language.equals("terminal", ignoreCase = true) ||
        codeBlock.language.equals("shell", ignoreCase = true) ||
        codeBlock.language.equals("bash", ignoreCase = true)
    val label = codeBlock.title?.trim()?.takeIf { it.isNotBlank() }
        ?: codeBlock.language?.trim()?.takeIf { it.isNotBlank() }
        ?: if (isConsole) "Console" else "Code"
    val containerColor = if (isConsole) Color(0xFF0B1020) else Color(0xFF111827)
    val headerColor = if (isConsole) Color(0xFF67E8F9) else Color(0xFFA7F3D0)
    val bodyColor = Color(0xFFE5E7EB)
    val clipboard = LocalClipboardManager.current
    val copyText = codeBlock.code.trim()
    Surface(
        shape = RoundedCornerShape(16.dp),
        color = containerColor,
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = LocalFlatSpecTextHorizontalPadding.current, vertical = 6.dp)
            .semantics {
                contentDescription = buildString {
                    append(if (isConsole) "Console log" else "Code block")
                    label.takeIf { it.isNotBlank() }?.let { language ->
                        append(", ")
                        append(language)
                    }
                    codeBlock.code.trim().takeIf { it.isNotBlank() }?.let { code ->
                        append(". ")
                        append(code)
                    }
                }
            }
    ) {
        Column(modifier = Modifier.fillMaxWidth().padding(12.dp)) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 6.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = label.uppercase(),
                    style = MaterialTheme.typography.labelSmall,
                    color = headerColor,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f)
                )
                if (copyText.isNotBlank()) {
                    Icon(
                        imageVector = Icons.Filled.ContentCopy,
                        contentDescription = if (isConsole) "Copy shell output" else "Copy code",
                        tint = Color.White,
                        modifier = Modifier
                            .clip(RoundedCornerShape(999.dp))
                            .background(Color.White.copy(alpha = 0.14f))
                            .clickable {
                                clipboard.setText(AnnotatedString(copyText))
                            }
                            .semantics {
                                role = Role.Button
                            }
                            .padding(6.dp)
                            .size(16.dp)
                    )
                }
            }
            val codeTextModifier = if (isConsole) {
                Modifier.fillMaxWidth()
            } else {
                Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
            }
            Text(
                text = codeBlock.code.ifBlank { " " },
                style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                color = bodyColor,
                softWrap = isConsole,
                modifier = codeTextModifier
            )
        }
    }
}

// moved from FlatSpecRenderer.kt (codeBlockFromProps)
internal fun codeBlockFromProps(
    props: Map<String, Any?>,
    defaultLanguage: String? = null,
    isConsole: Boolean = false
): FencedCodeBlock {
    val rawContent = listOf("code", "text", "content", "value", "output", "log", "logs")
        .firstNotNullOfOrNull { key -> props[key]?.toString()?.takeIf { it.isNotBlank() } }
        .orEmpty()
    val parsed = parseFencedCodeBlock(rawContent)
    val language = props["language"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
        ?: props["lang"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
        ?: props["syntax"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
        ?: parsed?.language
        ?: defaultLanguage
    val title = props["title"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
        ?: props["label"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
    val console = isConsole ||
        props["kind"]?.toString()?.contains("console", ignoreCase = true) == true ||
        props["role"]?.toString()?.contains("console", ignoreCase = true) == true ||
        language.equals("console", ignoreCase = true) ||
        language.equals("terminal", ignoreCase = true)
    return FencedCodeBlock(
        code = parsed?.code ?: rawContent,
        language = language,
        title = title,
        isConsole = console
    )
}
