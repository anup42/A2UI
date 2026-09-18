package com.samsung.genuicraft.sdk.internal.renderer.flat.domain

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
import androidx.compose.material.icons.filled.Language
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
import com.samsung.genuicraft.sdk.internal.renderer.flat.parse.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.expr.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.runtime.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.*

// moved from FlatSpecRenderer.kt (looksLikeProcessStateTable)
internal fun looksLikeProcessStateTable(
    headers: List<String>,
    rows: List<List<String>>,
    domain: String = "generic"
): Boolean {
    if (rows.size !in 2..8) return false
    if (domain !in setOf("status", "schedule", "generic")) return false
    val tokens = headers.map(::normalizeTableHeaderForMatch)
    val hasState = tokens.any { token ->
        token == "state" ||
            token == "ui state" ||
            token == "screen state" ||
            token == "status" ||
            token == "step" ||
            token == "stage"
    }
    val hasVisuals = tokens.any { token ->
        token.contains("visual") ||
            token.contains("screen") ||
            token.contains("view") ||
            token.contains("interface")
    }
    val hasFeedback = tokens.any { token ->
        token.contains("feedback") ||
            token.contains("message") ||
            token.contains("text") ||
            token.contains("result")
    }
    val rowSignals = rows.flatten().joinToString(" ").lowercase().let { combined ->
        listOf("scan", "success", "invalid", "error", "ready", "state", "overlay", "check-in", "qr").count {
            combined.contains(it)
        }
    }
    return hasState && (hasVisuals || hasFeedback) && rowSignals >= 2
}

// moved from FlatSpecRenderer.kt (looksLikeIncidentStatusTable)
internal fun looksLikeIncidentStatusTable(
    headers: List<String>,
    rows: List<List<String>>,
    domain: String = "generic"
): Boolean {
    if (rows.isEmpty()) return false
    if (domain !in setOf("status", "generic")) return false
    val tokens = headers.map(::normalizeTableHeaderForMatch)
    val hasComponent = tokens.any { token ->
        token == "component" ||
            token == "service" ||
            token == "system" ||
            token == "module" ||
            token.contains("service")
    }
    val hasStatus = tokens.any { token ->
        token == "status" ||
            token == "current status" ||
            token == "health" ||
            token.contains("status")
    }
    val hasNotes = tokens.any { token ->
        token == "notes" ||
            token == "impact" ||
            token == "details" ||
            token == "description" ||
            token.contains("message")
    }
    val rowSignals = rows.flatten().joinToString(" ").lowercase().let { combined ->
        listOf("outage", "degraded", "operational", "partial", "incident", "failure", "delayed").count {
            combined.contains(it)
        }
    }
    return hasComponent && hasStatus && hasNotes && rowSignals >= 1
}

// moved from FlatSpecRenderer.kt (RenderIncidentStatusDashboard)
@Composable
internal fun RenderIncidentStatusDashboard(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    landscape: Boolean
) {
    if (rows.isEmpty()) return
    val componentIndex = incidentComponentColumnIndex(headers)
    val statusIndex = incidentStatusColumnIndex(headers, componentIndex)
    val notesIndex = incidentNotesColumnIndex(headers, setOf(componentIndex, statusIndex))
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        shape = RoundedCornerShape(28.dp),
        color = Color.Transparent,
        tonalElevation = 0.dp,
        shadowElevation = 0.dp
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.linearGradient(
                        listOf(
                            Color(0xFF130D1B),
                            Color(0xFF2B1724),
                            Color(0xFF0F2433)
                        )
                    )
                )
                .padding(14.dp)
        ) {
            if (landscape) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(14.dp),
                    verticalAlignment = Alignment.Top
                ) {
                    IncidentStatusHero(
                        rows = rows,
                        componentIndex = componentIndex,
                        statusIndex = statusIndex,
                        modifier = Modifier.widthIn(min = 220.dp, max = 280.dp)
                    )
                    Column(
                        modifier = Modifier
                            .weight(1f)
                            .heightIn(max = 430.dp)
                            .verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(9.dp)
                    ) {
                        rows.forEachIndexed { rowIndex, row ->
                            IncidentServiceRowCard(
                                headers = headers,
                                row = row,
                                rowIndex = rowIndex,
                                componentIndex = componentIndex,
                                statusIndex = statusIndex,
                                notesIndex = notesIndex
                            )
                        }
                    }
                }
            } else {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    IncidentStatusHero(
                        rows = rows,
                        componentIndex = componentIndex,
                        statusIndex = statusIndex
                    )
                    rows.forEachIndexed { rowIndex, row ->
                        IncidentServiceRowCard(
                            headers = headers,
                            row = row,
                            rowIndex = rowIndex,
                            componentIndex = componentIndex,
                            statusIndex = statusIndex,
                            notesIndex = notesIndex
                        )
                    }
                }
            }
        }
    }
}

// moved from FlatSpecRenderer.kt (RenderProcessStateTable)
@Composable
internal fun RenderProcessStateTable(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    landscape: Boolean
) {
    if (rows.isEmpty()) return
    val stateIndex = processStateColumnIndex(headers)
    val visualIndex = processVisualColumnIndex(headers, stateIndex)
    val feedbackIndex = processFeedbackColumnIndex(headers, stateIndex)
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        shape = RoundedCornerShape(28.dp),
        color = Color.Transparent,
        tonalElevation = 0.dp,
        shadowElevation = 0.dp
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.linearGradient(
                        listOf(
                            Color(0xFF07111F),
                            Color(0xFF102A43),
                            Color(0xFF062E2E)
                        )
                    )
                )
                .padding(14.dp)
        ) {
            if (landscape) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(14.dp),
                    verticalAlignment = Alignment.Top
                ) {
                    ProcessScannerHero(
                        rows = rows,
                        stateIndex = stateIndex,
                        modifier = Modifier.widthIn(min = 220.dp, max = 280.dp)
                    )
                    Column(
                        modifier = Modifier
                            .weight(1f)
                            .heightIn(max = 430.dp)
                            .verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(9.dp)
                    ) {
                        rows.forEachIndexed { rowIndex, row ->
                            ProcessStateStepCard(
                                headers = headers,
                                row = row,
                                rowIndex = rowIndex,
                                stateIndex = stateIndex,
                                visualIndex = visualIndex,
                                feedbackIndex = feedbackIndex
                            )
                        }
                    }
                }
            } else {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    ProcessScannerHero(rows = rows, stateIndex = stateIndex)
                    rows.forEachIndexed { rowIndex, row ->
                        ProcessStateStepCard(
                            headers = headers,
                            row = row,
                            rowIndex = rowIndex,
                            stateIndex = stateIndex,
                            visualIndex = visualIndex,
                            feedbackIndex = feedbackIndex
                        )
                    }
                }
            }
        }
    }
}
