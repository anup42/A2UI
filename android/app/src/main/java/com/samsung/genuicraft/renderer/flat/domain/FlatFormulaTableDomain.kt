package com.samsung.genuicraft.renderer.flat.domain

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
import com.samsung.genuicraft.renderer.flat.parse.*
import com.samsung.genuicraft.renderer.flat.expr.*
import com.samsung.genuicraft.renderer.flat.runtime.*
import com.samsung.genuicraft.renderer.flat.compose.table.*

// moved from FlatSpecRenderer.kt (isFormulaVariableHeaderSet)
internal fun isFormulaVariableHeaderSet(headers: List<String>): Boolean {
    val tokens = headers.map(::normalizeTableHeaderForMatch)
    val hasVariable = tokens.any { token ->
        token in setOf("variable", "symbol", "term", "parameter")
    }
    val hasDescription = tokens.any { token ->
        token.contains("description") || token.contains("meaning") || token.contains("definition")
    }
    val hasValue = tokens.any { token ->
        token in setOf("value", "amount", "input value", "given")
    }
    return headers.size in 2..4 && hasVariable && (hasDescription || hasValue)
}

// moved from FlatSpecRenderer.kt (isCalculationBreakdownHeaderSet)
internal fun isCalculationBreakdownHeaderSet(headers: List<String>): Boolean {
    val tokens = headers.map(::normalizeTableHeaderForMatch)
    val hasLabel = tokens.any { token ->
        token in setOf("component", "item", "metric", "field", "label", "cost component")
    }
    val hasAmount = tokens.any { token ->
        token in setOf("amount", "value", "cost", "total", "payment") ||
            token.contains("amount") ||
            token.contains("payment")
    }
    return headers.size == 2 && hasLabel && hasAmount
}

// moved from FlatSpecRenderer.kt (isFormulaVariablesTable)
internal fun isFormulaVariablesTable(table: FlatDirectTableModel): Boolean =
    table.domain == "formula" && isFormulaVariableHeaderSet(table.columns.map { it.label })

// moved from FlatSpecRenderer.kt (isCalculationBreakdownTable)
internal fun isCalculationBreakdownTable(table: FlatDirectTableModel): Boolean {
    if (table.domain != "formula") return false
    if (!isCalculationBreakdownHeaderSet(table.columns.map { it.label })) return false
    return table.rows.any { row ->
        row.any { value ->
            value.contains('$') ||
                value.contains('€') ||
                value.contains('£') ||
                value.contains('₹') ||
                looksLikeNumericTableValue(value)
        }
    }
}

// moved from FlatSpecRenderer.kt (RenderFormulaVariablesTable)
@Composable
internal fun RenderFormulaVariablesTable(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier
) {
    if (rows.isEmpty()) return
    val variableIndex = headers.indexOfFirst { normalizeTableHeaderForMatch(it) in setOf("variable", "symbol", "term", "parameter", "input") }
        .takeIf { it >= 0 } ?: 0
    val descriptionIndex = headers.indexOfFirst {
        normalizeTableHeaderForMatch(it).let { token ->
            token.contains("description") || token.contains("meaning") || token.contains("definition")
        }
    }
    val valueIndex = headers.indexOfFirst {
        normalizeTableHeaderForMatch(it).let { token ->
            token in setOf("value", "amount", "input value", "given")
        }
    }
    Card(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = "Formula variables table with ${rows.size} variables"
            },
        shape = RoundedCornerShape(18.dp),
        colors = flatSpecCardColors(),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.62f))
                    .padding(horizontal = 10.dp, vertical = 7.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text(
                    text = headers.getOrNull(variableIndex).orEmpty().ifBlank { "Variable" },
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.width(58.dp)
                )
                Text(
                    text = headers.getOrNull(descriptionIndex).orEmpty().ifBlank { "Meaning" },
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f)
                )
                if (valueIndex >= 0) {
                    Text(
                        text = headers.getOrNull(valueIndex).orEmpty().ifBlank { "Value" },
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.End,
                        modifier = Modifier.widthIn(min = 72.dp)
                    )
                }
            }
            rows.forEachIndexed { index, row ->
                val variable = row.getOrNull(variableIndex).orEmpty().trim().ifBlank { "-" }
                val description = row.getOrNull(descriptionIndex).orEmpty().trim()
                val value = row.getOrNull(valueIndex).orEmpty().trim()
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 10.dp, vertical = 8.dp)
                        .semantics(mergeDescendants = true) {
                            contentDescription = listOf(variable, description, value)
                                .filter(String::isNotBlank)
                                .joinToString(": ")
                        },
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .width(58.dp)
                            .heightIn(min = 36.dp)
                            .clip(RoundedCornerShape(11.dp))
                            .background(MaterialTheme.colorScheme.primaryContainer),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = variable,
                            style = MaterialTheme.typography.titleMedium.copy(
                                fontWeight = FontWeight.Bold,
                                fontFamily = FontFamily.Monospace
                            ),
                            color = MaterialTheme.colorScheme.onPrimaryContainer,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                    Text(
                        text = parseBoldMarkdown(description),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                        modifier = Modifier.weight(1f)
                    )
                    if (valueIndex >= 0) {
                        Text(
                            text = value,
                            style = MaterialTheme.typography.labelLarge.copy(
                                fontWeight = FontWeight.SemiBold,
                                fontFamily = FontFamily.Monospace
                            ),
                            color = MaterialTheme.colorScheme.primary,
                            textAlign = TextAlign.End,
                            modifier = Modifier.widthIn(min = 72.dp)
                        )
                    }
                }
                if (index < rows.lastIndex) {
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.55f))
                }
            }
        }
    }
}

// moved from FlatSpecRenderer.kt (RenderCalculationBreakdownTable)
@Composable
internal fun RenderCalculationBreakdownTable(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier
) {
    if (rows.isEmpty()) return
    val labelIndex = headers.indexOfFirst {
        normalizeTableHeaderForMatch(it) in setOf("component", "item", "metric", "field", "label", "cost component")
    }.takeIf { it >= 0 } ?: 0
    val amountIndex = headers.indexOfFirst {
        normalizeTableHeaderForMatch(it).let { token ->
            token.contains("amount") || token.contains("payment") || token == "value" || token == "cost" || token == "total"
        }
    }.takeIf { it >= 0 } ?: rows.firstOrNull()?.indices?.firstOrNull { it != labelIndex } ?: 1
    val labelHeader = headers.getOrNull(labelIndex).orEmpty().ifBlank { "Item" }
    val amountHeader = headers.getOrNull(amountIndex).orEmpty().ifBlank { "Value" }
    val inputStyleTable = normalizeTableHeaderForMatch(labelHeader) in setOf("metric", "input", "field") &&
        normalizeTableHeaderForMatch(amountHeader) == "value"
    val labelWeight = if (inputStyleTable) 0.66f else 0.54f
    val amountWeight = 1f - labelWeight
    Card(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = "Calculation breakdown table with ${rows.size} rows"
            },
        shape = RoundedCornerShape(18.dp),
        colors = flatSpecCardColors(),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.62f))
                    .padding(horizontal = 10.dp, vertical = 7.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    text = labelHeader,
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(labelWeight)
                )
                Text(
                    text = amountHeader,
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.End,
                    modifier = Modifier.weight(amountWeight)
                )
            }
            rows.forEachIndexed { index, row ->
                val label = row.getOrNull(labelIndex).orEmpty().trim()
                val amount = row.getOrNull(amountIndex).orEmpty().trim()
                val highlight = label.contains("monthly", ignoreCase = true) ||
                    label.contains("lifetime", ignoreCase = true) ||
                    label.contains("total", ignoreCase = true)
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 10.dp, vertical = 8.dp)
                        .semantics(mergeDescendants = true) {
                            contentDescription = "$label: $amount"
                        },
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        text = parseBoldMarkdown(label),
                        style = MaterialTheme.typography.bodyMedium.copy(
                            fontWeight = if (highlight) FontWeight.SemiBold else FontWeight.Normal
                        ),
                        color = MaterialTheme.colorScheme.onSurface,
                        modifier = Modifier.weight(labelWeight)
                    )
                    Text(
                        text = amount,
                        style = MaterialTheme.typography.bodyMedium.copy(
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace
                        ),
                        color = if (highlight) {
                            MaterialTheme.colorScheme.primary
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        },
                        textAlign = TextAlign.End,
                        maxLines = 1,
                        softWrap = false,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(amountWeight)
                    )
                }
                if (index < rows.lastIndex) {
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.6f))
                }
            }
        }
    }
}
