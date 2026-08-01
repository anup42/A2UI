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

// moved from FlatSpecRenderer.kt (RenderTextField)
@OptIn(ExperimentalFoundationApi::class)
@Composable
internal fun RenderTextField(
    props: Map<String, Any?>,
    onSetState: (String, Any?) -> Unit,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    modifier: Modifier = Modifier
) {
    val computedFunctions = LocalFlatSpecComputedFunctions.current
    val label = props["label"]?.toString().orEmpty().ifBlank { "Input" }
    val placeholder = props["placeholder"]?.toString()?.trim().orEmpty()
    val bindPath = bindPathFromValueExpression(props["value"], repeatScope) ?: props["statePath"]?.toString()
    val value = FlatExprResolver.resolveString(props["value"], state, repeatScope, computedFunctions)
    var localValue by remember(label) { mutableStateOf(value) }
    val textValue = if (!bindPath.isNullOrBlank()) value else localValue
    val bringIntoViewRequester = remember { BringIntoViewRequester() }
    val coroutineScope = rememberCoroutineScope()

    OutlinedTextField(
        value = textValue,
        onValueChange = { next ->
            if (!bindPath.isNullOrBlank()) {
                onSetState(bindPath, next)
            } else {
                localValue = next
            }
        },
        label = { Text(label) },
        placeholder = if (placeholder.isBlank()) null else ({ Text(placeholder) }),
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
            .bringIntoViewRequester(bringIntoViewRequester)
            .onFocusChanged { focusState ->
                if (focusState.isFocused) {
                    coroutineScope.launch {
                        bringIntoViewRequester.bringIntoView()
                    }
                }
            }
    )
}

// moved from FlatSpecRenderer.kt (RenderCheckBox)
@Composable
internal fun RenderCheckBox(
    props: Map<String, Any?>,
    onSetState: (String, Any?) -> Unit,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    modifier: Modifier = Modifier
) {
    val computedFunctions = LocalFlatSpecComputedFunctions.current
    val label = props["label"]?.toString().orEmpty().ifBlank { "Option" }
    val bindPath = bindPathFromValueExpression(props["value"], repeatScope) ?: props["statePath"]?.toString()
    val checked = FlatExprResolver.resolveBoolean(props["value"], state, repeatScope, computedFunctions)
    var localChecked by remember(label) { mutableStateOf(checked) }
    val isChecked = if (!bindPath.isNullOrBlank()) checked else localChecked

    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
            .toggleable(
                value = isChecked,
                role = Role.Checkbox,
                onValueChange = { next ->
                    if (!bindPath.isNullOrBlank()) {
                        onSetState(bindPath, next)
                    } else {
                        localChecked = next
                    }
                }
            )
            .accessibilitySemantics(
                props = props,
                fallbackLabel = label,
                state = if (isChecked) "Checked" else "Not checked",
                mergeDescendants = true
            ),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Checkbox(
            checked = isChecked,
            onCheckedChange = null
        )
        // Explicit colour: these labels sit outside any Surface, so
        // LocalContentColor defaults to black and the text was almost invisible
        // on a dark background. Found by rendering on device.
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface
        )
    }
}

// moved from FlatSpecRenderer.kt (RenderChoicePicker)
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun RenderChoicePicker(
    props: Map<String, Any?>,
    onSetState: (String, Any?) -> Unit,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    modifier: Modifier = Modifier
) {
    val computedFunctions = LocalFlatSpecComputedFunctions.current
    val label = props["label"]?.toString().orEmpty().ifBlank { "Choose" }
    val bindPath = bindPathFromValueExpression(props["value"], repeatScope) ?: props["statePath"]?.toString()
    val resolvedValue = FlatExprResolver.resolve(props["value"], state, repeatScope, computedFunctions)
    val multiSelect = resolveChoicePickerMultiSelect(props, resolvedValue)
    val boundSelection = when (resolvedValue) {
        is List<*> -> resolvedValue.mapNotNull { it?.toString() }.toSet()
        is String -> if (resolvedValue.isBlank()) emptySet() else setOf(resolvedValue)
        null -> emptySet()
        else -> setOf(resolvedValue.toString())
    }
    // Fallback for an unbound picker. Without this the control looked enabled
    // but every tap was a no-op, matching TextField/CheckBox/Slider which all
    // already keep local state when there is no statePath.
    var localSelection by remember(label) { mutableStateOf(boundSelection) }
    val selected = if (!bindPath.isNullOrBlank()) boundSelection else localSelection
    val maxSelections = (props["maxSelections"] as? Number)?.toInt()
        ?: (props["maxAllowedSelections"] as? Number)?.toInt()
    val options = (props["options"] as? List<*>)?.mapNotNull { optionAny ->
        val optionMap = optionAny as? Map<*, *> ?: return@mapNotNull null
        val optionLabel = optionMap["label"]?.toString().orEmpty()
        val optionValue = optionMap["value"]?.toString().orEmpty()
        if (optionValue.isBlank()) null else optionLabel.ifBlank { optionValue } to optionValue
    }.orEmpty()

    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurface
        )
        Spacer(Modifier.height(6.dp))
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            options.forEach { (optionLabel, optionValue) ->
                val active = selected.contains(optionValue)
                val commit: (Set<String>) -> Unit = { next ->
                    if (!bindPath.isNullOrBlank()) {
                        // Single-select writes the scalar back. Writing a List
                        // here would permanently change the state value's type
                        // and break any $state/eq comparison against a string.
                        onSetState(bindPath, if (multiSelect) next.toList() else next.firstOrNull())
                    } else {
                        localSelection = next
                    }
                }
                val onClick: () -> Unit = {
                    if (multiSelect) {
                        val next = if (active) {
                            selected - optionValue
                        } else if (maxSelections != null && selected.size >= maxSelections) {
                            selected
                        } else {
                            selected + optionValue
                        }
                        commit(next)
                    } else {
                        // Mutually exclusive: re-tapping the active option keeps
                        // it selected rather than clearing to an empty value.
                        commit(setOf(optionValue))
                    }
                }
                val optionModifier = Modifier.semantics {
                    contentDescription = optionLabel
                    stateDescription = if (active) "Selected" else "Not selected"
                }
                if (active) {
                    Button(onClick = onClick, modifier = optionModifier) {
                        Text(optionLabel)
                    }
                } else {
                    OutlinedButton(onClick = onClick, modifier = optionModifier) {
                        Text(optionLabel)
                    }
                }
            }
        }
    }
}

// moved from FlatSpecRenderer.kt (resolveChoicePickerMultiSelect)
/**
 * Resolves single- vs multi-select for a ChoicePicker.
 *
 * Declared intent wins; otherwise the *type* of the current value decides, so a
 * spec whose state holds a string is treated as single-select and never widened
 * to a list. Defaults to multi-select to preserve the previous behaviour for
 * specs that declare nothing and start from an empty value.
 */
internal fun resolveChoicePickerMultiSelect(
    props: Map<String, Any?>,
    resolvedValue: Any?
): Boolean {
    (props["multiple"] as? Boolean)?.let { return it }
    val declared = (props["mode"] ?: props["selection"] ?: props["variant"])
        ?.toString()
        ?.trim()
        ?.lowercase(Locale.US)
    when (declared) {
        "single", "singleselection", "single_selection", "mutuallyexclusive",
        "mutually_exclusive", "radio", "one" -> return false
        "multiple", "multi", "multipleselection", "multiple_selection",
        "many" -> return true
    }
    return when (resolvedValue) {
        is List<*> -> true
        is String -> false
        else -> true
    }
}

// moved from FlatSpecRenderer.kt (RenderSlider)
@Composable
internal fun RenderSlider(
    props: Map<String, Any?>,
    onSetState: (String, Any?) -> Unit,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    modifier: Modifier = Modifier
) {
    val computedFunctions = LocalFlatSpecComputedFunctions.current
    val label = props["label"]?.toString().orEmpty()
    val bindPath = bindPathFromValueExpression(props["value"], repeatScope) ?: props["statePath"]?.toString()
    val min = (props["min"] as? Number)?.toFloat() ?: 0f
    val max = (props["max"] as? Number)?.toFloat() ?: 100f
    val requestedStep = (props["step"] as? Number)?.toFloat()?.takeIf { it > 0f }
    val steps = requestedStep
        ?.takeIf { max > min }
        ?.let { (((max - min) / it).roundToInt() - 1).coerceAtLeast(0) }
        ?: 0
    val resolved = FlatExprResolver.resolve(props["value"], state, repeatScope, computedFunctions)
    val initial = (resolved as? Number)?.toFloat() ?: min
    var localValue by remember(label) { mutableStateOf(initial.coerceIn(min, max)) }
    val current = if (!bindPath.isNullOrBlank()) {
        ((FlatSpecParser.getAtPath(state, bindPath) as? Number)?.toFloat() ?: initial).coerceIn(min, max)
    } else {
        localValue
    }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
    ) {
        if (label.isNotBlank()) {
            Text(
                "$label: ${current.roundToInt()}",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurface
            )
            Spacer(Modifier.height(4.dp))
        }
        Slider(
            value = current,
            valueRange = min..max,
            steps = steps,
            modifier = Modifier.semantics {
                contentDescription = label.ifBlank { "Slider" }
                stateDescription = "${current.roundToInt()} of ${max.roundToInt()}"
            },
            onValueChange = { next ->
                if (!bindPath.isNullOrBlank()) {
                    onSetState(bindPath, next)
                } else {
                    localValue = next
                }
            }
        )
    }
}

// moved from FlatSpecRenderer.kt (RenderDateTimeInput)
@OptIn(ExperimentalFoundationApi::class)
@Composable
internal fun RenderDateTimeInput(
    props: Map<String, Any?>,
    onSetState: (String, Any?) -> Unit,
    state: Map<String, Any?>,
    repeatScope: RepeatScope?,
    modifier: Modifier = Modifier
) {
    val computedFunctions = LocalFlatSpecComputedFunctions.current
    val label = props["label"]?.toString().orEmpty().ifBlank { "Date/Time" }
    val bindPath = bindPathFromValueExpression(props["value"], repeatScope) ?: props["statePath"]?.toString()
    val resolvedValue = FlatExprResolver.resolveString(props["value"], state, repeatScope, computedFunctions)
    var localValue by remember(label) { mutableStateOf(resolvedValue) }
    val value = if (!bindPath.isNullOrBlank()) resolvedValue else localValue
    val bringIntoViewRequester = remember { BringIntoViewRequester() }
    val coroutineScope = rememberCoroutineScope()
    val mode = flatDateTimeMode(props)
    // Manual entry stays available unless a spec opts out, so existing specs
    // that relied on free-text keep working.
    val allowManualEntry = props["allowManualEntry"] != false
    var showPicker by remember(label) { mutableStateOf(false) }

    val commit: (String) -> Unit = { next ->
        if (!bindPath.isNullOrBlank()) onSetState(bindPath, next) else localValue = next
    }

    OutlinedTextField(
        value = value,
        onValueChange = { next -> if (allowManualEntry) commit(next) },
        readOnly = !allowManualEntry,
        label = { Text(label) },
        placeholder = { Text(flatDateTimePlaceholder(mode)) },
        trailingIcon = {
            IconButton(
                onClick = { showPicker = true },
                modifier = Modifier.semantics {
                    contentDescription = "Pick $label"
                }
            ) {
                Icon(
                    imageVector = if (mode == FlatDateTimeMode.TIME) {
                        Icons.Filled.Schedule
                    } else {
                        Icons.Filled.CalendarToday
                    },
                    contentDescription = null
                )
            }
        },
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
            .bringIntoViewRequester(bringIntoViewRequester)
            .onFocusChanged { focusState ->
                if (focusState.isFocused) {
                    coroutineScope.launch {
                        bringIntoViewRequester.bringIntoView()
                    }
                }
            }
    )

    if (showPicker) {
        FlatDateTimePickerDialog(
            mode = mode,
            currentValue = value,
            onDismiss = { showPicker = false },
            onPicked = { picked ->
                commit(picked)
                showPicker = false
            }
        )
    }
}

// moved from FlatSpecRenderer.kt (FlatDateTimeMode)
internal enum class FlatDateTimeMode { DATE, TIME, DATE_TIME }

// moved from FlatSpecRenderer.kt (flatDateTimeMode)
/** Reads the declared picker mode; defaults to date-only. */
internal fun flatDateTimeMode(props: Map<String, Any?>): FlatDateTimeMode {
    val declared = (props["mode"] ?: props["variant"] ?: props["inputType"] ?: props["format"])
        ?.toString()
        ?.trim()
        ?.lowercase(Locale.US)
        .orEmpty()
    val enableDate = props["enableDate"] as? Boolean
    val enableTime = props["enableTime"] as? Boolean
    if (enableDate == true && enableTime == true) return FlatDateTimeMode.DATE_TIME
    if (enableTime == true && enableDate != true) return FlatDateTimeMode.TIME
    return when {
        declared.contains("datetime") || declared.contains("date_time") -> FlatDateTimeMode.DATE_TIME
        declared == "time" || declared.contains("time") && !declared.contains("date") ->
            FlatDateTimeMode.TIME
        else -> FlatDateTimeMode.DATE
    }
}

// moved from FlatSpecRenderer.kt (flatDateTimePlaceholder)
internal fun flatDateTimePlaceholder(mode: FlatDateTimeMode): String = when (mode) {
    FlatDateTimeMode.DATE -> "YYYY-MM-DD"
    FlatDateTimeMode.TIME -> "HH:MM"
    FlatDateTimeMode.DATE_TIME -> "YYYY-MM-DD HH:MM"
}

// moved from FlatSpecRenderer.kt (formatFlatDate)
/** ISO-8601 formatting, so a picked value round-trips through validation. */
internal fun formatFlatDate(year: Int, month: Int, day: Int): String =
    "%04d-%02d-%02d".format(year, month, day)


// moved from FlatSpecRenderer.kt (formatFlatTime)
internal fun formatFlatTime(hour: Int, minute: Int): String = "%02d:%02d".format(hour, minute)


// moved from FlatSpecRenderer.kt (flatDatePrefixOf)
/** Extracts an ISO date prefix from an existing value, if present. */
internal fun flatDatePrefixOf(value: String): String? =
    Regex("""\d{4}-\d{2}-\d{2}""").find(value)?.value


// moved from FlatSpecRenderer.kt (FlatDateTimePickerDialog)
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun FlatDateTimePickerDialog(
    mode: FlatDateTimeMode,
    currentValue: String,
    onDismiss: () -> Unit,
    onPicked: (String) -> Unit
) {
    // DATE_TIME picks the date first, then the time, composing an ISO value.
    var pendingDate by remember { mutableStateOf(flatDatePrefixOf(currentValue)) }
    val needsDate = mode != FlatDateTimeMode.TIME && pendingDate == null
    if (needsDate) {
        val dateState = rememberDatePickerState()
        DatePickerDialog(
            onDismissRequest = onDismiss,
            confirmButton = {
                TextButton(
                    onClick = {
                        val millis = dateState.selectedDateMillis
                        if (millis == null) {
                            onDismiss()
                        } else {
                            val date = Instant.ofEpochMilli(millis)
                                .atZone(ZoneOffset.UTC)
                                .toLocalDate()
                            val iso = formatFlatDate(
                                date.year,
                                date.monthValue,
                                date.dayOfMonth
                            )
                            if (mode == FlatDateTimeMode.DATE) onPicked(iso) else pendingDate = iso
                        }
                    }
                ) { Text("OK") }
            },
            dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } }
        ) {
            DatePicker(state = dateState)
        }
        return
    }

    val timeState = rememberTimePickerState()
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(
                onClick = {
                    val time = formatFlatTime(timeState.hour, timeState.minute)
                    val prefix = pendingDate
                    onPicked(if (prefix == null) time else "$prefix $time")
                }
            ) { Text("OK") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
        text = { TimePicker(state = timeState) }
    )
}
