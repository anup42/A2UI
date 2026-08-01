package com.samsung.genuicraft.renderer.flat.runtime

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
import com.samsung.genuicraft.renderer.flat.expr.*
import com.samsung.genuicraft.renderer.flat.parse.*
import com.samsung.genuicraft.renderer.flat.model.*

// moved from FlatSpecRenderer.kt (FlatActionRuntime)
internal data class FlatActionExecution(
    val action: String,
    val params: Map<String, Any?>,
    val success: Boolean,
    val stateMutations: List<FlatStateMutation> = emptyList()
)

internal data class FlatActionExecutionResult(
    val actions: List<FlatActionExecution>
) {
    val attempted: Int get() = actions.size
    val succeeded: Int get() = actions.count { it.success }
    val stateMutations: List<FlatStateMutation> get() = actions.flatMap(FlatActionExecution::stateMutations)
}

internal data class FlatWatchTrigger(
    val elementId: String,
    val watchKey: String,
    val statePath: String,
    val value: Any?,
    val actionBinding: Any?
) {
    val fingerprint: String = "$watchKey|$statePath|${canonicalWatchValue(value)}"
}

/** One mutation cascade shares a global action budget and visited watch fingerprints. */
internal class FlatWatchCascadeTransaction(
    private val maxActions: Int,
    private val idleBoundaryNanos: Long = 100_000_000L
) {
    private val visited = mutableSetOf<String>()
    private var lastEmissionNanos: Long? = null
    private var terminationReported = false
    var remainingActions: Int = maxActions
        private set

    fun beginEmission(nowNanos: Long = System.nanoTime()) {
        val previous = lastEmissionNanos
        if (previous != null && nowNanos - previous > idleBoundaryNanos) reset()
        lastEmissionNanos = nowNanos
    }

    fun enter(trigger: FlatWatchTrigger): Boolean =
        remainingActions > 0 && visited.add(trigger.fingerprint)

    fun consume(attemptedActions: Int) {
        remainingActions = (remainingActions - attemptedActions.coerceAtLeast(1)).coerceAtLeast(0)
    }

    fun claimTerminationDiagnostic(): Boolean {
        if (terminationReported) return false
        terminationReported = true
        return true
    }

    fun reset() {
        visited.clear()
        remainingActions = maxActions
        lastEmissionNanos = null
        terminationReported = false
    }
}

private fun canonicalWatchValue(value: Any?): String = when (value) {
    null -> "null"
    is Map<*, *> -> value.entries
        .sortedBy { it.key?.toString().orEmpty() }
        .joinToString(prefix = "{", postfix = "}") { (key, item) ->
            "${key?.toString().orEmpty()}:${canonicalWatchValue(item)}"
        }
    is List<*> -> value.joinToString(prefix = "[", postfix = "]") { canonicalWatchValue(it) }
    else -> "${value::class.simpleName}:$value"
}

internal object FlatActionRuntime {

    /** Compatibility facade for the existing composable callback shape. */
    fun execute(
        actionCandidate: Any?,
        stateStore: MutableMap<String, Any?>,
        repeatScope: RepeatScope?,
        computedFunctions: Map<String, FlatComputedFunction>,
        onOpenUrl: (String) -> Unit,
        onDiagnostic: (String) -> Unit = {},
        elements: Map<String, FlatElement> = emptyMap()
    ): Int = executeDetailed(
        actionCandidate = actionCandidate,
        stateStore = stateStore,
        repeatScope = repeatScope,
        computedFunctions = computedFunctions,
        onOpenUrl = onOpenUrl,
        onDiagnostic = { diagnostic -> onDiagnostic(diagnostic.message) },
        elements = elements
    ).attempted

    fun executeDetailed(
        actionCandidate: Any?,
        stateStore: MutableMap<String, Any?>,
        repeatScope: RepeatScope?,
        computedFunctions: Map<String, FlatComputedFunction>,
        onOpenUrl: (String) -> Unit,
        onDiagnostic: (FlatDiagnostic) -> Unit = {},
        elements: Map<String, FlatElement> = emptyMap(),
        elementId: String? = null
    ): FlatActionExecutionResult {
        val bindings = toActionBindings(actionCandidate)
        val executions = mutableListOf<FlatActionExecution>()
        bindings.forEach { binding ->
            val actionName = binding["action"]?.toString()?.trim()?.lowercase().orEmpty()
            if (actionName.isBlank()) return@forEach

            val rawParams = toStringKeyMap(binding["params"]) ?: emptyMap()
            val resolvedParams = rawParams.mapValues { (_, value) ->
                FlatExprResolver.resolve(value, stateStore, repeatScope, computedFunctions)
            }
            val mutations = mutableListOf<FlatStateMutation>()
            var success = false
            val displayName = canonicalActionName(actionName)

            fun invalid(message: String) {
                onDiagnostic(
                    FlatDiagnostic(
                        code = FlatDiagnostic.Code.INVALID_ACTION,
                        severity = FlatDiagnostic.Severity.WARNING,
                        message = message,
                        elementId = elementId,
                        details = mapOf("action" to displayName)
                    )
                )
            }

            when (actionName) {
                "openurl" -> {
                    val url = firstNonBlankString(resolvedParams, "url", "href", "link", "targetUrl")
                    val safeUrl = SafeContentPolicy.sanitizeActionUrl(url)
                    if (!safeUrl.isNullOrBlank()) {
                        onOpenUrl(safeUrl)
                        success = true
                    } else {
                        invalid("Action '$displayName' requires a safe https or tel URL.")
                    }
                }

                "setstate" -> {
                    val path = resolveStatePathParam(
                        rawParams,
                        stateStore,
                        repeatScope,
                        computedFunctions,
                        "statePath",
                        "path"
                    )
                    if (path.isNotBlank() && rawParams.containsKey("value")) {
                        val before = deepCopyValue(FlatSpecParser.getAtPath(stateStore, path))
                        val after = resolvedParams["value"]
                        FlatSpecParser.setAtPath(stateStore, path, after)
                        mutations += FlatStateMutation(path, FlatStateMutation.Operation.SET, before, deepCopyValue(after))
                        success = true
                    } else {
                        invalid("Action '$displayName' requires statePath/path and an explicit value.")
                    }
                }

                "pushstate" -> {
                    val path = resolveStatePathParam(
                        rawParams,
                        stateStore,
                        repeatScope,
                        computedFunctions,
                        "statePath",
                        "path"
                    )
                    if (path.isNotBlank() && rawParams.containsKey("value")) {
                        val before = deepCopyValue(FlatSpecParser.getAtPath(stateStore, path))
                        val current = (before as? List<*>)?.toMutableList()
                            ?: mutableListOf()
                        val generatedId = UUID.randomUUID().toString()
                        val value = replaceGeneratedIdToken(resolvedParams["value"], generatedId)
                        current.add(value)
                        val after = current.toList()
                        FlatSpecParser.setAtPath(stateStore, path, after)
                        mutations += FlatStateMutation(path, FlatStateMutation.Operation.PUSH, before, deepCopyValue(after))
                        val clearStatePath = resolveStatePathParam(
                            rawParams,
                            stateStore,
                            repeatScope,
                            computedFunctions,
                            "clearStatePath"
                        )
                        if (clearStatePath.isNotBlank()) {
                            val clearBefore = deepCopyValue(FlatSpecParser.getAtPath(stateStore, clearStatePath))
                            FlatSpecParser.setAtPath(stateStore, clearStatePath, null)
                            mutations += FlatStateMutation(
                                clearStatePath,
                                FlatStateMutation.Operation.SET,
                                clearBefore,
                                null
                            )
                        }
                        success = true
                    } else {
                        invalid("Action '$displayName' requires statePath/path and an explicit value.")
                    }
                }

                "removestate" -> {
                    val path = resolveStatePathParam(
                        rawParams,
                        stateStore,
                        repeatScope,
                        computedFunctions,
                        "statePath",
                        "path"
                    )
                    val index = toInt(resolvedParams["index"])
                    if (path.isNotBlank() && index != null) {
                        val current = (FlatSpecParser.getAtPath(stateStore, path) as? List<*>)?.toMutableList()
                        if (current != null && index in current.indices) {
                            val before = deepCopyValue(current)
                            current.removeAt(index)
                            val after = current.toList()
                            FlatSpecParser.setAtPath(stateStore, path, after)
                            mutations += FlatStateMutation(path, FlatStateMutation.Operation.REMOVE, before, deepCopyValue(after))
                            success = true
                        } else {
                            invalid("Action '$displayName' index is outside the target list.")
                        }
                    } else {
                        invalid("Action '$displayName' requires statePath/path and a non-negative integer index.")
                    }
                }

                "validateform" -> {
                    val targetPath = resolveStatePathParam(
                        rawParams,
                        stateStore,
                        repeatScope,
                        computedFunctions,
                        "resultStatePath",
                        "statePath",
                        "path"
                    )
                        .ifBlank { "/formValidation" }
                    val validationResult = FlatFormValidation.validate(elements, stateStore)
                    val before = deepCopyValue(FlatSpecParser.getAtPath(stateStore, targetPath))
                    FlatSpecParser.setAtPath(stateStore, targetPath, validationResult)
                    mutations += FlatStateMutation(
                        targetPath,
                        FlatStateMutation.Operation.VALIDATE,
                        before,
                        deepCopyValue(validationResult)
                    )
                    success = true
                }

                else -> onDiagnostic(
                    FlatDiagnostic(
                        code = FlatDiagnostic.Code.UNSUPPORTED_ACTION,
                        severity = FlatDiagnostic.Severity.WARNING,
                        message = "Unsupported action '$actionName'. Supported: openUrl, setState, pushState, removeState, validateForm.",
                        elementId = elementId,
                        details = mapOf("action" to actionName)
                    )
                )
            }
            executions += FlatActionExecution(
                action = displayName,
                params = resolvedParams,
                success = success,
                stateMutations = mutations
            )
        }
        return FlatActionExecutionResult(executions)
    }

    private fun canonicalActionName(actionName: String): String = when (actionName) {
        "openurl" -> "openUrl"
        "setstate" -> "setState"
        "pushstate" -> "pushState"
        "removestate" -> "removeState"
        "validateform" -> "validateForm"
        else -> actionName
    }

    private fun toActionBindings(actionCandidate: Any?): List<Map<String, Any?>> {
        return when (actionCandidate) {
            is List<*> -> actionCandidate.mapNotNull { entry ->
                val binding = toStringKeyMap(entry) ?: return@mapNotNull null
                binding.takeIf { it["action"] is String }
            }

            else -> {
                val binding = toStringKeyMap(actionCandidate)
                if (binding != null && binding["action"] is String) listOf(binding) else emptyList()
            }
        }
    }

    private fun resolveStatePathParam(
        rawParams: Map<String, Any?>,
        state: Map<String, Any?>,
        repeatScope: RepeatScope?,
        computedFunctions: Map<String, FlatComputedFunction>,
        vararg keys: String
    ): String {
        keys.forEach { key ->
            val rawValue = rawParams[key] ?: return@forEach
            val rawMap = toStringKeyMap(rawValue)
            val resolved = if (rawMap != null && rawMap.containsKey("\$item")) {
                resolveBindItemPath(rawMap["\$item"]?.toString(), repeatScope)
            } else {
                FlatExprResolver.resolve(rawValue, state, repeatScope, computedFunctions)?.toString()
            }
            val normalized = resolved?.trim().orEmpty()
            if (normalized.isNotBlank()) {
                return normalized
            }
        }
        return ""
    }

    private fun firstNonBlankString(map: Map<String, Any?>, vararg keys: String): String {
        keys.forEach { key ->
            val value = map[key]?.toString()?.trim().orEmpty()
            if (value.isNotBlank()) {
                return value
            }
        }
        return ""
    }

    private fun toInt(value: Any?): Int? {
        return when (value) {
            is Number -> value.toInt()
            is String -> value.toIntOrNull()
            else -> null
        }
    }

    private fun replaceGeneratedIdToken(value: Any?, generatedId: String): Any? {
        return when (value) {
            "\$id" -> generatedId
            is Map<*, *> -> value.entries.associate { (k, v) ->
                k.toString() to replaceGeneratedIdToken(v, generatedId)
            }

            is List<*> -> value.map { replaceGeneratedIdToken(it, generatedId) }
            else -> value
        }
    }
}

// moved from FlatSpecRenderer.kt (FlatWatchRuntime)
internal class FlatWatchRuntime(elements: Map<String, FlatElement>) {

    private val entries: List<WatchEntry> = elements.flatMap { (elementId, element) ->
        element.watch.orEmpty().map { (statePath, binding) ->
            WatchEntry(
                elementId = elementId,
                key = "$elementId::$statePath",
                statePath = normalizePointer(statePath),
                actionBinding = binding
            )
        }
    }
    private val lastValues = mutableMapOf<String, Any?>()
    private var initialized = false

    fun collectTriggered(state: Map<String, Any?>): List<Any?> {
        return collectTriggeredEntries(state).map(FlatWatchTrigger::actionBinding)
    }

    fun collectTriggeredEntries(state: Map<String, Any?>): List<FlatWatchTrigger> {
        if (entries.isEmpty()) return emptyList()
        val triggered = mutableListOf<FlatWatchTrigger>()
        entries.forEach { entry ->
            val currentValue = deepCopyValue(FlatSpecParser.getAtPath(state, entry.statePath))
            val previousValue = lastValues[entry.key]
            if (initialized && !deepEquals(previousValue, currentValue)) {
                triggered += FlatWatchTrigger(
                    elementId = entry.elementId,
                    watchKey = entry.key,
                    statePath = entry.statePath,
                    value = currentValue,
                    actionBinding = entry.actionBinding
                )
            }
            lastValues[entry.key] = currentValue
        }
        if (!initialized) {
            initialized = true
        }
        return triggered
    }
}
