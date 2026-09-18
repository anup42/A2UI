package com.samsung.genuicraft.sdk.internal.renderer.flat.legacy

import com.samsung.genuicraft.sdk.internal.renderer.*

/**
 * Legacy presentation heuristics: they infer table semantics by matching
 * **English column-header strings**.
 *
 * This is the layer DoD #4 exists to remove. Two problems: a non-English column
 * label routes to a plain table instead of the intended domain card layout, and
 * every new domain means another keyword list.
 *
 * They are isolated here, and gated by
 * [com.samsung.genuicraft.sdk.internal.pipeline.FlatSpecIngestMode.COMPATIBILITY],
 * so switching to IR-declared `domain`/`presentation` is a one-line flip rather
 * than a rewrite. The flag currently defaults to **true**: turning it off is a
 * behaviour change that moves table routing on 28 of 50 golden records and needs
 * a deliberate GenUI metric re-baseline, so it must not be flipped silently.
 */

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
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.*

// moved from FlatSpecRenderer.kt (isWeatherHeaderLabel)
internal fun isWeatherHeaderLabel(label: String): Boolean {
    if (label.isBlank()) return false
    val token = label.lowercase()
    val keywords = listOf(
        "weather",
        "temp",
        "temperature",
        "condition",
        "high",
        "low",
        "max",
        "min",
        "humidity",
        "wind",
        "rain",
        "rainy",
        "precip",
        "forecast",
        "climate",
        "sunshine",
        "sunny",
        "uv",
        "feels"
    )
    return keywords.any { keyword -> token.contains(keyword) }
}

// moved from FlatSpecRenderer.kt (isFlightHeaderLabel)
internal fun isFlightHeaderLabel(label: String): Boolean {
    if (label.isBlank()) return false
    val token = label.lowercase()
    val keywords = listOf(
        "airline",
        "carrier",
        "flight",
        "depart",
        "departure",
        "arrival",
        "arrive",
        "duration",
        "fare",
        "price",
        "cost",
        "stops",
        "layover"
    )
    return keywords.any { keyword -> token.contains(keyword) }
}

// moved from FlatSpecRenderer.kt (isStrongFlightHeaderLabel)
internal fun isStrongFlightHeaderLabel(label: String): Boolean {
    if (label.isBlank()) return false
    val token = label.lowercase()
    val keywords = listOf(
        "airline",
        "carrier",
        "flight",
        "depart",
        "departure",
        "arrival",
        "arrive",
        "takeoff",
        "landing",
        "origin",
        "destination"
    )
    return keywords.any { keyword -> token.contains(keyword) }
}

// moved from FlatSpecRenderer.kt (isRestaurantHeaderLabel)
internal fun isRestaurantHeaderLabel(label: String): Boolean {
    if (label.isBlank()) return false
    val token = normalizeTableHeaderForMatch(label)
    val keywords = listOf(
        "restaurant",
        "place",
        "name",
        "rating",
        "review",
        "price",
        "cuisine",
        "type",
        "address",
        "hours",
        "open",
        "photo",
        "image",
        "maps",
        "website",
        "direction",
        "call"
    )
    return keywords.any { keyword -> token.contains(keyword) }
}

// moved from FlatSpecRenderer.kt (isBookingEntityHeaderLabel)
internal fun isBookingEntityHeaderLabel(label: String): Boolean {
    if (label.isBlank()) return false
    val token = label.lowercase()
    val keywords = listOf(
        "hotel",
        "property",
        "provider",
        "option",
        "listing",
        "vendor",
        "airline",
        "plan",
        "package",
        "name"
    )
    return keywords.any { keyword -> token.contains(keyword) }
}

// moved from FlatSpecRenderer.kt (isBookingValueHeaderLabel)
internal fun isBookingValueHeaderLabel(label: String): Boolean {
    if (label.isBlank()) return false
    val token = label.lowercase()
    val keywords = listOf(
        "price",
        "cost",
        "fare",
        "rate",
        "night",
        "duration",
        "room",
        "amenity",
        "wifi",
        "rating",
        "review",
        "book",
        "reserve",
        "deal",
        "url",
        "link"
    )
    return keywords.any { keyword -> token.contains(keyword) }
}

// moved from FlatSpecRenderer.kt (isScheduleHeaderLabel)
internal fun isScheduleHeaderLabel(label: String): Boolean {
    if (label.isBlank()) return false
    val token = label.lowercase()
    val timeKeywords = listOf("time", "date", "day", "slot", "start", "end")
    val eventKeywords = listOf("event", "activity", "agenda", "session", "task", "title", "stop", "location")
    return timeKeywords.any { token.contains(it) } || eventKeywords.any { token.contains(it) }
}

// moved from FlatSpecRenderer.kt (isStatusHeaderLabel)
internal fun isStatusHeaderLabel(label: String): Boolean {
    if (label.isBlank()) return false
    val token = label.lowercase()
    val keywords = listOf(
        "status",
        "state",
        "stage",
        "progress",
        "eta",
        "updated",
        "resolved",
        "tracking",
        "phase"
    )
    return keywords.any { keyword -> token.contains(keyword) }
}

// moved from FlatSpecRenderer.kt (isComparisonFeatureHeader)
internal fun isComparisonFeatureHeader(label: String): Boolean {
    if (label.isBlank()) return false
    val token = label.lowercase()
    val keywords = listOf(
        "feature",
        "metric",
        "criteria",
        "criterion",
        "attribute",
        "spec",
        "dimension",
        "parameter",
        "category",
        "aspect",
        "factor",
        "topic"
    )
    return keywords.any { keyword -> token.contains(keyword) }
}

// moved from FlatSpecRenderer.kt (isComparisonEntityHeader)
internal fun isComparisonEntityHeader(label: String): Boolean {
    if (label.isBlank()) return false
    val token = normalizeTableHeaderForMatch(label)
    val keywords = listOf(
        "model",
        "product",
        "item",
        "option",
        "name",
        "plan",
        "package",
        "provider",
        "service",
        "tool",
        "device",
        "site",
        "destination",
        "hotel",
        "route",
        "airline",
        "project",
        "task",
        "drink",
        "recipe"
    )
    return keywords.any { keyword -> token.contains(keyword) }
}

// moved from FlatSpecRenderer.kt (isNewsTableHeaderSet)
internal fun isNewsTableHeaderSet(headers: List<String>, domain: String = "generic"): Boolean {
    if (domain == "news") return true
    val tokens = headers.map(::normalizeTableHeaderForMatch)
    val hasTitle = tokens.any { token ->
        token in setOf("article", "headline", "title", "story", "news") ||
            token.contains("headline") ||
            token.contains("article")
    }
    val hasSource = tokens.any { token ->
        token == "source" ||
            token == "publisher" ||
            token == "publication" ||
            token == "source name" ||
            token == "source url"
    }
    val hasTime = tokens.any { token ->
        token.contains("published") ||
            token.contains("pub date") ||
            token.contains("time") ||
            token.contains("date")
    }
    val hasArticleUrl = tokens.any { token ->
        token.contains("article url") ||
            token == "url" ||
            token == "link" ||
            token.contains("read")
    }
    val hasNewsMedia = tokens.any { token ->
        token.contains("image url") ||
            token.contains("source icon") ||
            token.contains("thumbnail")
    }
    return hasTitle && hasSource && (hasTime || hasArticleUrl || hasNewsMedia)
}

// moved from FlatSpecRenderer.kt (inferTableDomainFromHeaders)
internal fun inferTableDomainFromHeaders(headers: List<String>): String {
    val weatherSignals = headers.count(::isWeatherHeaderLabel)
    val weatherPeriodSignals = headers.count(::isWeatherPeriodHeaderLabel)
    val flightSignals = headers.count(::isFlightHeaderLabel)
    val strongFlightSignals = headers.count(::isStrongFlightHeaderLabel)
    val restaurantSignals = headers.count(::isRestaurantHeaderLabel)
    val bookingEntitySignals = headers.count(::isBookingEntityHeaderLabel)
    val bookingValueSignals = headers.count(::isBookingValueHeaderLabel)
    val scheduleSignals = headers.count(::isScheduleHeaderLabel)
    val statusSignals = headers.count(::isStatusHeaderLabel)
    val featureLike = isComparisonFeatureHeader(headers.firstOrNull().orEmpty())
    val entityLike = isComparisonEntityHeader(headers.firstOrNull().orEmpty())
    return when {
        isPlaylistTableHeaderSet(headers) -> "playlist"
        isFormulaVariableHeaderSet(headers) || isCalculationBreakdownHeaderSet(headers) -> "formula"
        isNewsTableHeaderSet(headers) -> "news"
        weatherSignals >= 2 -> "weather"
        weatherSignals >= 1 && weatherPeriodSignals >= 1 -> "weather"
        strongFlightSignals >= 1 && flightSignals >= 2 -> "flight"
        restaurantSignals >= 2 -> "restaurants"
        bookingEntitySignals >= 1 && bookingValueSignals >= 2 -> "booking"
        scheduleSignals >= 2 && statusSignals >= 1 -> "status"
        scheduleSignals >= 2 -> "schedule"
        featureLike && headers.size >= 3 -> "comparison"
        entityLike && headers.size >= 4 -> "comparison"
        else -> "generic"
    }
}

// moved from FlatSpecRenderer.kt (isWeatherPeriodHeaderLabel)
internal fun isWeatherPeriodHeaderLabel(label: String): Boolean {
    val token = normalizeTableHeaderForMatch(label)
    return token in setOf("day", "date", "time", "hour", "period")
}
