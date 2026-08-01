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

// moved from FlatSpecRenderer.kt (looksLikeMarketHoldingsTable)
internal fun looksLikeMarketHoldingsTable(
    headers: List<String>,
    rows: List<List<String>>,
    domain: String = "generic"
): Boolean {
    if (headers.size < 3 || rows.isEmpty()) return false
    if (domain !in setOf("generic", "comparison", "finance", "portfolio", "market", "investment", "schedule")) {
        return false
    }
    val tokens = headers.map(::normalizeTableHeaderForMatch)
    val hasTicker = tokens.any { token ->
        token in setOf("ticker", "symbol", "stock", "asset", "holding", "security") ||
            token.contains("ticker") ||
            token.contains("symbol")
    }
    val hasCurrentValue = tokens.any { token ->
        (token.contains("current") && token.contains("value")) ||
            token.contains("market value") ||
            (token == "value") ||
            (token.contains("price") && !token.contains("change"))
    }
    val hasChange = tokens.any { token ->
        token.contains("change") ||
            token.contains("return") ||
            token.contains("gain") ||
            token.contains("loss") ||
            token.contains("p l")
    }
    val hasPercentColumn = tokens.any { token ->
        token.contains("pct") ||
            token.contains("percent") ||
            token.contains("percentage")
    } || headers.any { header -> header.contains('%') }
    val firstColumnIndex = marketTickerColumnIndex(headers)
    val tickerLikeRows = rows.count { row ->
        row.getOrNull(firstColumnIndex).orEmpty().trim().matches(Regex("""[A-Z][A-Z0-9.\-]{1,6}"""))
    }
    val currencyOrPercentValues = rows.flatten().count { value ->
        val token = value.trim()
        token.contains('$') ||
            token.contains('%') ||
            token.startsWith("+") ||
            token.startsWith("-")
    }
    return hasTicker &&
        hasCurrentValue &&
        (hasChange || hasPercentColumn) &&
        tickerLikeRows >= maxOf(1, rows.size / 2) &&
        currencyOrPercentValues >= rows.size
}

// moved from FlatSpecRenderer.kt (RenderMarketHoldingsTable)
@Composable
internal fun RenderMarketHoldingsTable(
    headers: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier
) {
    if (rows.isEmpty()) return
    val tickerIndex = marketTickerColumnIndex(headers)
    val valueIndex = marketValueColumnIndex(headers, tickerIndex)
    val amountIndex = marketChangeAmountColumnIndex(headers, setOf(tickerIndex, valueIndex))
    val percentIndex = marketChangePercentColumnIndex(headers, setOf(tickerIndex, valueIndex) + listOfNotNull(amountIndex))
    Card(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        shape = RoundedCornerShape(18.dp),
        colors = flatSpecCardColors(),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 10.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(7.dp)
        ) {
            rows.forEachIndexed { rowIndex, row ->
                MarketHoldingRow(
                    headers = headers,
                    row = row,
                    rowIndex = rowIndex,
                    tickerIndex = tickerIndex,
                    valueIndex = valueIndex,
                    amountIndex = amountIndex,
                    percentIndex = percentIndex
                )
            }
        }
    }
}
