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

// moved from FlatSpecRenderer.kt (PlaylistTrackRow)
internal data class PlaylistTrackRow(
    val number: String,
    val title: String,
    val artist: String?,
    val chips: List<String>,
    val sourceRow: List<String>
)


// moved from FlatSpecRenderer.kt (isPlaylistTableHeaderSet)
internal fun isPlaylistTableHeaderSet(headers: List<String>, domain: String = "generic"): Boolean {
    if (domain == "playlist") return true
    val tokens = headers.map(::normalizeTableHeaderForMatch)
    val hasTrackNumber = headers.any { it.trim() == "#" } ||
        tokens.any { token ->
            token in setOf("no", "number", "track number", "track no", "tracknumber", "track")
        }
    val hasTrackTitle = tokens.any { token ->
        token == "track" ||
            token == "song" ||
            token == "title" ||
            token == "songtitle" ||
            token == "tracktitle" ||
            token.contains("track title") ||
            token.contains("track name") ||
            token.contains("song title") ||
            token.contains("song name")
    }
    val hasArtist = tokens.any { token ->
        token == "artist" ||
            token.contains("artist") ||
            token.contains("performer") ||
            token.contains("band")
    }
    val hasMusicMetadata = tokens.any { token ->
        token.contains("album") ||
            token.contains("genre") ||
            token.contains("mood") ||
            token.contains("tempo") ||
            token.contains("phase") ||
            token.contains("set")
    }
    return hasTrackTitle && (hasTrackNumber || hasArtist || hasMusicMetadata)
}

// moved from FlatSpecRenderer.kt (RenderPlaylistTableRows)
@Composable
internal fun RenderPlaylistTableRows(
    headers: List<String>,
    rows: List<List<String>>,
    props: Map<String, Any?>,
    modifier: Modifier = Modifier,
    landscape: Boolean
) {
    if (rows.isEmpty()) return
    val tracks = buildPlaylistTrackRows(headers, rows)
    val trackCount = tracks.size
    val title = props["title"]?.toString()?.trim()?.takeIf { it.isNotBlank() } ?: "Playlist"
    val subtitle = props["subtitle"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
        ?: "$trackCount ${if (trackCount == 1) "track" else "tracks"}"
    val tags = playlistStringListProp(props["mood"]) + playlistStringListProp(props["genre"])
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = tableAccessibilitySummary(headers, rows)
            },
        shape = RoundedCornerShape(26.dp),
        color = Color.Transparent,
        tonalElevation = 0.dp,
        shadowElevation = 0.dp
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.linearGradient(
                        colors = listOf(
                            Color(0xFF070712),
                            Color(0xFF172044),
                            Color(0xFF082F35)
                        )
                    )
                )
                .padding(14.dp)
        ) {
            if (landscape) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(18.dp),
                    verticalAlignment = Alignment.Top
                ) {
                    PlaylistHeroBlock(
                        title = title,
                        subtitle = subtitle,
                        tags = tags,
                        trackCount = trackCount,
                        landscape = true,
                        modifier = Modifier.widthIn(min = 210.dp, max = 260.dp)
                    )
                    PlaylistTrackList(
                        headers = headers,
                        tracks = tracks,
                        dense = true,
                        modifier = Modifier
                            .weight(1f)
                            .heightIn(max = 430.dp)
                            .verticalScroll(rememberScrollState())
                    )
                }
            } else {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    PlaylistHeroBlock(
                        title = title,
                        subtitle = subtitle,
                        tags = tags,
                        trackCount = trackCount,
                        landscape = false,
                        modifier = Modifier.fillMaxWidth()
                    )
                    PlaylistTrackList(
                        headers = headers,
                        tracks = tracks,
                        dense = false,
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }
        }
    }
}
