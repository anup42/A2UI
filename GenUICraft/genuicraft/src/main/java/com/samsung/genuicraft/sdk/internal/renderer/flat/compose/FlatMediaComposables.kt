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

// moved from FlatSpecRenderer.kt (RenderImage)
@Composable
internal fun RenderImage(
    props: Map<String, Any?>,
    onOpenUrl: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    val resolveAssetUrl = LocalFlatSpecAssetResolver.current
    val rawUrl = resolveMediaUrlCandidate(props, IMAGE_PROP_KEYS)
    val candidateUrls = buildList {
        add(rawUrl)
        addAll(extractMediaUrlTokens(props["fallbackUrl"]))
        addAll(extractMediaUrlTokens(props["fallbackUrls"]))
        addAll(extractMediaUrlTokens(props["fallback"]))
        addAll(extractMediaUrlTokens(props["alternates"]))
        addAll(extractMediaUrlTokens(props["photoUrls"]))
        addAll(extractMediaUrlTokens(props["photos"]))
    }
        .mapNotNull { candidate ->
            SafeContentPolicy.sanitizeMediaUrl(candidate, SafeContentPolicy.MediaKind.IMAGE)
                ?.let(resolveAssetUrl)
                ?.let(::resolveCoilMediaModel)
                ?.takeIf { it.isNotBlank() }
        }
        .distinct()
    val url = candidateUrls.firstOrNull() ?: return
    val resolvedUrl = url
    val contentScale = resolveImageScale(props)
    val context = LocalContext.current
    val imageLoader = rememberFlatImageLoader()
    val actionUrl = SafeContentPolicy.sanitizeActionUrl(extractMediaUrlToken(props["actionUrl"])).orEmpty()
    val widthDp = asDp(props["width"])
    val heightDp = asDp(props["height"])
    val aspectRatio = parseAspectRatio(props["aspectRatio"]) ?: if (heightDp == null) 16f / 9f else null
    var imageModifier = modifier
    imageModifier = if (widthDp != null) imageModifier.width(widthDp) else imageModifier.fillMaxWidth()
    if (heightDp != null) {
        imageModifier = imageModifier.height(heightDp)
    }
    if (aspectRatio != null) {
        imageModifier = imageModifier.aspectRatio(aspectRatio)
    }
    imageModifier = imageModifier.clip(RoundedCornerShape(12.dp))
    val clickableModifier = if (actionUrl.isNotBlank()) {
        imageModifier.clickable(
            role = Role.Button,
            onClickLabel = accessibilityString(props, "onClickLabel", "actionLabel") ?: "Open image"
        ) { onOpenUrl(actionUrl) }
    } else {
        imageModifier
    }
    val generatedRestaurantVisual = parseGeneratedRestaurantVisual(resolvedUrl)
        ?: parseGeneratedRestaurantVisual(url)
    if (generatedRestaurantVisual != null) {
        RenderGeneratedRestaurantVisual(
            visual = generatedRestaurantVisual,
            modifier = clickableModifier,
            imageLabel = accessibilityLabel(props, "${generatedRestaurantVisual.title} restaurant image")
        )
        return
    }
    var failed by remember(candidateUrls) { mutableStateOf(false) }
    var activeIndex by remember(candidateUrls) { mutableIntStateOf(0) }
    val activeUrl = candidateUrls.getOrElse(activeIndex) { url }
    val imageLabel = accessibilityLabel(props, props["alt"]?.toString() ?: "Image")
    val placeholderBg = MaterialTheme.colorScheme.surfaceContainerHighest
    val iconLikeImage = isIconLikeMediaUrl(resolvedUrl) || isIconLikeMediaUrl(url)

    Box(
        modifier = clickableModifier
            .background(placeholderBg)
            .accessibilitySemantics(
                props = props,
                fallbackLabel = imageLabel,
                semanticRole = Role.Button.takeIf { actionUrl.isNotBlank() },
                state = "Image unavailable".takeIf { failed },
                mergeDescendants = true
        ),
        contentAlignment = Alignment.Center
    ) {
        if (iconLikeImage && !failed) {
            AsyncImage(
                model = ImageRequest.Builder(context)
                    .data(activeUrl)
                    .applyFlatSpecRemoteImageHeaders(activeUrl)
                    .crossfade(true)
                    .allowHardware(false)
                    .build(),
                imageLoader = imageLoader,
                contentDescription = null,
                contentScale = ContentScale.Fit,
                colorFilter = ColorFilter.tint(MaterialTheme.colorScheme.onSurfaceVariant),
                modifier = Modifier.size(42.dp),
                onSuccess = { failed = false },
                onError = {
                    val nextIndex = activeIndex + 1
                    if (nextIndex < candidateUrls.size) {
                        activeIndex = nextIndex
                        failed = false
                        Log.w(
                            FLAT_SPEC_RENDERER_TAG,
                            "RenderImage icon-like media failed for ${imageLogLabel(activeUrl)}; retrying candidate ${nextIndex + 1}/${candidateUrls.size}."
                        )
                    } else {
                        failed = true
                        Log.w(
                            FLAT_SPEC_RENDERER_TAG,
                            "RenderImage received icon-like media URL from ${imageLogLabel(activeUrl)}; compact icon load failed."
                        )
                    }
                }
            )
        } else if (failed) {
            Icon(
                imageVector = Icons.Filled.Image,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(24.dp)
            )
        } else {
            AsyncImage(
                model = ImageRequest.Builder(context)
                    .data(activeUrl)
                    .applyFlatSpecRemoteImageHeaders(activeUrl)
                    .crossfade(true)
                    .allowHardware(false)
                    .build(),
                imageLoader = imageLoader,
                contentDescription = null,
                contentScale = contentScale,
                modifier = Modifier.fillMaxSize(),
                onSuccess = { failed = false },
                onError = {
                    val failedUrl = activeUrl
                    val nextIndex = activeIndex + 1
                    if (nextIndex < candidateUrls.size) {
                        activeIndex = nextIndex
                        failed = false
                        Log.w(
                            FLAT_SPEC_RENDERER_TAG,
                            "RenderImage failed for ${imageLogLabel(failedUrl)}; retrying candidate ${nextIndex + 1}/${candidateUrls.size}."
                        )
                    } else {
                        failed = true
                        Log.w(
                            FLAT_SPEC_RENDERER_TAG,
                            "RenderImage failed for ${imageLogLabel(failedUrl)} with ${candidateUrls.size} candidate(s)."
                        )
                    }
                }
            )
        }
    }
}

// moved from FlatSpecRenderer.kt (RenderCatalogIcon)
@Composable
private fun RenderCatalogIcon(
    vector: ImageVector,
    props: Map<String, Any?>,
    modifier: Modifier = Modifier
) {
    val iconSize = resolveIconSize(props)
    val basePadding = asFlatSpacingDp(props["padding"]) ?: asFlatSpacingDp(props["iconPadding"]) ?: 4.dp
    val horizontalPadding =
        asFlatSpacingDp(props["paddingHorizontal"]) ?: asFlatSpacingDp(props["iconPaddingHorizontal"]) ?: basePadding
    val verticalPadding =
        asFlatSpacingDp(props["paddingVertical"]) ?: asFlatSpacingDp(props["iconPaddingVertical"]) ?: basePadding
    Icon(
        imageVector = vector,
        // Vectors tint to the theme, which a rasterized remote SVG cannot do.
        tint = MaterialTheme.colorScheme.onSurfaceVariant,
        contentDescription = if (isAccessibilityDecorative(props)) null else accessibilityLabel(props),
        modifier = modifier
            .padding(horizontal = horizontalPadding, vertical = verticalPadding)
            .size(iconSize)
    )
}

// moved from FlatSpecRenderer.kt (RenderIcon)
@Composable
internal fun RenderIcon(
    props: Map<String, Any?>,
    modifier: Modifier = Modifier
) {
    val resolveAssetUrl = LocalFlatSpecAssetResolver.current
    val rawUrl = resolveMediaUrlCandidate(props, ICON_PROP_KEYS)

    // A bundled vector for an IR-declared name comes first. Previously a bare
    // name like `calendar_today` failed the URL sanitizer and the element
    // rendered nothing; it also means icons still render with no network.
    FlatIconCatalog.vectorFor(rawUrl)?.let { vector ->
        RenderCatalogIcon(vector, props, modifier)
        return
    }

    val safeRawUrl = SafeContentPolicy.sanitizeMediaUrl(rawUrl, SafeContentPolicy.MediaKind.ICON) ?: return
    val url = resolveCoilMediaModel(resolveAssetUrl(safeRawUrl))
    if (url.isBlank()) return
    val context = LocalContext.current
    val imageLoader = rememberFlatImageLoader()
    val iconSize = resolveIconSize(props)
    val basePadding = asFlatSpacingDp(props["padding"]) ?: asFlatSpacingDp(props["iconPadding"]) ?: 4.dp
    val horizontalPadding =
        asFlatSpacingDp(props["paddingHorizontal"]) ?: asFlatSpacingDp(props["iconPaddingHorizontal"]) ?: basePadding
    val verticalPadding =
        asFlatSpacingDp(props["paddingVertical"]) ?: asFlatSpacingDp(props["iconPaddingVertical"]) ?: basePadding
    val iconModifier = modifier
        .padding(horizontal = horizontalPadding, vertical = verticalPadding)
        .size(iconSize)
    var failed by remember(url) { mutableStateOf(false) }
    val iconDescription = if (isAccessibilityDecorative(props)) {
        null
    } else {
        accessibilityLabel(props)
    }
    if (failed) {
        Icon(
            imageVector = Icons.Filled.Image,
            contentDescription = iconDescription?.let { "$it unavailable" },
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = iconModifier
        )
    } else {
        AsyncImage(
            model = ImageRequest.Builder(context)
                .data(url)
                .applyFlatSpecRemoteImageHeaders(url)
                .crossfade(false)
                .allowHardware(false)
                .build(),
            imageLoader = imageLoader,
            contentDescription = iconDescription,
            contentScale = ContentScale.Fit,
            colorFilter = ColorFilter.tint(MaterialTheme.colorScheme.onSurface),
            modifier = iconModifier,
            onSuccess = { failed = false },
            onError = { failed = true }
        )
    }
}

// moved from FlatSpecRenderer.kt (RenderVideo)
@Composable
internal fun RenderVideo(
    props: Map<String, Any?>,
    onOpenUrl: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    val url = SafeContentPolicy.sanitizeMediaUrl(
        resolveMediaUrlCandidate(props, MEDIA_PROP_KEYS),
        SafeContentPolicy.MediaKind.VIDEO
    ).orEmpty()
    if (url.isBlank()) return
    val label = accessibilityLabel(props, "Video")
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
            .clip(RoundedCornerShape(8.dp))
            .clickable(
                role = Role.Button,
                onClickLabel = accessibilityString(props, "onClickLabel", "actionLabel") ?: "Open video"
            ) { onOpenUrl(url) }
            .accessibilitySemantics(
                props = props,
                fallbackLabel = label,
                semanticRole = Role.Button,
                mergeDescendants = true
            ),
        color = MaterialTheme.colorScheme.surfaceVariant
    ) {
        // Inline playback is deliberately out of scope: a video frame is not
        // reproducible, which would make DatasetRenderCaptureActivity's
        // screenshots non-deterministic and corrupt the GenUI metric. A poster
        // plus an explicit play affordance conveys "this is media" without that
        // cost.
        RenderMediaPosterTile(
            props = props,
            fallbackTitle = props["title"]?.toString()?.takeIf { it.isNotBlank() } ?: "Video",
            subtitle = flatMediaDurationLabel(props)
        )
    }
}

// moved from FlatSpecRenderer.kt (RenderMediaPosterTile)
/**
 * Poster tile shared by [RenderVideo] and [RenderAudioPlayer].
 *
 * Replaces a bare grey box whose only content was the literal text "Video" or
 * the description. Falls back to that layout when no poster is declared, so a
 * spec without poster art is unchanged.
 */
@Composable
private fun RenderMediaPosterTile(
    props: Map<String, Any?>,
    fallbackTitle: String,
    subtitle: String?
) {
    val resolveAssetUrl = LocalFlatSpecAssetResolver.current
    val posterRaw = firstNonBlankStringProp(props, "poster", "posterUrl", "thumbnail", "thumbnailUrl", "image")
    val posterUrl = posterRaw
        ?.let { SafeContentPolicy.sanitizeMediaUrl(it, SafeContentPolicy.MediaKind.IMAGE) }
        ?.let { resolveCoilMediaModel(resolveAssetUrl(it)) }
        ?.takeIf { it.isNotBlank() }
    val imageLoader = rememberFlatImageLoader()
    val context = LocalContext.current
    var posterFailed by remember(posterUrl) { mutableStateOf(false) }

    Box(modifier = Modifier.fillMaxWidth()) {
        if (posterUrl != null && !posterFailed) {
            AsyncImage(
                model = ImageRequest.Builder(context)
                    .data(posterUrl)
                    .applyFlatSpecRemoteImageHeaders(posterUrl)
                    .crossfade(false)
                    .build(),
                imageLoader = imageLoader,
                contentDescription = null,
                contentScale = ContentScale.Crop,
                onError = { posterFailed = true },
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 140.dp, max = 200.dp)
            )
        }
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Icon(
                imageVector = Icons.Filled.PlayCircle,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(32.dp)
            )
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = fallbackTitle,
                    style = MaterialTheme.typography.bodyMedium,
                    maxLines = 2
                )
                if (!subtitle.isNullOrBlank()) {
                    Text(
                        text = subtitle,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

// moved from FlatSpecRenderer.kt (RenderAudioPlayer)
@Composable
internal fun RenderAudioPlayer(
    props: Map<String, Any?>,
    onOpenUrl: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    val url = SafeContentPolicy.sanitizeMediaUrl(
        resolveMediaUrlCandidate(props, MEDIA_PROP_KEYS),
        SafeContentPolicy.MediaKind.AUDIO
    ).orEmpty()
    if (url.isBlank()) return
    val description = props["description"]?.toString().orEmpty().ifBlank { "Audio" }
    val label = accessibilityLabel(props, description)
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
            .clip(RoundedCornerShape(8.dp))
            .clickable(
                role = Role.Button,
                onClickLabel = accessibilityString(props, "onClickLabel", "actionLabel") ?: "Open audio"
            ) { onOpenUrl(url) }
            .accessibilitySemantics(
                props = props,
                fallbackLabel = label,
                semanticRole = Role.Button,
                mergeDescendants = true
            ),
        color = MaterialTheme.colorScheme.surfaceVariant
    ) {
        RenderMediaPosterTile(
            props = props,
            fallbackTitle = description,
            subtitle = flatMediaDurationLabel(props)
        )
    }
}
