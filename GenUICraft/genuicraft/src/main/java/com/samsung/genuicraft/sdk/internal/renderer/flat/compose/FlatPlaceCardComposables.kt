package com.samsung.genuicraft.sdk.internal.renderer.flat.compose

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.OpenInNew
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.Directions
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Language
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.samsung.genuicraft.sdk.internal.renderer.LocalFlatInteractionEnabled
import com.samsung.genuicraft.sdk.internal.renderer.LocalFlatSpecAssetResolver
import com.samsung.genuicraft.sdk.internal.renderer.applyFlatSpecRemoteImageHeaders
import com.samsung.genuicraft.sdk.internal.security.SafeContentPolicy
import com.samsung.genuicraft.sdk.internal.theme.genUiCardBorderColor

internal data class PlaceCardPhoto(val url: String, val title: String, val onClickUrl: String?)
internal data class PlaceCardAction(val label: String, val url: String)
internal data class PlaceCardModel(
    val title: String,
    val rating: String,
    val reviewCount: String,
    val address: String,
    val addressUrl: String?,
    val images: List<PlaceCardPhoto>,
    val actions: List<PlaceCardAction>,
    val description: String,
)

/** A photo link is only an explicit, already-safe web URL from the source payload. */
internal fun placeCardPhotoOpenUrl(raw: Any?): String? {
    val url = raw?.toString()?.trim().orEmpty()
    if (!url.startsWith("http://", ignoreCase = true) &&
        !url.startsWith("https://", ignoreCase = true)) return null
    return SafeContentPolicy.sanitizeActionUrl(url)?.takeIf { it == url }
}

/** Keep source text and ordering; only omit URLs that cannot be rendered or opened safely. */
internal fun placeCardFromProps(props: Map<String, Any?>): PlaceCardModel {
    val images = (props["images"] as? List<*>).orEmpty().mapNotNull { raw ->
        val item = raw as? Map<*, *> ?: return@mapNotNull null
        val url = SafeContentPolicy.sanitizeMediaUrl(
            item["url"]?.toString(), SafeContentPolicy.MediaKind.IMAGE
        ) ?: return@mapNotNull null
        PlaceCardPhoto(
            url = url,
            title = item["title"]?.toString()?.trim().orEmpty(),
            onClickUrl = placeCardPhotoOpenUrl(item["onClickUrl"]),
        )
    }
    val actions = (props["actions"] as? List<*>).orEmpty().mapNotNull { raw ->
        val item = raw as? Map<*, *> ?: return@mapNotNull null
        val label = item["label"]?.toString()?.trim().orEmpty()
        val url = SafeContentPolicy.sanitizeActionUrl(item["url"]?.toString())
        if (label.isBlank() || url == null) return@mapNotNull null
        PlaceCardAction(label = label, url = url)
    }
    return PlaceCardModel(
        title = props["title"]?.toString()?.trim().orEmpty(),
        rating = props["rating"]?.toString()?.trim().orEmpty(),
        reviewCount = props["reviewCount"]?.toString()?.trim().orEmpty(),
        address = props["address"]?.toString()?.trim().orEmpty(),
        addressUrl = SafeContentPolicy.sanitizeActionUrl(props["addressUrl"]?.toString()),
        images = images,
        actions = actions,
        description = props["description"]?.toString()?.trim().orEmpty(),
    )
}

internal fun placeCardPhotoDescription(index: Int, count: Int, status: String, title: String): String =
    "Photo ${index + 1} of $count, $status: $title"

@Composable
private fun placeCardAccent(): Color = if (isSystemInDarkTheme()) Color(0xFF71DCD0) else Color(0xFF087E77)

/** Native, compact place card; parent surfaces retain ownership of vertical scrolling. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun RenderPlaceCard(context: FlatRenderContext) {
    val place = remember(context.props) { placeCardFromProps(context.props) }
    val interactionEnabled = LocalFlatInteractionEnabled.current
    val accent = placeCardAccent()
    Card(
        modifier = context.modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
            .semantics { contentDescription = "Place card: ${place.title}" },
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, genUiCardBorderColor()),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
    ) {
        Column(
            modifier = Modifier.fillMaxWidth().padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(9.dp),
        ) {
            if (place.title.isNotBlank()) {
                Text(
                    text = place.title,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.semantics { heading() },
                )
            }
            if (place.rating.isNotBlank() || place.reviewCount.isNotBlank()) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                    modifier = Modifier.semantics(mergeDescendants = true) {},
                ) {
                    if (place.rating.isNotBlank()) {
                        Icon(
                            imageVector = Icons.Filled.Star,
                            contentDescription = null,
                            tint = Color(0xFFEEA72B),
                            modifier = Modifier.size(17.dp),
                        )
                        Text(
                            text = place.rating,
                            style = MaterialTheme.typography.labelLarge,
                            fontWeight = FontWeight.SemiBold,
                        )
                    }
                    if (place.reviewCount.isNotBlank()) {
                        Text(
                            text = place.reviewCount,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
            if (place.address.isNotBlank()) {
                val addressModifier = if (interactionEnabled && place.addressUrl != null) {
                    Modifier.clickable(role = Role.Button, onClickLabel = "Open address") {
                        context.onOpenUrl(place.addressUrl)
                    }
                } else Modifier
                Row(
                    verticalAlignment = Alignment.Top,
                    horizontalArrangement = Arrangement.spacedBy(5.dp),
                    modifier = addressModifier,
                ) {
                    Icon(
                        imageVector = Icons.Filled.LocationOn,
                        contentDescription = null,
                        tint = accent,
                        modifier = Modifier.size(17.dp),
                    )
                    Text(
                        text = place.address,
                        style = MaterialTheme.typography.bodySmall,
                        color = if (place.addressUrl != null) accent else MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            if (place.description.isNotBlank()) {
                Text(
                    text = place.description,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            if (place.images.isNotEmpty()) {
                BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
                    val photoWidth = (maxWidth * 0.8f).coerceAtMost(300.dp)
                    LazyRow(
                        modifier = Modifier.semantics { contentDescription = "Place photos: ${place.title}" },
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        itemsIndexed(place.images, key = { index, _ -> index }) { index, photo ->
                            PlaceCardPhotoFrame(
                                photo = photo,
                                index = index,
                                count = place.images.size,
                                fallbackTitle = place.title,
                                interactionEnabled = interactionEnabled,
                                onOpenUrl = context.onOpenUrl,
                                modifier = Modifier.width(photoWidth).height(146.dp),
                            )
                        }
                    }
                }
            }
            if (place.actions.isNotEmpty()) {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(7.dp),
                    verticalArrangement = Arrangement.spacedBy(7.dp),
                ) {
                    place.actions.forEach { action ->
                        PlaceCardActionChip(action, interactionEnabled, accent, context.onOpenUrl)
                    }
                }
            }
        }
    }
}

@Composable
private fun PlaceCardPhotoFrame(
    photo: PlaceCardPhoto,
    index: Int,
    count: Int,
    fallbackTitle: String,
    interactionEnabled: Boolean,
    onOpenUrl: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val resolveAssetUrl = LocalFlatSpecAssetResolver.current
    val model = remember(photo.url, resolveAssetUrl) { resolveAssetUrl(photo.url) }
    val context = LocalContext.current
    val imageLoader = rememberFlatImageLoader()
    val request = remember(context, model) {
        ImageRequest.Builder(context)
            .data(model)
            .applyFlatSpecRemoteImageHeaders(model)
            .crossfade(true)
            .allowHardware(false)
            .build()
    }
    var status by remember(photo.url) { mutableStateOf("loading") }
    val title = photo.title.ifBlank { fallbackTitle }.ifBlank { "Place" }
    val clickableModifier = if (photo.onClickUrl != null) {
        modifier.clickable(
            enabled = interactionEnabled,
            role = Role.Button,
            onClickLabel = "Open photo link",
        ) { onOpenUrl(photo.onClickUrl) }
    } else modifier
    Box(
        modifier = clickableModifier
            .clip(RoundedCornerShape(12.dp))
            .background(MaterialTheme.colorScheme.surfaceContainerHigh)
            .semantics(mergeDescendants = true) {
                contentDescription = placeCardPhotoDescription(index, count, status, title)
            },
        contentAlignment = Alignment.Center,
    ) {
        if (status != "loaded") {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(5.dp),
            ) {
                Icon(
                    imageVector = Icons.Filled.Image,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.size(23.dp),
                )
                Text(
                    text = if (status == "unavailable") "Photo unavailable" else "Loading photo",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        AsyncImage(
            model = request,
            imageLoader = imageLoader,
            contentDescription = null,
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize(),
            onLoading = { status = "loading" },
            onSuccess = { status = "loaded" },
            onError = { status = "unavailable" },
        )
    }
}

@Composable
private fun PlaceCardActionChip(
    action: PlaceCardAction,
    interactionEnabled: Boolean,
    accent: Color,
    onOpenUrl: (String) -> Unit,
) {
    val icon = when {
        action.url.startsWith("tel:", ignoreCase = true) ||
            action.label.contains("call", ignoreCase = true) -> Icons.Filled.Call
        action.label.contains("direction", ignoreCase = true) ||
            action.label.contains("map", ignoreCase = true) -> Icons.Filled.Directions
        action.label.contains("website", ignoreCase = true) -> Icons.Filled.Language
        else -> Icons.AutoMirrored.Filled.OpenInNew
    }
    Surface(
        modifier = Modifier.clickable(
            enabled = interactionEnabled,
            role = Role.Button,
            onClickLabel = action.label,
        ) { onOpenUrl(action.url) },
        shape = RoundedCornerShape(50),
        color = accent.copy(alpha = if (isSystemInDarkTheme()) 0.18f else 0.09f),
        border = BorderStroke(1.dp, accent.copy(alpha = 0.38f)),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 11.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(5.dp),
        ) {
            Icon(icon, contentDescription = null, modifier = Modifier.size(16.dp), tint = accent)
            Text(
                text = action.label,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.SemiBold,
                color = accent,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}
