package com.samsung.genuicraft.sdk.internal.renderer

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
import com.samsung.genuicraft.sdk.internal.renderer.flat.legacy.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.*

// moved from FlatSpecRenderer.kt (renderRestaurantRowsIfPossible)
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun renderRestaurantRowsIfPossible(
    headers: List<String>,
    rows: List<List<String>>,
    onOpenUrl: (String) -> Unit,
    modifier: Modifier = Modifier,
    title: String? = null
): Boolean {
    if (rows.isEmpty()) return false
    val titleIndex = restaurantTitleIndex(headers)
    val ratingIndex = restaurantRatingIndex(headers)
    val reviewCountIndex = restaurantReviewCountIndex(headers)
    val priceIndex = restaurantPriceIndex(headers)
    val statusIndex = restaurantStatusIndex(headers)
    val addressIndex = restaurantAddressIndex(headers)
    val descriptionIndex = restaurantDescriptionIndex(headers)
    val tagsIndex = restaurantTagsIndex(headers)
    val photoIndex = restaurantPhotoIndex(headers)
    val photoIndexes = restaurantPhotoIndexes(headers)
    val mapsIndex = restaurantMapsIndex(headers)
    val websiteIndex = restaurantWebsiteIndex(headers)
    val bookUrlIndex = restaurantBookUrlIndex(headers)
    val actionLabelIndex = restaurantActionLabelIndex(headers)
    val amenitiesIndex = restaurantAmenitiesIndex(headers)
    val phoneIndex = restaurantPhoneIndex(headers)
    val hasRestaurantSignal =
        headers.any(::isRestaurantHeaderLabel) &&
            (ratingIndex != null || addressIndex != null || photoIndex != null || mapsIndex != null || websiteIndex != null || bookUrlIndex != null)
    if (!hasRestaurantSignal) return false

    val shownRows = rows.filter { row -> row.any { value -> value.trim().isNotBlank() } }
    if (shownRows.isEmpty()) return false

    Column(
        modifier = modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = "Restaurant results with ${shownRows.size} places"
            },
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        title?.trim()?.takeIf { it.isNotBlank() }?.let { headingText ->
            Text(
                text = parseBoldMarkdown(headingText),
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                color = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 2.dp, vertical = 6.dp)
                    .semantics { heading() }
            )
        }
        shownRows.forEachIndexed { index, row ->
            val restaurantName = row.getOrNull(titleIndex).orEmpty().trim().ifBlank { "Restaurant ${index + 1}" }
            val rating = ratingIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty()
            val reviews = reviewCountIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty()
            val price = priceIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty()
            val status = statusIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty()
            val address = addressIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty()
            val description = descriptionIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty()
            val amenities = amenitiesIndex?.let { splitRestaurantTags(row.getOrNull(it).orEmpty()) }.orEmpty()
            val tags = (tagsIndex?.let { splitRestaurantTags(row.getOrNull(it).orEmpty()) }.orEmpty() + amenities)
                .distinct()
                .take(7)
            val photos = photoIndexes
                .flatMap { splitRestaurantPhotoUrls(row.getOrNull(it).orEmpty()) }
                .distinct()
                .take(3)
                .ifEmpty { listOf("genuicraft://visual/restaurant?title=${Uri.encode(restaurantName)}") }
            val mapsUrl = mapsIndex?.let { SafeContentPolicy.sanitizeActionUrl(row.getOrNull(it).orEmpty().trim()) }
            val websiteUrl = websiteIndex?.let { SafeContentPolicy.sanitizeActionUrl(row.getOrNull(it).orEmpty().trim()) }
            val bookUrl = bookUrlIndex?.let { SafeContentPolicy.sanitizeActionUrl(row.getOrNull(it).orEmpty().trim()) }
            val hasReservationSignal = tags.any { tag ->
                tag.contains("reservation", ignoreCase = true) ||
                    tag.contains("reserve", ignoreCase = true) ||
                    tag.contains("book", ignoreCase = true)
            }
            val rawActionLabel = actionLabelIndex
                ?.let { row.getOrNull(it).orEmpty().trim() }
                ?.takeIf { it.isNotBlank() && !isLikelyHttpUrl(it) }
                ?.let { label ->
                    if (label.contains("reserve", ignoreCase = true) ||
                        label.contains("book table", ignoreCase = true)
                    ) {
                        "Reserve Table"
                    } else {
                        label
                    }
                }
            val actionLabel = when {
                hasReservationSignal && !bookUrl.isNullOrBlank() ->
                    rawActionLabel
                        ?.takeIf { it.contains("reserve", ignoreCase = true) }
                        ?: "Reserve Table"
                hasReservationSignal && !mapsUrl.isNullOrBlank() ->
                    rawActionLabel
                        ?.takeIf { it.contains("reserve", ignoreCase = true) }
                        ?: "Reserve Table"
                !rawActionLabel.isNullOrBlank() -> rawActionLabel
                else -> when {
                    !bookUrl.isNullOrBlank() -> "Menu / Details"
                    !websiteUrl.isNullOrBlank() -> "Website"
                    !mapsUrl.isNullOrBlank() && hasReservationSignal -> "Reserve Table"
                    !mapsUrl.isNullOrBlank() -> "Directions"
                    else -> ""
                }
            }
            val phone = phoneIndex?.let { row.getOrNull(it).orEmpty().trim() }.orEmpty()
            val phoneDialUrl = SafeContentPolicy.sanitizePhoneDialUrl(phone)
            val hasReserveAction = actionLabel.contains("reserve", ignoreCase = true) ||
                actionLabel.contains("book table", ignoreCase = true)
            val reserveUrl = bookUrl.takeIf { hasReserveAction && !it.isNullOrBlank() }

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics(mergeDescendants = true) {
                        contentDescription = tableRowAccessibilitySummary(headers, row)
                    },
                shape = RoundedCornerShape(20.dp),
                colors = flatSpecCardColors(),
                elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
                border = flatSpecCardBorder()
            ) {
                Column(
                    modifier = Modifier.padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    RestaurantPhotoStrip(
                        name = restaurantName,
                        photos = photos,
                        onOpenUrl = onOpenUrl
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        verticalAlignment = Alignment.Top
                    ) {
                        Text(
                            text = parseBoldMarkdown(restaurantName),
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                            color = MaterialTheme.colorScheme.onSurface,
                            modifier = Modifier.weight(1f)
                        )
                    }
                    RestaurantRatingChips(
                        rating = rating,
                        reviews = reviews,
                        price = price
                    )
                    if (status.isNotBlank()) {
                        val openLike = status.contains("open", ignoreCase = true) && !status.contains("closed", ignoreCase = true)
                        val darkTheme = isSystemInDarkTheme()
                        Surface(
                            shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                            color = if (openLike) {
                                if (darkTheme) Color(0xFF163529) else Color(0xFFEAF7F0)
                            } else {
                                MaterialTheme.colorScheme.errorContainer.copy(alpha = if (darkTheme) 0.30f else 0.52f)
                            },
                            border = BorderStroke(
                                1.dp,
                                if (openLike) {
                                    Color(0xFF18A767).copy(alpha = if (darkTheme) 0.28f else 0.22f)
                                } else {
                                    MaterialTheme.colorScheme.error.copy(alpha = if (darkTheme) 0.26f else 0.18f)
                                }
                            )
                        ) {
                            Text(
                                text = parseBoldMarkdown(status),
                                style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.Medium),
                                color = if (openLike) {
                                    if (darkTheme) Color(0xFF8DDBB3) else Color(0xFF1C6B45)
                                } else {
                                    MaterialTheme.colorScheme.onErrorContainer
                                },
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                            )
                        }
                    }
                    if (description.isNotBlank()) {
                        Text(
                            text = parseBoldMarkdown(description),
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurface,
                            lineHeight = MaterialTheme.typography.bodyMedium.lineHeight,
                            maxLines = 3,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                    if (address.isNotBlank()) {
                        Text(
                            text = parseBoldMarkdown(address),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                    if (tags.isNotEmpty()) {
                        FlowRow(
                            horizontalArrangement = Arrangement.spacedBy(7.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            tags.forEach { tag ->
                                RestaurantTagChip(tag)
                            }
                        }
                    }
                    FlowRow(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        if (!reserveUrl.isNullOrBlank()) {
                            RestaurantActionPill(
                                label = "Reserve Table",
                                icon = Icons.Filled.EventAvailable,
                                primary = true,
                                onClick = { onOpenUrl(reserveUrl) }
                            )
                        }
                        if (!phoneDialUrl.isNullOrBlank()) {
                            RestaurantActionPill(
                                label = "Call",
                                icon = Icons.Filled.Call,
                                onClick = { onOpenUrl(phoneDialUrl) }
                            )
                        }
                        if (!mapsUrl.isNullOrBlank()) {
                            RestaurantActionPill(
                                label = "Directions",
                                icon = Icons.Filled.Directions,
                                onClick = { onOpenUrl(mapsUrl) }
                            )
                        }
                        if (!websiteUrl.isNullOrBlank()) {
                            RestaurantActionPill(
                                label = "Website",
                                icon = Icons.Filled.Language,
                                onClick = { onOpenUrl(websiteUrl) }
                            )
                        }
                        if (reserveUrl.isNullOrBlank() &&
                            bookUrl != null &&
                            bookUrl != websiteUrl &&
                            bookUrl != mapsUrl &&
                            actionLabel.isNotBlank()
                        ) {
                            RestaurantActionPill(
                                label = actionLabel,
                                icon = Icons.Filled.Link,
                                primary = true,
                                onClick = { onOpenUrl(bookUrl) }
                            )
                        }
                    }
                }
            }
        }
    }
    return true
}

// moved from FlatSpecRenderer.kt (renderBookingRowsIfPossible)
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun renderBookingRowsIfPossible(
    headers: List<String>,
    rows: List<List<String>>,
    onOpenUrl: (String) -> Unit
): Boolean {
    if (rows.isEmpty()) return false
    val titleIndex = findTableColumnIndex(
        headers = headers,
        keywords = listOf("hotel", "property", "provider", "option", "listing", "vendor", "airline", "name", "route", "plan")
    ) ?: 0
    val priceIndex = findTableColumnIndex(
        headers = headers,
        keywords = listOf("price", "cost", "fare", "rate", "night", "budget")
    )
    val actionLabelIndex = bookingActionLabelIndex(headers)
    val imageIndex = bookingImageIndex(headers)
    val ratingIndex = bookingRatingIndex(headers)
    val reviewCountIndex = bookingReviewCountIndex(headers)
    val classIndex = bookingClassIndex(headers)
    val amenitiesIndex = bookingAmenitiesIndex(headers)
    val mapIndex = bookingMapIndex(headers)
    val websiteIndex = bookingWebsiteIndex(headers)
    val photosDataIndex = bookingPhotosDataIndex(headers)
    val linkIndex = findTableColumnIndex(
        headers = headers,
        keywords = listOf("book", "booking", "reserve", "action url", "url", "link", "website"),
        exclude = setOfNotNull(imageIndex, mapIndex, photosDataIndex)
    )
    val secondaryIndex = findTableColumnIndex(
        headers = headers,
        keywords = listOf("duration", "time", "date", "location", "room", "type", "class", "stops", "status"),
        exclude = setOfNotNull(titleIndex, priceIndex, ratingIndex, reviewCountIndex, amenitiesIndex)
    )
    val hasBookingSignal = priceIndex != null || linkIndex != null || headers.any(::isBookingEntityHeaderLabel)
    if (!hasBookingSignal) return false

    val shownRows = rows.filter { row ->
        row.any { value -> value.trim().isNotBlank() }
    }
    if (shownRows.isEmpty()) return false

    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        shownRows.forEach { row ->
            val title = row.getOrNull(titleIndex).orEmpty().trim().ifBlank { "Option" }
            val price = priceIndex?.let { index -> row.getOrNull(index).orEmpty().trim() }.orEmpty()
            val secondary = secondaryIndex?.let { index -> row.getOrNull(index).orEmpty().trim() }.orEmpty()
            val rating = ratingIndex?.let { index -> row.getOrNull(index).orEmpty().trim() }.orEmpty()
            val reviews = reviewCountIndex?.let { index -> row.getOrNull(index).orEmpty().trim() }.orEmpty()
            val hotelClass = classIndex?.let { index -> row.getOrNull(index).orEmpty().trim() }.orEmpty()
            val amenities = amenitiesIndex?.let { index -> splitRestaurantTags(row.getOrNull(index).orEmpty()) }.orEmpty()
            val actionUrl = linkIndex?.let { index -> row.getOrNull(index).orEmpty().trim() }
                ?.let(SafeContentPolicy::sanitizeActionUrl)
            val mapsUrl = mapIndex?.let { index -> row.getOrNull(index).orEmpty().trim() }
                ?.let(SafeContentPolicy::sanitizeActionUrl)
            val websiteUrl = websiteIndex?.let { index -> row.getOrNull(index).orEmpty().trim() }
                ?.let(SafeContentPolicy::sanitizeActionUrl)
            val photosDataUrl = photosDataIndex?.let { index -> row.getOrNull(index).orEmpty().trim() }
                ?.let(SafeContentPolicy::sanitizeActionUrl)
            val actionLabel = bookingRowActionLabel(headers, row) ?: "Open Option"
            val imageUrls = bookingRowImageUrls(headers, row)
            val excludedChipIndexes = setOfNotNull(
                titleIndex,
                priceIndex,
                secondaryIndex,
                linkIndex,
                actionLabelIndex,
                imageIndex,
                ratingIndex,
                reviewCountIndex,
                classIndex,
                amenitiesIndex,
                mapIndex,
                websiteIndex,
                photosDataIndex
            )
            val chips = buildList {
                headers.forEachIndexed { index, header ->
                    if (index in excludedChipIndexes) {
                        return@forEachIndexed
                    }
                    val value = row.getOrNull(index).orEmpty().trim()
                    if (value.isBlank() || isLikelyHttpUrl(value)) return@forEachIndexed
                    add(header.ifBlank { "Detail" } to value)
                }
            }.take(5)

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics {
                        contentDescription = tableRowAccessibilitySummary(headers, row)
                    },
                shape = RoundedCornerShape(22.dp),
                colors = flatSpecCardColors(),
                elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
                border = flatSpecCardBorder()
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    if (imageUrls.isNotEmpty()) {
                        BookingPhotoStrip(
                            name = title,
                            photos = imageUrls,
                            onOpenUrl = onOpenUrl
                        )
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        verticalAlignment = Alignment.Top
                    ) {
                        Column(
                            modifier = Modifier.weight(1f),
                            verticalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Text(
                                text = parseBoldMarkdown(title),
                                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            val subtitle = listOf(hotelClass, secondary)
                                .filter { it.isNotBlank() }
                                .distinct()
                                .joinToString(" • ")
                            if (subtitle.isNotBlank()) {
                                Text(
                                    text = parseBoldMarkdown(subtitle),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                        if (price.isNotBlank()) {
                            Surface(
                                shape = RoundedCornerShape(GenUiTokens.RadiusPill),
                                color = MaterialTheme.colorScheme.primaryContainer
                            ) {
                                Text(
                                    text = parseBoldMarkdown(price),
                                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp)
                                )
                            }
                        }
                    }
                    if (rating.isNotBlank() || reviews.isNotBlank()) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            if (rating.isNotBlank()) {
                                BookingMetricChip(
                                    label = rating,
                                    icon = Icons.Filled.Star,
                                    emphasized = true
                                )
                            }
                            if (reviews.isNotBlank()) {
                                BookingMetricChip(
                                    label = reviews,
                                    icon = Icons.Filled.RateReview
                                )
                            }
                        }
                    }
                    if (amenities.isNotEmpty()) {
                        FlowRow(
                            horizontalArrangement = Arrangement.spacedBy(7.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            amenities.take(6).forEach { amenity ->
                                BookingDetailChip(amenity)
                            }
                        }
                    }
                    if (chips.isNotEmpty()) {
                        FlowRow(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            chips.forEach { (label, value) ->
                                BookingDetailChip("${label.trim()}: ${value.trim()}")
                            }
                        }
                    }
                    val actions = listOfNotNull(
                        actionUrl?.let { actionLabel to (Icons.Filled.EventAvailable to it) },
                        mapsUrl?.takeIf { it != actionUrl }?.let { "Directions" to (Icons.Filled.Directions to it) },
                        websiteUrl?.takeIf { it != actionUrl && it != mapsUrl }?.let { "Website" to (Icons.Filled.Language to it) },
                        photosDataUrl?.takeIf { it != actionUrl && it != mapsUrl && it != websiteUrl }?.let { "Photos" to (Icons.Filled.Image to it) }
                    )
                    if (actions.isNotEmpty()) {
                        FlowRow(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            actions.forEachIndexed { index, (label, iconAndUrl) ->
                                RestaurantActionPill(
                                    label = label,
                                    icon = iconAndUrl.first,
                                    primary = index == 0,
                                    onClick = { onOpenUrl(iconAndUrl.second) }
                                )
                            }
                        }
                    }
                }
            }
        }
    }
    return true
}
