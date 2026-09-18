package com.samsung.genuicraft.sdk.internal.renderer.native.intents.flight

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.sizeIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.draw.clip
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FlightTakeoff
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import coil.ImageLoader
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.samsung.genuicraft.sdk.internal.theme.GenUiCardTone
import com.samsung.genuicraft.sdk.internal.theme.GenUiTokens
import com.samsung.genuicraft.sdk.internal.theme.genUiCardBorderColor
import com.samsung.genuicraft.sdk.internal.theme.genUiCardColors
import com.samsung.genuicraft.sdk.internal.theme.genUiCardContainerColor
import com.samsung.genuicraft.sdk.internal.renderer.native.FlightRow
import com.samsung.genuicraft.sdk.internal.renderer.native.NativeTextFormatter
import com.samsung.genuicraft.sdk.internal.security.SafeContentPolicy
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.*

internal object NativeFlightUiRenderer {
    @Composable
    fun RenderFlightRows(
        rows: List<FlightRow>,
        onOpenUrl: (String) -> Unit = {}
    ) {
        if (rows.isEmpty()) {
            return
        }

        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            rows.forEach { row ->
                val airline = sanitize(row.airline)
                val fare = sanitize(row.fare.orEmpty()).ifBlank { null }
                val (fareValue, fareMeta) = NativeFlightSemantics.splitFareDisplay(fare)
                val depart = sanitize(row.depart.orEmpty()).ifBlank { null }
                val arrive = sanitize(row.arrive.orEmpty()).ifBlank { null }
                val originCode = sanitize(row.originCode.orEmpty())
                    .ifBlank { NativeFlightSemantics.extractAirportCode(depart.orEmpty()).orEmpty() }
                    .ifBlank { null }
                val destinationCode = sanitize(row.destinationCode.orEmpty())
                    .ifBlank { NativeFlightSemantics.extractAirportCode(arrive.orEmpty()).orEmpty() }
                    .ifBlank { null }
                val departPoint = NativeFlightSemantics.parseFlightPoint(depart, originCode)
                val arrivePoint = NativeFlightSemantics.parseFlightPoint(arrive, destinationCode)
                val departDisplay = departPoint.time ?: departPoint.code ?: depart
                val arriveDisplay = arrivePoint.time ?: arrivePoint.code ?: arrive
                val duration = NativeFlightSemantics.normalizeDurationLabel(row.duration)
                val stopLabel = NativeFlightSemantics.normalizeStopLabel(row.stops, row.status, depart, arrive)
                val statusLabel = NativeFlightSemantics.normalizeFlightStatus(row.status, stopLabel)
                val topStatus = statusLabel?.takeIf { NativeFlightSemantics.looksLikePunctualityStatus(it) }
                val arrivalStatus = statusLabel?.takeUnless { NativeFlightSemantics.looksLikePunctualityStatus(it) }
                val centerMeta = stopLabel ?: arrivalStatus?.takeIf { it.length <= 22 }
                val promoMeta = arrivalStatus?.takeIf { it.length > 22 }
                val hasTimeRow = departDisplay != null || arriveDisplay != null || duration != null
                val actionUrl = remember(row.actionUrl) { SafeContentPolicy.sanitizeActionUrl(row.actionUrl) }
                val actionLabel = sanitize(row.actionLabel.orEmpty()).ifBlank { "View fare" }

                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .semantics(mergeDescendants = true) {
                            contentDescription = flightRowAccessibilityLabel(
                                airline = airline,
                                fareValue = fareValue,
                                fareMeta = fareMeta,
                                depart = departDisplay,
                                arrive = arriveDisplay,
                                duration = duration,
                                stopLabel = stopLabel,
                                statusLabel = statusLabel
                            )
                        },
                    shape = RoundedCornerShape(24.dp),
                    colors = genUiCardColors(GenUiCardTone.Neutral),
                    elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                    border = BorderStroke(
                        width = GenUiTokens.BorderMd,
                        color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.55f)
                    ),
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 14.dp, vertical = 12.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(10.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(
                                modifier = Modifier.weight(1f),
                                horizontalArrangement = Arrangement.spacedBy(10.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                AirlineBadge(airline = airline, logoUrl = row.logoUrl)
                                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                    MarkdownText(
                                        text = airline,
                                        style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                    topStatus?.let { statusValue ->
                                        MarkdownText(
                                            text = statusValue,
                                            style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold),
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }
                            }
                            if (fareValue != null) {
                                Column(
                                    modifier = Modifier.sizeIn(minWidth = 84.dp),
                                    horizontalAlignment = Alignment.End,
                                    verticalArrangement = Arrangement.spacedBy(2.dp)
                                ) {
                                    MarkdownText(
                                        text = fareValue,
                                        style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
                                        color = MaterialTheme.colorScheme.onSurface,
                                        textAlign = TextAlign.End
                                    )
                                    fareMeta?.let { fareSuffix ->
                                        MarkdownText(
                                            text = fareSuffix,
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            textAlign = TextAlign.End
                                        )
                                    }
                                }
                            }
                        }

                        if (hasTimeRow) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(10.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                FlightTimeCell(
                                    title = departDisplay,
                                    subtitle = departPoint.code,
                                    align = TextAlign.Start,
                                    modifier = Modifier.weight(1f),
                                    emphasis = true
                                )
                                Column(
                                    modifier = Modifier.weight(1.1f),
                                    verticalArrangement = Arrangement.spacedBy(5.dp),
                                    horizontalAlignment = Alignment.CenterHorizontally
                                ) {
                                    duration?.let { durationValue ->
                                        MarkdownText(
                                            text = durationValue,
                                            style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                                            color = MaterialTheme.colorScheme.onSurface,
                                            textAlign = TextAlign.Center
                                        )
                                    }
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(horizontal = 2.dp),
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                                    ) {
                                        HorizontalDivider(
                                            modifier = Modifier.weight(1f),
                                            color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.75f)
                                        )
                                        Icon(
                                            imageVector = Icons.Filled.FlightTakeoff,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.onSurfaceVariant,
                                            modifier = Modifier.size(14.dp)
                                        )
                                        HorizontalDivider(
                                            modifier = Modifier.weight(1f),
                                            color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.75f)
                                        )
                                    }
                                    centerMeta?.let { meta ->
                                        FlightMetaChip(meta)
                                    }
                                }
                                FlightTimeCell(
                                    title = arriveDisplay,
                                    subtitle = arrivePoint.code,
                                    align = TextAlign.End,
                                    modifier = Modifier.weight(1f),
                                    emphasis = true
                                )
                            }
                            promoMeta?.let { promo ->
                                MarkdownText(
                                    text = promo,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.fillMaxWidth()
                                )
                            }
                        } else {
                            val meta = listOfNotNull(stopLabel, arrivalStatus, fareValue, fareMeta)
                            if (meta.isNotEmpty()) {
                                MarkdownText(
                                    text = meta.joinToString(" | "),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                        actionUrl?.let { safeUrl ->
                            Button(
                                onClick = { onOpenUrl(safeUrl) },
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(999.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Filled.FlightTakeoff,
                                    contentDescription = null,
                                    modifier = Modifier.size(16.dp)
                                )
                                Text(
                                    text = actionLabel,
                                    modifier = Modifier.padding(start = 8.dp)
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    private fun flightRowAccessibilityLabel(
        airline: String,
        fareValue: String?,
        fareMeta: String?,
        depart: String?,
        arrive: String?,
        duration: String?,
        stopLabel: String?,
        statusLabel: String?
    ): String {
        return buildString {
            airline.takeIf { it.isNotBlank() }?.let { append(it) }
            depart?.takeIf { it.isNotBlank() }?.let { value ->
                if (isNotEmpty()) append(". ")
                append("Depart ")
                append(value)
            }
            arrive?.takeIf { it.isNotBlank() }?.let { value ->
                if (isNotEmpty()) append(". ")
                append("Arrive ")
                append(value)
            }
            duration?.takeIf { it.isNotBlank() }?.let { value ->
                if (isNotEmpty()) append(". ")
                append(value)
            }
            stopLabel?.takeIf { it.isNotBlank() }?.let { value ->
                if (isNotEmpty()) append(". ")
                append(value)
            }
            statusLabel?.takeIf { it.isNotBlank() }?.let { value ->
                if (isNotEmpty()) append(". ")
                append(value)
            }
            fareValue?.takeIf { it.isNotBlank() }?.let { value ->
                if (isNotEmpty()) append(". ")
                append(value)
            }
            fareMeta?.takeIf { it.isNotBlank() }?.let { value ->
                if (isNotEmpty()) append(" ")
                append(value)
            }
        }.ifBlank { "Flight option" }
    }

    @Composable
    private fun AirlineBadge(airline: String, logoUrl: String?) {
        val accent = airlineAccentColor(airline)
        val code = NativeFlightSemantics.airlineBadgeCode(airline)
        val resolvedLogo = remember(airline, logoUrl) { airlineLogoUrl(airline, logoUrl) }
        var logoFailed by remember(resolvedLogo) { mutableStateOf(false) }
        val context = LocalContext.current
        val imageLoader = rememberFlatImageLoader()
        Box(
            modifier = Modifier
                .size(42.dp)
                .clip(RoundedCornerShape(999.dp)),
            contentAlignment = Alignment.Center
        ) {
            if (resolvedLogo != null && !logoFailed) {
                AsyncImage(
                    model = ImageRequest.Builder(context)
                        .data(resolvedLogo)
                        .crossfade(true)
                        .allowHardware(false)
                        .build(),
                    imageLoader = imageLoader,
                    contentDescription = null,
                    contentScale = ContentScale.Fit,
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(6.dp),
                    onError = { logoFailed = true }
                )
            } else {
                Icon(
                    imageVector = Icons.Filled.FlightTakeoff,
                    contentDescription = null,
                    tint = accent,
                    modifier = Modifier.size(18.dp)
                )
            }
            if (code.isNotBlank() && (resolvedLogo == null || logoFailed)) {
                Text(
                    text = code,
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                    color = accent,
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(bottom = 1.dp)
                )
            }
        }
    }

    private fun airlineLogoUrl(airline: String, explicitLogoUrl: String?): String? {
        explicitLogoUrl
            ?.trim()
            ?.takeIf { looksLikeMediaUrl(it) }
            ?.let { return it }
        val code = NativeFlightSemantics.airlineBadgeCode(airline).trim()
        if (code.length !in 2..3 || !code.all { it.isLetterOrDigit() }) {
            return null
        }
        return "https://www.gstatic.com/flights/airline_logos/70px/${code.uppercase()}.png"
    }

    private fun looksLikeMediaUrl(value: String): Boolean {
        return value.startsWith("http://", ignoreCase = true) ||
            value.startsWith("https://", ignoreCase = true) ||
            value.startsWith("../assets/", ignoreCase = true) ||
            value.startsWith("assets/", ignoreCase = true) ||
            value.startsWith("file:", ignoreCase = true)
    }

    @Composable
    private fun FlightMetaChip(text: String) {
        MarkdownText(
            text = text,
            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier
                .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                .background(genUiCardContainerColor(GenUiCardTone.Neutral))
                .border(
                    width = GenUiTokens.BorderMd,
                    color = genUiCardBorderColor(),
                    shape = RoundedCornerShape(GenUiTokens.RadiusPill)
                )
                .padding(horizontal = 8.dp, vertical = 3.dp)
        )
    }

    @Composable
    private fun FlightTimeCell(
        title: String?,
        subtitle: String?,
        align: TextAlign,
        modifier: Modifier = Modifier,
        emphasis: Boolean = false
    ) {
        Column(
            modifier = modifier,
            verticalArrangement = Arrangement.spacedBy(2.dp),
            horizontalAlignment = when (align) {
                TextAlign.End -> Alignment.End
                TextAlign.Center -> Alignment.CenterHorizontally
                else -> Alignment.Start
            }
        ) {
            if (!title.isNullOrBlank()) {
                MarkdownText(
                    text = title,
                    style = if (emphasis) {
                        MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                    } else {
                        MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold)
                    },
                    color = MaterialTheme.colorScheme.onSurface,
                    textAlign = align
                )
            }
            if (!subtitle.isNullOrBlank()) {
                MarkdownText(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = align
                )
            }
        }
    }

    @Composable
    private fun MarkdownText(
        text: String,
        style: TextStyle,
        color: Color,
        modifier: Modifier = Modifier,
        textAlign: TextAlign? = null
    ) {
        val displayText = remember(text) { NativeTextFormatter.sanitizeDisplayText(text, preserveMarkdown = true) }
        if (displayText.isBlank()) {
            return
        }
        val hasMarkdownInline = remember(displayText) { NativeTextFormatter.containsMarkdownInlineFormatting(displayText) }
        val hasMarkdownHeading = remember(displayText) { NativeTextFormatter.containsMarkdownHeading(displayText) }
        val hasMarkdown = hasMarkdownInline || hasMarkdownHeading
        val plainText = remember(displayText) { NativeTextFormatter.sanitizeDisplayText(displayText) }
        if (!hasMarkdown) {
            Text(
                text = plainText,
                style = style,
                color = color,
                modifier = modifier,
                textAlign = textAlign
            )
            return
        }
        val parsedText = remember(displayText, style, hasMarkdownHeading) {
            if (hasMarkdownHeading) {
                NativeTextFormatter.parseMarkdownWithHeadings(displayText, style)
            } else {
                NativeTextFormatter.parseInlineMarkdown(displayText)
            }
        }
        val effectiveStyle = style.copy(fontWeight = FontWeight.Normal)
        Text(
            text = parsedText,
            style = effectiveStyle,
            color = color,
            modifier = modifier,
            textAlign = textAlign
        )
    }

    @Composable
    private fun airlineAccentColor(airline: String): Color {
        val normalized = NativeFlightSemantics.normalizeMatchText(airline)
        return when {
            normalized.contains("indigo") -> Color(0xFF3F51B5)
            normalized.contains("akasa") -> Color(0xFF6D1B7B)
            normalized.contains("air india express") -> Color(0xFFE53935)
            normalized == "air india" || normalized.startsWith("air india ") -> Color(0xFFD32F2F)
            normalized.contains("vistara") -> Color(0xFF6A1B9A)
            normalized.contains("spicejet") -> Color(0xFFD84315)
            normalized.contains("emirates") -> Color(0xFFB71C1C)
            else -> MaterialTheme.colorScheme.primary
        }
    }

    private fun sanitize(value: String): String = NativeTextFormatter.sanitizeDisplayText(value)
}
