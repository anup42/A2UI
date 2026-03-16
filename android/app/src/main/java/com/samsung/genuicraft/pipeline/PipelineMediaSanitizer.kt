package com.samsung.genuicraft.pipeline

import android.content.Context
import android.content.SharedPreferences
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.net.URI
import java.net.URL
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.util.Locale

internal object PipelineMediaSanitizer {

    val URL_TOKEN_REGEX = Regex(
        """(?i)(?:https?://|//)[^\s<>\]]+|(?<![@\w])(?:www\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:[/?#][^\s<>\]]*)?"""
    )
    val HOST_LABEL_REGEX = Regex("""(?i)^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$""")

    const val DEFAULT_STAGE3_CATALOG_ID = "https://genui.local/specification/v0_9/standard_catalog.json"
    const val APP_PREFS_NAME = "genuicraft_prefs"
    const val PREF_STAGE3_CATALOG_ID = "stage3_catalog_id"
    private val FLIGHT_AIRLINE_REGEX = Regex(
        """(?i)\b(indigo|air india express|air india|akasa air|spicejet|vistara|emirates|qatar airways|qatar|lufthansa|united|delta|british airways|alliance air|flydubai|etihad|go first|goair)\b"""
    )
    private const val FLIGHT_FALLBACK_ICON_URL = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/airplane.svg"
    private val FLIGHT_ALLOWED_BOOTSTRAP_ICONS = setOf(
        "airplane",
        "airplane-fill",
        "ticket-perforated",
        "ticket-perforated-fill",
        "clock",
        "calendar",
        "calendar2-event",
        "geo-alt",
        "geo-alt-fill",
        "map",
        "signpost",
        "signpost-split",
        "building",
        "shop",
        "cash-stack",
        "currency-rupee"
    )
    private val FLIGHT_TIME_REGEX = Regex("""\b\d{1,2}:\d{2}(?:\s?(?:AM|PM))?\b""", RegexOption.IGNORE_CASE)
    private val FLIGHT_DURATION_REGEX = Regex("""(?i)\b\d+\s*h(?:\s*\d+\s*m)?\b|\b\d+\s*m\b""")
    private val FLIGHT_STOPS_REGEX = Regex("""(?i)\bnon[-\s]?stop\b|\bdirect\b|\b\d+\s*stop(?:s)?\b""")
    private val FLIGHT_FARE_REGEX = Regex("""(?i)(?:\u20B9|rs\.?|inr)\s*\d[\d,]*(?:\.\d+)?""")

    private data class FlightListRow(
        val airline: String,
        val departure: String?,
        val arrival: String?,
        val duration: String?,
        val stops: String?,
        val fare: String?
    )

    // ── Flight functions ───────────────────────────────────────────────

    fun ensureFlightQuickActions(
        responseText: String,
        queryText: String
    ): String {
        if (!looksLikeFlightQuery(queryText) && !looksLikeFlightContent(responseText)) {
            return responseText
        }

        val hasActionWithUrl = Regex(
            """(?im)^\s*Action:\s*\[Button:\s*.+?\]\s*(?:https?://|//|www\.|(?:[a-z0-9-]+\.)+[a-z]{2,24})\S*"""
        )
            .containsMatchIn(responseText)
        val hasQuickActionsWithUrl = Regex(
            """(?is)Quick\s*Actions.*(?:https?://|//|www\.|(?:[a-z0-9-]+\.)+[a-z]{2,24})"""
        )
            .containsMatchIn(responseText)
        if (hasActionWithUrl || hasQuickActionsWithUrl) {
            return responseText
        }

        val sourceUrls = extractUrlsForQuickActions(responseText)
        val actionLines = if (sourceUrls.isNotEmpty()) {
            sourceUrls.take(3).mapIndexed { index, url ->
                "Action: [Button: ${quickActionLabelForUrl(url, index)}] $url"
            }
        } else {
            listOf(
                "Action: [Button: Search on MakeMyTrip] https://www.makemytrip.com/flights/",
                "Action: [Button: Search on Skyscanner] https://www.skyscanner.co.in/",
                "Action: [Button: Search on Goibibo] https://www.goibibo.com/flights/"
            )
        }

        return buildString {
            append(responseText.trimEnd())
            append("\n\nQuick Actions\n")
            append(actionLines.joinToString(separator = "\n"))
        }
    }

    fun sanitizeFlightInlineMedia(
        responseText: String,
        queryText: String
    ): String {
        if (!looksLikeFlightQuery(queryText) && !looksLikeFlightContent(responseText)) {
            return responseText
        }
        val normalized = responseText.replace("\r\n", "\n")
        if (normalized.isBlank()) {
            return responseText
        }
        val lines = normalized.split('\n')
        val output = mutableListOf<String>()
        var removedAny = false

        lines.forEach { rawLine ->
            val trimmed = rawLine.trim()
            if (trimmed.isBlank()) {
                output += rawLine
                return@forEach
            }
            if (!isFlightMediaCandidateLine(trimmed)) {
                output += rawLine
                return@forEach
            }

            val sanitizedLine = sanitizeFlightMediaCandidateLine(trimmed)
            if (sanitizedLine.isNullOrBlank()) {
                removedAny = true
                return@forEach
            }

            val leadingWhitespace = rawLine.takeWhile { it.isWhitespace() }
            output += leadingWhitespace + sanitizedLine
            if (sanitizedLine != trimmed) {
                removedAny = true
            }
        }

        if (!removedAny) {
            return responseText
        }
        return output.joinToString("\n")
            .replace(Regex("""\n{3,}"""), "\n\n")
            .trimEnd()
    }

    fun ensureFlightListContent(
        responseText: String,
        queryText: String
    ): String {
        if (!looksLikeFlightQuery(queryText) && !looksLikeFlightContent(responseText)) {
            return responseText
        }
        if (responseContainsFlightList(responseText)) {
            return responseText
        }

        val rows = extractFlightListRows(responseText)
        if (rows.isEmpty()) {
            return responseText
        }

        val tableBlock = buildFlightComparisonBlock(rows)
        val normalized = responseText.trimEnd()
        val quickActionsMatch = Regex("""(?im)^\s*Quick\s*Actions\s*$""").find(normalized)
        if (quickActionsMatch == null) {
            return "$normalized\n\n$tableBlock"
        }

        val head = normalized.substring(0, quickActionsMatch.range.first).trimEnd()
        val tail = normalized.substring(quickActionsMatch.range.first).trimStart()
        return buildString {
            append(head)
            append("\n\n")
            append(tableBlock)
            append("\n\n")
            append(tail)
        }.trimEnd()
    }

    fun isFlightMediaCandidateLine(line: String): Boolean {
        val trimmed = line.trim()
        val lower = trimmed.lowercase(Locale.US)
        if (lower.startsWith("media:") ||
            lower.startsWith("image:") ||
            lower.startsWith("icon:") ||
            lower.startsWith("images:") ||
            lower.startsWith("icons:") ||
            lower.startsWith("assets:") ||
            lower.startsWith("asset:") ||
            lower.startsWith("logo:") ||
            lower.startsWith("logos:")
        ) {
            return true
        }
        return trimmed.contains("Image=", ignoreCase = true) ||
            trimmed.contains("Icon=", ignoreCase = true) ||
            Regex("""!\[[^\]]*]\((https?://\S+|/assets/\S+|assets/\S+)\)""").containsMatchIn(trimmed)
    }

    fun sanitizeFlightMediaCandidateLine(line: String): String? {
        val trimmed = line.trim()
        val lower = trimmed.lowercase(Locale.US)
        if (lower in setOf("media:", "images:", "image:", "icons:", "icon:", "assets:", "asset:", "logos:", "logo:")) {
            return null
        }

        val imageUrl = Regex("""(?i)\bImage\s*=\s*(https?://\S+|/assets/\S+|assets/\S+|\S+)""")
            .find(trimmed)
            ?.groupValues
            ?.getOrNull(1)
            ?.let(::sanitizeMediaUrlToken)
        val iconUrl = Regex("""(?i)\bIcon\s*=\s*(https?://\S+|/assets/\S+|assets/\S+|\S+)""")
            .find(trimmed)
            ?.groupValues
            ?.getOrNull(1)
            ?.let(::sanitizeMediaUrlToken)
        if (imageUrl != null || iconUrl != null) {
            val parts = mutableListOf<String>()
            if (imageUrl != null && isAllowedFlightMediaUrl(imageUrl, mediaType = "image", line = trimmed)) {
                parts += "Image=$imageUrl"
            }
            if (iconUrl != null && isAllowedFlightMediaUrl(iconUrl, mediaType = "icon", line = trimmed)) {
                parts += "Icon=$iconUrl"
            }
            return if (parts.isEmpty()) null else "Media: " + parts.joinToString(" ")
        }

        val imageColon = Regex("""(?i)^\s*Image\s*:\s*(https?://\S+|/assets/\S+|assets/\S+|\S+)\s*$""")
            .find(trimmed)
            ?.groupValues
            ?.getOrNull(1)
            ?.let(::sanitizeMediaUrlToken)
        if (imageColon != null) {
            return if (isAllowedFlightMediaUrl(imageColon, mediaType = "image", line = trimmed)) {
                "Image: $imageColon"
            } else {
                null
            }
        }

        val iconColon = Regex("""(?i)^\s*Icon\s*:\s*(https?://\S+|/assets/\S+|assets/\S+|\S+)\s*$""")
            .find(trimmed)
            ?.groupValues
            ?.getOrNull(1)
            ?.let(::sanitizeMediaUrlToken)
        if (iconColon != null) {
            return if (isAllowedFlightMediaUrl(iconColon, mediaType = "icon", line = trimmed)) {
                "Icon: $iconColon"
            } else {
                null
            }
        }

        val markdownMatch = Regex("""!\[([^\]]*)]\((https?://\S+|/assets/\S+|assets/\S+)\)""")
            .find(trimmed)
        if (markdownMatch != null) {
            val label = markdownMatch.groupValues.getOrNull(1).orEmpty()
            val mediaUrl = sanitizeMediaUrlToken(markdownMatch.groupValues.getOrNull(2).orEmpty())
            return if (isAllowedFlightMediaUrl(mediaUrl, mediaType = "image", line = "$label $trimmed")) {
                trimmed
            } else {
                null
            }
        }

        val labelUrlLine = Regex("""^\s*([^:]{1,80})\s*:\s*(https?://\S+|/assets/\S+|assets/\S+|\S+)\s*$""")
            .find(trimmed)
        if (labelUrlLine != null) {
            val label = labelUrlLine.groupValues.getOrNull(1).orEmpty().trim()
            val mediaUrl = sanitizeMediaUrlToken(labelUrlLine.groupValues.getOrNull(2).orEmpty())
            return if (isAllowedFlightMediaUrl(mediaUrl, mediaType = "image", line = "$label:")) {
                "$label: $mediaUrl"
            } else {
                null
            }
        }
        return null
    }

    fun isAllowedFlightMediaUrl(url: String, mediaType: String, line: String): Boolean {
        if (!looksLikeUsableInlineMediaUrl(url)) {
            return false
        }
        val normalizedUrl = sanitizeMediaUrlToken(url).lowercase(Locale.US)
        val normalizedLine = line.lowercase(Locale.US)
        val blockedTokens = listOf(
            "tokyo", "kyoto", "phuket", "weather", "cloud", "rain", "sun", "sunny",
            "beach", "temple", "noodles", "restaurant", "food", "cat", "dog", "monkey"
        )
        if (blockedTokens.any { normalizedUrl.contains(it) || normalizedLine.contains(it) }) {
            return false
        }

        val airlineTokens = listOf(
            "airline", "airlines", "airindia", "air-india", "airindiaexpress",
            "goindigo", "indigo", "akasa", "spicejet", "vistara", "emirates",
            "qatarairways", "britishairways", "delta", "united", "lufthansa"
        )
        val hasAirlineHint = airlineTokens.any { normalizedUrl.contains(it) || normalizedLine.contains(it) }
        val hasLogoHint = normalizedUrl.contains("logo") ||
            normalizedLine.contains("logo") ||
            normalizedLine.contains("airline") ||
            normalizedLine.contains("carrier")

        val isBootstrapFlightIcon = normalizedUrl.contains("bootstrap-icons") &&
            (normalizedUrl.contains("airplane") ||
                normalizedUrl.contains("ticket") ||
                normalizedUrl.contains("clock") ||
                normalizedUrl.contains("calendar"))

        return when (mediaType.lowercase(Locale.US)) {
            "icon" -> hasAirlineHint || hasLogoHint || isBootstrapFlightIcon
            "image" -> hasAirlineHint || hasLogoHint
            else -> hasAirlineHint || hasLogoHint
        }
    }

    // ── Travel functions ───────────────────────────────────────────────

    fun ensureTravelInlineMedia(
        responseText: String,
        queryText: String
    ): String {
        if (!looksLikeTravelQuery(queryText) && !looksLikeTravelContent(responseText)) {
            return responseText
        }
        if (hasTravelMediaCoverage(responseText)) {
            return responseText
        }

        val normalized = responseText.replace("\r\n", "\n").trim()
        if (normalized.isBlank()) {
            return responseText
        }

        val lines = normalized.split('\n')
        val output = mutableListOf<String>()
        val locationKeyword = extractTravelLocationKeyword(queryText)
        var inserted = 0
        val maxInsertions = 4

        lines.forEachIndexed { index, rawLine ->
            val line = rawLine.trimEnd()
            output += rawLine

            if (inserted >= maxInsertions) {
                return@forEachIndexed
            }

            val trimmed = line.trim()
            if (!shouldAttachTravelMediaAfterLine(trimmed)) {
                return@forEachIndexed
            }
            if (hasNearbyMediaLine(lines, index)) {
                return@forEachIndexed
            }

            val mediaLine = buildTravelMediaLine(
                line = trimmed,
                locationKeyword = locationKeyword
            )
            output += mediaLine
            inserted += 1
        }

        if (inserted == 0) {
            val mediaLine = buildTravelMediaLine(
                line = queryText,
                locationKeyword = locationKeyword
            )
            return buildString {
                append(normalized)
                append("\n\n")
                append(mediaLine)
            }
        }

        return output.joinToString(separator = "\n").trimEnd()
    }

    fun sanitizeTravelInlineMedia(
        responseText: String,
        queryText: String
    ): String {
        if (!looksLikeTravelQuery(queryText) && !looksLikeTravelContent(responseText)) {
            return responseText
        }
        val normalized = responseText.replace("\r\n", "\n")
        if (normalized.isBlank()) {
            return responseText
        }

        val locationKeyword = extractTravelLocationKeyword(queryText)
        val lines = normalized.split('\n')
        val output = mutableListOf<String>()
        var changed = false
        var contextLine = queryText

        lines.forEach { rawLine ->
            val trimmed = rawLine.trim()
            if (trimmed.isBlank()) {
                output += rawLine
                return@forEach
            }
            if (!isTravelMediaCandidateLine(trimmed)) {
                if (!trimmed.startsWith("Action:", ignoreCase = true) &&
                    !trimmed.startsWith("Source", ignoreCase = true) &&
                    !trimmed.startsWith("Quick Actions", ignoreCase = true)
                ) {
                    contextLine = trimmed
                }
                output += rawLine
                return@forEach
            }

            val sanitized = sanitizeTravelMediaCandidateLine(
                line = trimmed,
                contextLine = contextLine,
                locationKeyword = locationKeyword
            )
            if (sanitized.isNullOrBlank()) {
                changed = true
                return@forEach
            }
            val leadingWhitespace = rawLine.takeWhile { it.isWhitespace() }
            output += leadingWhitespace + sanitized
            if (sanitized != trimmed) {
                changed = true
            }
        }

        if (!changed) {
            return responseText
        }
        return output.joinToString("\n")
            .replace(Regex("""\n{3,}"""), "\n\n")
            .trimEnd()
    }

    fun looksLikeTravelQuery(queryText: String): Boolean {
        val normalized = queryText.lowercase(Locale.US)
        return normalized.contains("travel") ||
            normalized.contains("trip") ||
            normalized.contains("itinerary") ||
            normalized.contains("places to visit") ||
            normalized.contains("things to do") ||
            normalized.contains("attractions") ||
            normalized.contains("visit")
    }

    fun looksLikeTravelContent(text: String): Boolean {
        val normalized = text.lowercase(Locale.US)
        return normalized.contains("day 1") ||
            normalized.contains("itinerary") ||
            normalized.contains("places to visit") ||
            normalized.contains("things to do") ||
            normalized.contains("attraction") ||
            normalized.contains("must-visit")
    }

    fun isTravelMediaCandidateLine(line: String): Boolean {
        val trimmed = line.trim()
        val lower = trimmed.lowercase(Locale.US)
        if (lower.startsWith("media:") ||
            lower.startsWith("image:") ||
            lower.startsWith("icon:") ||
            lower.startsWith("images:") ||
            lower.startsWith("icons:") ||
            lower.startsWith("assets:") ||
            lower.startsWith("asset:")
        ) {
            return true
        }
        return trimmed.contains("Image=", ignoreCase = true) ||
            trimmed.contains("Icon=", ignoreCase = true) ||
            Regex("""!\[[^\]]*]\((https?://\S+|/assets/\S+|assets/\S+)\)""").containsMatchIn(trimmed)
    }

    fun sanitizeTravelMediaCandidateLine(
        line: String,
        contextLine: String,
        locationKeyword: String
    ): String? {
        val trimmed = line.trim()
        val lower = trimmed.lowercase(Locale.US)
        if (lower in setOf("media:", "images:", "image:", "icons:", "icon:", "assets:", "asset:")) {
            return null
        }

        val imageUrl = Regex("""(?i)\bImage\s*=\s*(https?://\S+|/assets/\S+|assets/\S+|\S+)""")
            .find(trimmed)
            ?.groupValues
            ?.getOrNull(1)
            ?.let(::sanitizeMediaUrlToken)
        val iconUrl = Regex("""(?i)\bIcon\s*=\s*(https?://\S+|/assets/\S+|assets/\S+|\S+)""")
            .find(trimmed)
            ?.groupValues
            ?.getOrNull(1)
            ?.let(::sanitizeMediaUrlToken)
        val imageColonUrl = Regex("""(?i)^\s*Image\s*:\s*(https?://\S+|/assets/\S+|assets/\S+|\S+)\s*$""")
            .find(trimmed)
            ?.groupValues
            ?.getOrNull(1)
            ?.let(::sanitizeMediaUrlToken)
        val iconColonUrl = Regex("""(?i)^\s*Icon\s*:\s*(https?://\S+|/assets/\S+|assets/\S+|\S+)\s*$""")
            .find(trimmed)
            ?.groupValues
            ?.getOrNull(1)
            ?.let(::sanitizeMediaUrlToken)
        val markdownUrl = Regex("""!\[[^\]]*]\((https?://\S+|/assets/\S+|assets/\S+)\)""")
            .find(trimmed)
            ?.groupValues
            ?.getOrNull(1)
            ?.let(::sanitizeMediaUrlToken)
        val labeledUrl = Regex("""^\s*([A-Za-z][A-Za-z\s_-]{1,30})\s*:\s*(https?://\S+|/assets/\S+|assets/\S+|\S+)\s*$""")
            .find(trimmed)

        val hasImageSignal = imageUrl != null ||
            imageColonUrl != null ||
            markdownUrl != null ||
            (
                labeledUrl != null &&
                    labeledUrl.groupValues
                        .getOrNull(1)
                        .orEmpty()
                        .lowercase(Locale.US)
                        .contains(Regex("""(?i)\b(image|photo|pic|picture|cover|banner)\b"""))
                )
        val iconCandidate = iconUrl ?: iconColonUrl

        val travelIconUrl = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/${pickTravelIconName(contextLine)}.svg"
        val rewrittenImage = if (hasImageSignal) {
            travelIconUrl
        } else {
            null
        }
        val rewrittenIcon = if (!iconCandidate.isNullOrBlank() && looksLikeUsableInlineMediaUrl(iconCandidate)) {
            iconCandidate
        } else {
            travelIconUrl
        }

        if (rewrittenImage == null && rewrittenIcon.isBlank()) {
            return null
        }
        return when {
            rewrittenImage != null -> "Media: Image=$rewrittenImage Icon=$rewrittenIcon"
            else -> "Media: Icon=$rewrittenIcon"
        }
    }

    fun hasNearbyMediaLine(lines: List<String>, index: Int): Boolean {
        val start = maxOf(0, index)
        val end = minOf(lines.lastIndex, index + 2)
        for (cursor in start..end) {
            val candidate = lines[cursor].trim()
            if (candidate.isBlank()) {
                continue
            }
            if (candidate.startsWith("Media:", ignoreCase = true)) {
                return true
            }
            if (candidate.contains("Image=", ignoreCase = true) || candidate.contains("Icon=", ignoreCase = true)) {
                return true
            }
        }
        return false
    }

    fun shouldAttachTravelMediaAfterLine(line: String): Boolean {
        if (line.isBlank()) {
            return false
        }
        val normalized = line
            .trim()
            .replace(Regex("""^#+\s*"""), "")
        if (containsUrlLikeToken(normalized)) {
            return false
        }
        if (normalized.startsWith("Media:", ignoreCase = true) ||
            normalized.startsWith("Action:", ignoreCase = true) ||
            normalized.startsWith("Source", ignoreCase = true) ||
            normalized.startsWith("Sources", ignoreCase = true) ||
            normalized.startsWith("Quick Actions", ignoreCase = true)
        ) {
            return false
        }
        if (Regex("""(?i)^(option\s*\d+|day\s*\d+|place\s*\d+|stop\s*\d+|attraction\s*\d+)\s*[:\-]""").containsMatchIn(normalized)) {
            return true
        }
        if (normalized.contains('|')) {
            return false
        }
        if (normalized.startsWith("-") || normalized.startsWith("\u2022")) {
            return false
        }

        val wordCount = normalized.split(Regex("""\s+""")).count { it.isNotBlank() }
        if (wordCount in 2..12 && normalized.length <= 96 && !normalized.endsWith(".")) {
            return true
        }
        return Regex("""(?i)^(day\s*\d+|place\s*\d+|stop\s*\d+|attraction\s*\d+)\b""")
            .containsMatchIn(normalized)
    }

    fun containsUrlLikeToken(value: String): Boolean {
        val normalized = value.lowercase(Locale.US)
        return normalized.contains("http://") ||
            normalized.contains("https://") ||
            Regex("""(?i)\b(?:www\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}\b""")
                .containsMatchIn(value)
    }

    fun buildTravelMediaLine(
        line: String,
        locationKeyword: String
    ): String {
        val iconName = pickTravelIconName(line)
        val iconUrl = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/$iconName.svg"
        return "Media: Image=$iconUrl Icon=$iconUrl"
    }

    fun buildTravelImageKeyword(line: String, locationKeyword: String): String {
        val stopwords = setOf(
            "the", "and", "for", "with", "from", "into", "your", "this", "that", "day",
            "place", "visit", "best", "top", "must", "to", "in", "of", "at", "on", "a", "an"
        )
        val words = line.lowercase(Locale.US)
            .replace(Regex("""[^a-z0-9\s-]"""), " ")
            .split(Regex("""\s+"""))
            .filter { it.length >= 3 && it !in stopwords }
            .take(3)
        val merged = buildList {
            add(locationKeyword)
            addAll(words)
        }
            .distinct()
            .joinToString(",")
            .ifBlank { "$locationKeyword,travel" }
        return URLEncoder.encode(merged, StandardCharsets.UTF_8.name())
            .replace("+", "%20")
    }

    fun pickTravelIconName(line: String): String {
        val normalized = line.lowercase(Locale.US)
        return when {
            normalized.contains("beach") || normalized.contains("island") || normalized.contains("sea") || normalized.contains("bay") -> "water"
            normalized.contains("temple") || normalized.contains("shrine") || normalized.contains("buddha") || normalized.contains("old town") -> "building"
            normalized.contains("market") || normalized.contains("food") || normalized.contains("street") -> "shop"
            normalized.contains("night") || normalized.contains("sunset") -> "moon-stars"
            normalized.contains("view") || normalized.contains("hike") || normalized.contains("trail") -> "signpost-split"
            normalized.contains("boat") || normalized.contains("pier") -> "geo-alt"
            else -> "geo-alt"
        }
    }

    fun hasTravelMediaCoverage(text: String): Boolean =
        hasInlineImageUrl(text) && hasInlineIconUrl(text)

    fun hasInlineImageUrl(text: String): Boolean {
        val imageAssignment = Regex(
            """(?im)\bImage\s*=\s*(https?://\S+|/assets/\S+|assets/\S+)"""
        )
            .findAll(text)
            .map { sanitizeMediaUrlToken(it.groupValues[1]) }
            .any { looksLikeUsableInlineMediaUrl(it) }
        if (imageAssignment) {
            return true
        }
        val imageColon = Regex(
            """(?im)^\s*Image\s*:\s*(https?://\S+|/assets/\S+|assets/\S+)"""
        )
            .findAll(text)
            .map { sanitizeMediaUrlToken(it.groupValues[1]) }
            .any { looksLikeUsableInlineMediaUrl(it) }
        return imageColon
    }

    fun hasInlineIconUrl(text: String): Boolean {
        val iconAssignment = Regex(
            """(?im)\bIcon\s*=\s*(https?://\S+|/assets/\S+|assets/\S+)"""
        )
            .findAll(text)
            .map { sanitizeMediaUrlToken(it.groupValues[1]) }
            .any { looksLikeUsableInlineMediaUrl(it) }
        if (iconAssignment) {
            return true
        }
        val iconColon = Regex(
            """(?im)^\s*Icon\s*:\s*(https?://\S+|/assets/\S+|assets/\S+)"""
        )
            .findAll(text)
            .map { sanitizeMediaUrlToken(it.groupValues[1]) }
            .any { looksLikeUsableInlineMediaUrl(it) }
        return iconColon
    }

    fun extractTravelLocationKeyword(queryText: String): String {
        val itineraryPattern = Regex("""(?i)\b\d+\s*day\s+([a-z][a-z0-9-]{2,30})\b""")
            .find(queryText)
            ?.groupValues
            ?.getOrNull(1)
            ?.lowercase(Locale.US)
        if (!itineraryPattern.isNullOrBlank()) {
            return itineraryPattern
        }

        val prepositionMatch = Regex("""(?i)\b(?:in|at|to|from|for)\s+([a-z][a-z0-9-]{2,30})\b""")
            .findAll(queryText)
            .lastOrNull()
            ?.groupValues
            ?.getOrNull(1)
            ?.lowercase(Locale.US)
        if (!prepositionMatch.isNullOrBlank()) {
            return prepositionMatch
        }

        val blocked = setOf(
            "show", "best", "top", "places", "visit", "travel", "trip", "itinerary",
            "things", "todo", "to", "in", "for", "with", "day", "days", "family",
            "activities", "activity", "quick", "action", "actions", "plan", "plans"
        )
        val token = queryText.lowercase(Locale.US)
            .split(Regex("""[^a-z0-9-]+"""))
            .firstOrNull { it.length >= 3 && it !in blocked }
        return token ?: "travel"
    }

    // ── General-purpose media injection ──────────────────────────────

    /**
     * Ensures every response has at least some inline media by injecting
     * icon URLs after section headings that lack a nearby Media line.
     * Icons are topic-matched via [pickGeneralIconName] so they are always
     * content-relevant. Runs after travel/flight-specific sanitizers so it
     * only fills remaining gaps.
     */
    fun ensureGeneralInlineMedia(
        responseText: String,
        queryText: String
    ): String {
        if (hasInlineImageUrl(responseText) || hasInlineIconUrl(responseText)) {
            return responseText
        }

        val normalized = responseText.replace("\r\n", "\n").trim()
        if (normalized.isBlank()) {
            return responseText
        }

        val topicKeyword = extractTopicKeyword(queryText)
        val lines = normalized.split('\n')
        val output = mutableListOf<String>()
        var inserted = 0
        val maxInsertions = 3

        lines.forEachIndexed { index, rawLine ->
            val line = rawLine.trimEnd()
            output += rawLine

            if (inserted >= maxInsertions) {
                return@forEachIndexed
            }

            val trimmed = line.trim()
            if (!isContentHeadingLine(trimmed)) {
                return@forEachIndexed
            }
            if (hasNearbyMediaLine(lines, index)) {
                return@forEachIndexed
            }

            val iconName = pickGeneralIconName(trimmed, topicKeyword)
            val iconUrl = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/$iconName.svg"
            output += "Media: Image=$iconUrl Icon=$iconUrl"
            inserted += 1
        }

        if (inserted == 0) {
            val iconName = pickGeneralIconName(queryText, topicKeyword)
            val iconUrl = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/$iconName.svg"
            return buildString {
                append(normalized)
                append("\nMedia: Image=$iconUrl Icon=$iconUrl")
            }
        }

        return output.joinToString(separator = "\n").trimEnd()
    }

    private fun isContentHeadingLine(line: String): Boolean {
        if (line.isBlank()) return false
        // Markdown heading
        if (line.startsWith("#")) return true
        // Numbered/named block headings like "Option 1:", "Day 1:", etc.
        if (Regex("""(?i)^(option|day|place|stop|step|item|category|section)\s*\d+\s*[:\-]""")
                .containsMatchIn(line)
        ) return true
        // Short title-like line (2-10 words, no trailing period, no URL, no bullet)
        val stripped = line.replace(Regex("""^#+\s*"""), "").trim()
        if (stripped.startsWith("-") || stripped.startsWith("•") || stripped.startsWith("|")) return false
        if (containsUrlLikeToken(stripped)) return false
        if (stripped.startsWith("Media:", ignoreCase = true) || stripped.startsWith("Action:", ignoreCase = true)) return false
        if (stripped.startsWith("Source", ignoreCase = true) || stripped.startsWith("Quick Actions", ignoreCase = true)) return false
        val wordCount = stripped.split(Regex("""\s+""")).count { it.isNotBlank() }
        return wordCount in 2..10 && stripped.length <= 80 && !stripped.endsWith(".")
    }

    fun extractTopicKeyword(queryText: String): String {
        val blocked = setOf(
            "show", "me", "what", "is", "are", "the", "best", "top", "how", "why",
            "tell", "about", "give", "find", "get", "list", "compare", "which",
            "please", "can", "you", "some", "any", "good", "great", "between",
            "and", "for", "with", "from", "this", "that", "should"
        )
        val tokens = queryText.lowercase(Locale.US)
            .replace(Regex("""[^a-z0-9\s-]"""), " ")
            .split(Regex("""\s+"""))
            .filter { it.length >= 3 && it !in blocked }
            .take(2)
        return if (tokens.isNotEmpty()) {
            tokens.joinToString("-")
        } else {
            "topic"
        }
    }

    private fun buildGeneralImageKeyword(line: String, topicKeyword: String): String {
        val stopwords = setOf(
            "the", "and", "for", "with", "from", "into", "your", "this", "that",
            "best", "top", "most", "key", "main", "overview", "summary", "comparison",
            "to", "in", "of", "at", "on", "a", "an"
        )
        val words = line.lowercase(Locale.US)
            .replace(Regex("""^#+\s*"""), "")
            .replace(Regex("""[^a-z0-9\s-]"""), " ")
            .split(Regex("""\s+"""))
            .filter { it.length >= 3 && it !in stopwords }
            .take(2)
        val merged = buildList {
            add(topicKeyword)
            addAll(words)
        }
            .distinct()
            .joinToString("-")
            .ifBlank { topicKeyword }
        return URLEncoder.encode(merged, StandardCharsets.UTF_8.name())
            .replace("+", "%20")
    }

    private fun pickGeneralIconName(line: String, topicKeyword: String): String {
        val normalized = (line + " " + topicKeyword).lowercase(Locale.US)
        return when {
            normalized.contains("weather") || normalized.contains("climate") || normalized.contains("temperature") -> "cloud-sun"
            normalized.contains("flight") || normalized.contains("airline") || normalized.contains("airport") -> "airplane"
            normalized.contains("hotel") || normalized.contains("stay") || normalized.contains("accommodation") -> "house"
            normalized.contains("food") || normalized.contains("restaurant") || normalized.contains("cuisine") -> "cup-hot"
            normalized.contains("shop") || normalized.contains("market") || normalized.contains("store") || normalized.contains("price") -> "shop"
            normalized.contains("sport") || normalized.contains("fitness") || normalized.contains("game") -> "trophy"
            normalized.contains("tech") || normalized.contains("computer") || normalized.contains("software") || normalized.contains("code") -> "laptop"
            normalized.contains("music") || normalized.contains("song") || normalized.contains("concert") -> "music-note"
            normalized.contains("book") || normalized.contains("read") || normalized.contains("study") || normalized.contains("learn") -> "book"
            normalized.contains("health") || normalized.contains("medical") || normalized.contains("doctor") -> "heart-pulse"
            normalized.contains("car") || normalized.contains("drive") || normalized.contains("vehicle") -> "car-front"
            normalized.contains("train") || normalized.contains("rail") || normalized.contains("metro") -> "train-front"
            normalized.contains("beach") || normalized.contains("sea") || normalized.contains("ocean") -> "water"
            normalized.contains("mountain") || normalized.contains("hike") || normalized.contains("trek") -> "signpost-split"
            normalized.contains("city") || normalized.contains("town") || normalized.contains("urban") -> "building"
            normalized.contains("nature") || normalized.contains("park") || normalized.contains("garden") -> "tree"
            normalized.contains("money") || normalized.contains("finance") || normalized.contains("invest") || normalized.contains("budget") -> "cash-stack"
            normalized.contains("travel") || normalized.contains("trip") || normalized.contains("visit") || normalized.contains("tour") -> "geo-alt"
            normalized.contains("time") || normalized.contains("schedule") || normalized.contains("plan") -> "calendar"
            else -> "star"
        }
    }

    fun looksLikeFlightQuery(queryText: String): Boolean {
        val normalized = queryText.lowercase(Locale.US)
        return normalized.contains("flight") ||
            normalized.contains("airline") ||
            normalized.contains("departure") ||
            normalized.contains("arrival")
    }

    fun looksLikeFlightContent(text: String): Boolean {
        val normalized = text.lowercase(Locale.US)
        return normalized.contains("airline") &&
            (normalized.contains("departure") || normalized.contains("arrival")) &&
            normalized.contains("fare")
    }

    fun responseContainsFlightList(text: String): Boolean {
        val normalized = text.replace("\r\n", "\n")
        val lines = normalized.lines().map { it.trim() }

        val headerIndex = lines.indexOfFirst { line ->
            Regex("""(?i)^airline\s*\|\s*departure\s*\|\s*arrival\s*\|\s*duration\s*\|\s*stops\s*\|\s*fare\s*$""")
                .matches(line)
        }
        if (headerIndex >= 0) {
            var rowCount = 0
            var cursor = headerIndex + 1
            while (cursor < lines.size) {
                val line = lines[cursor]
                if (line.isBlank()) break
                if (!line.contains('|')) break
                val cells = line.split('|').map { it.trim() }.filter { it.isNotBlank() }
                val separatorRow = cells.all { it.matches(Regex("""[-:]+""")) }
                if (!separatorRow && cells.size >= 6) {
                    rowCount++
                }
                cursor++
            }
            if (rowCount >= 2) {
                return true
            }
        }

        val pipeRows = lines.count { line ->
            if (line.isBlank() || !line.contains('|')) {
                return@count false
            }
            val cells = line.split('|').map { it.trim() }.filter { it.isNotBlank() }
            cells.size >= 6 && !line.startsWith("Action:", ignoreCase = true)
        }
        if (pipeRows >= 3) {
            return true
        }

        val optionCount = Regex("""(?im)^\s*Option\s+\d+\s*:""").findAll(normalized).count()
        return optionCount >= 2
    }

    // ── URL functions ──────────────────────────────────────────────────

    fun extractUrlsForQuickActions(text: String): List<String> {
        return URL_TOKEN_REGEX.findAll(text)
            .mapNotNull { normalizeExternalUrlCandidate(it.value) }
            .distinct()
            .toList()
    }

    fun normalizeUrlTokensForDisplay(text: String): String {
        if (text.isBlank()) {
            return text
        }
        return URL_TOKEN_REGEX.replace(text) { match ->
            normalizeExternalUrlCandidate(match.value) ?: match.value
        }
    }

    fun normalizeExternalUrlCandidate(value: String): String? {
        val token = value.trim().trim('"', '\'').trimEnd('.', ',', ';', ')', ']', '}')
        if (token.isBlank()) {
            return null
        }
        if (token.startsWith("http://", ignoreCase = true) || token.startsWith("https://", ignoreCase = true)) {
            return token
        }
        if (token.startsWith("//")) {
            return "https:$token"
        }
        if (!token.startsWith("www.", ignoreCase = true) &&
            !Regex("""(?i)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:[/?#].*)?""").matches(token)
        ) {
            return null
        }

        val host = token
            .removePrefix("www.")
            .substringBefore('/')
            .substringBefore('?')
            .substringBefore('#')
            .lowercase(Locale.US)
        if (!isLikelyPublicDomainHost(host)) {
            return null
        }
        return "https://$token"
    }

    fun isLikelyPublicDomainHost(host: String): Boolean {
        if (host.isBlank() || host.contains('_')) {
            return false
        }
        val labels = host.split('.').filter { it.isNotBlank() }
        if (labels.size < 2 || labels.any { !HOST_LABEL_REGEX.matches(it) }) {
            return false
        }
        val tld = labels.last().lowercase(Locale.US)
        if (!tld.all { it in 'a'..'z' } || tld.length !in 2..24) {
            return false
        }
        if (
            tld in setOf(
                "png", "jpg", "jpeg", "svg", "webp", "gif", "bmp", "ico",
                "json", "xml", "txt", "csv", "md", "pdf", "zip", "apk"
            )
        ) {
            return false
        }
        return true
    }

    fun quickActionLabelForUrl(url: String, index: Int): String {
        val host = runCatching { URL(url).host.lowercase(Locale.US) }.getOrDefault("")
        return when {
            host.contains("makemytrip") -> "Search on MakeMyTrip"
            host.contains("skyscanner") -> "Search on Skyscanner"
            host.contains("goibibo") -> "Search on Goibibo"
            host.contains("goindigo") -> "Open IndiGo"
            host.contains("airindia") -> "Open Air India"
            host.contains("akasaair") -> "Open Akasa Air"
            host.contains("flightsfrom") -> "Open FlightsFrom"
            else -> "Open Source ${index + 1}"
        }
    }

    private fun extractFlightListRows(text: String): List<FlightListRow> {
        val normalized = text.replace("\r\n", "\n")
        val rows = mutableListOf<FlightListRow>()
        val lines = normalized.lines().map { it.trim() }.filter { it.isNotBlank() }

        lines.forEach { line ->
            if (line.contains('|')) {
                val cells = line.split('|').map { it.trim() }.filter { it.isNotBlank() }
                if (cells.size >= 6 && !cells[0].equals("airline", ignoreCase = true)) {
                    val airline = normalizeAirlineName(cells[0])
                    val candidate = FlightListRow(
                        airline = airline,
                        departure = cells.getOrNull(1),
                        arrival = cells.getOrNull(2),
                        duration = cells.getOrNull(3),
                        stops = cells.getOrNull(4),
                        fare = cells.getOrNull(5)
                    )
                    if (hasFlightValueSignal(candidate)) {
                        rows += candidate
                    }
                }
                return@forEach
            }

            val airlineMatches = FLIGHT_AIRLINE_REGEX.findAll(line).toList()
            if (airlineMatches.isEmpty()) {
                return@forEach
            }
            val times = FLIGHT_TIME_REGEX.findAll(line).map { it.value.uppercase(Locale.US) }.toList()
            val duration = FLIGHT_DURATION_REGEX.find(line)?.value
            val stops = FLIGHT_STOPS_REGEX.find(line)?.value
            val fare = FLIGHT_FARE_REGEX.find(line)?.value
            airlineMatches.forEach { airlineMatch ->
                rows += FlightListRow(
                    airline = normalizeAirlineName(airlineMatch.value),
                    departure = times.getOrNull(0),
                    arrival = times.getOrNull(1),
                    duration = duration,
                    stops = stops,
                    fare = fare
                )
            }
        }

        val withSignals = rows.filter(::hasFlightValueSignal)
            .distinctBy { row ->
                listOf(
                    row.airline.lowercase(Locale.US),
                    row.departure.orEmpty().lowercase(Locale.US),
                    row.arrival.orEmpty().lowercase(Locale.US),
                    row.fare.orEmpty().lowercase(Locale.US)
                ).joinToString("|")
            }

        if (withSignals.isNotEmpty()) {
            return withSignals.take(6)
        }

        return FLIGHT_AIRLINE_REGEX.findAll(normalized)
            .map { normalizeAirlineName(it.value) }
            .distinct()
            .take(6)
            .map { airline ->
                FlightListRow(
                    airline = airline,
                    departure = null,
                    arrival = null,
                    duration = null,
                    stops = null,
                    fare = null
                )
            }
            .toList()
    }

    private fun hasFlightValueSignal(row: FlightListRow): Boolean {
        return !row.fare.isNullOrBlank() ||
            !row.duration.isNullOrBlank() ||
            !row.stops.isNullOrBlank() ||
            !row.departure.isNullOrBlank() ||
            !row.arrival.isNullOrBlank()
    }

    private fun normalizeAirlineName(value: String): String {
        val normalized = value.trim().lowercase(Locale.US)
        return when {
            normalized.contains("air india express") -> "Air India Express"
            normalized == "air india" || normalized.startsWith("air india ") -> "Air India"
            normalized.contains("akasa") -> "Akasa Air"
            normalized.contains("indigo") -> "IndiGo"
            normalized.contains("spicejet") -> "SpiceJet"
            normalized.contains("vistara") -> "Vistara"
            normalized.contains("british airways") -> "British Airways"
            normalized.contains("qatar") -> "Qatar Airways"
            normalized.contains("lufthansa") -> "Lufthansa"
            normalized.contains("emirates") -> "Emirates"
            normalized.contains("alliance air") -> "Alliance Air"
            normalized.contains("flydubai") -> "flydubai"
            normalized.contains("etihad") -> "Etihad Airways"
            normalized.contains("go first") || normalized.contains("goair") -> "Go First"
            normalized.contains("delta") -> "Delta Air Lines"
            normalized.contains("united") -> "United Airlines"
            else -> value.trim()
        }
    }

    private fun buildFlightComparisonBlock(rows: List<FlightListRow>): String {
        return buildString {
            append("Flight Comparison\n")
            append("Airline | Departure | Arrival | Duration | Stops | Fare\n")
            rows.forEach { row ->
                append(
                    listOf(
                        row.airline.ifBlank { "Airline" },
                        row.departure.orEmpty().ifBlank { "--" },
                        row.arrival.orEmpty().ifBlank { "--" },
                        row.duration.orEmpty().ifBlank { "--" },
                        row.stops.orEmpty().ifBlank { "--" },
                        row.fare.orEmpty().ifBlank { "--" }
                    ).joinToString(" | ")
                )
                append('\n')
            }
        }.trimEnd()
    }

    // ── GenUI validation ───────────────────────────────────────────────

    fun responseContainsInlineMedia(text: String): Boolean {
        val assignmentRegex = Regex(
            """(?im)\b(?:Media:\s*)?(?:Image|Icon)\s*=\s*(https?://\S+|/assets/\S+|assets/\S+|\S+)"""
        )
        val colonRegex = Regex(
            """(?im)^\s*(?:Image|Icon)\s*:\s*(https?://\S+|/assets/\S+|assets/\S+|\S+)"""
        )
        val candidates = mutableListOf<String>()
        assignmentRegex.findAll(text).forEach { match ->
            candidates += sanitizeMediaUrlToken(match.groupValues[1])
        }
        colonRegex.findAll(text).forEach { match ->
            candidates += sanitizeMediaUrlToken(match.groupValues[1])
        }
        return candidates.any(::looksLikeUsableInlineMediaUrl)
    }

    fun genUiPreservesInlineMedia(jsonText: String): Boolean {
        return genUiPreservesInlineImages(jsonText) || genUiPreservesInlineIcons(jsonText)
    }

    fun genUiPreservesInlineImages(jsonText: String): Boolean {
        return Regex(
            """"component"\s*:\s*"Image"[\s\S]{0,320}"(?:url|src|source|image)"\s*:\s*"(?:https?://|/assets/|assets/)"""",
            setOf(RegexOption.IGNORE_CASE)
        ).containsMatchIn(jsonText) ||
            Regex(
                """"component"\s*:\s*"Image"[\s\S]{0,420}"(?:url|src|source|image)"\s*:\s*\{\s*"literalString"\s*:\s*"(?:https?://|/assets/|assets/)"""",
                setOf(RegexOption.IGNORE_CASE)
            ).containsMatchIn(jsonText)
    }

    fun genUiPreservesInlineIcons(jsonText: String): Boolean {
        return Regex(
            """"component"\s*:\s*"Icon"[\s\S]{0,240}"(?:url|icon|name|glyph|asset)"\s*:\s*"[^"]+"""",
            setOf(RegexOption.IGNORE_CASE)
        ).containsMatchIn(jsonText) ||
            Regex(
                """"component"\s*:\s*"Icon"[\s\S]{0,320}"(?:url|icon|name|glyph|asset)"\s*:\s*\{\s*"literalString"\s*:\s*"[^"]+"""",
                setOf(RegexOption.IGNORE_CASE)
            ).containsMatchIn(jsonText)
    }

    fun responseContainsActionButtons(text: String): Boolean {
        return Regex(
            """(?im)^\s*Action:\s*\[Button:\s*.+?\]\s*(?:https?://|//|www\.|(?:[a-z0-9-]+\.)+[a-z]{2,24})\S*"""
        )
            .containsMatchIn(text)
    }

    fun genUiPreservesActionButtons(jsonText: String): Boolean {
        return Regex("""(?i)"call"\s*:\s*"openUrl"""").containsMatchIn(jsonText) ||
            (
                Regex("""(?i)"component"\s*:\s*"Button"""").containsMatchIn(jsonText) &&
                    Regex("""(?i)"url"\s*:\s*"https?://""").containsMatchIn(jsonText)
                )
    }

    private fun extractFirstInlineImageUrl(text: String): String? {
        val candidates = mutableListOf<String>()
        Regex("""(?im)\bImage\s*=\s*(https?://\S+|/assets/\S+|assets/\S+)""")
            .findAll(text)
            .forEach { candidates += sanitizeMediaUrlToken(it.groupValues[1]) }
        Regex("""(?im)^\s*Image\s*:\s*(https?://\S+|/assets/\S+|assets/\S+)""")
            .findAll(text)
            .forEach { candidates += sanitizeMediaUrlToken(it.groupValues[1]) }
        Regex("""!\[[^\]]*]\((https?://\S+|/assets/\S+|assets/\S+)\)""")
            .findAll(text)
            .forEach { candidates += sanitizeMediaUrlToken(it.groupValues[1]) }
        return candidates.firstOrNull(::looksLikeUsableInlineMediaUrl)
    }

    private fun extractFirstInlineIconUrl(text: String): String? {
        val candidates = mutableListOf<String>()
        Regex("""(?im)\bIcon\s*=\s*(https?://\S+|/assets/\S+|assets/\S+)""")
            .findAll(text)
            .forEach { candidates += sanitizeMediaUrlToken(it.groupValues[1]) }
        Regex("""(?im)^\s*Icon\s*:\s*(https?://\S+|/assets/\S+|assets/\S+)""")
            .findAll(text)
            .forEach { candidates += sanitizeMediaUrlToken(it.groupValues[1]) }
        return candidates.firstOrNull(::looksLikeUsableInlineMediaUrl)
    }

    fun ensureGenUiHasImageComponent(
        jsonText: String,
        stage2Response: String,
        queryText: String
    ): String {
        val parsed = runCatching { JsonParser.parseString(jsonText) }.getOrNull() ?: return jsonText
        val payload = normalizeGenUiPayload(parsed)
        if (!payload.isJsonArray) {
            return jsonText
        }

        val messages = payload.asJsonArray
        var components: JsonArray? = null
        messages.forEach { message ->
            if (!message.isJsonObject) return@forEach
            val update = message.asJsonObject.getAsJsonObject("updateComponents") ?: return@forEach
            val candidate = update.get("components")
            if (candidate != null && candidate.isJsonArray) {
                components = candidate.asJsonArray
            }
        }
        val componentList = components ?: return jsonText
        val topicMediaUrl = buildTopicFallbackMediaUrl(queryText)
        val isFlight = looksLikeFlightQuery(queryText) || looksLikeFlightContent(stage2Response)
        var changed = false

        componentList.forEach { component ->
            if (!component.isJsonObject) return@forEach
            val obj = component.asJsonObject
            if (!jsonStringOrNull(obj.get("component")).equals("Image", ignoreCase = true)) {
                return@forEach
            }
            val currentUrl = jsonStringOrNull(obj.get("url")).orEmpty()
            if (isFlight) {
                if (!currentUrl.equals(topicMediaUrl, ignoreCase = true)) {
                    obj.addProperty("url", topicMediaUrl)
                    changed = true
                }
            } else if (!looksLikeUsableInlineMediaUrl(currentUrl)) {
                obj.addProperty("url", topicMediaUrl)
                changed = true
            }
        }

        val alreadyHasImage = componentList.any { component ->
            component.isJsonObject &&
                jsonStringOrNull(component.asJsonObject.get("component")).equals("Image", ignoreCase = true) &&
                looksLikeUsableInlineMediaUrl(jsonStringOrNull(component.asJsonObject.get("url")).orEmpty())
        }
        if (alreadyHasImage) {
            return if (changed) messages.toString() else jsonText
        }

        val mediaUrl = if (looksLikeTravelQuery(queryText) || looksLikeTravelContent(stage2Response)) {
            extractFirstInlineImageUrl(stage2Response)
                ?: extractFirstInlineIconUrl(stage2Response)
                ?: topicMediaUrl
        } else {
            topicMediaUrl
        }
        if (!looksLikeUsableInlineMediaUrl(mediaUrl)) {
            return jsonText
        }

        val rootComponent = componentList.firstOrNull { component ->
            component.isJsonObject &&
                jsonStringOrNull(component.asJsonObject.get("id")) == "root"
        }?.asJsonObject ?: return jsonText

        val mediaId = buildUniqueComponentId(componentList, base = "auto_media")
        componentList.add(
            JsonObject().apply {
                addProperty("id", mediaId)
                addProperty("component", "Image")
                addProperty("url", mediaUrl)
                addProperty("fit", "cover")
            }
        )
        prependRootChildId(rootComponent, mediaId)
        return messages.toString()
    }

    private fun buildUniqueComponentId(components: JsonArray, base: String): String {
        val existing = components.mapNotNull { component ->
            if (!component.isJsonObject) return@mapNotNull null
            jsonStringOrNull(component.asJsonObject.get("id"))
        }.toSet()
        var index = 1
        while (true) {
            val candidate = "${base}_$index"
            if (candidate !in existing) {
                return candidate
            }
            index += 1
        }
    }

    private fun prependRootChildId(root: JsonObject, childId: String) {
        val existingChildren = root.get("children")
        if (existingChildren != null && existingChildren.isJsonArray) {
            val current = existingChildren.asJsonArray.mapNotNull { child ->
                if (child.isJsonPrimitive && child.asJsonPrimitive.isString) child.asString else null
            }
            if (current.contains(childId)) {
                return
            }
            val updated = JsonArray().apply {
                add(childId)
                current.forEach { add(it) }
            }
            root.add("children", updated)
            return
        }

        val singleChild = root.get("child")?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
        if (!singleChild.isNullOrBlank()) {
            root.remove("child")
            root.add("children", JsonArray().apply {
                add(childId)
                add(singleChild)
            })
            return
        }

        root.add("children", JsonArray().apply { add(childId) })
    }

    private fun jsonStringOrNull(element: JsonElement?): String? {
        if (element == null || element.isJsonNull) {
            return null
        }
        if (element.isJsonPrimitive && element.asJsonPrimitive.isString) {
            return element.asString
        }
        if (element.isJsonObject) {
            val literal = element.asJsonObject.get("literalString")
            if (literal != null && literal.isJsonPrimitive && literal.asJsonPrimitive.isString) {
                return literal.asString
            }
        }
        return null
    }

    private fun buildTopicFallbackMediaUrl(queryText: String): String {
        if (looksLikeFlightQuery(queryText)) {
            return FLIGHT_FALLBACK_ICON_URL
        }
        val topicKeyword = extractTopicKeyword(queryText)
        val iconName = pickGeneralIconName(queryText, topicKeyword)
        return "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/$iconName.svg"
    }

    fun rewriteUnstableMediaHostsInGenUi(
        jsonText: String,
        queryText: String
    ): String {
        val replacement = buildTopicFallbackMediaUrl(queryText)
        return Regex(
            """https?://(?:www\.)?(?:loremflickr\.com|picsum\.photos|placehold\.co|dummyimage\.com)\S*""",
            RegexOption.IGNORE_CASE
        ).replace(jsonText, replacement)
    }

    fun normalizeFlightMediaInGenUi(
        jsonText: String,
        queryText: String,
        stage2Response: String
    ): String {
        if (!looksLikeFlightQuery(queryText) && !looksLikeFlightContent(stage2Response)) {
            return jsonText
        }
        val bootstrapPattern = Regex(
            """https?://(?:cdn\.jsdelivr\.net/npm|unpkg\.com)/bootstrap-icons(?:@[^/]+)?/icons/([a-z0-9-]+)\.svg""",
            RegexOption.IGNORE_CASE
        )
        var normalized = bootstrapPattern.replace(jsonText) { match ->
            val iconName = match.groupValues.getOrNull(1).orEmpty().lowercase(Locale.US)
            if (iconName in FLIGHT_ALLOWED_BOOTSTRAP_ICONS) {
                match.value
            } else {
                FLIGHT_FALLBACK_ICON_URL
            }
        }
        normalized = Regex(
            """https?://[^\s"'\\]*(?:weatherapi\.com|openweathermap\.org|accuweather\.com|weather\.com)[^\s"'\\]*""",
            RegexOption.IGNORE_CASE
        ).replace(normalized, FLIGHT_FALLBACK_ICON_URL)
        normalized = Regex(
            """https?://[^\s"'\\]*(?:cloud|weather|rain|storm|snow|sunny|overcast)[^\s"'\\]*\.svg""",
            RegexOption.IGNORE_CASE
        ).replace(normalized, FLIGHT_FALLBACK_ICON_URL)
        return normalized
    }

    fun ensureGenUiHasInlineTextMedia(
        jsonText: String,
        queryText: String
    ): String {
        val parsed = runCatching { JsonParser.parseString(jsonText) }.getOrNull() ?: return jsonText
        val payload = normalizeGenUiPayload(parsed)
        if (!payload.isJsonArray) {
            return jsonText
        }

        val components = payload.asJsonArray
            .firstNotNullOfOrNull { message ->
                if (!message.isJsonObject) return@firstNotNullOfOrNull null
                val update = message.asJsonObject.getAsJsonObject("updateComponents") ?: return@firstNotNullOfOrNull null
                val candidate = update.get("components")
                if (candidate != null && candidate.isJsonArray) candidate.asJsonArray else null
            } ?: return jsonText

        val mediaUrl = buildTopicFallbackMediaUrl(queryText)
        var changed = false
        components.forEach { component ->
            if (changed || !component.isJsonObject) return@forEach
            val obj = component.asJsonObject
            if (!jsonStringOrNull(obj.get("component")).equals("Text", ignoreCase = true)) {
                return@forEach
            }

            val textElement = obj.get("text")
            val textValue = jsonStringOrNull(textElement).orEmpty()
            if (textValue.isBlank()) {
                return@forEach
            }
            if (textValue.contains("Media:", ignoreCase = true) || textValue.contains("Image=", ignoreCase = true)) {
                return@forEach
            }

            val injected = "Media: Image=$mediaUrl Icon=$mediaUrl\n$textValue"
            if (textElement != null && textElement.isJsonObject && textElement.asJsonObject.has("literalString")) {
                textElement.asJsonObject.addProperty("literalString", injected)
            } else {
                obj.addProperty("text", injected)
            }
            changed = true
        }

        return if (changed) payload.asJsonArray.toString() else jsonText
    }

    fun sanitizeMediaUrlToken(value: String): String =
        value.trim().trim('\'', '"').trimEnd('.', ',', ';', ')', ']')

    fun looksLikeUsableInlineMediaUrl(value: String): Boolean {
        val normalized = value.trim()
        if (normalized.isBlank()) {
            return false
        }
        val lower = normalized.lowercase(Locale.US)
        if (
            lower in setOf(
                "<image_url>",
                "<icon_url>",
                "<url>",
                "image_url",
                "icon_url",
                "url",
                "n/a",
                "na",
                "none",
                "null",
                "--"
            ) ||
            lower.contains("placeholder") ||
            lower.contains("<") ||
            lower.contains(">")
        ) {
            return false
        }
        if (lower.startsWith("/assets/") || lower.startsWith("assets/")) {
            return true
        }

        val pathWithoutQuery = lower.substringBefore('?').substringBefore('#')
        if (
            pathWithoutQuery.endsWith(".png") ||
            pathWithoutQuery.endsWith(".jpg") ||
            pathWithoutQuery.endsWith(".jpeg") ||
            pathWithoutQuery.endsWith(".svg") ||
            pathWithoutQuery.endsWith(".webp")
        ) {
            return true
        }

        val uri = runCatching { URI(normalized) }.getOrNull() ?: return false
        val scheme = uri.scheme?.lowercase(Locale.US) ?: return false
        if (scheme != "http" && scheme != "https") {
            return false
        }
        val host = uri.host?.lowercase(Locale.US).orEmpty()
        val path = uri.path?.lowercase(Locale.US).orEmpty()
        if (
            host.contains("cdn.jsdelivr.net") ||
            host.contains("raw.githubusercontent.com") ||
            host.contains("upload.wikimedia.org") ||
            host.contains("imgur.com") ||
            host.contains("gstatic.com") ||
            host.contains("twimg.com") ||
            host.contains("loremflickr.com") ||
            host.contains("picsum.photos")
        ) {
            return true
        }
        return path.contains("/icon") || path.contains("/icons/") || path.contains("/image") || path.contains("/images/")
    }

    // ── Payload ────────────────────────────────────────────────────────

    fun normalizeGenUiPayload(json: JsonElement): JsonElement {
        if (json.isJsonArray) {
            return json
        }
        if (!json.isJsonObject) {
            return json
        }

        val obj = json.asJsonObject
        val directArray = obj.get("genui_json")
        if (directArray != null && directArray.isJsonArray) {
            return directArray
        }
        val messages = obj.get("messages")
        if (messages != null && messages.isJsonArray) {
            return messages
        }
        val payload = obj.get("payload")
        if (payload != null) {
            if (payload.isJsonArray) {
                return payload
            }
            if (payload.isJsonObject) {
                val payloadObj = payload.asJsonObject
                val payloadMessages = payloadObj.get("messages")
                if (payloadMessages != null && payloadMessages.isJsonArray) {
                    return payloadMessages
                }
            }
        }
        return json
    }

    fun buildFallbackGenUi(stage2Response: String, catalogId: String): JsonArray {
        val textValue = stage2Response.trim().ifBlank { "No content generated." }
        val fallbackMediaUrl = extractFirstInlineImageUrl(stage2Response)
            ?: extractFirstInlineIconUrl(stage2Response)
        val surfaceId = "surface_live"
        return JsonArray().apply {
            add(
                JsonObject().apply {
                    addProperty("version", "v0.9")
                    add("createSurface", JsonObject().apply {
                        addProperty("surfaceId", surfaceId)
                        addProperty("catalogId", catalogId)
                    })
                }
            )
            add(
                JsonObject().apply {
                    addProperty("version", "v0.9")
                    add("updateComponents", JsonObject().apply {
                        addProperty("surfaceId", surfaceId)
                        add("components", JsonArray().apply {
                            add(JsonObject().apply {
                                addProperty("id", "root")
                                addProperty("component", "Column")
                                add("children", JsonArray().apply {
                                    if (!fallbackMediaUrl.isNullOrBlank()) {
                                        add("media_1")
                                    }
                                    add("text_1")
                                })
                            })
                            if (!fallbackMediaUrl.isNullOrBlank()) {
                                add(JsonObject().apply {
                                    addProperty("id", "media_1")
                                    addProperty("component", "Image")
                                    addProperty("url", fallbackMediaUrl)
                                    addProperty("fit", "cover")
                                })
                            }
                            add(JsonObject().apply {
                                addProperty("id", "text_1")
                                addProperty("component", "Text")
                                addProperty("variant", "body")
                                addProperty("text", textValue)
                            })
                        })
                    })
                }
            )
        }
    }

    fun resolveStage3CatalogId(prefs: SharedPreferences): String {
        val override = prefs
            .getString(PREF_STAGE3_CATALOG_ID, null)
            ?.trim()
            .orEmpty()
        return if (override.isNotBlank()) override else DEFAULT_STAGE3_CATALOG_ID
    }
}
