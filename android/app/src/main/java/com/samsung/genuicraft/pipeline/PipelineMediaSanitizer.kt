package com.samsung.genuicraft.pipeline

import android.content.Context
import android.content.SharedPreferences
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.security.SafeContentPolicy
import java.net.URL
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.util.Locale

internal object PipelineMediaSanitizer {

    val URL_TOKEN_REGEX = Regex(
        """(?i)(?:https?://|//)[^\s<>\]]+|(?<![@\w/\\])(?:www\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:[/?#][^\s<>\]]*)?"""
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

    data class ResponseLinkRepairResult(
        val jsonText: String,
        val sourcesAdded: Int,
        val actionsAdded: Int
    ) {
        val changed: Boolean get() = sourcesAdded > 0 || actionsAdded > 0
    }

    // -- Flight functions ---------------------------------------------------

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

    // -- Travel functions ---------------------------------------------------

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
        // Preserve real, usable API image URLs (e.g. MCP hotel thumbnails from googleusercontent.com).
        val candidateImageUrl = imageUrl ?: imageColonUrl ?: markdownUrl
        val rewrittenImage = if (hasImageSignal) {
            if (candidateImageUrl != null && looksLikeUsableInlineImageUrl(candidateImageUrl)) {
                candidateImageUrl
            } else {
                null
            }
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
        return "Media: Icon=$iconUrl"
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
            .any { looksLikeUsableInlineImageUrl(it) }
        if (imageAssignment) {
            return true
        }
        val imageColon = Regex(
            """(?im)^\s*Image\s*:\s*(https?://\S+|/assets/\S+|assets/\S+)"""
        )
            .findAll(text)
            .map { sanitizeMediaUrlToken(it.groupValues[1]) }
            .any { looksLikeUsableInlineImageUrl(it) }
        if (imageColon) {
            return true
        }
        return Regex("""!\[[^\]]*]\((https?://\S+|/assets/\S+|assets/\S+)\)""")
            .findAll(text)
            .map { sanitizeMediaUrlToken(it.groupValues[1]) }
            .any { looksLikeUsableInlineImageUrl(it) }
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
        if (!itineraryPattern.isNullOrBlank() &&
            itineraryPattern !in setOf("itinerary", "vacation", "travel", "trip", "holiday")
        ) {
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

    // -- General-purpose media injection ------------------------------------

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
            output += "Media: Icon=$iconUrl"
            inserted += 1
        }

        if (inserted == 0) {
            val iconName = pickGeneralIconName(queryText, topicKeyword)
            val iconUrl = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/$iconName.svg"
            return buildString {
                append(normalized)
                append("\nMedia: Icon=$iconUrl")
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
        if (stripped.startsWith("-") || stripped.startsWith("*") || stripped.startsWith("|")) return false
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

    // -- URL functions ------------------------------------------------------

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
            normalizeExternalUrlCandidate(match.value).orEmpty()
        }
    }

    fun normalizeExternalUrlCandidate(value: String): String? =
        SafeContentPolicy.sanitizeActionUrl(value)

    fun isLikelyPublicDomainHost(host: String): Boolean =
        SafeContentPolicy.isLikelyPublicDomainHost(host)

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

    // -- GenUI validation ---------------------------------------------------

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
        runCatching { JsonParser.parseString(jsonText) }.getOrNull()?.let { parsed ->
            val payload = normalizeGenUiPayload(parsed)
            if (payload.isJsonObject && FlatSpecContract.looksLikeFlatSpec(payload)) {
                val elements = payload.asJsonObject.getAsJsonObject("elements")
                if (elements != null && elements.entrySet().any { (_, node) ->
                        if (!node.isJsonObject) return@any false
                        val element = node.asJsonObject
                        jsonStringOrNull(element.get("type")).equals("Image", ignoreCase = true) &&
                            imageUrlFromProps(element.get("props")?.takeIf { it.isJsonObject }?.asJsonObject) != null
                    }
                ) {
                    return true
                }
            } else if (payload.isJsonArray) {
                val hasLegacyImage = payload.asJsonArray.any { message ->
                    if (!message.isJsonObject) return@any false
                    val update = message.asJsonObject.getAsJsonObject("updateComponents") ?: return@any false
                    val components = update.getAsJsonArray("components") ?: return@any false
                    components.asSequence().any { component ->
                        if (!component.isJsonObject) {
                            false
                        } else {
                            val obj = component.asJsonObject
                            jsonStringOrNull(obj.get("component")).equals("Image", ignoreCase = true) &&
                                listOf("url", "src", "source", "image")
                                    .mapNotNull { key -> jsonStringOrNull(obj.get(key)) }
                                    .any(::looksLikeUsableInlineImageUrl)
                        }
                    }
                }
                if (hasLegacyImage) {
                    return true
                }
            }
        }
        // Phase 2+ flat spec: "type":"Image" with a media URL/source in props.
        if (Regex(""""type"\s*:\s*"Image"[\s\S]{0,420}"(?:url|src|source|image)"\s*:\s*"(?:https?://|/assets/|assets/|\.\./assets/)""",
                setOf(RegexOption.IGNORE_CASE)).containsMatchIn(jsonText)) return true
        // Phase 2+ flat spec: "$item" image reference inside elements
        if (Regex(""""type"\s*:\s*"Image"[\s\S]{0,320}"\${"$"}item"\s*:\s*"[^"]+"""",
                setOf(RegexOption.IGNORE_CASE)).containsMatchIn(jsonText)) return true
        // Legacy format: "component":"Image"
        return Regex(
            """"component"\s*:\s*"Image"[\s\S]{0,320}"(?:url|src|source|image)"\s*:\s*"(?:https?://|/assets/|assets/)"""",
            setOf(RegexOption.IGNORE_CASE)
        ).containsMatchIn(jsonText) ||
            Regex(
                """"component"\s*:\s*"Image"[\s\S]{0,420}"(?:url|src|source|image)"\s*:\s*\{\s*"literalString"\s*:\s*"(?:https?://|/assets/|assets/)"""",
                setOf(RegexOption.IGNORE_CASE)
            ).containsMatchIn(jsonText)
    }

    private fun imageUrlFromProps(props: JsonObject?): String? {
        if (props == null) {
            return null
        }
        return listOf("url", "src", "source", "image")
            .firstNotNullOfOrNull { key -> extractImageUrlFromJsonValue(props.get(key)) }
    }

    private fun extractImageUrlFromJsonValue(element: JsonElement?): String? {
        val direct = jsonStringOrNull(element)
        if (!direct.isNullOrBlank() && looksLikeUsableInlineImageUrl(direct)) {
            return direct
        }
        if (element != null && element.isJsonObject) {
            val obj = element.asJsonObject
            return listOf("uri", "url", "src", "path", "value")
                .mapNotNull { key -> jsonStringOrNull(obj.get(key)) }
                .firstOrNull(::looksLikeUsableInlineImageUrl)
        }
        return null
    }

    fun genUiPreservesInlineIcons(jsonText: String): Boolean {
        // Phase 2+ flat spec: "type":"Icon" with "name" prop
        if (Regex(""""type"\s*:\s*"Icon"[\s\S]{0,240}"name"\s*:\s*"[^"]+"""",
                setOf(RegexOption.IGNORE_CASE)).containsMatchIn(jsonText)) return true
        // Legacy format: "component":"Icon"
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
        // Phase 2+ flat spec: "call":"openUrl" or "call":"setState"
        if (Regex("""(?i)"call"\s*:\s*"(?:openUrl|setState)"""").containsMatchIn(jsonText)) return true
        // Phase 2+ flat spec: Button type with action
        if (Regex("""(?i)"type"\s*:\s*"Button"""").containsMatchIn(jsonText) &&
            (
                Regex("""(?i)"action"\s*:\s*\{""").containsMatchIn(jsonText) ||
                    Regex("""(?i)"action"\s*:\s*"(?:openUrl|setState|pushState|removeState|validateForm|emitEvent)"""")
                        .containsMatchIn(jsonText)
                )
        ) return true
        // Legacy format
        return Regex("""(?i)"call"\s*:\s*"openUrl"""").containsMatchIn(jsonText) ||
            (
                Regex("""(?i)"component"\s*:\s*"Button"""").containsMatchIn(jsonText) &&
                    Regex("""(?i)"url"\s*:\s*"https?://""").containsMatchIn(jsonText)
                )
    }

    fun ensureResponseLinksInGenUi(
        jsonText: String,
        stage2Response: String
    ): ResponseLinkRepairResult {
        val sourceLinks = extractStage2SourceLinks(stage2Response)
        val actionLinks = extractStage2ActionButtons(stage2Response)
        if (sourceLinks.isEmpty() && actionLinks.isEmpty()) {
            return ResponseLinkRepairResult(jsonText, sourcesAdded = 0, actionsAdded = 0)
        }

        val root = runCatching { JsonParser.parseString(jsonText).asJsonObject }.getOrNull()
            ?: return ResponseLinkRepairResult(jsonText, sourcesAdded = 0, actionsAdded = 0)
        val rootId = root.get("root")?.asString?.takeIf { it.isNotBlank() }
            ?: return ResponseLinkRepairResult(jsonText, sourcesAdded = 0, actionsAdded = 0)
        val elements = root.getAsJsonObject("elements")
            ?: return ResponseLinkRepairResult(jsonText, sourcesAdded = 0, actionsAdded = 0)
        val rootElement = elements.getAsJsonObject(rootId)
            ?: return ResponseLinkRepairResult(jsonText, sourcesAdded = 0, actionsAdded = 0)
        val rootChildren = rootElement.getAsJsonArray("children") ?: JsonArray().also {
            rootElement.add("children", it)
        }

        var sourcesAdded = 0
        var actionsAdded = 0

        if (sourceLinks.isNotEmpty() && !genUiHasSourceSection(jsonText)) {
            val section = appendFlatLinkSection(
                elements = elements,
                idPrefix = "mcp_sources",
                title = "Sources",
                links = sourceLinks,
                buttonVariant = "borderless"
            )
            rootChildren.add(section)
            sourcesAdded = sourceLinks.size
        }

        if (actionLinks.isNotEmpty() && !genUiPreservesActionButtons(jsonText)) {
            val section = appendFlatLinkSection(
                elements = elements,
                idPrefix = "mcp_quick_actions",
                title = "Quick Actions",
                links = actionLinks,
                buttonVariant = "primary"
            )
            rootChildren.add(section)
            actionsAdded = actionLinks.size
        }

        return if (sourcesAdded > 0 || actionsAdded > 0) {
            ResponseLinkRepairResult(root.toString(), sourcesAdded, actionsAdded)
        } else {
            ResponseLinkRepairResult(jsonText, sourcesAdded = 0, actionsAdded = 0)
        }
    }

    private data class LinkSpec(val label: String, val url: String)

    private fun appendFlatLinkSection(
        elements: JsonObject,
        idPrefix: String,
        title: String,
        links: List<LinkSpec>,
        buttonVariant: String
    ): String {
        val sectionId = uniqueElementId(elements, "${idPrefix}_card")
        val stackId = uniqueElementId(elements, "${idPrefix}_stack")
        val titleId = uniqueElementId(elements, "${idPrefix}_title")

        val sectionChildren = JsonArray().apply { add(stackId) }
        elements.add(sectionId, JsonObject().apply {
            addProperty("type", "Card")
            add("props", JsonObject().apply { addProperty("contentPadding", "md") })
            add("children", sectionChildren)
        })

        val stackChildren = JsonArray().apply { add(titleId) }
        elements.add(stackId, JsonObject().apply {
            addProperty("type", "Stack")
            add("props", JsonObject().apply {
                addProperty("direction", "vertical")
                addProperty("gap", "sm")
            })
            add("children", stackChildren)
        })

        elements.add(titleId, JsonObject().apply {
            addProperty("type", "Text")
            add("props", JsonObject().apply {
                addProperty("text", title)
                addProperty("variant", "h3")
            })
            add("children", JsonArray())
        })

        links.distinctBy { it.url }.take(4).forEachIndexed { index, link ->
            val buttonId = uniqueElementId(elements, "${idPrefix}_button_${index + 1}")
            stackChildren.add(buttonId)
            elements.add(buttonId, JsonObject().apply {
                addProperty("type", "Button")
                add("props", JsonObject().apply {
                    addProperty("label", link.label.ifBlank { "Open link" })
                    addProperty("variant", buttonVariant)
                })
                add("on", JsonObject().apply {
                    add("press", JsonObject().apply {
                        addProperty("action", "openUrl")
                        add("params", JsonObject().apply { addProperty("url", link.url) })
                    })
                })
                add("children", JsonArray())
            })
        }

        return sectionId
    }

    private fun uniqueElementId(elements: JsonObject, base: String): String {
        var candidate = base
        var counter = 2
        while (elements.has(candidate)) {
            candidate = "${base}_$counter"
            counter++
        }
        return candidate
    }

    private fun genUiHasSourceSection(jsonText: String): Boolean {
        return Regex(
            """"(?:text|title|label)"\s*:\s*"(?:Data\s+)?Sources?"""",
            RegexOption.IGNORE_CASE
        ).containsMatchIn(jsonText)
    }

    private fun extractStage2SourceLinks(text: String): List<LinkSpec> {
        val lines = text.replace("\r\n", "\n").lines()
        val links = mutableListOf<LinkSpec>()
        var inSources = false
        lines.forEach { rawLine ->
            val line = rawLine.trim()
            if (line.isBlank()) return@forEach
            val heading = line
                .replace(Regex("""^#{1,6}\s*"""), "")
                .trim()
                .trimEnd(':')
                .trim()
            if (heading.equals("Sources", ignoreCase = true) ||
                heading.equals("Source", ignoreCase = true) ||
                heading.equals("References", ignoreCase = true)
            ) {
                inSources = true
                return@forEach
            }
            if (inSources && line.startsWith("##")) {
                inSources = false
            }
            if (!inSources) return@forEach

            val url = URL_TOKEN_REGEX.find(line)?.value
                ?.let(::normalizeExternalUrlCandidate)
                ?: return@forEach
            val label = line
                .replace(Regex("""^[-*\u2022]\s*"""), "")
                .replace(url, "")
                .trim()
                .trimEnd(':', '-', '\u2013', '\u2014')
                .trim()
                .ifBlank { quickActionLabelForUrl(url, links.size) }
            links += LinkSpec(label = label, url = url)
        }
        return links.distinctBy { it.url }.take(4)
    }

    private fun extractStage2ActionButtons(text: String): List<LinkSpec> {
        return Regex(
            """(?im)^\s*Action:\s*\[Button:\s*(.+?)]\s*(https?://\S+|//\S+|www\.\S+|(?:[a-z0-9-]+\.)+[a-z]{2,24}\S*)\s*$"""
        ).findAll(text)
            .mapNotNull { match ->
                val label = match.groupValues.getOrNull(1)?.trim().orEmpty()
                val url = match.groupValues.getOrNull(2)
                    ?.let(::normalizeExternalUrlCandidate)
                    ?: return@mapNotNull null
                LinkSpec(label = label.ifBlank { quickActionLabelForUrl(url, 0) }, url = url)
            }
            .distinctBy { it.url }
            .take(4)
            .toList()
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
        return candidates.firstOrNull(::looksLikeUsableInlineImageUrl)
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
        if (payload.isJsonObject && FlatSpecContract.looksLikeFlatSpec(payload)) {
            val changed = ensureFlatSpecHasImageComponent(
                payload = payload.asJsonObject,
                stage2Response = stage2Response,
                queryText = queryText
            )
            return if (changed) payload.toString() else jsonText
        }
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
        val fallbackImageUrl = extractFirstInlineImageUrl(stage2Response) ?: topicMediaUrl
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
            } else if (!looksLikeUsableInlineImageUrl(currentUrl)) {
                obj.addProperty("url", fallbackImageUrl)
                changed = true
            }
        }

        val alreadyHasImage = componentList.any { component ->
            component.isJsonObject &&
                jsonStringOrNull(component.asJsonObject.get("component")).equals("Image", ignoreCase = true) &&
                looksLikeUsableInlineImageUrl(jsonStringOrNull(component.asJsonObject.get("url")).orEmpty())
        }
        if (alreadyHasImage) {
            return if (changed) messages.toString() else jsonText
        }

        val mediaUrl = extractFirstInlineImageUrl(stage2Response)
            ?: if (looksLikeTravelQuery(queryText) || looksLikeTravelContent(stage2Response)) {
                extractFirstInlineIconUrl(stage2Response) ?: topicMediaUrl
            } else {
                topicMediaUrl
            }
        if (!looksLikeUsableInlineImageUrl(mediaUrl)) {
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

    private fun ensureFlatSpecHasImageComponent(
        payload: JsonObject,
        stage2Response: String,
        queryText: String
    ): Boolean {
        val elements = payload.getAsJsonObject("elements") ?: return false
        val topicMediaUrl = buildTopicFallbackMediaUrl(queryText)
        val fallbackImageUrl = extractFirstInlineImageUrl(stage2Response) ?: topicMediaUrl
        val isFlight = looksLikeFlightQuery(queryText) || looksLikeFlightContent(stage2Response)
        var changed = false
        var hasImage = false

        elements.entrySet().forEach { (_, node) ->
            if (!node.isJsonObject) return@forEach
            val element = node.asJsonObject
            val type = jsonStringOrNull(element.get("type")).orEmpty()
            if (!type.equals("Image", ignoreCase = true)) {
                return@forEach
            }
            val props = element.getAsJsonObject("props") ?: JsonObject().also {
                element.add("props", it)
                changed = true
            }
            val currentUrl = jsonStringOrNull(props.get("url")).orEmpty()
            if (isFlight) {
                if (!currentUrl.equals(topicMediaUrl, ignoreCase = true)) {
                    props.addProperty("url", topicMediaUrl)
                    changed = true
                }
            } else if (!looksLikeUsableInlineImageUrl(currentUrl)) {
                props.addProperty("url", fallbackImageUrl)
                changed = true
            }
            if (looksLikeUsableInlineImageUrl(jsonStringOrNull(props.get("url")).orEmpty())) {
                hasImage = true
            }
        }

        if (hasImage) {
            return changed
        }

        val mediaUrl = extractFirstInlineImageUrl(stage2Response)
            ?: if (looksLikeTravelQuery(queryText) || looksLikeTravelContent(stage2Response)) {
                extractFirstInlineIconUrl(stage2Response) ?: topicMediaUrl
            } else {
                topicMediaUrl
            }
        if (!looksLikeUsableInlineImageUrl(mediaUrl)) {
            return changed
        }

        val mediaId = buildUniqueFlatElementId(elements, base = "auto_media")
        elements.add(
            mediaId,
            JsonObject().apply {
                addProperty("type", "Image")
                add("props", JsonObject().apply {
                    addProperty("url", mediaUrl)
                    addProperty("fit", "cover")
                })
                add("children", JsonArray())
            }
        )
        prependRootFlatChildId(payload, mediaId)
        return true
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

    fun preserveRestaurantPhotosInGenUi(
        jsonText: String,
        stage2Response: String
    ): String {
        val photosByName = extractRestaurantPhotoUrls(stage2Response)
        val phonesByName = extractRestaurantPhoneNumbers(stage2Response)
        if (photosByName.isEmpty() && phonesByName.isEmpty()) return jsonText

        val parsed = runCatching { JsonParser.parseString(jsonText) }.getOrNull() ?: return jsonText
        val payload = normalizeGenUiPayload(parsed)
        if (!payload.isJsonObject || !FlatSpecContract.looksLikeFlatSpec(payload)) return jsonText

        val spec = payload.asJsonObject
        val state = spec.get("state")?.takeIf { it.isJsonObject }?.asJsonObject ?: return jsonText
        val elements = spec.getAsJsonObject("elements") ?: return jsonText
        var changed = false

        state.entrySet().forEach { (_, value) ->
            if (!value.isJsonArray) return@forEach
            value.asJsonArray.forEach { row ->
                val rowObj = row.takeIf { it.isJsonObject }?.asJsonObject ?: return@forEach
                val name = firstRowString(rowObj, "restaurant", "name", "place", "title") ?: return@forEach
                val normalizedName = normalizeRestaurantName(name)
                val photoUrls = photosByName[normalizedName]
                if (photoUrls != null) {
                    if (!rowHasImage(rowObj)) {
                        rowObj.addProperty("photoUrl", photoUrls.first())
                        changed = true
                    }
                    if (!rowHasPhotoList(rowObj) && photoUrls.size > 1) {
                        rowObj.addProperty("photoUrls", photoUrls.joinToString(", "))
                        changed = true
                    }
                }
                val phone = phonesByName[normalizedName]
                if (phone != null && !rowHasPhone(rowObj)) {
                    rowObj.addProperty("phone", phone)
                    changed = true
                }
            }
        }

        elements.entrySet().forEach { (_, node) ->
            val element = node.takeIf { it.isJsonObject }?.asJsonObject ?: return@forEach
            if (!jsonStringOrNull(element.get("type")).equals("Table", ignoreCase = true)) return@forEach
            val props = element.getAsJsonObject("props") ?: return@forEach
            if (!isRestaurantTableProps(props)) return@forEach
            val columns = props.get("columns")?.takeIf { it.isJsonArray }?.asJsonArray ?: return@forEach
            if (!columns.any { columnHasKeyOrLabel(it, "photo", "photourl", "photo url", "image") }) {
                columns.add(JsonObject().apply {
                    addProperty("key", "photoUrl")
                    addProperty("label", "Photo")
                })
                changed = true
            }
            if (!columns.any { columnHasKeyOrLabel(it, "photos", "photourls", "photo urls", "images") }) {
                columns.add(JsonObject().apply {
                    addProperty("key", "photoUrls")
                    addProperty("label", "Photos")
                })
                changed = true
            }
            if (phonesByName.isNotEmpty() && !columns.any { columnHasKeyOrLabel(it, "phone", "telephone", "call") }) {
                columns.add(JsonObject().apply {
                    addProperty("key", "phone")
                    addProperty("label", "Phone")
                })
                changed = true
            }
        }

        return if (changed) spec.toString() else jsonText
    }

    fun preserveNewsMediaInGenUi(
        jsonText: String,
        stage2Response: String
    ): String {
        val mediaByArticle = extractNewsMediaRows(stage2Response)
        if (mediaByArticle.isEmpty()) return jsonText

        val parsed = runCatching { JsonParser.parseString(jsonText) }.getOrNull() ?: return jsonText
        val payload = normalizeGenUiPayload(parsed)
        if (!payload.isJsonObject || !FlatSpecContract.looksLikeFlatSpec(payload)) return jsonText

        val spec = payload.asJsonObject
        val state = spec.get("state")?.takeIf { it.isJsonObject }?.asJsonObject ?: return jsonText
        val elements = spec.getAsJsonObject("elements") ?: return jsonText
        var changed = false

        state.entrySet().forEach { (_, value) ->
            if (!value.isJsonArray) return@forEach
            value.asJsonArray.forEach { row ->
                val rowObj = row.takeIf { it.isJsonObject }?.asJsonObject ?: return@forEach
                val article = firstRowString(rowObj, "article", "title", "headline", "story", "news") ?: return@forEach
                val media = mediaByArticle[normalizeNewsArticleTitle(article)] ?: return@forEach
                if (!rowHasImage(rowObj) && media.imageUrl.isNotBlank()) {
                    rowObj.addProperty("imageUrl", media.imageUrl)
                    changed = true
                }
                if (!rowHasSourceIcon(rowObj) && media.sourceIcon.isNotBlank()) {
                    rowObj.addProperty("sourceIcon", media.sourceIcon)
                    changed = true
                }
                if (!rowHasUrlField(rowObj, "articleUrl", "article URL", "url", "link") && media.articleUrl.isNotBlank()) {
                    rowObj.addProperty("articleUrl", media.articleUrl)
                    changed = true
                }
                if (!rowHasUrlField(rowObj, "sourceUrl", "source URL", "publisherUrl") && media.sourceUrl.isNotBlank()) {
                    rowObj.addProperty("sourceUrl", media.sourceUrl)
                    changed = true
                }
                if (jsonStringOrNull(rowObj.get("actionLabel")).isNullOrBlank() && media.actionLabel.isNotBlank()) {
                    rowObj.addProperty("actionLabel", media.actionLabel)
                    changed = true
                }
            }
        }

        elements.entrySet().forEach { (_, node) ->
            val element = node.takeIf { it.isJsonObject }?.asJsonObject ?: return@forEach
            if (!jsonStringOrNull(element.get("type")).equals("Table", ignoreCase = true)) return@forEach
            val props = element.getAsJsonObject("props") ?: return@forEach
            if (!isNewsTableProps(props)) return@forEach
            if (!jsonStringOrNull(props.get("domain")).equals("news", ignoreCase = true)) {
                props.addProperty("domain", "news")
                changed = true
            }
            if (jsonStringOrNull(props.get("preferredPresentation")).isNullOrBlank()) {
                props.addProperty("preferredPresentation", "cards")
                changed = true
            }
            val columns = props.get("columns")?.takeIf { it.isJsonArray }?.asJsonArray ?: return@forEach
            if (!columns.any { columnHasKeyOrLabel(it, "image", "imageurl", "image url", "photo", "thumbnail") }) {
                columns.add(JsonObject().apply {
                    addProperty("key", "imageUrl")
                    addProperty("label", "Image URL")
                })
                changed = true
            }
            if (!columns.any { columnHasKeyOrLabel(it, "sourceicon", "source icon", "publishericon", "publisher icon") }) {
                columns.add(JsonObject().apply {
                    addProperty("key", "sourceIcon")
                    addProperty("label", "Source Icon")
                })
                changed = true
            }
            if (!columns.any { columnHasKeyOrLabel(it, "articleurl", "article url", "link", "url") }) {
                columns.add(JsonObject().apply {
                    addProperty("key", "articleUrl")
                    addProperty("label", "Article URL")
                })
                changed = true
            }
            if (!columns.any { columnHasKeyOrLabel(it, "sourceurl", "source url", "publisherurl", "publisher url") }) {
                columns.add(JsonObject().apply {
                    addProperty("key", "sourceUrl")
                    addProperty("label", "Source URL")
                })
                changed = true
            }
        }

        return if (changed) spec.toString() else jsonText
    }

    private data class NewsMediaRow(
        val imageUrl: String,
        val sourceIcon: String,
        val articleUrl: String,
        val sourceUrl: String,
        val actionLabel: String
    )

    private fun extractNewsMediaRows(stage2Response: String): Map<String, NewsMediaRow> {
        val lines = stage2Response.lines()
        val result = linkedMapOf<String, NewsMediaRow>()
        lines.forEachIndexed { index, line ->
            if (!line.trimStart().startsWith("|")) return@forEachIndexed
            if (!line.contains("Article", ignoreCase = true) && !line.contains("Headline", ignoreCase = true)) {
                return@forEachIndexed
            }
            if (!line.contains("Image", ignoreCase = true) && !line.contains("Source Icon", ignoreCase = true)) {
                return@forEachIndexed
            }
            val headers = splitMarkdownTableRow(line)
            val articleIndex = headers.indexOfFirst { header ->
                normalizedKey(header) in setOf("article", "title", "headline", "story", "news")
            }
            if (articleIndex < 0) return@forEachIndexed
            val imageIndex = headers.indexOfFirst { header ->
                normalizedKey(header) in setOf("image", "imageurl", "photo", "thumbnail", "media")
            }
            val sourceIconIndex = headers.indexOfFirst { header ->
                normalizedKey(header) in setOf("sourceicon", "publishericon", "icon")
            }
            val articleUrlIndex = headers.indexOfFirst { header ->
                normalizedKey(header) in setOf("articleurl", "articlelink", "url", "link", "readurl")
            }
            val sourceUrlIndex = headers.indexOfFirst { header ->
                normalizedKey(header) in setOf("sourceurl", "sourcelink", "publisherurl", "publisherlink")
            }
            val actionLabelIndex = headers.indexOfFirst { header ->
                normalizedKey(header) in setOf("actionlabel", "buttonlabel", "ctalabel", "action")
            }
            lines.drop(index + 2)
                .takeWhile { it.trimStart().startsWith("|") }
                .forEach { rowLine ->
                    val cells = splitMarkdownTableRow(rowLine)
                    val article = cells.getOrNull(articleIndex)?.trim().orEmpty()
                    if (article.isBlank()) return@forEach
                    val imageUrl = imageIndex
                        .takeIf { it >= 0 }
                        ?.let { cells.getOrNull(it).orEmpty() }
                        ?.let(::firstSafeImageUrl)
                        .orEmpty()
                    val sourceIcon = sourceIconIndex
                        .takeIf { it >= 0 }
                        ?.let { cells.getOrNull(it).orEmpty() }
                        ?.let(::firstSafeImageUrl)
                        .orEmpty()
                    val articleUrl = articleUrlIndex
                        .takeIf { it >= 0 }
                        ?.let { cells.getOrNull(it).orEmpty().trim() }
                        ?.let(SafeContentPolicy::sanitizeActionUrl)
                        .orEmpty()
                    val sourceUrl = sourceUrlIndex
                        .takeIf { it >= 0 }
                        ?.let { cells.getOrNull(it).orEmpty().trim() }
                        ?.let(SafeContentPolicy::sanitizeActionUrl)
                        .orEmpty()
                    val actionLabel = actionLabelIndex
                        .takeIf { it >= 0 }
                        ?.let { cells.getOrNull(it).orEmpty().trim() }
                        ?.takeIf { it.isNotBlank() && !SafeContentPolicy.looksLikeUrl(it) }
                        .orEmpty()
                    if (imageUrl.isNotBlank() || sourceIcon.isNotBlank()) {
                        result[normalizeNewsArticleTitle(article)] = NewsMediaRow(
                            imageUrl = imageUrl,
                            sourceIcon = sourceIcon,
                            articleUrl = articleUrl,
                            sourceUrl = sourceUrl,
                            actionLabel = actionLabel
                        )
                    }
                }
        }
        return result
    }

    private fun extractRestaurantPhotoUrls(stage2Response: String): Map<String, List<String>> {
        val lines = stage2Response.lines()
        val result = linkedMapOf<String, List<String>>()
        lines.forEachIndexed { index, line ->
            if (!line.trimStart().startsWith("|")) return@forEachIndexed
            if (!line.contains("Restaurant", ignoreCase = true) || !line.contains("Photo", ignoreCase = true)) {
                return@forEachIndexed
            }
            val headers = splitMarkdownTableRow(line)
            val restaurantIndex = headers.indexOfFirst { it.equals("Restaurant", ignoreCase = true) || it.equals("Name", ignoreCase = true) }
            val photoIndexes = headers.indices.filter { headerIndex ->
                val normalized = headers[headerIndex].lowercase(Locale.US).replace(Regex("[^a-z0-9]+"), "")
                normalized in setOf("photo", "photos", "photourl", "photourls", "image", "imageurl", "imageurls", "media")
            }
            val photoIndex = photoIndexes.firstOrNull { headerIndex ->
                val normalized = headers[headerIndex].lowercase(Locale.US).replace(Regex("[^a-z0-9]+"), "")
                normalized in setOf("photos", "photourls", "imageurls")
            } ?: photoIndexes.firstOrNull()
            if (restaurantIndex < 0 || photoIndex == null) return@forEachIndexed
            lines.drop(index + 2)
                .takeWhile { it.trimStart().startsWith("|") }
                .forEach { rowLine ->
                    val cells = splitMarkdownTableRow(rowLine)
                    val name = cells.getOrNull(restaurantIndex)?.trim().orEmpty()
                    val rawPhotoCell = cells.getOrNull(photoIndex).orEmpty()
                    val photoUrls = safeImageUrls(rawPhotoCell)
                    if (name.isNotBlank() && photoUrls.isNotEmpty()) {
                        result[normalizeRestaurantName(name)] = photoUrls
                    }
                }
        }
        return result
    }

    private fun extractRestaurantPhoneNumbers(stage2Response: String): Map<String, String> {
        val lines = stage2Response.lines()
        val result = linkedMapOf<String, String>()
        lines.forEachIndexed { index, line ->
            if (!line.trimStart().startsWith("|")) return@forEachIndexed
            if (!line.contains("Restaurant", ignoreCase = true) || !line.contains("Phone", ignoreCase = true)) {
                return@forEachIndexed
            }
            val headers = splitMarkdownTableRow(line)
            val restaurantIndex = headers.indexOfFirst { it.equals("Restaurant", ignoreCase = true) || it.equals("Name", ignoreCase = true) }
            val phoneIndex = headers.indexOfFirst {
                val normalized = it.lowercase(Locale.US).replace(Regex("[^a-z0-9]+"), "")
                normalized in setOf("phone", "telephone", "call", "nationalphone", "internationalphone")
            }
            if (restaurantIndex < 0 || phoneIndex < 0) return@forEachIndexed
            lines.drop(index + 2)
                .takeWhile { it.trimStart().startsWith("|") }
                .forEach { rowLine ->
                    val cells = splitMarkdownTableRow(rowLine)
                    val name = cells.getOrNull(restaurantIndex)?.trim().orEmpty()
                    val rawPhone = cells.getOrNull(phoneIndex)?.trim().orEmpty()
                    if (name.isNotBlank() && SafeContentPolicy.sanitizePhoneDialUrl(rawPhone) != null) {
                        result[normalizeRestaurantName(name)] = rawPhone
                    }
                }
        }
        return result
    }

    private fun splitMarkdownTableRow(line: String): List<String> =
        line.trim().trim('|').split('|').map { it.trim() }

    private fun firstSafeImageUrl(value: String): String? =
        safeImageUrls(value).firstOrNull()

    private fun safeImageUrls(value: String): List<String> =
        Regex("""https?://[^\s,|]+""")
            .findAll(value)
            .mapNotNull { match ->
                SafeContentPolicy.sanitizeMediaUrl(match.value, SafeContentPolicy.MediaKind.IMAGE)
            }
            .distinct()
            .take(5)
            .toList()

    private fun rowHasImage(row: JsonObject): Boolean =
        listOf("photoUrl", "photo", "image", "imageUrl", "thumbnail", "mediaImage").any { key ->
            val value = jsonStringOrNull(row.get(key)) ?: return@any false
            SafeContentPolicy.isSafeMediaUrl(value, SafeContentPolicy.MediaKind.IMAGE)
        }

    private fun rowHasPhotoList(row: JsonObject): Boolean =
        listOf("photoUrls", "photos", "images", "media").any { key ->
            val value = jsonStringOrNull(row.get(key)) ?: return@any false
            safeImageUrls(value).isNotEmpty()
        }

    private fun rowHasPhone(row: JsonObject): Boolean =
        listOf("phone", "telephone", "call", "nationalPhoneNumber", "internationalPhoneNumber").any { key ->
            SafeContentPolicy.sanitizePhoneDialUrl(jsonStringOrNull(row.get(key))) != null
        }

    private fun firstRowString(row: JsonObject, vararg keys: String): String? {
        keys.forEach { key ->
            jsonStringOrNull(row.get(key))?.trim()?.takeIf { it.isNotBlank() }?.let { return it }
        }
        return null
    }

    private fun normalizeRestaurantName(value: String): String =
        value.lowercase(Locale.US).replace(Regex("[^a-z0-9]+"), "")

    private fun normalizedKey(value: String): String =
        value.lowercase(Locale.US).replace(Regex("[^a-z0-9]+"), "")

    private fun normalizeNewsArticleTitle(value: String): String =
        normalizedKey(value).take(96)

    private fun rowHasSourceIcon(row: JsonObject): Boolean =
        listOf("sourceIcon", "sourceicon", "publisherIcon", "icon", "mediaIcon", "iconUrl").any { key ->
            val value = jsonStringOrNull(row.get(key)) ?: return@any false
            SafeContentPolicy.isSafeMediaUrl(value, SafeContentPolicy.MediaKind.ICON) ||
                SafeContentPolicy.isSafeMediaUrl(value, SafeContentPolicy.MediaKind.IMAGE)
        }

    private fun rowHasUrlField(row: JsonObject, vararg keys: String): Boolean =
        keys.any { key ->
            val value = jsonStringOrNull(row.get(key)) ?: return@any false
            SafeContentPolicy.sanitizeActionUrl(value) != null
        }

    private fun isNewsTableProps(props: JsonObject): Boolean {
        val domain = jsonStringOrNull(props.get("domain")).orEmpty().lowercase(Locale.US)
        if (domain == "news") return true
        val columns = props.get("columns")?.takeIf { it.isJsonArray }?.asJsonArray ?: return false
        return columns.any { columnHasKeyOrLabel(it, "article", "title", "headline", "story", "news") } &&
            columns.any { columnHasKeyOrLabel(it, "source", "publisher", "publication") }
    }

    private fun isRestaurantTableProps(props: JsonObject): Boolean {
        val domain = jsonStringOrNull(props.get("domain")).orEmpty().lowercase(Locale.US)
        if (domain in setOf("restaurant", "restaurants", "place", "places", "dining")) return true
        val columns = props.get("columns")?.takeIf { it.isJsonArray }?.asJsonArray ?: return false
        return columns.any { columnHasKeyOrLabel(it, "restaurant", "name", "place") } &&
            columns.any { columnHasKeyOrLabel(it, "rating", "reviews", "address", "mapsurl", "maps url") }
    }

    private fun columnHasKeyOrLabel(column: JsonElement, vararg tokens: String): Boolean {
        val obj = column.takeIf { it.isJsonObject }?.asJsonObject ?: return false
        val values = listOfNotNull(jsonStringOrNull(obj.get("key")), jsonStringOrNull(obj.get("label")))
            .map { it.lowercase(Locale.US).replace(Regex("[^a-z0-9]+"), "") }
        return tokens
            .map { it.lowercase(Locale.US).replace(Regex("[^a-z0-9]+"), "") }
            .any { token -> values.any { it == token } }
    }

    data class SafeGenUiResult(
        val jsonText: String,
        val removedMediaCount: Int,
        val removedActionCount: Int
    ) {
        val changed: Boolean get() = removedMediaCount > 0 || removedActionCount > 0
    }

    fun enforceSafeGenUiContent(jsonText: String): SafeGenUiResult {
        val parsed = runCatching { JsonParser.parseString(jsonText) }.getOrNull()
            ?: return SafeGenUiResult(jsonText, 0, 0)
        val payload = normalizeGenUiPayload(parsed)
        if (!payload.isJsonObject || !FlatSpecContract.looksLikeFlatSpec(payload)) {
            return SafeGenUiResult(jsonText, 0, 0)
        }
        val spec = payload.asJsonObject
        val elements = spec.getAsJsonObject("elements") ?: return SafeGenUiResult(jsonText, 0, 0)
        val state = spec.get("state")?.takeIf { it.isJsonObject }?.asJsonObject
        var removedMedia = 0
        var removedActions = 0

        elements.entrySet().forEach { (_, node) ->
            if (!node.isJsonObject) return@forEach
            val element = node.asJsonObject
            val type = jsonStringOrNull(element.get("type")).orEmpty().lowercase(Locale.US)
            val props = element.getAsJsonObject("props")
            if (props != null) {
                removedMedia += sanitizeElementMediaProps(type, props, state)
                removedActions += sanitizeUrlFieldsInProps(type, props)
            }
            element.getAsJsonObject("on")?.let { removedActions += sanitizeActionObject(it) }
            element.getAsJsonObject("watch")?.let { removedActions += sanitizeActionObject(it) }
        }

        return if (removedMedia > 0 || removedActions > 0) {
            SafeGenUiResult(spec.toString(), removedMedia, removedActions)
        } else {
            SafeGenUiResult(jsonText, 0, 0)
        }
    }

    private fun sanitizeElementMediaProps(
        type: String,
        props: JsonObject,
        state: JsonObject?
    ): Int {
        var removed = 0
        fun sanitizeKeys(keys: List<String>, kind: SafeContentPolicy.MediaKind) {
            keys.forEach { key ->
                val value = jsonStringOrNull(props.get(key)) ?: return@forEach
                if (!SafeContentPolicy.looksLikeUrl(value) && !SafeContentPolicy.isLocalAssetUrl(value) && !SafeContentPolicy.isGeneratedVisualUrl(value)) {
                    return@forEach
                }
                val safe = SafeContentPolicy.sanitizeMediaUrl(value, kind)
                if (safe == null) {
                    props.remove(key)
                    removed += 1
                } else if (safe != value) {
                    props.addProperty(key, safe)
                }
            }
        }

        when (type) {
            "image" -> sanitizeKeys(listOf("url", "src", "image", "source", "name"), SafeContentPolicy.MediaKind.IMAGE)
            "icon" -> sanitizeKeys(listOf("name", "icon", "source", "url", "src"), SafeContentPolicy.MediaKind.ICON)
            "video" -> sanitizeKeys(listOf("url", "src", "source"), SafeContentPolicy.MediaKind.VIDEO)
            "audioplayer" -> sanitizeKeys(listOf("url", "src", "source"), SafeContentPolicy.MediaKind.AUDIO)
            "table" -> {
                removed += sanitizeTableMediaProps(props, state)
            }
        }
        return removed
    }

    private fun sanitizeTableMediaProps(props: JsonObject, state: JsonObject?): Int {
        var removed = 0
        val imageKeys = setOf("image", "imageurl", "photo", "photourl", "thumbnail", "mediaimage")
        val iconKeys = setOf("icon", "iconurl", "mediaicon")
        val actionKeys = setOf("url", "href", "link", "actionurl", "bookingurl", "buttonurl", "sourceurl", "targeturl")
        props.get("entityMedia")
            ?.takeIf { it.isJsonObject }
            ?.asJsonObject
            ?.entrySet()
            ?.forEach { (_, value) ->
            if (!value.isJsonObject) return@forEach
            val media = value.asJsonObject
            listOf("image", "url", "src", "source", "path").forEach { key ->
                val raw = jsonStringOrNull(media.get(key)) ?: return@forEach
                val safe = SafeContentPolicy.sanitizeMediaUrl(raw, SafeContentPolicy.MediaKind.IMAGE)
                if (safe == null) {
                    media.remove(key)
                    removed += 1
                } else if (safe != raw) {
                    media.addProperty(key, safe)
                }
            }
        }

        val rows = props.get("rows")
            ?.takeIf { it.isJsonArray }
            ?.asJsonArray
            ?: jsonStringOrNull(props.get("statePath"))
                ?.let { pointer -> state?.let { jsonAtPointer(it, pointer) } }
                ?.takeIf { it.isJsonArray }
                ?.asJsonArray
            ?: return removed
        rows.forEach { row ->
            if (!row.isJsonObject) return@forEach
            val rowObj = row.asJsonObject
            rowObj.entrySet().toList().forEach { (key, value) ->
                val normalizedKey = key.lowercase(Locale.US).replace(Regex("[^a-z0-9]+"), "")
                val raw = jsonStringOrNull(value) ?: return@forEach
                val kind = when {
                    normalizedKey in imageKeys -> SafeContentPolicy.MediaKind.IMAGE
                    normalizedKey in iconKeys -> SafeContentPolicy.MediaKind.ICON
                    normalizedKey in actionKeys -> null
                    else -> return@forEach
                }
                val safe = if (kind == null) {
                    SafeContentPolicy.sanitizeActionUrl(raw)
                } else {
                    SafeContentPolicy.sanitizeMediaUrl(raw, kind)
                }
                if (safe == null) {
                    rowObj.remove(key)
                    removed += 1
                } else if (safe != raw) {
                    rowObj.addProperty(key, safe)
                }
            }
        }
        return removed
    }

    private fun sanitizeUrlFieldsInProps(type: String, props: JsonObject): Int {
        if (type in setOf("image", "icon", "video", "audioplayer", "table")) return 0
        var removed = 0
        val keys = listOf("url", "actionUrl", "bookingUrl", "buttonUrl", "sourceUrl", "targetUrl", "href", "link")
        keys.forEach { key ->
            val raw = jsonStringOrNull(props.get(key)) ?: return@forEach
            if (!SafeContentPolicy.looksLikeUrl(raw)) return@forEach
            val safe = SafeContentPolicy.sanitizeActionUrl(raw)
            if (safe == null) {
                props.remove(key)
                removed += 1
            } else if (safe != raw) {
                props.addProperty(key, safe)
            }
        }
        return removed
    }

    private fun jsonAtPointer(root: JsonObject, pointer: String): JsonElement? {
        val normalized = pointer.trim()
        if (normalized.isBlank()) return null
        if (normalized == "/" || normalized == "$") return root
        val tokens = normalized
            .removePrefix("$")
            .trimStart('/')
            .split('/')
            .filter { it.isNotBlank() }
        var current: JsonElement = root
        tokens.forEach { rawToken ->
            val token = rawToken.replace("~1", "/").replace("~0", "~")
            current = when {
                current.isJsonObject -> current.asJsonObject.get(token) ?: return null
                current.isJsonArray -> token.toIntOrNull()
                    ?.takeIf { it >= 0 && it < current.asJsonArray.size() }
                    ?.let { current.asJsonArray[it] }
                    ?: return null
                else -> return null
            }
        }
        return current
    }

    private fun sanitizeActionObject(actions: JsonObject): Int {
        var removed = 0
        actions.entrySet().toList().forEach { (key, value) ->
            val sanitized = sanitizeActionBinding(value)
            if (sanitized == null) {
                actions.remove(key)
                removed += 1
            } else if (sanitized !== value) {
                actions.add(key, sanitized)
            }
        }
        return removed
    }

    private fun sanitizeActionBinding(value: JsonElement): JsonElement? {
        if (value.isJsonArray) {
            val output = JsonArray()
            value.asJsonArray.forEach { child ->
                sanitizeActionBinding(child)?.let(output::add)
            }
            return output.takeIf { it.size() > 0 }
        }
        if (!value.isJsonObject) return value
        val obj = value.asJsonObject
        val action = jsonStringOrNull(obj.get("action")).orEmpty().lowercase(Locale.US)
        if (action != "openurl") return value
        val params = obj.get("params")?.takeIf { it.isJsonObject }?.asJsonObject ?: return value
        val urlKey = listOf("url", "href", "link", "targetUrl").firstOrNull { params.has(it) } ?: return value
        val raw = jsonStringOrNull(params.get(urlKey)) ?: return value
        val safe = SafeContentPolicy.sanitizeActionUrl(raw) ?: return null
        if (safe == raw) return value
        val copy = obj.deepCopy()
        copy.getAsJsonObject("params").addProperty(urlKey, safe)
        return copy
    }

    fun ensureGenUiHasInlineTextMedia(
        jsonText: String,
        queryText: String
    ): String {
        val parsed = runCatching { JsonParser.parseString(jsonText) }.getOrNull() ?: return jsonText
        val payload = normalizeGenUiPayload(parsed)
        if (payload.isJsonObject && FlatSpecContract.looksLikeFlatSpec(payload)) {
            val changed = ensureFlatSpecHasInlineTextMedia(payload.asJsonObject, queryText)
            return if (changed) payload.toString() else jsonText
        }
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

    private fun ensureFlatSpecHasInlineTextMedia(
        payload: JsonObject,
        queryText: String
    ): Boolean {
        val elements = payload.getAsJsonObject("elements") ?: return false
        val mediaUrl = buildTopicFallbackMediaUrl(queryText)
        elements.entrySet().forEach { (_, node) ->
            if (!node.isJsonObject) return@forEach
            val element = node.asJsonObject
            val type = jsonStringOrNull(element.get("type")).orEmpty()
            if (!type.equals("Text", ignoreCase = true)) return@forEach

            val props = element.getAsJsonObject("props") ?: return@forEach
            val textValue = jsonStringOrNull(props.get("text")).orEmpty()
            if (textValue.isBlank()) return@forEach
            if (textValue.contains("Media:", ignoreCase = true) || textValue.contains("Image=", ignoreCase = true)) {
                return@forEach
            }

            props.addProperty("text", "Media: Image=$mediaUrl Icon=$mediaUrl\n$textValue")
            return true
        }
        return false
    }

    fun sanitizeMediaUrlToken(value: String): String =
        value.trim().trim('\'', '"').trimEnd('.', ',', ';', ')', ']')

    fun looksLikeUsableInlineMediaUrl(value: String): Boolean {
        return SafeContentPolicy.isSafeMediaUrl(value, SafeContentPolicy.MediaKind.IMAGE) ||
            SafeContentPolicy.isSafeMediaUrl(value, SafeContentPolicy.MediaKind.ICON)
    }

    fun looksLikeUsableInlineImageUrl(value: String): Boolean {
        return SafeContentPolicy.isSafeMediaUrl(value, SafeContentPolicy.MediaKind.IMAGE)
    }

    private fun looksLikeIconOnlyMediaUrl(value: String): Boolean {
        return SafeContentPolicy.isIconOnlyMediaUrl(value)
    }

    // -- Payload ------------------------------------------------------------

    fun normalizeGenUiPayload(json: JsonElement): JsonElement {
        FlatSpecContract.normalizeToFlatSpec(json).spec?.let { return it }
        return json
    }

    fun buildFallbackFlatSpec(stage2Response: String, catalogId: String): JsonObject {
        return FlatSpecContract.buildFallbackFlatSpec(stage2Response)
    }

    fun buildFallbackGenUi(stage2Response: String, catalogId: String): JsonArray {
        val textValue = stage2Response.trim().ifBlank { "No content generated." }
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
                                    add("text_1")
                                })
                            })
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

    private fun buildUniqueFlatElementId(elements: JsonObject, base: String): String {
        var index = 1
        while (true) {
            val candidate = "${base}_$index"
            if (!elements.has(candidate)) {
                return candidate
            }
            index += 1
        }
    }

    private fun prependRootFlatChildId(payload: JsonObject, childId: String) {
        val rootId = payload.get("root")
            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?: return
        val rootElement = payload.getAsJsonObject("elements")?.getAsJsonObject(rootId) ?: return
        val children = rootElement.get("children")
            ?.takeIf { it.isJsonArray }
            ?.asJsonArray
            ?: JsonArray().also { rootElement.add("children", it) }
        val existing = children.mapNotNull { child ->
            if (child.isJsonPrimitive && child.asJsonPrimitive.isString) child.asString else null
        }
        if (existing.contains(childId)) return

        val updated = JsonArray().apply {
            add(childId)
            existing.forEach { add(it) }
        }
        rootElement.add("children", updated)
    }

    fun resolveStage3CatalogId(prefs: SharedPreferences): String {
        val override = prefs
            .getString(PREF_STAGE3_CATALOG_ID, null)
            ?.trim()
            .orEmpty()
        return if (override.isNotBlank()) override else DEFAULT_STAGE3_CATALOG_ID
    }
}
