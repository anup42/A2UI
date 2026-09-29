package com.samsung.genuicraft.sdk.internal.renderer.native.intents.flight

import com.samsung.genuicraft.sdk.internal.renderer.native.FlightPoint
import com.samsung.genuicraft.sdk.internal.renderer.native.FlightRow
import com.samsung.genuicraft.sdk.internal.renderer.native.FlightTableColumns
import com.samsung.genuicraft.sdk.internal.renderer.native.NativeTextFormatter

import java.util.Locale

internal object NativeFlightSemantics {
    private val flightClock = Regex(
        """(?<![\d:])(\d{1,2}):([0-5]\d)(?:\s*(AM|PM))?(?:\s*(?:\(\s*\+(\d+)\s*\)|\+(\d+)))?(?![\d:])""",
        RegexOption.IGNORE_CASE
    )
    private val rangeSeparator = Regex("""->|[-\u2013\u2014\u2192]|\bto\b""", RegexOption.IGNORE_CASE)
    private val airportAroundSeparator = Regex("""\s*(?:\(?[A-Z]{3}\)?)?\s*""")

    /** Split only an explicit two-clock range; never infer a day offset or a missing time. */
    fun parseFlightTimeRange(value: String?): Pair<String, String>? {
        val text = NativeTextFormatter.sanitizeDisplayText(value.orEmpty())
        val clocks = flightClock.findAll(text).toList()
        if (clocks.size != 2 || clocks.any { !isValidFlightClock(it) }) return null
        val betweenStart = clocks[0].range.last + 1
        val between = text.substring(betweenStart, clocks[1].range.first)
        val separator = rangeSeparator.find(between) ?: return null
        if (!airportAroundSeparator.matches(between.substring(0, separator.range.first)) ||
            !airportAroundSeparator.matches(between.substring(separator.range.last + 1))) return null
        val split = betweenStart + separator.range.first
        return text.substring(0, split).trim() to
            text.substring(split + separator.value.length).trim()
    }

    private fun isValidFlightClock(match: MatchResult): Boolean {
        val hours = match.groupValues[1].toInt()
        return hours in if (match.groupValues[3].isBlank()) 0..23 else 1..12
    }

    fun hasCombinedFlightTimes(header: List<String>, body: List<List<String>>): Boolean {
        val columns = detectFlightColumns(header) ?: return false
        return body.any { row ->
            parseFlightTimeRange(readCell(row, columns.depart)) != null ||
                parseFlightTimeRange(readCell(row, columns.arrive)) != null
        }
    }

    fun parseFlightPoint(value: String?, fallbackCode: String?): FlightPoint {
        val raw = NativeTextFormatter.sanitizeDisplayText(value.orEmpty()).trim()
        val fallback = NativeTextFormatter.sanitizeDisplayText(fallbackCode.orEmpty()).ifBlank { null }
        if (raw.isBlank()) {
            return FlightPoint(time = null, code = fallback)
        }

        val time = normalizeFlightTime(raw)
        val inlineCode = Regex("""\b([A-Z]{3})\b""")
            .find(raw.uppercase(Locale.US))
            ?.groupValues
            ?.getOrNull(1)
            ?.uppercase(Locale.US)
        return FlightPoint(
            time = time,
            code = inlineCode ?: fallback
        )
    }

    fun normalizeFlightTime(value: String): String? {
        val cleaned = NativeTextFormatter.sanitizeDisplayText(value).trim()
        if (cleaned.isBlank()) {
            return null
        }
        val match = flightClock.find(cleaned)?.takeIf(::isValidFlightClock)
            ?: return null
        val hhmm = "${match.groupValues[1]}:${match.groupValues[2]}"
        val suffix = match.groupValues[3].uppercase(Locale.US)
        val dayOffset = match.groupValues[4].ifBlank { match.groupValues[5] }
        val ampm = if (suffix.isBlank()) "" else " $suffix"
        val plus = if (dayOffset.isBlank()) "" else "+$dayOffset"
        return "$hhmm$ampm$plus"
    }

    fun normalizeDurationLabel(value: String?): String? {
        val raw = NativeTextFormatter.sanitizeDisplayText(value.orEmpty()).trim()
        if (raw.isBlank()) {
            return null
        }
        val match = Regex("""(?i)(?<![\d.])(\d+(?:\.\d+)?)\s*h(?:ours?)?\s*(?:(\d+)\s*m(?:in(?:ute)?s?)?)?""").find(raw)
        if (match != null) {
            val hours = match.groupValues[1]
            val mins = match.groupValues.getOrNull(2).orEmpty()
            return if (mins.isBlank()) "${hours}h" else "${hours}h ${mins}m"
        }
        val minsOnly = Regex("""(?i)(\d+)\s*m(?:in(?:ute)?s?)?""").find(raw)?.groupValues?.getOrNull(1)
        if (!minsOnly.isNullOrBlank()) {
            return "${minsOnly}m"
        }
        return raw
    }

    fun splitFareDisplay(fare: String?): Pair<String?, String?> {
        val raw = NativeTextFormatter.sanitizeDisplayText(fare.orEmpty()).trim()
        if (raw.isBlank()) {
            return null to null
        }
        val compact = raw
            .replace(Regex("""\s+"""), " ")
            .replace("â‚¹", "\u20B9")
            .replace("Â₹", "\u20B9")
        val lower = compact.lowercase(Locale.US)
        val amount = Regex("""(?i)((?:[\u20B9$\u20AC\u00A3]|rs\.?|inr)\s?\d[\d,]*(?:\.\d+)?)""")
            .find(compact)
            ?.groupValues
            ?.getOrNull(1)
        val normalizedAmount = amount
            ?.replace(Regex("""\s+"""), "")
            ?.replace(Regex("""(?i)^(rs\.?|inr)"""), "\u20B9")
        val suffix = when {
            lower.contains("/adult") || lower.contains("per adult") -> "/adult"
            lower.contains("/person") || lower.contains("per person") -> "/person"
            else -> null
        }
        if (!normalizedAmount.isNullOrBlank()) {
            return normalizedAmount to suffix
        }
        return when {
            lower.contains("/adult") -> compact.replace(Regex("""(?i)\s*/\s*adult"""), "").trim() to "/adult"
            lower.contains("per adult") -> compact.replace(Regex("""(?i)\s*per\s*adult"""), "").trim() to "/adult"
            lower.contains("/person") -> compact.replace(Regex("""(?i)\s*/\s*person"""), "").trim() to "/person"
            lower.contains("per person") -> compact.replace(Regex("""(?i)\s*per\s*person"""), "").trim() to "/person"
            else -> compact to null
        }
    }

    fun normalizeStopLabel(
        rawStops: String?,
        rawStatus: String?,
        depart: String?,
        arrive: String?
    ): String? {
        canonicalizeStopLabel(rawStops)?.let { return it }
        canonicalizeStopLabel(rawStatus)?.let { return it }
        return null
    }

    fun canonicalizeStopLabel(value: String?): String? {
        val text = NativeTextFormatter.sanitizeDisplayText(value.orEmpty()).trim()
        if (text.isBlank()) {
            return null
        }
        val normalized = normalizeMatchText(text)
        return when {
            normalized.contains("non stop") || normalized.contains("nonstop") || normalized.contains("direct") ||
                normalized == "0 stop" || normalized == "0 stops" -> "Non-stop"

            Regex("""\b1\b.*\bstop""").containsMatchIn(normalized) || normalized.contains("one stop") -> "1 stop"
            Regex("""\b2\b.*\bstop""").containsMatchIn(normalized) || normalized.contains("two stop") -> "2 stops"
            normalized.contains("stop") || normalized.contains("layover") || normalized.contains("connection") -> text
            else -> null
        }
    }

    fun normalizeFlightStatus(rawStatus: String?, stopLabel: String?): String? {
        val status = NativeTextFormatter.sanitizeDisplayText(rawStatus.orEmpty()).trim()
        if (status.isBlank()) {
            return null
        }
        val normalized = normalizeMatchText(status)
        if (normalized in setOf("direct", "non stop", "nonstop")) {
            return null
        }
        if (stopLabel != null && canonicalizeStopLabel(status) != null) {
            return null
        }
        return status
    }

    fun looksLikePunctualityStatus(text: String): Boolean {
        val normalized = normalizeMatchText(text)
        return normalized.contains("on time") ||
            normalized.contains("punctual") ||
            Regex("""\b\d{1,3}\s*%""").containsMatchIn(text)
    }

    fun airlineBadgeCode(airline: String): String {
        val normalized = normalizeMatchText(airline)
        val explicit = when {
            normalized.contains("indigo") -> "6E"
            normalized.contains("air india express") -> "IX"
            normalized == "air india" || normalized.startsWith("air india ") -> "AI"
            normalized.contains("akasa") -> "QP"
            normalized.contains("vistara") -> "UK"
            normalized.contains("spicejet") -> "SG"
            normalized.contains("emirates") -> "EK"
            normalized.contains("british airways") -> "BA"
            normalized.contains("qatar") -> "QR"
            else -> ""
        }
        if (explicit.isNotBlank()) {
            return explicit
        }

        val initials = airline
            .split(Regex("""[^A-Za-z0-9]+"""))
            .filter { it.isNotBlank() }
            .take(2)
            .map { token ->
                token.firstOrNull { it.isLetterOrDigit() }?.uppercaseChar()?.toString().orEmpty()
            }
            .joinToString("")
        return initials.take(2)
    }

    fun normalizeMatchText(value: String): String {
        return value
            .lowercase(Locale.US)
            .replace(Regex("[^a-z0-9]+"), " ")
            .trim()
    }

    fun looksLikeTimeValue(value: String): Boolean {
        val normalized = value.trim().uppercase(Locale.US)
        return Regex("""^\d{1,2}:\d{2}(\s?(AM|PM))?(\+\d+)?$""").matches(normalized)
    }

    fun looksLikeDurationValue(value: String): Boolean {
        val normalized = value.trim().lowercase(Locale.US)
        return Regex("""\d+\s*h""").containsMatchIn(normalized) || Regex("""\d+\s*m""").containsMatchIn(normalized)
    }

    fun looksLikeFareValue(value: String): Boolean {
        val normalized = value.trim()
        if (normalized.isBlank()) {
            return false
        }
        val withRupee = normalized
            .replace("â‚¹", "\u20B9")
            .replace("Â₹", "\u20B9")
        return Regex("""(?i)(?:\u20B9|rs\.?|inr)\s*\d[\d,]*(?:\.\d+)?""").containsMatchIn(withRupee) ||
            Regex("""(?i)\bfrom\s*(?:\u20B9|rs\.?|inr)?\s*\d""").containsMatchIn(withRupee) ||
            Regex("""\d[\d,]{3,}(?:\.\d+)?""").matches(withRupee)
    }

    fun looksLikeAirlineValue(value: String): Boolean {
        val normalized = value.trim()
        if (normalized.length < 3) return false
        if (looksLikeTimeValue(normalized) || looksLikeDurationValue(normalized) || looksLikeFareValue(normalized)) {
            return false
        }
        if (Regex("""^[A-Z]{3}$""").matches(normalized.uppercase(Locale.US))) {
            return false
        }
        return normalized.any { it.isLetter() } && !normalized.equals("non-stop", ignoreCase = true)
    }

    fun buildFlightRows(header: List<String>, body: List<List<String>>): List<FlightRow>? {
        if (header.isEmpty() || body.isEmpty()) {
            return null
        }
        val detectedColumns = detectFlightColumns(header) ?: return null
        val columns = resolveFlightColumns(detectedColumns, body)
        val originCode = columns.depart?.let { extractAirportCode(header.getOrNull(it).orEmpty()) }
        val destinationCode = columns.arrive?.let { extractAirportCode(header.getOrNull(it).orEmpty()) }
        val rows = body.mapNotNull { row ->
            val airline = readCell(row, columns.airline).orEmpty()
            if (airline.isBlank()) {
                return@mapNotNull null
            }
            val rawDepart = readCell(row, columns.depart)
            val rawArrive = readCell(row, columns.arrive)
            val departRange = parseFlightTimeRange(rawDepart)
            val arriveRange = parseFlightTimeRange(rawArrive)
            val actionUrl = readCell(row, columns.actionUrl)?.takeIf(::looksLikeActionUrl)
            val representedColumns = setOfNotNull(
                columns.airline, columns.depart, columns.arrive, columns.duration, columns.stops,
                columns.fare, columns.status, columns.logo,
                columns.actionUrl.takeIf { actionUrl != null },
                columns.actionLabel.takeIf { actionUrl != null }
            )
            val details = header.indices.filterNot { it in representedColumns }.mapNotNull { index ->
                val value = readCell(row, index) ?: return@mapNotNull null
                val label = if (Regex("""(?:\s*\[\d+])+\s*""").matches(value)) "Sources" else header[index]
                label to value
            }
            FlightRow(
                airline = airline,
                depart = departRange?.first ?: rawDepart ?: arriveRange?.first,
                originCode = originCode,
                arrive = arriveRange?.second ?: rawArrive ?: departRange?.second,
                destinationCode = destinationCode,
                duration = readCell(row, columns.duration),
                stops = readCell(row, columns.stops),
                fare = readCell(row, columns.fare),
                status = readCell(row, columns.status),
                logoUrl = readCell(row, columns.logo)?.takeIf(::looksLikeMediaUrl),
                actionUrl = actionUrl,
                actionLabel = readCell(row, columns.actionLabel)
                    ?.takeIf { it.isNotBlank() && !looksLikeActionUrl(it) },
                details = details
            )
        }
        return rows.takeIf { it.isNotEmpty() }
    }

    fun detectFlightColumns(header: List<String>): FlightTableColumns? {
        val normalized = header.map { normalizeHeaderToken(it) }
        val flightSignal = normalized.count { token ->
            token.contains("airline") ||
                token.contains("carrier") ||
                token.contains("flight") ||
                token.contains("depart") ||
                token.contains("arrival") ||
                token.contains("arrive") ||
                token.contains("duration") ||
                token.contains("fare") ||
                token.contains("price") ||
                token.contains("cost") ||
                token.contains("stop") ||
                token.contains("layover") ||
                token.contains("status") ||
                token.contains("time")
        }
        if (flightSignal < 2) {
            return null
        }

        val logo = findHeaderIndex(normalized, listOf("logo", "icon", "image", "thumbnail", "airline logo"))
        val airline = findHeaderIndex(
            normalized,
            listOf("airline", "carrier", "operator", "flight", "route"),
            exclude = setOfNotNull(logo)
        )
            ?: return null
        val depart = findHeaderIndex(normalized, listOf("depart", "departure", "takeoff", "from", "origin"), exclude = setOfNotNull(airline, logo))
            ?: normalized.indices.firstOrNull { index ->
                index !in setOfNotNull(airline, logo) &&
                    normalized[index] in setOf("time", "times", "timing", "timings", "schedule", "flight time", "flight times")
            }
        val arrive = findHeaderIndex(normalized, listOf("arrive", "arrival", "landing", "to", "destination"), exclude = setOfNotNull(airline, logo, depart))
        val duration = findHeaderIndex(normalized, listOf("duration", "travel time", "elapsed"), exclude = setOfNotNull(airline, logo, depart, arrive))
        val stops = findHeaderIndex(normalized, listOf("stop", "stops", "layover", "connection", "type"), exclude = setOfNotNull(airline, logo, depart, arrive, duration))
        val fare = findHeaderIndex(normalized, listOf("fare", "price", "cost", "amount", "rate"), exclude = setOfNotNull(airline, logo, depart, arrive, duration, stops))
        val actionUrl = findHeaderIndex(
            normalized,
            listOf("booking url", "book url", "action url", "cta url", "url", "link", "website"),
            exclude = setOfNotNull(airline, logo, depart, arrive, duration, stops, fare)
        )
        val actionLabel = findHeaderIndex(
            normalized,
            listOf("action label", "button label", "cta label", "label"),
            exclude = setOfNotNull(airline, logo, depart, arrive, duration, stops, fare, actionUrl)
        )
        val status = findHeaderIndex(
            normalized,
            listOf("status", "on time", "punctual", "delay"),
            exclude = setOfNotNull(airline, logo, depart, arrive, duration, stops, fare, actionUrl, actionLabel)
        )

        val contentSignals = listOf(depart, arrive, duration, stops, fare, status).count { it != null }
        if (contentSignals < 2) {
            return null
        }

        return FlightTableColumns(
            airline = airline,
            depart = depart,
            arrive = arrive,
            duration = duration,
            stops = stops,
            fare = fare,
            status = status,
            logo = logo,
            actionUrl = actionUrl,
            actionLabel = actionLabel
        )
    }

    fun resolveFlightColumns(
        detected: FlightTableColumns,
        body: List<List<String>>
    ): FlightTableColumns {
        if (body.isEmpty()) {
            return detected
        }
        val columnCount = body.maxOfOrNull { it.size } ?: return detected
        val sampleRows = body.take(5)
        val indices = (0 until columnCount).toList()

        fun score(index: Int?, predicate: (String) -> Boolean): Float {
            if (index == null || index !in indices) return 0f
            val values = sampleRows.mapNotNull { row -> row.getOrNull(index)?.trim()?.takeIf { it.isNotBlank() } }
            if (values.isEmpty()) return 0f
            return values.count(predicate).toFloat() / values.size.toFloat()
        }

        fun bestIndex(
            candidates: List<Int>,
            predicate: (String) -> Boolean,
            exclude: Set<Int> = emptySet()
        ): Int? {
            return candidates
                .filterNot { it in exclude }
                .maxByOrNull { idx -> score(idx, predicate) }
                ?.takeIf { score(it, predicate) >= 0.5f }
        }

        val timeScore: (String) -> Boolean = { looksLikeTimeValue(it) || parseFlightTimeRange(it) != null }
        val durationScore: (String) -> Boolean = { looksLikeDurationValue(it) }
        val fareScore: (String) -> Boolean = { looksLikeFareValue(it) }
        val stopScore: (String) -> Boolean = { canonicalizeStopLabel(it) != null }
        val airlineScore: (String) -> Boolean = { looksLikeAirlineValue(it) }

        var airline = detected.airline
        var depart = detected.depart
        var arrive = detected.arrive
        var duration = detected.duration
        var stops = detected.stops
        var fare = detected.fare
        val logo = detected.logo
        val actionUrl = detected.actionUrl
        val actionLabel = detected.actionLabel

        val airlineLooksWrong = score(airline, timeScore) >= 0.5f || score(airline, fareScore) >= 0.5f
        if (airlineLooksWrong || score(airline, airlineScore) < 0.4f) {
            bestIndex(indices, airlineScore, exclude = setOfNotNull(depart, arrive, duration, stops, fare, logo, actionUrl, actionLabel))
                ?.let { airline = it }
        }

        if (depart == null || score(depart, timeScore) < 0.5f) {
            bestIndex(indices, timeScore, exclude = setOfNotNull(airline, arrive, logo, actionUrl, actionLabel))
                ?.let { depart = it }
        }
        if (arrive == null || score(arrive, timeScore) < 0.5f || arrive == depart) {
            bestIndex(indices, timeScore, exclude = setOfNotNull(airline, depart, logo, actionUrl, actionLabel))
                ?.let { arrive = it }
        }
        if (duration == null || score(duration, durationScore) < 0.4f) {
            duration = bestIndex(indices, durationScore, exclude = setOfNotNull(airline, depart, arrive, logo, actionUrl, actionLabel))
        }
        if (fare == null || score(fare, fareScore) < 0.4f) {
            fare = bestIndex(indices, fareScore, exclude = setOfNotNull(airline, depart, arrive, duration, stops, logo, actionUrl, actionLabel))
        }
        if (stops == null || score(stops, stopScore) < 0.4f) {
            // A generated Details cell can contain both duration and stops.
            stops = bestIndex(indices, stopScore, exclude = setOfNotNull(airline, depart, arrive, fare, logo, actionUrl, actionLabel))
        }

        return detected.copy(
            airline = airline,
            depart = depart,
            arrive = arrive,
            duration = duration,
            stops = stops,
            fare = fare
        )
    }

    fun findHeaderIndex(
        normalizedHeader: List<String>,
        keywords: List<String>,
        exclude: Set<Int> = emptySet()
    ): Int? {
        return normalizedHeader.indices.firstOrNull { index ->
            index !in exclude && keywords.any { key -> normalizedHeader[index].contains(key) }
        }
    }

    fun normalizeHeaderToken(value: String): String {
        return value
            .lowercase(Locale.US)
            .replace(Regex("[^a-z0-9]+"), " ")
            .trim()
    }

    fun readCell(row: List<String>, index: Int?): String? {
        if (index == null || index !in row.indices) {
            return null
        }
        return row[index].trim().takeIf { it.isNotEmpty() }
    }

    private fun looksLikeMediaUrl(value: String): Boolean {
        val trimmed = value.trim()
        return trimmed.startsWith("http://", ignoreCase = true) ||
            trimmed.startsWith("https://", ignoreCase = true) ||
            trimmed.startsWith("../assets/", ignoreCase = true) ||
            trimmed.startsWith("assets/", ignoreCase = true) ||
            trimmed.startsWith("file:", ignoreCase = true)
    }

    private fun looksLikeActionUrl(value: String): Boolean {
        val trimmed = value.trim()
        return trimmed.startsWith("https://", ignoreCase = true) ||
            trimmed.startsWith("http://", ignoreCase = true) ||
            trimmed.startsWith("www.", ignoreCase = true)
    }

    fun extractAirportCode(headerText: String): String? {
        if (headerText.isBlank()) {
            return null
        }
        val match = Regex("""\(([A-Za-z]{3})\)""").find(headerText)
        return match?.groupValues?.getOrNull(1)?.uppercase(Locale.US)
    }
}
