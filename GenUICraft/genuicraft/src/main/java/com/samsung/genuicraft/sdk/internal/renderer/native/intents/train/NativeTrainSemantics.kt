package com.samsung.genuicraft.sdk.internal.renderer.native.intents.train

import java.util.Locale

internal data class NativeTrainField(
    val label: String,
    val value: String
)

internal data class NativeTrainRow(
    val service: NativeTrainField,
    val station: NativeTrainField?,
    val departure: NativeTrainField?,
    val duration: NativeTrainField?,
    val seating: NativeTrainField?,
    val extraFields: List<NativeTrainField>
)

/**
 * Losslessly maps a table to the small rail profile used by native train cards.
 *
 * Detection is deliberately header/title based. Cell values are never parsed or corrected: malformed
 * times, source placeholders, duplicate rows, and any other generated text remain authoritative.
 */
internal object NativeTrainSemantics {
    fun buildTrainRows(
        headers: List<String>,
        rows: List<List<String>>,
        title: String? = null
    ): List<NativeTrainRow>? {
        if (headers.isEmpty() || rows.isEmpty()) return null

        val normalizedHeaders = headers.map(::normalizeForMatch)
        if (!hasRailEvidence(normalizedHeaders, title) || hasConflictingDomainIdentity(normalizedHeaders)) {
            return null
        }

        val rolesByIndex = normalizedHeaders.map(::roleForHeader)
        val serviceIndex = rolesByIndex.indexOfFirst { it == TrainFieldRole.SERVICE }.takeIf { it >= 0 }
            ?: return null
        val stationIndex = rolesByIndex.indexOfFirst { it == TrainFieldRole.STATION }.takeIf { it >= 0 }
        val departureIndex = rolesByIndex.indexOfFirst { it == TrainFieldRole.DEPARTURE }.takeIf { it >= 0 }
        val durationIndex = rolesByIndex.indexOfFirst { it == TrainFieldRole.DURATION }.takeIf { it >= 0 }
        val seatingIndex = rolesByIndex.indexOfFirst { it == TrainFieldRole.SEATING }.takeIf { it >= 0 }

        val profileIndexes = listOfNotNull(stationIndex, departureIndex, durationIndex, seatingIndex)
        if ((stationIndex == null && departureIndex == null) || profileIndexes.size < 2) return null

        val representedIndexes = setOfNotNull(
            serviceIndex,
            stationIndex,
            departureIndex,
            durationIndex,
            seatingIndex
        )
        val mappedRows = ArrayList<NativeTrainRow>(rows.size)
        for (row in rows) {
            val service = fieldAt(headers, row, serviceIndex, TrainFieldRole.SERVICE) ?: return null
            val station = stationIndex?.let { fieldAt(headers, row, it, TrainFieldRole.STATION) }
            val departure = departureIndex?.let { fieldAt(headers, row, it, TrainFieldRole.DEPARTURE) }
            val duration = durationIndex?.let { fieldAt(headers, row, it, TrainFieldRole.DURATION) }
            val seating = seatingIndex?.let { fieldAt(headers, row, it, TrainFieldRole.SEATING) }

            // A row with only an identity cannot be rendered as a train schedule without guessing.
            if (station == null && departure == null) return null

            val extraFields = row.indices.mapNotNull { index ->
                if (index in representedIndexes) return@mapNotNull null
                val value = row[index]
                if (value.isBlank()) return@mapNotNull null
                NativeTrainField(
                    label = extraFieldLabel(headers.getOrNull(index), index),
                    value = value
                )
            }
            mappedRows += NativeTrainRow(
                service = service,
                station = station,
                departure = departure,
                duration = duration,
                seating = seating,
                extraFields = extraFields
            )
        }
        return mappedRows
    }

    private fun fieldAt(
        headers: List<String>,
        row: List<String>,
        index: Int,
        role: TrainFieldRole
    ): NativeTrainField? {
        val value = row.getOrNull(index) ?: return null
        if (value.isBlank()) return null
        return NativeTrainField(
            label = semanticLabel(headers.getOrNull(index).orEmpty(), role),
            value = value
        )
    }

    private fun hasRailEvidence(normalizedHeaders: List<String>, title: String?): Boolean {
        val normalizedTitle = normalizeForMatch(title.orEmpty())
        val titleSignalsRail = normalizedTitle.hasToken("train") ||
            normalizedTitle.hasToken("rail") ||
            normalizedTitle.hasToken("railway")
        val headersSignalRail = normalizedHeaders.any { header ->
            header.hasToken("train") || header.hasToken("rail") || header.hasToken("railway")
        }
        return titleSignalsRail || headersSignalRail
    }

    private fun hasConflictingDomainIdentity(normalizedHeaders: List<String>): Boolean {
        return normalizedHeaders.any { header ->
            header.hasAnyToken("airline", "flight", "airport", "carrier", "iata", "layover") ||
                header.hasAnyToken(
                    "weather",
                    "forecast",
                    "temperature",
                    "humidity",
                    "precipitation"
                ) ||
                header.hasAnyToken("product", "brand", "seller", "sku")
        }
    }

    private fun roleForHeader(header: String): TrainFieldRole? {
        return when {
            header in SERVICE_HEADERS ||
                (header.hasAnyToken("train", "rail", "railway") &&
                    header.hasAnyToken("service", "name", "number", "no")) -> TrainFieldRole.SERVICE

            header in DEPARTURE_TIME_HEADERS ||
                (header.hasAnyToken("departure", "depart") &&
                    header.hasAnyToken("time", "schedule")) -> TrainFieldRole.DEPARTURE

            header in STATION_HEADERS ||
                (header.hasAnyToken("departure", "departing", "origin", "boarding", "from") &&
                    header.hasToken("station")) -> TrainFieldRole.STATION

            header in DURATION_HEADERS || header.hasToken("duration") ||
                (header.hasAnyToken("journey", "travel", "elapsed") && header.hasToken("time")) ->
                TrainFieldRole.DURATION

            header in SEATING_HEADERS ||
                (header.hasAnyToken("seat", "seating", "travel", "coach") &&
                    header.hasAnyToken("class", "classes")) -> TrainFieldRole.SEATING

            else -> null
        }
    }

    private fun semanticLabel(rawHeader: String, role: TrainFieldRole): String {
        val normalized = normalizeForMatch(rawHeader)
        val terseAlias = when (role) {
            TrainFieldRole.SERVICE -> normalized == "service"
            TrainFieldRole.STATION -> normalized in setOf("departure", "origin", "from")
            TrainFieldRole.DEPARTURE -> normalized == "departure time"
            TrainFieldRole.DURATION -> normalized == "duration"
            TrainFieldRole.SEATING -> normalized in setOf("class", "classes")
        }
        if (!terseAlias) return friendlyHeaderLabel(rawHeader)
        return when (role) {
            TrainFieldRole.SERVICE -> "Service"
            TrainFieldRole.STATION -> "Departure station"
            TrainFieldRole.DEPARTURE -> "Departure time"
            TrainFieldRole.DURATION -> "Journey duration"
            TrainFieldRole.SEATING -> "Seating class"
        }
    }

    private fun extraFieldLabel(rawHeader: String?, index: Int): String {
        return rawHeader
            ?.takeIf { it.isNotBlank() }
            ?.let(::friendlyHeaderLabel)
            ?: "Column ${index + 1}"
    }

    private fun friendlyHeaderLabel(rawHeader: String): String {
        val trimmed = rawHeader.trim()
        if (trimmed.isEmpty()) return trimmed

        val isMachineHeader = '_' in trimmed || CAMEL_CASE_BOUNDARY.containsMatchIn(trimmed) ||
            (trimmed.none(Char::isWhitespace) && trimmed.all { it.isLowerCase() || it.isDigit() })
        if (!isMachineHeader) return trimmed

        val words = trimmed
            .replace(CAMEL_CASE_BOUNDARY, "$1 $2")
            .replace('_', ' ')
            .trim()
            .split(Regex("""\s+"""))
            .filter(String::isNotBlank)
        return words.mapIndexed { index, word ->
            when {
                word.any(Char::isDigit) || (word.length > 1 && word.all(Char::isUpperCase)) -> word
                index == 0 -> word.lowercase(Locale.US).replaceFirstChar(Char::uppercaseChar)
                else -> word.lowercase(Locale.US)
            }
        }.joinToString(" ")
    }

    private fun normalizeForMatch(value: String): String {
        return value
            .replace(CAMEL_CASE_BOUNDARY, "$1 $2")
            .lowercase(Locale.US)
            .replace(Regex("""[^a-z0-9]+"""), " ")
            .trim()
            .replace(Regex("""\s+"""), " ")
    }

    private fun String.hasToken(token: String): Boolean = split(' ').any { it == token }

    private fun String.hasAnyToken(vararg tokens: String): Boolean {
        val words = split(' ').toSet()
        return tokens.any { it in words }
    }

    private enum class TrainFieldRole {
        SERVICE,
        STATION,
        DEPARTURE,
        DURATION,
        SEATING
    }

    private val CAMEL_CASE_BOUNDARY = Regex("([a-z0-9])([A-Z])")

    private val SERVICE_HEADERS = setOf(
        "service",
        "train",
        "train service",
        "rail service",
        "railway service",
        "train name",
        "service name",
        "train number",
        "train no"
    )
    private val STATION_HEADERS = setOf(
        "departure",
        "departure station",
        "departing station",
        "origin",
        "origin station",
        "boarding station",
        "from",
        "from station"
    )
    private val DEPARTURE_TIME_HEADERS = setOf(
        "departure time",
        "typical departure time",
        "depart time",
        "scheduled departure",
        "scheduled departure time",
        "departure schedule",
        "time of departure"
    )
    private val DURATION_HEADERS = setOf(
        "duration",
        "journey duration",
        "journey time",
        "travel time",
        "elapsed time",
        "typical journey duration"
    )
    private val SEATING_HEADERS = setOf(
        "class",
        "classes",
        "seating class",
        "seat class",
        "travel class",
        "coach class",
        "available classes",
        "class availability"
    )
}
