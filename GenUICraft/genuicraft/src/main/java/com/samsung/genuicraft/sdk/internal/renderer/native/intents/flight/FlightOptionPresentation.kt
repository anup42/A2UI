package com.samsung.genuicraft.sdk.internal.renderer.native.intents.flight

import com.samsung.genuicraft.sdk.internal.renderer.native.FlightRow
import java.util.Locale

/**
 * Narrow adapter for the Option/Details cards produced from literal flight-offer lists.
 * Entries stay index-aligned so a caller can render a native flight card for a clear offer and
 * keep the original generic entity card for an ambiguous aggregator or multi-fare entry.
 */
internal object FlightOptionPresentation {
    private val fareAmount = Regex("""(?i)(?:₹|INR\b|Rs\.?|\$|€|£)\s*\d[\d,]*(?:\.\d+)?""")
    private val fareQualifier = Regex(
        """(?i)(?:\bstarting\s+from\s+about|\bstarting\s+from|\bstarting\s+at|\bfrom|\babout|\baround|\bapprox(?:imately)?\.?)\s*(?:\*\*)?$"""
    )
    private val fareRangeTail = Regex(
        """(?i)^(?:\*\*)?\s*(?:[-–—]|\bto\b)\s*(?:(?:₹|INR\b|Rs\.?|\$|€|£)\s*)?\d[\d,]*(?:\.\d+)?"""
    )
    private val adjacentCitations = Regex("""^(?:\*\*)?(\s*(?:\[\d+\])+)""")
    private val clockLike = Regex("""(?<![\d:])\d{1,2}:\d{2}(?![\d:])""")
    private val unsupportedOffset = Regex(
        """(?i)(?<![\d:])\d{1,2}:\d{2}(?:\s*[AP]M)?\s*(?:\+(?!\d)|\(\s*\+(?!\d))"""
    )
    private val departure = Regex(
        """(?i)\b(?:departing|departure|departs|leaving|leaves)\b(?:\s+at|\s*:)?\s*\*{0,2}((?<![\d:])\d{1,2}:[0-5]\d(?:\s*[AP]M)?(?:\s*(?:\(\s*\+\d+\s*\)|\+\d+))?)(?![\d:])\*{0,2}"""
    )
    private val arrival = Regex(
        """(?i)\b(?:arriving|arrival|arrives)\b(?:\s+at|\s*:)?\s*\*{0,2}((?<![\d:])\d{1,2}:[0-5]\d(?:\s*[AP]M)?(?:\s*(?:\(\s*\+\d+\s*\)|\+\d+))?)(?![\d:])\*{0,2}"""
    )
    private val departureAfterClock = Regex(
        """(?i)(?<![\d:])(\d{1,2}:[0-5]\d(?:\s*[AP]M)?(?:\s*(?:\(\s*\+\d+\s*\)|\+\d+))?)(?![\d:])\*{0,2}\s+\b(?:departure|departing|departs|leaving|leaves)\b"""
    )
    private val arrivalAfterClock = Regex(
        """(?i)(?<![\d:])(\d{1,2}:[0-5]\d(?:\s*[AP]M)?(?:\s*(?:\(\s*\+\d+\s*\)|\+\d+))?)(?![\d:])\*{0,2}\s+\b(?:arrival|arriving|arrives)\b"""
    )
    private val offerLanguage = Regex("""(?i)\b(?:flight|flights|fare|fares|one-way|non-stop|nonstop|departing|departure|arriving|arrival)\b""")
    private val nonStop = Regex("""(?i)\bnon[- ]?stop\b""")
    private val missingFareNote = Regex(
        """(?i)\b(?:fare|price|cost)\s+(?:is\s+)?(?:not\s+(?:shown|available|provided|listed)|unavailable)\b"""
    )
    private val flightNumber = Regex("""(?i)^[A-Z0-9]{2,3}[- ]?\d{1,4}[A-Z]?$""")
    private val knownAirlines = setOf(
        "akasa air", "air india", "air india express", "indigo", "vistara", "spicejet",
        "emirates", "british airways", "qatar airways", "lufthansa", "singapore airlines",
    )
    private val aggregatorNames = setOf("ixigo", "easemytrip", "cleartrip", "makemytrip", "google flights", "skyscanner")

    /** Returns null for a non-flight table; otherwise one nullable native row per authored row. */
    fun buildFlightOptionRows(headers: List<String>, rows: List<List<String>>): List<FlightRow?>? {
        if (headers.size != 2 || rows.size !in 2..24) return null
        val first = headers[0].trim().lowercase(Locale.ROOT)
        val second = headers[1].trim().lowercase(Locale.ROOT)
        if (first !in setOf("option", "airline") || second != "details") return null
        if (rows.any { it.size != 2 }) return null
        val parsed = rows.map { row -> parseOffer(row[0], row[1], declaredAirline = first == "airline") }
        return parsed.takeIf { offers -> offers.count { it != null } >= 2 }
    }

    private fun parseOffer(title: String, details: String, declaredAirline: Boolean): FlightRow? {
        val airline = title.trim()
        if (airline != title || airline.length !in 3..44 || details.isBlank() ||
            !isAirline(airline, declaredAirline)) return null
        val fares = fareAmount.findAll(details).toList()
        // A route summary or fare range must not be presented as one airline's single offer.
        val missingFare = missingFareNote.containsMatchIn(details)
        if (fares.size > 1 || (fares.isEmpty() && !missingFare) || (fares.isNotEmpty() && missingFare)) return null
        val fare = fares.singleOrNull()
        val afterFare = fare?.let { details.substring(it.range.last + 1) }.orEmpty()
        if (fare != null && (fareRangeTail.containsMatchIn(afterFare) ||
            afterFare.removePrefix("**").trimStart().startsWith("+"))) return null
        if (!offerLanguage.containsMatchIn(details)) return null

        val departureMatches = departure.findAll(details).toList() + departureAfterClock.findAll(details).toList()
        val arrivalMatches = arrival.findAll(details).toList() + arrivalAfterClock.findAll(details).toList()
        if (unsupportedOffset.containsMatchIn(details)) return null
        if (departureMatches.size > 1 || arrivalMatches.size > 1) return null
        val hasOneEndpoint = departureMatches.isNotEmpty() != arrivalMatches.isNotEmpty()
        if (hasOneEndpoint) return null
        val depart = departureMatches.singleOrNull()?.groupValues?.get(1)
            ?.let(NativeFlightSemantics::normalizeFlightTime)
        val arrive = arrivalMatches.singleOrNull()?.groupValues?.get(1)
            ?.let(NativeFlightSemantics::normalizeFlightTime)
        if (departureMatches.isNotEmpty() && (depart == null || arrive == null)) return null
        if (missingFare && (depart == null || arrive == null)) return null
        // Every clock must be the labeled endpoint we display. In particular, an unlabeled
        // connection/window or a malformed endpoint stays in the original generic card.
        val clocks = clockLike.findAll(details).toList()
        if (clocks.size != departureMatches.size + arrivalMatches.size) return null
        if (Regex("""(?i)\b(?:departing|departure|departs|leaving|leaves|arriving|arrival|arrives)\b""")
                .containsMatchIn(details) && departureMatches.isEmpty()) return null

        val fareText = if (fare == null) {
            "Fare not shown"
        } else {
            val beforeFare = details.substring(0, fare.range.first)
            val qualifier = fareQualifier.find(beforeFare)?.value
                ?.replace("**", "")?.trim()
            val citationSuffix = adjacentCitations.find(afterFare)?.groupValues?.get(1).orEmpty()
            listOfNotNull(qualifier?.takeIf { it.isNotBlank() }, fare.value).joinToString(" ") + citationSuffix
        }
        return FlightRow(
            airline = airline,
            depart = depart,
            originCode = null,
            arrive = arrive,
            destinationCode = null,
            duration = null,
            stops = nonStop.find(details)?.value,
            fare = fareText,
            status = null,
            details = listOf("Details" to details),
        )
    }

    private fun isAirline(title: String, declaredAirline: Boolean): Boolean {
        val normalized = title.lowercase(Locale.ROOT).replace(Regex("""\s+"""), " ")
        if (normalized in aggregatorNames || title.any { it in "[]:{}" }) return false
        if (normalized in knownAirlines) return true
        val knownPrefix = knownAirlines.filter { normalized.startsWith("$it ") }.maxByOrNull { it.length }
        if (knownPrefix != null) return flightNumber.matches(normalized.removePrefix("$knownPrefix ").trim())
        val airlineToken = Regex("""(?i)\b(?:air|airlines|airways|aviation|jet)\b""").containsMatchIn(title)
        if (airlineToken) return true
        // An explicit Airline column is additional evidence, but still require a name-like value.
        return declaredAirline && title.firstOrNull()?.isUpperCase() == true &&
            title.any(Char::isLetter) && title.none(Char::isDigit)
    }
}
