package com.samsung.genuicraft.sdk.internal.renderer.flat.domain

import java.util.Locale

internal data class SightseeingTableProfile(val primaryIndex: Int, val detailIndexes: List<Int>)

/** Only named places with multiple explicit visit facts qualify; cell values are never interpreted. */
internal fun sightseeingTableProfile(headers: List<String>, rows: List<List<String>>): SightseeingTableProfile? {
    if (headers.isEmpty() || rows.isEmpty()) return null
    val normalized = headers.map { it.lowercase(Locale.US).replace(Regex("[_-]+"), " ").trim() }
    if (normalized.any { it in setOf("feature", "factor", "point", "step", "stage", "task", "action", "status") }) return null
    val primary = normalized.indexOfFirst {
        it in setOf("sight", "sights", "sight name", "place", "place name", "attraction", "attractions", "attraction name", "point of interest")
    }.takeIf { it >= 0 } ?: return null
    val hours = normalized.indexOfFirst { it.contains("hours") || it.contains("opening time") }.takeIf { it >= 0 }
    val duration = normalized.indexOfFirst { it.contains("duration") }.takeIf { it >= 0 }
    val fee = normalized.indexOfFirst {
        it.contains("fee") || it.contains("admission") || (it.contains("entry") && (it.contains("price") || it.contains("cost")))
    }.takeIf { it >= 0 }
    val details = listOfNotNull(hours, duration, fee).distinct()
    if (details.size < 2 || rows.any { it.size > headers.size || it.getOrNull(primary).isNullOrBlank() }) return null
    return SightseeingTableProfile(primary, details)
}
