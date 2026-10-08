package com.samsung.genuicraft.sdk.internal.renderer.native.intents.airquality

import java.util.Locale

internal data class NativeAirQualityField(val label: String, val value: String)

internal data class NativeAirQualitySummary(
    val fields: List<NativeAirQualityField>,
    val readingIndex: Int,
    val categoryIndex: Int?,
) {
    val reading: NativeAirQualityField get() = fields[readingIndex]
    val detailFields: List<NativeAirQualityField> get() = fields.filterIndexed { index, _ -> index != readingIndex }
}

/** A narrow, lossless key/value AQI profile. Supplied categories never come from numeric thresholds. */
internal object NativeAirQualitySemantics {
    fun buildSummary(
        headers: List<String>,
        rows: List<List<String>>,
        preferredPresentation: String? = null,
    ): NativeAirQualitySummary? {
        if (preferredPresentation?.trim()?.equals("table", ignoreCase = true) == true) return null
        if (headers.size != 2 || rows.isEmpty() || rows.any { it.size > 2 }) return null
        val normalizedHeaders = headers.map(::normalize)
        if (normalizedHeaders[0] !in setOf("detail", "details", "metric", "field", "label", "attribute", "key") ||
            normalizedHeaders[1] !in setOf("value", "values", "reading", "result")) return null

        // Matching uses labels only; every original field, unknown value and duplicate is retained.
        val fields = rows.map { NativeAirQualityField(it.getOrNull(0).orEmpty(), it.getOrNull(1).orEmpty()) }
        val labels = fields.map { normalize(it.label) }
        val readingIndex = labels.indices.firstOrNull { index ->
            val label = labels[index]
            ("aqi" in label.split(' ') || label.contains("air quality index")) &&
                label.split(' ').none { it in setOf("category", "status", "pollutant", "time", "date") } &&
                fields[index].value.isNotBlank()
        } ?: return null
        val categoryIndex = labels.indexOfFirst { label ->
            label == "category" || label in setOf("pollution category", "air quality category", "aqi category", "pollution level")
        }.takeIf { it >= 0 }
        val pollutantPresent = labels.any { "pollutant" in it.split(' ') }
        if (categoryIndex == null && !pollutantPresent) return null
        return NativeAirQualitySummary(fields, readingIndex, categoryIndex)
    }

    private fun normalize(value: String): String = value.trim().lowercase(Locale.ROOT)
        .replace(Regex("[^a-z0-9]+"), " ").trim().replace(Regex("\\s+"), " ")
}
