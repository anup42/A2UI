package com.samsung.genuicraft.renderer.native.intents.weather

import com.samsung.genuicraft.renderer.native.WeatherRow

/** Presentation selects existing fields only; it never supplies weather conditions or readings. */
internal object NativeWeatherPresentationPolicy {
    /** Draw a probability rail only for an explicit, bounded percentage, never an amount or range. */
    fun precipitationFraction(value: String): Float? {
        val match = Regex("""^\s*(\d{1,3}(?:\.\d+)?)\s*%\s*(?:\[\d+])?\s*$""").matchEntire(value) ?: return null
        val percent = match.groupValues[1].toFloatOrNull()?.takeIf { it in 0f..100f } ?: return null
        return percent / 100f
    }

    fun shouldUseForecastPanel(rows: List<WeatherRow>): Boolean =
        rows.size > 1 || rows.singleOrNull()?.let { row ->
            !row.high.isNullOrBlank() && !row.low.isNullOrBlank() && row.temp.isNullOrBlank()
        } == true

    fun hasExplicitCondition(condition: String?): Boolean = !condition.isNullOrBlank()

    fun precipitationIndex(metrics: List<Pair<String, String>>): Int {
        fun isPrecipitation(metric: Pair<String, String>): Boolean {
            val (label, value) = metric
            val normalized = label.lowercase(java.util.Locale.ROOT)
            return value.isNotBlank() &&
                (normalized.contains("rain") || normalized.contains("precip") || normalized.contains("shower") || normalized.trim() == "pop")
        }
        // Prefer a bounded chance when a row supplies both rainfall amount and probability.
        val probabilityIndex = metrics.indexOfFirst {
            isPrecipitation(it) && isProbabilityLabel(it.first) && precipitationFraction(it.second) != null
        }
        return if (probabilityIndex >= 0) probabilityIndex else metrics.indexOfFirst(::isPrecipitation)
    }

    private fun isProbabilityLabel(label: String): Boolean {
        val text = label.lowercase(java.util.Locale.ROOT)
        return text.contains("probab") || text.contains("chance") || text.trim() == "pop"
    }

    /** A probability symbol describes the supplied metric, not an invented daily condition. */
    fun forecastArtwork(row: WeatherRow): WeatherArtworkKind {
        if (hasExplicitCondition(row.condition)) return NativeWeatherArtwork.kindFor(row.condition, row.period)
        val precipitation = row.metrics.getOrNull(precipitationIndex(row.metrics))
        val fraction = precipitation?.takeIf { isProbabilityLabel(it.first) }?.second?.let(::precipitationFraction)
        return when {
            fraction == null -> WeatherArtworkKind.TEMPERATURE
            fraction > 0f -> WeatherArtworkKind.RAIN
            else -> WeatherArtworkKind.DROP
        }
    }

    fun heroMetrics(row: WeatherRow, temperature: String): List<Pair<String, String>> = buildList {
        row.high?.takeIf { it.isNotBlank() }?.let { add("High" to it) }
        row.low?.takeIf { it.isNotBlank() }?.let { add("Low" to it) }
        row.temp?.takeIf { it.isNotBlank() }?.let { add("Temperature" to it) }
        if (isEmpty() && temperature.isNotBlank()) add("Temperature" to temperature)
        addAll(row.metrics)
    }

    fun accessibilityDescription(row: WeatherRow): String = buildList {
        row.period.takeIf { it.isNotBlank() }?.let(::add)
        row.date?.takeIf { it.isNotBlank() && it != row.period }?.let(::add)
        row.condition?.takeIf { it.isNotBlank() }?.let(::add)
        heroMetrics(row, "").forEach { (label, value) ->
            listOf(label, value).filter { it.isNotBlank() }.joinToString(": ")
                .takeIf { it.isNotBlank() }?.let(::add)
        }
    }.joinToString(". ").ifBlank { "Weather forecast item" }
}
