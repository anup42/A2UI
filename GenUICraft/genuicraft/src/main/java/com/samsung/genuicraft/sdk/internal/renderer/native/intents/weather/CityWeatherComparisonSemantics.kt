package com.samsung.genuicraft.sdk.internal.renderer.native.intents.weather

import java.util.Locale

internal enum class CityWeatherFieldKind { CURRENT, CONDITION, HIGH, LOW, RAIN, OTHER }

internal data class CityWeatherField(
    val label: String,
    val value: String,
    val kind: CityWeatherFieldKind,
)

internal data class CityWeatherComparisonRow(
    val cityLabel: String,
    val city: String,
    val fields: List<CityWeatherField>,
) {
    /** Use only supplied condition text for artwork. Never infer a forecast from high/low or rain notes. */
    val iconCondition: String?
        get() = fields.firstOrNull { it.kind == CityWeatherFieldKind.CONDITION && it.value.isNotBlank() }?.value
            ?: fields.firstOrNull { it.kind == CityWeatherFieldKind.CURRENT && it.value.isNotBlank() }?.value
}

/** A narrow route for entity rows that compare weather across distinct cities. */
internal object CityWeatherComparisonSemantics {
    fun buildRows(
        headers: List<String>,
        rows: List<List<String>>,
        domain: String,
    ): List<CityWeatherComparisonRow>? {
        if (domain !in setOf("weather", "comparison") || headers.size < 3 || rows.size < 2) return null
        val cityHeader = headers.first()
        if (normalize(cityHeader) !in setOf("city", "cities", "location", "place")) return null

        val fieldKinds = headers.drop(1).map(::fieldKind)
        val weatherFields = fieldKinds.count { it != CityWeatherFieldKind.OTHER }
        if (weatherFields < 2 ||
            fieldKinds.none { it in setOf(CityWeatherFieldKind.CURRENT, CityWeatherFieldKind.CONDITION,
                CityWeatherFieldKind.HIGH, CityWeatherFieldKind.LOW) }
        ) return null

        val distinctCities = HashSet<String>()
        val result = rows.map { row ->
            val city = row.firstOrNull()?.trim().orEmpty()
            if (city.isBlank() || !distinctCities.add(city.lowercase(Locale.ROOT)) ||
                row.drop(headers.size).any { it.isNotBlank() }
            ) return null
            CityWeatherComparisonRow(
                cityLabel = cityHeader,
                city = city,
                fields = headers.drop(1).mapIndexed { index, label ->
                    CityWeatherField(label, row.getOrNull(index + 1).orEmpty(), fieldKinds[index])
                },
            )
        }
        if (result.any { cityRow -> cityRow.fields.none { it.value.isNotBlank() && it.kind != CityWeatherFieldKind.OTHER } }) {
            return null
        }
        return result
    }

    private fun fieldKind(header: String): CityWeatherFieldKind {
        val token = normalize(header)
        val words = token.split(' ')
        return when {
            token.contains("rain") || token.contains("precip") -> CityWeatherFieldKind.RAIN
            "high" in words || token.contains("maximum temperature") || token == "max temp" -> CityWeatherFieldKind.HIGH
            "low" in words || token.contains("minimum temperature") || token == "min temp" -> CityWeatherFieldKind.LOW
            token.contains("current") || token.contains("temperature now") -> CityWeatherFieldKind.CURRENT
            token.contains("condition") || token.contains("weather") || token.contains("forecast") -> CityWeatherFieldKind.CONDITION
            else -> CityWeatherFieldKind.OTHER
        }
    }

    private fun normalize(text: String): String = text.lowercase(Locale.ROOT)
        .replace(Regex("[^a-z0-9]+"), " ")
        .trim()
}
