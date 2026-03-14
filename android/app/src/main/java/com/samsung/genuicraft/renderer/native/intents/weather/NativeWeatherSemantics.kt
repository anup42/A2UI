package com.samsung.genuicraft.renderer.native.intents.weather

import com.samsung.genuicraft.renderer.native.TextBlock
import com.samsung.genuicraft.renderer.native.WeatherCurrentDetails
import com.samsung.genuicraft.renderer.native.WeatherRow
import com.samsung.genuicraft.renderer.native.WeatherTableColumns
import com.samsung.genuicraft.renderer.native.NativeTextFormatter
import java.time.LocalDate
import java.time.LocalTime
import java.util.Locale

internal object NativeWeatherSemantics {
    private val TABLE_PLACEHOLDER_CELL_REGEX = Regex("""^[:\-\u2013\u2014]+$""")

    fun buildWeatherRows(
        header: List<String>,
        body: List<List<String>>
    ): List<WeatherRow>? {
        if (header.isEmpty() || body.isEmpty()) {
            return null
        }
        val columns = detectWeatherColumns(header) ?: return null
        val rows = body.mapNotNull { row ->
            val period = normalizeWeatherCell(readCell(row, columns.period)).orEmpty()
            if (period.isBlank()) {
                return@mapNotNull null
            }

            val metrics = buildList {
                normalizeWeatherCell(readCell(row, columns.precip))?.let { add("Precip" to it) }
                normalizeWeatherCell(readCell(row, columns.wind))?.let { add("Wind" to it) }
                normalizeWeatherCell(readCell(row, columns.humidity))?.let { add("Humidity" to it) }
                normalizeWeatherCell(readCell(row, columns.uv))?.let { add("UV" to it) }
            }

            WeatherRow(
                period = period,
                date = normalizeWeatherCell(readCell(row, columns.date)),
                condition = normalizeWeatherCell(readCell(row, columns.condition)),
                high = normalizeWeatherCell(readCell(row, columns.high)),
                low = normalizeWeatherCell(readCell(row, columns.low)),
                temp = normalizeWeatherCell(readCell(row, columns.temp)),
                metrics = metrics
            )
        }
        return rows.takeIf { it.isNotEmpty() }
    }

    fun detectWeatherColumns(header: List<String>): WeatherTableColumns? {
        val normalized = header.map { normalizeWeatherHeader(it) }
        val weatherSignal = normalized.count { token ->
            token.contains("weather") ||
                token.contains("forecast") ||
                token.contains("condition") ||
                token.contains("temp") ||
                token.contains("high") ||
                token.contains("low") ||
                token.contains("precip") ||
                token.contains("rain") ||
                token.contains("humidity") ||
                token.contains("wind") ||
                token.contains("uv")
        }
        if (weatherSignal < 2) {
            return null
        }

        val period = findHeaderIndex(normalized, listOf("day", "time", "hour", "period", "date")) ?: return null
        val date = findHeaderIndex(normalized, listOf("date"), exclude = setOf(period))
        val condition = findHeaderIndex(normalized, listOf("condition", "forecast", "weather", "summary"), exclude = setOf(period))
        val temp = findHeaderIndex(
            normalized,
            listOf("temperature", "temp", "high low", "high low c", "high low f"),
            exclude = setOf(period) + listOfNotNull(date, condition)
        )
        val high = findHeaderIndex(normalized, listOf("high", "max"), exclude = setOf(period) + listOfNotNull(date, condition, temp))
        val low = findHeaderIndex(normalized, listOf("low", "min"), exclude = setOf(period) + listOfNotNull(date, condition, temp, high))
        val precip = findHeaderIndex(normalized, listOf("precip", "rain", "chance", "pop"), exclude = setOf(period))
        val wind = findHeaderIndex(normalized, listOf("wind"), exclude = setOf(period))
        val humidity = findHeaderIndex(normalized, listOf("humidity"), exclude = setOf(period))
        val uv = findHeaderIndex(normalized, listOf("uv"), exclude = setOf(period))

        val contentSignals = listOf(condition, temp, high, low, precip, wind, humidity, uv).count { it != null }
        if (contentSignals < 2) {
            return null
        }

        return WeatherTableColumns(
            period = period,
            date = date,
            condition = condition,
            high = high,
            low = low,
            temp = temp,
            precip = precip,
            wind = wind,
            humidity = humidity,
            uv = uv
        )
    }

    fun weatherTemperatureText(row: WeatherRow): String {
        val temp = formatTemperatureCellValue(row.temp)
        if (!temp.isNullOrBlank()) {
            return temp
        }
        val high = formatTemperatureCellValue(row.high)
        val low = formatTemperatureCellValue(row.low)
        if (!high.isNullOrBlank() && !low.isNullOrBlank()) {
            return "$high / $low"
        }
        return high ?: low ?: ""
    }

    fun formatTemperatureCellValue(value: String?): String? {
        val raw = normalizeWeatherCell(value) ?: return null
        var formatted = raw
            .replace("º", "°")
            .replace("℃", "°C")
            .replace("℉", "°F")
            .replace(
                Regex("""(?i)(-?\d{1,2}(?:\.\d+)?)\s*(?:°\s*)?([CF])\b""")
            ) { match ->
                val number = match.groupValues[1]
                val unit = match.groupValues[2].uppercase(Locale.US)
                "$number°$unit"
            }
        if (formatted.contains('°')) {
            return formatted
        }

        val rangePattern = Regex(
            """^\s*-?\d{1,2}(?:\.\d+)?\s*(?:/|–|-|to)\s*-?\d{1,2}(?:\.\d+)?\s*$""",
            RegexOption.IGNORE_CASE
        )
        if (rangePattern.matches(formatted)) {
            formatted = formatted.replace(Regex("""-?\d{1,2}(?:\.\d+)?""")) { match ->
                "${match.value}°"
            }
            return formatted
        }

        val single = Regex("""^\s*(-?\d{1,2}(?:\.\d+)?)\s*$""").matchEntire(formatted)
        if (single != null) {
            return "${single.groupValues[1]}°"
        }
        return formatted
    }

    fun orderWeatherRows(rows: List<WeatherRow>): List<WeatherRow> {
        val todayIndex = rows.indexOfFirst { isTodayWeatherRow(it) }
        if (todayIndex <= 0) {
            return rows
        }
        return buildList {
            add(rows[todayIndex])
            rows.forEachIndexed { index, row ->
                if (index != todayIndex) {
                    add(row)
                }
            }
        }
    }

    fun isTodayWeatherRow(row: WeatherRow): Boolean {
        val period = normalizeWeatherText(row.period)
        val date = normalizeWeatherText(row.date.orEmpty())
        if (period.contains("today") || date.contains("today")) {
            return true
        }
        val today = LocalDate.now()
        val fullDay = today.dayOfWeek.getDisplayName(java.time.format.TextStyle.FULL, Locale.US).lowercase(Locale.US)
        val shortDay = today.dayOfWeek.getDisplayName(java.time.format.TextStyle.SHORT, Locale.US).lowercase(Locale.US)
        return period.contains(fullDay) || period.contains(shortDay) || date.contains(fullDay) || date.contains(shortDay)
    }

    fun normalizeWeatherText(value: String): String =
        value.lowercase(Locale.US).replace(Regex("[^a-z0-9]+"), " ").trim()

    fun normalizeWeatherCell(value: String?): String? {
        val normalized = value?.trim().orEmpty()
        if (isPlaceholderTableCellValue(normalized)) {
            return null
        }
        return normalized
    }

    fun isCurrentWeatherHeading(title: String): Boolean {
        val normalized = normalizeWeatherText(title)
        return normalized.contains("current condition") ||
            normalized.contains("current weather") ||
            normalized == "currently" ||
            normalized.contains("now")
    }

    fun collectCurrentWeatherFollowUpLines(
        blocks: List<TextBlock>,
        startIndex: Int
    ): Pair<List<String>, Int> {
        val lines = mutableListOf<String>()
        var consumed = 0
        var cursor = startIndex
        while (cursor < blocks.size && consumed < 6) {
            val candidate = blocks[cursor]
            val text = when (candidate) {
                is TextBlock.Title -> candidate.text
                is TextBlock.Heading -> candidate.text
                is TextBlock.Paragraph -> candidate.text
                is TextBlock.Bullets -> candidate.items.joinToString(" ")
                else -> null
            }?.trim().orEmpty()
            if (text.isBlank()) {
                break
            }

            val normalized = normalizeWeatherText(text)
            val startsNewSection =
                normalized.contains("forecast") ||
                    normalized.contains("hourly") ||
                    normalized.contains("sources") ||
                    normalized.contains("quick action") ||
                    normalized.contains("recommendation")
            if (startsNewSection) {
                break
            }

            if (!isLikelyCurrentWeatherDetailLine(text)) {
                break
            }
            lines += text
            consumed++
            cursor++
        }
        return lines to consumed
    }

    fun isLikelyCurrentWeatherDetailLine(text: String): Boolean {
        val normalized = text.lowercase(Locale.US)
        return Regex("""(?<!\d)-?\d{1,2}(?:\.\d+)?\s*°""").containsMatchIn(text) ||
            normalized.contains("feels like") ||
            normalized.contains("humidity") ||
            normalized.contains("wind") ||
            normalized.contains("uv index") ||
            normalized.contains("chance of rain") ||
            normalized.contains("precip") ||
            normalized in setOf(
                "clear",
                "sunny",
                "partly cloudy",
                "cloudy",
                "overcast",
                "rain",
                "rainy",
                "thunderstorm",
                "snow",
                "fog",
                "mist"
            )
    }

    fun inferWeatherConditionFromSection(
        title: String,
        sectionBlocks: List<TextBlock>
    ): String? {
        sectionBlocks.forEach { block ->
            if (block is TextBlock.MediaCards) {
                block.entries.forEach { entry ->
                    if (entry.iconLike) {
                        inferWeatherConditionFromIconUrl(entry.url)?.let { return it }
                    }
                }
            }
        }

        val normalizedTitle = normalizeWeatherText(title)
        val weatherContext =
            normalizedTitle.contains("weather") ||
                normalizedTitle.contains("condition") ||
                normalizedTitle.contains("forecast")
        if (!weatherContext) {
            return null
        }

        val textPool = buildString {
            append(title)
            sectionBlocks.forEach { block ->
                when (block) {
                    is TextBlock.Paragraph -> {
                        append(' ')
                        append(block.text)
                    }

                    is TextBlock.Bullets -> {
                        block.items.forEach { item ->
                            append(' ')
                            append(item)
                        }
                    }

                    else -> Unit
                }
            }
        }.lowercase(Locale.US)
        return inferWeatherConditionFromTextPool(textPool)
    }

    fun extractCurrentWeatherIconUrl(sectionBlocks: List<TextBlock>): String? {
        sectionBlocks.forEach { block ->
            if (block is TextBlock.MediaCards) {
                block.entries.firstOrNull { entry ->
                    val normalized = entry.url.trim().lowercase(Locale.US)
                    val hasUsableUrl = normalized.isNotBlank() && !isLikelyPlaceholderMediaToken(normalized)
                    hasUsableUrl && (entry.iconLike || looksLikeCompactIconUrl(normalized))
                }?.let { return it.url }
            }
        }
        return null
    }

    fun inferWeatherConditionFromTextPool(textPool: String): String? {
        return when {
            textPool.contains("thunder") || textPool.contains("storm") || textPool.contains("lightning") -> "Thunderstorm"
            textPool.contains("snow") || textPool.contains("sleet") || textPool.contains("blizzard") -> "Snow"
            textPool.contains("rain") || textPool.contains("shower") || textPool.contains("drizzle") -> "Rain"
            textPool.contains("fog") || textPool.contains("mist") || textPool.contains("haze") -> "Fog"
            textPool.contains("cloud") || textPool.contains("overcast") || textPool.contains("partly cloudy") -> "Cloudy"
            textPool.contains("clear") -> "Clear"
            textPool.contains("sun") || textPool.contains("fair") -> "Sunny"
            textPool.contains("wind") || textPool.contains("breeze") -> "Windy"
            else -> null
        }
    }

    fun buildCurrentWeatherDetails(
        title: String,
        sectionBlocks: List<TextBlock>,
        fallbackCondition: String?,
        additionalLines: List<String> = emptyList()
    ): WeatherCurrentDetails? {
        if (!isCurrentWeatherHeading(title)) {
            return null
        }

        val rawLines = mutableListOf<String>()
        sectionBlocks.forEach { block ->
            when (block) {
                is TextBlock.Paragraph -> rawLines += block.text
                is TextBlock.Bullets -> rawLines += block.items
                else -> Unit
            }
        }
        rawLines += additionalLines
        if (rawLines.isEmpty()) {
            return null
        }

        val merged = rawLines.joinToString(" ")
            .replace(Regex("""\s+"""), " ")
            .trim()
        if (merged.isBlank()) {
            return null
        }

        val iconUrl = extractCurrentWeatherIconUrl(sectionBlocks)
        val iconCondition = iconUrl?.let(::inferWeatherConditionFromIconUrl)
        val condition = iconCondition
            ?: inferWeatherConditionFromTextPool(merged.lowercase(Locale.US))
            ?: fallbackCondition
        val temperature = extractTemperatureValue(merged)
        val feelsLike = extractFeelsLikeValue(merged)
        val humidity = extractHumidityValue(merged)
        val wind = extractWindValue(merged)
        val rainChance = extractRainChanceValue(merged)
        val uvIndex = extractUvIndexValue(merged)
        val summary = merged
            .split(Regex("""(?<=[.!?])\s+"""))
            .filter { it.isNotBlank() }
            .take(2)
            .joinToString(" ")
            .trim()
            .takeIf { it.isNotBlank() }

        return WeatherCurrentDetails(
            iconUrl = iconUrl,
            condition = condition,
            temperature = temperature,
            feelsLike = feelsLike,
            humidity = humidity,
            wind = wind,
            rainChance = rainChance,
            uvIndex = uvIndex,
            summary = summary
        ).takeIf {
            !it.iconUrl.isNullOrBlank() ||
                !it.condition.isNullOrBlank() ||
                !it.temperature.isNullOrBlank() ||
                !it.feelsLike.isNullOrBlank() ||
                !it.humidity.isNullOrBlank() ||
                !it.wind.isNullOrBlank() ||
                !it.rainChance.isNullOrBlank() ||
                !it.uvIndex.isNullOrBlank()
        }
    }

    fun extractTemperatureValue(text: String): String? {
        Regex("""(?<!\d)(-?\d{1,2}(?:\.\d+)?)\s*(?:°\s*([CF])|([CF]))\b""", RegexOption.IGNORE_CASE)
            .find(text)
            ?.let { match ->
                val value = match.groupValues[1]
                val unit = match.groupValues.getOrNull(2).orEmpty().ifBlank {
                    match.groupValues.getOrNull(3).orEmpty()
                }
                return formatTemperatureReading(value, unit)
            }

        Regex("""(?<!\d)(-?\d{1,2}(?:\.\d+)?)\s*°\b""")
            .find(text)
            ?.let { match ->
                return formatTemperatureReading(match.groupValues[1], null)
            }

        Regex("""(?i)\b(?:temperature|temp|currently|current)\b[^-\d]{0,16}(-?\d{1,2}(?:\.\d+)?)\s*(?:°\s*([CF])|([CF]))?""")
            .find(text)
            ?.let { match ->
                val value = match.groupValues[1]
                val unit = match.groupValues.getOrNull(2).orEmpty().ifBlank {
                    match.groupValues.getOrNull(3).orEmpty()
                }
                return formatTemperatureReading(value, unit)
            }

        return null
    }

    fun extractFeelsLikeValue(text: String): String? {
        Regex("""(?i)\bfeels?\s+like\b[^-\d]{0,16}(-?\d{1,2}(?:\.\d+)?)\s*(?:°\s*([CF])|([CF]))?""")
            .find(text)
            ?.let { match ->
                val value = match.groupValues[1]
                val unit = match.groupValues.getOrNull(2).orEmpty().ifBlank {
                    match.groupValues.getOrNull(3).orEmpty()
                }
                return formatTemperatureReading(value, unit)
            }
        return null
    }

    fun formatTemperatureReading(value: String, unit: String?): String {
        val cleanValue = value.trim()
        val cleanUnit = unit.orEmpty().trim().uppercase(Locale.US)
        return if (cleanUnit.isBlank()) "$cleanValue°" else "$cleanValue°$cleanUnit"
    }

    fun extractHumidityValue(text: String): String? =
        Regex("""(?i)\bhumidity\b[^0-9]{0,12}(\d{1,3})\s*%?""")
            .find(text)
            ?.groupValues
            ?.getOrNull(1)
            ?.let { "${it}%" }

    fun extractWindValue(text: String): String? {
        val explicit = Regex(
            """(?i)\bwinds?\b[^0-9]{0,24}(\d{1,3}(?:\.\d+)?)\s*(km/?h|kph|mph|m/s)"""
        ).find(text)
        if (explicit != null) {
            return "${explicit.groupValues[1]} ${explicit.groupValues[2]}"
        }
        val normalized = text.lowercase(Locale.US)
        return when {
            normalized.contains("wind is calm") || normalized.contains("winds are calm") || normalized.contains("calm wind") -> "Calm"
            normalized.contains("breezy") -> "Breezy"
            else -> null
        }
    }

    fun extractRainChanceValue(text: String): String? {
        Regex("""(?i)(\d{1,3})\s*%\s*(?:chance of rain|chance of precipitation|rain chance|precip(?:itation)? chance)""")
            .find(text)
            ?.groupValues
            ?.getOrNull(1)
            ?.let { return "${it}%" }
        Regex("""(?i)\b(?:chance of rain|chance of precipitation|rain chance|precip(?:itation)? chance)\b[^0-9]{0,16}(\d{1,3})\s*%?""")
            .find(text)
            ?.groupValues
            ?.getOrNull(1)
            ?.let { return "${it}%" }
        return null
    }

    fun extractUvIndexValue(text: String): String? =
        Regex("""(?i)\buv(?:\s+index)?\b[^0-9]{0,10}(\d{1,2}(?:\.\d+)?)""")
            .find(text)
            ?.groupValues
            ?.getOrNull(1)

    fun inferWeatherConditionFromIconUrl(rawUrl: String): String? {
        val normalized = rawUrl.lowercase(Locale.US)
        extractWeatherApiIconCode(normalized)?.let { code ->
            val isNight = normalized.contains("/night/")
            mapWeatherApiIconCode(code, isNight = isNight)?.let { return it }
        }
        extractOpenWeatherIconCode(normalized)?.let { code ->
            mapOpenWeatherIconCode(code)?.let { return it }
        }
        return inferWeatherConditionFromTextPool(normalized)
    }

    fun extractWeatherApiIconCode(normalizedUrl: String): Int? {
        return Regex("""/(\d{3,4})\.(png|jpg|jpeg|webp|svg)(?:[?#].*)?$""")
            .find(normalizedUrl)
            ?.groupValues
            ?.getOrNull(1)
            ?.toIntOrNull()
    }

    fun mapWeatherApiIconCode(code: Int, isNight: Boolean): String? {
        return when (code) {
            113 -> if (isNight) "Clear" else "Sunny"
            116 -> "Partly Cloudy"
            119, 122 -> "Cloudy"
            143, 248, 260 -> "Fog"
            176, 263, 266, 281, 284, 293, 296, 299, 302, 305, 308, 311, 314, 353, 356, 359 -> "Rain"
            179, 182, 185, 317, 320, 323, 326, 329, 332, 335, 338, 362, 365, 368, 371, 374, 377 -> "Snow"
            200, 386, 389, 392, 395 -> "Thunderstorm"
            227, 230 -> "Blizzard"
            350 -> "Ice"
            else -> null
        }
    }

    fun extractOpenWeatherIconCode(normalizedUrl: String): String? {
        return Regex("""/([0-9]{2}[dn])(?:@\dx)?\.(png|jpg|jpeg|webp|svg)(?:[?#].*)?$""")
            .find(normalizedUrl)
            ?.groupValues
            ?.getOrNull(1)
            ?.lowercase(Locale.US)
    }

    fun mapOpenWeatherIconCode(code: String): String? {
        return when (code) {
            "01d" -> "Sunny"
            "01n" -> "Clear"
            "02d", "02n" -> "Partly Cloudy"
            "03d", "03n", "04d", "04n" -> "Cloudy"
            "09d", "09n", "10d", "10n" -> "Rain"
            "11d", "11n" -> "Thunderstorm"
            "13d", "13n" -> "Snow"
            "50d", "50n" -> "Fog"
            else -> null
        }
    }

    fun isNightWeatherContext(condition: String?): Boolean {
        val normalized = normalizeWeatherText(condition.orEmpty())
        if (normalized.contains("night") || normalized.contains("tonight") || normalized.contains("overnight") || normalized.contains("evening")) {
            return true
        }
        if (normalized.contains("day") || normalized.contains("today") || normalized.contains("afternoon") || normalized.contains("morning")) {
            return false
        }
        val hour = LocalTime.now().hour
        return hour < 6 || hour >= 18
    }

    private fun readCell(row: List<String>, index: Int?): String? {
        if (index == null || index !in row.indices) {
            return null
        }
        return row[index].trim().takeIf { it.isNotEmpty() }
    }

    private fun findHeaderIndex(
        normalizedHeader: List<String>,
        keywords: List<String>,
        exclude: Set<Int> = emptySet()
    ): Int? {
        return normalizedHeader.indices.firstOrNull { index ->
            index !in exclude && keywords.any { key -> normalizedHeader[index].contains(key) }
        }
    }

    private fun normalizeWeatherHeader(value: String): String {
        return value
            .lowercase(Locale.US)
            .replace(Regex("[^a-z0-9]+"), " ")
            .trim()
    }

    private fun isPlaceholderTableCellValue(value: String): Boolean {
        val normalized = value.trim()
        if (normalized.isEmpty()) {
            return true
        }
        val compact = normalized.replace(Regex("""\s+"""), "")
        if (TABLE_PLACEHOLDER_CELL_REGEX.matches(compact)) {
            return true
        }
        return when (normalized.lowercase(Locale.US)) {
            "na", "n/a", "null", "none", "not available" -> true
            else -> false
        }
    }

    private fun isLikelyPlaceholderMediaToken(value: String): Boolean {
        val normalized = value
            .trim()
            .trim('\'', '"')
            .lowercase(Locale.US)
        if (normalized.isBlank()) {
            return true
        }
        return normalized in setOf(
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
        ) || normalized.contains("placeholder")
    }

    private fun looksLikeCompactIconUrl(value: String): Boolean {
        val normalized = value.lowercase(Locale.US)
        val iconPathLike = Regex("""(?:^|/)(?:icon|icons)(?:/|[-_.]|$)""")
            .containsMatchIn(normalized)
        return iconPathLike ||
            normalized.contains("weatherapi.com/weather/") ||
            normalized.contains("/weather/64x64/") ||
            normalized.contains("/weather/128x128/") ||
            normalized.contains("/weather/icons/") ||
            normalized.contains("openweathermap.org/img/wn/") ||
            normalized.contains("/img/wn/") ||
            Regex("""/\d{2}[dn](?:@\dx)?\.(png|webp|jpg|jpeg)(?:[?#].*)?$""").containsMatchIn(normalized)
    }
}
