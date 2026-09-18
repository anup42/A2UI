package com.samsung.genuicraft.sdk.internal.renderer.native.intents.weather

import com.samsung.genuicraft.sdk.internal.renderer.native.TextBlock
import com.samsung.genuicraft.sdk.internal.renderer.native.WeatherCurrentDetails
import com.samsung.genuicraft.sdk.internal.renderer.native.WeatherRow
import com.samsung.genuicraft.sdk.internal.renderer.native.WeatherTableColumns
import com.samsung.genuicraft.sdk.internal.renderer.native.NativeTextFormatter
import java.time.LocalDate
import java.time.LocalTime
import java.time.format.DateTimeFormatter
import java.time.format.DateTimeFormatterBuilder
import java.time.format.ResolverStyle
import java.util.Locale

internal object NativeWeatherSemantics {
    private val TABLE_PLACEHOLDER_CELL_REGEX = Regex("""^[:\-\u2013\u2014]+$""")
    private val TEMPERATURE_WITH_UNIT_REGEX =
        Regex("""(?i)(-?\d{1,3}(?:\.\d+)?)\s*(?:\u00B0\s*)?([CF])\b""")
    private val TEMPERATURE_HEADER_UNIT_REGEX =
        Regex("""(?i)(?:\u00B0\s*|\(\s*|\b)([CF])(?:\s*\)|\b)""")
    private val TEMPERATURE_NUMBER_TOKEN_REGEX =
        Regex("""(?i)(?<![A-Z0-9])(-?\d{1,3}(?:\.\d+)?)\s*(?:\u00B0\s*)?([CF])?(?![A-Z0-9])""")
    private val EXPLICIT_YEAR_REGEX = Regex("""\b\d{4}\b""")
    private val EXPLICIT_DATE_FORMATTERS =
        listOf(
            DateTimeFormatter.ISO_LOCAL_DATE.withResolverStyle(ResolverStyle.STRICT),
            strictDateFormatter("EEE, MMM d, uuuu"),
            strictDateFormatter("EEEE, MMMM d, uuuu"),
            strictDateFormatter("EEE MMM d uuuu"),
            strictDateFormatter("EEEE MMMM d uuuu"),
            strictDateFormatter("MMM d, uuuu"),
            strictDateFormatter("MMMM d, uuuu"),
            strictDateFormatter("MMM d uuuu"),
            strictDateFormatter("MMMM d uuuu"),
            strictDateFormatter("d MMM uuuu"),
            strictDateFormatter("d MMMM uuuu"),
            strictDateFormatter("M/d/uuuu"),
        )

    private data class TemperatureReading(
        val value: String,
        val unit: String?
    )

    fun buildWeatherRows(
        header: List<String>,
        body: List<List<String>>
    ): List<WeatherRow>? {
        if (header.isEmpty() || body.isEmpty()) {
            return null
        }
        val columns = detectWeatherColumns(header) ?: return null
        val tableTemperatureUnit =
            listOf(columns.temp, columns.high, columns.low)
                .mapNotNull { index -> temperatureUnitFromHeader(index?.let(header::getOrNull)) }
                .distinct()
                .singleOrNull()
        val rows = body.mapNotNull { row ->
            val period = normalizeWeatherCell(readCell(row, columns.period)).orEmpty()
            if (period.isBlank()) {
                return@mapNotNull null
            }

            val metrics = buildWeatherRowMetrics(header, row, columns)

            WeatherRow(
                period = period,
                date = normalizeWeatherCell(readCell(row, columns.date)),
                condition = normalizeWeatherCell(readCell(row, columns.condition)),
                high = temperatureCellWithHeaderUnit(header, row, columns.high, tableTemperatureUnit),
                low = temperatureCellWithHeaderUnit(header, row, columns.low, tableTemperatureUnit),
                temp = temperatureCellWithHeaderUnit(header, row, columns.temp, tableTemperatureUnit),
                metrics = metrics
            )
        }
        if (rows.isEmpty()) {
            return null
        }
        val weatherLikeRows = rows.count(::isWeatherLikeRow)
        if (weatherLikeRows == 0) {
            return null
        }
        return rows
    }

    private fun temperatureCellWithHeaderUnit(
        header: List<String>,
        row: List<String>,
        column: Int?,
        tableTemperatureUnit: String?,
    ): String? {
        val value = normalizeWeatherCell(readCell(row, column)) ?: return null
        val unit =
            temperatureUnitFromHeader(column?.let(header::getOrNull))
                ?: tableTemperatureUnit
                ?: return value
        return TEMPERATURE_NUMBER_TOKEN_REGEX.replace(value) { match ->
            val suppliedUnit = match.groupValues[2].uppercase(Locale.US)
            "${match.groupValues[1]}\u00B0${suppliedUnit.ifBlank { unit }}"
        }
    }

    private fun temperatureUnitFromHeader(header: String?): String? {
        val value = header?.trim().orEmpty()
        val namedUnit =
            when {
                Regex("""(?i)\bcelsius\b""").containsMatchIn(value) -> "C"
                Regex("""(?i)\bfahrenheit\b""").containsMatchIn(value) -> "F"
                else -> null
            }
        return namedUnit
            ?: TEMPERATURE_HEADER_UNIT_REGEX.find(value)?.groupValues?.getOrNull(1)?.uppercase(Locale.US)
    }

    fun buildCurrentWeatherRowsFromKeyValueTable(
        header: List<String>,
        body: List<List<String>>
    ): List<WeatherRow>? {
        if (body.size < 3) {
            return null
        }
        val pairs = body.mapNotNull { row ->
            val label = row.getOrNull(0)?.trim().orEmpty()
            val value = row.getOrNull(1)?.trim().orEmpty()
            if (label.isBlank() || value.isBlank() || isPlaceholderTableCellValue(value)) {
                null
            } else {
                label to value
            }
        }
        if (pairs.size < 3) {
            return null
        }

        val normalizedLabels = pairs.map { (label, _) -> normalizeWeatherText(label) }
        val weatherLabelSignals = normalizedLabels.count(::isCurrentWeatherMetricLabel)
        if (weatherLabelSignals < 3) {
            return null
        }

        fun firstValueFor(vararg keys: String): String? {
            return pairs.firstOrNull { (label, _) ->
                val normalized = normalizeWeatherText(label)
                keys.any { key -> normalized == key || normalized.contains(key) }
            }?.second?.trim()?.takeIf { it.isNotBlank() }
        }

        val condition = firstValueFor("current condition", "condition", "weather", "forecast", "summary")
        val temperature = firstValueFor("current temperature", "temperature", "temp")
        val high = firstValueFor("high", "max")
        val low = firstValueFor("low", "min")
        val hasStrongWeatherValue =
            !temperature.isNullOrBlank() ||
                !high.isNullOrBlank() ||
                !low.isNullOrBlank() ||
                isWeatherConditionLabel(condition)
        if (!hasStrongWeatherValue) {
            return null
        }

        val consumedKeys = setOf(
            "current condition",
            "condition",
            "weather",
            "forecast",
            "summary",
            "current temperature",
            "temperature",
            "temp",
            "high",
            "max",
            "low",
            "min"
        )
        val metrics = pairs.mapNotNull { (label, value) ->
            val normalized = normalizeWeatherText(label)
            val consumed = consumedKeys.any { key -> normalized == key || normalized.contains(key) }
            if (consumed || value.isBlank()) {
                null
            } else {
                normalizeWeatherMetricLabel(label) to value
            }
        }

        return listOf(
            WeatherRow(
                period = firstValueFor("day", "date", "period").takeIf { !it.isNullOrBlank() } ?: "Forecast",
                date = null,
                condition = condition,
                high = high,
                low = low,
                temp = temperature,
                metrics = metrics
            )
        )
    }

    fun detectWeatherColumns(header: List<String>): WeatherTableColumns? {
        val normalized = header.map { normalizeWeatherHeader(it) }
        val strictWeatherSignal = normalized.count { token ->
            token.contains("weather") ||
                token.contains("forecast") ||
                token.contains("temp") ||
                token.contains("high") ||
                token.contains("low") ||
                token.contains("precip") ||
                token.contains("rain") ||
                token.contains("humidity") ||
                token.contains("wind") ||
                token.contains("uv")
        }
        if (strictWeatherSignal == 0) {
            return null
        }
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
        val period = findHeaderIndex(normalized, listOf("day", "time", "hour", "period", "date")) ?: return null
        val date = findHeaderIndex(normalized, listOf("date"), exclude = setOf(period))
        val condition = findHeaderIndex(normalized, listOf("condition", "forecast", "weather", "summary"), exclude = setOf(period))
        val hasForecastOnlyShape = condition != null && strictWeatherSignal >= 1
        if (weatherSignal < 2 && !hasForecastOnlyShape) {
            return null
        }
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
        if (contentSignals < 2 && !hasForecastOnlyShape) {
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

    private fun buildWeatherRowMetrics(
        header: List<String>,
        row: List<String>,
        columns: WeatherTableColumns
    ): List<Pair<String, String>> {
        val consumed = setOfNotNull(
            columns.period,
            columns.date,
            columns.condition,
            columns.high,
            columns.low,
            columns.temp
        )
        val explicitMetricIndexes = listOfNotNull(
            columns.precip,
            columns.wind,
            columns.humidity,
            columns.uv
        )
        val metrics = linkedMapOf<Int, Pair<String, String>>()

        explicitMetricIndexes.forEach { index ->
            normalizeWeatherCell(readCell(row, index))?.let { value ->
                metrics[index] = displayWeatherMetricLabel(header.getOrNull(index), index) to value
            }
        }

        header.indices
            .filterNot { index -> index in consumed || index in metrics.keys }
            .forEach { index ->
                val value = normalizeWeatherCell(readCell(row, index)) ?: return@forEach
                val label = displayWeatherMetricLabel(header.getOrNull(index), index)
                if (label.isNotBlank() && isUsefulWeatherExtraMetric(label, value)) {
                    metrics[index] = label to value
                }
            }

        return metrics.values.toList()
    }

    private fun displayWeatherMetricLabel(rawLabel: String?, index: Int): String {
        val clean = rawLabel
            ?.replace(Regex("""[_\-]+"""), " ")
            ?.replace(Regex("""\s+"""), " ")
            ?.trim()
            .orEmpty()
        if (clean.isBlank()) {
            return "Detail ${index + 1}"
        }
        val normalized = normalizeWeatherHeader(clean)
        return when {
            normalized.contains("precip") -> "Precip"
            normalized.contains("rain") && normalized.contains("chance") -> "Rain Chance"
            normalized == "rain" -> "Rain"
            normalized.contains("humidity") -> "Humidity"
            normalized.contains("wind") -> "Wind"
            normalized.contains("uv") -> "UV"
            normalized.contains("cloth") || normalized.contains("wear") || normalized.contains("outfit") -> "What to wear"
            normalized.contains("advice") || normalized.contains("recommend") -> "Advice"
            normalized.contains("best") && normalized.contains("window") -> "Best window"
            else -> clean.split(' ').joinToString(" ") { token ->
                token.replaceFirstChar { char ->
                    if (char.isLowerCase()) char.titlecase(Locale.US) else char.toString()
                }
            }
        }
    }

    private fun isUsefulWeatherExtraMetric(label: String, value: String): Boolean {
        val normalizedLabel = normalizeWeatherHeader(label)
        val normalizedValue = value.trim()
        if (normalizedValue.isBlank() || isPlaceholderTableCellValue(normalizedValue)) {
            return false
        }
        if (normalizedLabel in setOf("source", "sources", "url", "link", "image", "icon")) {
            return false
        }
        return true
    }

    fun weatherTemperatureText(row: WeatherRow): String {
        val temp = collapseCompositeTemperature(formatTemperatureCellValue(row.temp))
        if (!temp.isNullOrBlank()) {
            return temp
        }
        val high = formatTemperatureCellValue(row.high)
        val low = formatTemperatureCellValue(row.low)
        if (!high.isNullOrBlank() && !low.isNullOrBlank()) {
            return formatHighLowTemperature(high, low)
        }
        return collapseCompositeTemperature(high ?: low).orEmpty()
    }


    fun formatTemperatureCellValue(value: String?): String? {
        val raw = normalizeWeatherCell(value) ?: return null
        var formatted = normalizeTemperatureText(raw)
            .replace(
                Regex("(?i)(-?\\d{1,2}(?:\\.\\d+)?)\\s*(?:\\u00B0\\s*)?([CF])\\b")
            ) { match ->
                val number = match.groupValues[1]
                val unit = match.groupValues[2].uppercase(Locale.US)
                "$number\u00B0$unit"
            }
        if (formatted.contains("\u00B0")) {
            return formatted
        }

        val rangePattern = Regex(
            "^\\s*-?\\d{1,2}(?:\\.\\d+)?\\s*(?:/|\\u2013|\\u2014|-|to)\\s*-?\\d{1,2}(?:\\.\\d+)?\\s*$",
            RegexOption.IGNORE_CASE
        )
        if (rangePattern.matches(formatted)) {
            formatted = formatted.replace(Regex("-?\\d{1,2}(?:\\.\\d+)?")) { match ->
                "${match.value}\u00B0"
            }
            return formatted
        }

        val single = Regex("^\\s*(-?\\d{1,2}(?:\\.\\d+)?)\\s*$").matchEntire(formatted)
        if (single != null) {
            return "${single.groupValues[1]}\u00B0"
        }
        return formatted
    }

    private fun formatHighLowTemperature(
        high: String,
        low: String
    ): String {
        val highReadings = extractTemperatureReadings(high)
        val lowReadings = extractTemperatureReadings(low)
        val preferredUnit = when {
            highReadings.any { it.unit == "C" } && lowReadings.any { it.unit == "C" } -> "C"
            highReadings.any { it.unit == "F" } && lowReadings.any { it.unit == "F" } -> "F"
            else -> null
        }

        if (preferredUnit != null) {
            val highPreferred = highReadings.firstOrNull { it.unit == preferredUnit }
            val lowPreferred = lowReadings.firstOrNull { it.unit == preferredUnit }
            if (highPreferred != null && lowPreferred != null) {
                return "${formatTemperatureReading(highPreferred.value, preferredUnit)} / " +
                    formatTemperatureReading(lowPreferred.value, preferredUnit)
            }
        }

        return "$high / $low"
    }

    private fun collapseCompositeTemperature(value: String?): String? {
        val formatted = value?.trim().orEmpty()
        if (formatted.isBlank()) {
            return null
        }

        val readings = extractTemperatureReadings(formatted)
        if (readings.size < 4) {
            return formatted
        }

        val preferredUnit = when {
            readings.count { it.unit == "C" } >= 2 -> "C"
            readings.count { it.unit == "F" } >= 2 -> "F"
            else -> null
        }
        if (preferredUnit != null) {
            val unitMatches = readings.filter { it.unit == preferredUnit }
            if (unitMatches.size >= 4) {
                return "${formatTemperatureReading(unitMatches[0].value, preferredUnit)} / " +
                    formatTemperatureReading(unitMatches[2].value, preferredUnit)
            }
            if (unitMatches.size >= 2) {
                return "${formatTemperatureReading(unitMatches[0].value, preferredUnit)} / " +
                    formatTemperatureReading(unitMatches[1].value, preferredUnit)
            }
        }

        val distinctReadings = readings
            .map { reading -> formatTemperatureReading(reading.value, reading.unit) }
            .distinct()
        if (distinctReadings.size >= 2) {
            return "${distinctReadings[0]} / ${distinctReadings[1]}"
        }
        return formatted
    }

    private fun extractTemperatureReadings(text: String): List<TemperatureReading> {
        val normalizedText = normalizeTemperatureText(text)
        return TEMPERATURE_WITH_UNIT_REGEX.findAll(normalizedText).map { match ->
            TemperatureReading(
                value = match.groupValues[1].trim(),
                unit = match.groupValues[2].trim().uppercase(Locale.US)
            )
        }.toList()
    }

    fun orderWeatherRows(rows: List<WeatherRow>): List<WeatherRow> = rows

    fun isTodayWeatherRow(row: WeatherRow): Boolean = isTodayWeatherRow(row, LocalDate.now())

    internal fun isTodayWeatherRow(row: WeatherRow, today: LocalDate): Boolean {
        val period = normalizeWeatherText(row.period)
        val date = normalizeWeatherText(row.date.orEmpty())
        if ("today" in period.split(' ') || "today" in date.split(' ')) {
            return true
        }
        return sequenceOf(row.date, row.period)
            .mapNotNull(::parseExplicitWeatherDate)
            .any { it == today }
    }

    private fun parseExplicitWeatherDate(value: String?): LocalDate? {
        val text = value?.trim().orEmpty()
        if (text.isBlank() || !EXPLICIT_YEAR_REGEX.containsMatchIn(text)) {
            return null
        }
        val normalized =
            text.replace(Regex("""(?i)\b(\d{1,2})(?:st|nd|rd|th)\b"""), "$1")
        return EXPLICIT_DATE_FORMATTERS.firstNotNullOfOrNull { formatter ->
            runCatching { LocalDate.parse(normalized, formatter) }.getOrNull()
        }
    }

    private fun strictDateFormatter(pattern: String): DateTimeFormatter =
        DateTimeFormatterBuilder()
            .parseCaseInsensitive()
            .appendPattern(pattern)
            .toFormatter(Locale.US)
            .withResolverStyle(ResolverStyle.STRICT)

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
        if (normalized.isBlank()) {
            return false
        }
        val hasCurrentMarker =
            normalized.contains("current") ||
                normalized.contains("currently") ||
                normalized.contains("now")
        val hasWeatherMarker =
            normalized.contains("weather") ||
                normalized.contains("condition") ||
                normalized.contains("temperature")
        return normalized.contains("current condition") ||
            normalized.contains("current weather") ||
            normalized == "currently" ||
            normalized == "weather now" ||
            normalized == "weather currently" ||
            (hasCurrentMarker && hasWeatherMarker)
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
        val normalizedTempText = normalizeTemperatureText(text)
        return Regex("(?<!\\d)-?\\d{1,2}(?:\\.\\d+)?\\s*\\u00B0").containsMatchIn(normalizedTempText) ||
            Regex("(?i)(?<!\\d)-?\\d{1,2}(?:\\.\\d+)?\\s*(?:\\u00B0\\s*)?[CF]\\b").containsMatchIn(normalizedTempText) ||
            normalized.contains("feels like") ||
            normalized.contains("humidity") ||
            normalized.contains("wind") ||
            normalized.contains("temperature") ||
            normalized.contains("temp ") ||
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
        val normalized = textPool.lowercase(Locale.US)
        return when {
            containsAnyWholeWord(normalized, setOf("thunder", "storm", "lightning", "thunderstorm")) -> "Thunderstorm"
            containsAnyWholeWord(normalized, setOf("snow", "sleet", "blizzard", "flurries")) -> "Snow"
            containsAnyWholeWord(normalized, setOf("rain", "rainy", "shower", "showers", "drizzle")) -> "Rain"
            containsAnyWholeWord(normalized, setOf("fog", "foggy", "mist", "haze", "hazy")) -> "Fog"
            normalized.contains("partly cloudy") ||
                containsAnyWholeWord(normalized, setOf("cloud", "cloudy", "overcast")) -> "Cloudy"

            containsWholeWord(normalized, "clear") -> "Clear"
            containsAnyWholeWord(normalized, setOf("sun", "sunny", "fair")) -> "Sunny"
            containsAnyWholeWord(normalized, setOf("wind", "windy", "breeze", "breezy")) -> "Windy"
            else -> null
        }
    }


    fun buildCurrentWeatherDetails(
        title: String,
        sectionBlocks: List<TextBlock>,
        fallbackCondition: String?,
        additionalLines: List<String> = emptyList()
    ): WeatherCurrentDetails? {
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
            .replace(Regex("\\s+"), " ")
            .trim()
        if (merged.isBlank()) {
            return null
        }
        val normalizedMerged = normalizeTemperatureText(merged)

        val iconUrl = extractCurrentWeatherIconUrl(sectionBlocks)
        val iconCondition = iconUrl?.let(::inferWeatherConditionFromIconUrl)
        val condition = iconCondition
            ?: inferWeatherConditionFromTextPool(normalizedMerged.lowercase(Locale.US))
            ?: fallbackCondition
        val temperature = extractTemperatureValue(normalizedMerged)
        val feelsLike = extractFeelsLikeValue(normalizedMerged)
        val humidity = extractHumidityValue(normalizedMerged)
        val wind = extractWindValue(normalizedMerged)
        val rainChance = extractRainChanceValue(normalizedMerged)
        val uvIndex = extractUvIndexValue(normalizedMerged)
        val summary = normalizedMerged
            .split(Regex("(?<=[.!?])\\s+"))
            .filter { it.isNotBlank() }
            .take(2)
            .joinToString(" ")
            .trim()
            .takeIf { it.isNotBlank() }
        val hasMetricSignals =
            !temperature.isNullOrBlank() ||
                !feelsLike.isNullOrBlank() ||
                !humidity.isNullOrBlank() ||
                !wind.isNullOrBlank() ||
                !rainChance.isNullOrBlank() ||
                !uvIndex.isNullOrBlank()
        val titleHasWeatherContext = normalizeWeatherText(title).contains("weather")
        val titleNormalized = normalizeWeatherText(title)
        val titleHasCurrentMarker =
            titleNormalized.contains("current") ||
                titleNormalized.contains("currently") ||
                titleNormalized.contains("now")
        val conditionIsWeatherLike = isWeatherConditionLabel(condition)
        val hasStrongWeatherData =
            !iconUrl.isNullOrBlank() ||
                hasMetricSignals ||
                (titleHasWeatherContext && conditionIsWeatherLike)
        val looksLikeCurrentSection =
            isCurrentWeatherHeading(title) ||
                titleHasCurrentMarker ||
                (titleHasWeatherContext && hasMetricSignals)
        if (!looksLikeCurrentSection) {
            return null
        }
        if (!hasStrongWeatherData && summary.isNullOrBlank()) {
            return null
        }

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
        )
    }

    fun extractTemperatureValue(text: String): String? {
        val normalizedText = normalizeTemperatureText(text)
        Regex("(?<!\\d)(-?\\d{1,2}(?:\\.\\d+)?)\\s*(?:\\u00B0\\s*([CF])|([CF]))\\b", RegexOption.IGNORE_CASE)
            .find(normalizedText)
            ?.let { match ->
                val value = match.groupValues[1]
                val unit = match.groupValues.getOrNull(2).orEmpty().ifBlank {
                    match.groupValues.getOrNull(3).orEmpty()
                }
                return formatTemperatureReading(value, unit)
            }

        Regex("(?<!\\d)(-?\\d{1,2}(?:\\.\\d+)?)\\s*\\u00B0\\b")
            .find(normalizedText)
            ?.let { match ->
                return formatTemperatureReading(match.groupValues[1], null)
            }

        Regex("(?i)\\b(?:temperature|temp|currently|current)\\b[^-\\d]{0,16}(-?\\d{1,2}(?:\\.\\d+)?)\\s*(?:\\u00B0\\s*([CF])|([CF]))?")
            .find(normalizedText)
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
        val normalizedText = normalizeTemperatureText(text)
        Regex("(?i)\\bfeels?\\s+like\\b[^-\\d]{0,16}(-?\\d{1,2}(?:\\.\\d+)?)\\s*(?:\\u00B0\\s*([CF])|([CF]))?")
            .find(normalizedText)
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
        return if (cleanUnit.isBlank()) "$cleanValue\u00B0" else "$cleanValue\u00B0$cleanUnit"
    }


    fun normalizeTemperatureText(value: String): String {
        val normalized = value
            .replace("\u00C2\u00B0", "\u00B0")
            .replace("\u00C2\u00BA", "\u00B0")
            .replace("\u00C2", "")
            .replace("º", "\u00B0")
            .replace("Â°", "\u00B0")
            .replace("Âº", "\u00B0")
            .replace(Regex("(?<!\\d)-?\\d{1,2}(?:\\.\\d+)?\\s*[^\\r\\n\\w%]{1,3}\\s*([CF])\\b", RegexOption.IGNORE_CASE)) { match ->
                val number = Regex("-?\\d{1,2}(?:\\.\\d+)?").find(match.value)?.value.orEmpty()
                val unit = match.groupValues[1].uppercase(Locale.US)
                if (number.isBlank()) match.value else "$number\u00B0$unit"
            }
        return normalized.replace(Regex("\u00B0\\s*([CF])", RegexOption.IGNORE_CASE)) { match ->
            "\u00B0${match.groupValues[1].uppercase(Locale.US)}"
        }
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
            "image",
            "icon",
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

    private fun isWeatherLikeRow(row: WeatherRow): Boolean {
        val hasTempOrMetrics =
            !row.temp.isNullOrBlank() ||
                !row.high.isNullOrBlank() ||
                !row.low.isNullOrBlank() ||
                row.metrics.isNotEmpty()
        if (hasTempOrMetrics) {
            return true
        }
        return isWeatherConditionLabel(row.condition)
    }

    private fun isWeatherConditionLabel(condition: String?): Boolean {
        val normalized = normalizeWeatherText(condition.orEmpty())
        if (normalized.isBlank()) {
            return false
        }
        return containsAnyWholeWord(
            normalized,
            setOf(
                "clear",
                "sunny",
                "cloudy",
                "overcast",
                "rain",
                "rainy",
                "drizzle",
                "thunderstorm",
                "storm",
                "snow",
                "fog",
                "mist",
                "haze",
                "windy"
            )
        ) || normalized.contains("partly cloudy")
    }

    private fun isCurrentWeatherMetricLabel(label: String): Boolean {
        val normalized = normalizeWeatherText(label)
        return normalized.contains("condition") ||
            normalized.contains("weather") ||
            normalized.contains("temperature") ||
            normalized.contains("temp") ||
            normalized.contains("high") ||
            normalized.contains("low") ||
            normalized.contains("feel") ||
            normalized.contains("rain") ||
            normalized.contains("precip") ||
            normalized.contains("wind") ||
            normalized.contains("gust") ||
            normalized.contains("humidity") ||
            normalized.contains("uv") ||
            normalized.contains("best window") ||
            normalized.contains("visibility") ||
            normalized.contains("pressure")
    }

    private fun normalizeWeatherMetricLabel(label: String): String {
        val clean = label.trim().replace(Regex("\\s+"), " ")
        if (clean.isBlank()) {
            return clean
        }
        if (clean.any { it.isLowerCase() }) {
            return clean
        }
        return clean.lowercase(Locale.US)
            .split(' ')
            .joinToString(" ") { token ->
                token.replaceFirstChar { char ->
                    if (char.isLowerCase()) char.titlecase(Locale.US) else char.toString()
                }
            }
    }

    private fun containsAnyWholeWord(text: String, words: Set<String>): Boolean {
        return words.any { containsWholeWord(text, it) }
    }

    private fun containsWholeWord(text: String, word: String): Boolean {
        return Regex("""\b${Regex.escape(word)}\b""").containsMatchIn(text)
    }
}


