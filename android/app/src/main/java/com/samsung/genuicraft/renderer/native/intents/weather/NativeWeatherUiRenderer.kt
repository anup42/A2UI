package com.samsung.genuicraft.renderer.native.intents.weather

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AcUnit
import androidx.compose.material.icons.filled.Air
import androidx.compose.material.icons.filled.Cloud
import androidx.compose.material.icons.filled.DarkMode
import androidx.compose.material.icons.filled.Grain
import androidx.compose.material.icons.filled.Thunderstorm
import androidx.compose.material.icons.filled.WbSunny
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.samsung.genuicraft.GenUiCardTone
import com.samsung.genuicraft.GenUiTokens
import com.samsung.genuicraft.genUiCardBorderColor
import com.samsung.genuicraft.genUiCardColors
import com.samsung.genuicraft.genUiCardContainerColor
import com.samsung.genuicraft.genUiTableContainerColor
import com.samsung.genuicraft.renderer.native.WeatherCurrentDetails
import com.samsung.genuicraft.renderer.native.WeatherRow
import kotlin.math.max

internal object NativeWeatherUiRenderer {
    private val TEMPERATURE_NUMBER_REGEX = Regex("""-?\d{1,3}(?:\.\d+)?""")

    private data class WeatherChartPoint(
        val label: String,
        val value: Float,
        val displayValue: String
    )

    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    fun RenderWeatherRows(
        rows: List<WeatherRow>,
        sanitizeDisplayText: (String) -> String,
        weatherTemperatureText: (WeatherRow) -> String,
        orderWeatherRows: (List<WeatherRow>) -> List<WeatherRow>,
        isTodayWeatherRow: (WeatherRow) -> Boolean,
        weatherConditionIcon: @Composable (String?, Dp) -> Unit
    ) {
        if (rows.isEmpty()) {
            return
        }
        val orderedRows = remember(rows) { orderWeatherRows(rows) }
        val todayRow = orderedRows.first()
        val laterRows = orderedRows.drop(1)
        val todayTemperature = weatherTemperatureText(todayRow)
        val todayCondition = sanitizeDisplayText(todayRow.condition.orEmpty())
        val todayDate = sanitizeDisplayText(todayRow.date.orEmpty()).ifBlank { null }
        val todayLabel = if (isTodayWeatherRow(todayRow)) "Today" else sanitizeDisplayText(todayRow.period)
        val chartPoints = remember(orderedRows) {
            buildWeatherChartPoints(
                rows = orderedRows,
                sanitizeDisplayText = sanitizeDisplayText,
                weatherTemperatureText = weatherTemperatureText
            )
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(GenUiTokens.RadiusXl),
            colors = CardDefaults.cardColors(containerColor = genUiCardContainerColor(GenUiCardTone.Neutral)),
            elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 12.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .semantics(mergeDescendants = true) {
                            contentDescription = weatherRowAccessibilityLabel(
                                period = todayLabel,
                                date = todayDate,
                                temperature = todayTemperature,
                                condition = todayCondition,
                                metrics = todayRow.metrics,
                                sanitizeDisplayText = sanitizeDisplayText
                            )
                        },
                    shape = RoundedCornerShape(GenUiTokens.RadiusXl),
                    colors = CardDefaults.cardColors(containerColor = Color.Transparent),
                    elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                ) {
                    val heroShape = RoundedCornerShape(GenUiTokens.RadiusXl)
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(heroShape)
                            .background(
                                Brush.linearGradient(
                                    listOf(
                                        MaterialTheme.colorScheme.primary.copy(alpha = 0.96f),
                                        MaterialTheme.colorScheme.tertiary.copy(alpha = 0.86f),
                                        Color(0xFF1E293B)
                                    )
                                )
                            )
                            .padding(horizontal = 18.dp, vertical = 18.dp)
                    ) {
                        Column(
                            modifier = Modifier.fillMaxWidth(),
                            verticalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(14.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Box(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                                        .background(Color.White.copy(alpha = 0.16f))
                                        .padding(10.dp)
                                ) {
                                    weatherConditionIcon(todayRow.condition, 56.dp)
                                }
                                Column(
                                    modifier = Modifier.weight(1f),
                                    verticalArrangement = Arrangement.spacedBy(3.dp)
                                ) {
                                    MarkdownText(
                                        text = todayLabel,
                                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                                        color = Color.White
                                    )
                                    todayDate?.let { dateValue ->
                                        MarkdownText(
                                            text = dateValue,
                                            style = MaterialTheme.typography.bodySmall,
                                            color = Color.White.copy(alpha = 0.78f)
                                        )
                                    }
                                    if (todayCondition.isNotBlank()) {
                                        MarkdownText(
                                            text = todayCondition,
                                            style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold),
                                            color = Color.White.copy(alpha = 0.92f),
                                            maxLines = 2,
                                            overflow = TextOverflow.Ellipsis
                                        )
                                    }
                                }

                                if (todayTemperature.isNotBlank()) {
                                    MarkdownText(
                                        text = sanitizeDisplayText(todayTemperature),
                                        style = MaterialTheme.typography.headlineLarge.copy(fontWeight = FontWeight.Bold),
                                        color = Color.White,
                                        textAlign = TextAlign.End
                                    )
                                }
                            }

                            if (todayRow.metrics.isNotEmpty()) {
                                FlowRow(
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    verticalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    todayRow.metrics.take(4).forEach { (label, value) ->
                                        val chipLabel = sanitizeDisplayText(label)
                                        val chipValue = sanitizeDisplayText(value)
                                        if (chipLabel.isNotBlank() && chipValue.isNotBlank()) {
                                            MarkdownText(
                                                text = "$chipLabel $chipValue",
                                                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                                                color = Color.White.copy(alpha = 0.92f),
                                                modifier = Modifier
                                                    .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                                                    .background(Color.White.copy(alpha = 0.14f))
                                                    .border(
                                                        GenUiTokens.BorderMd,
                                                        Color.White.copy(alpha = 0.20f),
                                                        RoundedCornerShape(GenUiTokens.RadiusPill)
                                                    )
                                                    .padding(horizontal = 10.dp, vertical = 6.dp)
                                            )
                                        }
                                    }
                                }
                            }

                            if (chartPoints.size >= 2) {
                                WeatherTrendChart(
                                    points = chartPoints,
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(112.dp)
                                )
                            } else if (orderedRows.size >= 2) {
                                WeatherConditionTimeline(
                                    rows = orderedRows,
                                    sanitizeDisplayText = sanitizeDisplayText,
                                    weatherConditionIcon = weatherConditionIcon,
                                    modifier = Modifier.fillMaxWidth()
                                )
                            }
                        }
                    }
                }

                if (laterRows.isNotEmpty()) {
                    MarkdownText(
                        text = "Forecast by day",
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }

                laterRows.forEach { row ->
                    val temperature = weatherTemperatureText(row)
                    val periodText = sanitizeDisplayText(row.period)
                    val dateText = sanitizeDisplayText(row.date.orEmpty()).ifBlank { null }
                    val conditionText = sanitizeDisplayText(row.condition.orEmpty()).ifBlank { null }
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .semantics(mergeDescendants = true) {
                                contentDescription = weatherRowAccessibilityLabel(
                                    period = periodText,
                                    date = dateText,
                                    temperature = temperature,
                                    condition = conditionText,
                                    metrics = row.metrics,
                                    sanitizeDisplayText = sanitizeDisplayText
                                )
                            },
                        shape = RoundedCornerShape(GenUiTokens.RadiusLg),
                        colors = genUiCardColors(GenUiCardTone.Neutral),
                        border = BorderStroke(GenUiTokens.BorderSm, genUiCardBorderColor()),
                        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 14.dp, vertical = 12.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(12.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Box(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                                        .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.10f))
                                        .padding(8.dp)
                                ) {
                                    weatherConditionIcon(row.condition, 28.dp)
                                }
                                Column(
                                    modifier = Modifier.weight(1f),
                                    verticalArrangement = Arrangement.spacedBy(2.dp)
                                ) {
                                    MarkdownText(
                                        text = periodText,
                                        style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                    dateText?.let { dateValue ->
                                        MarkdownText(
                                            text = dateValue,
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis
                                        )
                                    }
                                }
                                if (temperature.isNotBlank()) {
                                    MarkdownText(
                                        text = sanitizeDisplayText(temperature),
                                        style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
                                        color = MaterialTheme.colorScheme.onSurface,
                                        textAlign = TextAlign.End
                                    )
                                }
                            }

                            conditionText?.let { condition ->
                                MarkdownText(
                                    text = condition,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }

                            if (row.metrics.isNotEmpty()) {
                                FlowRow(
                                    horizontalArrangement = Arrangement.spacedBy(7.dp),
                                    verticalArrangement = Arrangement.spacedBy(7.dp)
                                ) {
                                    row.metrics.forEach { (label, value) ->
                                        val chipLabel = sanitizeDisplayText(label)
                                        val chipValue = sanitizeDisplayText(value)
                                        if (chipLabel.isNotBlank() && chipValue.isNotBlank()) {
                                            MarkdownText(
                                                text = "$chipLabel $chipValue",
                                                style = MaterialTheme.typography.labelSmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                modifier = Modifier
                                                    .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                                                    .background(genUiCardContainerColor(GenUiCardTone.Neutral))
                                                    .border(
                                                        GenUiTokens.BorderMd,
                                                        genUiCardBorderColor(),
                                                        RoundedCornerShape(GenUiTokens.RadiusPill)
                                                    )
                                                    .padding(horizontal = 9.dp, vertical = 5.dp)
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Composable
    private fun WeatherTrendChart(
        points: List<WeatherChartPoint>,
        modifier: Modifier = Modifier
    ) {
        val safePoints = remember(points) { points.take(7) }
        val minValue = remember(safePoints) { safePoints.minOfOrNull { it.value } ?: 0f }
        val maxValue = remember(safePoints) { safePoints.maxOfOrNull { it.value } ?: minValue }
        Column(
            modifier = modifier
                .clip(RoundedCornerShape(GenUiTokens.RadiusLg))
                .background(Color.White.copy(alpha = 0.13f))
                .border(
                    GenUiTokens.BorderMd,
                    Color.White.copy(alpha = 0.16f),
                    RoundedCornerShape(GenUiTokens.RadiusLg)
                )
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                MarkdownText(
                    text = "Temperature trend",
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                    color = Color.White.copy(alpha = 0.92f)
                )
                safePoints.firstOrNull()?.displayValue?.takeIf { it.isNotBlank() }?.let { value ->
                    MarkdownText(
                        text = value,
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                        color = Color.White
                    )
                }
            }

            Canvas(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp)
                    .semantics {
                        contentDescription = weatherTrendAccessibilityLabel(safePoints)
                    }
            ) {
                if (safePoints.size < 2) return@Canvas
                val top = 8f
                val bottom = size.height - 8f
                val range = max(1f, maxValue - minValue)
                fun x(index: Int): Float = if (safePoints.size == 1) {
                    size.width / 2f
                } else {
                    (size.width / (safePoints.size - 1)) * index
                }
                fun y(value: Float): Float = bottom - ((value - minValue) / range) * (bottom - top)

                val gridColor = Color.White.copy(alpha = 0.18f)
                drawLine(gridColor, Offset(0f, bottom), Offset(size.width, bottom), strokeWidth = 1.4f)
                drawLine(gridColor, Offset(0f, top), Offset(size.width, top), strokeWidth = 1.0f)

                val path = Path().apply {
                    safePoints.forEachIndexed { index, point ->
                        val px = x(index)
                        val py = y(point.value)
                        if (index == 0) moveTo(px, py) else lineTo(px, py)
                    }
                }
                drawPath(
                    path = path,
                    color = Color.White,
                    style = Stroke(width = 5f, cap = StrokeCap.Round)
                )
                safePoints.forEachIndexed { index, point ->
                    drawCircle(
                        color = Color.White,
                        radius = 6f,
                        center = Offset(x(index), y(point.value))
                    )
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                safePoints.forEach { point ->
                    MarkdownText(
                        text = point.label,
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.White.copy(alpha = 0.72f),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
        }
    }

    @Composable
    private fun WeatherConditionTimeline(
        rows: List<WeatherRow>,
        sanitizeDisplayText: (String) -> String,
        weatherConditionIcon: @Composable (String?, Dp) -> Unit,
        modifier: Modifier = Modifier
    ) {
        val visibleRows = remember(rows) { rows.take(5) }
        Column(
            modifier = modifier
                .clip(RoundedCornerShape(GenUiTokens.RadiusLg))
                .background(Color.White.copy(alpha = 0.13f))
                .border(
                    GenUiTokens.BorderMd,
                    Color.White.copy(alpha = 0.16f),
                    RoundedCornerShape(GenUiTokens.RadiusLg)
                )
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            MarkdownText(
                text = "Outlook trend",
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                color = Color.White.copy(alpha = 0.92f)
            )
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(72.dp)
                    .semantics {
                        contentDescription = weatherTimelineAccessibilityLabel(visibleRows, sanitizeDisplayText)
                    }
            ) {
                Canvas(modifier = Modifier.matchParentSize()) {
                    val y = size.height * 0.28f
                    drawLine(
                        color = Color.White.copy(alpha = 0.25f),
                        start = Offset(12f, y),
                        end = Offset(size.width - 12f, y),
                        strokeWidth = 3f,
                        cap = StrokeCap.Round
                    )
                }
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.Top
                ) {
                    visibleRows.forEach { row ->
                        val periodText = sanitizeDisplayText(row.period)
                        val conditionText = sanitizeDisplayText(row.condition.orEmpty())
                        Column(
                            modifier = Modifier.weight(1f),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(3.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                                    .background(Color.White.copy(alpha = 0.18f))
                                    .padding(5.dp)
                            ) {
                                weatherConditionIcon(row.condition, 22.dp)
                            }
                            MarkdownText(
                                text = compactWeatherChartLabel(periodText),
                                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                                color = Color.White,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                            if (conditionText.isNotBlank()) {
                                MarkdownText(
                                    text = conditionText,
                                    style = MaterialTheme.typography.labelSmall,
                                    color = Color.White.copy(alpha = 0.72f),
                                    textAlign = TextAlign.Center,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    private fun buildWeatherChartPoints(
        rows: List<WeatherRow>,
        sanitizeDisplayText: (String) -> String,
        weatherTemperatureText: (WeatherRow) -> String
    ): List<WeatherChartPoint> {
        return rows.mapNotNull { row ->
            val display = sanitizeDisplayText(weatherTemperatureText(row)).ifBlank {
                sanitizeDisplayText(row.high.orEmpty()).ifBlank { sanitizeDisplayText(row.temp.orEmpty()) }
            }
            val value = extractWeatherChartValue(row, display) ?: return@mapNotNull null
            WeatherChartPoint(
                label = compactWeatherChartLabel(sanitizeDisplayText(row.period)),
                value = value,
                displayValue = display
            )
        }
    }

    private fun extractWeatherChartValue(
        row: WeatherRow,
        display: String
    ): Float? {
        val preferredSources = listOf(row.high, row.temp, display, row.low)
        preferredSources.forEach { source ->
            val values = extractTemperatureNumbers(source.orEmpty())
            if (values.isNotEmpty()) {
                return values.maxOrNull()
            }
        }
        return null
    }

    private fun extractTemperatureNumbers(value: String): List<Float> {
        if (value.isBlank()) return emptyList()
        return TEMPERATURE_NUMBER_REGEX.findAll(value)
            .mapNotNull { match -> match.value.toFloatOrNull() }
            .toList()
    }

    private fun compactWeatherChartLabel(period: String): String {
        val clean = period.trim()
        if (clean.isBlank()) return ""
        val normalized = clean.lowercase()
        if (normalized == "today" || normalized == "tonight") return clean
        Regex("""(?i)\b(day\s*\d+|mon|tue|wed|thu|fri|sat|sun)\b""")
            .find(clean)
            ?.value
            ?.let { return it.replaceFirstChar { char -> char.uppercaseChar() } }
        return clean.split(Regex("""\s+""")).take(2).joinToString(" ").take(10)
    }

    private fun weatherTrendAccessibilityLabel(points: List<WeatherChartPoint>): String {
        if (points.isEmpty()) return "Weather temperature trend"
        return points.joinToString(
            prefix = "Weather temperature trend. ",
            separator = ". "
        ) { point ->
            listOf(point.label, point.displayValue).filter { it.isNotBlank() }.joinToString(": ")
        }
    }

    private fun weatherTimelineAccessibilityLabel(
        rows: List<WeatherRow>,
        sanitizeDisplayText: (String) -> String
    ): String {
        if (rows.isEmpty()) return "Weather outlook trend"
        return rows.joinToString(
            prefix = "Weather outlook trend. ",
            separator = ". "
        ) { row ->
            listOf(
                compactWeatherChartLabel(sanitizeDisplayText(row.period)),
                sanitizeDisplayText(row.condition.orEmpty())
            ).filter { it.isNotBlank() }.joinToString(": ")
        }
    }

    private fun weatherRowAccessibilityLabel(
        period: String,
        date: String?,
        temperature: String,
        condition: String?,
        metrics: List<Pair<String, String>>,
        sanitizeDisplayText: (String) -> String
    ): String {
        return buildString {
            sanitizeDisplayText(period).takeIf { it.isNotBlank() }?.let { append(it) }
            date?.let(sanitizeDisplayText)?.takeIf { it.isNotBlank() }?.let { value ->
                if (isNotEmpty()) append(". ")
                append(value)
            }
            sanitizeDisplayText(temperature).takeIf { it.isNotBlank() }?.let { value ->
                if (isNotEmpty()) append(". ")
                append(value)
            }
            condition?.let(sanitizeDisplayText)?.takeIf { it.isNotBlank() }?.let { value ->
                if (isNotEmpty()) append(". ")
                append(value)
            }
            metrics.forEach { (label, value) ->
                val cleanLabel = sanitizeDisplayText(label)
                val cleanValue = sanitizeDisplayText(value)
                if (cleanLabel.isNotBlank() && cleanValue.isNotBlank()) {
                    if (isNotEmpty()) append(". ")
                    append(cleanLabel)
                    append(": ")
                    append(cleanValue)
                }
            }
        }.ifBlank { "Weather forecast item" }
    }

    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    fun RenderCurrentWeatherDetails(
        titleText: String,
        titleStyle: TextStyle,
        details: WeatherCurrentDetails,
        sanitizeDisplayText: (String) -> String,
        inferWeatherConditionFromIconUrl: (String) -> String?,
        markdownText: @Composable (String, TextStyle, Color) -> Unit,
        weatherConditionIcon: @Composable (String?, Dp) -> Unit
    ) {
        val iconUrl = details.iconUrl?.trim().orEmpty()
        val iconBasedCondition = iconUrl
            .takeIf { it.isNotBlank() }
            ?.let(inferWeatherConditionFromIconUrl)
        val heroCondition = iconBasedCondition ?: details.condition
        val conditionText = (details.condition ?: heroCondition)?.let(sanitizeDisplayText).orEmpty()
        val temperatureText = details.temperature?.let(sanitizeDisplayText).orEmpty()
        val feelsLikeText = details.feelsLike?.let(sanitizeDisplayText).orEmpty()

        markdownText(
            titleText,
            titleStyle,
            MaterialTheme.colorScheme.onSurface
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            weatherConditionIcon(heroCondition, 124.dp)

            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                if (temperatureText.isNotBlank()) {
                    MarkdownText(
                        text = temperatureText,
                        style = MaterialTheme.typography.headlineLarge.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
                if (conditionText.isNotBlank()) {
                    MarkdownText(
                        text = conditionText,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
                if (feelsLikeText.isNotBlank()) {
                    MarkdownText(
                        text = "Feels like $feelsLikeText",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }

        val chips = buildList {
            details.humidity?.let { add("Humidity" to it) }
            details.wind?.let { add("Wind" to it) }
            details.rainChance?.let { add("Rain Chance" to it) }
            details.uvIndex?.let { add("UV Index" to it) }
        }
        if (chips.isNotEmpty()) {
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                chips.forEach { (label, value) ->
                    val chipValue = sanitizeDisplayText(value)
                    if (chipValue.isBlank()) return@forEach
                    MarkdownText(
                        text = "$label $chipValue",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier
                            .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                            .background(genUiCardContainerColor(GenUiCardTone.Neutral))
                            .border(
                                GenUiTokens.BorderMd,
                                genUiCardBorderColor(),
                                RoundedCornerShape(GenUiTokens.RadiusPill)
                            )
                            .padding(horizontal = 8.dp, vertical = 5.dp)
                    )
                }
            }
        }

        details.summary
            ?.let(sanitizeDisplayText)
            ?.takeIf { it.isNotBlank() }
            ?.let { summary ->
                markdownText(
                    summary,
                    MaterialTheme.typography.bodySmall,
                    MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
    }

    @Composable
    fun WeatherConditionIcon(
        condition: String?,
        size: Dp,
        sanitizeDisplayText: (String) -> String
    ) {
        val (icon, tint) = weatherConditionIconSpec(condition)
        Icon(
            imageVector = icon,
            contentDescription = sanitizeDisplayText(condition.orEmpty()).ifBlank { "Weather condition" },
            tint = tint,
            modifier = Modifier.size(size)
        )
    }

    @Composable
    private fun MarkdownText(
        text: String,
        style: TextStyle,
        color: Color,
        modifier: Modifier = Modifier,
        textAlign: TextAlign? = null,
        maxLines: Int = Int.MAX_VALUE,
        overflow: TextOverflow = TextOverflow.Clip
    ) {
        val displayText = remember(text) { sanitizeDisplayText(text, preserveMarkdown = true) }
        if (displayText.isBlank()) {
            return
        }
        val hasMarkdownInline = remember(displayText) { containsMarkdownInlineFormatting(displayText) }
        val hasMarkdownHeading = remember(displayText) { containsMarkdownHeading(displayText) }
        val hasMarkdown = hasMarkdownInline || hasMarkdownHeading
        val plainText = remember(displayText) { sanitizeDisplayText(displayText) }
        if (!hasMarkdown) {
            Text(
                text = plainText,
                style = style,
                color = color,
                modifier = modifier,
                textAlign = textAlign,
                maxLines = maxLines,
                overflow = overflow
            )
            return
        }
        val parsedText = remember(displayText, style, hasMarkdownHeading) {
            if (hasMarkdownHeading) {
                parseMarkdownWithHeadings(displayText, style)
            } else {
                parseInlineMarkdown(displayText)
            }
        }
        Text(
            text = parsedText,
            style = style.copy(fontWeight = FontWeight.Normal),
            color = color,
            modifier = modifier,
            textAlign = textAlign,
            maxLines = maxLines,
            overflow = overflow
        )
    }

    @Composable
    private fun weatherConditionIconSpec(condition: String?): Pair<ImageVector, Color> {
        val normalized = NativeWeatherSemantics.normalizeWeatherText(condition.orEmpty())
        return when {
            normalized.contains("thunder") || normalized.contains("storm") || normalized.contains("lightning") ->
                Icons.Filled.Thunderstorm to Color(0xFFE65B17)
            normalized.contains("snow") || normalized.contains("sleet") || normalized.contains("blizzard") ->
                Icons.Filled.AcUnit to Color(0xFF82B1FF)
            normalized.contains("rain") || normalized.contains("shower") || normalized.contains("drizzle") ->
                Icons.Filled.Grain to Color(0xFF2E65D4)
            normalized.contains("fog") || normalized.contains("mist") || normalized.contains("haze") ||
                normalized.contains("wind") || normalized.contains("breeze") ->
                Icons.Filled.Air to Color(0xFF25B871)
            normalized.contains("cloud") || normalized.contains("overcast") ->
                Icons.Filled.Cloud to Color(0xFF7A7A85)
            normalized.contains("clear") ->
                if (NativeWeatherSemantics.isNightWeatherContext(condition)) {
                    Icons.Filled.DarkMode to Color(0xFF7DA2FF)
                } else {
                    Icons.Filled.WbSunny to Color(0xFFFFB300)
                }

            normalized.contains("sun") ->
                Icons.Filled.WbSunny to Color(0xFFFFB300)

            else -> Icons.Filled.Cloud to MaterialTheme.colorScheme.primary
        }
    }

    private fun sanitizeDisplayText(text: String, preserveMarkdown: Boolean = false): String =
        com.samsung.genuicraft.renderer.native.NativeTextFormatter.sanitizeDisplayText(text, preserveMarkdown)

    private fun containsMarkdownInlineFormatting(text: String): Boolean =
        com.samsung.genuicraft.renderer.native.NativeTextFormatter.containsMarkdownInlineFormatting(text)

    private fun containsMarkdownHeading(text: String): Boolean =
        com.samsung.genuicraft.renderer.native.NativeTextFormatter.containsMarkdownHeading(text)

    private fun parseMarkdownWithHeadings(text: String, baseStyle: TextStyle) =
        com.samsung.genuicraft.renderer.native.NativeTextFormatter.parseMarkdownWithHeadings(text, baseStyle)

    private fun parseInlineMarkdown(text: String) =
        com.samsung.genuicraft.renderer.native.NativeTextFormatter.parseInlineMarkdown(text)
}
