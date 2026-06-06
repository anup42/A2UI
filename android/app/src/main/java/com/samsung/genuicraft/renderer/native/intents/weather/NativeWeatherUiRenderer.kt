package com.samsung.genuicraft.renderer.native.intents.weather

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.isSystemInDarkTheme
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
import androidx.compose.material.icons.filled.CheckCircle
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
        val displayValue: String,
        val lowValue: Float? = null,
        val lowDisplayValue: String? = null
    )

    private data class WeatherTemperatureRange(
        val primary: String,
        val secondary: String?
    )

    private data class WeatherHeroPalette(
        val gradient: List<Color>,
        val content: Color,
        val mutedContent: Color,
        val tileContainer: Color,
        val tileBorder: Color
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
        val darkMode = isSystemInDarkTheme()
        val chartPoints = remember(orderedRows) {
            buildWeatherChartPoints(
                rows = orderedRows,
                sanitizeDisplayText = sanitizeDisplayText,
                weatherTemperatureText = weatherTemperatureText
            )
        }

        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            WeatherHeroSummaryCard(
                row = todayRow,
                label = todayLabel,
                date = todayDate,
                temperature = todayTemperature,
                condition = todayCondition,
                sanitizeDisplayText = sanitizeDisplayText,
                weatherConditionIcon = weatherConditionIcon
            )

            if (chartPoints.size >= 3) {
                WeatherTrendChart(
                    points = chartPoints,
                    dark = darkMode,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(132.dp)
                )
            } else if (chartPoints.size >= 2) {
                WeatherTrendChart(
                    points = chartPoints,
                    dark = darkMode,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(124.dp)
                )
            } else if (orderedRows.size >= 2) {
                WeatherConditionTimeline(
                    rows = orderedRows,
                    sanitizeDisplayText = sanitizeDisplayText,
                    weatherConditionIcon = weatherConditionIcon,
                    modifier = Modifier.fillMaxWidth(),
                    dark = darkMode
                )
            }

            if (laterRows.isNotEmpty()) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    MarkdownText(
                        text = "Day-by-day details",
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                        color = MaterialTheme.colorScheme.onSurface,
                        modifier = Modifier.weight(1f)
                    )
                    MarkdownText(
                        text = "${orderedRows.size} days",
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.End
                    )
                }
            }

            laterRows.forEach { row ->
                WeatherDayDetailCard(
                    row = row,
                    temperature = weatherTemperatureText(row),
                    sanitizeDisplayText = sanitizeDisplayText,
                    weatherConditionIcon = weatherConditionIcon,
                    dark = darkMode
                )
            }
        }
    }

    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    private fun WeatherHeroSummaryCard(
        row: WeatherRow,
        label: String,
        date: String?,
        temperature: String,
        condition: String,
        sanitizeDisplayText: (String) -> String,
        weatherConditionIcon: @Composable (String?, Dp) -> Unit
    ) {
        val heroShape = RoundedCornerShape(GenUiTokens.RadiusXl)
        val heroPalette = weatherHeroPalette(row.condition)
        val heroMetrics = remember(row, temperature) { buildHeroWeatherMetrics(row, temperature) }
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .semantics(mergeDescendants = true) {
                    contentDescription = weatherRowAccessibilityLabel(
                        period = label,
                        date = date,
                        temperature = temperature,
                        condition = condition,
                        metrics = row.metrics,
                        sanitizeDisplayText = sanitizeDisplayText
                    )
                },
            shape = heroShape,
            colors = CardDefaults.cardColors(containerColor = Color.Transparent),
            elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(heroShape)
                    .background(Brush.linearGradient(heroPalette.gradient))
                    .padding(horizontal = 20.dp, vertical = 20.dp)
            ) {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(14.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(
                            modifier = Modifier.weight(1f),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            MarkdownText(
                                text = listOfNotNull(label, date).filter { it.isNotBlank() }.joinToString(", "),
                                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                                color = heroPalette.mutedContent,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                            if (temperature.isNotBlank()) {
                                MarkdownText(
                                    text = sanitizeDisplayText(temperature),
                                    style = MaterialTheme.typography.displaySmall.copy(fontWeight = FontWeight.Bold),
                                    color = heroPalette.content,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                            if (condition.isNotBlank()) {
                                MarkdownText(
                                    text = condition,
                                    style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold),
                                    color = heroPalette.content.copy(alpha = 0.90f),
                                    maxLines = 2,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                        }
                        Box(
                            modifier = Modifier.size(92.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            weatherConditionIcon(row.condition, 72.dp)
                        }
                    }

                    if (heroMetrics.isNotEmpty()) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            heroMetrics.take(3).forEach { (metricLabel, metricValue) ->
                                WeatherHeroMetricTile(
                                    label = sanitizeDisplayText(metricLabel),
                                    value = sanitizeDisplayText(metricValue),
                                    palette = heroPalette,
                                    modifier = Modifier.weight(1f)
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    @Composable
    private fun WeatherHeroMetricTile(
        label: String,
        value: String,
        palette: WeatherHeroPalette,
        modifier: Modifier = Modifier
    ) {
        if (label.isBlank() || value.isBlank()) {
            return
        }
        Column(
            modifier = modifier
                .clip(RoundedCornerShape(17.dp))
                .background(palette.tileContainer)
                .border(
                    GenUiTokens.BorderMd,
                    palette.tileBorder,
                    RoundedCornerShape(17.dp)
                )
                .padding(horizontal = 10.dp, vertical = 9.dp),
            verticalArrangement = Arrangement.spacedBy(3.dp)
        ) {
            MarkdownText(
                text = label,
                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                color = palette.mutedContent,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            MarkdownText(
                text = value,
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                color = palette.content,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }

    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    private fun WeatherDayDetailCard(
        row: WeatherRow,
        temperature: String,
        sanitizeDisplayText: (String) -> String,
        weatherConditionIcon: @Composable (String?, Dp) -> Unit,
        dark: Boolean
    ) {
        val periodText = sanitizeDisplayText(row.period)
        val dateText = sanitizeDisplayText(row.date.orEmpty()).ifBlank { null }
        val conditionText = sanitizeDisplayText(row.condition.orEmpty()).ifBlank { null }
        val advice = remember(row.metrics) { findWeatherAdvice(row.metrics) }
        val rainMetric = remember(row.metrics) { findRainMetric(row.metrics) }
        val daypartMetrics = remember(row.metrics) { findWeatherDaypartMetrics(row.metrics) }
        val rainPercent = remember(rainMetric) { rainMetric?.second?.let(::extractPercentValue) }
        val metricTiles = remember(row.metrics, advice, rainMetric, daypartMetrics) {
            val daypartKeys = daypartMetrics.map { metric -> weatherMetricKey(metric.first) }.toSet()
            row.metrics
                .filterNot { metric -> advice != null && metric.first == advice.first && metric.second == advice.second }
                .filterNot { metric -> rainMetric != null && metric.first == rainMetric.first && metric.second == rainMetric.second }
                .filterNot { metric -> weatherMetricKey(metric.first) in daypartKeys }
                .take(3)
        }
        val range = remember(temperature) { splitWeatherTemperatureRange(temperature) }
        val cardContainer = if (dark) {
            MaterialTheme.colorScheme.surfaceContainerHigh.copy(alpha = 0.92f)
        } else {
            MaterialTheme.colorScheme.surface.copy(alpha = 0.88f)
        }
        val cardBorder = if (dark) {
            MaterialTheme.colorScheme.outline.copy(alpha = 0.28f)
        } else {
            MaterialTheme.colorScheme.outline.copy(alpha = 0.14f)
        }
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
            shape = RoundedCornerShape(GenUiTokens.RadiusXl),
            colors = CardDefaults.cardColors(containerColor = cardContainer),
            border = BorderStroke(GenUiTokens.BorderMd, cardBorder),
            elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 15.dp, vertical = 15.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(11.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Box(
                        modifier = Modifier
                            .size(44.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        weatherConditionIcon(row.condition, 34.dp)
                    }
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(2.dp)
                    ) {
                        MarkdownText(
                            text = periodText,
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                            color = MaterialTheme.colorScheme.onSurface,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                        conditionText?.let { condition ->
                            MarkdownText(
                                text = condition,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
                    }
                    if (range.primary.isNotBlank()) {
                        Column(horizontalAlignment = Alignment.End) {
                            MarkdownText(
                                text = sanitizeDisplayText(range.primary),
                                style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
                                color = MaterialTheme.colorScheme.onSurface,
                                textAlign = TextAlign.End,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                            range.secondary?.takeIf { it.isNotBlank() }?.let { low ->
                                MarkdownText(
                                    text = "${sanitizeDisplayText(low)} low",
                                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    textAlign = TextAlign.End,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                        }
                    }
                }

                dateText?.let { date ->
                    MarkdownText(
                        text = date,
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }

                if (metricTiles.isNotEmpty()) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        metricTiles.forEach { (label, value) ->
                            WeatherDayMetricTile(
                                label = sanitizeDisplayText(label),
                                value = sanitizeDisplayText(value),
                                dark = dark,
                                modifier = Modifier.weight(1f)
                            )
                        }
                    }
                }

                if (rainMetric != null || daypartMetrics.isNotEmpty()) {
                    val (label, value) = rainMetric ?: ("Hourly outlook" to "")
                    WeatherRainRiskStrip(
                        label = sanitizeDisplayText(label),
                        value = sanitizeDisplayText(value),
                        percent = rainPercent,
                        dark = dark,
                        dayparts = daypartMetrics.map { (partLabel, partValue) ->
                            sanitizeDisplayText(partLabel) to sanitizeDisplayText(partValue)
                        }
                    )
                }

                advice?.let { (_, value) ->
                    WeatherAdvicePanel(
                        advice = sanitizeDisplayText(value),
                        dark = dark
                    )
                }
            }
        }
    }

    @Composable
    private fun WeatherDayMetricTile(
        label: String,
        value: String,
        dark: Boolean,
        modifier: Modifier = Modifier
    ) {
        if (label.isBlank() || value.isBlank()) {
            return
        }
        Column(
            modifier = modifier
                .clip(RoundedCornerShape(16.dp))
                .background(
                    if (dark) {
                        MaterialTheme.colorScheme.surfaceContainerHighest.copy(alpha = 0.74f)
                    } else {
                        MaterialTheme.colorScheme.surfaceContainerLow.copy(alpha = 0.90f)
                    }
                )
                .border(
                    GenUiTokens.BorderMd,
                    MaterialTheme.colorScheme.outline.copy(alpha = if (dark) 0.22f else 0.12f),
                    RoundedCornerShape(16.dp)
                )
                .padding(horizontal = 10.dp, vertical = 9.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            MarkdownText(
                text = label,
                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            MarkdownText(
                text = value,
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
        }
    }

    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    private fun WeatherRainRiskStrip(
        label: String,
        value: String,
        percent: Float?,
        dark: Boolean,
        dayparts: List<Pair<String, String>> = emptyList()
    ) {
        val safeDayparts = dayparts.filter { (_, partValue) -> partValue.isNotBlank() }
        if (label.isBlank() || (value.isBlank() && safeDayparts.isEmpty())) {
            return
        }
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(18.dp))
                .background(weatherRainAccent(dark).copy(alpha = if (dark) 0.16f else 0.10f))
                .border(
                    GenUiTokens.BorderMd,
                    weatherRainAccent(dark).copy(alpha = if (dark) 0.30f else 0.20f),
                    RoundedCornerShape(18.dp)
                )
                .padding(horizontal = 12.dp, vertical = 11.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                MarkdownText(
                    text = if (isRainLikeLabel(label)) "Rain risk" else label,
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                    color = weatherRainAccent(dark),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                if (value.isNotBlank()) {
                    MarkdownText(
                        text = value,
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                        color = weatherRainAccent(dark),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
            percent?.let { safePercent ->
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp)
                        .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                        .background(MaterialTheme.colorScheme.onSurface.copy(alpha = 0.08f))
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth(safePercent.coerceIn(0f, 100f) / 100f)
                            .height(8.dp)
                            .clip(RoundedCornerShape(GenUiTokens.RadiusPill))
                            .background(
                                Brush.linearGradient(
                                    listOf(Color(0xFF0EA5E9), Color(0xFF2563EB), Color(0xFF14B8A6))
                                )
                            )
                    )
                }
            }
            if (safeDayparts.isNotEmpty()) {
                FlowRow(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    safeDayparts.forEach { (partLabel, partValue) ->
                        WeatherDaypartChip(
                            label = partLabel,
                            value = partValue,
                            dark = dark
                        )
                    }
                }
            }
        }
    }

    @Composable
    private fun WeatherDaypartChip(
        label: String,
        value: String,
        dark: Boolean
    ) {
        if (label.isBlank() || value.isBlank()) {
            return
        }
        Column(
            modifier = Modifier
                .clip(RoundedCornerShape(14.dp))
                .background(
                    if (dark) {
                        Color.White.copy(alpha = 0.10f)
                    } else {
                        Color.White.copy(alpha = 0.82f)
                    }
                )
                .border(
                    GenUiTokens.BorderMd,
                    weatherRainAccent(dark).copy(alpha = if (dark) 0.24f else 0.18f),
                    RoundedCornerShape(14.dp)
                )
                .padding(horizontal = 9.dp, vertical = 7.dp),
            verticalArrangement = Arrangement.spacedBy(2.dp)
        ) {
            MarkdownText(
                text = label,
                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            MarkdownText(
                text = value,
                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }

    @Composable
    private fun WeatherAdvicePanel(
        advice: String,
        dark: Boolean
    ) {
        if (advice.isBlank()) {
            return
        }
        val accent = if (dark) Color(0xFF86EFAC) else Color(0xFF047857)
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(18.dp))
                .background(accent.copy(alpha = if (dark) 0.12f else 0.09f))
                .border(
                    GenUiTokens.BorderMd,
                    accent.copy(alpha = if (dark) 0.22f else 0.14f),
                    RoundedCornerShape(18.dp)
                )
                .padding(horizontal = 12.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.Top
        ) {
            Box(
                modifier = Modifier
                    .size(30.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(accent.copy(alpha = if (dark) 0.18f else 0.16f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Filled.CheckCircle,
                    contentDescription = "Advice",
                    tint = accent,
                    modifier = Modifier.size(20.dp)
                )
            }
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(3.dp)
            ) {
                MarkdownText(
                    text = "What to wear",
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                    color = accent,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                MarkdownText(
                    text = advice,
                    style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold),
                    color = MaterialTheme.colorScheme.onSurface,
                    maxLines = 4,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }

    private fun buildHeroWeatherMetrics(
        row: WeatherRow,
        temperature: String
    ): List<Pair<String, String>> {
        val metrics = mutableListOf<Pair<String, String>>()
        val highLow = when {
            !row.high.isNullOrBlank() && !row.low.isNullOrBlank() -> "${row.high} / ${row.low}"
            temperature.contains("/") -> temperature
            else -> null
        }
        if (!highLow.isNullOrBlank()) {
            metrics += "High / Low" to highLow
        } else if (temperature.isNotBlank()) {
            metrics += "Temp" to temperature
        }
        findRainMetric(row.metrics)?.let { metrics += it }
        findFirstMetric(row.metrics, "wind")?.let { metrics += it }
        findFirstMetric(row.metrics, "humidity")?.let { metrics += it }
        findFirstMetric(row.metrics, "uv")?.let { metrics += it }
        return metrics.distinctBy { metric -> NativeWeatherSemantics.normalizeWeatherText(metric.first) }
    }

    private fun findWeatherAdvice(metrics: List<Pair<String, String>>): Pair<String, String>? =
        metrics.firstOrNull { (label, value) ->
            value.isNotBlank() && isAdviceLikeLabel(label)
        }

    private fun findRainMetric(metrics: List<Pair<String, String>>): Pair<String, String>? =
        metrics.firstOrNull { (label, value) ->
            value.isNotBlank() && isRainLikeLabel(label)
        }

    private fun findWeatherDaypartMetrics(metrics: List<Pair<String, String>>): List<Pair<String, String>> {
        val order = listOf("morning", "afternoon", "evening", "night")
        val byKey = metrics
            .filter { (label, value) -> value.isNotBlank() && weatherMetricKey(label) in order }
            .associateBy { (label, _) -> weatherMetricKey(label) }
        return order.mapNotNull { key -> byKey[key] }
    }

    private fun findFirstMetric(
        metrics: List<Pair<String, String>>,
        vararg labelTokens: String
    ): Pair<String, String>? =
        metrics.firstOrNull { (label, value) ->
            val normalized = NativeWeatherSemantics.normalizeWeatherText(label)
            value.isNotBlank() && labelTokens.any { token -> normalized.contains(token) }
        }

    private fun isAdviceLikeLabel(label: String): Boolean {
        val normalized = NativeWeatherSemantics.normalizeWeatherText(label)
        return normalized.contains("wear") ||
            normalized.contains("cloth") ||
            normalized.contains("outfit") ||
            normalized.contains("advice") ||
            normalized.contains("recommend")
    }

    private fun isRainLikeLabel(label: String): Boolean {
        val normalized = NativeWeatherSemantics.normalizeWeatherText(label)
        return normalized.contains("rain") ||
            normalized.contains("precip") ||
            normalized.contains("shower")
    }

    private fun weatherMetricKey(label: String): String =
        NativeWeatherSemantics.normalizeWeatherText(label).trim()

    @Composable
    private fun weatherRainAccent(dark: Boolean): Color =
        if (dark) Color(0xFF38BDF8) else Color(0xFF2563EB)

    private fun extractPercentValue(value: String): Float? {
        val match = Regex("""(\d{1,3}(?:\.\d+)?)\s*%""").find(value) ?: return null
        return match.groupValues.getOrNull(1)?.toFloatOrNull()?.coerceIn(0f, 100f)
    }

    private fun splitWeatherTemperatureRange(value: String): WeatherTemperatureRange {
        val clean = value.trim()
        if (clean.isBlank()) {
            return WeatherTemperatureRange("", null)
        }
        val parts = Regex("""\s*(?:/|\u2013|\u2014|\bto\b|\s-\s)\s*""", RegexOption.IGNORE_CASE)
            .split(clean, limit = 2)
            .map { it.trim() }
            .filter { it.isNotBlank() }
        return if (parts.size >= 2) {
            WeatherTemperatureRange(parts[0], parts[1])
        } else {
            WeatherTemperatureRange(clean, null)
        }
    }

    @Composable
    private fun WeatherTrendChart(
        points: List<WeatherChartPoint>,
        modifier: Modifier = Modifier,
        dark: Boolean = true
    ) {
        val safePoints = remember(points) { points.take(7) }
        val minValue = remember(safePoints) {
            safePoints.flatMap { point -> listOfNotNull(point.value, point.lowValue) }.minOrNull() ?: 0f
        }
        val maxValue = remember(safePoints) {
            safePoints.flatMap { point -> listOfNotNull(point.value, point.lowValue) }.maxOrNull() ?: minValue
        }
        val panelColor = if (dark) Color.White.copy(alpha = 0.13f) else Color.White.copy(alpha = 0.72f)
        val borderColor = if (dark) Color.White.copy(alpha = 0.16f) else MaterialTheme.colorScheme.outline.copy(alpha = 0.10f)
        val primaryColor = if (dark) Color.White else Color(0xFF2563EB)
        val secondaryColor = if (dark) Color.White.copy(alpha = 0.48f) else Color(0xFF0EA5E9).copy(alpha = 0.62f)
        val fillColor = if (dark) Color.White.copy(alpha = 0.10f) else Color(0xFF0EA5E9).copy(alpha = 0.12f)
        val labelColor = if (dark) Color.White.copy(alpha = 0.72f) else MaterialTheme.colorScheme.onSurfaceVariant
        val titleColor = if (dark) Color.White.copy(alpha = 0.92f) else MaterialTheme.colorScheme.onSurface
        Column(
            modifier = modifier
                .clip(RoundedCornerShape(GenUiTokens.RadiusLg))
                .background(panelColor)
                .border(
                    GenUiTokens.BorderMd,
                    borderColor,
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
                    color = titleColor
                )
                safePoints.firstOrNull()?.displayValue?.takeIf { it.isNotBlank() }?.let { value ->
                    MarkdownText(
                        text = value,
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                        color = titleColor
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

                val gridColor = if (dark) Color.White.copy(alpha = 0.18f) else Color(0xFF172033).copy(alpha = 0.10f)
                drawLine(gridColor, Offset(0f, bottom), Offset(size.width, bottom), strokeWidth = 1.4f)
                drawLine(gridColor, Offset(0f, top), Offset(size.width, top), strokeWidth = 1.0f)

                val highPath = Path().apply {
                    safePoints.forEachIndexed { index, point ->
                        val px = x(index)
                        val py = y(point.value)
                        if (index == 0) moveTo(px, py) else lineTo(px, py)
                    }
                }
                val lowPoints = safePoints.mapIndexedNotNull { index, point ->
                    point.lowValue?.let { value -> index to value }
                }
                if (lowPoints.size >= 2) {
                    val bandPath = Path().apply {
                        safePoints.forEachIndexed { index, point ->
                            val px = x(index)
                            val py = y(point.value)
                            if (index == 0) moveTo(px, py) else lineTo(px, py)
                        }
                        lowPoints.asReversed().forEach { (index, value) ->
                            lineTo(x(index), y(value))
                        }
                        close()
                    }
                    drawPath(path = bandPath, color = fillColor)
                    val lowPath = Path().apply {
                        lowPoints.forEachIndexed { lowIndex, (index, value) ->
                            val px = x(index)
                            val py = y(value)
                            if (lowIndex == 0) moveTo(px, py) else lineTo(px, py)
                        }
                    }
                    drawPath(
                        path = lowPath,
                        color = secondaryColor,
                        style = Stroke(width = 3f, cap = StrokeCap.Round)
                    )
                }
                drawPath(
                    path = highPath,
                    color = primaryColor,
                    style = Stroke(width = 5f, cap = StrokeCap.Round)
                )
                safePoints.forEachIndexed { index, point ->
                    drawCircle(
                        color = primaryColor,
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
                        color = labelColor,
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
        modifier: Modifier = Modifier,
        dark: Boolean = true
    ) {
        val visibleRows = remember(rows) { rows.take(5) }
        val panelColor = if (dark) Color.White.copy(alpha = 0.13f) else Color.White.copy(alpha = 0.72f)
        val borderColor = if (dark) Color.White.copy(alpha = 0.16f) else MaterialTheme.colorScheme.outline.copy(alpha = 0.10f)
        val titleColor = if (dark) Color.White.copy(alpha = 0.92f) else MaterialTheme.colorScheme.onSurface
        val labelColor = if (dark) Color.White else MaterialTheme.colorScheme.onSurface
        val secondaryTextColor = if (dark) Color.White.copy(alpha = 0.72f) else MaterialTheme.colorScheme.onSurfaceVariant
        val lineColor = if (dark) Color.White.copy(alpha = 0.25f) else MaterialTheme.colorScheme.primary.copy(alpha = 0.20f)
        Column(
            modifier = modifier
                .clip(RoundedCornerShape(GenUiTokens.RadiusLg))
                .background(panelColor)
                .border(
                    GenUiTokens.BorderMd,
                    borderColor,
                    RoundedCornerShape(GenUiTokens.RadiusLg)
                )
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            MarkdownText(
                text = "Outlook trend",
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                color = titleColor
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
                        color = lineColor,
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
                            weatherConditionIcon(row.condition, 26.dp)
                            MarkdownText(
                                text = compactWeatherChartLabel(periodText),
                                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                                color = labelColor,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                            if (conditionText.isNotBlank()) {
                                MarkdownText(
                                    text = conditionText,
                                    style = MaterialTheme.typography.labelSmall,
                                    color = secondaryTextColor,
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
            val value = extractWeatherHighChartValue(row, display) ?: return@mapNotNull null
            val lowValue = extractWeatherLowChartValue(row, display)
            WeatherChartPoint(
                label = compactWeatherChartLabel(sanitizeDisplayText(row.period)),
                value = value,
                displayValue = display,
                lowValue = lowValue,
                lowDisplayValue = row.low
            )
        }
    }

    private fun extractWeatherHighChartValue(
        row: WeatherRow,
        display: String
    ): Float? {
        val preferredSources = listOf(display, row.high, row.temp, row.low)
        preferredSources.forEach { source ->
            val values = extractTemperatureNumbers(source.orEmpty())
            if (values.isNotEmpty()) {
                return values.maxOrNull()
            }
        }
        return null
    }

    private fun extractWeatherLowChartValue(
        row: WeatherRow,
        display: String
    ): Float? {
        extractTemperatureNumbers(display).takeIf { it.size >= 2 }?.let { values ->
            return values.minOrNull()
        }
        extractTemperatureNumbers(row.low.orEmpty()).takeIf { it.isNotEmpty() }?.let { values ->
            return values.minOrNull()
        }
        extractTemperatureNumbers(row.temp.orEmpty()).takeIf { it.size >= 2 }?.let { values ->
            return values.minOrNull()
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

    @Composable
    private fun weatherHeroPalette(condition: String?): WeatherHeroPalette {
        val dark = isSystemInDarkTheme()
        val key = condition.orEmpty().lowercase()
        val gradient = when {
            dark && (key.contains("thunder") || key.contains("storm")) -> listOf(
                Color(0xFF111827),
                Color(0xFF1D4ED8),
                Color(0xFF0F172A)
            )

            dark && (key.contains("rain") || key.contains("shower") || key.contains("drizzle")) -> listOf(
                Color(0xFF0F172A),
                Color(0xFF075985),
                Color(0xFF115E59)
            )

            dark && (key.contains("cloud") || key.contains("overcast") || key.contains("fog") || key.contains("mist")) -> listOf(
                Color(0xFF111827),
                Color(0xFF334155),
                Color(0xFF0F766E)
            )

            dark && (key.contains("sun") || key.contains("clear") || key.contains("hot")) -> listOf(
                Color(0xFF0F172A),
                Color(0xFFB45309),
                Color(0xFF075985)
            )

            dark -> listOf(
                Color(0xFF0F172A),
                Color(0xFF0369A1),
                Color(0xFF0F766E)
            )

            key.contains("thunder") || key.contains("storm") -> listOf(
                Color(0xFFE0E7FF),
                Color(0xFFE0F2FE),
                Color(0xFFD1FAE5)
            )

            key.contains("rain") || key.contains("shower") || key.contains("drizzle") -> listOf(
                Color(0xFFE0F2FE),
                Color(0xFFDBEAFE),
                Color(0xFFD1FAE5)
            )

            key.contains("cloud") || key.contains("overcast") || key.contains("fog") || key.contains("mist") -> listOf(
                Color(0xFFE2E8F0),
                Color(0xFFE0F2FE),
                Color(0xFFEFF6FF)
            )

            key.contains("sun") || key.contains("clear") || key.contains("hot") -> listOf(
                Color(0xFFFEF3C7),
                Color(0xFFE0F2FE),
                Color(0xFFDBEAFE)
            )

            key.contains("snow") || key.contains("ice") || key.contains("cold") -> listOf(
                Color(0xFFE0F2FE),
                Color(0xFFF8FAFC),
                Color(0xFFDBEAFE)
            )

            else -> listOf(
                Color(0xFFE0F2FE),
                Color(0xFFDBEAFE),
                Color(0xFFD1FAE5)
            )
        }
        val content = if (dark) Color.White else Color(0xFF0F172A)
        val mutedContent = if (dark) {
            Color.White.copy(alpha = 0.78f)
        } else {
            Color(0xFF334155).copy(alpha = 0.86f)
        }
        return WeatherHeroPalette(
            gradient = gradient,
            content = content,
            mutedContent = mutedContent,
            tileContainer = if (dark) Color.White.copy(alpha = 0.13f) else Color.White.copy(alpha = 0.68f),
            tileBorder = if (dark) Color.White.copy(alpha = 0.18f) else Color(0xFF38BDF8).copy(alpha = 0.28f)
        )
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
