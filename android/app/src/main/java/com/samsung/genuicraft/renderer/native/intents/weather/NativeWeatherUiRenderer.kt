package com.samsung.genuicraft.renderer.native.intents.weather

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
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

internal object NativeWeatherUiRenderer {
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

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(GenUiTokens.RadiusXl),
            colors = CardDefaults.cardColors(containerColor = genUiTableContainerColor()),
            elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 10.dp, vertical = 10.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(GenUiTokens.RadiusLg),
                    colors = genUiCardColors(GenUiCardTone.Primary),
                    elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 12.dp, vertical = 12.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(10.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(
                                modifier = Modifier.weight(1f),
                                horizontalArrangement = Arrangement.spacedBy(10.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                weatherConditionIcon(todayRow.condition, 32.dp)
                                Column(
                                    verticalArrangement = Arrangement.spacedBy(2.dp)
                                ) {
                                    MarkdownText(
                                        text = todayLabel,
                                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                    todayDate?.let { dateValue ->
                                        MarkdownText(
                                            text = dateValue,
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }
                            }

                            if (todayTemperature.isNotBlank()) {
                                MarkdownText(
                                    text = sanitizeDisplayText(todayTemperature),
                                    style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Bold),
                                    color = MaterialTheme.colorScheme.onSurface,
                                    textAlign = TextAlign.End
                                )
                            }
                        }

                        if (todayCondition.isNotBlank()) {
                            MarkdownText(
                                text = todayCondition,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        if (todayRow.metrics.isNotEmpty()) {
                            FlowRow(
                                horizontalArrangement = Arrangement.spacedBy(6.dp),
                                verticalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                todayRow.metrics.forEach { (label, value) ->
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
                                                .padding(horizontal = 8.dp, vertical = 5.dp)
                                        )
                                    }
                                }
                            }
                        }
                    }
                }

                laterRows.forEach { row ->
                    val temperature = weatherTemperatureText(row)
                    val periodText = sanitizeDisplayText(row.period)
                    val dateText = sanitizeDisplayText(row.date.orEmpty()).ifBlank { null }
                    val conditionText = sanitizeDisplayText(row.condition.orEmpty()).ifBlank { null }
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(GenUiTokens.RadiusLg),
                        colors = genUiCardColors(GenUiCardTone.Neutral),
                        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 12.dp, vertical = 10.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(10.dp),
                                verticalAlignment = Alignment.Top
                            ) {
                                Row(
                                    modifier = Modifier.weight(1f),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    verticalAlignment = Alignment.Top
                                ) {
                                    weatherConditionIcon(row.condition, 26.dp)
                                    Column(
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
                                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                                    verticalArrangement = Arrangement.spacedBy(6.dp)
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
                                                    .padding(horizontal = 8.dp, vertical = 5.dp)
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
