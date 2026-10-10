package com.samsung.genuicraft.sdk.internal.renderer.native.intents.airquality

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import com.samsung.genuicraft.sdk.internal.renderer.CitationText as Text
import java.util.Locale

/** Only an authored category can choose a status accent; the numeric reading is never classified. */
internal fun authoredAirQualityAccent(summary: NativeAirQualitySummary): Color? {
    val category = summary.categoryIndex?.let { summary.fields.getOrNull(it) }?.value
        ?.trim()?.takeIf(String::isNotBlank) ?: return null
    return when (category.lowercase(Locale.ROOT)) {
        "good", "satisfactory" -> Color(0xFF1B6845)
        "moderate" -> Color(0xFF7A5700)
        "poor" -> Color(0xFF8A4300)
        "very poor" -> Color(0xFF922E2E)
        "severe" -> Color(0xFF682C64)
        else -> Color(0xFF315987)
    }
}

/** One compact panel; all source fields wrap at natural height without truncation. */
internal object NativeAirQualityUiRenderer {
    @Composable
    fun RenderSummary(summary: NativeAirQualitySummary, title: String? = null, modifier: Modifier = Modifier) {
        val colors = MaterialTheme.colorScheme
        val category = summary.categoryIndex?.let { summary.fields.getOrNull(it) }
            ?.takeIf { it.value.isNotBlank() }
        val accent = authoredAirQualityAccent(summary) ?: colors.primary
        val darkSurface = colors.surface.luminance() < 0.5f
        Surface(
            modifier = modifier.fillMaxWidth().semantics { contentDescription = "Air quality summary" },
            shape = RoundedCornerShape(16.dp),
            color = colors.surface,
            border = BorderStroke(1.dp, accent.copy(alpha = if (darkSurface) 0.45f else 0.24f)),
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(7.dp),
            ) {
                title?.takeIf(String::isNotBlank)?.let {
                    Text(it, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                }
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(12.dp))
                        .background(accent.copy(alpha = if (darkSurface) 0.25f else 0.09f))
                        .padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(5.dp),
                ) {
                    category?.let { authored ->
                        Surface(shape = RoundedCornerShape(50.dp), color = accent) {
                            // Display the complete supplied category, including any citation markers.
                            Text(
                                authored.value,
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                                style = MaterialTheme.typography.labelMedium,
                                color = Color.White,
                                fontWeight = FontWeight.Bold,
                            )
                        }
                    }
                    Text(summary.reading.label, style = MaterialTheme.typography.labelMedium, color = colors.onSurfaceVariant)
                    // Keep the full supplied reading in one Text node, including scale and citations.
                    Text(
                        summary.reading.value,
                        style = MaterialTheme.typography.displaySmall.copy(fontFeatureSettings = "tnum"),
                        color = colors.onSurface,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.semantics { heading() },
                    )
                }
                if (summary.fields.size > 1) {
                    HorizontalDivider(modifier = Modifier.padding(vertical = 3.dp), color = colors.outlineVariant)
                }
                summary.fields.forEachIndexed { index, field ->
                    if (index != summary.readingIndex) {
                        Text(
                            text = buildAnnotatedString {
                                withStyle(SpanStyle(color = colors.onSurfaceVariant)) {
                                    append(field.label)
                                    if (field.label.isNotBlank() && field.value.isNotBlank()) append(": ")
                                }
                                withStyle(SpanStyle(color = colors.onSurface, fontWeight =
                                    if (index == summary.categoryIndex) FontWeight.Bold else FontWeight.Medium)) {
                                    append(field.value)
                                }
                            },
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    }
                }
            }
        }
    }
}
