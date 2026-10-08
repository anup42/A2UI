package com.samsung.genuicraft.sdk.internal.renderer.native.intents.airquality

import androidx.compose.foundation.BorderStroke
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
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import com.samsung.genuicraft.sdk.internal.renderer.CitationText as Text

/** One compact neutral panel; all source fields wrap at natural height without truncation. */
internal object NativeAirQualityUiRenderer {
    @Composable
    fun RenderSummary(summary: NativeAirQualitySummary, title: String? = null, modifier: Modifier = Modifier) {
        val colors = MaterialTheme.colorScheme
        Surface(
            modifier = modifier.fillMaxWidth().semantics { contentDescription = "Air quality summary" },
            shape = RoundedCornerShape(16.dp),
            color = colors.surface,
            border = BorderStroke(1.dp, colors.outlineVariant.copy(alpha = 0.5f)),
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(7.dp),
            ) {
                title?.takeIf(String::isNotBlank)?.let {
                    Text(it, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
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
