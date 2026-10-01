package com.samsung.genuicraft.renderer.native.text

import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import java.util.Locale

internal object NativeVariantStyleResolver {
    @Composable
    fun textStyleForVariant(variant: String): TextStyle = when (variant.lowercase(Locale.US)) {
        "h1" -> MaterialTheme.typography.displayLarge
        "h2" -> MaterialTheme.typography.displayMedium
        "h3" -> MaterialTheme.typography.displaySmall
        "h4" -> MaterialTheme.typography.titleLarge
        "h5" -> MaterialTheme.typography.titleMedium
        "h6" -> MaterialTheme.typography.titleSmall
        "caption" -> MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Normal)
        else -> MaterialTheme.typography.bodyMedium
    }

    @Composable
    fun textColorForVariant(variant: String): Color = when (variant.lowercase(Locale.US)) {
        "h1", "h2", "h3", "h4", "h5", "h6" -> MaterialTheme.colorScheme.onSurface
        "caption" -> MaterialTheme.colorScheme.onSurfaceVariant
        else -> MaterialTheme.colorScheme.onSurface
    }
}
