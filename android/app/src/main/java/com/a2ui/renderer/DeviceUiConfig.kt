package com.samsung.genuicraft

import android.content.res.Configuration
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.Immutable
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.unit.LayoutDirection
import java.util.Locale
import kotlin.math.roundToInt

enum class DeviceSizeClass {
    Compact,
    Medium,
    Expanded
}

@Immutable
data class DeviceUiConfig(
    val screenWidthDp: Int,
    val screenHeightDp: Int,
    val smallestWidthDp: Int,
    val density: Float,
    val fontScale: Float,
    val isLandscape: Boolean,
    val isDarkTheme: Boolean,
    val localeTag: String,
    val layoutDirection: LayoutDirection,
    val widthClass: DeviceSizeClass,
    val heightClass: DeviceSizeClass
) {
    fun allowLargeTextListLayout(): Boolean {
        return (screenWidthDp <= 320 && fontScale >= 1.15f) ||
            (screenWidthDp < 411 && fontScale >= 1.3f)
    }

    fun webViewTextZoomPercent(): Int {
        return (fontScale * 100f).roundToInt().coerceIn(85, 220)
    }

    fun orientationLabel(): String = if (isLandscape) "Landscape" else "Portrait"

    fun layoutDirectionLabel(): String = if (layoutDirection == LayoutDirection.Rtl) "RTL" else "LTR"
}

@Composable
fun rememberDeviceUiConfig(): DeviceUiConfig {
    val configuration = LocalConfiguration.current
    val density = LocalDensity.current
    val layoutDirection = LocalLayoutDirection.current
    val darkTheme = isSystemInDarkTheme()

    return DeviceUiConfig(
        screenWidthDp = configuration.screenWidthDp,
        screenHeightDp = configuration.screenHeightDp,
        smallestWidthDp = configuration.smallestScreenWidthDp,
        density = density.density,
        fontScale = density.fontScale,
        isLandscape = configuration.orientation == Configuration.ORIENTATION_LANDSCAPE,
        isDarkTheme = darkTheme,
        localeTag = configuration.primaryLocale().toLanguageTag(),
        layoutDirection = layoutDirection,
        widthClass = widthClassFor(configuration.screenWidthDp),
        heightClass = heightClassFor(configuration.screenHeightDp)
    )
}

private fun widthClassFor(widthDp: Int): DeviceSizeClass {
    return when {
        widthDp < 600 -> DeviceSizeClass.Compact
        widthDp < 840 -> DeviceSizeClass.Medium
        else -> DeviceSizeClass.Expanded
    }
}

private fun heightClassFor(heightDp: Int): DeviceSizeClass {
    return when {
        heightDp < 480 -> DeviceSizeClass.Compact
        heightDp < 900 -> DeviceSizeClass.Medium
        else -> DeviceSizeClass.Expanded
    }
}

private fun Configuration.primaryLocale(): Locale {
    return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
        val locales = locales
        if (locales.isEmpty) Locale.getDefault() else locales[0]
    } else {
        @Suppress("DEPRECATION")
        locale ?: Locale.getDefault()
    }
}

fun DeviceSizeClass.cssToken(): String {
    return when (this) {
        DeviceSizeClass.Compact -> "compact"
        DeviceSizeClass.Medium -> "medium"
        DeviceSizeClass.Expanded -> "expanded"
    }
}
