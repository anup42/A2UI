package com.samsung.genuicraft

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CardColors
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

object GenUiTokens {
    val RadiusSm = 8.dp
    val RadiusMd = 12.dp
    val RadiusLg = 22.dp
    val RadiusXl = 26.dp
    val RadiusPill = 100.dp

    val BorderSm = 0.5.dp
    val BorderMd = 1.dp
    val BorderLg = 2.dp

    val ElevationSm = 4.dp
    val ElevationMd = 12.dp
    val ElevationLg = 20.dp
    val ElevationXl = 32.dp
}

enum class GenUiCardTone {
    Neutral,
    Primary,
    Positive,
    Warning,
    Error
}

private val LightScheme: ColorScheme = lightColorScheme(
    primary = Color(0xFF387AFF),
    onPrimary = Color(0xFFFFFFFF),
    secondary = Color(0xFF2E65D4),
    onSecondary = Color(0xFFFFFFFF),
    tertiary = Color(0xFFE65B17),
    onTertiary = Color(0xFFFFFFFF),
    secondaryContainer = Color(0xFFD6FFEB),
    onSecondaryContainer = Color(0xFF074225),
    tertiaryContainer = Color(0xFFFFEDDA),
    onTertiaryContainer = Color(0xFF4D1E08),
    background = Color(0xFFF1F1F3),
    onBackground = Color(0xFF010102),
    surface = Color(0xFFFCFCFF),
    onSurface = Color(0xFF252528),
    surfaceVariant = Color(0xFFE9E9EC),
    onSurfaceVariant = Color(0xFF4D4D52),
    surfaceTint = Color(0xFF387AFF),
    outline = Color(0xFFB7B7BB),
    outlineVariant = Color(0xFFE4E4E7),
    error = Color(0xFFD93E36),
    onError = Color(0xFFFFFFFF),
    scrim = Color(0x99000000)
)

private val DarkScheme: ColorScheme = darkColorScheme(
    primary = Color(0xFF387AFF),
    onPrimary = Color(0xFFFFFFFF),
    secondary = Color(0xFF578FFF),
    onSecondary = Color(0xFFFFFFFF),
    tertiary = Color(0xFFFF8249),
    onTertiary = Color(0xFF010102),
    secondaryContainer = Color(0xFF0C7542),
    onSecondaryContainer = Color(0xFFB0FFD9),
    tertiaryContainer = Color(0xFF993D0F),
    onTertiaryContainer = Color(0xFFFFD3BD),
    background = Color(0xFF010102),
    onBackground = Color(0xFFFCFCFF),
    surface = Color(0xFF17171A),
    onSurface = Color(0xFFE9E9EC),
    surfaceVariant = Color(0xFF17171A),
    onSurfaceVariant = Color(0xFFB7B7BB),
    surfaceTint = Color(0xFF578FFF),
    outline = Color(0xFF636368),
    outlineVariant = Color(0xFF3A3A3D),
    error = Color(0xFFD93E36),
    onError = Color(0xFFFFFFFF),
    scrim = Color(0x99000000)
)

private val GenUiTypography = Typography(
    displayLarge = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.W700,
        fontSize = 34.sp,
        lineHeight = 43.sp,
        letterSpacing = 0.sp
    ),
    displayMedium = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.W600,
        fontSize = 30.sp,
        letterSpacing = 0.sp
    ),
    displaySmall = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.W600,
        fontSize = 26.sp,
        letterSpacing = 0.sp
    ),
    headlineLarge = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.W700,
        fontSize = 21.sp,
        letterSpacing = 0.sp
    ),
    headlineMedium = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.W600,
        fontSize = 21.sp,
        letterSpacing = 0.sp
    ),
    headlineSmall = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.W600,
        fontSize = 19.sp,
        letterSpacing = 0.sp
    ),
    titleLarge = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.W700,
        fontSize = 19.sp,
        letterSpacing = 0.sp
    ),
    titleMedium = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.W600,
        fontSize = 17.sp,
        letterSpacing = 0.sp
    ),
    titleSmall = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.W600,
        fontSize = 15.sp,
        letterSpacing = 0.sp
    ),
    bodyLarge = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.W400,
        fontSize = 17.sp,
        letterSpacing = 0.sp
    ),
    bodyMedium = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.W400,
        fontSize = 15.sp,
        letterSpacing = 0.sp
    ),
    bodySmall = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.W400,
        fontSize = 14.sp,
        letterSpacing = 0.sp
    ),
    labelLarge = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.W600,
        fontSize = 17.sp,
        letterSpacing = 0.sp
    ),
    labelMedium = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.W600,
        fontSize = 13.sp,
        letterSpacing = 0.sp
    ),
    labelSmall = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.W400,
        fontSize = 11.sp,
        letterSpacing = 0.sp
    )
)

private val GenUiShapes = Shapes(
    small = RoundedCornerShape(GenUiTokens.RadiusMd),
    medium = RoundedCornerShape(GenUiTokens.RadiusLg),
    large = RoundedCornerShape(GenUiTokens.RadiusXl)
)

@Composable
fun genUiBackgroundBrush(): Brush {
    val dark = isSystemInDarkTheme()
    val scheme = MaterialTheme.colorScheme
    val top = scheme.surfaceContainerLowest.copy(alpha = if (dark) 0.30f else 0.36f)
    val midPrimary = lerp(
        scheme.surfaceContainerLow,
        scheme.primaryContainer,
        if (dark) 0.24f else 0.18f
    )
    val midAccent = lerp(
        scheme.surfaceContainer,
        scheme.tertiaryContainer,
        if (dark) 0.22f else 0.16f
    )
    return Brush.verticalGradient(
        colors = listOf(
            top,
            midPrimary.copy(alpha = if (dark) 0.44f else 0.52f),
            midAccent.copy(alpha = if (dark) 0.40f else 0.48f),
            top
        )
    )
}

@Composable
fun genUiCardContainerColor(tone: GenUiCardTone = GenUiCardTone.Neutral): Color {
    val dark = isSystemInDarkTheme()
    val scheme = MaterialTheme.colorScheme
    val tokenColor = when (tone) {
        GenUiCardTone.Neutral -> scheme.surfaceContainerLow
        GenUiCardTone.Primary -> scheme.primaryContainer
        GenUiCardTone.Positive -> scheme.secondaryContainer
        GenUiCardTone.Warning -> scheme.tertiaryContainer
        GenUiCardTone.Error -> scheme.errorContainer
    }
    val blended = lerp(
        tokenColor,
        scheme.surface,
        if (dark) 0.10f else 0.06f
    )
    val alpha = when (tone) {
        GenUiCardTone.Neutral -> if (dark) 0.58f else 0.68f
        GenUiCardTone.Primary -> if (dark) 0.62f else 0.74f
        GenUiCardTone.Positive -> if (dark) 0.60f else 0.72f
        GenUiCardTone.Warning -> if (dark) 0.60f else 0.72f
        GenUiCardTone.Error -> if (dark) 0.62f else 0.74f
    }
    return blended.copy(alpha = alpha)
}

@Composable
fun genUiCardColors(tone: GenUiCardTone = GenUiCardTone.Neutral): CardColors {
    return CardDefaults.cardColors(containerColor = genUiCardContainerColor(tone))
}

@Composable
fun genUiCardBorderColor(): Color {
    val dark = isSystemInDarkTheme()
    return MaterialTheme.colorScheme.outlineVariant.copy(alpha = if (dark) 0.82f else 0.92f)
}

@Composable
fun GenUiCraftTheme(content: @Composable () -> Unit) {
    val context = LocalContext.current
    val darkTheme = isSystemInDarkTheme()
    val baseScheme = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
    } else if (darkTheme) {
        DarkScheme
    } else {
        LightScheme
    }
    val colorScheme = baseScheme.copy(
        // Keep text/readability stable on translucent windows in Samsung One UI by
        // avoiding semi-transparent text colors that can produce pale text bounding boxes.
        onSurfaceVariant = baseScheme.onSurfaceVariant.copy(alpha = 1f),
        outline = baseScheme.outline.copy(alpha = 1f),
        outlineVariant = baseScheme.outlineVariant.copy(alpha = 1f)
    )

    MaterialTheme(
        colorScheme = colorScheme,
        typography = GenUiTypography,
        shapes = GenUiShapes,
        content = content
    )
}
