package com.samsung.genuicraft.sdk.internal.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CardColors
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Renderer-owned design tokens, independent of the host application's theme. */
internal object GenUiTokens {
    val RadiusSm = 8.dp
    val RadiusMd = 12.dp
    val RadiusLg = 22.dp
    val RadiusXl = 26.dp
    val RadiusPill = 100.dp

    val BorderSm = 0.5.dp
    val BorderMd = 1.dp
    val BorderLg = 2.dp

    val ElevationSm = 0.dp
    val ElevationMd = 0.dp
    val ElevationLg = 0.dp
    val ElevationXl = 0.dp
}

internal enum class GenUiCardTone {
    Neutral,
    Primary,
    Positive,
    Warning,
    Error,
}

private val LightColors = lightColorScheme(
    primary = Color(0xFF245BBF),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFE8F0FF),
    onPrimaryContainer = Color(0xFF183F83),
    secondary = Color(0xFF46607C),
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFEAF0F6),
    onSecondaryContainer = Color(0xFF26384D),
    tertiary = Color(0xFF8B5A19),
    onTertiary = Color.White,
    tertiaryContainer = Color(0xFFFFF2DB),
    onTertiaryContainer = Color(0xFF67420D),
    background = Color(0xFFF5F7FB),
    onBackground = Color(0xFF172033),
    surface = Color.White,
    onSurface = Color(0xFF172033),
    surfaceVariant = Color(0xFFEAF0F6),
    onSurfaceVariant = Color(0xFF526078),
    surfaceDim = Color(0xFFE5EAF2),
    surfaceBright = Color.White,
    surfaceContainerLowest = Color.White,
    surfaceContainerLow = Color(0xFFF8FAFD),
    surfaceContainer = Color(0xFFF0F3F8),
    surfaceContainerHigh = Color(0xFFEAF0F6),
    surfaceContainerHighest = Color(0xFFE2E8F0),
    outline = Color(0xFF8592A7),
    outlineVariant = Color(0xFFDDE4EF),
)

private val DarkColors = darkColorScheme(
    primary = Color(0xFFA8C7FF),
    onPrimary = Color(0xFF12376E),
    primaryContainer = Color(0xFF233D63),
    onPrimaryContainer = Color(0xFFD8E7FF),
    secondary = Color(0xFFB6C9E1),
    onSecondary = Color(0xFF183047),
    secondaryContainer = Color(0xFF293B50),
    onSecondaryContainer = Color(0xFFDEEBFA),
    tertiary = Color(0xFFF1C17A),
    onTertiary = Color(0xFF472D0B),
    tertiaryContainer = Color(0xFF513C20),
    onTertiaryContainer = Color(0xFFFFE1B0),
    background = Color(0xFF101620),
    onBackground = Color(0xFFEAF0FA),
    surface = Color(0xFF192331),
    onSurface = Color(0xFFEAF0FA),
    surfaceVariant = Color(0xFF2B394D),
    onSurfaceVariant = Color(0xFFB7C4D8),
    surfaceDim = Color(0xFF101620),
    surfaceBright = Color(0xFF303D51),
    surfaceContainerLowest = Color(0xFF101620),
    surfaceContainerLow = Color(0xFF1B2635),
    surfaceContainer = Color(0xFF202D3F),
    surfaceContainerHigh = Color(0xFF26354A),
    surfaceContainerHighest = Color(0xFF2F4057),
    outline = Color(0xFF8392A9),
    outlineVariant = Color(0xFF37475E),
)

private val RendererTypography = Typography(
    headlineMedium = TextStyle(fontSize = 26.sp, lineHeight = 33.sp, fontWeight = FontWeight.SemiBold),
    titleLarge = TextStyle(fontSize = 22.sp, lineHeight = 29.sp, fontWeight = FontWeight.SemiBold),
    titleMedium = TextStyle(fontSize = 18.sp, lineHeight = 25.sp, fontWeight = FontWeight.SemiBold),
    titleSmall = TextStyle(fontSize = 16.sp, lineHeight = 23.sp, fontWeight = FontWeight.SemiBold),
    bodyLarge = TextStyle(fontSize = 16.sp, lineHeight = 25.sp),
    bodyMedium = TextStyle(fontSize = 15.sp, lineHeight = 23.sp),
    bodySmall = TextStyle(fontSize = 13.sp, lineHeight = 20.sp),
    labelLarge = TextStyle(fontSize = 14.sp, lineHeight = 20.sp, fontWeight = FontWeight.Medium),
    labelMedium = TextStyle(fontSize = 12.sp, lineHeight = 18.sp, fontWeight = FontWeight.Medium),
    labelSmall = TextStyle(fontSize = 11.sp, lineHeight = 16.sp, fontWeight = FontWeight.Medium),
)

@Composable
internal fun GenUiRendererTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = if (isSystemInDarkTheme()) DarkColors else LightColors,
        typography = RendererTypography,
        shapes = Shapes(
            small = RoundedCornerShape(GenUiTokens.RadiusMd),
            medium = RoundedCornerShape(GenUiTokens.RadiusLg),
            large = RoundedCornerShape(GenUiTokens.RadiusXl),
        ),
        content = content,
    )
}

@Composable
internal fun genUiCardContainerColor(tone: GenUiCardTone = GenUiCardTone.Neutral): Color {
    val dark = isSystemInDarkTheme()
    val scheme = MaterialTheme.colorScheme
    val tokenColor = when (tone) {
        GenUiCardTone.Neutral -> scheme.surface
        GenUiCardTone.Primary -> scheme.primaryContainer
        GenUiCardTone.Positive -> scheme.secondaryContainer
        GenUiCardTone.Warning -> scheme.tertiaryContainer
        GenUiCardTone.Error -> scheme.errorContainer
    }
    return if (tone == GenUiCardTone.Neutral) tokenColor
        else lerp(scheme.surface, tokenColor, if (dark) 0.72f else 0.62f)
}

@Composable
internal fun genUiCardColors(tone: GenUiCardTone = GenUiCardTone.Neutral): CardColors =
    CardDefaults.cardColors(containerColor = genUiCardContainerColor(tone))

@Composable
internal fun genUiCardBorderColor(): Color =
    MaterialTheme.colorScheme.outlineVariant.copy(alpha = if (isSystemInDarkTheme()) 0.72f else 0.62f)

@Composable
internal fun genUiMediaFrameBorderColor(): Color =
    if (isSystemInDarkTheme()) Color(0xFF636368) else Color(0xFFB7B7BB)

@Composable
internal fun genUiTableContainerColor(): Color {
    val scheme = MaterialTheme.colorScheme
    return lerp(
        scheme.surfaceContainerLow,
        scheme.surfaceContainerHighest,
        if (isSystemInDarkTheme()) 0.24f else 0.40f,
    )
}
