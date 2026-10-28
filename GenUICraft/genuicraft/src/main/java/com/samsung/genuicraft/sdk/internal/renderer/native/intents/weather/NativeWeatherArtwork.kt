package com.samsung.genuicraft.sdk.internal.renderer.native.intents.weather

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import java.util.Locale
import kotlin.math.cos
import kotlin.math.sin

internal enum class WeatherArtworkKind { SUN, PARTLY_CLOUDY, CLOUD, RAIN, STORM, SNOW, SLEET, WIND, FOG, NIGHT, DROP, TEMPERATURE }

/** Local vector artwork: no network, font glyph, image decoding or asset-pack dependency. */
internal object NativeWeatherArtwork {
    fun kindFor(condition: String?, timeContext: String? = null): WeatherArtworkKind {
        val text = condition.orEmpty().lowercase(Locale.ROOT)
        val snow = listOf("snow", "sleet", "flurr", "ice", "freez", "hail", "blizzard").any(text::contains)
        val rain = listOf("rain", "drizzle").any(text::contains) || (text.contains("shower") && !snow)
        val airborneDust = listOf("dust", "sandstorm", "sand storm").any(text::contains)
        return when {
            listOf("thunder", "lightning").any(text::contains) ||
                (text.contains("storm") && !snow && !airborneDust) -> WeatherArtworkKind.STORM
            airborneDust -> WeatherArtworkKind.FOG
            rain && snow -> WeatherArtworkKind.SLEET
            snow -> WeatherArtworkKind.SNOW
            rain -> WeatherArtworkKind.RAIN
            listOf("fog", "mist", "haze", "smog").any(text::contains) -> WeatherArtworkKind.FOG
            listOf("wind", "breeze", "gust").any(text::contains) -> WeatherArtworkKind.WIND
            (text.contains("sun") && text.contains("cloud")) ||
                text.contains("partly") || text.contains("mostly sunny") ||
                text.contains("sunny interval") -> WeatherArtworkKind.PARTLY_CLOUDY
            text.contains("cloud") || text.contains("overcast") -> WeatherArtworkKind.CLOUD
            text.contains("sun") -> WeatherArtworkKind.SUN
            text.contains("clear") || text.contains("fair") ->
                if (Regex("""\b(night|tonight|overnight|evening)\b""", RegexOption.IGNORE_CASE)
                        .containsMatchIn(listOfNotNull(condition, timeContext).joinToString(" "))) {
                    WeatherArtworkKind.NIGHT
                } else WeatherArtworkKind.SUN
            else -> WeatherArtworkKind.TEMPERATURE
        }
    }

    @Composable
    fun Symbol(kind: WeatherArtworkKind, modifier: Modifier, description: String? = null) {
        Canvas(modifier.then(if (description == null) Modifier else Modifier.semantics { contentDescription = description })) {
            scale(size.width / 100f, size.height / 100f, pivot = Offset.Zero) {
                when (kind) {
                    WeatherArtworkKind.SUN -> sun(50f, 50f, 22f)
                    WeatherArtworkKind.PARTLY_CLOUDY -> { sun(35f, 34f, 17f); cloud(9f, 36f, 82f) }
                    WeatherArtworkKind.CLOUD -> { cloud(26f, 24f, 63f, muted = true); cloud(7f, 39f, 82f) }
                    WeatherArtworkKind.RAIN -> { cloud(8f, 20f, 84f); rain() }
                    WeatherArtworkKind.STORM -> {
                        cloud(8f, 19f, 84f, muted = true)
                        val bolt = Path().apply {
                            moveTo(53f, 49f); lineTo(35f, 77f); lineTo(48f, 77f)
                            lineTo(40f, 95f); lineTo(68f, 64f); lineTo(54f, 64f); close()
                        }
                        drawPath(bolt, Brush.verticalGradient(listOf(Color(0xFFFFDB54), Color(0xFFF49B21)), 50f, 96f))
                    }
                    WeatherArtworkKind.SNOW -> { cloud(8f, 18f, 84f); snowflake(32f, 80f, 9f); snowflake(68f, 80f, 9f) }
                    WeatherArtworkKind.SLEET -> { cloud(8f, 18f, 84f); drop(31f, 78f, 11f); snowflake(68f, 80f, 9f) }
                    WeatherArtworkKind.WIND -> wind(false)
                    WeatherArtworkKind.FOG -> { cloud(13f, 18f, 73f, muted = true); wind(true) }
                    WeatherArtworkKind.NIGHT -> {
                        val moon = Path().apply {
                            moveTo(67f, 13f); cubicTo(21f, 7f, 11f, 74f, 51f, 87f)
                            cubicTo(69f, 94f, 89f, 82f, 93f, 67f)
                            cubicTo(57f, 77f, 41f, 37f, 67f, 13f); close()
                        }
                        drawPath(moon, Brush.linearGradient(listOf(Color(0xFFFFE6A1), Color(0xFFFFBB4B))))
                        drawCircle(Color(0xFF83C4FF), 4f, Offset(81f, 25f))
                        drawCircle(Color(0xFF83C4FF), 2.5f, Offset(69f, 43f))
                    }
                    WeatherArtworkKind.DROP -> drop(50f, 53f, 31f)
                    WeatherArtworkKind.TEMPERATURE -> {
                        drawRoundRect(Color(0xFF5FAAF0), Offset(39f, 12f), Size(22f, 62f), CornerRadius(11f))
                        drawRoundRect(Color(0xFFE8F6FF), Offset(44f, 17f), Size(12f, 57f), CornerRadius(6f))
                        drawCircle(Color(0xFF5FAAF0), 21f, Offset(50f, 76f))
                        drawRoundRect(Color(0xFFFF9650), Offset(46f, 40f), Size(8f, 38f), CornerRadius(4f))
                        drawCircle(Color(0xFFFF9650), 14f, Offset(50f, 76f))
                        listOf(28f, 40f, 52f).forEach { y -> drawLine(Color(0xFF5FAAF0), Offset(68f, y), Offset(79f, y), 4f, StrokeCap.Round) }
                    }
                }
            }
        }
    }

    private fun DrawScope.sun(x: Float, y: Float, radius: Float) {
        repeat(8) { index ->
            val angle = Math.PI * index / 4
            val start = Offset(x + cos(angle).toFloat() * (radius + 7f), y + sin(angle).toFloat() * (radius + 7f))
            val end = Offset(x + cos(angle).toFloat() * (radius + 15f), y + sin(angle).toFloat() * (radius + 15f))
            drawLine(Color(0xFFF6B331), start, end, 5f, StrokeCap.Round)
        }
        drawCircle(Brush.radialGradient(listOf(Color(0xFFFFE586), Color(0xFFFFB529)), center = Offset(x - radius / 3, y - radius / 3), radius = radius * 1.8f), radius, Offset(x, y))
        drawCircle(Color(0xFFFFCF59), radius, Offset(x, y), style = Stroke(1.5f))
    }

    private fun DrawScope.cloud(x: Float, y: Float, width: Float, muted: Boolean = false) {
        fun shape(dy: Float) = Path().apply {
            moveTo(x + width * .22f, y + width * .48f + dy)
            cubicTo(x - width * .04f, y + width * .48f + dy, x - width * .03f, y + width * .16f + dy, x + width * .22f, y + width * .18f + dy)
            cubicTo(x + width * .28f, y - width * .10f + dy, x + width * .64f, y - width * .04f + dy, x + width * .67f, y + width * .17f + dy)
            cubicTo(x + width * .99f, y + width * .08f + dy, x + width * 1.13f, y + width * .48f + dy, x + width * .82f, y + width * .48f + dy)
            close()
        }
        drawPath(shape(4f), Color(0xFF366EA4).copy(alpha = .14f))
        drawPath(shape(0f), Brush.verticalGradient(if (muted) listOf(Color(0xFFBED8ED), Color(0xFF77A9D0)) else listOf(Color.White, Color(0xFFC8E7FA)), y, y + width * .55f))
        drawPath(shape(0f), Color(0xFF78B2D6).copy(alpha = .8f), style = Stroke(1.8f))
    }

    private fun DrawScope.rain() {
        drop(28f, 76f, 9f); drop(50f, 85f, 9f); drop(73f, 76f, 9f)
    }

    private fun DrawScope.drop(x: Float, y: Float, radius: Float) {
        val path = Path().apply {
            moveTo(x, y - radius)
            cubicTo(x + radius * .35f, y - radius * .35f, x + radius, y + radius * .15f, x + radius * .65f, y + radius * .6f)
            cubicTo(x + radius * .4f, y + radius * 1.2f, x - radius * .7f, y + radius, x - radius * .7f, y + radius * .35f)
            cubicTo(x - radius * .8f, y - radius * .1f, x - radius * .3f, y - radius * .6f, x, y - radius); close()
        }
        drawPath(path, Brush.linearGradient(listOf(Color(0xFF60CAFF), Color(0xFF2568D8)), Offset(x - radius, y - radius), Offset(x + radius, y + radius)))
        drawCircle(Color.White.copy(alpha = .65f), radius * .13f, Offset(x - radius * .28f, y + radius * .2f))
    }

    private fun DrawScope.snowflake(x: Float, y: Float, radius: Float) {
        repeat(3) { index ->
            val angle = Math.PI * index / 3
            val dx = cos(angle).toFloat() * radius; val dy = sin(angle).toFloat() * radius
            drawLine(Color(0xFF40A5DB), Offset(x - dx, y - dy), Offset(x + dx, y + dy), 3f, StrokeCap.Round)
        }
    }

    private fun DrawScope.wind(fog: Boolean) {
        val base = if (fog) 66f else 36f
        repeat(3) { index ->
            val y = base + index * 12f
            drawLine(Color(0xFF65AAC6), Offset(15f + index * 6f, y), Offset(83f - index * 6f, y), 5f, StrokeCap.Round)
        }
        if (!fog) drawArc(Color(0xFF65AAC6), 190f, 260f, false, Offset(60f, 16f), Size(25f, 22f), style = Stroke(5f, cap = StrokeCap.Round))
    }
}
