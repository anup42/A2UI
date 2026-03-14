package com.samsung.genuicraft.renderer.native.intents.weather

import androidx.compose.runtime.Composable
import com.samsung.genuicraft.renderer.native.NativeTextFormatter
import com.samsung.genuicraft.renderer.native.WeatherRow
import com.samsung.genuicraft.renderer.native.intents.NativeIntentModule
import com.samsung.genuicraft.renderer.native.intents.NativeIntentRenderModel

internal object NativeWeatherIntentModule : NativeIntentModule {
    private data class WeatherIntentModel(
        val rows: List<WeatherRow>
    ) : NativeIntentRenderModel

    override val id: String = "weather"

    override fun resolve(
        header: List<String>,
        body: List<List<String>>
    ): NativeIntentRenderModel? {
        val rows = NativeWeatherSemantics.buildWeatherRows(header, body) ?: return null
        return WeatherIntentModel(rows)
    }

    @Composable
    override fun render(
        model: NativeIntentRenderModel
    ): Boolean {
        val weatherModel = model as? WeatherIntentModel ?: return false
        NativeWeatherUiRenderer.RenderWeatherRows(
            rows = weatherModel.rows,
            sanitizeDisplayText = NativeTextFormatter::sanitizeDisplayText,
            weatherTemperatureText = NativeWeatherSemantics::weatherTemperatureText,
            orderWeatherRows = NativeWeatherSemantics::orderWeatherRows,
            isTodayWeatherRow = NativeWeatherSemantics::isTodayWeatherRow,
            weatherConditionIcon = { condition, size ->
                NativeWeatherUiRenderer.WeatherConditionIcon(
                    condition = condition,
                    size = size,
                    sanitizeDisplayText = NativeTextFormatter::sanitizeDisplayText
                )
            }
        )
        return true
    }
}
