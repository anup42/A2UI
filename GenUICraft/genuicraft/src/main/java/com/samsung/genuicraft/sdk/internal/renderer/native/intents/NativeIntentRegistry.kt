package com.samsung.genuicraft.sdk.internal.renderer.native.intents

import androidx.compose.runtime.Composable
import com.samsung.genuicraft.sdk.internal.renderer.native.intents.flight.NativeFlightIntentModule
import com.samsung.genuicraft.sdk.internal.renderer.native.intents.weather.NativeWeatherIntentModule

internal object NativeIntentRegistry {
    private val modules: List<NativeIntentModule> = listOf(
        NativeFlightIntentModule,
        NativeWeatherIntentModule
    )

    fun resolve(
        header: List<String>,
        body: List<List<String>>
    ): NativeIntentRenderModel? {
        for (module in modules) {
            val model = module.resolve(header, body)
            if (model != null) {
                return model
            }
        }
        return null
    }

    @Composable
    fun render(
        model: NativeIntentRenderModel
    ): Boolean {
        for (module in modules) {
            if (module.render(model)) {
                return true
            }
        }
        return false
    }
}
