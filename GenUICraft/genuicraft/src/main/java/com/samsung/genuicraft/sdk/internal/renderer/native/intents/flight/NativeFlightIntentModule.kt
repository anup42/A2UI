package com.samsung.genuicraft.sdk.internal.renderer.native.intents.flight

import androidx.compose.runtime.Composable
import com.samsung.genuicraft.sdk.internal.renderer.native.FlightRow
import com.samsung.genuicraft.sdk.internal.renderer.native.intents.NativeIntentModule
import com.samsung.genuicraft.sdk.internal.renderer.native.intents.NativeIntentRenderModel

internal object NativeFlightIntentModule : NativeIntentModule {
    private data class FlightIntentModel(
        val rows: List<FlightRow>
    ) : NativeIntentRenderModel

    override val id: String = "flight"

    override fun resolve(
        header: List<String>,
        body: List<List<String>>
    ): NativeIntentRenderModel? {
        val rows = NativeFlightSemantics.buildFlightRows(header, body) ?: return null
        return FlightIntentModel(rows)
    }

    @Composable
    override fun render(
        model: NativeIntentRenderModel
    ): Boolean {
        val flightModel = model as? FlightIntentModel ?: return false
        NativeFlightUiRenderer.RenderFlightRows(flightModel.rows)
        return true
    }
}
