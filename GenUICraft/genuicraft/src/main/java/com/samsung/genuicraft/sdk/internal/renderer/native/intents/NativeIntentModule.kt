package com.samsung.genuicraft.sdk.internal.renderer.native.intents

import androidx.compose.runtime.Composable

internal interface NativeIntentModule {
    val id: String

    fun resolve(
        header: List<String>,
        body: List<List<String>>
    ): NativeIntentRenderModel?

    @Composable
    fun render(
        model: NativeIntentRenderModel
    ): Boolean
}
