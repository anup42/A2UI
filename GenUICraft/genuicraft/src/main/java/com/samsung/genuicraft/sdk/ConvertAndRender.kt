package com.samsung.genuicraft.sdk

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** Caller owns its coroutine/lifecycle and loading/error UI. Rendering occurs only after validation. */
suspend fun GenUiView.convertAndRender(converter: GenUiConverter, request: GenUiRequest): GenUiConversionResult {
    val result = converter.convert(request)
    if (result is GenUiConversionResult.Success) withContext(Dispatchers.Main.immediate) { render(result.document) }
    return result
}
