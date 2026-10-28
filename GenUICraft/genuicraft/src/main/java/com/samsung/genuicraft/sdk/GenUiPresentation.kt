package com.samsung.genuicraft.sdk

import com.google.gson.JsonPrimitive
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiCanonicalGraph
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiWireCodec
import com.samsung.genuicraft.sdk.internal.pipeline.EntityListPresentation
import com.samsung.genuicraft.sdk.internal.pipeline.GenUiIrCodec

/** Lossless presentation adaptation, separate from raw-model validation and DSL recovery. */
internal object GenUiPresentation {
    data class Result(val document: GenUiDocument, val convertedLists: Int = 0) {
        val diagnostics: List<String>
            get() = if (convertedLists == 0) emptyList() else listOf(
                "SDK presentation: organized $convertedLists entity list(s) into cards; " +
                    "retained the supplied descriptions and citations. Raw model output is unchanged.",
            )
    }

    fun prepare(document: GenUiDocument): Result = runCatching {
        val graph = GenUiIrCodec.decode(JsonPrimitive(document.a2uiJson.ifBlank { document.express })).canonicalGraph
        val adapted = EntityListPresentation.normalize(graph)
        if (adapted.convertedLists == 0) return Result(document)
        require(A2uiCanonicalGraph.validate(adapted.graph).isValid) { "Invalid presentation projection." }
        Result(
            document.copy(
                express = A2uiExpressCodec.encode(adapted.graph, shortenIds = false),
                a2uiJson = A2uiWireCodec.encode(adapted.graph, shortenIds = false).toString(),
            ),
            adapted.convertedLists,
        )
    }.getOrElse { Result(document) } // Presentation must not turn an accepted answer into a failure.
}
