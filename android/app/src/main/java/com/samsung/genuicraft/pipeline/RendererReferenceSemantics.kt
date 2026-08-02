package com.samsung.genuicraft.pipeline

import com.google.gson.JsonObject

/** Active renderer reference inventory; legacy name remains an offline alias. */
internal object RendererReferenceSemantics {
    fun references(element: JsonObject): List<FlatSpecReferenceSemantics.Reference> =
        FlatSpecReferenceSemantics.references(element)
}
