package com.samsung.genuicraft.sdk.internal.renderer.flat.expr

/** Opaque until display, so components that resolve a value twice cannot evaluate literal text. */
internal data class FlatLiteralText(val value: String) {
    override fun toString(): String = value
}
