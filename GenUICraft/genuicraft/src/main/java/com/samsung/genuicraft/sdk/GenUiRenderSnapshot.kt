package com.samsung.genuicraft.sdk

/**
 * A native rendering revision for one generation attempt. Previews are provisional, not a
 * conversion success. Hosts must discard them on cancellation/failure and persist only the final
 * conversion result. [elapsedMs] measures conversion-to-publication, not first screen paint.
 */
data class GenUiRenderSnapshot(
    val surfaceKey: String,
    val attempt: Int,
    val revision: Long,
    val document: GenUiDocument,
    val elapsedMs: Long,
    val readyComponentCount: Int,
    val isFinal: Boolean = false,
)
