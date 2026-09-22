package com.samsung.genuicraft

import java.util.Locale

internal enum class GemmaModelSource(val preferenceValue: String) {
    MANAGED_DOWNLOAD("managed_download"),
    LOCAL_FILE("local_file");

    companion object {
        fun fromPreference(value: String?): GemmaModelSource =
            entries.firstOrNull { it.preferenceValue == value } ?: MANAGED_DOWNLOAD
    }
}

internal data class DemoActionAvailability(
    val convertEnabled: Boolean,
    val renderEnabled: Boolean,
)

internal fun demoActionAvailability(
    working: Boolean,
    modelSource: GemmaModelSource,
    managedModelReady: Boolean,
): DemoActionAvailability = DemoActionAvailability(
    convertEnabled = !working && (
        modelSource == GemmaModelSource.LOCAL_FILE || managedModelReady
    ),
    renderEnabled = !working,
)

internal fun formatDownloadBytes(bytes: Long): String {
    if (bytes < 0L) return "unknown"
    val gigabytes = bytes.toDouble() / 1_000_000_000.0
    return if (gigabytes >= 1.0) {
        String.format(Locale.US, "%.2f GB", gigabytes)
    } else {
        String.format(Locale.US, "%.1f MB", bytes.toDouble() / 1_000_000.0)
    }
}
