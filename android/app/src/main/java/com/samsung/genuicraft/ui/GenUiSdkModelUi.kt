package com.samsung.genuicraft

import com.samsung.genuicraft.sdk.GenUiTrainedConverter
import com.samsung.genuicraft.sdk.GenUiProvider
import com.samsung.genuicraft.sdk.provider.Gemma4Config
import java.io.File
import java.util.Locale
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

internal const val PREFERENCE_E2B_MODEL_CHOICE = "e2b_model_choice"
internal const val PREFERENCE_OFFICIAL_E2B_MODEL_PATH = "model_path"
internal const val PREFERENCE_TRAINED_E2B_W4_MODEL_PATH = "trained_e2b_v10_w4_model_path"
internal const val PREFERENCE_E2B_MTP_ENABLED = "e2b_mtp_enabled"

internal enum class E2bModelChoice(
    val preferenceValue: String,
    val profile: String,
) {
    OFFICIAL_E2B(
        preferenceValue = "official_e2b",
        profile = "official_e2b",
    ),
    TRAINED_E2B_V10_W4(
        preferenceValue = "trained_e2b_v10_w4",
        profile = GenUiTrainedConverter.PROFILE,
    );

    companion object {
        fun fromPreference(value: String?): E2bModelChoice =
            entries.firstOrNull { it.preferenceValue == value } ?: OFFICIAL_E2B
    }
}

internal data class LocalModelReadiness(
    val usable: Boolean,
    val message: String,
)

internal fun trainedE2bW4Readiness(modelPath: String): LocalModelReadiness {
    val normalizedPath = modelPath.trim()
    if (normalizedPath.isEmpty()) {
        return LocalModelReadiness(false, "Trained E2B model path is empty.")
    }
    val modelFile = File(normalizedPath)
    if (!modelFile.isAbsolute) {
        return LocalModelReadiness(false, "Trained E2B model path must be absolute.")
    }
    if (!modelFile.name.lowercase(Locale.US).endsWith(".litertlm")) {
        return LocalModelReadiness(false, "Trained E2B model must be a .litertlm file.")
    }
    if (!modelFile.isFile) {
        return LocalModelReadiness(false, "Trained E2B model not found: $normalizedPath")
    }
    if (!modelFile.canRead()) {
        return LocalModelReadiness(false, "Trained E2B model is not readable: $normalizedPath")
    }
    if (modelFile.length() <= 0L) {
        return LocalModelReadiness(false, "Trained E2B model is empty: $normalizedPath")
    }
    return LocalModelReadiness(
        usable = true,
        message = "Ready · ${formatDownloadBytes(modelFile.length())} · GPU",
    )
}

internal fun sdkDemoActionAvailability(
    working: Boolean,
    useGemma: Boolean,
    e2bModelChoice: E2bModelChoice,
    officialModelSource: GemmaModelSource,
    managedOfficialModelReady: Boolean,
    trainedModelReady: Boolean,
): DemoActionAvailability {
    if (useGemma && e2bModelChoice == E2bModelChoice.TRAINED_E2B_V10_W4) {
        return DemoActionAvailability(
            convertEnabled = !working && trainedModelReady,
            renderEnabled = !working,
        )
    }
    return demoActionAvailability(
        working = working,
        useGemma = useGemma,
        modelSource = officialModelSource,
        managedModelReady = managedOfficialModelReady,
    )
}

internal fun trainedE2bW4Config(
    modelPath: String,
    enableMetrics: Boolean,
    enableMtp: Boolean = false,
): Gemma4Config = Gemma4Config(
    modelPath = modelPath,
    accelerator = "GPU",
    maxContextTokens = 8_192,
    maxOutputTokens = 2_048,
    enableThinking = false,
    thinkingTokenBudget = 0,
    enableSpeculativeDecoding = enableMtp,
    enableMetrics = enableMetrics,
)

internal fun sdkDemoProviderKey(
    useGemma: Boolean,
    e2bModelChoice: E2bModelChoice,
    modelPath: String,
    enableMetrics: Boolean,
    enableMtp: Boolean = e2bModelChoice == E2bModelChoice.OFFICIAL_E2B,
): String = if (useGemma) {
    "gemma:profile=${e2bModelChoice.profile}:path=${modelPath.trim()}:metrics=$enableMetrics:mtp=$enableMtp"
} else {
    "gauss:profile=gauss30b:metrics=$enableMetrics"
}

/** Fully releases an old native provider before a replacement can be constructed. */
internal suspend fun closeProviderBeforeReplacement(provider: GenUiProvider?) {
    withContext(NonCancellable) {
        provider?.closeAndAwait()
    }
    // Do not continue into replacement construction when cancellation arrived during cleanup.
    coroutineContext.ensureActive()
}
