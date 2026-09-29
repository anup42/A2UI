package com.samsung.genuicraft

import com.samsung.genuicraft.sdk.GenUiTrainedConverter
import com.samsung.genuicraft.sdk.GenUiProvider
import com.samsung.genuicraft.sdk.GenUiModelProfiles
import com.samsung.genuicraft.sdk.provider.Gemma4Config
import com.samsung.genuicraft.sdk.provider.Gemma4GpuPrecision
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

internal fun trainedE2bW4Readiness(modelPath: String): LocalModelReadiness =
    selectTrainedE2bModel(modelPath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP32).readiness

internal fun sdkDemoActionAvailability(
    working: Boolean,
    e2bModelChoice: E2bModelChoice,
    officialModelSource: GemmaModelSource,
    managedOfficialModelReady: Boolean,
    trainedModelReady: Boolean,
): DemoActionAvailability {
    if (e2bModelChoice == E2bModelChoice.TRAINED_E2B_V10_W4) {
        return DemoActionAvailability(
            convertEnabled = !working && trainedModelReady,
            renderEnabled = !working,
        )
    }
    return demoActionAvailability(
        working = working,
        modelSource = officialModelSource,
        managedModelReady = managedOfficialModelReady,
    )
}

internal fun trainedE2bW4Config(
    modelPath: String,
    enableMetrics: Boolean,
    enableMtp: Boolean = false,
    gpuPrecision: Gemma4GpuPrecision = Gemma4GpuPrecision.FP32,
): Gemma4Config = GenUiModelProfiles.trainedE2b(
    modelPath = modelPath,
    enableMtp = enableMtp,
    enableMetrics = enableMetrics,
    gpuPrecision = gpuPrecision,
)

internal fun sdkDemoProviderKey(
    e2bModelChoice: E2bModelChoice,
    modelPath: String,
    enableMetrics: Boolean,
    enableMtp: Boolean = e2bModelChoice == E2bModelChoice.OFFICIAL_E2B,
    gpuPrecision: Gemma4GpuPrecision = Gemma4GpuPrecision.FP32,
): String =
    "gemma:profile=${e2bModelChoice.profile}:path=${modelPath.trim()}:metrics=$enableMetrics:mtp=$enableMtp:precision=$gpuPrecision"

/** Fully releases an old native provider before a replacement can be constructed. */
internal suspend fun closeProviderBeforeReplacement(provider: GenUiProvider?) {
    withContext(NonCancellable) {
        provider?.closeAndAwait()
    }
    // Do not continue into replacement construction when cancellation arrived during cleanup.
    coroutineContext.ensureActive()
}
