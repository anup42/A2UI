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
internal const val PREFERENCE_TRAINED_E2B_PREVIOUS_R64_MODEL_PATH = "trained_e2b_previous_r64_model_path"
internal const val PREFERENCE_E2B_MTP_ENABLED = "e2b_mtp_enabled"
internal const val PREFERENCE_GEMMA270_MODEL_PATH = "gemma3_270m_model_path"
internal const val PREFERENCE_GEMMA270_ACCELERATOR = "gemma3_270m_accelerator"

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
    ),
    TRAINED_E2B_PREVIOUS_R64(
        preferenceValue = "trained_e2b_previous_r64",
        profile = GenUiTrainedConverter.PROFILE,
    ),
    TRAINED_GEMMA3_270M(
        preferenceValue = "trained_gemma3_270m",
        profile = "trained_gemma3_270m",
    );

    val supportsMtp: Boolean get() = this != TRAINED_GEMMA3_270M
    val isTrainedE2b: Boolean get() =
        this == TRAINED_E2B_V10_W4 || this == TRAINED_E2B_PREVIOUS_R64

    val displayName: String get() = when (this) {
        OFFICIAL_E2B -> "Official E2B"
        TRAINED_E2B_V10_W4 -> "Trained E2B · Latest"
        TRAINED_E2B_PREVIOUS_R64 -> "Trained E2B · Previous R64"
        TRAINED_GEMMA3_270M -> "Gemma 3 270M"
    }

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

internal fun gemma270ModelReadiness(modelPath: String): LocalModelReadiness {
    val file = java.io.File(modelPath.trim())
    val correctSelection = com.samsung.genuicraft.inference.OnDeviceModelCatalog.entryForModelPath(
        file.absolutePath)?.id == "gemma3_270m_a2ui_express_int8"
    if (!correctSelection) return LocalModelReadiness(false,
        "Use the Gemma 270M A2UI Express package named gemma-3-270m-a2ui-express-int8.litertlm.")
    val usable = file.isAbsolute &&
        file.isFile && file.canRead() && file.length() >= 200_000_000L
    return LocalModelReadiness(usable, if (usable) "Gemma 270M package found · ready for inference"
        else "Select a readable Gemma 270M .litertlm package (at least 200 MB).")
}

internal fun sdkDemoActionAvailability(
    working: Boolean,
    e2bModelChoice: E2bModelChoice,
    officialModelSource: GemmaModelSource,
    managedOfficialModelReady: Boolean,
    trainedModelReady: Boolean,
    gemma270ModelReady: Boolean = false,
): DemoActionAvailability {
    if (e2bModelChoice == E2bModelChoice.TRAINED_GEMMA3_270M) {
        return DemoActionAvailability(!working && gemma270ModelReady, !working)
    }
    if (e2bModelChoice.isTrainedE2b) {
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
    gemma270Accelerator: String = "CPU",
): String = if (e2bModelChoice == E2bModelChoice.TRAINED_GEMMA3_270M) {
    "gemma:profile=${e2bModelChoice.profile}:path=${modelPath.trim()}:metrics=$enableMetrics:" +
        "mtp=false:precision=MODEL_DEFAULT:accelerator=${gemma270Accelerator.uppercase(java.util.Locale.ROOT)}"
} else {
    "gemma:profile=${e2bModelChoice.profile}:path=${modelPath.trim()}:metrics=$enableMetrics:mtp=$enableMtp:precision=$gpuPrecision"
}

/** Fully releases an old native provider before a replacement can be constructed. */
internal suspend fun closeProviderBeforeReplacement(provider: GenUiProvider?) {
    withContext(NonCancellable) {
        provider?.closeAndAwait()
    }
    // Do not continue into replacement construction when cancellation arrived during cleanup.
    coroutineContext.ensureActive()
}
