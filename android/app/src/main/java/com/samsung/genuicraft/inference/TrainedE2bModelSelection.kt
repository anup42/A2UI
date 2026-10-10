package com.samsung.genuicraft

import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.provider.Gemma4GpuPrecision
import java.io.File
import java.util.Locale

/** The saved path names the original trained package. Corrected FP16 uses a prepared sibling. */
internal const val CORRECTED_FP16_MODEL_FILE_NAME = "model-fp16-corrected.litertlm"
private const val CORRECTED_FP16_POLICY = "genuicraft-fp16-rope-qdq-v1"
private const val CORRECTED_FP16_CONTEXT_TOKENS = 8_192
private const val LEGACY_SOURCE_SHA256 = "de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62"
private const val LEGACY_MODEL_SHA256 = "7f01bdf1c6ba9bdf658e57c75001fc35a42edad88238bc5dab0a5f7dcf3de373"
private const val NEW_SOURCE_SHA256 = "1fefbea8a1b6d526d33fe5fba6cf4b212f8c81d3226710972516b432419eacdc"
private const val NEW_MODEL_SHA256 = "26aa7eedbdfc4fe80321edc7a24aa1cfe85057739e03ffb1b7de7ac5bdb8fdb3"
private const val PINNED_SOURCE_BYTES = 2_588_147_712L
private val SHA256_HEX = Regex("[0-9a-f]{64}")

internal data class CorrectedFp16Profile(
    val originalFileNames: Set<String>,
    val sourceSha256: String,
    val sourceBytes: Long,
    val transformVersion: String,
    val modelSha256: String,
)

private val correctedFp16Profiles = listOf(
    CorrectedFp16Profile(
        originalFileNames = setOf(
            "gemma4_e2b_a2ui_mobile_r64_qat_compatible.litertlm",
            "gemma4_e2b_a2ui_mobile.litertlm",
            "e2b_v10_w4.litertlm",
        ),
        sourceSha256 = LEGACY_SOURCE_SHA256,
        sourceBytes = PINNED_SOURCE_BYTES,
        transformVersion = "pinned-r64-rope-target-mtp-8192-v1",
        modelSha256 = LEGACY_MODEL_SHA256,
    ),
    CorrectedFp16Profile(
        originalFileNames = setOf("e2b_qat_grpo_20261009.litertlm"),
        sourceSha256 = NEW_SOURCE_SHA256,
        sourceBytes = PINNED_SOURCE_BYTES,
        transformVersion = "e2b-rope-target-mtp-8192-v2",
        modelSha256 = NEW_MODEL_SHA256,
    ),
)

internal data class TrainedE2bModelSelection(
    val modelPath: String,
    val gpuPrecision: Gemma4GpuPrecision,
    val readiness: LocalModelReadiness,
    val mtpSupported: Boolean,
)

/** Cheap UI checks; provisioning must verify the original SHA outside the main thread. */
internal fun selectTrainedE2bModel(
    originalModelPath: String,
    precision: InferenceBackendSettings.TrainedE2bGpuPrecision,
): TrainedE2bModelSelection = selectTrainedE2bModel(
    originalModelPath, precision, correctedFp16Profiles,
)

/** Profile injection is kept inside the app module for small-file contract tests. */
internal fun selectTrainedE2bModel(
    originalModelPath: String,
    precision: InferenceBackendSettings.TrainedE2bGpuPrecision,
    profiles: List<CorrectedFp16Profile>,
): TrainedE2bModelSelection {
    val original = File(originalModelPath.trim())
    if (precision == InferenceBackendSettings.TrainedE2bGpuPrecision.FP32) {
        val readiness = basicModelReadiness(originalModelPath, "Trained E2B model")
        return TrainedE2bModelSelection(
            original.absolutePath,
            Gemma4GpuPrecision.FP32,
            readiness,
            mtpSupported = true,
        )
    }

    val sourceReadiness = basicModelReadiness(originalModelPath, "Original trained E2B model")
    if (!sourceReadiness.usable) {
        return TrainedE2bModelSelection(
            original.absolutePath,
            Gemma4GpuPrecision.FP16_CORRECTED,
            sourceReadiness,
            mtpSupported = false,
        )
    }
    val profile = profiles.firstOrNull { candidate ->
        candidate.originalFileNames.any { it.equals(original.name, ignoreCase = true) }
    }
    if (profile == null || !SHA256_HEX.matches(profile.modelSha256)) {
        return TrainedE2bModelSelection(
            original.absolutePath,
            Gemma4GpuPrecision.FP16_CORRECTED,
            LocalModelReadiness(false, "No independently verified corrected FP16 package is available for this original model."),
            mtpSupported = false,
        )
    }
    val prepared = File(original.parentFile, CORRECTED_FP16_MODEL_FILE_NAME)
    val modelReadiness = basicModelReadiness(
        prepared.absolutePath, "Corrected FP16 model", allowCorrectedName = true,
    )
    if (!modelReadiness.usable) {
        return TrainedE2bModelSelection(
            prepared.absolutePath,
            Gemma4GpuPrecision.FP16_CORRECTED,
            modelReadiness,
            mtpSupported = false,
        )
    }

    val manifest = File("${prepared.absolutePath}.fp16.json")
    if (!manifest.isFile || !manifest.canRead()) {
        return TrainedE2bModelSelection(
            prepared.absolutePath,
            Gemma4GpuPrecision.FP16_CORRECTED,
            LocalModelReadiness(false, "Corrected FP16 manifest missing: ${manifest.absolutePath}"),
            mtpSupported = false,
        )
    }
    if (manifest.length() >= 1024 * 1024) {
        return TrainedE2bModelSelection(
            prepared.absolutePath,
            Gemma4GpuPrecision.FP16_CORRECTED,
            LocalModelReadiness(false, "Corrected FP16 manifest is too large."),
            mtpSupported = false,
        )
    }
    val manifestData = runCatching { JsonParser.parseString(manifest.readText()).asJsonObject }.getOrNull()
    val manifestValid = runCatching {
        val drafterFlag = manifestData?.get("drafter_rope_corrected")
        manifestData != null &&
            manifestData.get("schema_version").asInt == 1 &&
            manifestData.get("precision_policy").asString == CORRECTED_FP16_POLICY &&
            manifestData.get("max_context_tokens").asInt == CORRECTED_FP16_CONTEXT_TOKENS &&
            manifestData.get("target_rope_corrected").asBoolean &&
            manifestData.get("transform_version").asString == profile.transformVersion &&
            manifestData.get("source_bytes").asLong == profile.sourceBytes &&
            original.length() == profile.sourceBytes &&
            manifestData.get("model_bytes").asLong == prepared.length() &&
            manifestData.get("source_sha256").asString == profile.sourceSha256 &&
            manifestData.get("model_sha256").asString == profile.modelSha256 &&
            drafterFlag != null && drafterFlag.isJsonPrimitive && drafterFlag.asJsonPrimitive.isBoolean
    }.getOrDefault(false)
    if (!manifestValid) {
        return TrainedE2bModelSelection(
            prepared.absolutePath,
            Gemma4GpuPrecision.FP16_CORRECTED,
            LocalModelReadiness(false, "Corrected FP16 manifest is incomplete or does not match the model."),
            mtpSupported = false,
        )
    }
    val drafterCorrected = manifestData!!.get("drafter_rope_corrected").asBoolean
    return TrainedE2bModelSelection(
        prepared.absolutePath,
        Gemma4GpuPrecision.FP16_CORRECTED,
        LocalModelReadiness(
            true,
            "Ready for SDK verification · ${formatDownloadBytes(prepared.length())} · GPU FP16 (corrected)" +
                if (drafterCorrected) "" else " · MTP unavailable (drafter not corrected)",
        ),
        mtpSupported = drafterCorrected,
    )
}

internal fun basicModelReadiness(
    modelPath: String,
    modelName: String,
    allowCorrectedName: Boolean = false,
): LocalModelReadiness {
    val pathError = basicPathError(modelPath, modelName, allowCorrectedName)
    if (pathError != null) return LocalModelReadiness(false, pathError)
    val modelFile = File(modelPath.trim())
    if (!modelFile.isFile) return LocalModelReadiness(false, "$modelName not found: $modelPath")
    if (!modelFile.canRead()) return LocalModelReadiness(false, "$modelName is not readable: $modelPath")
    if (modelFile.length() <= 0L) return LocalModelReadiness(false, "$modelName is empty: $modelPath")
    return LocalModelReadiness(true, "Ready · ${formatDownloadBytes(modelFile.length())} · GPU FP32")
}

private fun basicPathError(
    modelPath: String,
    modelName: String,
    allowCorrectedName: Boolean = false,
): String? {
    val normalized = modelPath.trim()
    if (normalized.isEmpty()) return "$modelName path is empty."
    val modelFile = File(normalized)
    if (!modelFile.isAbsolute) return "$modelName path must be absolute."
    if (!modelFile.name.lowercase(Locale.US).endsWith(".litertlm")) {
        return "$modelName must be a .litertlm file."
    }
    if (!allowCorrectedName && modelFile.name.equals(CORRECTED_FP16_MODEL_FILE_NAME, ignoreCase = true)) {
        return "Keep the original trained model path here and select FP16 (corrected) separately."
    }
    return null
}
