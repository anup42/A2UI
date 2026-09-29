package com.samsung.genuicraft

import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.provider.Gemma4GpuPrecision
import java.io.File
import java.util.Locale

/** The saved path names the original trained package. Corrected FP16 uses a prepared sibling. */
internal const val CORRECTED_FP16_MODEL_FILE_NAME = "model-fp16-corrected.litertlm"
private const val CORRECTED_FP16_POLICY = "genuicraft-fp16-rope-qdq-v1"
private const val CORRECTED_FP16_CONTEXT_TOKENS = 8_192
private val SHA256_HEX = Regex("[0-9a-fA-F]{64}")

internal data class TrainedE2bModelSelection(
    val modelPath: String,
    val gpuPrecision: Gemma4GpuPrecision,
    val readiness: LocalModelReadiness,
    val mtpSupported: Boolean,
)

/** A cheap UI check; the SDK verifies the model digest again before loading FP16. */
internal fun selectTrainedE2bModel(
    originalModelPath: String,
    precision: InferenceBackendSettings.TrainedE2bGpuPrecision,
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

    val sourcePathError = basicPathError(originalModelPath, "Original trained E2B model")
    if (sourcePathError != null) {
        return TrainedE2bModelSelection(
            original.absolutePath,
            Gemma4GpuPrecision.FP16_CORRECTED,
            LocalModelReadiness(false, sourcePathError),
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
    val manifestData = runCatching { JsonParser.parseString(manifest.readText()).asJsonObject }.getOrNull()
    val manifestValid = runCatching {
        manifestData != null &&
            manifestData.get("schema_version").asInt == 1 &&
            manifestData.get("precision_policy").asString == CORRECTED_FP16_POLICY &&
            manifestData.get("max_context_tokens").asInt == CORRECTED_FP16_CONTEXT_TOKENS &&
            manifestData.get("target_rope_corrected").asBoolean &&
            manifestData.get("model_bytes").asLong == prepared.length() &&
            SHA256_HEX.matches(manifestData.get("source_sha256").asString) &&
            SHA256_HEX.matches(manifestData.get("model_sha256").asString) &&
            manifestData.has("drafter_rope_corrected")
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
