package com.samsung.genuicraft.inference

import java.io.File
import java.util.Locale

/**
 * Stage-3 on-device backend hook for exported LiteRT-LM/Gemma IR models.
 *
 * The runtime dependency is intentionally not wired here yet; training/export code writes
 * model packages and manifests that this backend will load once the LiteRT runtime is added.
 */
class OnDeviceLitertBackend(
    private val modelPath: String
) : InferenceBackend {

    override fun generate(request: InferenceBackend.GenerateRequest): InferenceBackend.GenerateResponse {
        val health = checkHealth()
        if (!health.healthy) {
            return InferenceBackend.GenerateResponse(
                text = "",
                rawResponse = null,
                error = health.errorMessage ?: "On-device model is not ready.",
                streamDurationMs = null
            )
        }
        return InferenceBackend.GenerateResponse(
            text = "",
            rawResponse = null,
            error = "On-device LiteRT IR runtime is not bundled yet. Exported model is configured at $modelPath, but runtime loading still needs the LiteRT/Gemma implementation.",
            streamDurationMs = null
        )
    }

    override fun checkHealth(): InferenceBackend.HealthCheckResult {
        val normalized = modelPath.trim()
        if (normalized.isBlank()) {
            return InferenceBackend.HealthCheckResult(
                healthy = false,
                errorMessage = "On-device IR model path is empty. Export a training package and select its model_manifest.json or .litertlm package."
            )
        }
        val file = File(normalized)
        if (!file.exists()) {
            return InferenceBackend.HealthCheckResult(
                healthy = false,
                errorMessage = "On-device IR model path does not exist: $normalized"
            )
        }
        return InferenceBackend.HealthCheckResult(healthy = true)
    }

    override fun classifyError(error: String): InferenceBackend.ErrorClass {
        val lower = error.lowercase(Locale.US)
        return if (lower.contains("not bundled") || lower.contains("not ready")) {
            InferenceBackend.ErrorClass.UNKNOWN
        } else {
            InferenceBackend.ErrorClass.TRANSIENT
        }
    }
}
