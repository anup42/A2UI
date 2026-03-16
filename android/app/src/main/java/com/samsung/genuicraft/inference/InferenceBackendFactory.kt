package com.samsung.genuicraft.inference

import com.samsung.genuicraft.InferenceBackendSettings

object InferenceBackendFactory {

    fun create(
        provider: InferenceBackendSettings.Provider,
        apiKey: String,
        model: String,
        localServerBaseUrl: String,
        localModelPath: String
    ): InferenceBackend {
        return when (provider) {
            InferenceBackendSettings.Provider.GEMINI ->
                GeminiBackend(apiKey, model)
            InferenceBackendSettings.Provider.LOCAL_SERVER ->
                LocalServerBackend(localServerBaseUrl, localModelPath)
        }
    }
}
