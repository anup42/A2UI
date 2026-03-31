package com.samsung.genuicraft.inference

import com.samsung.genuicraft.InferenceBackendSettings

object InferenceBackendFactory {

    fun create(
        provider: InferenceBackendSettings.Provider,
        apiKey: String,
        model: String,
        geminiApiMode: InferenceBackendSettings.GeminiApiMode,
        vertexProjectId: String,
        vertexLocation: String,
        vertexAccessToken: String,
        vertexExpressApiKey: String,
        localServerBaseUrl: String,
        localModelPath: String
    ): InferenceBackend {
        return when (provider) {
            InferenceBackendSettings.Provider.GEMINI ->
                GeminiBackend(
                    apiKey = apiKey,
                    model = model,
                    apiMode = geminiApiMode,
                    vertexProjectId = vertexProjectId,
                    vertexLocation = vertexLocation,
                    vertexAccessToken = vertexAccessToken,
                    vertexExpressApiKey = vertexExpressApiKey
                )
            InferenceBackendSettings.Provider.LOCAL_SERVER ->
                LocalServerBackend(localServerBaseUrl, localModelPath)
        }
    }
}
