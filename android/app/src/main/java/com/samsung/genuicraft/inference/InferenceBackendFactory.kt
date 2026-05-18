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
        azureOpenAiApiKey: String,
        azureOpenAiResponsesEndpoint: String,
        azureOpenAiDeployment: String,
        localServerBaseUrl: String,
        localModelPath: String,
        onDeviceModelPath: String
    ): InferenceBackend {
        return when (provider) {
            InferenceBackendSettings.Provider.AZURE_OPENAI ->
                AzureOpenAiResponsesBackend(
                    apiKey = azureOpenAiApiKey,
                    responsesEndpoint = azureOpenAiResponsesEndpoint,
                    deployment = azureOpenAiDeployment
                )
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
            InferenceBackendSettings.Provider.ON_DEVICE_LITERT ->
                OnDeviceLitertBackend(onDeviceModelPath)
        }
    }
}
