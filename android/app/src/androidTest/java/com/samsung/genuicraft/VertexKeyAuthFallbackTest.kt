package com.samsung.genuicraft

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.samsung.genuicraft.inference.InferenceBackend
import com.samsung.genuicraft.inference.InferenceBackendFactory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VertexKeyAuthFallbackTest {

    @Test
    fun geminiApiMode_isAlwaysExpress() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        InferenceBackendSettings.setGeminiApiMode(
            context,
            InferenceBackendSettings.GeminiApiMode.AI_STUDIO_DIRECT
        )
        assertEquals(
            InferenceBackendSettings.GeminiApiMode.VERTEX_AI_EXPRESS_API_KEY,
            InferenceBackendSettings.getGeminiApiMode(context)
        )
    }

    @Test
    fun expressMode_requiresVertexExpressApiKey() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        val backend = InferenceBackendFactory.create(
            provider = InferenceBackendSettings.Provider.GEMINI,
            apiKey = "",
            model = GeminiModelSettings.DEFAULT_RESPONSE_MODEL,
            geminiApiMode = InferenceBackendSettings.GeminiApiMode.VERTEX_AI_EXPRESS_API_KEY,
            vertexProjectId = InferenceBackendSettings.getVertexProjectId(context),
            vertexLocation = InferenceBackendSettings.getVertexLocation(context),
            vertexAccessToken = InferenceBackendSettings.getVertexAccessToken(context),
            vertexExpressApiKey = "",
            azureOpenAiApiKey = "",
            azureOpenAiResponsesEndpoint = "",
            azureOpenAiDeployment = "",
            localServerBaseUrl = InferenceBackendSettings.getLocalServerBaseUrl(context),
            localModelPath = InferenceBackendSettings.getLocalModelPath(context),
            onDeviceModelPath = InferenceBackendSettings.getOnDeviceModelPath(context),
        )

        val response = backend.generate(
            InferenceBackend.GenerateRequest(
                prompt = "Reply with OK only.",
                systemPrompt = null,
                temperature = 0.0,
                maxOutputTokens = 64,
                jsonMode = false
            )
        )

        val error = response.error.orEmpty()
        assertTrue(error.contains("Vertex Express API key is missing", ignoreCase = true))
        assertFalse(error.contains("AI Studio", ignoreCase = true))
    }
}
