package com.samsung.genuicraft

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.samsung.genuicraft.inference.InferenceBackend
import com.samsung.genuicraft.inference.InferenceBackendFactory
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VertexKeyAuthFallbackTest {

    @Test
    fun vertexMode_doesNotFailWith401() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        GeminiApiKeyProvider.refresh(context)

        val geminiApiKey = GeminiApiKeyProvider.stage3ApiKey(context).trim()
        val vertexExpressApiKey = GeminiApiKeyProvider.vertexExpressApiKey(context).trim()
        assertTrue("Gemini API key must be configured", geminiApiKey.isNotBlank())
        assertTrue("Vertex Express API key must be configured", vertexExpressApiKey.isNotBlank())

        val backend = InferenceBackendFactory.create(
            provider = InferenceBackendSettings.Provider.GEMINI,
            apiKey = geminiApiKey,
            model = "gemini-2.5-flash",
            geminiApiMode = InferenceBackendSettings.GeminiApiMode.VERTEX_AI_EXPRESS_API_KEY,
            vertexProjectId = InferenceBackendSettings.getVertexProjectId(context),
            vertexLocation = InferenceBackendSettings.getVertexLocation(context),
            vertexAccessToken = InferenceBackendSettings.getVertexAccessToken(context),
            vertexExpressApiKey = vertexExpressApiKey,
            localServerBaseUrl = InferenceBackendSettings.getLocalServerBaseUrl(context),
            localModelPath = InferenceBackendSettings.getLocalModelPath(context)
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
        assertFalse(
            "Vertex-mode request should not fail with HTTP 401 after fallback fix. error=$error",
            error.contains("HTTP 401", ignoreCase = true)
        )
    }
}

