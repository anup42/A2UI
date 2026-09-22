package com.samsung.genuicraft

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.samsung.genuicraft.inference.GeminiBackend
import com.samsung.genuicraft.inference.InferenceBackend
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VertexApiSmokeTest {

    @Test
    fun vertexExpress_defaultModelsReturnText() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Context>()
        GeminiApiKeyProvider.refresh(context)

        val apiKey = GeminiApiKeyProvider.vertexExpressApiKey(context).trim()
        require(apiKey.isNotBlank()) {
            "Vertex Express API key is missing. Configure GEMINI_VERTEX_EXPRESS_API_KEY."
        }

        listOf(
            GeminiModelSettings.DEFAULT_RESPONSE_MODEL,
            GeminiModelSettings.DEFAULT_IR_MODEL,
        ).distinct().forEach { model ->
            val backend = GeminiBackend(
                apiKey = apiKey,
                model = model,
                apiMode = InferenceBackendSettings.GeminiApiMode.VERTEX_AI_EXPRESS_API_KEY,
                vertexProjectId = InferenceBackendSettings.getVertexProjectId(context),
                vertexLocation = InferenceBackendSettings.getVertexLocation(context),
                vertexAccessToken = InferenceBackendSettings.getVertexAccessToken(context),
                vertexExpressApiKey = apiKey,
            )
            val response = backend.generate(
                InferenceBackend.GenerateRequest(
                    prompt = "Reply with one short sentence saying Vertex smoke test passed.",
                    systemPrompt = null,
                    temperature = 0.2,
                    maxOutputTokens = 256,
                    jsonMode = false
                )
            )

            val lowerError = response.error.orEmpty().lowercase()
            assertFalse(
                "Vertex auth/config appears invalid for $model: ${response.error}",
                lowerError.contains("401") ||
                    lowerError.contains("unauthenticated") ||
                    lowerError.contains("permission_denied") ||
                    lowerError.contains("api key")
            )
            assertTrue(
                "Vertex call failed for $model: ${response.error ?: "unknown error"}",
                response.error.isNullOrBlank()
            )
            assertTrue(
                "Vertex returned empty text for $model. Raw=${response.rawResponse?.take(260)}",
                response.text.trim().isNotEmpty()
            )
        }
    }
}
