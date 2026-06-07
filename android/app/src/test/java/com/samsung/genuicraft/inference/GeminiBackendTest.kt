package com.samsung.genuicraft.inference

import com.samsung.genuicraft.InferenceBackendSettings
import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.net.URL

class GeminiBackendTest {
    @Test
    fun gemmaModelsUseGeminiApiEndpoint() {
        val endpoint = buildEndpointFor(model = "gemma-4-31b-it")

        assertEquals("generativelanguage.googleapis.com", endpoint.host)
        assertTrue(endpoint.path.endsWith("/v1beta/models/gemma-4-31b-it:generateContent"))
        assertTrue(endpoint.query.contains("gemini-key"))
        assertFalse(endpoint.query.contains("vertex-key"))
    }

    @Test
    fun geminiModelsUseVertexExpressEndpoint() {
        val endpoint = buildEndpointFor(model = "gemini-2.5-flash")

        assertEquals("aiplatform.googleapis.com", endpoint.host)
        assertTrue(endpoint.path.endsWith("/v1/publishers/google/models/gemini-2.5-flash:generateContent"))
        assertTrue(endpoint.query.contains("vertex-key"))
        assertFalse(endpoint.query.contains("gemini-key"))
    }

    @Test
    fun extractTextSkipsGemmaThoughtParts() {
        val raw = """
            {
              "candidates": [
                {
                  "content": {
                    "parts": [
                      { "text": "internal reasoning must not leak", "thought": true },
                      { "text": "{\"root\":\"main\",\"elements\":{}}" }
                    ]
                  }
                }
              ]
            }
        """.trimIndent()

        val backend = backend(model = "gemma-4-31b-it")
        val method = GeminiBackend::class.java.getDeclaredMethod("extractGeminiText", String::class.java)
        method.isAccessible = true
        val extraction = method.invoke(backend, raw)
        val textField = extraction.javaClass.getDeclaredField("text")
        textField.isAccessible = true
        val text = textField.get(extraction) as String

        assertEquals("{\"root\":\"main\",\"elements\":{}}", text)
    }

    @Test
    fun gemmaRequestDisablesThoughtSummaries() {
        val backend = backend(model = "gemma-4-31b-it")
        val method = GeminiBackend::class.java.getDeclaredMethod(
            "buildRequestPayload",
            InferenceBackend.GenerateRequest::class.java
        )
        method.isAccessible = true

        val payload = method.invoke(
            backend,
            InferenceBackend.GenerateRequest(
                prompt = "Return JSON.",
                systemPrompt = null,
                temperature = 0.0,
                maxOutputTokens = 256,
                jsonMode = true,
                enableGoogleSearch = false,
                cachedContentName = null,
                structuredOutput = false
            )
        ) as String
        val generationConfig = JsonParser.parseString(payload)
            .asJsonObject
            .getAsJsonObject("generationConfig")
        val thinkingConfig = generationConfig.getAsJsonObject("thinkingConfig")

        assertEquals(false, thinkingConfig.get("includeThoughts").asBoolean)
    }

    private fun buildEndpointFor(model: String): URL {
        val backend = backend(model)
        val method = GeminiBackend::class.java.getDeclaredMethod("buildGenerateEndpoint")
        method.isAccessible = true
        return method.invoke(backend) as URL
    }

    private fun backend(model: String): GeminiBackend {
        return GeminiBackend(
            apiKey = "gemini-key",
            model = model,
            apiMode = InferenceBackendSettings.GeminiApiMode.VERTEX_AI_EXPRESS_API_KEY,
            vertexProjectId = "project-id",
            vertexLocation = "us-central1",
            vertexAccessToken = "",
            vertexExpressApiKey = "vertex-key"
        )
    }
}
