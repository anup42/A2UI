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
    fun vertexExpressModeIsHonoredForGemmaModelNames() {
        val endpoint = buildEndpointFor(model = "gemma-4-31b-it")

        assertEquals("aiplatform.googleapis.com", endpoint.host)
        assertTrue(endpoint.path.endsWith("/v1/publishers/google/models/gemma-4-31b-it:generateContent"))
        assertEquals(null, endpoint.query)
    }

    @Test
    fun aiStudioModeUsesGeminiApiEndpointForGemma() {
        val endpoint = buildEndpointFor(
            model = "gemma-4-31b-it",
            apiMode = InferenceBackendSettings.GeminiApiMode.AI_STUDIO_DIRECT,
        )

        assertEquals("generativelanguage.googleapis.com", endpoint.host)
        assertTrue(endpoint.path.endsWith("/v1beta/models/gemma-4-31b-it:generateContent"))
        assertEquals(null, endpoint.query)
    }

    @Test
    fun geminiModelsUseVertexExpressEndpoint() {
        val endpoint = buildEndpointFor(model = "gemini-2.5-flash")

        assertEquals("aiplatform.googleapis.com", endpoint.host)
        assertTrue(endpoint.path.endsWith("/v1/publishers/google/models/gemini-2.5-flash:generateContent"))
        assertEquals(null, endpoint.query)
    }

    @Test
    fun latestGeminiModelUsesVertexExpressEndpoint() {
        val endpoint = buildEndpointFor(model = "gemini-3.8-flash")

        assertEquals("aiplatform.googleapis.com", endpoint.host)
        assertTrue(endpoint.path.endsWith("/v1/publishers/google/models/gemini-3.8-flash:generateContent"))
        assertEquals(null, endpoint.query)
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

        val backend = backend(
            model = "gemma-4-31b-it",
            apiMode = InferenceBackendSettings.GeminiApiMode.AI_STUDIO_DIRECT,
        )
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
        val backend = backend(
            model = "gemma-4-31b-it",
            apiMode = InferenceBackendSettings.GeminiApiMode.AI_STUDIO_DIRECT,
        )
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

    @Test
    fun gemini3RequestOmitsLegacyTemperature() {
        val generationConfig = generationConfigFor("gemini-3.8-flash")

        assertFalse(generationConfig.has("temperature"))
        assertEquals(256, generationConfig.get("maxOutputTokens").asInt)
    }

    @Test
    fun gemini25RequestKeepsLegacyTemperature() {
        val generationConfig = generationConfigFor("gemini-2.5-flash")

        assertTrue(generationConfig.has("temperature"))
        assertEquals(0.25, generationConfig.get("temperature").asDouble, 0.0)
    }

    @Test
    fun billingFailureGetsActionableVertexExpressHint() {
        val backend = backend(model = "gemini-2.5-flash-lite")

        val hint = backend.buildHttpErrorHint(
            code = 403,
            responseBody = "This API method requires billing to be enabled.",
        )

        assertTrue(hint.contains("billing is disabled", ignoreCase = true))
        assertTrue(hint.contains("active Vertex Express trial/project"))
    }

    private fun buildEndpointFor(
        model: String,
        apiMode: InferenceBackendSettings.GeminiApiMode =
            InferenceBackendSettings.GeminiApiMode.VERTEX_AI_EXPRESS_API_KEY,
    ): URL {
        val backend = backend(model, apiMode)
        val method = GeminiBackend::class.java.getDeclaredMethod("buildGenerateEndpoint")
        method.isAccessible = true
        return method.invoke(backend) as URL
    }

    private fun generationConfigFor(model: String) =
        JsonParser.parseString(
            GeminiBackend::class.java.getDeclaredMethod(
                "buildRequestPayload",
                InferenceBackend.GenerateRequest::class.java,
            ).apply { isAccessible = true }.invoke(
                backend(model),
                InferenceBackend.GenerateRequest(
                    prompt = "Return OK.",
                    systemPrompt = null,
                    temperature = 0.25,
                    maxOutputTokens = 256,
                    jsonMode = false,
                    enableGoogleSearch = false,
                    cachedContentName = null,
                    structuredOutput = false,
                ),
            ) as String,
        ).asJsonObject.getAsJsonObject("generationConfig")

    private fun backend(
        model: String,
        apiMode: InferenceBackendSettings.GeminiApiMode =
            InferenceBackendSettings.GeminiApiMode.VERTEX_AI_EXPRESS_API_KEY,
    ): GeminiBackend {
        return GeminiBackend(
            apiKey = "gemini-key",
            model = model,
            apiMode = apiMode,
            vertexProjectId = "project-id",
            vertexLocation = "us-central1",
            vertexAccessToken = "",
            vertexExpressApiKey = "vertex-key"
        )
    }
}
