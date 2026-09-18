package com.samsung.genuicraft.sdk.provider

import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.GenUiPrompt
import java.io.IOException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.SocketPolicy
import org.junit.Assert.*
import org.junit.Test

class Gauss30bProviderTest {
    @Test fun `server usage includes prompt counts without claiming native decode speed`() {
        val decoder = GaussResponseDecoder()
        decoder.accept("""{"choices":[{"message":{"content":"Ready"},"finish_reason":"stop"}]}""")
        decoder.accept("""{"choices":[],"usage":{"prompt_tokens":123,"completion_tokens":42}}""")
        decoder.accept("[DONE]")
        val output = decoder.finish("test")
        assertEquals(123, output.metrics?.inputTokens)
        assertEquals(42, output.metrics?.outputTokens)
        assertEquals(output.outputTokens, output.metrics?.outputTokens)
        assertNull(output.metrics?.decodeTokensPerSecond)
    }

    @Test fun `missing server usage stays unavailable`() {
        val decoder = GaussResponseDecoder()
        decoder.accept("""{"choices":[{"message":{"content":"Ready"},"finish_reason":"stop"}]}""")
        val output = decoder.finish("test")
        assertNull(output.outputTokens)
        assertNull(output.metrics)
    }

    @Test fun `streamed reasoning cannot become rendered content`() = runBlocking {
        MockWebServer().use { server ->
            server.enqueue(MockResponse().setHeader("Content-Type", "text/event-stream").setBody(
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"secret reasoning\"}}]}\n\n" +
                    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"<a2ui>\\nroot=Text(\\\"Ready\\\")\\n</a2ui>\"},\"finish_reason\":\"stop\"}]}\n\n" +
                    "data: {\"choices\":[],\"usage\":{\"completion_tokens\":42}}\n\ndata: [DONE]\n\n"
            ))
            Gauss30bProvider(GaussConfig(endpoint = server.url("/v1/chat/completions").toString())).use { provider ->
                val result = provider.generate(GenUiPrompt("system", "response"))
                assertFalse(result.text.contains("reasoning"))
                assertEquals(42, result.outputTokens)
                val request = server.takeRequest(2, TimeUnit.SECONDS)!!
                assertNull(request.getHeader("Authorization"))
                val json = JsonParser.parseString(request.body.readUtf8()).asJsonObject
                assertEquals("gaussa-30b-v0.5-128k", json.get("model").asString)
                assertEquals("low", json.getAsJsonObject("chat_template_kwargs").get("reasoning_strength").asString)
            }
        }
    }

    @Test fun `truncated output is rejected even when it contains a closing tag`() {
        val decoder = GaussResponseDecoder()
        decoder.accept("""{"choices":[{"message":{"content":"<a2ui>root=Text(\"Ready\")</a2ui>"},"finish_reason":"length"}]}""")
        assertThrows(IllegalArgumentException::class.java) { decoder.finish("test") }
    }

    @Test fun `bounded nonstream response still decodes normally`() = runBlocking {
        MockWebServer().use { server ->
            server.enqueue(
                MockResponse()
                    .setHeader("Content-Type", "application/json")
                    .setBody(
                        """{"choices":[{"index":0,"message":{"content":"<a2ui>root=Text(\"Ready\")</a2ui>"},"finish_reason":"stop"}],"usage":{"completion_tokens":9}}""",
                    ),
            )
            Gauss30bProvider(GaussConfig(endpoint = server.url("/v1/chat/completions").toString())).use { provider ->
                val result = provider.generate(GenUiPrompt("system", "response"))
                assertEquals("<a2ui>root=Text(\"Ready\")</a2ui>", result.text)
                assertEquals(9, result.outputTokens)
            }
        }
    }

    @Test fun `declared oversized nonstream response is rejected`() = runBlocking {
        MockWebServer().use { server ->
            server.enqueue(
                MockResponse()
                    .setHeader("Content-Type", "application/json")
                    .setBody("x".repeat((GAUSS_MAX_RESPONSE_BYTES + 1).toInt())),
            )
            Gauss30bProvider(GaussConfig(endpoint = server.url("/v1/chat/completions").toString())).use { provider ->
                val failure = expectIoFailure { provider.generate(GenUiPrompt("system", "response")) }
                assertTrue(failure.message.orEmpty().contains("response exceeds"))
            }
        }
    }

    @Test fun `chunked oversized SSE response is rejected while streaming`() = runBlocking {
        MockWebServer().use { server ->
            val oversized = "data: " + "x".repeat((GAUSS_MAX_RESPONSE_BYTES + 1).toInt()) + "\n\n"
            server.enqueue(
                MockResponse()
                    .setHeader("Content-Type", "text/event-stream")
                    .setChunkedBody(oversized, 4_096),
            )
            Gauss30bProvider(GaussConfig(endpoint = server.url("/v1/chat/completions").toString())).use { provider ->
                val failure = expectIoFailure { provider.generate(GenUiPrompt("system", "response")) }
                assertTrue(failure.message.orEmpty().contains("response exceeds"))
            }
        }
    }

    @Test fun `redirect response is not followed`() = runBlocking {
        MockWebServer().use { destination ->
            MockWebServer().use { origin ->
                destination.enqueue(MockResponse().setBody("unexpected"))
                origin.enqueue(
                    MockResponse()
                        .setResponseCode(307)
                        .setHeader("Location", destination.url("/captured")),
                )
                Gauss30bProvider(GaussConfig(endpoint = origin.url("/v1/chat/completions").toString())).use { provider ->
                    val failure = expectIoFailure { provider.generate(GenUiPrompt("system", "private response")) }
                    assertTrue(failure.message.orEmpty().contains("Gauss HTTP 307"))
                    assertEquals(0, destination.requestCount)
                }
            }
        }
    }

    @Test fun `cancellation stops waiting for network response`() = runBlocking {
        MockWebServer().use { server ->
            server.enqueue(MockResponse().setSocketPolicy(SocketPolicy.NO_RESPONSE))
            Gauss30bProvider(GaussConfig(endpoint = server.url("/v1/chat/completions").toString())).use { provider ->
                val request = async { provider.generate(GenUiPrompt("system", "response")) }
                kotlinx.coroutines.delay(100)
                request.cancelAndJoin()
                assertTrue(request.isCancelled)
            }
        }
    }

    @Test fun `close cancels an active call and rejects later generation`() = runBlocking {
        MockWebServer().use { server ->
            server.enqueue(MockResponse().setSocketPolicy(SocketPolicy.NO_RESPONSE))
            val provider = Gauss30bProvider(GaussConfig(endpoint = server.url("/v1/chat/completions").toString()))
            try {
                val pending = async(Dispatchers.IO) {
                    runCatching { provider.generate(GenUiPrompt("system", "response")) }.exceptionOrNull()
                }
                kotlinx.coroutines.delay(100)
                provider.close()
                assertTrue(withTimeout(2_000) { pending.await() } is IOException)
                try {
                    provider.generate(GenUiPrompt("system", "response"))
                    fail("Generation started after close")
                } catch (_: IllegalStateException) {
                    // Expected.
                }
            } finally {
                provider.close()
            }
        }
    }

    @Test fun `call registration and close are linearizable`() {
        val client = OkHttpClient()
        val request = Request.Builder().url("http://localhost/").build()
        val executor = Executors.newFixedThreadPool(2)
        try {
            repeat(250) {
                val lifecycle = GaussCallLifecycle()
                val call = client.newCall(request)
                val start = CountDownLatch(1)
                val registration = executor.submit<Boolean> {
                    start.await()
                    lifecycle.register(call)
                }
                val closing = executor.submit<List<okhttp3.Call>?> {
                    start.await()
                    lifecycle.closeAndDrain()
                }
                start.countDown()
                val registered = registration.get(2, TimeUnit.SECONDS)
                val drained = closing.get(2, TimeUnit.SECONDS)!!
                assertFalse("A registered call was missed by close", registered && call !in drained)
                assertFalse(lifecycle.register(client.newCall(request)))
                assertNull(lifecycle.closeAndDrain())
            }
        } finally {
            executor.shutdownNow()
            client.dispatcher.executorService.shutdownNow()
            client.connectionPool.evictAll()
        }
    }

    private suspend fun expectIoFailure(block: suspend () -> Unit): IOException = try {
        block()
        throw AssertionError("Expected IOException")
    } catch (failure: IOException) {
        failure
    }
}
