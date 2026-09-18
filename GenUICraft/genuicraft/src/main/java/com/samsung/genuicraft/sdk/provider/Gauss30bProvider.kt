package com.samsung.genuicraft.sdk.provider

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.GenUiGenerationMetrics
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiProvider
import java.io.FilterInputStream
import java.io.IOException
import java.io.InputStream
import java.net.URI
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.suspendCancellableCoroutine
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.ResponseBody

internal const val GAUSS_MAX_RESPONSE_BYTES = 2_000_000L
private const val GAUSS_MAX_ANSWER_CHARS = 2_000_000

data class GaussConfig(
    val endpoint: String = "https://gaussa.post-train.win/v1/chat/completions",
    val model: String = "gaussa-30b-v0.5-128k",
    val apiKey: String = "",
    val reasoningStrength: String = "low",
    val timeoutMs: Long = 180_000,
)

/** OpenAI-compatible Gauss adapter. Reasoning is never treated as renderable content. */
class Gauss30bProvider(private val config: GaussConfig = GaussConfig()) : GenUiProvider {
    override val id: String = "gauss_30b"
    private val client: OkHttpClient
    private val calls = GaussCallLifecycle()

    init {
        val uri = URI(config.endpoint)
        require(uri.scheme == "https" || (uri.scheme == "http" && uri.host in setOf("localhost", "127.0.0.1", "::1"))) {
            "Gauss endpoint must use HTTPS (HTTP is allowed only for loopback tests)."
        }
        require(config.model.isNotBlank()) { "Gauss model must not be blank." }
        require(config.timeoutMs > 0) { "Gauss timeout must be positive." }
        require(config.reasoningStrength in setOf("none", "low", "medium", "high", "xhigh")) {
            "Unsupported Gauss reasoning strength."
        }
        client = OkHttpClient.Builder()
            .followRedirects(false)
            .followSslRedirects(false)
            .connectTimeout(20, TimeUnit.SECONDS)
            .readTimeout(config.timeoutMs, TimeUnit.MILLISECONDS)
            .callTimeout(config.timeoutMs, TimeUnit.MILLISECONDS).build()
    }

    override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput {
        calls.checkOpen()
        val payload = JsonObject().apply {
            addProperty("model", config.model)
            addProperty("temperature", prompt.temperature)
            addProperty("max_tokens", prompt.maxOutputTokens)
            addProperty("stream", true)
            add("stream_options", JsonObject().apply { addProperty("include_usage", true) })
            add("chat_template_kwargs", JsonObject().apply { addProperty("reasoning_strength", config.reasoningStrength) })
            add("messages", com.google.gson.JsonArray().apply {
                add(JsonObject().apply { addProperty("role", "system"); addProperty("content", prompt.system) })
                add(JsonObject().apply { addProperty("role", "user"); addProperty("content", prompt.user) })
            })
        }
        val request = Request.Builder().url(config.endpoint).header("User-Agent", "GenUICraft/0.1 (Android)")
            .post(payload.toString().toRequestBody("application/json".toMediaType()))
            .apply { if (config.apiKey.isNotBlank()) header("Authorization", "Bearer ${config.apiKey}") }.build()
        return suspendCancellableCoroutine { continuation ->
            val call = client.newCall(request)
            if (!calls.register(call)) {
                continuation.resumeWithException(IllegalStateException("Gauss provider has been closed."))
                return@suspendCancellableCoroutine
            }
            continuation.invokeOnCancellation {
                call.cancel()
                calls.unregister(call)
            }
            if (!continuation.isActive) {
                call.cancel()
                calls.unregister(call)
                return@suspendCancellableCoroutine
            }
            val callback = object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    calls.unregister(call)
                    if (continuation.isActive) continuation.resumeWithException(e)
                }
                override fun onResponse(call: Call, response: Response) {
                    try {
                        response.use {
                            if (!it.isSuccessful) throw IOException("Gauss HTTP ${it.code}; check endpoint, model, and credentials.")
                            val body = it.body ?: throw IOException("Gauss response has no body.")
                            val output = if (it.header("Content-Type").orEmpty().contains("text/event-stream")) {
                                val decoder = GaussResponseDecoder()
                                val event = StringBuilder()
                                body.openBoundedReader().useLines { lines ->
                                    lines.forEach { line ->
                                        if (line.isBlank()) {
                                            if (event.isNotEmpty()) { decoder.accept(event.toString()); event.setLength(0) }
                                        } else if (line.startsWith("data:")) {
                                            if (event.isNotEmpty()) event.append('\n')
                                            event.append(line.removePrefix("data:").trimStart())
                                            require(event.length <= GAUSS_MAX_ANSWER_CHARS) { "Gauss SSE event exceeds size limit." }
                                        }
                                    }
                                }
                                if (event.isNotEmpty()) decoder.accept(event.toString())
                                decoder.finish(config.model)
                            } else {
                                val data = body.openBoundedReader().use { reader -> reader.readText() }
                                GaussResponseDecoder().apply { accept(data) }.finish(config.model)
                            }
                            if (continuation.isActive) continuation.resume(output)
                        }
                    } catch (e: Exception) {
                        if (continuation.isActive) continuation.resumeWithException(e)
                    } finally {
                        calls.unregister(call)
                    }
                }
            }
            try {
                call.enqueue(callback)
            } catch (failure: Exception) {
                calls.unregister(call)
                if (continuation.isActive) continuation.resumeWithException(failure)
            }
        }
    }

    override fun close() {
        val activeCalls = calls.closeAndDrain() ?: return
        activeCalls.forEach(Call::cancel)
        client.dispatcher.cancelAll()
        client.connectionPool.evictAll()
        client.dispatcher.executorService.shutdown()
    }
}

/** Linearizes call registration with close so a request cannot start after cancellation is drained. */
internal class GaussCallLifecycle {
    private val lock = Any()
    @Volatile private var closed = false
    private val activeCalls = linkedSetOf<Call>()

    fun checkOpen() {
        check(!closed) { "Gauss provider has been closed." }
    }

    fun register(call: Call): Boolean = synchronized(lock) {
        if (closed) false else activeCalls.add(call)
    }

    fun unregister(call: Call) {
        synchronized(lock) { activeCalls.remove(call) }
    }

    /** Returns null after the first close; otherwise returns every call registered before close. */
    fun closeAndDrain(): List<Call>? = synchronized(lock) {
        if (closed) return@synchronized null
        closed = true
        activeCalls.toList().also { activeCalls.clear() }
    }
}

private fun ResponseBody.openBoundedReader(): java.io.BufferedReader {
    val declaredLength = contentLength()
    if (declaredLength > GAUSS_MAX_RESPONSE_BYTES) {
        throw IOException("Gauss response exceeds the $GAUSS_MAX_RESPONSE_BYTES byte size limit.")
    }
    val charset = contentType()?.charset(Charsets.UTF_8) ?: Charsets.UTF_8
    return GaussBoundedInputStream(byteStream(), GAUSS_MAX_RESPONSE_BYTES).reader(charset).buffered()
}

/** Counts decoded response bytes even when Content-Length is absent or removed by decompression. */
internal class GaussBoundedInputStream(
    input: InputStream,
    private val maxBytes: Long,
) : FilterInputStream(input) {
    private var bytesRead = 0L

    override fun read(): Int {
        if (bytesRead == maxBytes) return probeEndOfStream()
        return super.read().also { if (it >= 0) bytesRead++ }
    }

    override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
        if (length == 0) return 0
        if (bytesRead == maxBytes) return probeEndOfStream()
        val allowed = minOf(length.toLong(), maxBytes - bytesRead).toInt()
        return super.read(buffer, offset, allowed).also { read ->
            if (read > 0) bytesRead += read
        }
    }

    private fun probeEndOfStream(): Int {
        if (super.read() == -1) return -1
        throw IOException("Gauss response exceeds the $maxBytes byte size limit.")
    }
}

internal class GaussResponseDecoder {
    private val answer = StringBuilder()
    private var finishReason: String? = null
    private var outputTokens: Int? = null
    private var inputTokens: Int? = null

    fun accept(data: String) {
        if (data.trim() == "[DONE]") return
        val chunk = JsonParser.parseString(data).asJsonObject
        if (chunk.has("error")) throw IOException("Gauss returned a provider error.")
        chunk.getAsJsonObject("usage")?.let { usage ->
            usage.get("completion_tokens")?.takeUnless { it.isJsonNull }
                ?.let { outputTokens = it.asInt.takeIf { count -> count >= 0 } }
            usage.get("prompt_tokens")?.takeUnless { it.isJsonNull }
                ?.let { inputTokens = it.asInt.takeIf { count -> count >= 0 } }
        }
        chunk.getAsJsonArray("choices")?.forEach { element ->
            val choice = element.asJsonObject
            if (choice.get("index")?.asInt?.let { it != 0 } == true) return@forEach
            choice.get("finish_reason")?.takeUnless { it.isJsonNull }?.let { finishReason = it.asString }
            val message = choice.getAsJsonObject("delta") ?: choice.getAsJsonObject("message")
            message?.get("content")?.takeUnless { it.isJsonNull }?.let { answer.append(it.asString) }
            require(answer.length <= GAUSS_MAX_ANSWER_CHARS) { "Gauss answer exceeds size limit." }
        }
    }

    fun finish(model: String): GenUiModelOutput {
        require(finishReason == "stop") { "Gauss completion did not finish normally (finish_reason=$finishReason)." }
        require(answer.isNotBlank()) { "Gauss returned no answer content; reasoning may have consumed the output budget." }
        val metrics = if (inputTokens != null || outputTokens != null) {
            GenUiGenerationMetrics(inputTokens, outputTokens, decodeTokensPerSecond = null)
        } else null
        return GenUiModelOutput(answer.toString(), "Gauss/$model", outputTokens, metrics)
    }
}
