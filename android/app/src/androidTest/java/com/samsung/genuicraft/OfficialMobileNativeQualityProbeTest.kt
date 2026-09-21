package com.samsung.genuicraft

import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ThinkingConfig
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Runs a hash-bound batch of already-formatted evaluation prompts through one
 * GPU engine. Every case uses a fresh Conversation. Before decoding, the
 * runtime-rendered complete conversation must byte-for-byte match the
 * independently formatted expected prompt. The expected completion is
 * never staged on the device.
 *
 * Required instrumentation arguments: modelPath, requestPath, label.
 * The JSONL request rows contain id, messages, the checkpoint-formatted prompt
 * and hash, bound chat-template kwargs, and input/output limits. Results are
 * appended after every case so a failed/terminated batch retains an
 * unambiguous partial record.
 */
@RunWith(AndroidJUnit4::class)
class OfficialMobileNativeQualityProbeTest {

    @OptIn(ExperimentalApi::class)
    @Test
    fun evaluatesBoundPromptsWithOneGpuEngine() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val arguments = InstrumentationRegistry.getArguments()
        val modelFile = requiredFile(arguments.getString("modelPath"), "modelPath")
        val requestFile = requiredFile(arguments.getString("requestPath"), "requestPath")
        val label = sanitizeLabel(arguments.getString("label").orEmpty())
        // The bound training evaluator remains target-only by default. A separate
        // small probe may explicitly enable MTP when the package has a drafter.
        val mtpEnabled = arguments.getString("mtp")?.toBooleanStrictOrNull() ?: false
        val caseTimeoutSeconds = arguments.getString("caseTimeoutSeconds")?.toLongOrNull()
            ?: error("Pass -e caseTimeoutSeconds <positive integer>")
        require(caseTimeoutSeconds > 0) { "caseTimeoutSeconds must be positive" }
        val requests = requestFile.useLines { lines ->
            lines.filter { it.isNotBlank() }.map(::JSONObject).toList()
        }
        assertTrue("Native quality request batch is empty", requests.isNotEmpty())
        val ids = requests.map { it.getString("id") }
        assertEquals("Request IDs must be unique", ids.size, ids.toSet().size)
        val maxInputTokens = requests.maxOf { it.getInt("max_input_tokens") }
        val maxNewTokens = requests.map { it.getInt("max_new_tokens") }.toSet()
        assertEquals("One batch must use one decode limit", 1, maxNewTokens.size)
        val outputLimit = maxNewTokens.single()
        assertTrue("max_new_tokens must be in 1..4096", outputLimit in 1..4_096)

        val reportDir = File(context.filesDir, REPORT_DIRECTORY).apply {
            check(mkdirs() || isDirectory) { "Could not create report directory: $absolutePath" }
        }
        val resultFile = File(reportDir, "$label.jsonl")
        val manifestFile = File(reportDir, "$label.manifest.json")
        check(!resultFile.exists() && !manifestFile.exists()) {
            "Refusing to overwrite native quality reports for label=$label"
        }
        val cacheDir = File(context.cacheDir, "$CACHE_DIRECTORY/$label").apply {
            check(mkdirs() || isDirectory) { "Could not create cache directory: $absolutePath" }
        }
        val manifest = JSONObject()
            .put("schema_version", 1)
            .put("label", label)
            .put("model_path", modelFile.canonicalPath)
            .put("model_size_bytes", modelFile.length())
            .put("request_path", requestFile.canonicalPath)
            .put("request_count", requests.size)
            .put("requested_backend", "GPU")
            .put("mtp_enabled", mtpEnabled)
            .put("engine_instance_count", 1)
            .put("fresh_conversation_per_case", true)
            .put("raw_session", false)
            .put("conversation_template_applied", true)
            .put("runtime_rendered_prompt_required", true)
            .put("native_input_token_ids_available", false)
            .put("native_output_token_ids_available", false)
            .put(
                "token_id_unavailable_reason",
                "LiteRT Android 0.16.1 public Conversation API exposes neither tokenize nor generated token IDs",
            )
            .put("max_input_tokens", maxInputTokens)
            .put("max_new_tokens", outputLimit)
            .put("sampler", samplerJson())
            .put("stop_policy_version", "a2ui-envelope-v1")
            .put("device_model", Build.MODEL)
            .put("device_product", Build.PRODUCT)
            .put("android_sdk", Build.VERSION.SDK_INT)
            .put("supported_abis", Build.SUPPORTED_ABIS.joinToString(","))
            .put("completed_cases", 0)
            .put("success", false)
        manifestFile.writeText(manifest.toString(2) + "\n")

        ExperimentalFlags.enableSpeculativeDecoding = mtpEnabled
        ExperimentalFlags.enableBenchmark = true
        loadGpuSamplerDependencies()
        var engine: Engine? = null
        val batchStarted = SystemClock.elapsedRealtime()
        try {
            Log.i(LOG_TAG, "run_label=$label")
            emitStatus(instrumentation, "engine_start", null, 0, requests.size)
            engine = Engine(
                EngineConfig(
                    modelPath = modelFile.canonicalPath,
                    backend = Backend.GPU(),
                    maxNumTokens = maxInputTokens + outputLimit,
                    cacheDir = cacheDir.absolutePath,
                )
            )
            engine.initialize()
            manifest.put("engine_init_elapsed_ms", SystemClock.elapsedRealtime() - batchStarted)
            emitStatus(instrumentation, "engine_ready", null, 0, requests.size)

            resultFile.bufferedWriter(Charsets.UTF_8).use { writer ->
                requests.forEachIndexed { index, request ->
                    val id = request.getString("id")
                    val prompt = request.getString("formatted_prompt")
                    val expectedPromptSha256 = request.getString("prompt_sha256")
                    assertEquals(
                        "Prompt transport hash changed for $id",
                        expectedPromptSha256,
                        sha256(prompt),
                    )
                    assertEquals(outputLimit, request.getInt("max_new_tokens"))
                    val messages = request.getJSONArray("messages")
                    val templateContext = request.getJSONObject("chat_template_kwargs").let { value ->
                        value.keys().asSequence().associateWith { key -> value.get(key) }
                    }
                    val thinkingValue = templateContext["enable_thinking"]
                    require(thinkingValue == null || thinkingValue is Boolean) {
                        "Bound chat_template_kwargs.enable_thinking must be Boolean for $id"
                    }
                    val thinkingEnabled = thinkingValue as? Boolean ?: false
                    assertTrue("Conversation must contain at least one message for $id", messages.length() > 0)
                    val nativeMessages = (0 until messages.length()).map { offset ->
                        val item = messages.getJSONObject(offset)
                        val text = item.getString("content")
                        when (item.getString("role")) {
                            "system" -> Message.system(text)
                            "user" -> Message.user(text)
                            "assistant", "model" -> Message.model(text)
                            else -> error("Unsupported message role for $id at $offset")
                        }
                    }
                    val finalMessage = nativeMessages.last()
                    assertEquals(
                        "Final prompt message must be user for $id",
                        com.google.ai.edge.litertlm.Role.USER,
                        finalMessage.role,
                    )
                    val initialMessages = nativeMessages.dropLast(1)
                    emitStatus(instrumentation, "case_start", id, index, requests.size)
                    val started = SystemClock.elapsedRealtime()
                    val raw = StringBuilder()
                    var outputTokenCount = 0
                    var inputTokenCount: Int? = null
                    var decodeTokensPerSecond: Double? = null
                    var prefillTokensPerSecond: Double? = null
                    var renderedPromptSha256 = ""
                    var closingBoundary: Int? = null
                    engine.createConversation(
                        ConversationConfig(
                            initialMessages = initialMessages,
                            samplerConfig = SamplerConfig(
                                topK = 1,
                                topP = 1.0,
                                temperature = 0.0,
                                seed = 42,
                            ),
                            channels = emptyList(),
                            extraContext = templateContext,
                            maxOutputToken = outputLimit,
                            thinkingConfig = ThinkingConfig(enableThinking = thinkingEnabled),
                        )
                    ).use { conversation ->
                        // Before the first send, LiteRT 0.16.1 renders the complete
                        // history with the final message. Concatenating the preface
                        // duplicates that history. Native inserts BOS separately.
                        val renderedMessage = conversation.renderMessageIntoString(finalMessage, templateContext)
                        val renderedPrompt = if (renderedMessage.startsWith("<bos>")) renderedMessage
                            else "<bos>" + renderedMessage
                        renderedPromptSha256 = sha256(renderedPrompt)
                        assertEquals(
                            "LiteRT conversation template differs from checkpoint prompt for $id",
                            prompt,
                            renderedPrompt,
                        )
                        val done = CountDownLatch(1)
                        val asyncFailure = AtomicReference<Throwable?>(null)
                        conversation.sendMessageAsync(
                            finalMessage,
                            object : MessageCallback {
                                override fun onMessage(message: Message) {
                                    try {
                                        check(message.channels.isEmpty()) {
                                            "LiteRT-LM returned separated channels for $id; " +
                                                "raw HF-equivalent text is unavailable"
                                        }
                                        val chunk = message.contents.contents.joinToString(separator = "") { content ->
                                            when (content) {
                                                is Content.Text -> content.text
                                                else -> error(
                                                    "LiteRT-LM returned non-text content for $id: " +
                                                        content.javaClass.name
                                                )
                                            }
                                        }
                                        raw.append(chunk)
                                        if (closingBoundary == null) {
                                            closingBoundary = closingSentinelEnd(raw)
                                            if (closingBoundary != null) conversation.cancelProcess()
                                        }
                                    } catch (error: Throwable) {
                                        asyncFailure.compareAndSet(null, error)
                                        runCatching { conversation.cancelProcess() }
                                        done.countDown()
                                    }
                                }

                                override fun onDone() {
                                    done.countDown()
                                }

                                override fun onError(error: Throwable) {
                                    if (closingBoundary == null) asyncFailure.compareAndSet(null, error)
                                    done.countDown()
                                }
                            },
                            extraContext = templateContext,
                            maxOutputToken = outputLimit,
                            thinkingConfig = ThinkingConfig(enableThinking = thinkingEnabled),
                        )
                        if (!done.await(caseTimeoutSeconds, TimeUnit.SECONDS)) {
                            conversation.cancelProcess()
                            error("Native generation exceeded case timeout for $id")
                        }
                        asyncFailure.get()?.let { throw it }
                        val benchmark = conversation.getBenchmarkInfo()
                        outputTokenCount = benchmark.lastDecodeTokenCount
                        fun counter(getter: String): Double? = runCatching {
                            (benchmark.javaClass.getMethod(getter).invoke(benchmark) as Number)
                                .toDouble().takeIf { it.isFinite() && it >= 0.0 }
                        }.getOrNull()
                        inputTokenCount = counter("getLastPrefillTokenCount")?.toInt()
                        decodeTokensPerSecond = counter("getLastDecodeTokensPerSecond")
                        prefillTokensPerSecond = counter("getLastPrefillTokensPerSecond")
                    }
                    val stopReason = when {
                        closingBoundary != null -> "closing_sentinel"
                        outputTokenCount >= outputLimit -> "max_new_tokens"
                        else -> "eos_or_native_stop"
                    }
                    val result = JSONObject()
                        .put("schema_version", 1)
                        .put("id", id)
                        .put("prompt_sha256", expectedPromptSha256)
                        .put("prompt_transport_verified", true)
                        .put("runtime_rendered_prompt_sha256", renderedPromptSha256)
                        .put("runtime_template_prompt_verified", renderedPromptSha256 == expectedPromptSha256)
                        .put("raw_generated_text", raw.toString())
                        .put("generation_elapsed_ms", SystemClock.elapsedRealtime() - started)
                        .put("native_output_token_count", outputTokenCount)
                        .put("native_input_token_count", inputTokenCount ?: JSONObject.NULL)
                        .put("native_decode_tokens_per_second", decodeTokensPerSecond ?: JSONObject.NULL)
                        .put("native_prefill_tokens_per_second", prefillTokensPerSecond ?: JSONObject.NULL)
                        .put("native_stop_observed", closingBoundary != null || outputTokenCount < outputLimit)
                        .put("decode_limit_reached", outputTokenCount >= outputLimit)
                        .put("stop_reason", stopReason)
                        .put("stop_policy_version", "a2ui-envelope-v1")
                        .put(
                            "raw_output_scope",
                            "chunks_actually_returned_by_runtime; continuation_after_stop_unobserved",
                        )
                        .put("sampler", samplerJson())
                        .put("requested_backend", "GPU")
                        .put("mtp_enabled", mtpEnabled)
                        .put("raw_session", false)
                        .put("conversation_template_applied", true)
                        .put("native_input_token_ids_available", false)
                        .put("native_output_token_ids_available", false)
                    writer.append(result.toString()).append('\n')
                    writer.flush()
                    manifest.put("completed_cases", index + 1)
                    manifestFile.writeText(manifest.toString(2) + "\n")
                    emitStatus(instrumentation, "case_done", id, index + 1, requests.size)
                }
            }
            manifest
                .put("success", true)
                .put("batch_elapsed_ms", SystemClock.elapsedRealtime() - batchStarted)
            emitStatus(instrumentation, "finished", null, requests.size, requests.size)
        } catch (failure: Throwable) {
            manifest
                .put("success", false)
                .put("batch_elapsed_ms", SystemClock.elapsedRealtime() - batchStarted)
                .put("error_type", failure.javaClass.name)
                .put("error", failure.message.orEmpty())
            Log.e(LOG_TAG, "Official mobile native quality batch failed", failure)
            throw failure
        } finally {
            val closeFailure = runCatching { engine?.close() }.exceptionOrNull()
            manifest.put("engine_closed", closeFailure == null)
            closeFailure?.let {
                manifest.put("close_error_type", it.javaClass.name)
                manifest.put("close_error", it.message.orEmpty())
            }
            manifestFile.writeText(manifest.toString(2) + "\n")
            Log.i(LOG_TAG, "OFFICIAL_MOBILE_NATIVE report=${resultFile.absolutePath}")
        }
    }

    private fun requiredFile(value: String?, name: String): File {
        val path = requireNotNull(value?.trim()?.takeIf { it.isNotEmpty() }) {
            "Pass -e $name /absolute/device/path"
        }
        return File(path).also {
            assertTrue("$name does not exist: $path", it.isFile)
            assertTrue("$name is empty: $path", it.length() > 0L)
        }
    }

    private fun emitStatus(
        instrumentation: android.app.Instrumentation,
        phase: String,
        id: String?,
        completed: Int,
        total: Int,
    ) {
        val event = JSONObject()
            .put("phase", phase)
            .put("completed", completed)
            .put("total", total)
        if (id != null) event.put("id", id)
        instrumentation.sendStatus(
            0,
            Bundle().apply { putString(STATUS_KEY, event.toString()) },
        )
        Log.i(LOG_TAG, "$STATUS_KEY=${event}")
    }

    private fun sha256(value: String): String = MessageDigest.getInstance("SHA-256")
        .digest(value.toByteArray(Charsets.UTF_8))
        .joinToString("") { "%02x".format(it) }

    private fun samplerJson(): JSONObject = JSONObject()
        .put("do_sample", false)
        .put("top_k", 1)
        .put("top_p", 1.0)
        .put("temperature", 0.0)
        .put("seed", 42)

    /** Mirrors generation_policy.closing_sentinel_end for </a2ui>. */
    private fun closingSentinelEnd(text: CharSequence): Int? {
        val sentinel = "</a2ui>"
        var quote: Char? = null
        var escaped = false
        var index = 0
        while (index < text.length) {
            val char = text[index]
            if (quote != null) {
                when {
                    escaped -> escaped = false
                    char == '\\' -> escaped = true
                    char == quote -> quote = null
                }
            } else if (char == '"' || char == '\'') {
                quote = char
            } else if (index + sentinel.length <= text.length &&
                (0 until sentinel.length).all { text[index + it] == sentinel[it] }
            ) {
                return index + sentinel.length
            }
            index += 1
        }
        return null
    }

    private fun sanitizeLabel(value: String): String {
        val normalized = value.replace(Regex("[^A-Za-z0-9._-]+"), "_").trim('_')
        return normalized.ifBlank { "native_quality" }.take(96)
    }

    private fun loadGpuSamplerDependencies() {
        listOf("c++_shared", "LiteRt", "LiteRtTopKOpenClSampler").forEach { library ->
            runCatching { System.loadLibrary(library) }
                .onFailure { Log.w(LOG_TAG, "GPU sampler dependency failed name=$library", it) }
        }
    }

    private companion object {
        private const val LOG_TAG = "OfficialMobileNative"
        private const val STATUS_KEY = "a2ui_native_event"
        private const val REPORT_DIRECTORY = "official_mobile_native_quality_reports"
        private const val CACHE_DIRECTORY = "official_mobile_native_quality"
    }
}
