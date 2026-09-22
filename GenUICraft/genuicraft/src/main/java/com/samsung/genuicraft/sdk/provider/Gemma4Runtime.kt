package com.samsung.genuicraft.sdk.provider

import android.os.Build
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Capabilities
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ThinkingConfig
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiPromptRole
import com.samsung.genuicraft.sdk.GenUiGenerationMetrics
import com.samsung.genuicraft.sdk.GenUiGenerationSessionMetrics
import com.samsung.genuicraft.sdk.GenUiGenerationFinishReason
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runInterruptible
import kotlinx.coroutines.suspendCancellableCoroutine

internal fun interface Gemma4RuntimeFactory {
    fun create(config: ValidatedGemma4Config): Gemma4Runtime
}

internal interface Gemma4Runtime : AutoCloseable {
    suspend fun generate(prompt: GenUiPrompt, maxOutputTokens: Int): Gemma4RuntimeOutput
    suspend fun generate(
        prompt: GenUiPrompt,
        maxOutputTokens: Int,
        onPartialText: (String) -> Unit,
    ): Gemma4RuntimeOutput = generate(prompt, maxOutputTokens).also { onPartialText(it.text) }
    fun cancelActive()
    suspend fun finishGenerationMetrics(): GenUiGenerationSessionMetrics? = null
    suspend fun awaitClosed()
    override fun close()
}

internal data class Gemma4RuntimeOutput(
    val text: String,
    val runtimeIdentity: String,
    val outputTokens: Int?,
    val metrics: GenUiGenerationMetrics? = null,
    val renderedPromptSha256: String? = null,
    val finishReason: GenUiGenerationFinishReason = GenUiGenerationFinishReason.COMPLETED,
    val finishDetail: String? = null,
)

/** LiteRT-LM implementation isolated behind [Gemma4Runtime] so lifecycle behavior is testable. */
@OptIn(ExperimentalApi::class)
internal class LiteRtGemma4Runtime(
    private val config: ValidatedGemma4Config,
) : Gemma4Runtime {
    private val closed = AtomicBoolean(false)
    private val activeConversation = AtomicReference<Conversation?>(null)
    private val worker: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "GenUICraft-Gemma4").apply { isDaemon = true }
    }

    // Accessed only by the single runtime worker.
    private var engineState: EngineState? = null

    override suspend fun generate(
        prompt: GenUiPrompt,
        maxOutputTokens: Int,
    ): Gemma4RuntimeOutput = generateInternal(prompt, maxOutputTokens, null)

    override suspend fun generate(
        prompt: GenUiPrompt,
        maxOutputTokens: Int,
        onPartialText: (String) -> Unit,
    ): Gemma4RuntimeOutput = generateInternal(prompt, maxOutputTokens, onPartialText)

    private suspend fun generateInternal(
        prompt: GenUiPrompt,
        maxOutputTokens: Int,
        onPartialText: ((String) -> Unit)?,
    ): Gemma4RuntimeOutput = submitCancellable { requestCancelled ->
        check(!closed.get()) { "Gemma 4 runtime has been closed." }
        var engineInitializationWallSeconds: Double? = null
        val state = engineState ?: run {
            val initializationStartedNanos = System.nanoTime()
            initializeEngine().also {
                engineInitializationWallSeconds =
                    (System.nanoTime() - initializationStartedNanos).coerceAtLeast(0L) /
                    1_000_000_000.0
                engineState = it
            }
        }
        throwIfRequestStopped(requestCancelled)
        val deterministic = prompt.temperature <= 0.0
        // Pass thinking explicitly: a null native optional can resolve to thinking disabled.
        // The formatting default bounds reasoning while leaving a generous final layout budget.
        val conversationConfig =
            ConversationConfig(
                systemInstruction = prompt.system.takeIf { it.isNotBlank() }?.let(Contents::of),
                initialMessages = prompt.initialMessages.map { message ->
                    when (message.role) {
                        GenUiPromptRole.USER -> Message.user(message.text)
                        GenUiPromptRole.MODEL -> Message.model(message.text)
                    }
                },
                samplerConfig = SamplerConfig(
                    temperature = prompt.temperature,
                    topK = if (deterministic) 1 else 32,
                    topP = if (deterministic) 1.0 else 0.9,
                ),
                maxOutputToken = maxOutputTokens,
                thinkingConfig = ThinkingConfig(
                    enableThinking = config.enableThinking,
                    thinkingTokenBudget = config.thinkingTokenBudget,
                ),
            )
        val conversation = LiteRtRuntimeFlags.withPromptTemplate(prompt.chatTemplateOverride) {
            state.engine.createConversation(conversationConfig)
        }
        if (requestCancelled.get() || closed.get()) {
            runCatching { conversation.cancelProcess() }
            conversation.close()
            throw java.util.concurrent.CancellationException(
                "Gemma 4 request was cancelled during engine initialization.",
            )
        }
        check(activeConversation.compareAndSet(null, conversation)) {
            "LiteRT-LM conversation serialization invariant was violated."
        }
        try {
            throwIfRequestStopped(requestCancelled)
            val renderedPromptSha256 = prompt.expectedRenderedPrompt?.let { expected ->
                val message = conversation.renderMessageIntoString(Message.user(prompt.user))
                // On the first turn native rendering returns the full history, but leaves
                // bos_token empty. The session inserts Gemma's BOS token separately.
                val rendered = "<bos>" + message
                check(rendered == expected) {
                    "Native prompt differs from the pinned training template " +
                        "(expected=${promptSha256(expected)}, actual=${promptSha256(rendered)}, " +
                        "messageChars=${message.length})."
                }
                promptSha256(rendered)
            }
            // Always use the async path so decode-loop detection also protects callers that do not
            // request UI streaming. The observer remains optional; repetition recovery is not.
            val stream = awaitGemma4Stream(
                start = { callback -> conversation.sendMessageAsync(prompt.user, callback) },
                cancel = { conversation.cancelProcess() },
                onPartialText = onPartialText ?: {},
                isCancelled = { requestCancelled.get() || closed.get() },
            )
            val responseText = if (onPartialText == null) stream.text.trim() else stream.text
            throwIfRequestStopped(requestCancelled)
            val metrics = if (config.enableMetrics) {
                runCatching {
                    readGemma4GenerationMetrics(
                        conversation = conversation,
                        includeEngineInitialization = engineInitializationWallSeconds != null,
                        engineInitializationWallSeconds = engineInitializationWallSeconds,
                    )
                }.getOrNull()
            } else null
            Gemma4RuntimeOutput(
                text = responseText,
                runtimeIdentity = state.runtimeIdentity,
                outputTokens = metrics?.outputTokens ?: runCatching { conversation.outputTokenCount() }.getOrNull(),
                metrics = metrics,
                renderedPromptSha256 = renderedPromptSha256,
                finishReason = if (stream.repetitionStop == null) {
                    GenUiGenerationFinishReason.COMPLETED
                } else {
                    GenUiGenerationFinishReason.REPETITION_LIMIT
                },
                finishDetail = stream.repetitionStop?.detail,
            )
        } finally {
            activeConversation.compareAndSet(conversation, null)
            conversation.close()
        }
    }

    override suspend fun finishGenerationMetrics(): GenUiGenerationSessionMetrics? {
        if (!config.enableMetrics) return null
        return submitCancellable { requestCancelled ->
            throwIfRequestStopped(requestCancelled)
            val state = engineState ?: return@submitCancellable null
            if (!state.speculativeDecodingEnabled) {
                return@submitCancellable GenUiGenerationSessionMetrics(
                    speculativeDecodingEnabled = false,
                )
            }
            // LiteRT-LM owns verified/drafted counters in the engine's MTP drafter and publishes
            // their ratio only from its destructor. End the metrics session after all conversion
            // attempts, then read the process's own bounded log window.
            val logBoundaryEpochMs = System.currentTimeMillis()
            val engineClosed = try {
                state.engine.close()
                true
            } catch (_: Exception) {
                false
            }
            if (engineState === state) engineState = null
            GenUiGenerationSessionMetrics(
                speculativeDecodingEnabled = true,
                drafterAcceptanceRate = if (engineClosed) {
                    MtpAcceptanceLogcat.readSince(logBoundaryEpochMs)
                } else {
                    null
                },
            )
        }
    }

    override fun cancelActive() {
        runCatching { activeConversation.get()?.cancelProcess() }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        cancelActive()
        try {
            worker.submit {
                activeConversation.getAndSet(null)?.let { conversation ->
                    runCatching { conversation.cancelProcess() }
                    runCatching { conversation.close() }
                }
                engineState?.engine?.close()
                engineState = null
            }
        } catch (_: RejectedExecutionException) {
            // The only normal shutdown path submits cleanup before shutting down the worker. This
            // branch is defensive for a concurrently failed executor; there is no safe thread on
            // which to close a possibly active native engine synchronously.
        }
        worker.shutdown()
    }

    override suspend fun awaitClosed() {
        check(closed.get()) { "Gemma 4 runtime must be closed before awaiting termination." }
        runInterruptible(Dispatchers.IO) {
            worker.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS)
        }
    }

    private fun initializeEngine(): EngineState {
        if (config.accelerator == Gemma4Accelerator.GPU) {
            Gemma4NativeLibraries.ensureGpuRuntimeLoaded()
        }
        // Capability belongs to the package, independent of whether this host requested MTP.
        // Keep it truthful in diagnostics when speculative decoding is explicitly disabled.
        val modelHasMtp = modelSupportsMtp(config.modelPath)
        val useMtp = resolveSpeculativeDecoding(
            accelerator = config.accelerator,
            enabledByHost = config.enableSpeculativeDecoding,
            modelSupportsMtp = modelHasMtp,
        )
        val backend = when (config.accelerator) {
            Gemma4Accelerator.GPU -> Backend.GPU()
            Gemma4Accelerator.CPU -> Backend.CPU(config.cpuThreads)
        }
        val cacheDir = config.cacheDir?.let(::prepareCacheDirectory)

        var initializedEngine: Engine? = null
        LiteRtRuntimeFlags.withEngineFlags(
            speculativeDecoding = useMtp,
            benchmark = config.enableMetrics,
        ) {
            try {
                // CPU must explicitly initialize without MTP even if another host component left
                // the process-global experimental flag enabled.
                initializedEngine = Engine(
                    EngineConfig(
                        modelPath = config.modelPath,
                        backend = backend,
                        maxNumTokens = config.maxContextTokens,
                        cacheDir = cacheDir,
                    ),
                ).also(Engine::initialize)
            } catch (failure: Throwable) {
                runCatching { initializedEngine?.close() }
                throw failure
            }
        }
        val engine = checkNotNull(initializedEngine)
        val mtpSuffix = if (useMtp) "+MTP" else ""
        android.util.Log.i("GenUICraftRuntime", "Gemma4 backend=${config.accelerator.name}; MTP=$useMtp; MTPRequested=${config.enableSpeculativeDecoding}; modelSupportsMtp=$modelHasMtp; thinking=${config.enableThinking}; metrics=${config.enableMetrics}")
        return EngineState(
            engine = engine,
            runtimeIdentity = "LiteRT-LM/Gemma4/${config.accelerator.name}$mtpSuffix",
            speculativeDecodingEnabled = useMtp,
        )
    }

    private fun modelSupportsMtp(modelPath: String): Boolean = runCatching {
        Capabilities(modelPath).use(Capabilities::hasSpeculativeDecodingSupport)
    }.getOrDefault(false)

    private fun prepareCacheDirectory(path: String): String {
        val directory = File(path)
        check((directory.isDirectory || directory.mkdirs()) && directory.canWrite()) {
            "Gemma 4 cacheDir is not a writable directory: $path"
        }
        return directory.canonicalPath
    }

    private fun throwIfRequestStopped(requestCancelled: AtomicBoolean) {
        if (requestCancelled.get() || closed.get()) {
            throw java.util.concurrent.CancellationException("Gemma 4 request was cancelled.")
        }
    }

    private suspend fun <T> submitCancellable(
        block: (requestCancelled: AtomicBoolean) -> T,
    ): T = suspendCancellableCoroutine { continuation ->
        if (closed.get()) {
            continuation.resumeWithException(IllegalStateException("Gemma 4 runtime has been closed."))
            return@suspendCancellableCoroutine
        }
        val requestCancelled = AtomicBoolean(false)
        val submitted = AtomicReference<Future<*>?>()
        val future = try {
            worker.submit {
                try {
                    val result = block(requestCancelled)
                    if (continuation.isActive) continuation.resume(result)
                } catch (failure: Throwable) {
                    if (continuation.isActive) continuation.resumeWithException(failure)
                }
            }
        } catch (rejected: RejectedExecutionException) {
            continuation.resumeWithException(IllegalStateException("Gemma 4 runtime has been closed.", rejected))
            return@suspendCancellableCoroutine
        }
        submitted.set(future)
        if (!continuation.isActive) {
            requestCancelled.set(true)
            cancelActive()
            future.cancel(true)
        }
        continuation.invokeOnCancellation {
            requestCancelled.set(true)
            cancelActive()
            submitted.get()?.cancel(true)
        }
    }

    private data class EngineState(
        val engine: Engine,
        val runtimeIdentity: String,
        val speculativeDecodingEnabled: Boolean,
    )

}

private fun promptSha256(text: String): String = MessageDigest.getInstance("SHA-256")
    .digest(text.toByteArray(Charsets.UTF_8)).joinToString("") { "%02x".format(it) }

/** Loads the two libraries whose initialization must precede LiteRT-LM GPU engine creation. */
internal object Gemma4NativeLibraries {
    @Volatile
    private var gpuRuntimeLoaded = false

    fun ensureGpuRuntimeLoaded() {
        if (gpuRuntimeLoaded) return
        synchronized(this) {
            if (gpuRuntimeLoaded) return
            requireArm64GpuAbi(Build.SUPPORTED_ABIS.asList())
            try {
                System.loadLibrary("LiteRt")
                System.loadLibrary("LiteRtTopKOpenClSampler")
            } catch (failure: LinkageError) {
                throw IllegalStateException(
                    "Gemma 4 GPU initialization could not load the packaged ARM64 LiteRT " +
                        "runtime. CPU fallback is not automatic; select accelerator=CPU " +
                        "explicitly if that is the intended backend.",
                    failure,
                )
            }
            gpuRuntimeLoaded = true
        }
    }
}

internal fun requireArm64GpuAbi(supportedAbis: List<String>) {
    check(supportedAbis.any { it.equals("arm64-v8a", ignoreCase = true) }) {
        "Gemma 4 GPU requires arm64-v8a, but this device reports: " +
            supportedAbis.ifEmpty { listOf("<none>") }.joinToString()
    }
}

internal fun resolveSpeculativeDecoding(
    accelerator: Gemma4Accelerator,
    enabledByHost: Boolean,
    modelSupportsMtp: Boolean,
): Boolean {
    require(accelerator != Gemma4Accelerator.GPU || !enabledByHost || modelSupportsMtp) {
        "GPU+MTP was requested, but the model package does not report speculative decoding " +
            "support. Disable MTP or use an MTP-capable Gemma 4 E2B export; the runtime " +
            "will not silently fall back to CPU."
    }
    return accelerator == Gemma4Accelerator.GPU && enabledByHost && modelSupportsMtp
}

private fun Message.textContent(): String = contents.contents.joinToString(separator = "") { content ->
    when (content) {
        is Content.Text -> content.text
        else -> ""
    }
}

private fun Conversation.outputTokenCount(): Int? = runCatching {
    val benchmark = javaClass.getMethod("getBenchmarkInfo").invoke(this)
    benchmark.javaClass.getMethod("getLastDecodeTokenCount").invoke(benchmark).positiveNativeTokenCount()
}.getOrNull()

/** LiteRT-LM exposes these Java getters but hides them from Kotlin metadata. */
internal fun readGemma4GenerationMetrics(
    conversation: Any,
    includeEngineInitialization: Boolean = true,
    engineInitializationWallSeconds: Double? = null,
): GenUiGenerationMetrics? {
    val benchmark = runCatching {
        conversation.javaClass.getMethod("getBenchmarkInfo").invoke(conversation)
    }.getOrNull() ?: return null
    fun value(getter: String): Any? = runCatching {
        benchmark.javaClass.getMethod(getter).invoke(benchmark)
    }.getOrNull()
    val inputTokens = value("getLastPrefillTokenCount").positiveNativeTokenCount()
    val outputTokens = value("getLastDecodeTokenCount").positiveNativeTokenCount()
    val decodeTokensPerSecond = (value("getLastDecodeTokensPerSecond") as? Number)
        ?.toDouble()?.takeIf { it.isFinite() && it > 0.0 }
    val prefillTokensPerSecond = (value("getLastPrefillTokensPerSecond") as? Number)
        ?.toDouble()?.takeIf { it.isFinite() && it > 0.0 }
    val timeToFirstTokenSeconds = (value("getTimeToFirstTokenInSecond") as? Number)
        ?.toDouble()?.takeIf { it.isFinite() && it >= 0.0 }
    val nativeInitializationPhaseSeconds = if (includeEngineInitialization) {
        (value("getInitTimeInSecond") as? Number)
            ?.toDouble()?.takeIf { it.isFinite() && it >= 0.0 }
    } else {
        null
    }
    val engineInitializationSeconds = if (includeEngineInitialization) {
        engineInitializationWallSeconds?.takeIf { it.isFinite() && it >= 0.0 }
    } else {
        null
    }
    if (
        inputTokens == null && outputTokens == null && decodeTokensPerSecond == null &&
        prefillTokensPerSecond == null && timeToFirstTokenSeconds == null &&
        engineInitializationSeconds == null && nativeInitializationPhaseSeconds == null
    ) return null
    return GenUiGenerationMetrics(
        inputTokens = inputTokens,
        outputTokens = outputTokens,
        decodeTokensPerSecond = decodeTokensPerSecond,
        prefillTokensPerSecond = prefillTokensPerSecond,
        timeToFirstTokenSeconds = timeToFirstTokenSeconds,
        engineInitializationSeconds = engineInitializationSeconds,
        engineInitializedForRequest = when {
            !includeEngineInitialization -> false
            engineInitializationSeconds != null || nativeInitializationPhaseSeconds != null -> true
            else -> null
        },
        nativeInitializationPhaseSeconds = nativeInitializationPhaseSeconds,
    )
}

/** Reads only this app UID/process's recent logcat lines; Android never grants cross-app logs. */
internal object MtpAcceptanceLogcat {
    private val acceptance = Regex(
        """(?m)^\s*(\d+(?:\.\d+)?)\s+.*MTP Drafter - Success rate:\s*""" +
            """([0-9]+(?:\.[0-9]+)?)\s*$""",
    )

    fun parse(logcat: String, notBeforeEpochMs: Long): Double? {
        val thresholdSeconds = (notBeforeEpochMs - 500L).coerceAtLeast(0L) / 1_000.0
        return acceptance.findAll(logcat).mapNotNull { match ->
            val timestamp = match.groupValues[1].toDoubleOrNull() ?: return@mapNotNull null
            val rate = match.groupValues[2].toDoubleOrNull() ?: return@mapNotNull null
            rate.takeIf { timestamp >= thresholdSeconds && it.isFinite() && it in 0.0..1.0 }
        }.lastOrNull()
    }

    fun readSince(notBeforeEpochMs: Long): Double? = runCatching {
        val process = ProcessBuilder(
            "logcat", "-d", "-t", "256", "-v", "epoch",
            "--pid=${android.os.Process.myPid()}",
        ).redirectErrorStream(true).start()
        if (!process.waitFor(2, TimeUnit.SECONDS)) {
            process.destroy()
            return@runCatching null
        }
        process.inputStream.bufferedReader().use { parse(it.readText(), notBeforeEpochMs) }
    }.getOrNull()
}

private fun Any?.positiveNativeTokenCount(): Int? {
    val count = (this as? Number)?.toDouble() ?: return null
    return count.takeIf { it.isFinite() && it > 0.0 && it <= Int.MAX_VALUE && it % 1.0 == 0.0 }?.toInt()
}
