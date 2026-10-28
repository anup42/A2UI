package com.samsung.genuicraft.sdk.provider

import android.app.ActivityManager
import android.content.ComponentCallbacks2
import android.content.Context
import android.content.res.Configuration
import com.samsung.genuicraft.sdk.GenUiGenerationSessionMetrics
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.telemetry.GenUiResourceMonitor
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/** Process-wide engine lifetime. Provider close releases a lease; it does not destroy a warm engine. */
object GenUiEngineManager {
    const val IDLE_RELEASE_MILLIS = 120_000L
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO +
        kotlinx.coroutines.CoroutineExceptionHandler { _, failure ->
            android.util.Log.e("GenUICraftEngine", "Background engine retirement failed", failure)
        })
    private val retired = MutableStateFlow<GenUiGenerationSessionMetrics?>(null)
    /** Latest completed engine lifetime, potentially covering several requests. Never per-request. */
    val retirementMetrics: StateFlow<GenUiGenerationSessionMetrics?> = retired.asStateFlow()
    @Volatile internal var resourceMonitor: GenUiResourceMonitor? = null
        private set
    @Volatile private var installed = false
    @Volatile private var pressureUntilNanos = 0L
    @Volatile private var activityManager: ActivityManager? = null

    internal val pool = ManagedGemmaRuntimePool(
        factory = Gemma4RuntimeFactory(::LiteRtGemma4Runtime), scope = scope,
        idleMillis = IDLE_RELEASE_MILLIS,
        releaseLegacy = LiteRtModelRunner::releaseCachedEngine,
        onRetirement = { retired.value = it },
        allowPrewarm = {
            val low = runCatching {
                val info = ActivityManager.MemoryInfo()
                activityManager?.getMemoryInfo(info)
                info.lowMemory
            }.getOrDefault(false)
            !low && System.nanoTime() >= pressureUntilNanos
        },
    )

    /** Register once using the Application context, also done by the Context-based GenUiSession. */
    @Synchronized fun install(context: Context) {
        if (installed) return
        val app = context.applicationContext
        activityManager = app.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        resourceMonitor = GenUiResourceMonitor.install(app)
        app.registerComponentCallbacks(object : ComponentCallbacks2 {
            override fun onConfigurationChanged(newConfig: Configuration) = Unit
            override fun onLowMemory() = memoryPressure()
            override fun onTrimMemory(level: Int) {
                // UI_HIDDEN alone is not memory pressure. The bounded idle timer remains active.
                if (level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW ||
                    level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL ||
                    level >= ComponentCallbacks2.TRIM_MEMORY_BACKGROUND) memoryPressure()
            }
        })
        installed = true
    }

    private fun memoryPressure() {
        pressureUntilNanos = System.nanoTime() + 30_000_000_000L
        scope.launch { runCatching { pool.retire("memory_pressure") } }
    }

    /** Drains any active request before releasing native memory; useful for explicit host teardown. */
    suspend fun retire(reason: String = "host_release") = pool.retire(reason)
    internal fun lease(config: ValidatedGemma4Config): Gemma4Runtime = pool.lease(config)
}

/** One lifecycle lock covers engine handoff, requests, prewarm and retirement, including legacy hosts. */
internal class ManagedGemmaRuntimePool(
    private val factory: Gemma4RuntimeFactory,
    private val scope: CoroutineScope,
    private val idleMillis: Long,
    private val releaseLegacy: () -> Unit = {},
    private val onRetirement: (GenUiGenerationSessionMetrics) -> Unit = {},
    private val allowPrewarm: () -> Boolean = { true },
) {
    private val mutex = Mutex()
    private var slot: Slot? = null
    private var idleJob: Job? = null
    private var legacyRetained = false
    private var failedDrain: Throwable? = null

    fun lease(config: ValidatedGemma4Config): Gemma4Runtime = Lease(config)

    suspend fun retire(reason: String) = mutex.withLock {
        idleJob?.cancel()
        idleJob = null
        retireLocked(reason)
        if (legacyRetained) { releaseLegacy(); legacyRetained = false }
    }

    suspend fun <T> withLegacyRuntime(block: () -> T): T = mutex.withLock {
        checkDrain()
        idleJob?.cancel()
        retireLocked("legacy_runtime_handoff")
        try {
            legacyRetained = true
            block()
        } finally { scheduleIdle() }
    }

    private suspend fun retireLocked(reason: String) {
        val old = slot ?: return
        // Always drain cleanup even if the caller was cancelled; no replacement can overlap it.
        withContext(NonCancellable) {
            try {
                old.runtime.retire(reason)?.let(onRetirement)
            } finally {
                try {
                    old.runtime.close()
                    old.runtime.awaitClosed()
                } catch (failure: Throwable) {
                    // Fail closed: native ownership is uncertain; never overlap a replacement.
                    failedDrain = failure
                    throw failure
                } finally { slot = null }
            }
        }
    }

    private fun checkDrain() {
        failedDrain?.let { throw IllegalStateException("Native engine cleanup failed; restart the app before reloading.", it) }
    }

    private fun scheduleIdle() {
        idleJob?.cancel()
        idleJob = scope.launch {
            var waited = 0L
            while (true) {
                // Recent Android versions may omit low-memory callbacks. Check while retained,
                // too; this only reads ActivityManager memory state, never GPU clocks or PSS.
                val interval = minOf(15_000L, (idleMillis - waited).coerceAtLeast(0L))
                delay(interval)
                waited += interval
                val done = mutex.withLock {
                    val pressure = !allowPrewarm()
                    if (pressure || waited >= idleMillis) {
                        retireLocked(if (pressure) "memory_pressure" else "idle_timeout")
                        if (legacyRetained) { releaseLegacy(); legacyRetained = false }
                        true
                    } else false
                }
                if (done) break
            }
        }
    }

    private data class Key(val config: ValidatedGemma4Config, val bytes: Long, val modified: Long,
        val manifestBytes: Long, val manifestModified: Long)
    private fun key(config: ValidatedGemma4Config): Key {
        val model = File(config.modelPath)
        val manifest = File(config.modelPath + ".fp16.json")
        return Key(config, model.length(), model.lastModified(), manifest.length(), manifest.lastModified())
    }
    private data class Slot(val key: Key, val runtime: Gemma4Runtime)

    private inner class Lease(private val config: ValidatedGemma4Config) : Gemma4Runtime {
        private val closed = AtomicBoolean(false)
        private val operations = Mutex()
        private val activeLock = Any()
        private var active: Gemma4Runtime? = null
        @Volatile private var metrics: GenUiGenerationSessionMetrics? = null

        private suspend fun <T> use(prewarm: Boolean = false, block: suspend (Gemma4Runtime) -> T): T? =
            operations.withLock { mutex.withLock runtimeLock@{
                check(!closed.get()) { "GenUICraft engine lease is closed." }
                checkDrain()
                if (prewarm && !allowPrewarm()) return@runtimeLock null
                idleJob?.cancel()
                try {
                    val wanted = key(config)
                    if (slot?.key != wanted) retireLocked("configuration_change")
                    if (legacyRetained) { releaseLegacy(); legacyRetained = false }
                    // Legacy hosts may have populated their cache before manager installation.
                    releaseLegacy()
                    val runtime = slot?.runtime ?: factory.create(config).also { slot = Slot(wanted, it) }
                    synchronized(activeLock) {
                        check(!closed.get()) { "GenUICraft engine lease is closed." }
                        active = runtime
                    }
                    val result = block(runtime)
                    check(!closed.get()) { "GenUICraft engine lease closed during request." }
                    metrics = try { runtime.finishGenerationMetrics() }
                    catch (cancelled: kotlinx.coroutines.CancellationException) { throw cancelled }
                    catch (_: Exception) { null }
                    result
                } catch (failure: Throwable) {
                    // Cancellation can resume its caller before native work unwinds. Drain first.
                    retireLocked("request_interrupted")
                    throw failure
                } finally {
                    synchronized(activeLock) { active = null }
                    scheduleIdle()
                }
            } }

        override suspend fun prewarm() { use(prewarm = true) { it.prewarm() } }
        override suspend fun generate(prompt: GenUiPrompt, maxOutputTokens: Int): Gemma4RuntimeOutput =
            checkNotNull(use { it.generate(prompt, maxOutputTokens) })
        override suspend fun generate(prompt: GenUiPrompt, maxOutputTokens: Int,
            onPartialText: (String) -> Unit): Gemma4RuntimeOutput =
            checkNotNull(use { it.generate(prompt, maxOutputTokens, onPartialText) })
        override suspend fun finishGenerationMetrics(): GenUiGenerationSessionMetrics? = metrics
        override fun cancelActive() = synchronized(activeLock) { active?.cancelActive(); Unit }
        override fun close() { if (closed.compareAndSet(false, true)) cancelActive() }
        override suspend fun awaitClosed() { check(closed.get()); operations.withLock {} }
    }
}
