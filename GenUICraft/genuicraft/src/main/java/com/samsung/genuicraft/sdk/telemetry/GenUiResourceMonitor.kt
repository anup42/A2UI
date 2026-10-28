package com.samsung.genuicraft.sdk.telemetry

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import java.io.File
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.ScheduledThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * Opt-in, read-only device sampler for native generation requests. Install once with an application
 * context and reuse the monitor. Each request gets at most 256 samples, scheduled once per second.
 * Unavailable GPU sysfs, memory, or thermal sources are recorded as null rather than estimated.
 */
class GenUiResourceMonitor private constructor(context: Context) : AutoCloseable {
    companion object {
        private const val SAMPLE_INTERVAL_MS = 1_000L
        private const val MAX_SAMPLES = 256

        @JvmStatic
        fun install(context: Context): GenUiResourceMonitor = GenUiResourceMonitor(context)
    }

    private val appContext = context.applicationContext ?: context
    private val scheduler = ScheduledThreadPoolExecutor(1) { runnable ->
        Thread(runnable, "GenUICraft-resource-monitor").apply { isDaemon = true }
    }.apply { removeOnCancelPolicy = true }

    /** Start monitoring one request; [Session.stop] returns its immutable observation snapshot. */
    fun start(requestId: String, runtimeBackend: String): Session {
        val source = AndroidResourceSource(appContext)
        val accumulator = ResourceSampleAccumulator(
            requestId = requestId,
            runtimeBackend = runtimeBackend,
            clockMs = SystemClock::elapsedRealtime,
            sampleAt = source::read,
            sampleIntervalMs = SAMPLE_INTERVAL_MS,
            maxSamples = MAX_SAMPLES,
        )
        val taskRef = AtomicReference<ScheduledFuture<*>?>()
        val task = try {
            // Fixed delay prevents catch-up bursts if a device read itself takes over one second.
            scheduler.scheduleWithFixedDelay(
                {
                    if (!accumulator.capture()) taskRef.get()?.cancel(false)
                },
                0L,
                SAMPLE_INTERVAL_MS,
                TimeUnit.MILLISECONDS,
            ).also(taskRef::set)
        } catch (_: RejectedExecutionException) {
            // A closed monitor never interrupts generation. Its session reports zero coverage.
            null
        }
        return Session(accumulator, task)
    }

    override fun close() {
        scheduler.shutdownNow()
    }

    class Session internal constructor(
        private val accumulator: ResourceSampleAccumulator,
        private val task: ScheduledFuture<*>?,
    ) : AutoCloseable {
        /** Safe to call again; the first stop fixes the elapsed duration and samples. */
        fun stop(): GenUiResourceSummary {
            task?.cancel(false)
            return accumulator.finish()
        }

        override fun close() {
            stop()
        }
    }
}

private class AndroidResourceSource(context: Context) {
    companion object { private const val PSS_INTERVAL_MS = 5_000L }

    private val activityManager = try {
        context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
    } catch (_: Exception) { null }
    private val powerManager = try {
        context.getSystemService(Context.POWER_SERVICE) as? PowerManager
    } catch (_: Exception) { null }
    private val processId = Process.myPid()
    private val gpuClockFile = File("/sys/class/kgsl/kgsl-3d0/gpuclk")
    private var lastPssBucket = -1L

    fun read(elapsedMs: Long): GenUiResourceSample {
        val gpuClock = try {
            gpuClockFile.bufferedReader(Charsets.US_ASCII).use { parseGpuClockHz(it.readLine()) }
        } catch (_: Exception) {
            null
        }
        val memory = try {
            activityManager?.let { manager ->
                ActivityManager.MemoryInfo().also(manager::getMemoryInfo)
            }
        } catch (_: Exception) {
            null
        }
        val pssBucket = elapsedMs / PSS_INTERVAL_MS
        val attemptPss = pssBucket != lastPssBucket
        if (attemptPss) lastPssBucket = pssBucket
        val pss = if (attemptPss) {
            try {
                activityManager?.getProcessMemoryInfo(intArrayOf(processId))
                    ?.firstOrNull()?.totalPss?.takeIf { it >= 0 }
            } catch (_: Exception) {
                null
            }
        } else null
        val thermal = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            try {
                powerManager?.currentThermalStatus
            } catch (_: Exception) {
                null
            }
        } else {
            null
        }
        return GenUiResourceSample(
            elapsedMs = elapsedMs,
            gpuClockHz = gpuClock,
            gpuClockSource = gpuClock?.let { gpuClockFile.path },
            availableMemoryBytes = memory?.availMem,
            totalMemoryBytes = memory?.totalMem,
            lowMemoryThresholdBytes = memory?.threshold,
            lowMemory = memory?.lowMemory,
            processPssKiB = pss,
            thermalStatus = thermal,
            processPssAttempted = attemptPss,
        )
    }
}
