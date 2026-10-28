package com.samsung.genuicraft.sdk.telemetry

import java.util.Collections

/** One read-only observation, timed from the start of a generation request. */
data class GenUiResourceSample(
    val elapsedMs: Long,
    /** Current GPU frequency in hertz, when a readable GPU sysfs source exists. */
    val gpuClockHz: Long? = null,
    val gpuClockSource: String? = null,
    val availableMemoryBytes: Long? = null,
    val totalMemoryBytes: Long? = null,
    val lowMemoryThresholdBytes: Long? = null,
    val lowMemory: Boolean? = null,
    /** PSS for this app process, in kibibytes. */
    val processPssKiB: Int? = null,
    /** Android PowerManager thermal status; null on unsupported or unreadable devices. */
    val thermalStatus: Int? = null,
    /** True only on a tick that requested the slower process-PSS read. */
    val processPssAttempted: Boolean = false,
)

/**
 * Descriptive request telemetry. Clock and thermal observations alone cannot establish the cause
 * of a slowdown. GPU, system-memory, and thermal readability counts use [sampleCount] as their
 * denominator. PSS is sampled less frequently: compare [processPssReadableSamples] with
 * [processPssAttemptedSamples]. [sampleCoverage] uses the number of one-second samples expected
 * over the entire request as its denominator.
 */
data class GenUiResourceSummary(
    val requestId: String,
    val runtimeBackend: String,
    val elapsedMs: Long,
    val sampleIntervalMs: Long,
    val maxSamples: Int,
    val sampleCount: Int,
    val expectedSampleCount: Long,
    val sampleCoverage: Double,
    val gpuClockReadableSamples: Int,
    val systemMemoryReadableSamples: Int,
    val processPssReadableSamples: Int,
    val thermalReadableSamples: Int,
    val gpuClockHzMin: Long?,
    val gpuClockHzMedian: Double?,
    val gpuClockHzMax: Long?,
    val availableMemoryBytesMin: Long?,
    val processPssKiBMax: Int?,
    val lowMemorySampleCount: Int,
    val peakThermalStatus: Int?,
    val samples: List<GenUiResourceSample>,
    val processPssAttemptedSamples: Int = 0,
)

/** Only a positive plain integer is a usable GPU clock; zero and read errors remain unknown. */
internal fun parseGpuClockHz(raw: String?): Long? {
    val value = raw?.trim() ?: return null
    if (value.isEmpty() || !value.all { it in '0'..'9' }) return null
    return value.toLongOrNull()?.takeIf { it > 0L }
}

/** Pure Kotlin accumulation so scheduling and Android reads can be tested independently. */
internal class ResourceSampleAccumulator(
    private val requestId: String,
    private val runtimeBackend: String,
    private val clockMs: () -> Long,
    private val sampleAt: (elapsedMs: Long) -> GenUiResourceSample,
    private val sampleIntervalMs: Long = 1_000L,
    private val maxSamples: Int = 256,
) {
    init {
        require(sampleIntervalMs > 0L)
        require(maxSamples > 0)
    }

    private val startedMs = clockMs()
    private val samples = ArrayList<GenUiResourceSample>(minOf(maxSamples, 256))
    private var finished: GenUiResourceSummary? = null

    /** Captures at most one observation. False means no more ticks should be scheduled. */
    fun capture(): Boolean {
        synchronized(this) {
            if (finished != null || samples.size >= maxSamples) return false
        }
        val elapsedMs = (clockMs() - startedMs).coerceAtLeast(0L)
        val sample = try {
            sampleAt(elapsedMs).copy(elapsedMs = elapsedMs)
        } catch (_: Exception) {
            GenUiResourceSample(elapsedMs)
        }
        // The device read may use Binder. Do not hold the lock while it runs: stop() must not
        // wait for a slow or stalled PSS read, and a sample finishing after stop is discarded.
        synchronized(this) {
            if (finished != null || samples.size >= maxSamples) return false
            samples.add(sample)
            return samples.size < maxSamples
        }
    }

    @Synchronized
    fun finish(): GenUiResourceSummary {
        finished?.let { return it }
        val elapsedMs = (clockMs() - startedMs).coerceAtLeast(0L)
        val snapshot = Collections.unmodifiableList(ArrayList(samples))
        val gpuClocks = snapshot.mapNotNull { it.gpuClockHz }.sorted()
        val expected = elapsedMs / sampleIntervalMs + 1L
        val median = when {
            gpuClocks.isEmpty() -> null
            gpuClocks.size % 2 == 1 -> gpuClocks[gpuClocks.size / 2].toDouble()
            else -> (gpuClocks[gpuClocks.size / 2 - 1].toDouble() +
                gpuClocks[gpuClocks.size / 2].toDouble()) / 2.0
        }
        return GenUiResourceSummary(
            requestId = requestId,
            runtimeBackend = runtimeBackend,
            elapsedMs = elapsedMs,
            sampleIntervalMs = sampleIntervalMs,
            maxSamples = maxSamples,
            sampleCount = snapshot.size,
            expectedSampleCount = expected,
            sampleCoverage = (snapshot.size.toDouble() / expected).coerceIn(0.0, 1.0),
            gpuClockReadableSamples = gpuClocks.size,
            systemMemoryReadableSamples = snapshot.count { it.availableMemoryBytes != null },
            processPssReadableSamples = snapshot.count { it.processPssKiB != null },
            thermalReadableSamples = snapshot.count { it.thermalStatus != null },
            gpuClockHzMin = gpuClocks.firstOrNull(),
            gpuClockHzMedian = median,
            gpuClockHzMax = gpuClocks.lastOrNull(),
            availableMemoryBytesMin = snapshot.mapNotNull { it.availableMemoryBytes }.minOrNull(),
            processPssKiBMax = snapshot.mapNotNull { it.processPssKiB }.maxOrNull(),
            lowMemorySampleCount = snapshot.count { it.lowMemory == true },
            peakThermalStatus = snapshot.mapNotNull { it.thermalStatus }.maxOrNull(),
            samples = snapshot,
            processPssAttemptedSamples = snapshot.count { it.processPssAttempted },
        ).also { finished = it }
    }
}
