package com.samsung.genuicraft.sdk.telemetry

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class GenUiResourceSummaryTest {
    @Test fun `GPU clock parser accepts plain positive sysfs hertz only`() {
        assertEquals(749_000_000L, parseGpuClockHz(" 749000000\n"))
        listOf<String?>(null, "", "0", "-1", "749 MHz", "1.5", "NaN",
            "99999999999999999999999999999").forEach { assertNull(parseGpuClockHz(it)) }
    }

    @Test fun `sampling is bounded and reports clock memory thermal and coverage separately`() {
        var nowMs = 10_000L
        val source = listOf(
            GenUiResourceSample(0, 600_000_000L, "gpuclk", 1_000L, 2_000L, 200L,
                false, 120, 0, true),
            GenUiResourceSample(0, null, null, 800L, 2_000L, 200L,
                true, null, 2),
            GenUiResourceSample(0, 400_000_000L, "gpuclk", null, null, null,
                null, 180, null, true),
        )
        var next = 0
        val accumulator = ResourceSampleAccumulator(
            requestId = "case-1",
            runtimeBackend = "GPU",
            clockMs = { nowMs },
            sampleAt = { source[next++] },
            maxSamples = 3,
        )

        assertTrue(accumulator.capture())
        nowMs += 1_000
        assertTrue(accumulator.capture())
        nowMs += 1_000
        assertFalse(accumulator.capture()) // Third sample reaches the cap.
        nowMs += 2_000
        assertFalse(accumulator.capture())

        val result = accumulator.finish()
        assertEquals("case-1", result.requestId)
        assertEquals("GPU", result.runtimeBackend)
        assertEquals(4_000L, result.elapsedMs)
        assertEquals(3, result.sampleCount)
        assertEquals(5L, result.expectedSampleCount)
        assertEquals(0.6, result.sampleCoverage, 0.0001)
        assertEquals(listOf(0L, 1_000L, 2_000L), result.samples.map { it.elapsedMs })
        assertEquals(2, result.gpuClockReadableSamples)
        assertEquals(2, result.systemMemoryReadableSamples)
        assertEquals(2, result.processPssReadableSamples)
        assertEquals(2, result.processPssAttemptedSamples)
        assertEquals(2, result.thermalReadableSamples)
        assertEquals(400_000_000L, result.gpuClockHzMin)
        assertEquals(500_000_000.0, result.gpuClockHzMedian!!, 0.1)
        assertEquals(600_000_000L, result.gpuClockHzMax)
        assertEquals(800L, result.availableMemoryBytesMin)
        assertEquals(180, result.processPssKiBMax)
        assertEquals(1, result.lowMemorySampleCount)
        assertEquals(2, result.peakThermalStatus)

        nowMs += 1_000
        assertSame(result, accumulator.finish())
        assertFalse(accumulator.capture())
    }

    @Test fun `unreadable observations remain unknown and sampling failure does not break request`() {
        var nowMs = 0L
        val accumulator = ResourceSampleAccumulator(
            requestId = "case-2",
            runtimeBackend = "CPU",
            clockMs = { nowMs },
            sampleAt = { throw IllegalStateException("permission denied") },
        )
        assertTrue(accumulator.capture())
        nowMs = 2_000L
        val result = accumulator.finish()
        assertEquals(1, result.sampleCount)
        assertEquals(3L, result.expectedSampleCount)
        assertEquals(1.0 / 3.0, result.sampleCoverage, 0.0001)
        assertEquals(0, result.gpuClockReadableSamples)
        assertEquals(0, result.systemMemoryReadableSamples)
        assertEquals(0, result.processPssReadableSamples)
        assertEquals(0, result.thermalReadableSamples)
        assertNull(result.gpuClockHzMin)
        assertNull(result.availableMemoryBytesMin)
        assertNull(result.processPssKiBMax)
        assertNull(result.peakThermalStatus)
    }

    @Test fun `stop does not wait for an in flight device read`() {
        val readStarted = CountDownLatch(1)
        val releaseRead = CountDownLatch(1)
        val stopFinished = CountDownLatch(1)
        val accumulator = ResourceSampleAccumulator(
            requestId = "case-3",
            runtimeBackend = "GPU",
            clockMs = { 0L },
            sampleAt = {
                readStarted.countDown()
                releaseRead.await()
                GenUiResourceSample(0L, gpuClockHz = 500_000_000L)
            },
        )
        val sampler = Thread { accumulator.capture() }
        sampler.start()
        try {
            assertTrue(readStarted.await(1, TimeUnit.SECONDS))
            var result: GenUiResourceSummary? = null
            val stopper = Thread {
                result = accumulator.finish()
                stopFinished.countDown()
            }
            stopper.start()
            assertTrue(stopFinished.await(1, TimeUnit.SECONDS))
            assertEquals(0, result?.sampleCount)
            stopper.join(1_000)
        } finally {
            releaseRead.countDown()
            sampler.join(1_000)
        }
        assertEquals(0, accumulator.finish().sampleCount)
    }
}
