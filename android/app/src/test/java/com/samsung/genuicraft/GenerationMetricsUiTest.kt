package com.samsung.genuicraft

import com.samsung.genuicraft.sdk.GenUiGenerationMetrics
import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.telemetry.GenUiResourceSample
import com.samsung.genuicraft.sdk.telemetry.GenUiResourceSummary
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class GenerationMetricsUiTest {
    @Test
    fun modelOutputUsesReportedMetricsAndLegacyOutputFallbackWithoutEstimating() {
        val output = GenUiModelOutput(
            text = "fixture",
            runtime = "Gemma/test",
            outputTokens = 80,
            metrics = GenUiGenerationMetrics(
                inputTokens = 120,
                outputTokens = null,
                decodeTokensPerSecond = null,
            ),
        )

        val metrics = output.toUiMetrics(providerCallElapsedNanos = 2_000_000_000L)

        assertEquals(120, metrics.inputTokens)
        assertEquals(80, metrics.outputTokens)
        assertNull(metrics.nativeDecodeTokensPerSecond)
        assertEquals(40.0, metrics.requestAverageTokensPerSecond!!, 0.0001)
    }

    @Test
    fun retrySummaryUsesTokenWeightedNativeDecodeAndMeasuredRequestWallTime() {
        val summary = GenerationMetricsUiState(
            attempts = listOf(
                GenerationAttemptUiMetrics(100, 100, 50.0, 2_000_000_000L),
                GenerationAttemptUiMetrics(200, 300, 100.0, 4_000_000_000L),
            ),
            reportedAttempts = 2,
            conversionElapsedMs = 7_000,
        )

        assertEquals(300L, summary.totalInputTokens)
        assertEquals(400L, summary.totalOutputTokens)
        assertEquals(80.0, summary.weightedNativeDecodeTokensPerSecond!!, 0.0001)
        assertEquals(400.0 / 6.0, summary.requestAverageTokensPerSecond!!, 0.0001)
    }

    @Test
    fun incompleteAttemptMetricsDoNotProduceMisleadingTotalsOrRates() {
        val summary = GenerationMetricsUiState(
            attempts = listOf(
                GenerationAttemptUiMetrics(null, 80, null, 2_000_000_000L),
                GenerationAttemptUiMetrics(90, null, 45.0, 1_000_000_000L),
            ),
            reportedAttempts = 2,
            conversionElapsedMs = 3_500,
        )

        assertNull(summary.totalInputTokens)
        assertNull(summary.totalOutputTokens)
        assertNull(summary.weightedNativeDecodeTokensPerSecond)
        assertNull(summary.requestAverageTokensPerSecond)
    }

    @Test
    fun conversionBreakdownReconcilesNativePhasesProviderWallAndPostProcessing() {
        val output = GenUiModelOutput(
            text = "fixture",
            runtime = "Gemma/test/GPU+MTP",
            metrics = GenUiGenerationMetrics(
                inputTokens = 3_174,
                outputTokens = 288,
                decodeTokensPerSecond = 57.6,
                prefillTokensPerSecond = 1_587.0,
                timeToFirstTokenSeconds = 4.0,
                engineInitializationSeconds = 2.0,
                engineInitializedForRequest = true,
            ),
        )
        val attempt = output.toUiMetrics(providerCallElapsedNanos = 10_000_000_000L)
        val summary = GenerationMetricsUiState(
            attempts = listOf(attempt),
            reportedAttempts = 1,
            conversionElapsedMs = 10_100,
            speculativeDecodingEnabled = true,
            drafterAcceptanceRate = 0.558642,
        )

        assertEquals(2_000_000_000L, summary.totalEngineInitializationNanos)
        assertEquals(2_000_000_000L, summary.totalPrefillElapsedNanos)
        assertEquals(5_000_000_000L, summary.totalDecodeElapsedNanos)
        assertEquals(1_000_000_000L, summary.totalProviderResidualNanos)
        assertEquals(100_000_000L, summary.validationAndRecoveryElapsedNanos)
        assertEquals("55.86%", formatPercentage(summary.drafterAcceptanceRate))
    }

    @Test
    fun drafterAcceptanceStaysPendingUntilEngineRetires() {
        val pending = GenerationMetricsUiState(
            attempts = emptyList(),
            reportedAttempts = 0,
            conversionElapsedMs = 0,
            speculativeDecodingEnabled = true,
            pendingRetirement = true,
            acceptanceScope = "engine_lifetime",
        )
        assertEquals("MTP acceptance: pending engine retirement", pending.drafterAcceptanceSummary())
        assertEquals(
            "Drafter acceptance: pending engine retirement. The native counter covers the engine lifetime.",
            pending.drafterAcceptanceDetail(),
        )

        val retired = pending.copy(pendingRetirement = false, drafterAcceptanceRate = 0.558642)
        assertEquals("MTP acceptance (engine lifetime): 55.86%", retired.drafterAcceptanceSummary())
        assertEquals(
            "Drafter acceptance: 55.86% (verified draft tokens / proposed draft tokens across the engine lifetime).",
            retired.drafterAcceptanceDetail(),
        )
    }

    @Test
    fun resourceSummaryPreservesCoverageReadabilityAndMeasuredUnits() {
        val resources = resourceSummary()
        val output = GenUiModelOutput(
            text = "fixture",
            runtime = "LiteRT-LM/Gemma4/GPU",
            metrics = GenUiGenerationMetrics(
                inputTokens = 20,
                outputTokens = 10,
                decodeTokensPerSecond = 5.0,
                resources = resources,
            ),
        )
        val attempt = output.toUiMetrics()
        assertEquals(resources, attempt.resources)
        assertEquals(
            listOf(
                "Resource samples: 4/6 expected (66.67% coverage; 1000 ms interval)",
                "GPU clock: 400.0 / 461.5 / 500.0 MHz min / median / max (3/4 readable)",
                "Process PSS peak: 256.0 MiB (4/4 reads); lowest available RAM: 2048.0 MiB (4/4 readable)",
                "Thermal status max: 3 (2/4 readable); low-memory: 1/4 memory-readable samples",
            ),
            resources.toUiLines(),
        )
    }

    @Test
    fun unreadableResourceSourcesStayUnavailable() {
        val resources = resourceSummary().copy(
            gpuClockReadableSamples = 0,
            systemMemoryReadableSamples = 0,
            processPssReadableSamples = 0,
            thermalReadableSamples = 0,
            gpuClockHzMin = null,
            gpuClockHzMedian = null,
            gpuClockHzMax = null,
            availableMemoryBytesMin = null,
            processPssKiBMax = null,
            lowMemorySampleCount = 0,
            peakThermalStatus = null,
        )
        assertEquals(
            listOf(
                "Resource samples: 4/6 expected (66.67% coverage; 1000 ms interval)",
                "GPU clock: unavailable (0/4 readable)",
                "Process PSS peak: unavailable (0/4 reads); lowest available RAM: unavailable (0/4 readable)",
                "Thermal status max: unavailable (0/4 readable); low-memory: unavailable",
            ),
            resources.toUiLines(),
        )
    }

    private fun resourceSummary() = GenUiResourceSummary(
        requestId = "test-request",
        runtimeBackend = "LiteRT-LM/Gemma4/GPU",
        elapsedMs = 5_000,
        sampleIntervalMs = 1_000,
        maxSamples = 256,
        sampleCount = 4,
        expectedSampleCount = 6,
        sampleCoverage = 4.0 / 6.0,
        gpuClockReadableSamples = 3,
        systemMemoryReadableSamples = 4,
        processPssReadableSamples = 4,
        processPssAttemptedSamples = 4,
        thermalReadableSamples = 2,
        gpuClockHzMin = 400_000_000L,
        gpuClockHzMedian = 461_500_000.0,
        gpuClockHzMax = 500_000_000L,
        availableMemoryBytesMin = 2_147_483_648L,
        processPssKiBMax = 262_144,
        lowMemorySampleCount = 1,
        peakThermalStatus = 3,
        samples = List(4) { index -> GenUiResourceSample(elapsedMs = index * 1_000L) },
    )
}
