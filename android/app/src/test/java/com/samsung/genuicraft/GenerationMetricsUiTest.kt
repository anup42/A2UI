package com.samsung.genuicraft

import com.samsung.genuicraft.sdk.GenUiGenerationMetrics
import com.samsung.genuicraft.sdk.GenUiModelOutput
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class GenerationMetricsUiTest {
    @Test
    fun modelOutputUsesReportedMetricsAndLegacyOutputFallbackWithoutEstimating() {
        val output = GenUiModelOutput(
            text = "fixture",
            runtime = "Gauss/test",
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
}
