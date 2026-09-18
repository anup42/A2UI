package com.samsung.genuicraft.sdk.provider

import com.samsung.genuicraft.sdk.GenUiGenerationMetrics
import com.samsung.genuicraft.sdk.GenUiModelOutput
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class Gemma4MetricsTest {
    @Test fun `reads exact pinned native getter values without deriving counts from response text`() {
        val result = readGemma4GenerationMetrics(ConversationSnapshot(BenchmarkSnapshot(719, 127L, 23.75)))
        assertEquals(GenUiGenerationMetrics(719, 127, 23.75), result)
    }

    @Test fun `missing benchmark or inaccessible getter leaves telemetry unavailable`() {
        assertNull(readGemma4GenerationMetrics(Any()))
        assertNull(readGemma4GenerationMetrics(ThrowingConversation()))
        assertNull(readGemma4GenerationMetrics(ConversationSnapshot(Any())))
    }

    @Test fun `partial native telemetry preserves only available actual values`() {
        assertEquals(GenUiGenerationMetrics(null, 45, null),
            readGemma4GenerationMetrics(ConversationSnapshot(DecodeCountOnly())))
        assertEquals(GenUiGenerationMetrics(719, null, 22.5),
            readGemma4GenerationMetrics(ConversationSnapshot(BenchmarkSnapshot(719, null, 22.5))))
    }

    @Test fun `invalid token counts are rejected instead of rounded clamped or estimated`() {
        listOf<Any?>(null, 0, -1, 1.5, Double.NaN, Double.POSITIVE_INFINITY,
            Int.MAX_VALUE.toLong() + 1L, "127").forEach { count ->
            assertEquals(GenUiGenerationMetrics(null, null, 22.5),
                readGemma4GenerationMetrics(ConversationSnapshot(BenchmarkSnapshot(count, count, 22.5))))
        }
        assertEquals(GenUiGenerationMetrics(Int.MAX_VALUE, 1, null),
            readGemma4GenerationMetrics(ConversationSnapshot(BenchmarkSnapshot(Int.MAX_VALUE, 1, null))))
    }

    @Test fun `native throughput must be finite and positive`() {
        listOf<Any?>(null, 0.0, -1.0, Double.NaN, Double.POSITIVE_INFINITY,
            Double.NEGATIVE_INFINITY, "22.5").forEach { rate ->
            assertEquals(GenUiGenerationMetrics(719, 127, null),
                readGemma4GenerationMetrics(ConversationSnapshot(BenchmarkSnapshot(719, 127, rate))))
        }
        assertNull(readGemma4GenerationMetrics(ConversationSnapshot(BenchmarkSnapshot(0, 0, Double.NaN))))
    }

    @Test fun `existing model output constructor and token property remain usable`() {
        val result = GenUiModelOutput("result", "runtime", 127)
        assertEquals(127, result.outputTokens)
        assertNull(result.metrics)
    }

    // These public Java-style getters match LiteRT-LM 0.15.0's metadata-hidden benchmark API.
    class ConversationSnapshot(private val benchmark: Any) {
        fun getBenchmarkInfo(): Any = benchmark
    }

    class ThrowingConversation {
        fun getBenchmarkInfo(): Any = error("Native benchmark unavailable")
    }

    class BenchmarkSnapshot(
        val lastPrefillTokenCount: Any?,
        val lastDecodeTokenCount: Any?,
        val lastDecodeTokensPerSecond: Any?,
    )

    class DecodeCountOnly {
        fun getLastDecodeTokenCount(): Int = 45
    }
}
