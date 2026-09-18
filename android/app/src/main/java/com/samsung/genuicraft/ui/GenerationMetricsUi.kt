package com.samsung.genuicraft

import com.samsung.genuicraft.sdk.GenUiModelOutput
import java.util.Locale

const val TOKEN_METRICS_SWITCH_TEST_TAG = "token_metrics_switch"
const val GENERATION_METRICS_PANEL_TEST_TAG = "generation_metrics_panel"
const val GENERATION_METRICS_ATTEMPT_TEST_TAG_PREFIX = "generation_metrics_attempt_"

internal data class GenerationAttemptUiMetrics(
    val inputTokens: Int?,
    val outputTokens: Int?,
    val nativeDecodeTokensPerSecond: Double?,
    val providerCallElapsedNanos: Long?,
) {
    val requestAverageTokensPerSecond: Double?
        get() {
            val tokens = outputTokens?.takeIf { it >= 0 } ?: return null
            val nanos = providerCallElapsedNanos?.takeIf { it > 0L } ?: return null
            return tokens.toDouble() * 1_000_000_000.0 / nanos.toDouble()
        }
}

internal data class GenerationMetricsUiState(
    val attempts: List<GenerationAttemptUiMetrics>,
    val reportedAttempts: Int,
    val conversionElapsedMs: Long,
) {
    val totalInputTokens: Long?
        get() = completeTokenTotal { it.inputTokens }

    val totalOutputTokens: Long?
        get() = completeTokenTotal { it.outputTokens }

    /** Actual output tokens divided by measured provider-call wall time across all attempts. */
    val requestAverageTokensPerSecond: Double?
        get() {
            if (attempts.isEmpty() || attempts.size != reportedAttempts) return null
            var outputTokens = 0L
            var elapsedNanos = 0L
            attempts.forEach { attempt ->
                val tokens = attempt.outputTokens?.takeIf { it >= 0 } ?: return null
                val nanos = attempt.providerCallElapsedNanos?.takeIf { it > 0L } ?: return null
                outputTokens += tokens
                if (Long.MAX_VALUE - elapsedNanos < nanos) return null
                elapsedNanos += nanos
            }
            return outputTokens.toDouble() * 1_000_000_000.0 / elapsedNanos.toDouble()
        }

    /** Token-weighted native decode rate across attempts; never inferred from wall time. */
    val weightedNativeDecodeTokensPerSecond: Double?
        get() {
            if (attempts.isEmpty() || attempts.size != reportedAttempts) return null
            var outputTokens = 0L
            var decodeSeconds = 0.0
            attempts.forEach { attempt ->
                val tokens = attempt.outputTokens?.takeIf { it > 0 } ?: return null
                val rate = attempt.nativeDecodeTokensPerSecond?.takeIf { it.isFinite() && it > 0.0 }
                    ?: return null
                outputTokens += tokens
                decodeSeconds += tokens.toDouble() / rate
            }
            return if (decodeSeconds.isFinite() && decodeSeconds > 0.0) {
                outputTokens.toDouble() / decodeSeconds
            } else {
                null
            }
        }

    private inline fun completeTokenTotal(value: (GenerationAttemptUiMetrics) -> Int?): Long? {
        if (attempts.isEmpty() || attempts.size != reportedAttempts) return null
        var total = 0L
        attempts.forEach { attempt ->
            val count = value(attempt)?.takeIf { it >= 0 } ?: return null
            total += count
        }
        return total
    }
}

internal fun GenUiModelOutput.toUiMetrics(providerCallElapsedNanos: Long? = null): GenerationAttemptUiMetrics {
    val reported = metrics
    return GenerationAttemptUiMetrics(
        inputTokens = reported?.inputTokens?.takeIf { it >= 0 },
        outputTokens = (reported?.outputTokens ?: this.outputTokens)?.takeIf { it >= 0 },
        nativeDecodeTokensPerSecond = reported?.decodeTokensPerSecond
            ?.takeIf { it.isFinite() && it > 0.0 },
        providerCallElapsedNanos = providerCallElapsedNanos?.takeIf { it >= 0L },
    )
}

internal fun formatTokenCount(value: Number?): String = value?.toString() ?: "unavailable"

internal fun formatTokensPerSecond(value: Double?): String =
    value?.takeIf { it.isFinite() && it >= 0.0 }
        ?.let { String.format(Locale.US, "%.2f token/s", it) }
        ?: "unavailable"

internal fun formatElapsedNanos(value: Long?): String = value?.takeIf { it >= 0L }
    ?.let { formatElapsedMillis(it / 1_000_000.0) }
    ?: "unavailable"

internal fun formatElapsedMillis(value: Long): String = formatElapsedMillis(value.toDouble())

private fun formatElapsedMillis(value: Double): String = when {
    value < 1_000.0 -> String.format(Locale.US, "%.0f ms", value)
    else -> String.format(Locale.US, "%.2f s", value / 1_000.0)
}
