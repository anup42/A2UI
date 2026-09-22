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
    val nativePrefillTokensPerSecond: Double? = null,
    val nativeTimeToFirstTokenSeconds: Double? = null,
    val nativeEngineInitializationSeconds: Double? = null,
    val engineInitializedForRequest: Boolean? = null,
    val nativeInitializationPhaseSeconds: Double? = null,
) {
    val requestAverageTokensPerSecond: Double?
        get() {
            val tokens = outputTokens?.takeIf { it >= 0 } ?: return null
            val nanos = providerCallElapsedNanos?.takeIf { it > 0L } ?: return null
            return tokens.toDouble() * 1_000_000_000.0 / nanos.toDouble()
        }

    val nativePrefillElapsedNanos: Long?
        get() = durationNanos(inputTokens, nativePrefillTokensPerSecond)

    val nativeDecodeElapsedNanos: Long?
        get() = durationNanos(outputTokens, nativeDecodeTokensPerSecond)

    val nativeEngineInitializationNanos: Long?
        get() = when (engineInitializedForRequest) {
            false -> 0L
            true -> secondsToNanos(nativeEngineInitializationSeconds)
            null -> secondsToNanos(nativeEngineInitializationSeconds)
        }

    val nativeTimeToFirstTokenNanos: Long?
        get() = secondsToNanos(nativeTimeToFirstTokenSeconds)

    val nativeInitializationPhaseNanos: Long?
        get() = secondsToNanos(nativeInitializationPhaseSeconds)

    /** Provider wall time not represented by native init, prefill, or decode compute counters. */
    val providerResidualNanos: Long?
        get() {
            val wall = providerCallElapsedNanos?.takeIf { it >= 0L } ?: return null
            val initialization = nativeEngineInitializationNanos ?: return null
            val prefill = nativePrefillElapsedNanos ?: return null
            val decode = nativeDecodeElapsedNanos ?: return null
            val accounted = safeNanosSum(initialization, prefill, decode) ?: return null
            return (wall - accounted).coerceAtLeast(0L)
        }
}

internal data class GenerationMetricsUiState(
    val attempts: List<GenerationAttemptUiMetrics>,
    val reportedAttempts: Int,
    val conversionElapsedMs: Long,
    val speculativeDecodingEnabled: Boolean? = null,
    val drafterAcceptanceRate: Double? = null,
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

    val totalProviderCallElapsedNanos: Long?
        get() = completeDurationTotal { it.providerCallElapsedNanos }

    val totalEngineInitializationNanos: Long?
        get() = completeDurationTotal { it.nativeEngineInitializationNanos }

    val totalPrefillElapsedNanos: Long?
        get() = completeDurationTotal { it.nativePrefillElapsedNanos }

    val totalDecodeElapsedNanos: Long?
        get() = completeDurationTotal { it.nativeDecodeElapsedNanos }

    val totalProviderResidualNanos: Long?
        get() = completeDurationTotal { it.providerResidualNanos }

    /** Prompt assembly, compilation, validation and generated-DSL recovery outside provider calls. */
    val validationAndRecoveryElapsedNanos: Long?
        get() {
            val providerNanos = totalProviderCallElapsedNanos ?: return null
            val conversionNanos = conversionElapsedMs.takeIf { it >= 0L }
                ?.let { safeMillisToNanos(it) } ?: return null
            return (conversionNanos - providerNanos).coerceAtLeast(0L)
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

    private inline fun completeDurationTotal(
        value: (GenerationAttemptUiMetrics) -> Long?,
    ): Long? {
        if (attempts.isEmpty() || attempts.size != reportedAttempts) return null
        var total = 0L
        attempts.forEach { attempt ->
            val nanos = value(attempt)?.takeIf { it >= 0L } ?: return null
            total = safeNanosSum(total, nanos) ?: return null
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
        nativePrefillTokensPerSecond = reported?.prefillTokensPerSecond
            ?.takeIf { it.isFinite() && it > 0.0 },
        nativeTimeToFirstTokenSeconds = reported?.timeToFirstTokenSeconds
            ?.takeIf { it.isFinite() && it >= 0.0 },
        nativeEngineInitializationSeconds = reported?.engineInitializationSeconds
            ?.takeIf { it.isFinite() && it >= 0.0 },
        engineInitializedForRequest = reported?.engineInitializedForRequest,
        nativeInitializationPhaseSeconds = reported?.nativeInitializationPhaseSeconds
            ?.takeIf { it.isFinite() && it >= 0.0 },
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

internal fun formatPercentage(value: Double?): String =
    value?.takeIf { it.isFinite() && it in 0.0..1.0 }
        ?.let { String.format(Locale.US, "%.2f%%", it * 100.0) }
        ?: "unavailable"

private fun durationNanos(tokens: Int?, tokensPerSecond: Double?): Long? {
    val count = tokens?.takeIf { it >= 0 } ?: return null
    val rate = tokensPerSecond?.takeIf { it.isFinite() && it > 0.0 } ?: return null
    return secondsToNanos(count.toDouble() / rate)
}

private fun secondsToNanos(seconds: Double?): Long? = seconds
    ?.takeIf { it.isFinite() && it >= 0.0 && it <= Long.MAX_VALUE / 1_000_000_000.0 }
    ?.let { (it * 1_000_000_000.0).toLong() }

private fun safeMillisToNanos(milliseconds: Long): Long? =
    if (milliseconds > Long.MAX_VALUE / 1_000_000L) null else milliseconds * 1_000_000L

private fun safeNanosSum(vararg values: Long): Long? {
    var total = 0L
    values.forEach { value ->
        if (value < 0L || Long.MAX_VALUE - total < value) return null
        total += value
    }
    return total
}
