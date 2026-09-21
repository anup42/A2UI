package com.samsung.genuicraft.sdk.provider

import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags

/**
 * Serializes access to LiteRT-LM's process-global experimental flags.
 *
 * Engine initialization and conversation creation must use the same lock across every provider;
 * otherwise one model can observe another view's MTP, benchmark, or prompt-template settings.
 */
@OptIn(ExperimentalApi::class)
internal object LiteRtRuntimeFlags {
    private val lock = Any()

    fun <T> withEngineFlags(
        speculativeDecoding: Boolean,
        benchmark: Boolean? = null,
        block: () -> T,
    ): T = synchronized(lock) {
        val previousMtp = ExperimentalFlags.enableSpeculativeDecoding
        val previousBenchmark = ExperimentalFlags.enableBenchmark
        try {
            ExperimentalFlags.enableSpeculativeDecoding = speculativeDecoding
            if (benchmark != null) ExperimentalFlags.enableBenchmark = benchmark
            block()
        } finally {
            ExperimentalFlags.enableSpeculativeDecoding = previousMtp
            ExperimentalFlags.enableBenchmark = previousBenchmark
        }
    }

    fun <T> withPromptTemplate(template: String?, block: () -> T): T = synchronized(lock) {
        val previousTemplate = ExperimentalFlags.overwritePromptTemplate
        try {
            ExperimentalFlags.overwritePromptTemplate = template
            block()
        } finally {
            ExperimentalFlags.overwritePromptTemplate = previousTemplate
        }
    }
}
