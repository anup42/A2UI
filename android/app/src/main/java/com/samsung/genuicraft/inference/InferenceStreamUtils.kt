package com.samsung.genuicraft.inference

import java.io.InputStream

internal object InferenceStreamUtils {

    data class StreamReadResult(
        val text: String,
        val streamDurationMs: Long?
    )

    fun readStreamWithTiming(stream: InputStream?): StreamReadResult {
        if (stream == null) {
            return StreamReadResult(text = "", streamDurationMs = null)
        }
        var firstChunkAtMs: Long? = null
        val builder = StringBuilder()
        stream.bufferedReader(Charsets.UTF_8).use { reader ->
            val buffer = CharArray(4096)
            while (true) {
                val read = reader.read(buffer)
                if (read <= 0) {
                    break
                }
                if (firstChunkAtMs == null) {
                    firstChunkAtMs = System.currentTimeMillis()
                }
                builder.append(buffer, 0, read)
            }
        }
        val streamDurationMs = firstChunkAtMs?.let { start ->
            (System.currentTimeMillis() - start).coerceAtLeast(0L)
        }
        return StreamReadResult(
            text = builder.toString(),
            streamDurationMs = streamDurationMs
        )
    }
}
