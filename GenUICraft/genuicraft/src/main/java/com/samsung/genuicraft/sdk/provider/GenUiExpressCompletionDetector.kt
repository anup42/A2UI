package com.samsung.genuicraft.sdk.provider

import com.samsung.genuicraft.sdk.GenUiCompiler
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressStreamScanner

/**
 * Recognizes a complete, strictly compilable Express document in cumulative native output.
 * The scanner distinguishes real document delimiters from text in strings and comments.
 * A Markdown fence or other prefix is deliberately ineligible for early completion.
 */
internal class GenUiExpressCompletionDetector {
    private var nextCloseSearchOffset = 0

    fun inspect(cumulativeText: String): Boolean {
        if (cumulativeText.length > A2uiExpressCodec.MAX_INPUT_CHARS) return false
        val openAt = cumulativeText.indexOfFirst { !it.isWhitespace() }
        if (openAt < 0 || !cumulativeText.startsWith(OPEN, openAt)) return false

        // Search the overlap so a closing delimiter split across callbacks is found once complete.
        val closeAt = cumulativeText.indexOf(CLOSE, nextCloseSearchOffset)
        nextCloseSearchOffset = (cumulativeText.length - CLOSE.length + 1).coerceAtLeast(0)
        if (closeAt < 0) return false

        val closed = runCatching {
            A2uiExpressStreamScanner.scan(cumulativeText, openAt + OPEN.length).closed
        }.getOrDefault(false)
        return closed && runCatching { GenUiCompiler.compile(cumulativeText) }.isSuccess
    }

    private companion object {
        const val OPEN = "<a2ui>"
        const val CLOSE = "</a2ui>"
    }
}
