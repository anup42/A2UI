package com.samsung.genuicraft.pipeline

import com.google.gson.JsonElement
import com.google.gson.JsonParser

internal object PipelineJsonExtractor {

    fun extractJsonElement(text: String): JsonElement? {
        val cleaned = text.trim()
        if (cleaned.isEmpty()) {
            return null
        }

        extractFencedBlock(cleaned)?.let { fenced ->
            try {
                return JsonParser.parseString(fenced)
            } catch (_: Exception) {
            }
        }

        if (cleaned.startsWith("[") || cleaned.startsWith("{")) {
            try {
                return JsonParser.parseString(cleaned)
            } catch (_: Exception) {
            }
        }

        findFirstBalanced(cleaned, '[', ']')?.let { arrayCandidate ->
            try {
                return JsonParser.parseString(arrayCandidate)
            } catch (_: Exception) {
            }
        }

        findFirstBalanced(cleaned, '{', '}')?.let { objectCandidate ->
            try {
                return JsonParser.parseString(objectCandidate)
            } catch (_: Exception) {
            }
        }

        return null
    }

    fun extractFencedBlock(text: String): String? {
        val start = text.indexOf("```")
        if (start < 0) {
            return null
        }
        val end = text.indexOf("```", startIndex = start + 3)
        if (end <= start) {
            return null
        }
        val block = text.substring(start + 3, end).trim()
        return if (block.startsWith("json", ignoreCase = true)) {
            block.removePrefix("json").trim()
        } else {
            block
        }
    }

    fun findFirstBalanced(text: String, open: Char, close: Char): String? {
        var index = text.indexOf(open)
        while (index >= 0) {
            val candidate = balancedSubstring(text, index, open, close)
            if (candidate != null) {
                return candidate
            }
            index = text.indexOf(open, startIndex = index + 1)
        }
        return null
    }

    fun balancedSubstring(text: String, start: Int, open: Char, close: Char): String? {
        var depth = 0
        var inString = false
        var escape = false

        for (i in start until text.length) {
            val ch = text[i]
            if (escape) {
                escape = false
                continue
            }
            if (ch == '\\' && inString) {
                escape = true
                continue
            }
            if (ch == '"') {
                inString = !inString
                continue
            }
            if (inString) {
                continue
            }
            if (ch == open) {
                depth += 1
            } else if (ch == close) {
                depth -= 1
                if (depth == 0) {
                    return text.substring(start, i + 1)
                }
            }
        }
        return null
    }

    fun buildFlatSpecRepairPrompt(rawText: String, failureReason: String? = null): String {
        val reasonLine = failureReason?.trim()?.takeIf { it.isNotEmpty() }?.let {
            "Failure reason: $it\n"
        }.orEmpty()
        return (
            "The previous output does not satisfy the required flat-spec contract.\n" +
                reasonLine +
                "Return ONLY one valid JSON object with this shape:\n" +
                "{\"root\":\"<id>\",\"state\":{...},\"elements\":{...}}\n\n" +
                "Rules:\n" +
                "- Do NOT emit legacy v0.9 message arrays (`createSurface` / `updateComponents`).\n" +
                "- `root` must reference an existing key in `elements`.\n" +
                "- Every element must contain `type`, `props`, and `children`.\n" +
                "- Every id in `children` must exist in `elements`.\n" +
                "- Return JSON only, no markdown.\n\n" +
                "Original output:\n${rawText.trim()}"
            )
    }
}
