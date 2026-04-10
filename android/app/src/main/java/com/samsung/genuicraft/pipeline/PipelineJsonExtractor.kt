package com.samsung.genuicraft.pipeline

import com.google.gson.JsonElement
import com.google.gson.JsonParser

internal object PipelineJsonExtractor {

    fun extractJsonElement(text: String): JsonElement? {
        val cleaned = text.trim()
        if (cleaned.isEmpty()) {
            return null
        }

        val candidates = linkedSetOf<String>()
        extractFencedBlock(cleaned)?.let { candidates += it }
        if (cleaned.startsWith("[") || cleaned.startsWith("{")) {
            candidates += cleaned
        }
        candidates += collectBalancedCandidates(cleaned, '[', ']')
        candidates += collectBalancedCandidates(cleaned, '{', '}')

        val parsedCandidates = candidates.mapNotNull { candidate ->
            runCatching { JsonParser.parseString(candidate) }.getOrNull()
        }
        if (parsedCandidates.isEmpty()) {
            return null
        }

        return parsedCandidates.maxByOrNull(::scoreJsonCandidate)
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

    private fun collectBalancedCandidates(text: String, open: Char, close: Char): List<String> {
        val out = mutableListOf<String>()
        var index = text.indexOf(open)
        while (index >= 0) {
            balancedSubstring(text, index, open, close)?.let { out += it }
            index = text.indexOf(open, startIndex = index + 1)
        }
        return out
    }

    private fun scoreJsonCandidate(element: JsonElement): Int {
        var score = 0
        val coerce = FlatSpecContract.coerceAndValidate(element)
        if (coerce.isValid) {
            val spec = coerce.spec
            val elements = spec?.getAsJsonObject("elements")
            val size = elements?.size() ?: 0
            score += 400
            score += size.coerceAtMost(80)
            val rootId = spec?.get("root")
                ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                ?.asString
                .orEmpty()
            if (rootId.isNotBlank() && elements?.has(rootId) == true) {
                score += 60
            }
        } else {
            val normalized = FlatSpecContract.normalizeToFlatSpec(element)
            if (normalized.spec != null) {
                score += 160
            } else if (element.isJsonArray) {
                score += 80
            } else if (element.isJsonObject) {
                score += 40
            }
        }

        if (element.isJsonObject) {
            val obj = element.asJsonObject
            if (obj.has("genui_json") || obj.has("messages") || obj.has("payload")) {
                score += 50
            }
        }
        score += element.toString().length.coerceAtMost(4000) / 200
        return score
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
                "- `root` must reference an existing key in `elements`.\n" +
                "- `elements` must be a non-empty object with at least 2 entries (root container + content).\n" +
                "- Do NOT return `{}` and do NOT return `\"elements\": {}`.\n" +
                "- Every element must contain `type`, `props`, and `children`.\n" +
                "- Every id in `children` must exist in `elements`.\n" +
                "- Prefer a Stack root container with direction set.\n" +
                "- Return a complete, renderable flat-spec even when source content is brief.\n" +
                "- Return JSON only, no markdown.\n\n" +
                "Required minimum skeleton (adapt ids/content as needed):\n" +
                "{\"root\":\"root\",\"state\":{},\"elements\":{\"root\":{\"type\":\"Stack\",\"props\":{\"direction\":\"vertical\"},\"children\":[\"content\"]},\"content\":{\"type\":\"Text\",\"props\":{\"text\":\"...\"},\"children\":[]}}}\n\n" +
                "Original output:\n${rawText.trim()}"
            )
    }
}
