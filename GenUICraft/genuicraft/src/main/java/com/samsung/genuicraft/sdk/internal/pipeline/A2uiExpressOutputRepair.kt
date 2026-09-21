package com.samsung.genuicraft.sdk.internal.pipeline

/**
 * Bounded repair for generated Express syntax. It never edits visible values, guesses component
 * types, fixes factual text, or invents missing UI. Candidate programs are accepted only after the
 * normal codec and canonical graph validator accept the result.
 */
internal object A2uiExpressOutputRepair {
    data class Repair(val express: String, val changes: List<String>)

    fun repair(input: String): Repair? {
        val candidates = candidates(input)
        for ((candidate, initialChanges) in candidates) {
            val graph = runCatching { A2uiExpressCodec.decode(candidate) }.getOrNull() ?: continue
            val validation = A2uiCanonicalGraph.validate(graph)
            if (!validation.isValid) continue
            val canonical = runCatching { A2uiExpressCodec.encode(graph) }.getOrNull() ?: continue
            return Repair(canonical, initialChanges.distinct())
        }
        return null
    }

    private fun candidates(input: String): List<Pair<String, List<String>>> {
        val withoutBom = input.removePrefix("\uFEFF")
        val original = withoutBom.trim()
        if (original.isEmpty()) return emptyList()
        val values = linkedMapOf<String, List<String>>()
        val baseChanges = if (withoutBom != input) listOf("Removed a leading UTF-8 BOM.") else emptyList()
        fun add(value: String, changes: List<String>) {
            val trimmed = value.trim()
            if (trimmed.startsWith("<a2ui>") && trimmed.length <= 120_000) values.putIfAbsent(trimmed, changes)
        }

        add(original, baseChanges)
        exactFenceBody(original)?.let {
            add(it, baseChanges + "Removed an exact Markdown code-fence wrapper.")
        }

        values.toList().forEach { (value, changes) ->
            normalizeMalformedClosingTag(value)?.let {
                add(it, changes + "Normalized a malformed trailing closing tag.")
            }
        }

        return values.map { it.key to it.value }
    }

    private fun exactFenceBody(value: String): String? {
        val match = Regex("(?s)^```(?:a2ui|express)?\\s*\\n(.*)\\n```$").matchEntire(value) ?: return null
        return match.groupValues[1]
    }

    private fun normalizeMalformedClosingTag(value: String): String? {
        val normalized = value.replace(
            Regex("(?s)</a2ui\\s*$"),
            "</a2ui>",
        )
        return normalized.takeIf { it != value }
    }
}
