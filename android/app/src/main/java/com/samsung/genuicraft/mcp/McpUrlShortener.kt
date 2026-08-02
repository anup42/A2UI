package com.samsung.genuicraft.mcp

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonPrimitive

/**
 * Masks URLs and local asset paths with placeholders like {{u1}}, {{u2}} before
 * sending response data to the Stage 3 LLM, and restores them in the GenUI IR after.
 *
 * Keeping this transformation lossless prevents external and device-local references
 * from entering the IR model context while preserving their exact values for rendering.
 */
object McpUrlShortener {

    private const val LOCAL_ASSET_PREFIX =
        """(?:[a-z]:[\\/]|/(?:data|sdcard|storage|mnt|android_asset)/|\.{1,2}[\\/]|(?:assets?|media|images?|res|drawable|mipmap|raw)[\\/])"""
    private const val LOCAL_ASSET_TAIL =
        """[^\s<>"'|]*[-\p{L}\p{N}._~/#%+&=\\]"""

    private val REFERENCE_REGEX = Regex(
        """(?i)(?:(["'])($LOCAL_ASSET_PREFIX[^"']*[-\p{L}\p{N}._~/#%+&=\\])\1|(\b[a-z][a-z0-9+.-]*://[^\s<>"']*[\p{L}\p{N}/#=&%+~_-]|\b(?:mailto|tel|geo|intent|genuicraft|data|javascript|blob|urn|sms|market):[^\s<>"']*[\p{L}\p{N}/#=&%+~_-]|(?<![\p{L}\p{N}_])$LOCAL_ASSET_PREFIX$LOCAL_ASSET_TAIL|@[a-z][a-z0-9_.-]*/[a-z0-9_.-]+))"""
    )

    private val PLACEHOLDER_REGEX = Regex("""\{\{u(\d+)\}\}""")
    private val WHOLE_LOCAL_ASSET_REGEX = Regex(
        """(?i)^(?:[a-z]:[\\/]|/(?:data|sdcard|storage|mnt|android_asset)/|\.{1,2}[\\/]|(?:assets?|media|images?|res|drawable|mipmap|raw)[\\/]|@[a-z][a-z0-9_.-]*/[a-z0-9_.-]+).+"""
    )

    data class ShortenResult(
        val shortenedText: String,
        val urlMap: Map<String, String> // placeholder -> original URL or local asset path
    )

    data class JsonRestoreResult(
        val jsonElement: JsonElement,
        val removedUnmappedReferenceCount: Int,
    )

    /**
     * Replaces each unique URL and local asset path in [text] with a compact placeholder.
     * Existing placeholder numbers are skipped to prevent collisions.
     */
    fun shorten(text: String): ShortenResult {
        val seen = linkedMapOf<String, String>()
        var counter = (PLACEHOLDER_REGEX.findAll(text)
            .mapNotNull { it.groupValues.getOrNull(1)?.toIntOrNull() }
            .maxOrNull() ?: 0) + 1

        fun placeholderFor(reference: String): String {
            return seen.getOrPut(reference) {
                var candidate: String
                do {
                    candidate = "{{u${counter++}}}"
                } while (text.contains(candidate) || candidate in seen.values)
                candidate
            }
        }

        // A JSON string value arrives here without its surrounding quotes. Treat a
        // path that occupies the whole value as one reference so spaces inside a
        // Windows or asset path cannot be mistaken for reference boundaries.
        val trimmed = text.trim()
        if ('\n' !in text && '\r' !in text && WHOLE_LOCAL_ASSET_REGEX.matches(trimmed)) {
            val placeholder = placeholderFor(trimmed)
            val start = text.indexOf(trimmed)
            val shortenedText = text.replaceRange(start, start + trimmed.length, placeholder)
            val urlMap = seen.entries.associate { (reference, token) -> token to reference }
            return ShortenResult(shortenedText, urlMap)
        }

        val shortened = REFERENCE_REGEX.replace(text) { match ->
            val quote = match.groupValues[1]
            val quotedReference = match.groupValues[2]
            if (quotedReference.isNotEmpty()) {
                "$quote${placeholderFor(quotedReference)}$quote"
            } else {
                placeholderFor(match.groupValues[3])
            }
        }
        val urlMap = seen.entries.associate { (reference, placeholder) -> placeholder to reference }
        return ShortenResult(shortened, urlMap)
    }

    /** Replaces placeholder tokens in [text] with their exact original references. */
    fun restore(text: String, urlMap: Map<String, String>): String {
        if (urlMap.isEmpty()) return text
        var result = text
        for ((placeholder, reference) in urlMap) {
            result = result.replace(placeholder, reference)
        }
        return result
    }

    /**
     * Restores references inside JSON string values so paths remain correctly escaped.
     * Any raw URL/path invented by the model and absent from [urlMap] is removed.
     */
    fun restoreJsonReferences(element: JsonElement, urlMap: Map<String, String>): JsonRestoreResult {
        val allowedReferences = urlMap.values.toSet()
        var removedCount = 0

        fun restoreElement(value: JsonElement): JsonElement {
            return when {
                value.isJsonObject -> JsonObject().also { restored ->
                    value.asJsonObject.entrySet().forEach { (key, child) ->
                        restored.add(key, restoreElement(child))
                    }
                }
                value.isJsonArray -> JsonArray().also { restored ->
                    value.asJsonArray.forEach { child -> restored.add(restoreElement(child)) }
                }
                value.isJsonPrimitive && value.asJsonPrimitive.isString -> {
                    val restoredText = restore(value.asString, urlMap)
                    val detected = shorten(restoredText)
                    var sanitizedText = detected.shortenedText
                    detected.urlMap.forEach { (placeholder, reference) ->
                        val replacement = if (reference in allowedReferences) {
                            reference
                        } else {
                            removedCount += 1
                            ""
                        }
                        sanitizedText = sanitizedText.replace(placeholder, replacement)
                    }
                    JsonPrimitive(sanitizedText)
                }
                else -> value.deepCopy()
            }
        }

        return JsonRestoreResult(
            jsonElement = restoreElement(element),
            removedUnmappedReferenceCount = removedCount,
        )
    }
}
