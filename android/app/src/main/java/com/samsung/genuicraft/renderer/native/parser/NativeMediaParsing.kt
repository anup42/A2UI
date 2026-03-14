package com.samsung.genuicraft.renderer.native.parser

import com.samsung.genuicraft.renderer.native.ParsedMediaEntry
import com.samsung.genuicraft.renderer.native.intents.flight.NativeFlightSemantics
import java.util.Locale

internal object NativeMediaParsing {
    private val INLINE_MEDIA_ASSIGNMENT_REGEX =
        Regex("""(?i)\b(Image|Icon)\s*=\s*(https?://\S+|/assets/\S+|assets/\S+|\S+)""")
    private val MARKDOWN_IMAGE_REGEX =
        Regex(
            """!\[([^\]]*)\]\((https?://[^\s)]+|//[^\s)]+|/assets/[^\s)]+|assets/[^\s)]+)\)""",
            RegexOption.IGNORE_CASE
        )

    fun collectMediaEntries(
        lines: List<String>,
        startIndex: Int,
        stripLeadingBulletMarker: (String) -> String,
        sanitizeUrlToken: (String) -> String,
        looksLikeImagePath: (String, String?) -> Boolean,
        looksLikeCompactIconUrl: (String) -> Boolean
    ): Pair<List<ParsedMediaEntry>, Int>? {
        var cursor = startIndex
        val entries = mutableListOf<ParsedMediaEntry>()
        var markerHeading: String? = null

        while (cursor < lines.size) {
            val line = lines[cursor].trim()
            if (line.isEmpty()) {
                cursor++
                continue
            }

            if (isMediaMarkerHeading(line)) {
                markerHeading = line.lowercase(Locale.US)
                cursor++
                continue
            }

            val mediaEntry = parseMediaEntryLine(
                line = line,
                stripLeadingBulletMarker = stripLeadingBulletMarker,
                looksLikeImagePath = looksLikeImagePath,
                looksLikeCompactIconUrl = looksLikeCompactIconUrl
            )
            if (mediaEntry != null) {
                entries += mediaEntry
                cursor++
                continue
            }

            val inlineEntries = parseInlineMediaEntries(
                line = line,
                sanitizeUrlToken = sanitizeUrlToken,
                looksLikeImagePath = looksLikeImagePath,
                looksLikeCompactIconUrl = looksLikeCompactIconUrl
            )
            if (inlineEntries.isNotEmpty()) {
                entries += inlineEntries
                cursor++
                continue
            }
            break
        }

        if (entries.isEmpty()) {
            return null
        }

        if (markerHeading?.startsWith("icons") == true &&
            entries.all { it.iconLike && isGenericUtilityIconLabel(it.label) }
        ) {
            return emptyList<ParsedMediaEntry>() to cursor
        }

        return entries to cursor
    }

    fun isMediaMarkerHeading(line: String): Boolean {
        val normalized = line.trim().lowercase(Locale.US)
        return normalized in setOf(
            "media:",
            "icons:",
            "icon:",
            "assets:",
            "asset:",
            "logos:",
            "logo:",
            "images:",
            "image:",
            "airline logos:"
        )
    }

    fun isInlineMediaLine(line: String): Boolean {
        val trimmed = line.trim()
        if (MARKDOWN_IMAGE_REGEX.containsMatchIn(trimmed)) {
            return true
        }
        if (trimmed.startsWith("Media:", ignoreCase = true)) {
            return INLINE_MEDIA_ASSIGNMENT_REGEX.containsMatchIn(trimmed)
        }
        if (trimmed.startsWith("Image:", ignoreCase = true) || trimmed.startsWith("Icon:", ignoreCase = true)) {
            return true
        }
        return INLINE_MEDIA_ASSIGNMENT_REGEX.containsMatchIn(trimmed)
    }

    private fun isGenericUtilityIconLabel(label: String): Boolean {
        val normalized = NativeFlightSemantics.normalizeMatchText(label)
        return normalized in setOf("airline", "time", "clock", "calendar", "date")
    }

    private fun parseMediaEntryLine(
        line: String,
        stripLeadingBulletMarker: (String) -> String,
        looksLikeImagePath: (String, String?) -> Boolean,
        looksLikeCompactIconUrl: (String) -> Boolean
    ): ParsedMediaEntry? {
        val normalized = stripLeadingBulletMarker(line)
        val parts = normalized.split(":", limit = 2)
        if (parts.size != 2) {
            return null
        }
        val label = parts[0].trim()
        val value = parts[1].trim()
        if (value.contains("Image=", ignoreCase = true) || value.contains("Icon=", ignoreCase = true)) {
            return null
        }
        if (label.isEmpty() || !looksLikeImagePath(value, label)) {
            return null
        }

        val lowerLabel = label.lowercase(Locale.US)
        val lowerValue = value.lowercase(Locale.US)
        val iconLike = lowerLabel.contains("icon") ||
            looksLikeCompactIconUrl(lowerValue) ||
            (lowerValue.endsWith(".svg") && !lowerLabel.contains("logo") && !lowerValue.contains("logo")) ||
            (lowerValue.contains("/icon") && !lowerLabel.contains("logo") && !lowerValue.contains("logo"))

        return ParsedMediaEntry(
            label = label,
            url = value,
            iconLike = iconLike
        )
    }

    private fun parseInlineMediaEntries(
        line: String,
        sanitizeUrlToken: (String) -> String,
        looksLikeImagePath: (String, String?) -> Boolean,
        looksLikeCompactIconUrl: (String) -> Boolean
    ): List<ParsedMediaEntry> {
        if (!isInlineMediaLine(line)) {
            return emptyList()
        }

        val assignmentEntries = INLINE_MEDIA_ASSIGNMENT_REGEX
            .findAll(line)
            .mapNotNull { match ->
                val mediaType = match.groupValues[1].trim()
                val rawValue = sanitizeUrlToken(match.groupValues[2])
                if (!looksLikeImagePath(rawValue, mediaType)) {
                    null
                } else {
                    val normalizedType = mediaType.lowercase(Locale.US)
                    val normalizedUrl = rawValue.lowercase(Locale.US)
                    val inlineIconLike = normalizedType.contains("icon") ||
                        looksLikeCompactIconUrl(normalizedUrl) ||
                        (normalizedUrl.endsWith(".svg") && !normalizedUrl.contains("logo")) ||
                        (normalizedUrl.contains("/icon") && !normalizedUrl.contains("logo"))
                    ParsedMediaEntry(
                        label = mediaType,
                        url = rawValue,
                        iconLike = inlineIconLike
                    )
                }
            }
            .toList()
        val markdownEntries = parseMarkdownImageEntries(line, sanitizeUrlToken, looksLikeImagePath, looksLikeCompactIconUrl)
        return (assignmentEntries + markdownEntries)
            .distinctBy { "${it.label.lowercase(Locale.US)}|${it.url.lowercase(Locale.US)}" }
    }

    private fun parseMarkdownImageEntries(
        line: String,
        sanitizeUrlToken: (String) -> String,
        looksLikeImagePath: (String, String?) -> Boolean,
        looksLikeCompactIconUrl: (String) -> Boolean
    ): List<ParsedMediaEntry> {
        return MARKDOWN_IMAGE_REGEX
            .findAll(line)
            .mapNotNull { match ->
                val label = match.groupValues.getOrNull(1)?.trim().orEmpty().ifBlank { "Image" }
                val rawValue = sanitizeUrlToken(match.groupValues.getOrNull(2).orEmpty())
                if (!looksLikeImagePath(rawValue, label)) {
                    null
                } else {
                    val normalizedLabel = label.lowercase(Locale.US)
                    val normalizedUrl = rawValue.lowercase(Locale.US)
                    val iconLike = normalizedLabel.contains("icon") ||
                        looksLikeCompactIconUrl(normalizedUrl) ||
                        (normalizedUrl.endsWith(".svg") && !normalizedUrl.contains("logo")) ||
                        (normalizedUrl.contains("/icon") && !normalizedUrl.contains("logo"))
                    ParsedMediaEntry(
                        label = label,
                        url = rawValue,
                        iconLike = iconLike
                    )
                }
            }
            .toList()
    }
}
