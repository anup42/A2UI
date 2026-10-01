package com.samsung.genuicraft.renderer.native.parser

import android.net.Uri
import com.samsung.genuicraft.renderer.native.ParsedButton
import java.util.Locale

internal object NativeSourceParsing {
    private val SOURCE_HEADING_REGEX = Regex(
        """^(?:sources?|references?|citations?|links?|source\s+links?|reference\s+links?|useful\s+links?|external\s+links?|related\s+links?)$""",
        RegexOption.IGNORE_CASE
    )
    private val MARKDOWN_SOURCE_LINK_REGEX =
        Regex(
            """\[(.+?)]\((https?://[^\s)]+|//[^\s)]+|www\.[^\s)]+|(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:[/?#][^\s)]*)?)\)""",
            RegexOption.IGNORE_CASE
        )
    private val URL_REGEX = Regex(
        """(?i)(?:https?://|//)[^\s<>\]]+|(?<![@\w])(?:www\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:[/?#][^\s<>\]]*)?"""
    )

    fun collectSourceLinks(
        lines: List<String>,
        startIndex: Int,
        inSourcesSection: Boolean,
        looksLikeSectionHeading: (String) -> Boolean,
        parseSourceLinksFromLine: (String) -> List<ParsedButton>
    ): Pair<List<ParsedButton>, Int>? {
        val firstLine = lines[startIndex].trim()
        val sourcePrefixed = isSourceHeadingLine(firstLine) || isSourcePrefixedLinkLine(firstLine)
        if (!inSourcesSection && !sourcePrefixed) {
            return null
        }

        var cursor = startIndex
        val links = mutableListOf<ParsedButton>()
        while (cursor < lines.size) {
            val line = lines[cursor].trim()
            if (line.isEmpty()) {
                cursor++
                continue
            }

            if (cursor != startIndex && looksLikeSectionHeading(line)) {
                break
            }

            val parsed = parseSourceLinksFromLine(line)
            if (parsed.isEmpty()) {
                if (sourcePrefixed && cursor == startIndex) {
                    // Accept "Sources:" heading-only line and continue parsing next lines.
                    cursor++
                    continue
                }
                if (links.isNotEmpty()) {
                    break
                }
                return null
            }
            links += parsed
            cursor++
        }

        if (links.isEmpty()) {
            return null
        }
        val distinct = links.distinctBy { canonicalSourceUrl(it.url) }
        return distinct to cursor
    }

    fun parseSourceLinksFromLine(
        line: String,
        stripLeadingBulletMarker: (String) -> String,
        sanitizeDisplayText: (String) -> String,
        sanitizeUrlToken: (String) -> String,
        toExternalUrl: (String) -> String?,
        containsUrlLikeToken: (String) -> Boolean
    ): List<ParsedButton> {
        val normalized = normalizeSourceLineForParsing(stripLeadingBulletMarker(line))
        if (normalized.isBlank()) {
            return emptyList()
        }

        val markdownLinks = MARKDOWN_SOURCE_LINK_REGEX
            .findAll(normalized)
            .mapNotNull { match ->
                val rawLabel = normalizeSourceLabel(sanitizeDisplayText(match.groupValues[1]).trim())
                val normalizedUrl = toExternalUrl(sanitizeUrlToken(match.groupValues[2])) ?: return@mapNotNull null
                val label = if (isUsefulSourceLabel(rawLabel, containsUrlLikeToken)) {
                    rawLabel
                } else {
                    buildSourceLabelFromUrl(normalizedUrl, 0)
                }
                ParsedButton(label = label, url = normalizedUrl)
            }
            .toList()
        if (markdownLinks.isNotEmpty()) {
            return markdownLinks.distinctBy { canonicalSourceUrl(it.url) }
        }

        val urlMatches = URL_REGEX.findAll(normalized).toList()
        if (urlMatches.isEmpty()) {
            return emptyList()
        }

        val links = mutableListOf<ParsedButton>()
        var cursor = 0
        urlMatches.forEachIndexed { index, match ->
            val rawUrl = sanitizeUrlToken(match.value)
            val normalizedUrl = toExternalUrl(rawUrl) ?: return@forEachIndexed
            val nextStart = urlMatches.getOrNull(index + 1)?.range?.first ?: normalized.length
            val labelChunk = normalized.substring(cursor, match.range.first).trim()
            val trailingChunk = normalized.substring((match.range.last + 1).coerceAtMost(normalized.length), nextStart).trim()

            var label = labelChunk.removeSuffix(":").trim().trim('|')
            if (index == 0) {
                label = label.replace(
                    Regex(
                        """^(?:sources?|references?|citations?|source\s+links?|reference\s+links?|links?)\s*:?\s*""",
                        RegexOption.IGNORE_CASE
                    ),
                    ""
                )
            }
            label = label
                .replace(Regex("""^\s*\d+\s*[\).:\-\u2013\u2014]?\s*"""), "")
                .replace(Regex("""^\s*[\[\(]\s*\d+\s*[\]\)]\s*[:\-\u2013\u2014]?\s*"""), "")
                .replace(Regex("""^\s*[-\u2013\u2014|:]\s*"""), "")
                .trim(' ', '-', '\u2013', '\u2014', '|', ':')
            label = normalizeSourceLabel(label)

            if (label.isBlank() && trailingChunk.isNotBlank() && !containsUrlLikeToken(trailingChunk)) {
                label = trailingChunk
                    .replace(Regex("""^\s*[-\u2013\u2014|:\u2022]\s*"""), "")
                    .trim(' ', '-', '\u2013', '\u2014', '|', ':')
                label = normalizeSourceLabel(label)
            }

            if (!isUsefulSourceLabel(label, containsUrlLikeToken)) {
                label = buildSourceLabelFromUrl(normalizedUrl, index)
            }

            links += ParsedButton(
                label = label,
                url = normalizedUrl
            )
            cursor = nextStart
        }
        return links.distinctBy { canonicalSourceUrl(it.url) }
    }

    fun isSourceHeadingLine(line: String): Boolean {
        if (line.isBlank()) {
            return false
        }
        val cleaned = normalizeSourceHeadingToken(line)
        if (cleaned.length > 44) {
            return false
        }
        return SOURCE_HEADING_REGEX.matches(cleaned)
    }

    fun isSourcePrefixedLinkLine(line: String): Boolean {
        val normalized = normalizeSourceLineForParsing(line)
        if (normalized.isBlank()) {
            return false
        }
        val prefix = normalizeSourceHeadingToken(normalized.substringBefore(':'))
        if (!SOURCE_HEADING_REGEX.matches(prefix)) {
            return false
        }
        val afterColon = normalized.substringAfter(':', "").trim()
        if (afterColon.isBlank()) {
            return false
        }
        return MARKDOWN_SOURCE_LINK_REGEX.containsMatchIn(afterColon) || URL_REGEX.containsMatchIn(afterColon)
    }

    fun normalizeSourceHeadingToken(line: String): String {
        var cleaned = line.trim()
        cleaned = cleaned.trimStart('-', '\u2022', '*').trim()
        cleaned = cleaned.replace(Regex("""^#{1,6}\s*"""), "")
        cleaned = cleaned
            .removeSurrounding("**")
            .removeSurrounding("__")
            .removeSurrounding("*")
            .removeSurrounding("_")
            .trim()
        cleaned = cleaned
            .removeSuffix(":")
            .removeSuffix("-")
            .removeSuffix("\u2013")
            .removeSuffix("\u2014")
            .trim()
        return cleaned
    }

    private fun normalizeSourceLineForParsing(line: String): String {
        if (line.isBlank()) {
            return line
        }
        var normalized = line.trim()
        normalized = normalized.replace(Regex("""^#{1,6}\s*"""), "")
        if (isSourceHeadingLine(normalized)) {
            return normalizeSourceHeadingToken(normalized) + ":"
        }
        return normalized
    }

    private fun normalizeSourceLabel(raw: String): String {
        if (raw.isBlank()) {
            return raw
        }
        return raw
            .replace(Regex("""^\s*[\[\(]\s*\d+\s*[\]\)]\s*[:\-\u2013\u2014]?\s*"""), "")
            .replace(
                Regex(
                    """^\s*(?:sources?|references?|citations?|source\s+links?|reference\s+links?|links?)\s*:?\s*""",
                    RegexOption.IGNORE_CASE
                ),
                ""
            )
            .trim()
            .trim('[', ']', '(', ')', '{', '}', '"', '\'', '-', '\u2013', '\u2014', '|', ':', ' ')
            .replace(Regex("""\s{2,}"""), " ")
    }

    private fun isUsefulSourceLabel(
        label: String,
        containsUrlLikeToken: (String) -> Boolean
    ): Boolean {
        val normalized = label.trim()
        if (normalized.isBlank()) {
            return false
        }
        if (containsUrlLikeToken(normalized)) {
            return false
        }
        if (Regex("""(?i)^(?:www\.)?(?:[a-z0-9-]+\.)+[a-z]{2,24}$""").matches(normalized)) {
            return false
        }
        val lower = normalized.lowercase(Locale.US)
        if (lower in setOf("source", "sources", "reference", "references", "citation", "citations", "link", "links")) {
            return false
        }
        return normalized.any { it.isLetter() }
    }

    private fun buildSourceLabelFromUrl(url: String, index: Int): String {
        val uri = runCatching { Uri.parse(url) }.getOrNull()
        val host = uri?.host?.removePrefix("www.")?.trim().orEmpty()
        if (host.isBlank()) {
            return "Source ${index + 1}"
        }
        val hostLabel = host
            .substringBefore('.')
            .replace('-', ' ')
            .replace('_', ' ')
            .split(Regex("""\s+"""))
            .filter { it.isNotBlank() }
            .joinToString(" ") { token ->
                token.lowercase(Locale.US).replaceFirstChar { ch ->
                    if (ch.isLowerCase()) ch.titlecase(Locale.US) else ch.toString()
                }
            }
            .ifBlank { host }
        val pathHint = uri?.pathSegments
            ?.firstOrNull { segment -> segment.length >= 3 && segment.any { it.isLetter() } }
            ?.replace('-', ' ')
            ?.replace('_', ' ')
            ?.split(Regex("""\s+"""))
            ?.filter { it.isNotBlank() }
            ?.take(3)
            ?.joinToString(" ") { token ->
                token.lowercase(Locale.US).replaceFirstChar { ch ->
                    if (ch.isLowerCase()) ch.titlecase(Locale.US) else ch.toString()
                }
            }
            ?.takeIf { it.isNotBlank() && !hostLabel.contains(it, ignoreCase = true) }

        return if (pathHint != null) "$hostLabel $pathHint" else hostLabel
    }

    private fun canonicalSourceUrl(url: String): String {
        val raw = url.trim()
        val uri = runCatching { Uri.parse(raw) }.getOrNull() ?: return raw.lowercase(Locale.US)
        val scheme = (uri.scheme ?: "https").lowercase(Locale.US)
        val host = uri.host?.lowercase(Locale.US)?.removePrefix("www.").orEmpty()
        if (host.isBlank()) {
            return raw.lowercase(Locale.US)
        }
        val path = uri.path.orEmpty().trimEnd('/')
        val query = uri.query.orEmpty().trim()
        return buildString {
            append(scheme)
            append("://")
            append(host)
            if (path.isNotBlank()) {
                append(path)
            }
            if (query.isNotBlank()) {
                append('?')
                append(query)
            }
        }
    }
}
