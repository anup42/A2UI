package com.samsung.genuicraft.sdk.internal.renderer.native.parser

import com.samsung.genuicraft.sdk.internal.renderer.native.BookingOption
import com.samsung.genuicraft.sdk.internal.renderer.native.ParsedButton
import com.samsung.genuicraft.sdk.internal.renderer.native.ParsedMediaEntry
import com.samsung.genuicraft.sdk.internal.renderer.native.StepEntry
import com.samsung.genuicraft.sdk.internal.renderer.native.TextBlock

internal object NativeTextBlockParser {
    private val BOLD_HEADING_REGEX = Regex("""^\s*(?:\*\*|__)\s*(.+?)\s*(?:\*\*|__)\s*$""")

    fun shouldUseStructuredBlocks(
        rawText: String,
        variant: String,
        isTableLikeLine: (String) -> Boolean,
        isBulletListLine: (String) -> Boolean,
        isInlineMediaLine: (String) -> Boolean,
        isMediaMarkerHeading: (String) -> Boolean,
        looksLikeStandaloneLinkLine: (String) -> Boolean,
        isSourceHeadingLine: (String) -> Boolean
    ): Boolean {
        if (variant in setOf("h1", "h2", "h3", "h4")) {
            return false
        }
        val normalized = rawText.replace("\r\n", "\n")
        val lines = normalized.lines().map { it.trim() }.filter { it.isNotEmpty() }
        if (lines.isEmpty()) {
            return false
        }

        val hasButtons = normalized.contains("[Button:", ignoreCase = true)
        val hasTable = lines.count(isTableLikeLine) >= 2
        val hasBullets = lines.count(isBulletListLine) >= 2
        val hasMedia = lines.any { isInlineMediaLine(it) || isMediaMarkerHeading(it) }
        val hasStandaloneLinks = lines.any(looksLikeStandaloneLinkLine)
        val hasSourceHeading = lines.any(isSourceHeadingLine)
        val hasTagLine = lines.any { it.startsWith("Tags:", ignoreCase = true) }
        val hasMultiLineLayout = lines.size >= 3
        return hasButtons ||
            hasTable ||
            hasBullets ||
            hasMedia ||
            hasStandaloneLinks ||
            hasSourceHeading ||
            hasTagLine ||
            hasMultiLineLayout ||
            normalized.contains("\n\n")
    }

    fun parseTextBlocks(
        rawText: String,
        collectSourceLinks: (List<String>, Int, Boolean) -> Pair<List<ParsedButton>, Int>?,
        collectBookingOptions: (List<String>, Int) -> Pair<List<BookingOption>, Int>?,
        collectTableRows: (List<String>, Int) -> Pair<List<List<String>>, Int>?,
        collectMediaEntries: (List<String>, Int) -> Pair<List<ParsedMediaEntry>, Int>?,
        collectNumberedSteps: (List<String>, Int) -> Pair<List<StepEntry>, Int>?,
        isBulletListLine: (String) -> Boolean,
        extractBulletListItem: (String) -> String?,
        isPlaceholderListEntry: (String) -> Boolean,
        parseButtonLine: (String) -> ParsedButton?,
        looksLikeStandaloneLinkLine: (String) -> Boolean,
        parseSourceLinksFromLine: (String) -> List<ParsedButton>,
        isSourcePrefixedLinkLine: (String) -> Boolean,
        looksLikeSectionHeading: (String) -> Boolean,
        isStructuredBoundary: (String) -> Boolean
    ): List<TextBlock> {
        val lines = rawText.replace("\r\n", "\n").split('\n')
        val blocks = mutableListOf<TextBlock>()
        var index = 0
        var renderedAny = false
        var inSourcesSection = false

        while (index < lines.size) {
            val line = lines[index].trim()
            if (line.isEmpty()) {
                index++
                continue
            }

            val markdownHeading = extractMarkdownBoldHeading(line)
            if (markdownHeading != null) {
                if (!renderedAny) {
                    blocks += TextBlock.Title(markdownHeading)
                } else {
                    blocks += TextBlock.Heading(markdownHeading)
                }
                inSourcesSection = NativeSourceParsing.isSourceHeadingLine(markdownHeading)
                renderedAny = true
                index++
                continue
            }

            val sourceLinks = collectSourceLinks(lines, index, inSourcesSection)
            if (sourceLinks != null) {
                val (links, nextIndex) = sourceLinks
                if (links.isNotEmpty()) {
                    blocks += TextBlock.Sources(links)
                    inSourcesSection = true
                    renderedAny = true
                    index = nextIndex
                    continue
                }
            }

            val bookingOptions = collectBookingOptions(lines, index)
            if (bookingOptions != null) {
                val (options, nextIndex) = bookingOptions
                if (options.isNotEmpty()) {
                    blocks += TextBlock.BookingCards(options)
                    renderedAny = true
                    index = nextIndex
                    continue
                }
            }

            val tableRows = collectTableRows(lines, index)
            if (tableRows != null) {
                val (rows, nextIndex) = tableRows
                blocks += TextBlock.Table(rows)
                renderedAny = true
                index = nextIndex
                continue
            }

            val mediaEntries = collectMediaEntries(lines, index)
            if (mediaEntries != null) {
                val (entries, nextIndex) = mediaEntries
                if (entries.isNotEmpty()) {
                    blocks += TextBlock.MediaCards(entries)
                    renderedAny = true
                }
                index = nextIndex
                continue
            }

            val numberedSteps = collectNumberedSteps(lines, index)
            if (numberedSteps != null) {
                val (steps, nextIndex) = numberedSteps
                if (steps.isNotEmpty()) {
                    blocks += TextBlock.NumberedSteps(steps)
                    renderedAny = true
                    index = nextIndex
                    continue
                }
            }

            // Tags: A | B | C → chip row
            if (line.startsWith("Tags:", ignoreCase = true)) {
                val tagPart = line.substringAfter(":").trim()
                val tags = tagPart.split("|").map { it.trim() }.filter { it.isNotBlank() }
                if (tags.isNotEmpty()) {
                    blocks += TextBlock.TagRow(tags)
                    renderedAny = true
                }
                index++
                continue
            }

            val actions = mutableListOf<ParsedButton>()
            var actionCursor = index
            while (actionCursor < lines.size) {
                val parsed = parseButtonLine(lines[actionCursor].trim()) ?: break
                actions += parsed
                actionCursor++
            }
            if (actions.isNotEmpty()) {
                blocks += TextBlock.Actions(actions)
                renderedAny = true
                index = actionCursor
                continue
            }

            if (isBulletListLine(line)) {
                val items = mutableListOf<String>()
                var cursor = index
                while (cursor < lines.size) {
                    val l = lines[cursor].trim()
                    if (!isBulletListLine(l)) break
                    val item = extractBulletListItem(l) ?: break
                    if (!isPlaceholderListEntry(item)) {
                        items += item
                    }
                    cursor++
                }
                if (items.isNotEmpty()) {
                    blocks += TextBlock.Bullets(items)
                    renderedAny = true
                }
                index = cursor
                continue
            }

            val inlineLinkButtons =
                if (looksLikeStandaloneLinkLine(line)) {
                    parseSourceLinksFromLine(line)
                } else {
                    emptyList()
                }
            if (inlineLinkButtons.isNotEmpty()) {
                val treatAsSources = inSourcesSection || isSourcePrefixedLinkLine(line)
                blocks += if (treatAsSources) {
                    TextBlock.Sources(inlineLinkButtons)
                } else {
                    TextBlock.Actions(inlineLinkButtons)
                }
                if (treatAsSources) {
                    inSourcesSection = true
                }
                renderedAny = true
                index++
                continue
            }

            if (!renderedAny) {
                blocks += TextBlock.Title(line)
                renderedAny = true
                index++
                continue
            }

            if (looksLikeSectionHeading(line)) {
                blocks += TextBlock.Heading(line)
                inSourcesSection = NativeSourceParsing.isSourceHeadingLine(line)
                index++
                continue
            }

            val paragraphLines = mutableListOf<String>()
            var cursor = index
            while (cursor < lines.size) {
                val candidate = lines[cursor].trim()
                if (candidate.isEmpty() || isStructuredBoundary(candidate)) {
                    break
                }
                paragraphLines += candidate
                cursor++
            }
            if (paragraphLines.isNotEmpty()) {
                blocks += TextBlock.Paragraph(paragraphLines.joinToString(" "))
                index = cursor
                continue
            }

            index++
        }

        return blocks
    }

    private fun extractMarkdownBoldHeading(line: String): String? {
        val match = BOLD_HEADING_REGEX.matchEntire(line) ?: return null
        val heading = match.groupValues.getOrNull(1)?.trim().orEmpty()
        if (heading.length !in 2..100) {
            return null
        }
        if (!heading.any { it.isLetter() }) {
            return null
        }
        return heading
    }
}
